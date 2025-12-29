import json
import torch
import numpy as np
import math
from typing import Iterator, List, Dict
from itertools import islice, chain
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def calculate_negative_log_likelihood(
    model_path: str,
    jsonl_path: str,
    target_token_length: int,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    chunk_size: int = 1000, # Number of samples to hold in RAM before sending to vLLM
    num_samples: int = 200000,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Calculates NLL using streaming data loading and vLLM.
    Input is read lazily from disk, preventing OOM on the inputs.
    """
    
    # 1. Initialize Tokenizer and vLLM
    # We use the HF tokenizer to handle truncation explicitly before vLLM
    print(f"Initializing model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        enforce_eager=False,
        gpu_memory_utilization=0.8,
        max_model_len=target_token_length + 1,
    )

    # prompt_logprobs=1 returns the logprob of the token that was actually matched/generated
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1, detokenize=False)
    
    # constant for log conversion
    ln_2 = np.log(2)

    # 2. Define the data generator
    def data_generator() -> Iterator[List[int]]:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Tokenize and Truncate immediately to save RAM
                token_ids = tokenizer.encode(
                    data["text"],
                    truncation=True,
                    max_length=target_token_length,
                    add_special_tokens=True
                )
                yield token_ids

    # 3. Process in Chunks
    # If the output tensor is too large for CPU RAM, you should write results to disk 
    # inside this loop instead of appending to 'all_results'.
    all_results = []
    
    # Create the iterator
    token_iter = data_generator()
    
    print(f"Starting streaming inference with chunk size {chunk_size}...")
    
    # Loop over batches of the dataset
    for batch_idx, batch_token_ids in enumerate(tqdm(batched(token_iter, chunk_size), total=math.ceil(num_samples / chunk_size) if num_samples is not None else None, 
                                   desc=f"Calculating Entropy for {model_path.split('/')[-1]}")):
        
        # Run vLLM on this chunk
        # vLLM handles internal batching for GPU throughput, but this loop manages CPU RAM.
        request_outputs = llm.generate(
            prompt_token_ids=batch_token_ids,
            sampling_params=sampling_params,
            use_tqdm=True
        )

        # Process results for this chunk
        chunk_entropies = []
        
        for i, output in enumerate(request_outputs):
            seq_logprobs = output.prompt_logprobs
            token_ids = batch_token_ids[i]
            
            # Extract logprobs for prediction (tokens 1 to N)
            # seq_logprobs[j] corresponds to the token at index j
            current_nlls = []
            
            # We predict token[j] given token[0...j-1]
            # prompt_logprobs list aligns with input tokens. 
            # Entry 0 is None. Entry 1 is logprob of token 1 given token 0.
            for j in range(1, len(seq_logprobs)):
                token_at_j = token_ids[j]
                step_logprobs = seq_logprobs[j]
                
                if step_logprobs is not None and token_at_j in step_logprobs:
                    log_prob = step_logprobs[token_at_j].logprob
                    # Convert ln to log2 and negate
                    current_nlls.append(-(log_prob / ln_2))
                else:
                    current_nlls.append(float('nan'))

            chunk_entropies.append(current_nlls)

        # Convert chunk to tensor
        # Create tensor filled with NaN
        chunk_tensor = torch.full((len(chunk_entropies), target_token_length - 1), float('nan'))
        
        for k, nll_list in enumerate(chunk_entropies):
            # Fill valid data
            valid_len = min(len(nll_list), target_token_length - 1)
            chunk_tensor[k, :valid_len] = torch.tensor(nll_list[:valid_len])
            
        all_results.append(chunk_tensor)
        
        # Optional: Explicit garbage collection if memory is extremely tight
        del request_outputs, batch_token_ids
        
    # 4. Concatenate all chunks
    if not all_results:
        return torch.empty(0)
        
    return torch.cat(all_results, dim=0)