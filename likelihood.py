import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from math import ceil

class JsonlIterableDataset(IterableDataset):
    """Sequential streaming dataset for jsonl lines of the form {"text": "..."}."""
    def __init__(self, jsonl_path: str, tokenizer, target_token_length: int):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.target_token_length = target_token_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            start, stride = 0, 1
        else:
            # Multi-worker: split work evenly
            start = worker_info.id
            stride = worker_info.num_workers

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % stride != start:
                    continue
                data = json.loads(line)
                text = data["text"]

                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.target_token_length,
                    return_tensors="pt"
                )
                yield {
                    "input_ids": tokens["input_ids"].squeeze(0),
                    "attention_mask": tokens["attention_mask"].squeeze(0),
                }


def calculate_negative_log_likelihood(
    model_path: str,
    jsonl_path: str,
    target_token_length: int,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 2,
    num_samples: int = None,
) -> torch.Tensor:
    """
    Streaming, batched NLL computation for a large jsonl dataset using deterministic sequential access.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", 
        attn_implementation="flash_attention_2" if "TinyLlama" not in model_path else "sdpa", trust_remote_code=True)
    model.eval()

    dataset = JsonlIterableDataset(jsonl_path, tokenizer, target_token_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    entropies = []

    for i, batch in enumerate(tqdm(dataloader, total=ceil(num_samples / batch_size) if num_samples is not None else None, 
                                   desc=f"Calculating Entropy for {model_path.split('/')[-1]}")):
        if i % 100 == 0: torch.cuda.empty_cache()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Per-token NLL
        logits = torch.softmax(logits[:, :, :len(tokenizer)].to(dtype=torch.float32), dim=-1)
        effective_probs = torch.gather(logits[:, :target_token_length, :], -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        entropy = -torch.log2(effective_probs)
        entropy[attention_mask[:, 1:] == 0] = torch.nan

        entropies.append(entropy.cpu())

    return torch.cat(entropies, dim=0)