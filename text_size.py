import json
import numpy as np
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import tqdm

# --- Global Tokenizer Initialization ---
# This function initializes the tokenizer for each worker process.
# We define it globally so it's available to the pool's initializer.
tokenizer = None

def init_tokenizer(model_path, target_token_length=1024):
    """Initializer for each worker process."""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.target_token_length = target_token_length

# --- Worker Function ---
# This is the function that each worker process will execute.
def process_line(line):
    """
    Processes a single line from the JSONL file to count tokens.
    Returns the token count or None if an error occurs.
    """
    try:
        # Load the JSON object from the line
        data = json.loads(line)
        text = data.get("text")

        if text and isinstance(text, str):
            # The global tokenizer, initialized for this process, is used here
            ids = tokenizer.encode(text, truncation=True, max_length=tokenizer.target_token_length)
            ids = ids[1:]
            s = tokenizer.decode(ids)
            return len(s.encode('utf-8')) * 8 / len(ids)
        else:
            # Return None for lines without a valid 'text' field
            return None
    except (json.JSONDecodeError, AttributeError):
        # Return None for malformed JSON or other errors
        return None

def calculate_text_size_per_token(file_path, model_path, target_token_length=1024):
    """
    Calculates token count statistics in a parallelized manner.

    Args:
        file_path (str): The path to the JSONL file.
    """
    init_tokenizer(model_path, target_token_length)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return
    
    if not lines:
        print("File is empty. No statistics to calculate.")
        return

    # Determine the number of processes to use
    num_processes = cpu_count() // 2
    print(f"Starting parallel processing with {num_processes} workers...")

    token_counts = []
    
    # Create a pool of worker processes
    # The initializer runs `init_tokenizer` once for each worker process.
    with Pool(processes=num_processes, initializer=init_tokenizer, initargs=(model_path, target_token_length)) as pool:
        # Use imap_unordered for efficiency, as order doesn't matter.
        results = list(tqdm.tqdm(pool.imap_unordered(process_line, lines), total=len(lines), desc="Processing lines"))

    # Filter out the None results from failed lines
    token_counts = [count for count in results if count is not None]

    if not token_counts:
        print("No valid text lines were found to calculate statistics.")
        return

    # Calculate and print statistics
    counts_array = np.array(token_counts)

    return {
        "file_path": {file_path},
        "tokenizer": {tokenizer.name_or_path},
        "vocab_size": {len(tokenizer)},
        "max_sample_length": target_token_length,
        "total_valid_lines": len(counts_array),
        "mean_text_size": round(np.mean(counts_array), 2),
        "min_text_size": np.min(counts_array),
        "max_text_size": np.max(counts_array),
    }