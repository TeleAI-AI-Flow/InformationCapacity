import argparse
import torch
from math import log2
from text_size import calculate_text_size_per_token
from likelihood import calculate_negative_log_likelihood
from flops import gqa_model_theoretical_flops, mla_model_theoretical_flops

def calculate_information_capacity(
    model_path: str,
    data_path: str,
    max_sample_length: int = 1024,
    batch_size: int = 1,
    numerator_bias: float = None,
    attention_mechanism: str = None,
) -> float:
    if attention_mechanism is None:
        attention_mechanism = "mla" if "deepseek" in model_path.lower() else "gqa"
    else:
        attention_mechanism = attention_mechanism.lower()
    if attention_mechanism != "gqa" and attention_mechanism != "mla":
        raise NotImplementedError("attention_mechanism argument should be either gqa or mla")
    
    if numerator_bias is None:
        if "mixed_text.jsonl" in data_path: numerator_bias = -24
        elif "Ch-FineWeb-Edu.jsonl" in data_path: numerator_bias = -18.7
        else: numerator_bias = -27
        print(f"numerator_bias is not designated, default to {numerator_bias} based on the data_path")
    
    ts_results = calculate_text_size_per_token(data_path, model_path, target_token_length=max_sample_length)
    avg_ts = ts_results["mean_text_size"]
    for k, v in ts_results.items(): print(f"{k}: {v}")

    nlls = calculate_negative_log_likelihood(model_path, data_path, max_sample_length, batch_size=batch_size, num_samples=ts_results["total_valid_lines"])
    avg_nll = torch.nanmean(nlls).item()
    print(f"Average negative log-likelihood: {avg_nll}")

    cfg_path = model_path + "/config.json"
    if attention_mechanism == "gqa":
        flops_results = gqa_model_theoretical_flops(cfg_path, gen_len=max_sample_length)
    elif attention_mechanism == "mla":
        flops_results = mla_model_theoretical_flops(cfg_path, gen_len=max_sample_length)
    per_token_flops = flops_results["decode_total_TFLOPs"] * 1e12 / max_sample_length
    for k, v in flops_results.items(): print(f"{k}: {v}")
    
    ic = (avg_ts - avg_nll + numerator_bias) / log2(per_token_flops)
    print(f"\nInformation capacity: {ic}")

    return ic


def main():
    parser = argparse.ArgumentParser(
        description="Compute the information capacity of a language model."
    )
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the dataset (JSONL format).")
    parser.add_argument("-l", "--max_sample_length", type=int, default=1024, help="Maximum token length for each sample.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("-n", "--numerator_bias", type=float, default=None, 
                        help="Optional numerator bias. If not set, inferred automatically.")
    parser.add_argument("-a", "--attention_mechanism", type=str, choices=["gqa", "mla"], default=None,
                        help="Specify attention mechanism ('gqa' or 'mla'). If not set, inferred automatically.")

    args = parser.parse_args()

    calculate_information_capacity(
        model_path=args.model_path,
        data_path=args.data_path,
        max_sample_length=args.max_sample_length,
        batch_size=args.batch_size,
        numerator_bias=args.numerator_bias,
        attention_mechanism=args.attention_mechanism,
    )


if __name__ == "__main__":
    main()