import json
from pathlib import Path
from typing import Union, Dict, Optional

def gqa_model_theoretical_flops(
    config_path: Union[str, Path],
    seq_len: int = 0,
    gen_len: int = 1024,
    batch_size: int = 1,
    prefill_logits: str = "all",   # "all" | "last" | "none"
) -> Dict[str, float]:
    """
    Compute theoretical FLOPs for an LLM with GQA given its Hugging Face config.json.

    Assumptions (dense Transformer, forward only):
    - 2 FLOPs per multiply-add.
    - Attention = dense GQA: Q & O project to d_model; K/V project to n_kv_heads * d_k
      where d_k = d_model / n_heads.
    - Attention core cost includes QK^T and softmax(QK^T) @ V.
    - MLP = gated (SwiGLU-like): two "up" matmuls + one "down" matmul. (handles special 
      cases of llama-4 and gpt-oss)
    - LM head (final logits) included; at prefill you can count logits for:
        * "all": logits for every prompt token (matches HF's default forward outputs),
        * "last": logits only for last prompt token (some gens do this),
        * "none": if you never materialize logits at prefill.
      At decode, logits are computed every step.

    Returns (TFLOPs):
        dict with detailed breakdown for prefill, decode, totals.
    """
    # ---- load config ----
    if "Ruyi" in config_path:
        import re
        pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:B|billion)', re.IGNORECASE)
        match = pattern.search(config_path)
        config_path = config_path.replace(match.group(0), "7B")
    cfg_path = Path(config_path)
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    if "gemma-3" in config_path:
        import re
        pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:B|billion)', re.IGNORECASE)
        match = pattern.search(config_path)
        param_count = float(match.group(1))
        if param_count >= 4:
            cfg = cfg["text_config"]
            cfg["vocab_size"] = 262208
            if param_count == 4: cfg["num_attention_heads"] = 8; cfg["num_key_value_heads"] = 4
            elif param_count == 12: cfg["num_attention_heads"] = 16; cfg["num_key_value_heads"] = 8
            elif param_count == 27: cfg["num_attention_heads"] = 32; cfg["num_key_value_heads"] = 16
    if "Llama-4" in config_path:
        cfg = cfg["text_config"]

    # ---- required hyperparams ----
    d_model = int(cfg["hidden_size"])
    n_layers = int(cfg.get("num_hidden_layers", cfg.get("n_layer"))) if "Ruyi" not in config_path else int(match.group(1)) * 4
    n_heads = int(cfg.get("num_attention_heads", cfg.get("n_head")))
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    if "Llama-4" in config_path: d_ff = cfg["intermediate_size_mlp"]
    elif ("Qwen1.5" in config_path or "Qwen2-" in config_path) and "B-A" in config_path:
        d_ff = cfg["intermediate_size"] + cfg["shared_expert_intermediate_size"]
    else: d_ff = int(cfg.get("intermediate_size", cfg.get("ffn_hidden_size"))) # llama-4 uses intermediate_size_mlp for main mlp
    vocab_size = int(cfg["vocab_size"])

    # per-head dimension (assume divisible)
    d_k = d_model // n_heads
    kv_dim = n_kv_heads * d_k

    B = batch_size
    L = seq_len
    T = gen_len

    # ---- helpers (FLOPs, not TFLOPs) ----
    # Projections per layer for a sequence of length L
    # Q: 2 * B * L * d_model * d_model
    # O: same
    # K,V: 2 * B * L * d_model * kv_dim each
    def proj_flops(L_tokens: int) -> int:
        q = 2 * B * L_tokens * d_model * d_model
        o = 2 * B * L_tokens * d_model * d_model
        k = 2 * B * L_tokens * d_model * kv_dim
        v = 2 * B * L_tokens * d_model * kv_dim
        return q + k + v + o

    # Attention core per layer
    # Prefill (quadratic): QK^T + (softmax@V) ≈ 4 * B * n_heads * L^2 * d_k
    # Decode (one step over cache length C): ≈ 4 * B * n_heads * C * d_k
    def attn_core_prefill_flops(L_tokens: int) -> int:
        return 4 * B * n_heads * (L_tokens ** 2) * d_k

    def attn_core_decode_flops(cache_len: int) -> int:
        return 4 * B * n_heads * cache_len * d_k

    # MLP per layer
    # Two up matmuls + one down: 2*B*L*d_model*d_ff + 2*B*L*d_model*d_ff + 2*B*L*d_ff*d_model = 6*B*L*d_model*d_ff
    def mlp_flops(L_tokens: int) -> int:
        # gpt-oss does not use gate function (6 → 4), registers per-expert intermediate size
        if "gpt-oss" in config_path: return 4 * B * L_tokens * d_model * d_ff * int(cfg["num_experts_per_tok"]) 
        # llama-4 use 2-layer mlp without gating on attn score, before the main mlp
        elif "Llama-4" in config_path: return B * L_tokens * d_model * (6 * d_ff + 4 * int(cfg["intermediate_size"])) 
        else: return 6 * B * L_tokens * d_model * d_ff

    # LM head (final linear to vocab) for N tokens: 2 * B * N * d_model * vocab_size
    def lm_head_flops(num_tokens: int) -> int:
        return 2 * B * num_tokens * d_model * vocab_size

    # ---- prefill (length L) ----
    proj_prefill_per_layer = proj_flops(L)
    attn_prefill_per_layer = attn_core_prefill_flops(L)
    mlp_prefill_per_layer  = mlp_flops(L)

    stack_prefill = n_layers * (proj_prefill_per_layer + attn_prefill_per_layer + mlp_prefill_per_layer)

    if prefill_logits == "all":
        lm_prefill = lm_head_flops(L)
    elif prefill_logits == "last":
        lm_prefill = lm_head_flops(1)
    elif prefill_logits == "none":
        lm_prefill = 0
    else:
        raise ValueError("prefill_logits must be one of {'all','last','none'}")

    prefill_total = stack_prefill + lm_prefill

    # ---- decode (T steps) ----
    # For each step, projections/MLP are for 1 new token.
    proj_decode_per_layer_per_step = proj_flops(1)
    mlp_decode_per_layer_per_step  = mlp_flops(1)

    # Attention core sums over growing cache lengths: L, L+1, ..., L+T-1
    # Sum_{t=0..T-1} 4 * B * n_heads * (L + t) * d_k = 4 * B * n_heads * d_k * (T*L + T*(T-1)/2)
    attn_decode_per_layer_total = 4 * B * n_heads * d_k * (T * L + (T * (T - 1)) // 2)

    stack_decode = n_layers * (
        T * (proj_decode_per_layer_per_step + mlp_decode_per_layer_per_step) + attn_decode_per_layer_total
    )

    # Logits at each decode step
    lm_decode = lm_head_flops(T)

    decode_total = stack_decode + lm_decode

    # ---- packing results (TFLOPs) ----
    toT = lambda x: x / 1e12

    results = {
        # Inputs
        "batch_size": B,
        "seq_len": L,
        "gen_len": T,
        "hidden_size": d_model,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_kv_heads": n_kv_heads,
        "intermediate_size": d_ff,
        "vocab_size": vocab_size,
        "prefill_logits_mode": prefill_logits,

        # Prefill breakdown
        "prefill_stack_TFLOPs": toT(stack_prefill),
        "prefill_proj_TFLOPs": toT(n_layers * proj_prefill_per_layer),
        "prefill_attn_core_TFLOPs": toT(n_layers * attn_prefill_per_layer),
        "prefill_mlp_TFLOPs": toT(n_layers * mlp_prefill_per_layer),
        "prefill_lm_head_TFLOPs": toT(lm_prefill),
        "prefill_total_TFLOPs": toT(prefill_total),

        # Decode breakdown
        "decode_stack_TFLOPs": toT(stack_decode),
        "decode_proj_TFLOPs": toT(n_layers * T * proj_decode_per_layer_per_step),
        "decode_attn_core_TFLOPs": toT(n_layers * attn_decode_per_layer_total),
        "decode_mlp_TFLOPs": toT(n_layers * T * mlp_decode_per_layer_per_step),
        "decode_lm_head_TFLOPs": toT(lm_decode),
        "decode_total_TFLOPs": toT(decode_total),

        # Totals
        "request_total_TFLOPs": toT(prefill_total + decode_total),
        "avg_decode_TFLOPs_per_token": toT(decode_total / max(T, 1)),
    }
    return results

def mla_model_theoretical_flops(
    config_path: Union[str, Path],
    seq_len: int = 0,
    gen_len: int = 1024,
    batch_size: int = 1,
    prefill_logits: str = "all",   # "all" | "last" | "none"
    attention_type: Optional[str] = None,  # "mha" | "mla" | None (auto-detect)
    mla_latents: Optional[int] = None,
    mla_mode: str = "reuse",  # "reuse" | "recompute"
) -> Dict[str, float]:
    """
    Compute theoretical FLOPs (TFLOPs) for DeepSeek-R1 (or similar) inference.

    Key points & assumptions (be sure to read):
    - This function supports both classic dense Multi-Head Attention (MHA)
      and DeepSeek's Multi-Head Latent Attention (MLA). MLA reduces the
      attention core from O(L^2) to O(L * M) where M is the number of latent
      tokens (per head or global depending on implementation). See DeepSeek-V2/V3 papers.
      MLA also admits two execution schemes: 'reuse' (compute latent KV once at prefill
      and reuse during decode) and 'recompute' (recompute / update latents per step).
      The hardware analysis and community descriptions motivated these cost models.
    - MoE MLP: we model a single shared expert (always executed) plus `num_experts_per_tok`
      *activated* experts per token (as reported in the config). We expose separate
      FLOP entries for shared vs activated experts.
    - Projection FLOPs follow your previous convention: 2 FLOPs per multiply-add,
      and we keep the same projection accounting for Q/K/V/O. The attention *core* cost
      is replaced with MLA formulas when used.
    - Because MLA variants differ in implementation details across repos, you can pass
      `mla_latents` to set the latent length (if None a conservative default is used).
      The default is chosen to reflect a moderate compression (an inferrable but tunable value).
    - All counts are for forward-only inference, and result units are TFLOPs.

    Parameters:
        mla_latents: recommended to pass a locale-specific sensible value (e.g., 64, 128, 256).
                    If None, the function will pick a conservative default: min(256, max(1, seq_len // 16)).
        mla_mode: "reuse" (default) counts the one-time cost to build latents at prefill and
                  then low-cost per-step decode attention against the smaller latent set.
                  "recompute" falls back to recomputing compressed latents per decode step
                  — yielding higher compute but lower memory footprint (useful to model
                  alternate execution strategies). See hardware-centric analysis.
    """

    cfg_path = Path(config_path)
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # ---- required hyperparams ----
    d_model = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    d_ff = int(cfg.get("moe_intermediate_size", cfg.get("intermediate_size")))
    vocab_size = int(cfg["vocab_size"])

    # MoE-specific
    n_experts_total = int(cfg.get("n_routed_experts", cfg.get("num_experts", cfg.get("num_local_experts", 0))))
    n_shared_experts = int(cfg.get("n_shared_experts", cfg.get("n_shared_experts", 0)))
    n_experts_per_tok = int(cfg.get("num_experts_per_tok", cfg.get("num_experts_per_tok", 0)))

    # Detect/override attention type:
    cfg_model_type = cfg.get("model_type", "").lower()
    if attention_type is None:
        # If model type contains deepseek or config contains MLA-related fields, default to mla
        if "deepseek" in cfg_model_type or cfg.get("moa") or cfg.get("n_group") is not None:
            attention_type = "mla"
        else:
            attention_type = "mha"

    # MLA default latent length (tunable). MLA papers/reports show M << L; choose conservative default.
    if mla_latents is None:
        mla_latents = int(cfg.get("kv_lora_rank", max(1, min(256, max(1, seq_len // 16)))))

    # per-head dimension (assume divisible)
    d_k = d_model // n_heads
    kv_dim = n_kv_heads * d_k

    B = batch_size
    L = seq_len
    T = gen_len

    # ---- helpers (FLOPs, NOT TFLOPs) ----
    # Linear projections per layer for a sequence of length L_tokens.
    # Keep original projection accounting for Q, O, K, V (this counts the input linear layers).
    def proj_flops(L_tokens: int) -> int:
        q = 2 * B * L_tokens * d_model * d_model      # Wq : d_model x d_model
        o = 2 * B * L_tokens * d_model * d_model      # Wo : d_model x d_model (output projection)
        # For K and V we keep the same "dense" projection accounting here. MLA adds separate
        # compression costs which we model in attention_core_mla below.
        k = 2 * B * L_tokens * d_model * kv_dim
        v = 2 * B * L_tokens * d_model * kv_dim
        return q + k + v + o

    # Dense attention core (classic quadratic)
    def attn_core_prefill_mha(L_tokens: int) -> int:
        # approximate QK^T + softmax@V cost
        return 4 * B * n_heads * (L_tokens ** 2) * d_k

    def attn_core_decode_mha(cache_len: int) -> int:
        return 4 * B * n_heads * cache_len * d_k

    # MLA attention core (approximate): replace L^2 with L * M.
    # We model two things:
    # 1) core: Q @ K_latent^T and softmax@V_latent -> ~ 4 * B * n_heads * L * M * d_k
    # 2) one-time compression cost at prefill to build the latent K/V (approximation).
    #    hardware analyses show there are two execution schemes: re-use (compress once) vs recompute.
    #    We approximate the one-time compression cost as: 2 * B * L * d_model * (mla_latents / max(1,L))
    #    which simplifies to ~ 2 * B * d_model * mla_latents (a compact, tunable approximation).
    # See DeepSeek papers and hardware analysis for details.
    def attn_core_prefill_mla(L_tokens: int) -> int:
        M = mla_latents
        core = 4 * B * n_heads * L_tokens * M * d_k
        # one-time compress cost (approximation; tunable)
        compress = int(2 * B * d_model * M)
        return core + compress

    def attn_core_decode_mla_reuse(L_tokens: int, T_steps: int) -> int:
        # If latents are reused, each decode step attends Q (1 token) against latent keys size M:
        # cost per step ~ 4 * B * n_heads * M * d_k
        return 4 * B * n_heads * d_k * (T_steps * mla_latents)

    def attn_core_decode_mla_recompute(L_tokens: int, T_steps: int) -> int:
        # recomputing latents each step approximates back toward classic cost (worse-case).
        # fall back to the MHA-like growing-cache sum as conservative upper bound:
        return 4 * B * n_heads * d_k * (T_steps * L_tokens + (T_steps * (T_steps - 1)) // 2)

    # MLP costs:
    # Single expert (SwiGLU-like gated): approx 6 * B * L * d_model * d_ff
    def single_expert_flops(L_tokens: int) -> int:
        return 6 * B * L_tokens * d_model * d_ff

    # MoE MLP breakdown: shared experts (n_shared_experts) executed every token
    # plus activated experts (n_experts_per_tok) *per-token* (sparse routing).
    # Note: some implementations add extra routing overhead; we ignore the small routing bookkeeping cost here.
    def moe_mlp_flops_shared(L_tokens: int) -> int:
        # FLOPs for shared (always executed). If config says n_shared_experts>1, multiply accordingly.
        return n_shared_experts * single_expert_flops(L_tokens)

    def moe_mlp_flops_activated(L_tokens: int) -> int:
        # Activated experts per token: each token runs n_experts_per_tok experts (sparse).
        return n_experts_per_tok * single_expert_flops(L_tokens)

    # LM head
    def lm_head_flops(num_tokens: int) -> int:
        return 2 * B * num_tokens * d_model * vocab_size

    # ---- PREFILL (length L) ----
    proj_prefill_per_layer = proj_flops(L)

    if attention_type == "mha":
        attn_prefill_per_layer = attn_core_prefill_mha(L)
        # no extra MLA compress cost
        mla_extra_prefill_per_layer = 0
    elif attention_type == "mla":
        attn_prefill_per_layer = attn_core_prefill_mla(L)
        # the compression cost is included in attn_core_prefill_mla as 'compress' term
        mla_extra_prefill_per_layer = max(0, attn_prefill_per_layer - (4 * B * n_heads * (L ** 2) * d_k))
    else:
        raise ValueError("attention_type must be one of {'mha','mla'}")

    # MLP (MoE)
    mlp_prefill_shared_per_layer = moe_mlp_flops_shared(L)
    mlp_prefill_activated_per_layer = moe_mlp_flops_activated(L)
    mlp_prefill_per_layer = mlp_prefill_shared_per_layer + mlp_prefill_activated_per_layer

    stack_prefill = n_layers * (proj_prefill_per_layer + attn_prefill_per_layer + mlp_prefill_per_layer)

    if prefill_logits == "all":
        lm_prefill = lm_head_flops(L)
    elif prefill_logits == "last":
        lm_prefill = lm_head_flops(1)
    elif prefill_logits == "none":
        lm_prefill = 0
    else:
        raise ValueError("prefill_logits must be one of {'all','last','none'}")

    prefill_total = stack_prefill + lm_prefill

    # ---- DECODE (T steps) ----
    proj_decode_per_layer_per_step = proj_flops(1)
    mlp_decode_per_layer_per_step_shared = moe_mlp_flops_shared(1)
    mlp_decode_per_layer_per_step_activated = moe_mlp_flops_activated(1)
    mlp_decode_per_layer_per_step = mlp_decode_per_layer_per_step_shared + mlp_decode_per_layer_per_step_activated

    if attention_type == "mha":
        # attention grows with cache: L, L+1, ..., L+T-1
        attn_decode_per_layer_total = 4 * B * n_heads * d_k * (T * L + (T * (T - 1)) // 2)
        mla_extra_decode_term = 0
    else:  # mla
        if mla_mode == "reuse":
            attn_decode_per_layer_total = attn_core_decode_mla_reuse(L, T)
            mla_extra_decode_term = 0  # compression cost already accounted in prefill
        elif mla_mode == "recompute":
            attn_decode_per_layer_total = attn_core_decode_mla_recompute(L, T)
            # recompute implies we pay full compress-like cost in decode as well;
            # approximate by adding the same compress cost per layer per decode (conservative)
            per_step_compress = int(2 * B * d_model * mla_latents)
            mla_extra_decode_term = n_layers * (per_step_compress * T)
        else:
            raise ValueError("mla_mode must be one of {'reuse','recompute'}")

    stack_decode = n_layers * (
        T * (proj_decode_per_layer_per_step + mlp_decode_per_layer_per_step) + attn_decode_per_layer_total
    ) + mla_extra_decode_term

    lm_decode = lm_head_flops(T)
    decode_total = stack_decode + lm_decode

    # ---- pack results (TFLOPs) ----
    toT = lambda x: x / 1e12

    results = {
        # Inputs / config readout
        "batch_size": B,
        "seq_len": L,
        "gen_len": T,
        "hidden_size": d_model,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_kv_heads": n_kv_heads,
        "intermediate_size": d_ff,
        "vocab_size": vocab_size,
        "num_experts_total": n_experts_total,
        "num_shared_experts": n_shared_experts,
        "num_experts_per_tok": n_experts_per_tok,
        "attention_type": attention_type,
        "mla_latents": mla_latents if attention_type == "mla" else None,
        "mla_mode": mla_mode if attention_type == "mla" else None,
        "prefill_logits_mode": prefill_logits,

        # Prefill breakdown
        "prefill_stack_TFLOPs": toT(stack_prefill),
        "prefill_proj_TFLOPs": toT(n_layers * proj_prefill_per_layer),
        "prefill_attn_core_TFLOPs": toT(n_layers * attn_prefill_per_layer),
        "prefill_mlp_shared_TFLOPs": toT(n_layers * mlp_prefill_shared_per_layer),
        "prefill_mlp_activated_TFLOPs": toT(n_layers * mlp_prefill_activated_per_layer),
        "prefill_mlp_TFLOPs": toT(n_layers * mlp_prefill_per_layer),
        "prefill_lm_head_TFLOPs": toT(lm_prefill),
        "prefill_total_TFLOPs": toT(prefill_total),

        # Decode breakdown
        "decode_stack_TFLOPs": toT(stack_decode),
        "decode_proj_TFLOPs": toT(n_layers * T * proj_decode_per_layer_per_step),
        "decode_attn_core_TFLOPs": toT(n_layers * attn_decode_per_layer_total),
        "decode_mlp_shared_TFLOPs": toT(n_layers * T * mlp_decode_per_layer_per_step_shared),
        "decode_mlp_activated_TFLOPs": toT(n_layers * T * mlp_decode_per_layer_per_step_activated),
        "decode_mlp_TFLOPs": toT(n_layers * T * mlp_decode_per_layer_per_step),
        "decode_lm_head_TFLOPs": toT(lm_decode),
        "decode_total_TFLOPs": toT(decode_total),

        # Totals
        "request_total_TFLOPs": toT(prefill_total + decode_total),
        "avg_decode_TFLOPs_per_token": toT(decode_total / max(T, 1)),
    }

    return results