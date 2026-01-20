import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# USER IMPORTS (edit this to match your file/module name)
# -----------------------------------------------------------------------------
# Expect your model code to define:
# - GPTConfig
# - GPT
#
# Example: from model import GPT, GPTConfig
try:
    from mistral_model import GPT, GPTConfig  # <-- rename "model" to your filename w/o .py
except Exception as e:
    print("FAILED to import your model. Edit the import in grade_mistral_exam.py.")
    raise


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def set_deterministic(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol=1e-5, rtol=1e-4):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"{name}: not close (max_abs={max_abs:.3e})")


def report(ok: bool, part: str, msg: str = ""):
    status = "PASS" if ok else "FAIL"
    if msg:
        print(f"[{status}] {part}: {msg}")
    else:
        print(f"[{status}] {part}")


def make_toy_vocab(vocab_size: int, device: str):
    # toy input ids
    B, T = 2, 16
    idx = torch.randint(low=0, high=vocab_size, size=(B, T), device=device)
    targets = torch.randint(low=0, high=vocab_size, size=(B, T), device=device)
    return idx, targets


def run_forward(model, idx, targets=None):
    out = model(idx, targets)
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise AssertionError("Expected model.forward to return (logits, loss).")
    logits, loss = out
    return logits, loss


def run_generate(model, idx, max_new_tokens: int, temperature=1.0, top_k=None):
    # Expect your GPT has .generate signature similar to your current one
    return model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)


# -----------------------------------------------------------------------------
# PART CHECKS
# -----------------------------------------------------------------------------
def part0_sanity(config: GPTConfig, model: torch.nn.Module, device: str):
    """
    Part 0: basic forward pass, no attribute errors, shapes correct.
    """
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        # If you kept vocab_size external, that's okay, but then your model should still run.
        vocab_size = 64

    idx, targets = make_toy_vocab(vocab_size=vocab_size, device=device)
    logits, loss = run_forward(model, idx, targets)

    if logits.ndim != 3:
        raise AssertionError(f"logits must be (B,T,V). Got {tuple(logits.shape)}")
    if logits.shape[:2] != idx.shape:
        raise AssertionError(f"logits first dims must match idx. logits={tuple(logits.shape)} idx={tuple(idx.shape)}")
    if logits.shape[-1] != vocab_size:
        raise AssertionError(f"logits last dim must be vocab_size={vocab_size}. Got {logits.shape[-1]}")
    if loss is None or loss.ndim != 0:
        raise AssertionError("loss must be a scalar tensor when targets provided.")


def part1_prenorm_structure(model: torch.nn.Module):
    """
    Part 1: pre-norm consistency.
    We can't fully prove the residual ordering, but we can sanity check:
    - RMSNorm modules exist and are used
    - No LayerNorm-only leftovers (not required, but typically Mistral-like uses RMSNorm)
    """
    # Basic module presence check
    norms = [m for m in model.modules() if m.__class__.__name__.lower().endswith("rmsnorm")]
    if len(norms) == 0:
        raise AssertionError("Expected RMSNorm modules (Mistral/LLaMA-style). Found none.")

    # Optional: ensure at least one attention checks
    attn_modules = [m for m in model.modules() if m.__class__.__name__.lower().find("attention") >= 0]
    if len(attn_modules) == 0:
        raise AssertionError("Expected an attention module to exist. Found none.")


def part2_rope_present(model: torch.nn.Module):
    """
    Part 2: RoPE present and absolute position embeddings removed/unused.
    We can check for presence of an embedding called 'wpe' and fail if it still exists.
    If you kept it but unused, this check will still fail—intentionally strict.
    """
    # We look for a submodule name "wpe" inside transformer
    module_names = set(dict(model.named_modules()).keys())
    buffer_names = set(dict(model.named_buffers()).keys())

    names = module_names | buffer_names
    if any(name.endswith("wpe") for name in names):
        raise AssertionError("Found positional embedding module 'wpe'. Mistral-like should use RoPE instead.")

    # RoPE existence check: you likely added a helper or buffer with 'rope' in its name
    # This is heuristic. Feel free to rename to satisfy this check.
    rope_like = [name for name in names if "rope" in name.lower()]
    if len(rope_like) == 0:
        raise AssertionError("Couldn't find any module/buffer with 'rope' in its name. Add RoPE helper/module.")


def part3_gqa_head_config(config: GPTConfig):
    """
    Part 3: GQA config check.
    Expect config.n_kv_head (or similar) and divisibility constraint.
    """
    n_head = getattr(config, "n_head", None)
    n_kv = getattr(config, "n_kv_head", None)
    if n_head is None:
        raise AssertionError("config.n_head missing.")
    if n_kv is None:
        raise AssertionError("config.n_kv_head missing (needed for GQA).")
    if n_head % n_kv != 0:
        raise AssertionError(f"Expected n_head % n_kv_head == 0, got {n_head} % {n_kv} != 0")


def part4_swa_mask_behavior(config: GPTConfig, model: torch.nn.Module, device: str):
    """
    Part 4: Sliding window behavior.
    This is tricky to verify without internal hooks.
    We do a functional test:
      - Create two inputs differing only in tokens *older than window*
      - If SWA is enabled, the last token logits should match closely
    """
    window = getattr(config, "window_size", None)
    if window is None:
        raise AssertionError("config.window_size missing (needed for sliding-window attention).")

    vocab_size = getattr(config, "vocab_size", 64)
    B, T = 1, max(16, window + 4)

    idx1 = torch.randint(0, vocab_size, (B, T), device=device)
    idx2 = idx1.clone()

    # Perturb tokens that are definitely outside the window for the final position.
    # Last position attends only to last `window` tokens.
    # So change earliest part: positions [0 : T-window-1]
    cut = max(0, T - window - 1)
    if cut == 0:
        # If T is too small, can't test
        raise AssertionError("Sequence too short to test sliding window masking. Increase T or reduce window.")

    idx2[:, :cut] = torch.randint(0, vocab_size, (B, cut), device=device)

    model.eval()
    with torch.no_grad():
        logits1, _ = run_forward(model, idx1, targets=None)
        logits2, _ = run_forward(model, idx2, targets=None)

    # Compare last-token logits
    last1 = logits1[:, -1, :]
    last2 = logits2[:, -1, :]
    assert_close(last1, last2, name="SWA last-token invariance test", atol=1e-3, rtol=1e-3)


def part5_cache_api(model: torch.nn.Module):
    """
    Part 5: KV-cache API existence.
    Expect attention forward signature supports cache.
    Since we cannot introspect signature reliably with torch.compile, we check by attribute hints.
    """
    # Heuristic: look for a method or attribute used for caching
    # You may add something like:
    # - model.forward(..., use_cache=True, past_kv=...)
    # - or attention.forward(..., past_kv=...)
    has_cache_flag = False
    for name, _ in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            has_cache_flag = True
            break
    if not has_cache_flag:
        raise AssertionError("Could not locate attention modules to validate cache API.")

    # Strictest check would require calling with use_cache/past_kv.
    # We'll do a dynamic call on model.forward if it supports kwargs.
    # You should update your forward to accept use_cache/past_kv and return present_kv.
    # This test intentionally fails until you implement that.
    import inspect

    sig = inspect.signature(model.forward)
    params = sig.parameters
    if "use_cache" not in params:
        raise AssertionError("model.forward must accept use_cache=... kwarg for KV caching.")
    if "past_kv" not in params:
        raise AssertionError("model.forward must accept past_kv=... kwarg for KV caching.")
    # Also check return convention
    # Expect: logits, loss, present_kv   (or logits, loss, cache)
    # We'll only check that returning 3 items works when use_cache=True
    # (Your model may choose a different API; then edit this harness accordingly.)


def part5_cache_equivalence(config: GPTConfig, model: torch.nn.Module, device: str):
    """
    Cache equivalence test:
    - Run generation with and without cache
    - Expect token outputs identical under deterministic sampling.
    """
    vocab_size = getattr(config, "vocab_size", 64)

    # Deterministic generation: temperature=1, but sampling introduces randomness.
    # We'll set a fixed seed before each run.
    prompt = torch.randint(0, vocab_size, (1, 8), device=device)

    model.eval()

    # You must implement a way to disable cache for comparison.
    # Suggested: model.generate(..., use_cache=False/True)
    # This harness expects generate signature supports use_cache.
    import inspect
    gen_sig = inspect.signature(model.generate)
    if "use_cache" not in gen_sig.parameters:
        raise AssertionError("generate() must accept use_cache=... to test cached vs uncached generation.")

    set_deterministic(123)
    out_nocache = model.generate(prompt.clone(), max_new_tokens=16, temperature=1.0, top_k=None, use_cache=False)

    set_deterministic(123)
    out_cache = model.generate(prompt.clone(), max_new_tokens=16, temperature=1.0, top_k=None, use_cache=True)

    if not torch.equal(out_nocache, out_cache):
        raise AssertionError("Cached and non-cached generation differ. Cache logic likely incorrect.")


def part6_rolling_cache_limit(config: GPTConfig, model: torch.nn.Module, device: str):
    """
    Rolling KV-cache limit:
    Ensure cache does not grow beyond window_size when SWA+cache enabled.
    This requires your model to expose cache length for debugging.
    """
    window = getattr(config, "window_size", None)
    if window is None:
        raise AssertionError("config.window_size missing.")

    # You must expose a debug getter for cache length, e.g. model.get_cache_len()
    if not hasattr(model, "get_cache_len"):
        raise AssertionError("Model must define get_cache_len() for grading rolling cache length.")

    vocab_size = getattr(config, "vocab_size", 64)
    prompt = torch.randint(0, vocab_size, (1, 8), device=device)

    model.eval()
    set_deterministic(0)

    _ = model.generate(prompt, max_new_tokens=window + 10, temperature=1.0, top_k=None, use_cache=True)

    cache_len = model.get_cache_len()
    if cache_len > window:
        raise AssertionError(f"Rolling cache too large: cache_len={cache_len} > window_size={window}")


def part8_swiglu_present(model: torch.nn.Module):
    """
    Part 8: SwiGLU MLP presence.
    We detect SiLU usage (or 'silu' functional) and at least two linear projections in MLPs.
    """
    # Check presence of SiLU module usage
    silu_modules = [m for m in model.modules() if m.__class__.__name__.lower() == "silu"]
    # Some people use F.silu without a module; we'll also accept a named module with 'swiglu'
    named = dict(model.named_modules()).keys()
    swiglu_like = [n for n in named if "swiglu" in n.lower()]

    if len(silu_modules) == 0 and len(swiglu_like) == 0:
        raise AssertionError("No SiLU/SwiGLU detected. Replace GELU MLP with SwiGLU-style gating.")

    # Heuristic: your MLP should have >1 linear proj (gate/up/down)
    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    if linear_count < 3:
        raise AssertionError("Too few Linear layers overall; SwiGLU requires multiple projections.")


def part9_topk_generation_correctness(config: GPTConfig, model: torch.nn.Module, device: str):
    """
    Part 9: top_k generation should not crash and must return valid tokens in vocab range.
    """
    vocab_size = getattr(config, "vocab_size", 64)
    prompt = torch.randint(0, vocab_size, (1, 8), device=device)
    model.eval()
    set_deterministic(999)

    out = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10, use_cache=True)

    if out.min().item() < 0 or out.max().item() >= vocab_size:
        raise AssertionError("Generated tokens out of vocab range.")


# -----------------------------------------------------------------------------
# MAIN GRADER
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--strict", action="store_true", help="Fail fast instead of continuing.")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    set_deterministic(1337)

    # -------------------------------------------------------------------------
    # Build a config for the exam.
    # IMPORTANT: You should update your GPTConfig defaults so these fields exist.
    # -------------------------------------------------------------------------
    config = GPTConfig()

    # For grading, we need these fields to exist after you upgrade your config.
    # If you keep vocab_size/block_size external, adapt this harness accordingly.
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 64
    if not hasattr(config, "block_size"):
        config.block_size = 128

    # Make these present so parts can run once implemented.
    # Your exam expects you to add them properly.
    if not hasattr(config, "n_kv_head"):
        # Leave it unset to force Part 3 failure until you implement it.
        pass
    if not hasattr(config, "window_size"):
        # Leave it unset to force Part 4 failure until you implement it.
        pass

    # Build model
    try:
        model = GPT(config=config, vocab_size=config.vocab_size, block_size=config.block_size).to(device)
    except TypeError:
        # If you changed the constructor to only take config, that's fine.
        model = GPT(config=config).to(device)

    # -------------------------------------------------------------------------
    # Run parts
    # -------------------------------------------------------------------------
    parts = []

    parts.append(("Part 0: Sanity", lambda: part0_sanity(config, model, device)))
    parts.append(("Part 1: Pre-Norm structure", lambda: part1_prenorm_structure(model)))

    # These will fail until you implement the corresponding upgrades:
    parts.append(("Part 2: RoPE present & no wpe", lambda: part2_rope_present(model)))
    parts.append(("Part 3: GQA config", lambda: part3_gqa_head_config(config)))
    parts.append(("Part 4: Sliding window invariance", lambda: part4_swa_mask_behavior(config, model, device)))

    # Cache parts
    parts.append(("Part 5a: Cache API exists", lambda: part5_cache_api(model)))
    parts.append(("Part 5b: Cache equivalence in generation", lambda: part5_cache_equivalence(config, model, device)))
    parts.append(("Part 6: Rolling cache length bounded", lambda: part6_rolling_cache_limit(config, model, device)))

    # MLP + topk checks
    parts.append(("Part 8: SwiGLU detected", lambda: part8_swiglu_present(model)))
    parts.append(("Part 9: top_k generation correctness", lambda: part9_topk_generation_correctness(config, model, device)))

    # Execute
    failures = 0
    for title, fn in parts:
        try:
            fn()
            report(True, title)
        except Exception as e:
            failures += 1
            report(False, title, msg=str(e))
            if args.strict:
                break

    print("-" * 80)
    if failures == 0:
        print("ALL PARTS PASSED")
        sys.exit(0)
    else:
        print(f"{failures} PART(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
