#!/usr/bin/env python3
"""Write Hugging Face reference ("golden") dumps for llmhost correctness tests.

Runs the official transformers implementation on CPU in fp32 (eager attention)
and writes one .npz per prompt plus a manifest.json:

  ids            int64  [T]          prompt token ids (tokenizer output, BOS included)
  embed          f32    [T, d]       output of embed_tokens
  layer_XX       f32    [T, d]       residual stream after decoder layer XX
  final_norm     f32    [T, d]       output of the final norm (embedding_norm for lfm2)
  logits         f32    [T, V]       lm_head output for every prompt position
  greedy_ids     int64  [G]          greedy continuation (fp32, no sampling), if --greedy > 0
  greedy_logits  f32    [G, V]       logits that chose each greedy token

Golden files are large and are not committed. manifest.json records the model
revision, library versions and sha256 of every file so a run can be reproduced
and checked.

Usage:
  python tools/ref_dump.py --model LiquidAI/LFM2-700M --out tests/golden/lfm2-700m
"""
import argparse, hashlib, json, os, platform, sys, time

import numpy as np
import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    # Short plain text: smallest case, first thing to get right.
    "short_en": "The capital of France is",
    # Code: different token statistics.
    "code_py": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    # Non-ASCII / byte-level BPE paths.
    "multilingual": "Grüße aus Berlin! 東京は日本の首都です。Привет, мир. 🚀",
    # Long enough to exercise conv state across many positions and all attention layers.
    "long_en": (
        "Liquid foundation models combine short gated convolutions with a small number of "
        "grouped-query attention layers. Because most layers are convolutions, the decode-time "
        "state is tiny: each convolution layer only needs to remember the last few tokens, while "
        "the attention layers keep a key-value cache. This makes the models attractive for "
        "on-device inference, where memory bandwidth, not compute, limits how fast tokens can be "
        "generated. In this test we feed a paragraph of a few hundred tokens so that every layer "
        "sees a realistic amount of history, and we compare every hidden state against the "
        "reference implementation. "
    ) * 3,
}
CHAT = {
    # Chat-templated prompt: checks template rendering + special tokens.
    "chat_basic": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Name three primary colors."},
    ],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2-700M")
    ap.add_argument("--revision", default=None, help="HF commit sha; defaults to current main, recorded in manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", type=int, default=128, help="greedy tokens to generate per prompt (0 = off)")
    ap.add_argument("--greedy-prompts", default="short_en,chat_basic,code_py")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    rev = args.revision or HfApi().model_info(args.model).sha
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=rev, dtype=dtype, attn_implementation="eager", low_cpu_mem_usage=True
    ).eval()

    base = model.model
    layers = base.layers
    final_norm = getattr(base, "embedding_norm", None) or getattr(base, "norm")
    captured = {}

    def hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t.detach().to(torch.float32)[0].numpy().copy()
        return fn

    handles = [base.embed_tokens.register_forward_hook(hook("embed"))]
    handles += [l.register_forward_hook(hook(f"layer_{i:02d}")) for i, l in enumerate(layers)]
    handles.append(final_norm.register_forward_hook(hook("final_norm")))

    prompts = {k: tok(v, return_tensors="pt").input_ids for k, v in PROMPTS.items()}
    for k, msgs in CHAT.items():
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts[k] = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        with open(os.path.join(args.out, f"{k}.template.txt"), "w") as f:
            f.write(text)

    files = {}
    greedy_set = set(filter(None, args.greedy_prompts.split(",")))
    for name, ids in prompts.items():
        captured.clear()
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        rec = dict(captured)
        rec["ids"] = ids[0].numpy().astype(np.int64)
        rec["logits"] = out.logits[0].to(torch.float32).numpy()
        if args.greedy and name in greedy_set:
            for h in handles:
                h.remove()
            with torch.no_grad():
                g = model.generate(
                    # Explicit all-ones mask: otherwise generate() infers padding from
                    # pad_token_id == eos and masks real EOS tokens inside chat prompts.
                    ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.greedy, do_sample=False, output_logits=True,
                    return_dict_in_generate=True, eos_token_id=None, pad_token_id=tok.eos_token_id,
                )
            rec["greedy_ids"] = g.sequences[0, ids.shape[1]:].numpy().astype(np.int64)
            rec["greedy_logits"] = torch.stack(g.logits, 0)[:, 0].to(torch.float32).numpy()
            handles = [base.embed_tokens.register_forward_hook(hook("embed"))]
            handles += [l.register_forward_hook(hook(f"layer_{i:02d}")) for i, l in enumerate(layers)]
            handles.append(final_norm.register_forward_hook(hook("final_norm")))
        path = os.path.join(args.out, f"{name}.npz")
        np.savez(path, **rec)
        files[name] = {
            "file": f"{name}.npz", "sha256": sha256(path), "tokens": int(ids.shape[1]),
            "greedy_tokens": int(rec["greedy_ids"].shape[0]) if "greedy_ids" in rec else 0,
            "keys": sorted(rec.keys()), "seconds": round(time.time() - t0, 2),
        }
        print(f"{name}: T={ids.shape[1]} keys={len(rec)} {files[name]['seconds']}s", flush=True)

    manifest = {
        "model": args.model, "revision": rev, "dtype": args.dtype, "attn_implementation": "eager",
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "machine": platform.machine(), "threads": args.threads,
        "num_layers": len(layers), "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "layer_types": getattr(model.config, "layer_types", None),
        "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
        "prompts": {k: PROMPTS.get(k) for k in PROMPTS} | {k: CHAT[k] for k in CHAT},
        "files": files,
        "layer_capture": "residual stream after each decoder layer (forward hook on layers[i])",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
