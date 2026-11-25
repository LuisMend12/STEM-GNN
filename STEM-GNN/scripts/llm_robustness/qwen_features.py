"""Node features for Cora from a local Qwen model: mean of the last hidden layer.

Each node's title and abstract (same MAX_CHARS truncation as the prior) goes through
the model with no prompt and no labels; hidden states are averaged over real tokens.
Only node features change: the graph structure is never derived from these vectors.

Usage (from repo root):
  python STEM-GNN/scripts/llm_robustness/qwen_features.py --device cuda
"""
import argparse
import hashlib
import json
import os
import time

import torch

from qwen_prior import CORA, MAX_CHARS, OUT


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token states over the attention mask, whichever side is padded."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer

    texts = list(torch.load(CORA, weights_only=False).raw_texts)
    inputs = [t[:MAX_CHARS].strip() for t in texts]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    # Longest first keeps padding small; results are written back by node id.
    order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
    feats = None
    t0 = time.time()
    for b in range(0, len(order), args.batch):
        idx = order[b:b + args.batch]
        batch = tok([inputs[i] for i in idx], return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            pooled = mean_pool(model(**batch).last_hidden_state, batch["attention_mask"]).float().cpu()
        if feats is None:
            feats = torch.full((len(texts), pooled.shape[-1]), float("nan"))
        feats[idx] = pooled
        if b % (args.batch * 20) == 0:
            print(json.dumps({"done": b + len(idx), "of": len(order), "s": round(time.time() - t0, 1)}), flush=True)
    if not torch.isfinite(feats).all():
        raise RuntimeError("some node features are missing or non-finite")
    meta = {"model": args.model, "dtype": args.dtype, "max_chars": MAX_CHARS, "pool": "mean_last_hidden",
            "n_nodes": len(texts), "dim": feats.shape[-1],
            "text_hash": hashlib.sha256("\x00".join(texts).encode()).hexdigest()}
    path = os.path.join(OUT, f"qwen_features_{args.model.split('/')[-1]}.pt")
    torch.save({"meta": meta, "features": feats}, path)
    print(json.dumps({"saved": path, "dim": feats.shape[-1], "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
