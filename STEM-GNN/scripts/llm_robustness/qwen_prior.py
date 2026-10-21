"""Zero-shot class prior for every Cora node from a local Qwen instruct model.

Qwen reads the paper's title and abstract and the 7 category names, and we
take the softmax over the next-token logits of the option letters A-G. One
forward pass per node, no generation, no labels. Results are checkpointed so
the job can be interrupted and resumed.

The prior is a robustness signal only (edge trust, pseudo-labels); the GNN
still makes every prediction.

Usage (from repo root):
  python STEM-GNN/scripts/llm_robustness/qwen_prior.py --threads 6
  python STEM-GNN/scripts/llm_robustness/qwen_prior.py --limit 16   # timing check
"""
import argparse
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
CORA = os.path.join(HERE, "..", "..", "dataset", "data", "single_graph", "Cora", "cora.pt")
OUT = os.path.join(ROOT, "experiments", "llm-robustness", "results")
LETTERS = "ABCDEFG"
MAX_CHARS = 1200


def build_prompt(tok, text, label_names):
    options = "\n".join(f"{LETTERS[i]}. {n.replace('_', ' ')}" for i, n in enumerate(label_names))
    msg = ("Classify this machine-learning paper into exactly one category.\n\n"
           f"Paper: {text[:MAX_CHARS].strip()}\n\nCategories:\n{options}\n\n"
           "Answer with the single letter of the best category.")
    chat = [{"role": "user", "content": msg}]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + "The answer is "


def letter_ids(tok):
    ids = []
    for ch in LETTERS:
        enc = tok.encode(ch, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {ch!r} is not a single token: {enc}")
        ids.append(enc[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N nodes (timing check)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = torch.load(CORA, weights_only=False)
    texts, labels, names = list(data.raw_texts), data.y, list(data.label_names)
    n = len(texts) if args.limit is None else min(args.limit, len(texts))
    tag = args.model.split("/")[-1]
    ckpt = os.path.join(OUT, f"qwen_prior_{tag}.partial.pt")
    final = os.path.join(OUT, f"qwen_prior_{tag}.pt")
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval()
    ids = letter_ids(tok)

    probs = torch.full((len(texts), len(names)), float("nan"))
    if os.path.exists(ckpt) and args.limit is None:
        probs = torch.load(ckpt)
    todo = [i for i in range(n) if torch.isnan(probs[i, 0])]
    t0 = time.time()
    for b in range(0, len(todo), args.batch):
        idx = todo[b:b + args.batch]
        batch = tok([build_prompt(tok, texts[i], names) for i in idx], return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float()
        probs[idx] = torch.softmax(logits[:, ids], dim=-1)
        done = b + len(idx)
        if done % 200 < args.batch or done == len(todo):
            rate = done / (time.time() - t0)
            print(json.dumps({"done": done, "todo": len(todo), "nodes_per_s": round(rate, 3),
                              "eta_min": round((len(todo) - done) / rate / 60, 1)}), flush=True)
            if args.limit is None:
                torch.save(probs, ckpt)

    have = ~torch.isnan(probs[:, 0])
    acc = (probs[have].argmax(-1) == labels[have]).float().mean().item()
    print(json.dumps({"nodes_scored": int(have.sum()), "zero_shot_acc": round(acc, 4),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    if args.limit is None and bool(have.all()):
        torch.save(probs, final)


if __name__ == "__main__":
    main()
