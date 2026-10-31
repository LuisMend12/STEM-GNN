"""Zero-shot class prior for every Cora node from a local Qwen instruct model (v2).

Qwen reads the paper's title and abstract plus the 7 category names; we take the
next-token logits of the space-prefixed option letters " A".." G" after the prompt
"... The answer is" (v1 used a trailing space and bare letters, which is
tokenization-inconsistent; see experiments/llm-anchor/protocol.md Amendment 1).
One forward pass per node, no generation, no labels. Raw letter logits are cached
(for validation-only temperature calibration) with metadata; resuming refuses a
cache whose metadata differs.

--perm k rotates the option order by k (class i is shown at letter (i+k) % 7) and maps
the scores back to the original class order: used for the position-bias audit.

The prior is a training signal only; the GNN still makes every prediction and the
graph structure is never weighted by it.

Usage (from repo root):
  python STEM-GNN/scripts/llm_robustness/qwen_prior.py --threads 14
  python STEM-GNN/scripts/llm_robustness/qwen_prior.py --perm 3 --nodes_file audit_nodes.txt
"""
import argparse
import hashlib
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
VERSION = "v2"


def order(n_cls, perm):
    """shown[j] = class index displayed at letter j."""
    return [(j - perm) % n_cls for j in range(n_cls)]


def build_prompt(tok, text, label_names, perm=0):
    shown = order(len(label_names), perm)
    options = "\n".join(f"{LETTERS[j]}. {label_names[c].replace('_', ' ')}" for j, c in enumerate(shown))
    msg = ("Classify this machine-learning paper into exactly one category.\n\n"
           f"Paper: {text[:MAX_CHARS].strip()}\n\nCategories:\n{options}\n\n"
           "Answer with the single letter of the best category.")
    chat = [{"role": "user", "content": msg}]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + "The answer is"


def letter_ids(tok, sample_prompt):
    """Ids of ' A'..' G', asserted single-token and context-consistent after the prompt."""
    base = tok(sample_prompt, add_special_tokens=False)["input_ids"]
    ids = []
    for ch in LETTERS:
        enc = tok.encode(" " + ch, add_special_tokens=False)
        full = tok(sample_prompt + " " + ch, add_special_tokens=False)["input_ids"]
        if len(enc) != 1 or full[:-1] != base or full[-1] != enc[0]:
            raise ValueError(f"' {ch}' is not a context-consistent single token: {enc}")
        ids.append(enc[0])
    return ids


def metadata(args, names, texts, tok_name, prompt_template):
    return {"version": VERSION, "model": args.model, "tokenizer": tok_name, "dtype": args.dtype,
            "max_chars": MAX_CHARS, "label_order": names, "perm": args.perm, "n_nodes": len(texts),
            "text_hash": hashlib.sha256("\x00".join(texts).encode()).hexdigest(),
            "prompt_hash": hashlib.sha256(prompt_template.encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N nodes (timing check)")
    ap.add_argument("--nodes_file", default=None, help="score only these node ids (one per line)")
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = torch.load(CORA, weights_only=False)
    texts, labels, names = list(data.raw_texts), data.y, list(data.label_names)
    if args.nodes_file:
        nodes = [int(l) for l in open(args.nodes_file) if l.strip()]
    else:
        nodes = list(range(len(texts) if args.limit is None else min(args.limit, len(texts))))
    tag = f"{args.model.split('/')[-1]}_{VERSION}_perm{args.perm}" + ("_audit" if args.nodes_file else "")
    ckpt = os.path.join(OUT, f"qwen_prior_{tag}.partial.pt")
    final = os.path.join(OUT, f"qwen_prior_{tag}.pt")
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    ids = letter_ids(tok, build_prompt(tok, texts[0], names, args.perm))
    shown = order(len(names), args.perm)
    meta = metadata(args, names, texts, args.model, build_prompt(tok, "", names, args.perm))

    store = {"meta": meta, "logits": torch.full((len(texts), len(names)), float("nan")),
             "letter_mass": torch.full((len(texts),), float("nan"))}
    if os.path.exists(ckpt) and args.limit is None:
        old = torch.load(ckpt)
        if old.get("meta") != meta:
            raise RuntimeError("partial cache metadata differs from this configuration; refusing to resume")
        store = old
    todo = [i for i in nodes if torch.isnan(store["logits"][i, 0])]
    t0 = time.time()
    for b in range(0, len(todo), args.batch):
        idx = todo[b:b + args.batch]
        batch = tok([build_prompt(tok, texts[i], names, args.perm) for i in idx], return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            full = model(**batch).logits[:, -1, :].float().cpu()
        letter_logits = full[:, ids]                     # [B, 7] in displayed-letter order
        back = torch.empty_like(letter_logits)
        back[:, shown] = letter_logits                   # map to original class order
        store["logits"][idx] = back
        store["letter_mass"][idx] = torch.softmax(full, -1)[:, ids].sum(-1)
        done = b + len(idx)
        if done % 200 < args.batch or done == len(todo):
            rate = done / (time.time() - t0)
            print(json.dumps({"done": done, "todo": len(todo), "nodes_per_s": round(rate, 3),
                              "eta_min": round((len(todo) - done) / rate / 60, 1)}), flush=True)
            if args.limit is None:
                torch.save(store, ckpt)

    have = ~torch.isnan(store["logits"][:, 0])
    probs = torch.softmax(store["logits"][have], -1)
    acc = (probs.argmax(-1) == labels[have]).float().mean().item()
    print(json.dumps({"nodes_scored": int(have.sum()), "zero_shot_acc": round(acc, 4),
                      "mean_letter_mass": round(store["letter_mass"][have].mean().item(), 4),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    if args.limit is None and bool(have[nodes].all()):
        torch.save(store, final)


if __name__ == "__main__":
    main()
