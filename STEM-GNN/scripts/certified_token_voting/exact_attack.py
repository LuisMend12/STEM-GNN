"""Exact-greedy structure attack on CTV (validates the PRBCD numbers).

CTV's score is linear in the edges, so the cheapest way to flip a node is
computable exactly: per target class c, sort all single-edge margin
reductions (insert an edge to j: v_jc - v_jy; delete neighbor j:
v_jy - v_jc) and take the smallest k whose cumulative reduction reaches the
margin. A global budget B is spent on the cheapest correctly-classified test
nodes first. Ignores the (attacker-favourable) reuse of one edge for two
targets, so it is a strong but not provably optimal global attack.

Usage (from repo root):
  python STEM-GNN/scripts/certified_token_voting/exact_attack.py --splits 0 1 2
"""
import argparse
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ctv_run", os.path.join(HERE, "run.py"))
ctv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctv)
probe, llm = ctv.probe, ctv.llm


def flip_costs(model, ei, y, test_idx):
    """Minimum number of incident edge edits to flip each test node (inf if impossible)."""
    v = model.votes
    scores = model(None, ei)
    n = v.size(0)
    nbr = [[] for _ in range(n)]
    for s, d in ei.t().tolist():
        nbr[d].append(s)
    costs = torch.full((test_idx.numel(),), float("inf"))
    for t, i in enumerate(test_idx.tolist()):
        yi = int(y[i])
        if scores[i].argmax() != yi:
            costs[t] = 0
            continue
        is_nbr = torch.zeros(n, dtype=torch.bool)
        is_nbr[nbr[i]] = True
        is_nbr[i] = True
        best = float("inf")
        for c in range(v.size(1)):
            if c == yi:
                continue
            margin = (scores[i, yi] - scores[i, c]).item()
            ins = (v[:, c] - v[:, yi])[~is_nbr]
            dele = (v[:, yi] - v[:, c])[torch.tensor(nbr[i], dtype=torch.long)] if nbr[i] else torch.zeros(0)
            gains = torch.cat([ins, dele]).sort(descending=True).values
            gains = gains[gains > 0]
            cum = gains.cumsum(0)
            hit = (cum >= margin).nonzero()
            if hit.numel():
                best = min(best, int(hit[0]) + 1)
        costs[t] = best
    return costs


def greedy_accuracy(costs, budget):
    correct = costs > 0
    spend, flipped = 0, 0
    for c in costs[correct].sort().values.tolist():
        if spend + c > budget:
            break
        spend += c
        flipped += 1
    return (int(correct.sum()) - flipped) / costs.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = ap.parse_args()
    torch.set_num_threads(2)
    x, ei, y, trm, vam, tem = probe.load_cora()
    d = torch.load(probe.CORA, weights_only=False)
    out = []
    for s in args.splits:
        tr, va, te = trm[s], vam[s], tem[s]
        test_idx = te.nonzero().view(-1)
        prior, _ = llm.zero_shot_prior(x, llm.class_embeddings(list(d.label_names)), y, va)
        logits, _ = ctv.tower_logits(x, y, tr, va, s, None)
        configs = {"locked": ctv.tune(logits, prior, ei, y, va), "EXPL a=2,T=0.1": dict(alpha=2.0, temp=0.1, beta=0.0),
                   "EXPL a=4,T=0.1": dict(alpha=4.0, temp=0.1, beta=0.0)}
        for name, cfg in configs.items():
            model = ctv.CTV(ctv.make_votes(logits, prior, cfg["temp"], cfg["beta"]), cfg["alpha"])
            costs = flip_costs(model, ei, y, test_idx)
            row = {"config": name, "split": s, **cfg, "clean": probe.accuracy(model, x, ei, y, te),
                   "median_flip_cost_correct": float(costs[costs > 0].median())}
            for r in args.rates:
                row[f"exact_greedy@{r}"] = greedy_accuracy(costs, int(r * ei.size(1) // 2))
            print(json.dumps(row), flush=True)
            out.append(row)
    with open(os.path.join(ctv.OUT_DIR, "exact_attack.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
