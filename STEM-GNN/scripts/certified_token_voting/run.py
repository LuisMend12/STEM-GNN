"""Certified Token Voting (CTV): experiments/certified-token-voting/protocol.md.

Token tower = STEM-lite run on an EMPTY edge set (tokens depend only on each
node's own text), so a structure attack cannot change any token. Prediction:
    score_c(i) = alpha * v_ic + sum_{j in N(i)} w_ij * v_jc,   v_j on the simplex
Each incident edge edit moves the margin by at most 1, which gives a
deterministic certificate: radius r_i = ceil(margin_i) - 1.

Usage (from repo root):
  python STEM-GNN/scripts/certified_token_voting/run.py --splits 0 --threads 2
"""
import argparse
import importlib.util
import itertools
import json
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, scatter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


llm = _load("llm_run", os.path.join(SCRIPTS, "llm_robustness", "run.py"))
probe = llm.probe
log = logging.getLogger("ctv")
OUT_DIR = os.path.join(SCRIPTS, "..", "..", "experiments", "certified-token-voting", "results")
ALPHAS, TEMPS, BETAS = (0.5, 1.0, 2.0, 4.0), (0.5, 1.0, 2.0), (0.0, 0.5)
MAX_K = 10


class CTV(nn.Module):
    """Linear in edge_weight: PRBCD's gradients are exact (no masking)."""

    def __init__(self, votes, alpha):
        super().__init__()
        self.register_buffer("votes", votes)
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None):
        w = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1))
        nb = scatter(w.unsqueeze(-1) * self.votes[edge_index[0]], edge_index[1], dim=0,
                     dim_size=self.votes.size(0), reduce="sum")
        return self.alpha * self.votes + nb


# ------------------------------------------------------------------ tower
def tower_logits(x, y, tr, va, split, pseudo):
    """STEM-lite on an empty graph: per-node tokens from own text only."""
    empty = torch.zeros((2, 0), dtype=torch.long)
    model = llm.train_stem(False, pseudo, 0.0, x, empty, y, tr, va, split, None)
    model.vq_mode = "hard"
    model.eval()
    with torch.no_grad():
        logits, _, _, codes = model.forward_full(x, empty)
    return logits, codes


def make_votes(logits, prior, temp, beta):
    v = F.softmax(logits / temp, -1)
    return (1 - beta) * v + beta * prior if beta > 0 else v


def tune(logits, prior, ei, y, va):
    best = None
    for a, t, b in itertools.product(ALPHAS, TEMPS, BETAS):
        m = CTV(make_votes(logits, prior, t, b), a)
        acc = probe.accuracy(m, None, ei, y, va)
        cert1 = certified_acc(m, ei, y, va, 1)
        key = (acc, cert1)
        if best is None or key > best[0]:
            best = (key, dict(alpha=a, temp=t, beta=b))
    return best[1]


# ------------------------------------------------------------ certificate
@torch.no_grad()
def radii(model, ei):
    s = model(None, ei)
    top2 = s.topk(2, -1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    r = torch.ceil(margin) - 1
    return top2.indices[:, 0], r.clamp_min(-1)


def certified_acc(model, ei, y, mask, k):
    pred, r = radii(model, ei)
    return ((pred == y) & (r >= k))[mask].float().mean().item()


def incident_edits(clean, pert, n):
    """Number of undirected edge changes incident to each node."""
    def pairs(e):
        e = e[:, e[0] < e[1]]
        return set(map(tuple, e.t().tolist()))
    diff = pairs(clean) ^ pairs(pert)
    cnt = torch.zeros(n, dtype=torch.long)
    for i, j in diff:
        cnt[i] += 1
        cnt[j] += 1
    return cnt


def certificate_violations(model, clean, pert, test_idx):
    pred, r = radii(model, clean)
    with torch.no_grad():
        pred_p = model(None, pert).argmax(-1)
    edits = incident_edits(clean, pert, pred.numel())
    covered = (r >= edits)[test_idx]
    flipped = (pred != pred_p)[test_idx]
    return int((covered & flipped).sum()), int(covered.sum())


def degree_buckets(model, ei, y, te):
    pred, r = radii(model, ei)
    deg = degree(ei[1], num_nodes=y.numel())
    out = {}
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 10_000)):
        m = te & (deg >= lo) & (deg < hi)
        if m.any():
            out[f"deg[{lo},{hi})"] = {"n": int(m.sum()), "acc": (pred == y)[m].float().mean().item(),
                                      "mean_radius_correct": r[m & (pred == y)].float().mean().item()}
    return out


# ------------------------------------------------------------------- main
def run_split(split, rates, use_pseudo, prior_name, fixed=None):
    x, ei, y, trm, vam, tem = probe.load_cora()
    tr, va, te = trm[split], vam[split], tem[split]
    test_idx = te.nonzero().view(-1)
    d = torch.load(probe.CORA, weights_only=False)
    prior, _ = llm.zero_shot_prior(x, llm.class_embeddings(list(d.label_names)), y, va)
    if prior_name != "minilm":
        prior = torch.load(os.path.join(SCRIPTS, "..", "..", "experiments", "llm-robustness",
                                        "results", f"qwen_prior_{prior_name}.pt"))
    pseudo = llm.pseudo_labels(prior, tr) if use_pseudo else None
    t0 = time.time()
    logits, _ = tower_logits(x, y, tr, va, split, pseudo)
    cfg = tune(logits, prior, ei, y, va) if fixed is None else dict(alpha=fixed[0], temp=fixed[1], beta=0.0)
    model = CTV(make_votes(logits, prior, cfg["temp"], cfg["beta"]), cfg["alpha"])
    name = f"CTV(prior={prior_name},pseudo={int(use_pseudo)})"
    if fixed is not None:
        name += f"[EXPL a={fixed[0]},T={fixed[1]}]"
    row = {"model": name, "split": split, **cfg,
           "tower_only_acc": (logits.argmax(-1)[te] == y[te]).float().mean().item(),
           "clean": probe.accuracy(model, x, ei, y, te),
           "certified": {k: certified_acc(model, ei, y, te, k) for k in range(MAX_K + 1)},
           "degree": degree_buckets(model, ei, y, te)}
    gen = torch.Generator().manual_seed(1000 + split)
    for r in rates:
        b = int(r * ei.size(1) // 2)
        row[f"random_add@{r}"] = probe.accuracy(model, x, probe.random_add(ei, x.size(0), b, gen), y, te)
        row[f"camouflage@{r}"] = probe.accuracy(model, x, probe.camouflage(x, ei, y, test_idx, b, gen), y, te)
        pert = probe.prbcd(model, x, ei, y, test_idx, b)
        row[f"prbcd@{r}"] = probe.accuracy(model, x, pert, y, te)
        row[f"cert_violations@{r}"], row[f"cert_covered@{r}"] = certificate_violations(model, ei, pert, test_idx)
    row["seconds"] = round(time.time() - t0, 1)
    log.info(json.dumps(row))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--prior", default="minilm", help="minilm or a Qwen tag, e.g. Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pseudo", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in args.splits:
        rows = [run_split(s, args.rates, bool(p), args.prior) for p in args.pseudo]
        rows += [run_split(s, args.rates, False, args.prior, fixed=f) for f in ((2.0, 0.1), (4.0, 0.1))]
        with open(os.path.join(OUT_DIR, f"split{s}_{args.prior}.json"), "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
