"""Run the protocol in experiments/stem-vq-robustness/protocol.md (H1-H4).

Usage (from repo root):
  python STEM-GNN/scripts/stem_vq_robustness/run.py --splits 0 --threads 3
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adaptive_robustness"))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "probe", os.path.join(HERE, "..", "adaptive_robustness", "run.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)  # load_cora, accuracy, prbcd, constants
from models import SAGE  # noqa: E402
from stem_lite import STEMLite, code_margin_loss, commitment_loss  # noqa: E402

log = logging.getLogger("stem")
OUT_DIR = os.path.join(HERE, "..", "..", "..", "experiments", "stem-vq-robustness", "results")
STAGE1_EPOCHS, STAGE3_EPOCHS, PATIENCE, LR, WD = 200, 300, 50, 0.01, 5e-4
STEM_VARIANTS = {
    "STEM-lite": dict(cons=0.0, margin=0.0),
    "STEM-lite+DropEdgeCons": dict(cons=1.0, margin=0.0),
    "STEM-lite+CodeMargin": dict(cons=0.0, margin=1.0),
    "STEM-lite+Both": dict(cons=1.0, margin=1.0),
}


# ---------------------------------------------------------------- training
def fit(model, loss_fn, x, ei, y, tr, va, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        val = probe.accuracy(model, x, ei, y, va)
        if val > best:
            best, wait, state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return best


def dropedge_cons(model, x, ei):
    d1, _ = dropout_edge(ei, p=0.3, force_undirected=True, training=True)
    d2, _ = dropout_edge(ei, p=0.3, force_undirected=True, training=True)
    return (F.softmax(model(x, d1), -1) - F.softmax(model(x, d2), -1)).pow(2).sum(-1).mean()


def train_stem(cfg, x, ei, y, tr, va, split):
    torch.manual_seed(split)
    model = STEMLite(x.size(1), int(y.max()) + 1)
    model.vq_mode = "none"
    fit(model, lambda: F.cross_entropy(model(x, ei)[tr], y[tr]), x, ei, y, tr, va, STAGE1_EPOCHS)
    model.init_codebook(x, ei, seed=split)
    model.vq_mode = "ste"

    def loss():
        logits, zh, sims, idx = model.forward_full(x, ei)
        total = F.cross_entropy(logits[tr], y[tr]) + 0.25 * commitment_loss(zh, sims, idx, model)
        if cfg["margin"] > 0:
            total = total + cfg["margin"] * code_margin_loss(sims)
        if cfg["cons"] > 0:
            total = total + cfg["cons"] * dropedge_cons(model, x, ei)
        return total

    val = fit(model, loss, x, ei, y, tr, va, STAGE3_EPOCHS)
    model.vq_mode = "hard"
    return model, val


def propsens(model, x, ei, coef, kind):
    """H4 variants of the repo's propagation_sensitivity_penalty."""
    dropped, _ = dropout_edge(ei, p=0.3, force_undirected=True, training=True)
    if kind == "preact":
        w = torch.ones(ei.size(1)), torch.ones(dropped.size(1))
        h, h_d = model.l1(x, ei, w[0]), model.l1(x, dropped, w[1])
    else:
        h, h_d = model.hidden(x, ei)[0], model.hidden(x, dropped)[0]
    change = (h - h_d).norm(dim=-1)
    if kind == "global":
        return coef * change.mean() / h.detach().norm(dim=-1).mean().clamp_min(1e-6)
    return coef * (change / h.detach().norm(dim=-1).clamp_min(1e-6)).mean()


SAGE_VARIANTS = {
    "SAGE": None,
    # (c=1,node) and (c=1,global) collapse to ~33% clean in the smoke test
    # (dead-ReLU collapse within ~6 epochs); see analysis.md. Not attacked.
    "SAGE+PropSens(c=0.1,node)": (0.1, "node"),
    "SAGE+PropSens(c=1,preact)": (1.0, "preact"),
}


def train_sage(spec, x, ei, y, tr, va, split):
    torch.manual_seed(split)
    model = SAGE(x.size(1), probe.HID, int(y.max()) + 1)

    def loss():
        total = F.cross_entropy(model(x, ei)[tr], y[tr])
        if spec is not None:
            total = total + propsens(model, x, ei, *spec)
        return total

    return model, fit(model, loss, x, ei, y, tr, va, probe.EPOCHS)


@torch.no_grad()
def zero_row_fraction(model, x, ei):
    model.eval()
    return (model.hidden(x, ei)[0].abs().sum(-1) == 0).float().mean().item()


# ------------------------------------------------------------- evaluation
def stem_attacks(model, x, ei, y, te, test_idx, budgets, transfer):
    row = {}
    for r, b in budgets.items():
        model.vq_mode = "hard"
        row[f"transfer@{r}"] = probe.accuracy(model, x, transfer[r], y, te)
        model.vq_mode = "ste"
        pert = probe.prbcd(model, x, ei, y, test_idx, b)
        model.vq_mode = "hard"
        row[f"prbcd_ste@{r}"] = probe.accuracy(model, x, pert, y, te)
        if r == 0.1:
            row["code_change@0.1"] = code_change(model, x, ei, pert, test_idx)
    return row


@torch.no_grad()
def code_change(model, x, ei, pert, test_idx):
    model.eval()
    _, _, _, a = model.forward_full(x, ei)
    _, _, _, b = model.forward_full(x, pert)
    return (a[test_idx] != b[test_idx]).float().mean().item()


def run_split(split, rates, include_hard):
    x, ei, y, trm, vam, tem = probe.load_cora()
    tr, va, te = trm[split], vam[split], tem[split]
    test_idx = te.nonzero().view(-1)
    budgets = {r: int(r * ei.size(1) // 2) for r in rates}
    rows, transfer = [], {}
    for name, spec in SAGE_VARIANTS.items():
        t0 = time.time()
        model, val = train_sage(spec, x, ei, y, tr, va, split)
        row = {"model": name, "split": split, "val": val, "clean": probe.accuracy(model, x, ei, y, te),
               "zero_rows": zero_row_fraction(model, x, ei)}
        for r, b in budgets.items():
            pert = probe.prbcd(model, x, ei, y, test_idx, b)
            row[f"prbcd@{r}"] = probe.accuracy(model, x, pert, y, te)
            if name == "SAGE":
                transfer[r] = pert
        rows.append(emit(row, t0))
    for name, cfg in STEM_VARIANTS.items():
        t0 = time.time()
        model, val = train_stem(cfg, x, ei, y, tr, va, split)
        row = {"model": name, "split": split, "val": val, "clean": probe.accuracy(model, x, ei, y, te)}
        row.update(stem_attacks(model, x, ei, y, te, test_idx, budgets, transfer))
        if include_hard and name == "STEM-lite":
            for r, b in budgets.items():
                model.vq_mode = "hard"
                row[f"prbcd_hard@{r}"] = probe.accuracy(model, x, probe.prbcd(model, x, ei, y, test_idx, b), y, te)
        rows.append(emit(row, t0))
    return rows


def emit(row, t0):
    row["seconds"] = round(time.time() - t0, 1)
    log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--no_hard", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in args.splits:
        rows = run_split(s, args.rates, not args.no_hard)
        with open(os.path.join(OUT_DIR, f"split{s}.json"), "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
