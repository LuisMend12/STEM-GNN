"""Adaptive-attack probe for candidate robustness mechanisms (Q1-Q3 of the brief).

Self-contained: it reads the OFA Cora file (frozen 384-d sentence embeddings,
10 provided splits) and needs only torch + torch_geometric (+ contrib PRBCD).

Threats (all evasion: train on the clean graph, attack at test time):
  random_add   random undirected edge insertion
  camouflage   PGD-Guard-style heuristic: link test nodes to the wrong-label node
               with the highest text cosine (defeats cosine gates by construction)
  prbcd        white-box adaptive PRBCD global attack on the test nodes,
               differentiating through each model's own edge weighting

Usage (from repo root):
  python STEM-GNN/scripts/adaptive_robustness/run.py --splits 0 1 2
"""
import argparse
import json
import logging
import os
import time
import warnings

import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce, dropout_edge, to_undirected

from models import GCN, MLP, SAGE, RUNGLite

warnings.filterwarnings("ignore", category=UserWarning)
from torch_geometric.contrib.nn import PRBCDAttack  # noqa: E402

log = logging.getLogger("probe")
HERE = os.path.dirname(os.path.abspath(__file__))
CORA = os.path.join(HERE, "..", "..", "dataset", "data", "single_graph", "Cora", "cora.pt")
HID, EPOCHS, LR, WD, PATIENCE = 64, 300, 0.01, 5e-4, 50


# --------------------------------------------------------------------- data
def load_cora():
    d = torch.load(CORA, weights_only=False)
    ei = coalesce(to_undirected(d.edge_index))
    ei = ei[:, ei[0] != ei[1]]
    return d.x.float(), ei, d.y.long(), d.train_masks, d.val_masks, d.test_masks


def edge_cos_quantile(x, ei, q):
    cos = F.cosine_similarity(x[ei[0]], x[ei[1]], dim=-1)
    return torch.quantile(cos, q).item()


# ------------------------------------------------------------------- models
def build(name, in_dim, n_cls, tau):
    table = {
        "MLP": lambda: MLP(in_dim, HID, n_cls),
        "GCN": lambda: GCN(in_dim, HID, n_cls),
        "SAGE": lambda: SAGE(in_dim, HID, n_cls),
        "SAGE+PropSens": lambda: SAGE(in_dim, HID, n_cls),
        "SAGE+DropEdgeCons": lambda: SAGE(in_dim, HID, n_cls),
        "SAGE+TextGate": lambda: SAGE(in_dim, HID, n_cls, mode="text_gate", tau=tau),
        "SAGE+TrustBudget": lambda: SAGE(in_dim, HID, n_cls, mode="trust_budget", tau=tau),
        "SAGE+TrustBudget+PropSens": lambda: SAGE(in_dim, HID, n_cls, mode="trust_budget", tau=tau),
        "APPNP-lite": lambda: RUNGLite(in_dim, HID, n_cls, penalty="l2", lam=0.3),
        "RUNG-lite": lambda: RUNGLite(in_dim, HID, n_cls, lam=0.3, delta=0.3, gamma=1.2),
        "RUNG-lite+Text": lambda: RUNGLite(in_dim, HID, n_cls, lam=0.1, delta=1.0, gamma=1.2,
                                            text=True, tau=tau),
    }
    return table[name]()


def regularizer(name, model, x, ei, coef):
    """Q2 regularizers. PropSens mirrors the repo's propagation_sensitivity_penalty
    (relative hidden-state change under 30% undirected edge drop, detached
    denominator). DropEdgeCons is a GRAND-style prediction-consistency baseline."""
    if "PropSens" in name:
        dropped, _ = dropout_edge(ei, p=0.3, force_undirected=True, training=True)
        h, _, _ = model.hidden(x, ei)
        h_d, _, _ = model.hidden(x, dropped)
        return coef * ((h - h_d).norm(dim=-1) / h.detach().norm(dim=-1).clamp_min(1e-6)).mean()
    if "DropEdgeCons" in name:
        d1, _ = dropout_edge(ei, p=0.3, force_undirected=True, training=True)
        d2, _ = dropout_edge(ei, p=0.3, force_undirected=True, training=True)
        p1, p2 = F.softmax(model(x, d1), -1), F.softmax(model(x, d2), -1)
        return coef * (p1 - p2).pow(2).sum(-1).mean()
    return torch.zeros(())


# ----------------------------------------------------------------- training
@torch.no_grad()
def accuracy(model, x, ei, y, mask, edge_weight=None):
    model.eval()
    return (model(x, ei, edge_weight).argmax(-1)[mask] == y[mask]).float().mean().item()


def train(name, model, x, ei, y, tr, va, coef):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, best_state, wait = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, ei)[tr], y[tr]) + regularizer(name, model, x, ei, coef)
        loss.backward()
        opt.step()
        val = accuracy(model, x, ei, y, va)
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best


# ------------------------------------------------------------------ attacks
def random_add(ei, n, budget, gen):
    existing = set(map(tuple, ei.t().tolist()))
    new = []
    while len(new) < budget:
        i, j = torch.randint(0, n, (2,), generator=gen).tolist()
        if i != j and (i, j) not in existing:
            existing.update({(i, j), (j, i)})
            new.append((i, j))
    add = torch.tensor(new).t()
    return coalesce(torch.cat([ei, to_undirected(add)], 1))


def camouflage(x, ei, y, test_idx, budget, gen):
    """Each of `budget` random test nodes gets one edge to its highest-cosine
    node with a different label (attacker knows labels: worst case for gates)."""
    xn = F.normalize(x, dim=-1)
    existing = set(map(tuple, ei.t().tolist()))
    order = test_idx[torch.randperm(test_idx.numel(), generator=gen)]
    sim = xn[order] @ xn.t()
    sim[y[order].unsqueeze(1) == y.unsqueeze(0)] = -2.0
    ranked = sim.argsort(-1, descending=True)
    new, rank = [], 0
    # Round-robin over test nodes, skipping existing/duplicate pairs, until the
    # budget of unique undirected edges is spent (Codex review finding 8).
    while len(new) < budget and rank < ranked.size(1):
        for t, row in zip(order.tolist(), ranked[:, rank].tolist()):
            if len(new) >= budget:
                break
            if t != row and (t, row) not in existing and y[t] != y[row]:
                existing.update({(t, row), (row, t)})
                new.append((t, row))
        rank += 1
    add = torch.tensor(new).t()
    return coalesce(torch.cat([ei, to_undirected(add)], 1))


def prbcd(model, x, ei, y, test_idx, budget):
    model.eval()
    attack = PRBCDAttack(model, block_size=100_000, epochs=100, epochs_resampling=60, log=False)
    pert_ei, _ = attack.attack(x, ei, y, budget=budget, idx_attack=test_idx)
    return pert_ei


# --------------------------------------------------------------------- main
def run_split(split, names, x, ei, y, masks, rates, coef, tau):
    tr, va, te = (m[split] for m in masks)
    test_idx = te.nonzero().view(-1)
    n_und = ei.size(1) // 2
    rows = []
    for name in names:
        torch.manual_seed(split)
        gen = torch.Generator().manual_seed(1000 + split)
        model = build(name, x.size(1), int(y.max()) + 1, tau)
        t0 = time.time()
        val = train(name, model, x, ei, y, tr, va, coef)
        row = {"model": name, "split": split, "val": val, "clean": accuracy(model, x, ei, y, te)}
        for r in rates:
            b = int(r * n_und)
            row[f"random_add@{r}"] = accuracy(model, x, random_add(ei, x.size(0), b, gen), y, te)
            row[f"camouflage@{r}"] = accuracy(model, x, camouflage(x, ei, y, test_idx, b, gen), y, te)
            if name != "MLP":
                row[f"prbcd@{r}"] = accuracy(model, x, prbcd(model, x, ei, y, test_idx, b), y, te)
            else:
                row[f"prbcd@{r}"] = row["clean"]
        row["seconds"] = round(time.time() - t0, 1)
        log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--coef", type=float, default=1.0, help="Q2 regularizer weight")
    ap.add_argument("--tau_q", type=float, default=0.1,
                    help="text-trust cutoff = this quantile of clean-edge cosines")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)

    x, ei, y, *masks = load_cora()
    tau = edge_cos_quantile(x, ei, args.tau_q)
    log.info(f"Cora: n={x.size(0)} undirected_edges={ei.size(1)//2} tau={tau:.3f}")
    names = args.models or ["MLP", "GCN", "SAGE", "SAGE+PropSens", "SAGE+DropEdgeCons",
                            "SAGE+TextGate", "SAGE+TrustBudget", "SAGE+TrustBudget+PropSens",
                            "APPNP-lite", "RUNG-lite", "RUNG-lite+Text"]
    rows = []
    for s in args.splits:
        rows += run_split(s, names, x, ei, y, masks, args.rates, args.coef, tau)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "tau": tau, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
