"""Message-collapse experiment on STEM-GNN's real Encoder
(experiments/message-collapse/protocol.md).

Compares baseline, TAEF (text-anchored per-edge distance floor; new), and
GEF (graph-level energy floor; EGNN-style nearest prior art), for the
non-MoE and MoE encoders, across depths.

Usage (from repo root):
  python STEM-GNN/scripts/message_collapse/run.py --splits 0 --threads 2
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, to_undirected

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, PKG)
from model.encoder import Encoder  # noqa: E402  (the real STEM-GNN encoder)

log = logging.getLogger("collapse")
CORA = os.path.join(PKG, "dataset", "data", "single_graph", "Cora", "cora.pt")
OUT = os.path.join(PKG, "..", "experiments", "message-collapse", "results")
HID, LR, WD, EPOCHS, PATIENCE, DROPOUT = 64, 0.01, 5e-4, 300, 50, 0.15
RHO, LAM = 0.5, 1.0
DEPTHS, METHODS = (2, 4, 8, 16), ("baseline", "TAEF", "GEF", "TAEF-rho0.25[EXPL]")


def load():
    d = torch.load(CORA, weights_only=False)
    ei = coalesce(to_undirected(d.edge_index))
    ei = ei[:, ei[0] != ei[1]]
    return d.x.float(), ei, d.y.long(), d.train_masks, d.val_masks, d.test_masks


def edge_dist(h, ei):
    return 1 - F.cosine_similarity(h[ei[0]], h[ei[1]], dim=-1)


class Model(nn.Module):
    """Real STEM Encoder + linear head; records every layer's hidden state."""

    def __init__(self, in_dim, n_cls, depth, moe):
        super().__init__()
        self.enc = Encoder(in_dim, HID, nn.ReLU, depth, backbone="sage", normalize="batch",
                           dropout=DROPOUT, moe=moe, num_experts=3, tau=1.0, moe_layers="all")
        self.head = nn.Linear(HID, n_cls)
        self.states = []
        self.enc.activation.register_forward_hook(lambda m, i, o: self.states.append(o))

    def forward(self, x, ei):
        self.states = []
        z = self.enc(x, ei)
        self.states.append(z)  # final layer (no activation applied)
        return self.head(z)


def floor_loss(method, states, x, ei):
    if method == "baseline":
        return torch.zeros(())
    dx = edge_dist(x, ei)
    if method.startswith("TAEF"):
        rho = 0.25 if "rho0.25" in method else RHO
        return LAM * torch.stack([F.relu(rho * dx - edge_dist(h, ei)).mean() for h in states]).mean()
    return LAM * torch.stack([F.relu(RHO * dx.mean() - edge_dist(h, ei).mean()) for h in states]).mean()


@torch.no_grad()
def accuracy(model, x, ei, y, mask):
    model.eval()
    return (model(x, ei).argmax(-1)[mask] == y[mask]).float().mean().item()


def train(model, method, x, ei, y, tr, va):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, state, wait = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x, ei)
        loss = F.cross_entropy(logits[tr], y[tr]) + floor_loss(method, model.states, x, ei)
        loss.backward()
        opt.step()
        val = accuracy(model, x, ei, y, va)
        if val > best:
            best, wait, state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return best


@torch.no_grad()
def collapse_metrics(model, x, ei, gen):
    model.eval()
    model.enc.enable_router_cache(True)
    model(x, ei)
    routers = model.enc.get_router_cache(reset=True) or []
    model.enc.enable_router_cache(False)
    h = model.states[-1]
    ratio = (edge_dist(h, ei).mean() / edge_dist(x, ei).mean()).item()
    i, j = torch.randint(0, h.size(0), (2, 20_000), generator=gen)
    mad = (1 - F.cosine_similarity(h[i], h[j], dim=-1)).mean().item()
    s = torch.linalg.svdvals(h - h.mean(0))
    p = s / s.sum()
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    per_layer_mad = [(1 - F.cosine_similarity(st[i], st[j], dim=-1)).mean().item() for st in model.states]
    out = {"energy_ratio": ratio, "mad": mad, "eff_rank": erank, "mad_per_layer": per_layer_mad}
    if routers:
        ents = []
        for w in routers:
            u = w.mean(0)
            ents.append((-(u * torch.log(u.clamp_min(1e-12))).sum() / torch.log(torch.tensor(float(u.numel())))).item())
        out["expert_entropy"] = sum(ents) / len(ents)
        out["expert_usage_per_layer"] = [w.mean(0).tolist() for w in routers]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    x, ei, y, trm, vam, tem = load()
    for s in args.splits:
        rows = []
        for seed in args.seeds:
          for moe in (False, True):
            for depth in args.depths:
                for method in METHODS:
                    torch.manual_seed(1000 * seed + s)
                    t0 = time.time()
                    model = Model(x.size(1), int(y.max()) + 1, depth, moe)
                    val = train(model, method, x, ei, y, trm[s], vam[s])
                    row = {"encoder": "MoE" if moe else "SAGE", "depth": depth, "method": method, "split": s,
                           "seed": seed, "threads": args.threads,
                           "val": val, "test": accuracy(model, x, ei, y, tem[s]),
                           **collapse_metrics(model, x, ei, torch.Generator().manual_seed(s))}
                    row["seconds"] = round(time.time() - t0, 1)
                    log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                         if k not in ("mad_per_layer", "expert_usage_per_layer")}))
                    rows.append(row)
                    with open(os.path.join(OUT, f"split{s}{args.tag}.json"), "w") as f:
                        json.dump({"args": vars(args), "rho": RHO, "lam": LAM, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
