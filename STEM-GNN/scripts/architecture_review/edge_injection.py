"""E1: STEM-GNN raw edge-text injection vs alternatives (experiments/architecture-review/protocol.md).

Uses the real Encoder/MySAGEConv; variants S2/S3 only wrap each conv's
forward to change what is passed as edge_attr, so the message function
ReLU(x_j + xe) stays the repo's own.

Usage (from repo root):
  python STEM-GNN/scripts/architecture_review/edge_injection.py --splits 0 --threads 4
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

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, PKG)
from model.encoder import Encoder  # noqa: E402

log = logging.getLogger("edge")
CORA = os.path.join(PKG, "dataset", "data", "single_graph", "Cora", "cora.pt")
OUT = os.path.join(PKG, "..", "experiments", "architecture-review", "results")
EDGE_TEXT = "feature edge. connected papers are cited together by other papers."
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
HID, LR, WD, EPOCHS, PATIENCE, DROPOUT = 384, 0.01, 5e-4, 300, 50, 0.15
VARIANTS = ("S0", "S1", "S2", "S3")


def load():
    from torch_geometric.utils import coalesce, to_undirected
    d = torch.load(CORA, weights_only=False)
    ei = coalesce(to_undirected(d.edge_index))
    return d.x.float(), ei[:, ei[0] != ei[1]], d.y.long(), d.train_masks, d.val_masks, d.test_masks


def edge_text_embedding():
    cache = os.path.join(OUT, "cora_edge_text_minilm.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    enc = AutoModel.from_pretrained(ENCODER, local_files_only=True).eval()
    b = tok([EDGE_TEXT], return_tensors="pt")
    with torch.no_grad():
        h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    c = F.normalize((h * m).sum(1) / m.sum(1), dim=-1)[0]
    torch.save(c, cache)
    return c


class Model(nn.Module):
    def __init__(self, in_dim, n_cls, depth, variant, edge_dim):
        super().__init__()
        self.enc = Encoder(in_dim, HID, nn.ReLU, depth, backbone="sage", normalize="batch", dropout=DROPOUT)
        self.head = nn.Linear(HID, n_cls)
        self.variant = variant
        if variant == "S3":
            self.edge_proj = nn.ModuleList(nn.Linear(edge_dim, layer.in_channels) for layer in self.enc.layers)
        for idx, layer in enumerate(self.enc.layers):
            layer.forward = self._wrap(layer, layer.forward, idx)

    def _wrap(self, layer, orig_forward, idx):
        def forward(x, edge_index, edge_attr=None, size=None):
            if self.variant == "S2" and idx > 0:
                edge_attr = None
            elif self.variant == "S3" and edge_attr is not None:
                edge_attr = self.edge_proj[idx](edge_attr)
            return orig_forward(x, edge_index, edge_attr, size)
        return forward

    def forward(self, x, ei, edge_attr):
        return self.head(self.enc(x, ei, None if self.variant == "S0" else edge_attr))


@torch.no_grad()
def accuracy(model, x, ei, ea, y, mask):
    model.eval()
    return (model(x, ei, ea).argmax(-1)[mask] == y[mask]).float().mean().item()


def train(model, x, ei, ea, y, tr, va):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, state, wait = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei, ea)[tr], y[tr]).backward()
        opt.step()
        val = accuracy(model, x, ei, ea, y, va)
        if val > best:
            best, wait, state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(state)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(OUT, exist_ok=True)
    x, ei, y, trm, vam, tem = load()
    c = edge_text_embedding()
    ea = c.unsqueeze(0).expand(ei.size(1), -1).contiguous()
    for s in args.splits:
        rows = []
        for seed in args.seeds:
            for depth in args.depths:
                for variant in VARIANTS:
                    torch.manual_seed(1000 * seed + s)
                    t0 = time.time()
                    model = Model(x.size(1), int(y.max()) + 1, depth, variant, c.numel())
                    val = train(model, x, ei, ea, y, trm[s], vam[s])
                    row = {"variant": variant, "depth": depth, "split": s, "seed": seed, "val": val,
                           "test": accuracy(model, x, ei, ea, y, tem[s]), "seconds": round(time.time() - t0, 1)}
                    log.info(json.dumps(row))
                    rows.append(row)
                    with open(os.path.join(OUT, f"edge_injection_split{s}.json"), "w") as f:
                        json.dump({"args": vars(args), "edge_text": EDGE_TEXT, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
