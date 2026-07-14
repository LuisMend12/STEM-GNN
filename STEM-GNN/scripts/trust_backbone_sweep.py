"""
Backbone comparison: GraphSAGE vs GCN, with LLM-generated gate signal.

Architecture:
  Backbone:  GCN (normalised A) or GraphSAGE (mean) on LLM embeddings
  Gate:      β_v from LLM cosine similarity — pure semantic oracle, no trained gate
             β_v = 1 - σ(cos(e_v, mean{e_u}))  [high cos → trust nbrs → low β]

Run from STEM-GNN/ root:
  python STEM-GNN/scripts/trust_backbone_sweep.py
"""
import os, sys, random, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_undirected, add_self_loops, degree
from torch_scatter import scatter_mean

from dataset.process_datasets import get_finetune_graph
from utils.preprocess import pre_node

# ── Backbone layers ───────────────────────────────────────────────────────────

class SAGELayer(nn.Module):
    """Standard GraphSAGE mean aggregation."""
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)

    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        agg = scatter_mean(x[col], row, dim=0, dim_size=N)
        return self.Ws(x) + self.Wn(agg)


class GCNLayer(nn.Module):
    """GCN with symmetric normalisation: h = D^{-1/2} A D^{-1/2} X W."""
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W = nn.Linear(in_d, out_d, bias=False)

    def forward(self, x, ei, norm):
        # norm[i] = 1 / sqrt(deg_i)  (precomputed, includes self-loops)
        row, col = ei; N = x.size(0)
        xw = self.W(x)
        # Normalise source then aggregate then normalise target
        agg = scatter_mean(norm[col].unsqueeze(-1) * xw[col],
                           row, dim=0, dim_size=N)
        return norm.unsqueeze(-1) * agg


# ── Gated models ──────────────────────────────────────────────────────────────

class TrustGNN(nn.Module):
    """2-layer GNN with optional LLM-generated gate signal."""

    def __init__(self, in_d, hid, nc, backbone="sage", gate="llm_cosine"):
        super().__init__()
        self.backbone = backbone
        self.gate     = gate
        if backbone == "sage":
            self.l1 = SAGELayer(in_d, hid)
            self.l2 = SAGELayer(hid, hid)
        else:  # gcn
            self.l1 = GCNLayer(in_d, hid)
            self.l2 = GCNLayer(hid, hid)
        self.clf = nn.Linear(hid, nc)

        # Confidence gate needs a linear pre-classifier
        if gate == "confidence":
            self.pre = nn.Linear(in_d, nc)
        # Learned MLP gate
        if gate == "learned":
            self.mlp = nn.Sequential(
                nn.Linear(in_d * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def _gate(self, x, ei, llm_feats, norm=None):
        """Compute per-node β_v from the chosen signal."""
        row, col = ei; N = x.size(0)
        if self.gate == "none":
            return None

        elif self.gate == "llm_cosine":
            # Use frozen LLM embeddings — pure semantic oracle
            ef = F.normalize(llm_feats, dim=-1)
            agg_llm = scatter_mean(ef[col], row, dim=0, dim_size=N)
            b = 1 - torch.sigmoid(F.cosine_similarity(ef, agg_llm, dim=-1))

        elif self.gate == "confidence":
            # Neighbour prediction confidence
            p = self.pre(llm_feats).detach()
            c = F.softmax(p, -1).max(-1).values
            b = 1 - scatter_mean(c[col], row, dim=0, dim_size=N)

        elif self.gate == "learned":
            ef = llm_feats
            agg = scatter_mean(ef[col], row, dim=0, dim_size=N)
            b = self.mlp(torch.cat([ef, agg], -1)).squeeze(-1).sigmoid()

        return b

    def _layer(self, layer, x, ei, beta, norm):
        row, col = ei; N = x.size(0)
        if self.backbone == "sage":
            h_full = layer(x, ei)
        else:
            h_full = layer(x, ei, norm)

        if beta is None:
            return h_full

        # Split self vs neigh contribution and gate
        if self.backbone == "sage":
            h_self  = layer.Ws(x)
            h_neigh = layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
        else:
            # For GCN, approximate: self-contribution vs aggregation
            h_self  = layer.W(x)
            h_neigh = h_full  # already the normalised aggregation

        return beta.unsqueeze(-1) * h_self + (1 - beta.unsqueeze(-1)) * h_neigh

    def forward(self, x, ei, llm_feats, norm=None):
        beta = self._gate(x, ei, llm_feats, norm)
        h = F.relu(self._layer(self.l1, x, ei, beta, norm))
        beta2 = self._gate(h, ei, llm_feats, norm)  # recompute from LLM (frozen)
        h = F.relu(self._layer(self.l2, h, ei, beta2, norm))
        return self.clf(h)


# ── Training ──────────────────────────────────────────────────────────────────

def train_eval(backbone, gate, x, ei, llm_feats, labels,
               trm, vm, tem, nc, seed, norm=None, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TrustGNN(x.size(1), 128, nc, backbone=backbone, gate=gate)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        m.train()
        F.cross_entropy(m(x, ei, llm_feats, norm)[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                lg = m(x, ei, llm_feats, norm)
            pred = lg.argmax(-1)
            va = (pred[vm] == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt


def gcn_norm(ei, N):
    """Compute D^{-1/2} for symmetric normalisation (with self-loops)."""
    ei_sl, _ = add_self_loops(ei, num_nodes=N)
    deg = degree(ei_sl[0], num_nodes=N).float()
    return deg.pow(-0.5).clamp(max=10)


def inject(ei, feat, labels, ratio):
    N = feat.size(0); n = int(ratio * N)
    if n == 0: return ei
    fn = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), N):
        if added >= n: break
        sim = fn[vi] @ fn.T; sim[vi] = -1; sim[labels == labels[vi]] = -1
        j = sim.argmax().item()
        if sim[j].item() >= 0.5: src.append(vi); dst.append(j); added += 1
    if not src: return ei
    ne = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return torch.cat([ei, ne], 1)


# ── Experiments ───────────────────────────────────────────────────────────────

BACKBONES = ["sage", "gcn"]
GATES     = ["none", "llm_cosine", "confidence", "learned"]
LABEL_B   = {"sage": "GraphSAGE", "gcn": "GCN"}
LABEL_G   = {"none": "No gate", "llm_cosine": "LLM cosine gate",
             "confidence": "Confidence gate", "learned": "Learned gate"}

def run_injection(name, x, ei_base, llm_feats, labels,
                  trm, vm, tem, nc, ratios, seeds=3, epochs=200):
    print(f"\n=== {name.upper()} injection sweep ===")
    results = {}
    N = x.size(0)
    for bb in BACKBONES:
        norm = gcn_norm(ei_base, N) if bb == "gcn" else None
        for gate in GATES:
            key = f"{bb}+{gate}"
            results[key] = []
            for r in ratios:
                ei = inject(ei_base, llm_feats, labels, r)
                n_ei = gcn_norm(ei, N) if bb == "gcn" else None
                acc = float(np.mean([
                    train_eval(bb, gate, x, ei, llm_feats, labels,
                               trm, vm, tem, nc, s, n_ei, epochs)
                    for s in range(seeds)
                ]))
                results[key].append(round(acc, 4))
                print(f"  {bb:4s}+{gate:<12} ratio={r:.0%}  {acc*100:.2f}%", flush=True)
    return {"ratios": ratios, "results": results}


def run_hetero(webkb_root, seeds=3, epochs=200):
    print("\n=== WebKB Heterophily (Texas / Wisconsin / Cornell) ===")
    out = {}
    for ds_name in ["Texas", "Wisconsin", "Cornell"]:
        ds   = WebKB(webkb_root, name=ds_name)
        data = ds[0]; N = data.num_nodes
        x    = F.normalize(data.x.float(), dim=-1)
        labels = data.y.long()
        nc   = int(labels.max().item()) + 1
        ei, _ = add_self_loops(to_undirected(data.edge_index, num_nodes=N), num_nodes=N)
        hom  = (labels[ei[0]] == labels[ei[1]]).float().mean().item()
        print(f"\n{ds_name}: {N} nodes | homophily={hom:.3f}")
        norm_gcn = gcn_norm(ei, N)

        results = {}
        n_splits = min(seeds, data.train_mask.size(1))
        for bb in BACKBONES:
            norm = norm_gcn if bb == "gcn" else None
            for gate in GATES:
                key = f"{bb}+{gate}"
                accs = []
                for si in range(n_splits):
                    trm = data.train_mask[:, si]
                    vm  = data.val_mask[:, si]
                    tem = data.test_mask[:, si]
                    # Use raw features as LLM proxy for WebKB (no text encoder available)
                    acc = train_eval(bb, gate, x, ei, x, labels,
                                     trm, vm, tem, nc, si, norm, epochs)
                    accs.append(round(float(acc), 4))
                results[key] = round(float(np.mean(accs)), 4)
                print(f"  {ds_name} {bb:4s}+{gate:<12} {results[key]*100:.2f}%", flush=True)

        out[ds_name] = {"homophily": round(hom, 4), "results": results}
    return out


if __name__ == "__main__":
    DATA_DIR  = "STEM-GNN/dataset"
    WEBKB_DIR = "/tmp/webkb"
    SEEDS = 3; EPOCHS = 200

    all_results = {}

    # Cora injection: backbone comparison
    print("Loading Cora...")
    dataset, spl, labels, nc, _ = get_finetune_graph(DATA_DIR, "cora")
    dataset = pre_node(dataset); data = dataset[0]; labels = labels.long()
    tf = data.node_text_feat; N = data.num_nodes; sp = spl[0]
    trm = torch.zeros(N, dtype=torch.bool); trm[sp["train"]] = True
    vm  = torch.zeros(N, dtype=torch.bool); vm[sp["valid"]]  = True
    tem = torch.zeros(N, dtype=torch.bool); tem[sp["test"]]  = True
    ec  = to_undirected(data.edge_index, num_nodes=N)

    all_results["cora_backbone"] = run_injection(
        "cora", tf, ec, tf, labels, trm, vm, tem, nc,
        ratios=[0.0, 0.1, 0.2, 0.3, 0.5], seeds=SEEDS, epochs=EPOCHS)

    # WebKB: backbone comparison on heterophily
    all_results["webkb_backbone"] = run_hetero(WEBKB_DIR, seeds=SEEDS, epochs=EPOCHS)

    out_path = "/home/lam23005/STEM-GNN/results/backbone_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Done. Saved → {out_path} ===")
