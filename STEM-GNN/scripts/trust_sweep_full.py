"""
Full trust gate sweep across all available datasets.
Run from STEM-GNN/ root:
  python STEM-GNN/scripts/trust_sweep_full.py
"""
import os, sys, random, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WebKB, WikiCS
from torch_geometric.utils import to_undirected, add_self_loops
from torch_scatter import scatter_mean

from dataset.process_datasets import get_finetune_graph
from utils.preprocess import pre_node

GATES = ["none", "A_cosine", "B_confidence", "C_learned", "D_llm"]


class TrustLayer(nn.Module):
    def __init__(self, in_d, out_d, gate, hid=64):
        super().__init__()
        self.gate = gate
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
        if gate == "C_learned":
            self.mlp = nn.Sequential(nn.Linear(in_d * 2, hid), nn.ReLU(),
                                     nn.Linear(hid, 1))

    def forward(self, x, ei, tf=None, pl=None):
        row, col = ei; N = x.size(0)
        agg = scatter_mean(x[col], row, dim=0, dim_size=N)
        so, no = self.Ws(x), self.Wn(agg)
        if self.gate == "none":
            return so + no
        elif self.gate == "A_cosine":
            b = 1 - torch.sigmoid(F.cosine_similarity(x, agg, dim=-1))
        elif self.gate == "B_confidence":
            if pl is None:
                b = torch.full((N,), 0.5, device=x.device)
            else:
                c = F.softmax(pl, -1).max(-1).values
                b = 1 - scatter_mean(c[col], row, dim=0, dim_size=N)
        elif self.gate == "C_learned":
            b = self.mlp(torch.cat([x, agg], -1)).squeeze(-1).sigmoid()
        elif self.gate == "D_llm":
            ta = scatter_mean(tf[col], row, dim=0, dim_size=N)
            b = 1 - torch.sigmoid(F.cosine_similarity(tf, ta, dim=-1))
        return b.unsqueeze(-1) * so + (1 - b.unsqueeze(-1)) * no


class TrustGNN(nn.Module):
    def __init__(self, in_d, hid, nc, gate):
        super().__init__()
        self.gate = gate
        self.l1  = TrustLayer(in_d, hid, gate)
        self.l2  = TrustLayer(hid, hid, gate)
        self.clf = nn.Linear(hid, nc)
        if gate == "B_confidence":
            self.pre = nn.Linear(in_d, nc)

    def forward(self, x, ei, tf=None):
        p = self.pre(x).detach() if self.gate == "B_confidence" else None
        h = F.relu(self.l1(x, ei, tf, p))
        p2 = self.clf(h).detach() if self.gate == "B_confidence" else None
        return self.clf(F.relu(self.l2(h, ei, tf, p2)))


def inject(ei, feat, labels, ratio):
    N = feat.size(0); n = int(ratio * N)
    if n == 0: return ei
    fn = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), N):
        if added >= n: break
        sim = fn[vi] @ fn.T
        sim[vi] = -1
        sim[labels == labels[vi]] = -1
        j = sim.argmax().item()
        if sim[j].item() >= 0.5:
            src.append(vi); dst.append(j); added += 1
    if not src: return ei
    ne = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return torch.cat([ei, ne], 1)


def train_eval(gate, x, ei, tf, labels, trm, vm, tem, nc, seed, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TrustGNN(x.size(1), 128, nc, gate)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        m.train()
        F.cross_entropy(m(x, ei, tf)[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad(): lg = m(x, ei, tf)
            pred = lg.argmax(-1)
            va = (pred[vm] == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt


def run_injection_sweep(name, x, ei_base, tf, labels, trm, vm, tem, nc,
                        ratios, seeds=3, epochs=200):
    print(f"\n=== {name.upper()} injection sweep ===")
    res = {g: [] for g in GATES}
    for r in ratios:
        ei = inject(ei_base, tf, labels, r)
        print(f"\n--- ratio={r:.0%}  edges={ei.size(1)} ---", flush=True)
        for g in GATES:
            t2 = tf if g == "D_llm" else None
            acc = float(np.mean([
                train_eval(g, x, ei, t2, labels, trm, vm, tem, nc, s, epochs)
                for s in range(seeds)
            ]))
            res[g].append(round(acc, 4))
            print(f"  {g:<16} {acc*100:.2f}%", flush=True)
    return {"ratios": ratios, "results": res}


def run_webkb_hetero(webkb_root, seeds=3, epochs=200):
    print("\n=== WebKB Heterophily (Texas / Wisconsin / Cornell) ===")
    out = {}
    for ds_name in ["Texas", "Wisconsin", "Cornell"]:
        try:
            ds   = WebKB(webkb_root, name=ds_name)
            data = ds[0]; N = data.num_nodes
            x    = F.normalize(data.x.float(), dim=-1)
            labels = data.y.long()
            nc   = int(labels.max().item()) + 1
            ei, _ = add_self_loops(to_undirected(data.edge_index, num_nodes=N), num_nodes=N)
            hom  = (labels[ei[0]] == labels[ei[1]]).float().mean().item()
            print(f"\n{ds_name}: {N} nodes | homophily={hom:.3f}", flush=True)

            res = {g: [] for g in GATES}
            n_splits = min(seeds, data.train_mask.size(1))
            for si in range(n_splits):
                trm = data.train_mask[:, si]
                vm  = data.val_mask[:, si]
                tem = data.test_mask[:, si]
                for g in GATES:
                    t2  = x if g == "D_llm" else None
                    acc = train_eval(g, x, ei, t2, labels, trm, vm, tem, nc, si, epochs)
                    res[g].append(round(float(acc), 4))
                    print(f"  {ds_name} split={si} {g:<16} {acc*100:.2f}%", flush=True)

            out[ds_name] = {
                "homophily": round(hom, 4),
                "results": {g: round(float(np.mean(v)), 4) for g, v in res.items()},
            }
        except Exception as e:
            print(f"  {ds_name} FAILED: {e}")
    return out


if __name__ == "__main__":
    DATA_DIR  = "STEM-GNN/dataset"
    WEBKB_DIR = "/tmp/webkb"
    SEEDS     = 3
    EPOCHS    = 200

    # Load existing cora results
    existing_path = "/home/lam23005/STEM-GNN/results/sweep_results.json"
    with open(existing_path) as f:
        all_results = json.load(f)

    # PubMed injection sweep
    dataset, spl, labels, nc, _ = get_finetune_graph(DATA_DIR, "pubmed")
    dataset = pre_node(dataset); data = dataset[0]; labels = labels.long()
    tf = data.node_text_feat; N = data.num_nodes
    sp = spl[0]
    trm = torch.zeros(N, dtype=torch.bool); trm[sp["train"]] = True
    vm  = torch.zeros(N, dtype=torch.bool); vm[sp["valid"]]  = True
    tem = torch.zeros(N, dtype=torch.bool); tem[sp["test"]]  = True
    ec  = to_undirected(data.edge_index, num_nodes=N)
    all_results["pubmed_sweep"] = run_injection_sweep(
        "pubmed", tf, ec, tf, labels, trm, vm, tem, nc,
        ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], seeds=SEEDS, epochs=EPOCHS)

    # WebKB heterophily (Texas + Wisconsin + Cornell)
    all_results["heterophily"] = run_webkb_hetero(WEBKB_DIR, seeds=SEEDS, epochs=EPOCHS)

    out_path = "/home/lam23005/STEM-GNN/results/sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== All done. Saved → {out_path} ===")
