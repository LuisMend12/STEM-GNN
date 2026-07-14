"""
Published baseline methods under confusing-edge injection.
Compares against our LLM gate approach.

Baselines:
  mlp    — 2-layer MLP on BOW, no graph (immune to injection by design)
  gcn    — 2-layer GCN (Kipf & Welling, ICLR 2017)
  gat    — 2-layer GAT (Velickovic et al., ICLR 2018)
  appnp  — Linear encoder + APPNP propagation (Klicpera et al., ICLR 2019)
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, degree
from torch_geometric.nn import GATConv, APPNP as APPNPProp
from torch_scatter import scatter_mean

# ── GCN conv (manual — avoids version issues) ─────────────────────────────────
class GCNConv(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei, n):
        ei2, _ = add_self_loops(ei, num_nodes=n)
        row, col = ei2
        deg  = degree(col, n, dtype=x.dtype)
        norm = (deg[row] * deg[col]).sqrt().clamp(min=1e-6)
        out  = self.lin(x)
        agg  = torch.zeros_like(out)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(out[col]), out[col] / norm.unsqueeze(-1))
        return agg

# ── Models ────────────────────────────────────────────────────────────────────
class MLPModel(nn.Module):
    def __init__(self, d, hid, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hid), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hid, nc)
        )
    def forward(self, x, ei):
        return self.net(x)

class GCNModel(nn.Module):
    def __init__(self, d, hid, nc):
        super().__init__()
        self.c1  = GCNConv(d, hid)
        self.c2  = GCNConv(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei):
        n = x.size(0)
        h = F.relu(F.dropout(self.c1(x, ei, n),  p=0.5, training=self.training))
        h = F.relu(F.dropout(self.c2(h, ei, n),  p=0.5, training=self.training))
        return self.clf(h)

class GATModel(nn.Module):
    def __init__(self, d, hid, nc, heads=8):
        super().__init__()
        self.c1  = GATConv(d,          hid,  heads=heads,     concat=True,  dropout=0.6)
        self.c2  = GATConv(hid*heads,  hid,  heads=1,         concat=False, dropout=0.6)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei):
        h = F.elu(self.c1(x, ei))
        h = F.elu(self.c2(h, ei))
        return self.clf(h)

class APPNPModel(nn.Module):
    def __init__(self, d, hid, nc, K=10, alpha=0.15):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(d, hid), nn.ReLU(),
                                   nn.Dropout(0.5), nn.Linear(hid, hid))
        self.prop = APPNPProp(K=K, alpha=alpha)
        self.clf  = nn.Linear(hid, nc)
    def forward(self, x, ei):
        h = F.dropout(self.enc(x), p=0.5, training=self.training)
        h = self.prop(h, ei)
        return self.clf(h)

# ── Train / eval ──────────────────────────────────────────────────────────────
def train_eval(model_cls, model_kwargs, bow_x, ei, labels,
               trm, vm, tem, seed, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m   = model_cls(**model_kwargs)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        m.train()
        F.cross_entropy(m(bow_x, ei)[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep+1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                logits = m(bow_x, ei)
            pred = logits.argmax(-1)
            va = (pred[vm]  == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt

# ── Injection ─────────────────────────────────────────────────────────────────
def encode_texts(texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        with torch.no_grad():
            embs.append(model.encode(texts[i:i+batch_size], convert_to_tensor=True,
                                     show_progress_bar=False).cpu())
    return torch.cat(embs, 0)

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
    ne = torch.tensor([src+dst, dst+src], dtype=torch.long)
    return torch.cat([ei, ne], 1)

# ── Config ────────────────────────────────────────────────────────────────────
BASELINES = ["mlp", "gcn", "gat", "appnp"]
SEEDS  = 3
EPOCHS = 200
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
CORA_PT = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(EXISTING) as f:
    all_results = json.load(f)

# ── Dataset loop ──────────────────────────────────────────────────────────────
for ds_name, bow_name, raw_pt, sweep_key in [
    ("Cora",   "Cora",   CORA_PT, "cora_sweep"),
    ("PubMed", "PubMed", PUB_PT,  "pubmed_sweep"),
]:
    print(f"\n{'='*60}")
    print(f"  {ds_name}  —  published baselines")
    print(f"{'='*60}", flush=True)

    pl_data = Planetoid(PL_ROOT, bow_name)[0]
    bow_x   = F.normalize(pl_data.x.float(), dim=-1)
    labels  = pl_data.y.long()
    nc      = int(labels.max().item()) + 1
    d       = bow_x.size(1)
    ei_base = to_undirected(pl_data.edge_index, num_nodes=pl_data.num_nodes)
    trm, vm, tem = pl_data.train_mask, pl_data.val_mask, pl_data.test_mask

    print(f"  Loading LLM embeddings for injection ...", flush=True)
    raw_data = torch.load(raw_pt)
    llm_e    = encode_texts(raw_data.raw_texts)

    model_configs = {
        "mlp":   (MLPModel,   dict(d=d,       hid=128, nc=nc)),
        "gcn":   (GCNModel,   dict(d=d,       hid=128, nc=nc)),
        "gat":   (GATModel,   dict(d=d,       hid=64,  nc=nc, heads=8)),
        "appnp": (APPNPModel, dict(d=d,       hid=128, nc=nc, K=10, alpha=0.15)),
    }

    for bname in BASELINES:
        print(f"\n  ── {bname} ──", flush=True)
        model_cls, model_kwargs = model_configs[bname]
        res_bl = []
        for r in RATIOS:
            ei = inject(ei_base, llm_e, labels, r)
            accs = [train_eval(model_cls, model_kwargs, bow_x, ei,
                               labels, trm, vm, tem, s, EPOCHS)
                    for s in range(SEEDS)]
            acc = float(np.mean(accs))
            res_bl.append(round(acc, 4))
            print(f"  {bname:<8} ratio={r:.0%}  {acc*100:.2f}%", flush=True)
        # Store with "bl_" prefix to distinguish from our methods
        all_results[sweep_key]["results"][f"bl_{bname}"] = res_bl

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== Saved → {OUT} ===")
