"""
All-LLM gate signals sweep — decoupled backbone + gate.

Backbone:  GNN (GraphSAGE) on ORIGINAL node features (BOW)
           Cora  → 1433-dim bag-of-words  (Planetoid)
           PubMed → 500-dim bag-of-words  (Planetoid)

Gate signal: all gates use FROZEN LLM embeddings computed from raw node text
             (multi-qa-distilbert-cos-v1, 768-dim)

Gate variants (all LLM-based):
  A  cosine      β_v = 1 − σ(cos(e_v, mean{e_u}))
  B  entropy     β_v = H_norm(softmax attention over neighbour LLM embeddings)
  C  mlp         β_v = σ(MLP(e_v ‖ mean{e_u}))   [trainable]
  D  variance    β_v = σ(10 · mean ||e_u − μ||²)
  E  max_cosine  β_v = 1 − max_u cos(e_v, e_u)
  F  nbr_agree   β_v = 1 − mean_{u1,u2} cos(e_u1, e_u2)
  G  bilinear    β_v = σ(e_v^T W mean{e_u})       [trainable, W 768→32]
  H  topk        β_v = 1 − mean of top-3 cos(e_v, e_u)
  I  cos_std     β_v = std of cos(e_v, e_u) over N(v)

Run from STEM-GNN/ root:
  python STEM-GNN/scripts/trust_gate/llm_gates_sweep.py
"""
import os, sys, random, json
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "STEM-GNN"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, add_self_loops
from torch_scatter import scatter_mean, scatter

# ── LLM encoder (frozen) ──────────────────────────────────────────────────────

def encode_texts(texts, batch_size=256):
    """Encode raw texts with DistilBERT into frozen 768-dim embeddings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            embs = model.encode(batch, convert_to_tensor=True,
                                show_progress_bar=False)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)   # [N, 768]

# ── Gate functions (all from LLM embeddings) ─────────────────────────────────

def gate_cosine(ei, e):
    """β_v = 1 − σ(cos(e_v, mean{e_u}))"""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
    return 1 - torch.sigmoid(F.cosine_similarity(ef, mu, dim=-1))


def gate_entropy(ei, e):
    """β_v = H(softmax attn) / log(deg)  — neighbourhood confusion."""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    scores = (ef[row] * ef[col]).sum(-1)
    s_max  = scatter(scores, row, dim=0, dim_size=N, reduce="max")
    exp_s  = torch.exp(scores - s_max[row])
    exp_sm = scatter(exp_s, row, dim=0, dim_size=N, reduce="sum").clamp(1e-8)
    a = exp_s / exp_sm[row]
    H = -scatter(a * torch.log(a.clamp(1e-8)), row, dim=0, dim_size=N, reduce="sum")
    deg = scatter(torch.ones(col.size(0), device=col.device),
                  row, dim=0, dim_size=N, reduce="sum").clamp(min=2)
    return (H / torch.log(deg)).clamp(0, 1)


def gate_variance(ei, e):
    """β_v = σ(10 · mean ||e_u − μ||²)  — neighbourhood heterogeneity."""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
    diff_sq = ((ef[col] - mu[row]) ** 2).sum(-1)
    var = scatter_mean(diff_sq, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)


def gate_max_cosine(ei, e):
    """β_v = 1 − max_u cos(e_v, e_u)  — single closest neighbour."""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos_per_edge = (ef[row] * ef[col]).sum(-1)
    max_cos = scatter(cos_per_edge, row, dim=0, dim_size=N, reduce="max")
    return 1 - torch.sigmoid(max_cos)


def gate_nbr_agree(ei, e):
    """β_v = 1 − mean cos(e_u1, e_u2) — do neighbours agree with each other?"""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    # For each node, compute mean pairwise cosine among its neighbours
    # Approximation: use ||mean_neigh||² ≈ mean of pairwise cosines (when normalised)
    mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
    # ||μ||² = mean pairwise cosine when all ||e_u||=1
    agreement = (mu * mu).sum(-1)   # in [0,1], 1 = all neighbours identical
    return 1 - agreement


def gate_topk(ei, e, k=3):
    """β_v = 1 − mean of top-k cos(e_v, e_u) — filter outlier neighbours (vectorised)."""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos_per_edge = (ef[row] * ef[col]).sum(-1)
    # Sort edges by (node asc, cosine desc) — fully vectorised, no Python loop
    sort_key  = row.float() * 1e9 - cos_per_edge.detach()
    sort_idx  = torch.argsort(sort_key)
    row_s     = row[sort_idx]
    cos_s     = cos_per_edge[sort_idx]
    positions = torch.arange(row_s.size(0), device=e.device)
    first_pos = scatter(positions.float(), row_s, dim=0, dim_size=N, reduce="min").long()
    rank      = positions - first_pos[row_s]
    mask      = rank < k
    topk_mean = scatter_mean(cos_s[mask], row_s[mask], dim=0, dim_size=N)
    return 1 - torch.sigmoid(topk_mean)


def gate_cos_std(ei, e):
    """β_v = std of cos(e_v, e_u) over N(v) — spread = neighbourhood uncertainty."""
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos_per_edge = (ef[row] * ef[col]).sum(-1)
    mu_cos = scatter_mean(cos_per_edge, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos_per_edge - mu_cos[row]) ** 2, row, dim=0, dim_size=N)
    return torch.sigmoid(var_cos.sqrt() * 10)


# ── Model ─────────────────────────────────────────────────────────────────────

class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)

    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))


class LLMGatedSAGE(nn.Module):
    """
    GraphSAGE on BOW features, gated by frozen LLM embeddings.
    Backbone and gate are fully decoupled.
    """
    def __init__(self, bow_dim, llm_dim, hid, nc, gate="cosine"):
        super().__init__()
        self.gate_type = gate
        self.l1  = SAGELayer(bow_dim, hid)
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
        if gate == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(llm_dim * 2, 128), nn.ReLU(),
                nn.Linear(128, 1))
        if gate == "bilinear":
            self.U = nn.Linear(llm_dim, 32, bias=False)
            self.V = nn.Linear(llm_dim, 32, bias=False)

    def _beta(self, ei, llm_e):
        if self.gate_type == "none":        return None
        if self.gate_type == "cosine":      return gate_cosine(ei, llm_e)
        if self.gate_type == "entropy":     return gate_entropy(ei, llm_e)
        if self.gate_type == "variance":    return gate_variance(ei, llm_e)
        if self.gate_type == "max_cosine":  return gate_max_cosine(ei, llm_e)
        if self.gate_type == "nbr_agree":   return gate_nbr_agree(ei, llm_e)
        if self.gate_type == "topk":        return gate_topk(ei, llm_e)
        if self.gate_type == "cos_std":     return gate_cos_std(ei, llm_e)
        if self.gate_type == "mlp":
            row, col = ei; N = llm_e.size(0)
            mu = scatter_mean(llm_e[col], row, dim=0, dim_size=N)
            return self.mlp(torch.cat([llm_e, mu], -1)).squeeze(-1).sigmoid()
        if self.gate_type == "bilinear":
            row, col = ei; N = llm_e.size(0)
            mu = scatter_mean(llm_e[col], row, dim=0, dim_size=N)
            return torch.sigmoid((self.U(llm_e) * self.V(mu)).sum(-1))

    def _apply_gate(self, layer, x, ei, beta):
        row, col = ei; N = x.size(0)
        h_self  = layer.Ws(x)
        h_neigh = layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
        if beta is None:
            return h_self + h_neigh
        return beta.unsqueeze(-1) * h_self + (1 - beta.unsqueeze(-1)) * h_neigh

    def forward(self, bow_x, ei, llm_e):
        b = self._beta(ei, llm_e)
        h = F.relu(self._apply_gate(self.l1, bow_x, ei, b))
        b = self._beta(ei, llm_e)   # recompute — llm_e is frozen
        h = F.relu(self._apply_gate(self.l2, h, ei, b))
        return self.clf(h)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_eval(gate, bow_x, ei, llm_e, labels, trm, vm, tem, nc, seed, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LLMGatedSAGE(bow_x.size(1), llm_e.size(1), 128, nc, gate=gate)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        m.train()
        F.cross_entropy(m(bow_x, ei, llm_e)[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                lg = m(bow_x, ei, llm_e)
            pred = lg.argmax(-1)
            va = (pred[vm]  == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt


def inject(ei, feat, labels, ratio):
    """Add text-similar, wrong-class confusing edges."""
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


# ── Config ────────────────────────────────────────────────────────────────────

GATES = ["none", "cosine", "entropy", "mlp", "variance",
         "max_cosine", "nbr_agree", "bilinear", "topk", "cos_std"]
LABEL = {
    "none":       "No gate (BOW baseline)",
    "cosine":     "A — LLM cosine",
    "entropy":    "B — LLM entropy",
    "mlp":        "C — LLM MLP",
    "variance":   "D — LLM variance",
    "max_cosine": "E — LLM max cosine",
    "nbr_agree":  "F — neighbour agreement",
    "bilinear":   "G — LLM bilinear",
    "topk":       "H — LLM top-k cosine",
    "cos_std":    "I — cosine std",
}
COLOR = {
    "none":       "#888888",
    "cosine":     "#4C9BE8",
    "entropy":    "#E86B4C",
    "mlp":        "#7C4CE8",
    "variance":   "#4CC8A0",
    "max_cosine": "#F59E0B",
    "nbr_agree":  "#10B981",
    "bilinear":   "#EC4899",
    "topk":       "#6366F1",
    "cos_std":    "#14B8A6",
}
SEEDS  = 3
EPOCHS = 200
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
CORA_PT = os.path.join(_root, "STEM-GNN/dataset/data/single_graph/Cora/cora.pt")
PUB_PT  = os.path.join(_root, "STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt")

all_results = {}

# ── Dataset loop ──────────────────────────────────────────────────────────────
for ds_name, bow_name, raw_pt in [
    ("Cora",   "Cora",   CORA_PT),
    ("PubMed", "PubMed", PUB_PT),
]:
    print(f"\n{'='*60}")
    print(f"  {ds_name.upper()}  —  BOW backbone + LLM gate")
    print(f"{'='*60}")

    # 1) Original BOW features (backbone)
    pl_ds   = Planetoid(PL_ROOT, bow_name)
    pl_data = pl_ds[0]
    bow_x   = F.normalize(pl_data.x.float(), dim=-1)   # [N, 1433/500]
    labels  = pl_data.y.long()
    nc      = int(labels.max().item()) + 1
    ei_base = to_undirected(pl_data.edge_index, num_nodes=pl_data.num_nodes)

    # 2) LLM gate embeddings from raw text
    print(f"  Encoding {ds_name} texts with DistilBERT ...", flush=True)
    raw_data = torch.load(raw_pt)
    llm_e    = encode_texts(raw_data.raw_texts)   # [N, 768], frozen
    print(f"  BOW: {bow_x.shape}  |  LLM: {llm_e.shape}", flush=True)

    # 3) Masks (Planetoid fixed split)
    N   = bow_x.size(0)
    trm = pl_data.train_mask
    vm  = pl_data.val_mask
    tem = pl_data.test_mask

    # 4) Injection sweep
    print(f"\n  Injection sweep:")
    res = {}
    for gate in GATES:
        res[gate] = []
        for r in RATIOS:
            ei = inject(ei_base, llm_e, labels, r)
            acc = float(np.mean([
                train_eval(gate, bow_x, ei, llm_e, labels,
                           trm, vm, tem, nc, s, EPOCHS)
                for s in range(SEEDS)
            ]))
            res[gate].append(round(acc, 4))
            print(f"    {LABEL[gate]:<28} ratio={r:.0%}  {acc*100:.2f}%", flush=True)

    all_results[f"{ds_name.lower()}_sweep"] = {"ratios": RATIOS, "results": res}

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== Done. Saved → {out_path} ===")
