"""
New LLM gate signals and fusion strategies.

Signals:
  attn_entropy  — softmax attention entropy over neighbours
  energy_dist   — statistical energy distance between node and neighbourhood
  spectral      — local graph Laplacian coherence (algebraic connectivity proxy)

Fusion:
  llm_gat       — per-edge LLM attention weights (LLM-GAT)
  llm_appnp     — personalized propagation with LLM teleportation coefficient
  moe_fusion    — 3 expert aggregators (mean/max/attn) routed by LLM signal
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_max, scatter_softmax

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── LLM encoder ───────────────────────────────────────────────────────────────
def encode_texts(texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("multi-qa-distilbert-cos-v1")
    m.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        with torch.no_grad():
            embs.append(m.encode(texts[i:i+batch_size], convert_to_tensor=True,
                                 show_progress_bar=False).cpu())
    return torch.cat(embs, 0)

# ══════════════════════════════════════════════════════════════════════════════
# NEW GATE SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def gate_attn_entropy(ei, e):
    """
    Signal O — Attention Entropy.
    Compute softmax attention weights a_{vu} = softmax(e_v·e_u/sqrt(d))
    over all neighbours u. High entropy = spread/uncertain = distrust nbrs.
    β_v = σ(normalised_entropy * 5)
    """
    row, col = ei; N = e.size(0); d = e.size(1)
    ef = F.normalize(e, dim=-1)
    # Per-edge attention logit
    logit = (ef[row] * ef[col]).sum(-1) / (d ** 0.5)
    # Softmax over each node's neighbourhood
    attn = scatter_softmax(logit, row, dim=0)          # [E]
    # Entropy per node: H = -Σ a log a
    eps = 1e-9
    ent = scatter(-attn * (attn + eps).log(), row, dim=0, dim_size=N, reduce="sum")
    # Normalise by log(degree) so entropy is in [0,1]
    deg  = scatter(torch.ones_like(row, dtype=torch.float), row,
                   dim=0, dim_size=N, reduce="sum").clamp(min=1)
    ent_norm = ent / deg.log().clamp(min=eps)          # [N]
    return torch.sigmoid(ent_norm * 5)

def gate_energy_dist(ei, e):
    """
    Signal P — Energy Distance.
    E_dist(v, N(v)) = 2·E[||e_v - e_u||] - E[||e_u - e_u'||]
    Measures how far node v is from its neighbourhood distribution.
    High energy dist = v is an outlier in its neighbourhood (or nbrs are mixed).
    β_v = σ(energy_dist * 2)
    """
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    # Term 1: mean ||e_v - e_u|| for each node v
    diff_vu  = (ef[row] - ef[col]).norm(dim=-1)        # [E]
    mean_vu  = scatter_mean(diff_vu, row, dim=0, dim_size=N)  # [N]
    # Term 2: mean ||e_u - e_u'|| — approx as variance of neighbour embeddings
    mu_nbr   = scatter_mean(ef[col], row, dim=0, dim_size=N)  # [N, d]
    var_nbr  = scatter_mean((ef[col] - mu_nbr[row]).norm(dim=-1),
                             row, dim=0, dim_size=N)   # [N]
    energy   = (2 * mean_vu - var_nbr).clamp(min=0)   # [N]
    return torch.sigmoid(energy * 2)

def gate_spectral(ei, e):
    """
    Signal Q — Spectral Coherence.
    Build local similarity-weighted adjacency for ego-network,
    compute its Fiedler value (2nd eigenvalue proxy) via:
      fiedler_proxy = 1 - max_u cos(e_v, e_u)  +  std of cosines
    Low max-sim and high std = disconnected neighbourhood = high β.
    (True Fiedler is O(n³); this is a fast closed-form proxy.)
    """
    row, col = ei; N = e.size(0)
    ef  = F.normalize(e, dim=-1)
    cos = (ef[row] * ef[col]).sum(-1)                  # [E]
    max_cos = scatter(cos, row, dim=0, dim_size=N, reduce="max")
    mu_cos  = scatter_mean(cos, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos - mu_cos[row])**2, row, dim=0, dim_size=N)
    std_cos = var_cos.sqrt()
    # Low max-cos + high std → fragmented, uncertain → trust self
    proxy = (1 - max_cos) + std_cos
    return torch.sigmoid(proxy * 3)

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE LAYERS
# ══════════════════════════════════════════════════════════════════════════════

class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))

# ══════════════════════════════════════════════════════════════════════════════
# NEW FUSION MODELS
# ══════════════════════════════════════════════════════════════════════════════

class LLMGATModel(nn.Module):
    """
    Fusion R — Per-edge LLM Attention (LLM-GAT).
    Replaces GAT's learned feature attention with frozen LLM cross-attention.
    Per-edge weight: w_{vu} = softmax_u( e_v^T W_q^T W_k e_u / sqrt(r) )
    Aggregation: h_v = W_self·x_v + Σ_u w_{vu}·W_neigh·x_u
    """
    def __init__(self, bow_dim, llm_dim, hid, nc, rank=64):
        super().__init__()
        self.Wq  = nn.Linear(llm_dim, rank, bias=False)
        self.Wk  = nn.Linear(llm_dim, rank, bias=False)
        self.l1  = SAGELayer(bow_dim, hid)   # reuse Ws, Wn
        self.l2  = SAGELayer(hid,    hid)
        self.clf = nn.Linear(hid, nc)
        self.hid = hid

    def _edge_weighted_agg(self, layer, x, ei, llm_e):
        row, col = ei; N = x.size(0)
        # Per-edge attention score from LLM
        q = self.Wq(llm_e[row]); k = self.Wk(llm_e[col])
        score = (q * k).sum(-1) / (q.size(-1) ** 0.5)
        w = scatter_softmax(score, row, dim=0)          # [E]
        # Weighted aggregation
        hs = layer.Ws(x)
        hn_edges = layer.Wn(x[col]) * w.unsqueeze(-1)  # [E, hid]
        hn = torch.zeros(N, layer.Wn.out_features, device=x.device)
        hn.scatter_add_(0, row.unsqueeze(-1).expand_as(hn_edges), hn_edges)
        return hs + hn

    def forward(self, bow_x, ei, llm_e, **kw):
        h = F.relu(self._edge_weighted_agg(self.l1, bow_x, ei, llm_e))
        h = F.relu(self._edge_weighted_agg(self.l2, h,     ei, llm_e))
        return self.clf(h)


class LLMAPPNPModel(nn.Module):
    """
    Fusion S — Personalized Propagation with LLM Teleportation (LLM-APPNP).
    Standard APPNP uses a fixed global alpha. Here alpha_v = beta_v from
    LLM variance gate — each node gets its own teleportation coefficient.
    h^(0) = MLP(x_v)
    h^(k) = (1-β_v)·A_hat·h^(k-1) + β_v·h^(0)   for k=1..K
    """
    def __init__(self, bow_dim, llm_dim, hid, nc, K=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(bow_dim, hid), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(hid, hid))
        self.clf = nn.Linear(hid, nc)
        self.K   = K

    def _beta(self, ei, e):
        # Variance gate (our best signal)
        row, col = ei; N = e.size(0)
        ef  = F.normalize(e, dim=-1)
        mu  = scatter_mean(ef[col], row, dim=0, dim_size=N)
        var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2,
                           row, dim=0, dim_size=N)
        return torch.sigmoid(var * 10)                 # [N]

    def _norm_agg(self, h, ei):
        row, col = ei; N = h.size(0)
        deg  = scatter(torch.ones_like(row, dtype=torch.float),
                       row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
        return scatter_mean(h[col], row, dim=0, dim_size=N)

    def forward(self, bow_x, ei, llm_e, **kw):
        h0   = F.dropout(self.enc(bow_x), p=0.5, training=self.training)
        beta = self._beta(ei, llm_e).unsqueeze(-1)     # [N,1]
        h    = h0
        for _ in range(self.K):
            h = (1 - beta) * self._norm_agg(h, ei) + beta * h0
        return self.clf(h)


class MoEFusionModel(nn.Module):
    """
    Fusion T — Mixture of Expert Aggregators.
    3 experts, each with different aggregation:
      Expert 0: mean aggregation  (standard SAGE)
      Expert 1: max  aggregation  (captures strongest signal)
      Expert 2: LLM-weighted attention aggregation
    Router: LLM entropy gate → softmax over 3 experts per node.
    h_v = Σ_k r_k(v) · Expert_k(x_v, N(v))
    """
    def __init__(self, bow_dim, llm_dim, hid, nc, rank=32):
        super().__init__()
        # Shared input transform
        self.lin_in  = nn.Linear(bow_dim, hid, bias=False)
        # Expert aggregators (different Wn per expert)
        self.Wn = nn.ModuleList([nn.Linear(bow_dim, hid, bias=False) for _ in range(3)])
        self.Ws = nn.ModuleList([nn.Linear(bow_dim, hid, bias=False) for _ in range(3)])
        # LLM-based router: maps [attn_entropy, energy_dist, spectral] → 3 logits
        self.router  = nn.Linear(3, 3, bias=True)
        # Expert 2 attention keys/queries
        self.Wq2 = nn.Linear(llm_dim, rank, bias=False)
        self.Wk2 = nn.Linear(llm_dim, rank, bias=False)
        # Layer 2
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)

    def _signals(self, ei, e):
        s_ent  = gate_attn_entropy(ei, e)              # [N]
        s_eng  = gate_energy_dist(ei, e)               # [N]
        s_spec = gate_spectral(ei, e)                  # [N]
        return torch.stack([s_ent, s_eng, s_spec], dim=-1)  # [N,3]

    def _expert_agg(self, k, x, ei, llm_e):
        row, col = ei; N = x.size(0)
        hs = self.Ws[k](x)
        if k == 0:   # mean
            hn = self.Wn[k](scatter_mean(x[col], row, dim=0, dim_size=N))
        elif k == 1: # max
            hn_e = self.Wn[k](x[col])
            hn, _ = scatter_max(hn_e, row, dim=0, out=torch.zeros(N, hn_e.size(-1), device=x.device))
        else:        # llm-weighted attention
            q = self.Wq2(llm_e[row]); kk = self.Wk2(llm_e[col])
            score = (q * kk).sum(-1) / (q.size(-1)**0.5)
            w  = scatter_softmax(score, row, dim=0)
            hn_e = self.Wn[k](x[col]) * w.unsqueeze(-1)
            hn   = torch.zeros(N, hn_e.size(-1), device=x.device)
            hn.scatter_add_(0, row.unsqueeze(-1).expand_as(hn_e), hn_e)
        return hs + hn

    def forward(self, bow_x, ei, llm_e, **kw):
        sigs    = self._signals(ei, llm_e)             # [N, 3]
        routing = F.softmax(self.router(sigs), dim=-1) # [N, 3]
        # Compute all 3 experts and route
        outs = torch.stack([self._expert_agg(k, bow_x, ei, llm_e)
                            for k in range(3)], dim=-1)  # [N, hid, 3]
        h = (outs * routing.unsqueeze(1)).sum(-1)       # [N, hid]
        h = F.relu(h)
        h = F.relu(self.l2(h, ei))
        return self.clf(h)


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD GATED SAGE (for signals O, P, Q)
# ══════════════════════════════════════════════════════════════════════════════

class GatedSAGE(nn.Module):
    def __init__(self, bow_dim, llm_dim, hid, nc, gate):
        super().__init__()
        self.gate = gate
        self.l1   = SAGELayer(bow_dim, hid)
        self.l2   = SAGELayer(hid,    hid)
        self.clf  = nn.Linear(hid, nc)

    def _beta(self, ei, e):
        if self.gate == "attn_entropy": return gate_attn_entropy(ei, e)
        if self.gate == "energy_dist":  return gate_energy_dist(ei, e)
        if self.gate == "spectral":     return gate_spectral(ei, e)
        raise ValueError(self.gate)

    def _apply(self, layer, x, ei, beta):
        row, col = ei; N = x.size(0)
        hs = layer.Ws(x)
        hn = layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
        b  = beta.unsqueeze(-1)
        return b * hs + (1 - b) * hn

    def forward(self, bow_x, ei, llm_e, **kw):
        b = self._beta(ei, llm_e)
        h = F.relu(self._apply(self.l1, bow_x, ei, b))
        b = self._beta(ei, llm_e)
        h = F.relu(self._apply(self.l2, h,     ei, b))
        return self.clf(h)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_eval(model, bow_x, ei, llm_e, labels, trm, vm, tem,
               seed, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        F.cross_entropy(model(bow_x, ei, llm_e)[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bow_x, ei, llm_e)
            pred = logits.argmax(-1)
            va = (pred[vm]  == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt

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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

METHODS = {
    # Signal gates (drop-in for GatedSAGE)
    "attn_entropy": ("O — Attn Entropy",    "signal"),
    "energy_dist":  ("P — Energy Distance", "signal"),
    "spectral":     ("Q — Spectral Coher.", "signal"),
    # Fusion models
    "llm_gat":      ("R — LLM-GAT (per-edge)", "fusion"),
    "llm_appnp":    ("S — LLM-APPNP",          "fusion"),
    "moe_fusion":   ("T — MoE Fusion",          "fusion"),
}

SEEDS   = 3
EPOCHS  = 200
RATIOS  = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
CORA_PT = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(EXISTING) as f:
    all_results = json.load(f)

for ds_name, bow_name, raw_pt, sweep_key in [
    ("Cora",   "Cora",   CORA_PT, "cora_sweep"),
    ("PubMed", "PubMed", PUB_PT,  "pubmed_sweep"),
]:
    print(f"\n{'='*60}")
    print(f"  {ds_name}  —  new signals & fusion methods")
    print(f"{'='*60}", flush=True)

    pl_data = Planetoid(PL_ROOT, bow_name)[0]
    bow_x   = F.normalize(pl_data.x.float(), dim=-1)
    labels  = pl_data.y.long()
    nc      = int(labels.max().item()) + 1
    d       = bow_x.size(1)
    ei_base = to_undirected(pl_data.edge_index, num_nodes=pl_data.num_nodes)
    trm, vm, tem = pl_data.train_mask, pl_data.val_mask, pl_data.test_mask

    print(f"  Encoding texts ...", flush=True)
    raw_data = torch.load(raw_pt)
    llm_e    = encode_texts(raw_data.raw_texts)
    print(f"  BOW {bow_x.shape}  LLM {llm_e.shape}", flush=True)

    for key, (label, kind) in METHODS.items():
        print(f"\n  ── {label} ──", flush=True)
        res = []
        for r in RATIOS:
            ei = inject(ei_base, llm_e, labels, r)
            accs = []
            for s in range(SEEDS):
                torch.manual_seed(s)
                if kind == "signal":
                    m = GatedSAGE(d, llm_e.size(1), 128, nc, gate=key)
                elif key == "llm_gat":
                    m = LLMGATModel(d, llm_e.size(1), 128, nc)
                elif key == "llm_appnp":
                    m = LLMAPPNPModel(d, llm_e.size(1), 128, nc, K=10)
                elif key == "moe_fusion":
                    m = MoEFusionModel(d, llm_e.size(1), 128, nc)
                accs.append(train_eval(m, bow_x, ei, llm_e, labels,
                                       trm, vm, tem, s, EPOCHS))
            acc = float(np.mean(accs))
            res.append(round(acc, 4))
            print(f"  {key:<15} ratio={r:.0%}  {acc*100:.2f}%", flush=True)
        all_results[sweep_key]["results"][key] = res

OUT = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== Saved → {OUT} ===")
