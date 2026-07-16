"""
Robustness at subgraph fusion.

Published robustness baselines:
  robust_gcn   — Gaussian repr + variance-based attention (Zhu et al. KDD 2019)
  gnn_guard    — per-edge trust weights from feature sim (Zhang & Zitnik NeurIPS 2020)

New LLM-guided fusion methods:
  llm_trimmed    — drop bottom-20% neighbours by LLM cosine before aggregating
  llm_consensus  — only aggregate neighbours LLM predicts same class (proto gate)
  llm_multiscale — separate 1-hop/2-hop, fused with LLM variance gate per node
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_softmax, scatter_max

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

CORA_DESCS = [
    "case based reasoning and analogical retrieval systems",
    "genetic algorithms evolutionary computation and optimization",
    "artificial neural networks deep learning and connectionist models",
    "probabilistic graphical models Bayesian inference and belief networks",
    "reinforcement learning reward-based agents and policy optimization",
    "rule learning inductive logic programming and symbolic methods",
    "computational learning theory formal analysis and complexity",
]
PUBMED_DESCS = [
    "experimental diabetes mellitus animal models laboratory research",
    "type 1 juvenile insulin-dependent diabetes mellitus treatment",
    "type 2 adult-onset non-insulin-dependent diabetes mellitus",
]

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 1 — RobustGCN (Zhu et al., KDD 2019)
# Gaussian node representations; low-variance neighbours get higher attention.
# ══════════════════════════════════════════════════════════════════════════════
class RobustGCNConv(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W_mu  = nn.Linear(in_d, out_d, bias=False)
        self.W_sig = nn.Linear(in_d, out_d, bias=False)

    def forward(self, mu, sig2, ei, N):
        row, col = ei
        # Variance-based attention: lower neighbour variance → more trust
        attn = torch.exp(-sig2[col].mean(-1).clamp(max=10))  # [E]
        attn_sum = scatter(attn, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
        w = attn / attn_sum                                    # [E] normalised

        # Aggregate means
        agg_mu  = scatter(mu[col]  * w.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        new_mu  = F.relu(self.W_mu(mu + agg_mu))

        # Propagate variances (weighted sum of neighbour variances)
        agg_sig = scatter(sig2[col] * (w**2).unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        new_sig = F.relu(self.W_sig(sig2 + agg_sig))
        return new_mu, new_sig

class RobustGCNModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        # Initial variance lives in same space as input features
        self.sig_init = nn.Linear(in_d, in_d, bias=False)
        self.c1  = RobustGCNConv(in_d, hid)   # in_d → hid
        self.c2  = RobustGCNConv(hid,  hid)   # hid  → hid
        self.clf = nn.Linear(hid, nc)

    def _encode(self, x, ei):
        N = x.size(0)
        mu   = x
        sig2 = F.relu(self.sig_init(x)) + 1e-4   # [N, in_d]
        mu,  sig2 = self.c1(mu,  sig2, ei, N)     # [N, hid], [N, hid]
        mu,  sig2 = self.c2(mu,  sig2, ei, N)     # [N, hid], [N, hid]
        return mu, sig2

    def forward(self, x, ei, **kw):
        mu, _ = self._encode(x, ei)
        return self.clf(mu)

    def kl_loss(self, x, ei):
        mu, sig2 = self._encode(x, ei)
        return 0.5 * (sig2 + mu**2 - sig2.clamp(min=1e-9).log() - 1).mean()

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2 — GNNGuard (Zhang & Zitnik, NeurIPS 2020)
# Recomputes per-edge trust weights from feature cosine similarity each layer.
# Prunes edges with near-zero similarity; normalises survivors.
# ══════════════════════════════════════════════════════════════════════════════
def gnnguard_weights(h, ei, N, threshold=0.0):
    row, col = ei
    hf  = F.normalize(h, dim=-1)
    sim = (hf[row] * hf[col]).sum(-1).clamp(min=threshold)   # [E]
    # Degree-normalised weights
    w_sum = scatter(sim, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
    return sim / w_sum                                         # [E]

class GNNGuardModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.lin1 = nn.Linear(in_d, hid, bias=False)
        self.lin2 = nn.Linear(hid,  hid, bias=False)
        self.clf  = nn.Linear(hid,  nc)

    def _conv(self, lin, x, ei, w, N):
        row, col = ei
        agg = scatter(x[col] * w.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(lin(x + agg))

    def forward(self, x, ei, **kw):
        N = x.size(0)
        # Layer 1: initial edge weights from raw features
        w1 = gnnguard_weights(x, ei, N)
        h  = self._conv(self.lin1, x, ei, w1, N)
        # Layer 2: recompute weights from hidden representations
        w2 = gnnguard_weights(h, ei, N)
        h  = self._conv(self.lin2, h, ei, w2, N)
        return self.clf(h)

# ══════════════════════════════════════════════════════════════════════════════
# NEW LLM FUSION METHODS
# ══════════════════════════════════════════════════════════════════════════════

class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei, N=None):
        if N is None: N = x.size(0)
        row, col = ei
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))


class LLMTrimmedModel(nn.Module):
    """
    LLM Trimmed Mean Fusion.
    For each node, rank neighbours by LLM cosine(e_v, e_u).
    Drop bottom trim_pct% before aggregating BOW features.
    This surgically removes the injected adversarial neighbours.
    """
    def __init__(self, bow_dim, llm_dim, hid, nc, trim_pct=0.2):
        super().__init__()
        self.trim = trim_pct
        self.l1   = SAGELayer(bow_dim, hid)
        self.l2   = SAGELayer(hid, hid)
        self.clf  = nn.Linear(hid, nc)

    def _trimmed_agg(self, layer, x, ei, llm_e):
        row, col = ei; N = x.size(0)
        ef  = F.normalize(llm_e, dim=-1)
        cos = (ef[row] * ef[col]).sum(-1)                     # [E] LLM cosine

        # For each node, keep only top (1-trim_pct) neighbours by LLM cosine
        # Use soft trimming: weight = σ((cos - quantile) / T)
        # Hard trimming via sorting:
        deg = scatter(torch.ones_like(row, dtype=torch.float),
                      row, dim=0, dim_size=N, reduce="sum").clamp(min=1)

        # Per-node rank of each edge (higher cos = lower rank = keep)
        # Use neg cosine as sort key so argsort gives descending order
        neg_cos  = -cos.detach()
        sort_idx = torch.argsort(row.float() * 1e9 + neg_cos)  # sort by (node, -cos)
        row_s    = row[sort_idx]

        positions = torch.arange(row_s.size(0), device=x.device).float()
        first_pos = scatter(positions, row_s, dim=0, dim_size=N, reduce="min").long()
        rank      = (positions - first_pos[row_s]).long()

        n_keep    = (deg * (1 - self.trim)).long().clamp(min=1)
        keep_mask = rank < n_keep[row]                         # [E]

        # Aggregation over kept edges
        hs  = layer.Ws(x)
        if keep_mask.any():
            r_k  = row[keep_mask]; c_k = col[keep_mask]
            agg  = scatter_mean(x[c_k], r_k, dim=0, dim_size=N)
        else:
            agg  = torch.zeros_like(x[:, :layer.Wn.in_features])
        hn = layer.Wn(agg)
        return hs + hn

    def forward(self, bow_x, ei, llm_e, **kw):
        h = F.relu(self._trimmed_agg(self.l1, bow_x, ei, llm_e))
        h = F.relu(self._trimmed_agg(self.l2, h,     ei, llm_e))
        return self.clf(h)


class LLMConsensusModel(nn.Module):
    """
    LLM Consensus Fusion.
    Use LLM prototype predictions to build a semantic-agreement mask:
    only aggregate neighbours where LLM predicts the SAME class as node v.
    Falls back to self-only when no consensus neighbours exist.
    """
    def __init__(self, bow_dim, llm_dim, hid, nc):
        super().__init__()
        self.l1  = SAGELayer(bow_dim, hid)
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)

    def _consensus_agg(self, layer, x, ei, llm_e, protos):
        row, col = ei; N = x.size(0)
        ef   = F.normalize(llm_e, dim=-1)
        pf   = F.normalize(protos, dim=-1)
        pred = (ef @ pf.T).argmax(-1)                         # [N] LLM pseudo-labels

        same = (pred[row] == pred[col])                        # [E] agreement mask
        hs   = layer.Ws(x)

        if same.any():
            r_s = row[same]; c_s = col[same]
            agg = scatter_mean(x[c_s], r_s, dim=0, dim_size=N)
        else:
            agg = torch.zeros_like(x[:, :layer.Wn.in_features])

        # For nodes with NO consensus neighbours, fall back to self
        has_consensus = scatter(same.float(), row, dim=0, dim_size=N, reduce="sum") > 0
        # Where no consensus, agg is already 0 → hs only (self transform)
        hn = layer.Wn(agg)
        return hs + hn

    def forward(self, bow_x, ei, llm_e, protos=None, **kw):
        h = F.relu(self._consensus_agg(self.l1, bow_x, ei, llm_e, protos))
        h = F.relu(self._consensus_agg(self.l2, h,     ei, llm_e, protos))
        return self.clf(h)


class LLMMultiScaleModel(nn.Module):
    """
    LLM Multi-Scale Fusion.
    Compute 1-hop and 2-hop subgraph representations separately.
    Use LLM variance gate to determine which scale to trust per node:
      β_v high → trust 1-hop (local, less attacked)
      β_v low  → trust 2-hop (wider context, more averaged out)
    Final: h_v = β_v · h_v^(1hop) + (1-β_v) · h_v^(2hop)
    """
    def __init__(self, bow_dim, llm_dim, hid, nc):
        super().__init__()
        # 1-hop branch
        self.enc1h = nn.Linear(bow_dim, hid, bias=False)
        self.Wn1h  = nn.Linear(bow_dim, hid, bias=False)
        # 2-hop branch
        self.enc2h = nn.Linear(bow_dim, hid, bias=False)
        self.Wn2h  = nn.Linear(bow_dim, hid, bias=False)
        # Shared classifier
        self.clf   = nn.Linear(hid, nc)

    def _hop1(self, x, ei):
        row, col = ei; N = x.size(0)
        return F.relu(self.enc1h(x) + self.Wn1h(scatter_mean(x[col], row, dim=0, dim_size=N)))

    def _hop2(self, x, ei):
        # 2-hop: aggregate neighbours-of-neighbours
        row, col = ei; N = x.size(0)
        # First pass: get neighbour means
        nbr_mean = scatter_mean(x[col], row, dim=0, dim_size=N)          # 1-hop mean
        # Second pass: aggregate those means
        nbr2_mean = scatter_mean(nbr_mean[col], row, dim=0, dim_size=N)  # 2-hop mean
        return F.relu(self.enc2h(x) + self.Wn2h(nbr2_mean))

    def _variance_gate(self, ei, llm_e):
        row, col = ei; N = llm_e.size(0)
        ef  = F.normalize(llm_e, dim=-1)
        mu  = scatter_mean(ef[col], row, dim=0, dim_size=N)
        var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2,
                           row, dim=0, dim_size=N)
        return torch.sigmoid(var * 10)                                    # [N] ∈ [0,1]

    def forward(self, bow_x, ei, llm_e, **kw):
        h1   = self._hop1(bow_x, ei)           # 1-hop subgraph representation
        h2   = self._hop2(bow_x, ei)           # 2-hop subgraph representation
        beta = self._variance_gate(ei, llm_e).unsqueeze(-1)  # [N,1]
        # High β → trust 1-hop (local, fewer injected edges)
        h    = beta * h1 + (1 - beta) * h2
        return self.clf(h)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_eval(model, bow_x, ei, llm_e, labels, trm, vm, tem,
               seed, epochs=200, protos=None, is_robust_gcn=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(bow_x, ei, llm_e=llm_e, protos=protos)
        loss   = F.cross_entropy(logits[trm], labels[trm])
        if is_robust_gcn:
            loss = loss + 5e-4 * model.kl_loss(bow_x, ei)
        loss.backward(); opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bow_x, ei, llm_e=llm_e, protos=protos)
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

SEEDS   = 3
EPOCHS  = 200
RATIOS  = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
CORA_PT = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(EXISTING) as f:
    all_results = json.load(f)

for ds_name, bow_name, raw_pt, sweep_key, class_descs in [
    ("Cora",   "Cora",   CORA_PT, "cora_sweep",   CORA_DESCS),
    ("PubMed", "PubMed", PUB_PT,  "pubmed_sweep",  PUBMED_DESCS),
]:
    print(f"\n{'='*60}")
    print(f"  {ds_name}  —  robust fusion methods")
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

    print(f"  Encoding class descriptions ...", flush=True)
    protos   = encode_texts(class_descs)

    # Zero-shot proto accuracy
    ef   = F.normalize(llm_e, dim=-1)
    pf   = F.normalize(protos, dim=-1)
    pred = (ef @ pf.T).argmax(-1)
    proto_acc = (pred == labels).float().mean().item()
    print(f"  Proto zero-shot acc: {proto_acc*100:.1f}%", flush=True)

    METHODS = [
        ("robust_gcn",    "RobustGCN (Zhu 2019)",          "baseline"),
        ("gnn_guard",     "GNNGuard (Zhang 2020)",          "baseline"),
        ("llm_trimmed",   "LLM Trimmed Mean Fusion",        "ours"),
        ("llm_consensus", "LLM Consensus Fusion",           "ours"),
        ("llm_multiscale","LLM Multi-Scale Fusion",         "ours"),
    ]

    for key, label, kind in METHODS:
        print(f"\n  ── {label} ──", flush=True)
        res = []
        for r in RATIOS:
            ei = inject(ei_base, llm_e, labels, r)
            accs = []
            for s in range(SEEDS):
                torch.manual_seed(s)
                if   key == "robust_gcn":
                    m = RobustGCNModel(d, 128, nc)
                    a = train_eval(m, bow_x, ei, llm_e, labels, trm, vm, tem,
                                   s, EPOCHS, is_robust_gcn=True)
                elif key == "gnn_guard":
                    m = GNNGuardModel(d, 128, nc)
                    a = train_eval(m, bow_x, ei, llm_e, labels, trm, vm, tem, s, EPOCHS)
                elif key == "llm_trimmed":
                    m = LLMTrimmedModel(d, llm_e.size(1), 128, nc)
                    a = train_eval(m, bow_x, ei, llm_e, labels, trm, vm, tem, s, EPOCHS)
                elif key == "llm_consensus":
                    m = LLMConsensusModel(d, llm_e.size(1), 128, nc)
                    a = train_eval(m, bow_x, ei, llm_e, labels, trm, vm, tem,
                                   s, EPOCHS, protos=protos)
                elif key == "llm_multiscale":
                    m = LLMMultiScaleModel(d, llm_e.size(1), 128, nc)
                    a = train_eval(m, bow_x, ei, llm_e, labels, trm, vm, tem, s, EPOCHS)
                accs.append(a)
            acc = float(np.mean(accs))
            res.append(round(acc, 4))
            print(f"  {key:<18} ratio={r:.0%}  {acc*100:.2f}%", flush=True)
        all_results[sweep_key]["results"][key] = res

OUT = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== Saved → {OUT} ===")
