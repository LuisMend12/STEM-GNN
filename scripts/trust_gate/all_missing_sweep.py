"""
Single sweep for all 11 missing methods — 2 seeds x 150 epochs.

New gate signals (O-Q):
  attn_entropy  — softmax attention entropy over neighbours
  energy_dist   — energy distance node vs neighbourhood distribution
  spectral      — local Laplacian coherence proxy

New fusion methods (R-T):
  llm_gat       — per-edge LLM cross-attention weights (LLM-GAT)
  llm_appnp     — APPNP with per-node LLM teleportation coefficient
  moe_fusion    — 3 expert aggregators routed by LLM signals

Robustness baselines:
  robust_gcn    — Gaussian repr + variance attention (Zhu et al. KDD 2019)
  gnn_guard     — per-edge trust from feature sim (Zhang & Zitnik NeurIPS 2020)

LLM subgraph fusion (novel, targeting fusion robustness):
  llm_trimmed    — drop bottom-20% neighbours by LLM cosine before aggregating
  llm_consensus  — only aggregate LLM-predicted same-class neighbours
  llm_multiscale — 1-hop vs 2-hop subgraph fused with LLM variance gate
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_softmax, scatter_max

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
# SHARED BACKBONE
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
# GATE SIGNAL FUNCTIONS (O, P, Q)
# ══════════════════════════════════════════════════════════════════════════════
def gate_attn_entropy(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    logit = (ef[row] * ef[col]).sum(-1) / (e.size(1) ** 0.5)
    attn  = scatter_softmax(logit, row, dim=0)
    eps   = 1e-9
    ent   = scatter(-(attn * (attn + eps).log()), row, dim=0, dim_size=N, reduce="sum")
    deg   = scatter(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
    return torch.sigmoid(ent / deg.log().clamp(min=eps) * 5)

def gate_energy_dist(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    diff_vu  = (ef[row] - ef[col]).norm(dim=-1)
    mean_vu  = scatter_mean(diff_vu, row, dim=0, dim_size=N)
    mu_nbr   = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var_nbr  = scatter_mean((ef[col] - mu_nbr[row]).norm(dim=-1), row, dim=0, dim_size=N)
    return torch.sigmoid((2 * mean_vu - var_nbr).clamp(min=0) * 2)

def gate_spectral(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos = (ef[row] * ef[col]).sum(-1)
    max_cos = scatter(cos, row, dim=0, dim_size=N, reduce="max")
    mu_cos  = scatter_mean(cos, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos - mu_cos[row])**2, row, dim=0, dim_size=N)
    return torch.sigmoid(((1 - max_cos) + var_cos.sqrt()) * 3)

def gate_variance(ei, e):
    row, col = ei; N = e.size(0)
    ef  = F.normalize(e, dim=-1)
    mu  = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

# ── Signal gates O, P, Q ──────────────────────────────────────────────────────
class GatedSAGE(nn.Module):
    def __init__(self, bow_dim, hid, nc, gate_fn):
        super().__init__()
        self.gate_fn = gate_fn
        self.l1 = SAGELayer(bow_dim, hid)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _apply(self, layer, x, ei, b):
        row, col = ei; N = x.size(0)
        b = b.unsqueeze(-1)
        return b * layer.Ws(x) + (1-b) * layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
    def forward(self, bow_x, ei, llm_e, **kw):
        b = self.gate_fn(ei, llm_e)
        h = F.relu(self._apply(self.l1, bow_x, ei, b))
        b = self.gate_fn(ei, llm_e)
        h = F.relu(self._apply(self.l2, h, ei, b))
        return self.clf(h)

# ── LLM-GAT (R): per-edge attention from LLM ─────────────────────────────────
class LLMGATModel(nn.Module):
    def __init__(self, bow_dim, llm_dim, hid, nc, rank=64):
        super().__init__()
        self.Wq = nn.Linear(llm_dim, rank, bias=False)
        self.Wk = nn.Linear(llm_dim, rank, bias=False)
        self.l1 = SAGELayer(bow_dim, hid)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _agg(self, layer, x, ei, llm_e):
        row, col = ei; N = x.size(0)
        q = self.Wq(llm_e[row]); k = self.Wk(llm_e[col])
        w = scatter_softmax((q * k).sum(-1) / (q.size(-1)**0.5), row, dim=0)
        hs = layer.Ws(x)
        hn = torch.zeros(N, layer.Wn.out_features, device=x.device)
        hn.scatter_add_(0, row.unsqueeze(-1).expand(row.size(0), layer.Wn.out_features),
                        layer.Wn(x[col]) * w.unsqueeze(-1))
        return hs + hn
    def forward(self, bow_x, ei, llm_e, **kw):
        h = F.relu(self._agg(self.l1, bow_x, ei, llm_e))
        h = F.relu(self._agg(self.l2, h,     ei, llm_e))
        return self.clf(h)

# ── LLM-APPNP (S): personalized propagation with LLM teleportation ────────────
class LLMAPPNPModel(nn.Module):
    def __init__(self, bow_dim, hid, nc, K=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(bow_dim, hid), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(hid, hid))
        self.clf = nn.Linear(hid, nc)
        self.K = K
    def forward(self, bow_x, ei, llm_e, **kw):
        row, col = ei; N = bow_x.size(0)
        h0   = F.dropout(self.enc(bow_x), p=0.5, training=self.training)
        beta = gate_variance(ei, llm_e).unsqueeze(-1)
        h = h0
        for _ in range(self.K):
            h = (1 - beta) * scatter_mean(h[col], row, dim=0, dim_size=N) + beta * h0
        return self.clf(h)

# ── MoE Fusion (T): 3 experts routed by LLM signals ──────────────────────────
class MoEFusionModel(nn.Module):
    def __init__(self, bow_dim, llm_dim, hid, nc, rank=32):
        super().__init__()
        self.Ws = nn.ModuleList([nn.Linear(bow_dim, hid, bias=False) for _ in range(3)])
        self.Wn = nn.ModuleList([nn.Linear(bow_dim, hid, bias=False) for _ in range(3)])
        self.Wq = nn.Linear(llm_dim, rank, bias=False)
        self.Wk = nn.Linear(llm_dim, rank, bias=False)
        self.router = nn.Linear(3, 3, bias=True)
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _experts(self, x, ei, llm_e):
        row, col = ei; N = x.size(0)
        outs = []
        for k in range(3):
            hs = self.Ws[k](x)
            if k == 0:   # mean
                hn = self.Wn[k](scatter_mean(x[col], row, dim=0, dim_size=N))
            elif k == 1: # max
                hn_e = self.Wn[k](x[col])
                hn, _ = scatter_max(hn_e, row, dim=0,
                                    out=torch.zeros(N, hn_e.size(-1), device=x.device))
            else:        # llm-weighted attention
                q = self.Wq(llm_e[row]); kk = self.Wk(llm_e[col])
                w = scatter_softmax((q * kk).sum(-1) / (q.size(-1)**0.5), row, dim=0)
                hn_e = self.Wn[k](x[col]) * w.unsqueeze(-1)
                hn = torch.zeros(N, hn_e.size(-1), device=x.device)
                hn.scatter_add_(0, row.unsqueeze(-1).expand_as(hn_e), hn_e)
            outs.append(hs + hn)
        return torch.stack(outs, dim=-1)                        # [N, hid, 3]
    def forward(self, bow_x, ei, llm_e, **kw):
        sigs = torch.stack([gate_attn_entropy(ei, llm_e),
                            gate_energy_dist(ei, llm_e),
                            gate_spectral(ei, llm_e)], dim=-1)  # [N,3]
        r = F.softmax(self.router(sigs), dim=-1)                # [N,3]
        h = F.relu((self._experts(bow_x, ei, llm_e) * r.unsqueeze(1)).sum(-1))
        h = F.relu(self.l2(h, ei))
        return self.clf(h)

# ── RobustGCN (Zhu 2019) ─────────────────────────────────────────────────────
class RobustGCNConv(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W_mu  = nn.Linear(in_d, out_d, bias=False)
        self.W_sig = nn.Linear(in_d, out_d, bias=False)
    def forward(self, mu, sig2, ei, N):
        row, col = ei
        attn = torch.exp(-sig2[col].mean(-1).clamp(max=10))
        attn = attn / scatter(attn, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
        agg_mu  = scatter(mu[col]   * attn.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        agg_sig = scatter(sig2[col] * (attn**2).unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(self.W_mu(mu + agg_mu)), F.relu(self.W_sig(sig2 + agg_sig))

class RobustGCNModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.sig_init = nn.Linear(in_d, in_d, bias=False)
        self.c1  = RobustGCNConv(in_d, hid)
        self.c2  = RobustGCNConv(hid,  hid)
        self.clf = nn.Linear(hid, nc)
    def _encode(self, x, ei):
        N = x.size(0)
        mu = x; sig2 = F.relu(self.sig_init(x)) + 1e-4
        mu, sig2 = self.c1(mu, sig2, ei, N)
        mu, sig2 = self.c2(mu, sig2, ei, N)
        return mu, sig2
    def forward(self, x, ei, **kw):
        mu, _ = self._encode(x, ei)
        return self.clf(mu)
    def kl_loss(self, x, ei):
        mu, sig2 = self._encode(x, ei)
        return 0.5 * (sig2 + mu**2 - sig2.clamp(min=1e-9).log() - 1).mean()

# ── GNNGuard (Zhang & Zitnik 2020) ───────────────────────────────────────────
class GNNGuardModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.lin1 = nn.Linear(in_d, hid, bias=False)
        self.lin2 = nn.Linear(hid,  hid, bias=False)
        self.clf  = nn.Linear(hid,  nc)
    def _weights(self, h, ei):
        row, col = ei; N = h.size(0)
        hf  = F.normalize(h, dim=-1)
        sim = (hf[row] * hf[col]).sum(-1).clamp(min=0)
        return sim / scatter(sim, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
    def _conv(self, lin, x, ei, w):
        row, col = ei; N = x.size(0)
        agg = scatter(x[col] * w.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(lin(x + agg))
    def forward(self, x, ei, **kw):
        N = x.size(0)
        h = self._conv(self.lin1, x,  ei, self._weights(x,  ei))
        h = self._conv(self.lin2, h,  ei, self._weights(h,  ei))
        return self.clf(h)

# ── LLM Trimmed Mean Fusion ───────────────────────────────────────────────────
class LLMTrimmedModel(nn.Module):
    def __init__(self, bow_dim, hid, nc, trim=0.2):
        super().__init__()
        self.trim = trim
        self.l1 = SAGELayer(bow_dim, hid)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _tagg(self, layer, x, ei, llm_e):
        row, col = ei; N = x.size(0)
        ef  = F.normalize(llm_e, dim=-1)
        cos = (ef[row] * ef[col]).sum(-1)
        deg = scatter(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
        sort_idx  = torch.argsort(row.float() * 1e9 - cos.detach())
        row_s     = row[sort_idx]; col_s = col[sort_idx]
        positions = torch.arange(row_s.size(0), device=x.device).float()
        first     = scatter(positions, row_s, dim=0, dim_size=N, reduce="min").long()
        rank      = (positions - first[row_s]).long()
        n_keep    = (deg * (1 - self.trim)).long().clamp(min=1)
        keep      = rank < n_keep[row_s]
        hs = layer.Ws(x)
        if keep.any():
            agg = scatter_mean(x[col_s[keep]], row_s[keep], dim=0, dim_size=N)
        else:
            agg = torch.zeros_like(x[:, :layer.Wn.in_features])
        return hs + layer.Wn(agg)
    def forward(self, bow_x, ei, llm_e, **kw):
        h = F.relu(self._tagg(self.l1, bow_x, ei, llm_e))
        h = F.relu(self._tagg(self.l2, h,     ei, llm_e))
        return self.clf(h)

# ── LLM Consensus Fusion ──────────────────────────────────────────────────────
class LLMConsensusModel(nn.Module):
    def __init__(self, bow_dim, hid, nc):
        super().__init__()
        self.l1 = SAGELayer(bow_dim, hid)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _cagg(self, layer, x, ei, llm_e, protos):
        row, col = ei; N = x.size(0)
        ef   = F.normalize(llm_e, dim=-1)
        pf   = F.normalize(protos, dim=-1)
        pred = (ef @ pf.T).argmax(-1)
        same = pred[row] == pred[col]
        hs   = layer.Ws(x)
        if same.any():
            agg = scatter_mean(x[col[same]], row[same], dim=0, dim_size=N)
        else:
            agg = scatter_mean(x[col], row, dim=0, dim_size=N)
        return hs + layer.Wn(agg)
    def forward(self, bow_x, ei, llm_e, protos=None, **kw):
        h = F.relu(self._cagg(self.l1, bow_x, ei, llm_e, protos))
        h = F.relu(self._cagg(self.l2, h,     ei, llm_e, protos))
        return self.clf(h)

# ── LLM Multi-Scale Fusion ────────────────────────────────────────────────────
class LLMMultiScaleModel(nn.Module):
    def __init__(self, bow_dim, hid, nc):
        super().__init__()
        self.enc1 = nn.Linear(bow_dim, hid, bias=False)
        self.Wn1  = nn.Linear(bow_dim, hid, bias=False)
        self.enc2 = nn.Linear(bow_dim, hid, bias=False)
        self.Wn2  = nn.Linear(bow_dim, hid, bias=False)
        self.clf  = nn.Linear(hid, nc)
    def forward(self, bow_x, ei, llm_e, **kw):
        row, col = ei; N = bow_x.size(0)
        nbr1 = scatter_mean(bow_x[col], row, dim=0, dim_size=N)
        nbr2 = scatter_mean(nbr1[col],  row, dim=0, dim_size=N)
        h1   = F.relu(self.enc1(bow_x) + self.Wn1(nbr1))
        h2   = F.relu(self.enc2(bow_x) + self.Wn2(nbr2))
        beta = gate_variance(ei, llm_e).unsqueeze(-1)
        return self.clf(beta * h1 + (1 - beta) * h2)

# ══════════════════════════════════════════════════════════════════════════════
# INJECTION + TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def inject(ei, feat, labels, ratio):
    N = feat.size(0); n = int(ratio * N)
    if n == 0: return ei
    fn = F.normalize(feat, dim=-1); src, dst, added = [], [], 0
    for vi in random.sample(range(N), N):
        if added >= n: break
        sim = fn[vi] @ fn.T; sim[vi] = -1; sim[labels == labels[vi]] = -1
        j = sim.argmax().item()
        if sim[j].item() >= 0.5: src.append(vi); dst.append(j); added += 1
    if not src: return ei
    return torch.cat([ei, torch.tensor([src+dst, dst+src], dtype=torch.long)], 1)

def run(model, bow_x, ei, llm_e, labels, trm, vm, tem, seed,
        epochs=150, protos=None, is_rgcn=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(bow_x, ei, llm_e=llm_e, protos=protos)
        loss   = F.cross_entropy(logits[trm], labels[trm])
        if is_rgcn: loss = loss + 5e-4 * model.kl_loss(bow_x, ei)
        loss.backward(); opt.step(); opt.zero_grad()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bow_x, ei, llm_e=llm_e, protos=protos)
            pred = logits.argmax(-1)
            va = (pred[vm]  == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — 11 missing methods
# ══════════════════════════════════════════════════════════════════════════════
SEEDS  = 2
EPOCHS = 150
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
CORA_PT = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(EXISTING) as f:
    results = json.load(f)

for ds_name, bow_name, raw_pt, skey, descs in [
    ("Cora",   "Cora",   CORA_PT, "cora_sweep",   CORA_DESCS),
    ("PubMed", "PubMed", PUB_PT,  "pubmed_sweep",  PUBMED_DESCS),
]:
    print(f"\n{'='*60}\n  {ds_name}\n{'='*60}", flush=True)
    pl   = Planetoid(PL_ROOT, bow_name)[0]
    bx   = F.normalize(pl.x.float(), dim=-1)
    lbl  = pl.y.long()
    nc   = int(lbl.max().item()) + 1
    d    = bx.size(1)
    ei_b = to_undirected(pl.edge_index, num_nodes=pl.num_nodes)
    trm, vm, tem = pl.train_mask, pl.val_mask, pl.test_mask

    print("  Encoding node texts...", flush=True)
    llm_e  = encode_texts(torch.load(raw_pt).raw_texts)
    print("  Encoding class descriptions...", flush=True)
    protos = encode_texts(descs)
    ldim   = llm_e.size(1)
    print(f"  BOW={bx.shape}  LLM={llm_e.shape}  Protos={protos.shape}", flush=True)

    METHODS = [
        # Signal gates
        ("attn_entropy",  lambda: GatedSAGE(d, 128, nc, gate_attn_entropy), False, None),
        ("energy_dist",   lambda: GatedSAGE(d, 128, nc, gate_energy_dist),  False, None),
        ("spectral",      lambda: GatedSAGE(d, 128, nc, gate_spectral),     False, None),
        # Fusion
        ("llm_gat",       lambda: LLMGATModel(d, ldim, 128, nc),            False, None),
        ("llm_appnp",     lambda: LLMAPPNPModel(d, 128, nc),                False, None),
        ("moe_fusion",    lambda: MoEFusionModel(d, ldim, 128, nc),         False, None),
        # Robustness baselines
        ("robust_gcn",    lambda: RobustGCNModel(d, 128, nc),               True,  None),
        ("gnn_guard",     lambda: GNNGuardModel(d, 128, nc),                False, None),
        # LLM subgraph fusion
        ("llm_trimmed",   lambda: LLMTrimmedModel(d, 128, nc),              False, None),
        ("llm_consensus", lambda: LLMConsensusModel(d, 128, nc),            False, protos),
        ("llm_multiscale",lambda: LLMMultiScaleModel(d, 128, nc),           False, None),
    ]

    for key, model_fn, is_rgcn, proto_arg in METHODS:
        if key in results[skey]["results"]:
            print(f"  SKIP {key} (already done)", flush=True)
            continue
        print(f"\n  ── {key} ──", flush=True)
        res = []
        for r in RATIOS:
            ei = inject(ei_b, llm_e, lbl, r)
            accs = [run(model_fn(), bx, ei, llm_e, lbl, trm, vm, tem,
                        s, EPOCHS, protos=proto_arg, is_rgcn=is_rgcn)
                    for s in range(SEEDS)]
            acc = float(np.mean(accs))
            res.append(round(acc, 4))
            print(f"  {key:<18} r={r:.0%}  {acc*100:.2f}%", flush=True)
        results[skey]["results"][key] = res
        # Save after each method so progress is not lost
        with open(EXISTING, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {key}", flush=True)

print("\n=== ALL DONE ===")
