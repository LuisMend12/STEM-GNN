"""
OGBN-Arxiv sweep — key methods only.
Scale adaptations:
  - OGB 128-dim word2vec features used as both BOW input and LLM proxy
    (computed from paper title+abstract, a valid text-derived signal)
  - Sampled injection: cosine search over 2000 random candidates per node
  - 1 seed x 100 epochs to stay within SIGXCPU budget
  - OGB predefined train/valid/test splits
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_softmax, scatter_max

OGB_ROOT = "/home/lam23005/STEM-GNN/data/ogb"
RESULTS  = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
SEEDS    = 1
EPOCHS   = 100
RATIOS   = [0.0, 0.1, 0.2, 0.3, 0.5]
SAMPLE   = 2000   # candidate nodes per injection target

os.makedirs(OGB_ROOT, exist_ok=True)

# ── Load OGBN-Arxiv ───────────────────────────────────────────────────────────
print("Loading OGBN-Arxiv...", flush=True)
from ogb.nodeproppred import PygNodePropPredDataset
ds   = PygNodePropPredDataset("ogbn-arxiv", root=OGB_ROOT)
data = ds[0]
split = ds.get_idx_split()

# OGB features: 128-dim averaged word2vec over title+abstract
bx   = F.normalize(data.x.float(), dim=-1)
lbl  = data.y.squeeze(-1).long()
nc   = ds.num_classes          # 40 arXiv CS subfields
d    = bx.size(1)              # 128
N    = data.num_nodes

# Masks from OGB splits
trm = torch.zeros(N, dtype=torch.bool); trm[split["train"]] = True
vm  = torch.zeros(N, dtype=torch.bool); vm[split["valid"]]  = True
tem = torch.zeros(N, dtype=torch.bool); tem[split["test"]]  = True

# Use word2vec features as the "LLM" proxy (text-derived, external to GNN)
llm_e = bx.clone()

# Undirected edges
ei_b = to_undirected(data.edge_index, num_nodes=N)

print(f"N={N}  edges={ei_b.size(1)}  dim={d}  classes={nc}", flush=True)
print(f"train={trm.sum()}  val={vm.sum()}  test={tem.sum()}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLED INJECTION (can't do 169k × 169k cosine)
# ══════════════════════════════════════════════════════════════════════════════
def inject_sampled(ei, feat, labels, ratio, sample=SAMPLE):
    n_inject = int(ratio * N)
    if n_inject == 0: return ei
    fn   = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    targets = random.sample(range(N), min(N, n_inject * 5))
    for vi in targets:
        if added >= n_inject: break
        # Sample candidates from different classes
        diff_cls = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff_cls.numel() == 0: continue
        cands = diff_cls[torch.randperm(diff_cls.numel())[:sample]]
        sims  = (fn[vi] * fn[cands]).sum(-1)
        best  = sims.argmax().item()
        if sims[best].item() >= 0.5:
            src.append(vi); dst.append(cands[best].item()); added += 1
    if not src: return ei
    new = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return torch.cat([ei, new], dim=1)

# ══════════════════════════════════════════════════════════════════════════════
# MODELS (same as all_missing_sweep.py)
# ══════════════════════════════════════════════════════════════════════════════
class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=x.size(0)))

def gate_variance(ei, e):
    row, col = ei; N = e.size(0)
    ef  = F.normalize(e, dim=-1)
    mu  = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)

def gate_attn_entropy(ei, e):
    row, col = ei; N = e.size(0)
    ef    = F.normalize(e, dim=-1)
    logit = (ef[row] * ef[col]).sum(-1) / (e.size(1)**0.5)
    attn  = scatter_softmax(logit, row, dim=0)
    eps   = 1e-9
    ent   = scatter(-(attn*(attn+eps).log()), row, dim=0, dim_size=N, reduce="sum")
    deg   = scatter(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
    return torch.sigmoid(ent / deg.log().clamp(min=eps) * 5)

def gate_energy_dist(ei, e):
    row, col = ei; N = e.size(0)
    ef      = F.normalize(e, dim=-1)
    diff_vu = (ef[row] - ef[col]).norm(dim=-1)
    mean_vu = scatter_mean(diff_vu, row, dim=0, dim_size=N)
    mu_nbr  = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var_nbr = scatter_mean((ef[col] - mu_nbr[row]).norm(dim=-1), row, dim=0, dim_size=N)
    return torch.sigmoid((2*mean_vu - var_nbr).clamp(min=0) * 2)

def gate_spectral(ei, e):
    row, col = ei; N = e.size(0)
    ef      = F.normalize(e, dim=-1)
    cos     = (ef[row]*ef[col]).sum(-1)
    max_cos = scatter(cos, row, dim=0, dim_size=N, reduce="max")
    mu_cos  = scatter_mean(cos, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos - mu_cos[row])**2, row, dim=0, dim_size=N)
    return torch.sigmoid(((1 - max_cos) + var_cos.sqrt()) * 3)

class PlainSAGE(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1 = SAGELayer(in_d, hid); self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei, **kw):
        h = F.relu(self.l1(x, ei)); h = F.relu(self.l2(h, ei))
        return self.clf(h)

class GatedSAGE(nn.Module):
    def __init__(self, in_d, hid, nc, gate_fn):
        super().__init__()
        self.gate_fn = gate_fn
        self.l1 = SAGELayer(in_d, hid); self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _apply(self, layer, x, ei, b):
        row, col = ei; N = x.size(0); b = b.unsqueeze(-1)
        return b*layer.Ws(x) + (1-b)*layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
    def forward(self, x, ei, llm_e, **kw):
        b = self.gate_fn(ei, llm_e)
        h = F.relu(self._apply(self.l1, x, ei, b))
        b = self.gate_fn(ei, llm_e)
        h = F.relu(self._apply(self.l2, h, ei, b))
        return self.clf(h)

class MLPGate(nn.Module):
    def __init__(self, in_d, hid, nc, llm_d):
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(llm_d*2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.l1 = SAGELayer(in_d, hid); self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _gate(self, ei, e):
        row, col = ei; N = e.size(0)
        mu = scatter_mean(e[col], row, dim=0, dim_size=N)
        return torch.sigmoid(self.gate_mlp(torch.cat([e, mu], dim=-1)).squeeze(-1))
    def _apply(self, layer, x, ei, b):
        row, col = ei; N = x.size(0); b = b.unsqueeze(-1)
        return b*layer.Ws(x) + (1-b)*layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
    def forward(self, x, ei, llm_e, **kw):
        b = self._gate(ei, llm_e)
        h = F.relu(self._apply(self.l1, x, ei, b))
        b = self._gate(ei, llm_e)
        h = F.relu(self._apply(self.l2, h, ei, b))
        return self.clf(h)

class LLMAPPNPModel(nn.Module):
    def __init__(self, in_d, hid, nc, K=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hid, hid))
        self.clf = nn.Linear(hid, nc); self.K = K
    def forward(self, x, ei, llm_e, **kw):
        row, col = ei; N = x.size(0)
        h0   = F.dropout(self.enc(x), p=0.5, training=self.training)
        beta = gate_variance(ei, llm_e).unsqueeze(-1)
        h = h0
        for _ in range(self.K):
            h = (1-beta)*scatter_mean(h[col], row, dim=0, dim_size=N) + beta*h0
        return self.clf(h)

class MoEFusionModel(nn.Module):
    def __init__(self, in_d, llm_d, hid, nc, rank=32):
        super().__init__()
        self.Ws = nn.ModuleList([nn.Linear(in_d, hid, bias=False) for _ in range(3)])
        self.Wn = nn.ModuleList([nn.Linear(in_d, hid, bias=False) for _ in range(3)])
        self.Wq = nn.Linear(llm_d, rank, bias=False); self.Wk = nn.Linear(llm_d, rank, bias=False)
        self.router = nn.Linear(3, 3, bias=True)
        self.l2 = SAGELayer(hid, hid); self.clf = nn.Linear(hid, nc)
    def _experts(self, x, ei, llm_e):
        row, col = ei; N = x.size(0); outs = []
        for k in range(3):
            hs = self.Ws[k](x)
            if k == 0:
                hn = self.Wn[k](scatter_mean(x[col], row, dim=0, dim_size=N))
            elif k == 1:
                hn_e = self.Wn[k](x[col])
                hn, _ = scatter_max(hn_e, row, dim=0, out=torch.zeros(N, hn_e.size(-1), device=x.device))
            else:
                q = self.Wq(llm_e[row]); kk = self.Wk(llm_e[col])
                w = scatter_softmax((q*kk).sum(-1)/(q.size(-1)**0.5), row, dim=0)
                hn_e = self.Wn[k](x[col]) * w.unsqueeze(-1)
                hn = torch.zeros(N, hn_e.size(-1), device=x.device)
                hn.scatter_add_(0, row.unsqueeze(-1).expand_as(hn_e), hn_e)
            outs.append(hs + hn)
        return torch.stack(outs, dim=-1)
    def forward(self, x, ei, llm_e, **kw):
        sigs = torch.stack([gate_attn_entropy(ei, llm_e),
                            gate_energy_dist(ei, llm_e),
                            gate_spectral(ei, llm_e)], dim=-1)
        r = F.softmax(self.router(sigs), dim=-1)
        h = F.relu((self._experts(x, ei, llm_e) * r.unsqueeze(1)).sum(-1))
        h = F.relu(self.l2(h, ei)); return self.clf(h)

class RobustGCNConv(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W_mu = nn.Linear(in_d, out_d, bias=False)
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
        self.c1 = RobustGCNConv(in_d, hid); self.c2 = RobustGCNConv(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _encode(self, x, ei):
        N = x.size(0); mu = x; sig2 = F.relu(self.sig_init(x)) + 1e-4
        mu, sig2 = self.c1(mu, sig2, ei, N); mu, sig2 = self.c2(mu, sig2, ei, N)
        return mu, sig2
    def forward(self, x, ei, **kw):
        mu, _ = self._encode(x, ei); return self.clf(mu)
    def kl_loss(self, x, ei):
        mu, sig2 = self._encode(x, ei)
        return 0.5 * (sig2 + mu**2 - sig2.clamp(min=1e-9).log() - 1).mean()

class GNNGuardModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.lin1 = nn.Linear(in_d, hid, bias=False); self.lin2 = nn.Linear(hid, hid, bias=False)
        self.clf = nn.Linear(hid, nc)
    def _weights(self, h, ei):
        row, col = ei; N = h.size(0)
        hf = F.normalize(h, dim=-1)
        sim = (hf[row]*hf[col]).sum(-1).clamp(min=0)
        return sim / scatter(sim, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
    def _conv(self, lin, x, ei, w):
        row, col = ei; N = x.size(0)
        agg = scatter(x[col]*w.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(lin(x + agg))
    def forward(self, x, ei, **kw):
        h = self._conv(self.lin1, x, ei, self._weights(x, ei))
        h = self._conv(self.lin2, h, ei, self._weights(h, ei))
        return self.clf(h)

# Standard baselines
class GCNModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1 = nn.Linear(in_d, hid, bias=False); self.l2 = nn.Linear(hid, hid, bias=False)
        self.clf = nn.Linear(hid, nc)
    def _conv(self, lin, x, ei):
        row, col = ei; N = x.size(0)
        deg = scatter(torch.ones(col.size(0), device=x.device), row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
        norm = (deg[row] * deg[col]).sqrt()
        agg = scatter(x[col] / norm.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(lin(agg))
    def forward(self, x, ei, **kw):
        h = self._conv(self.l1, x, ei); h = self._conv(self.l2, h, ei); return self.clf(h)

class GATModel(nn.Module):
    def __init__(self, in_d, hid, nc, heads=4):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.c1 = GATConv(in_d, hid//heads, heads=heads, dropout=0.6)
        self.c2 = GATConv(hid, hid, heads=1, concat=False, dropout=0.6)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei, **kw):
        h = F.elu(self.c1(x, ei)); h = F.elu(self.c2(h, ei)); return self.clf(h)

class APPNPModel(nn.Module):
    def __init__(self, in_d, hid, nc, K=10, alpha=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hid, hid))
        self.clf = nn.Linear(hid, nc); self.K = K; self.alpha = alpha
    def forward(self, x, ei, **kw):
        row, col = ei; N = x.size(0)
        h0 = F.dropout(self.enc(x), p=0.5, training=self.training)
        h = h0
        for _ in range(self.K):
            h = (1-self.alpha)*scatter_mean(h[col], row, dim=0, dim_size=N) + self.alpha*h0
        return self.clf(h)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def run(model, bx, ei, llm_e, lbl, trm, vm, tem, seed, epochs=EPOCHS, is_rgcn=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(bx, ei, llm_e=llm_e)
        loss = F.cross_entropy(logits[trm], lbl[trm])
        if is_rgcn: loss = loss + 5e-4 * model.kl_loss(bx, ei)
        loss.backward(); opt.step(); opt.zero_grad()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bx, ei, llm_e=llm_e)
            pred = logits.argmax(-1)
            va = (pred[vm]  == lbl[vm]).float().mean().item()
            ta = (pred[tem] == lbl[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt

# ══════════════════════════════════════════════════════════════════════════════
# METHODS
# ══════════════════════════════════════════════════════════════════════════════
METHODS = [
    ("bl_gcn",        lambda: GCNModel(d, 256, nc),                          False),
    ("bl_gat",        lambda: GATModel(d, 256, nc),                          False),
    ("bl_appnp",      lambda: APPNPModel(d, 256, nc),                        False),
    ("none",          lambda: PlainSAGE(d, 256, nc),                         False),
    ("variance",      lambda: GatedSAGE(d, 256, nc, gate_variance),          False),
    ("mlp",           lambda: MLPGate(d, 256, nc, d),                        False),
    ("spectral",      lambda: GatedSAGE(d, 256, nc, gate_spectral),          False),
    ("energy_dist",   lambda: GatedSAGE(d, 256, nc, gate_energy_dist),       False),
    ("attn_entropy",  lambda: GatedSAGE(d, 256, nc, gate_attn_entropy),      False),
    ("llm_appnp",     lambda: LLMAPPNPModel(d, 256, nc),                     False),
    ("moe_fusion",    lambda: MoEFusionModel(d, d, 256, nc),                 False),
    ("robust_gcn",    lambda: RobustGCNModel(d, 256, nc),                    True),
    ("gnn_guard",     lambda: GNNGuardModel(d, 256, nc),                     False),
]

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
with open(RESULTS) as f:
    results = json.load(f)

if "arxiv_sweep" not in results:
    results["arxiv_sweep"] = {"dataset": "ogbn-arxiv", "results": {}}

skey = "arxiv_sweep"

print(f"\n{'='*60}\n  OGBN-Arxiv\n{'='*60}", flush=True)

for key, model_fn, is_rgcn in METHODS:
    if key in results[skey]["results"]:
        print(f"  SKIP {key}", flush=True)
        continue
    print(f"\n  ── {key} ──", flush=True)
    res = []
    for r in RATIOS:
        ei = inject_sampled(ei_b, llm_e, lbl, r)
        accs = [run(model_fn(), bx, ei, llm_e, lbl, trm, vm, tem, s, is_rgcn=is_rgcn)
                for s in range(SEEDS)]
        acc = float(np.mean(accs))
        res.append(round(acc, 4))
        print(f"  {key:<18} r={r:.0%}  {acc*100:.2f}%", flush=True)
    results[skey]["results"][key] = res
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {key}", flush=True)

print("\n=== OGBN-Arxiv DONE ===", flush=True)
