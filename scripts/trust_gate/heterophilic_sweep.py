"""
Heterophilic graph sweep: Wisconsin, Cornell, Texas (WebKB).

These graphs have low homophily — neighbours are usually different-class.
The LLM uncertainty gate should permanently stay high (beta->1, trust self),
effectively turning the model into an MLP, which is actually correct here.

This tests whether the gate adapts correctly to graph homophily structure,
and whether our robustness advantages transfer beyond citation networks.

Methods: none, bl_gcn, bl_gat, variance, spectral, energy_dist,
         ensemble, llm_appnp, gnn_guard
Injection: semantic (cosine>=0.5, wrong-class)
Datasets: Wisconsin, Cornell, Texas
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_softmax

RESULTS_PATH = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
DATA_ROOT    = "/tmp/webkb"
SEEDS   = [0, 1]
EPOCHS  = 200          # more epochs: these graphs are tiny
HID     = 64           # smaller hidden dim for tiny graphs
LR      = 5e-3
WD      = 5e-4
RATIOS  = [0, 0.10, 0.20, 0.30, 0.50]
SAMPLE  = 500

# ── LLM class prototypes (WebKB: student/project/course/staff/faculty) ────────
WEBKB_DESCS = [
    "student personal academic homepage university",
    "course class syllabus lecture notes university",
    "faculty professor academic staff research university",
    "project research group laboratory university",
    "staff administrative university department",
]

def encode_texts(texts, batch=64):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("multi-qa-distilbert-cos-v1"); m.eval()
    out = []
    for i in range(0, len(texts), batch):
        with torch.no_grad():
            out.append(m.encode(texts[i:i+batch], convert_to_tensor=True,
                                show_progress_bar=False).cpu())
    return torch.cat(out, 0)

# ══════════════════════════════════════════════════════════════════════════════
# GATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def gate_variance(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var = scatter_mean((ef[col]-mu[row]).norm(dim=-1)**2, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)

def gate_energy_dist(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    diff = (ef[row]-ef[col]).norm(dim=-1)
    mean_dv = scatter_mean(diff, row, dim=0, dim_size=N)
    mu_n = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var_n = scatter_mean((ef[col]-mu_n[row]).norm(dim=-1), row, dim=0, dim_size=N)
    return torch.sigmoid((2*mean_dv - var_n).clamp(min=0)*2)

def gate_spectral(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos = (ef[row]*ef[col]).sum(-1)
    max_cos = scatter(cos, row, dim=0, dim_size=N, reduce="max")
    mu_cos = scatter_mean(cos, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos-mu_cos[row])**2, row, dim=0, dim_size=N)
    return torch.sigmoid(((1-max_cos)+var_cos.sqrt())*3)

def gate_ensemble(ei, e):
    return (gate_variance(ei, e)+gate_energy_dist(ei, e)+gate_spectral(ei, e))/3.0

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))

class SAGEModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1, self.l2 = SAGELayer(in_d, hid), SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei, **kw):
        return self.clf(F.relu(self.l2(F.relu(self.l1(x, ei)), ei)))

class GCNModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1 = nn.Linear(in_d, hid, bias=False)
        self.l2 = nn.Linear(hid,  hid, bias=False)
        self.clf = nn.Linear(hid, nc)
    def _gcn(self, lin, x, ei):
        row, col = ei; N = x.size(0)
        deg = scatter(torch.ones(ei.size(1), device=x.device), row, dim=0, dim_size=N).clamp(min=1)
        norm = (deg[row]*deg[col]).rsqrt()
        agg = scatter(x[col]*norm.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(lin(agg))
    def forward(self, x, ei, **kw):
        return self.clf(self._gcn(self.l2, self._gcn(self.l1, x, ei), ei))

class GATModel(nn.Module):
    def __init__(self, in_d, hid, nc, heads=4):
        super().__init__()
        assert hid % heads == 0
        self.h, self.dh = heads, hid // heads
        self.Wq = nn.Linear(in_d, hid, bias=False)
        self.Wk = nn.Linear(in_d, hid, bias=False)
        self.Wv = nn.Linear(in_d, hid, bias=False)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _gat(self, x, ei):
        row, col = ei; N = x.size(0)
        q = self.Wq(x).view(N, self.h, self.dh)
        k = self.Wk(x).view(N, self.h, self.dh)
        v = self.Wv(x).view(N, self.h, self.dh)
        score = (q[row]*k[col]).sum(-1)/self.dh**0.5
        w = torch.stack([scatter_softmax(score[:,hh], row, dim=0)
                         for hh in range(self.h)], -1)
        out = torch.zeros(N, self.h, self.dh, device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand(-1,self.h,self.dh),
                         v[col]*w.unsqueeze(-1))
        return out.view(N, self.h*self.dh)
    def forward(self, x, ei, **kw):
        return self.clf(F.relu(self.l2(F.relu(self._gat(x, ei)), ei)))

class GatedSAGE(nn.Module):
    def __init__(self, in_d, hid, nc, gate_fn):
        super().__init__()
        self.gfn = gate_fn
        self.l1, self.l2 = SAGELayer(in_d, hid), SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _apply(self, layer, x, ei, b):
        row, col = ei; N = x.size(0); b = b.unsqueeze(-1)
        return b*layer.Ws(x) + (1-b)*layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
    def forward(self, x, ei, llm_e, **kw):
        b = self.gfn(ei, llm_e)
        h = F.relu(self._apply(self.l1, x,  ei, b))
        b = self.gfn(ei, llm_e)
        h = F.relu(self._apply(self.l2, h,  ei, b))
        return self.clf(h)

class LLMAPPNPModel(nn.Module):
    def __init__(self, in_d, hid, nc, K=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(hid, hid))
        self.clf = nn.Linear(hid, nc); self.K = K
    def forward(self, x, ei, llm_e, **kw):
        row, col = ei; N = x.size(0)
        h0   = F.dropout(self.enc(x), 0.5, self.training)
        beta = gate_variance(ei, llm_e).unsqueeze(-1)
        h    = h0
        for _ in range(self.K):
            h = (1-beta)*scatter_mean(h[col], row, dim=0, dim_size=N) + beta*h0
        return self.clf(h)

class GNNGuardModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1 = nn.Linear(in_d, hid, bias=False)
        self.l2 = nn.Linear(hid,  hid, bias=False)
        self.clf = nn.Linear(hid, nc)
    def _w(self, h, ei):
        row, col = ei; N = h.size(0)
        s = (F.normalize(h,dim=-1)[row]*F.normalize(h,dim=-1)[col]).sum(-1).clamp(min=0)
        return s/scatter(s, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
    def _conv(self, lin, x, ei):
        row, col = ei; N = x.size(0)
        return F.relu(lin(x+scatter(x[col]*self._w(x,ei).unsqueeze(-1),
                                    row, dim=0, dim_size=N)))
    def forward(self, x, ei, **kw):
        return self.clf(self._conv(self.l2, self._conv(self.l1, x, ei), ei))

# ══════════════════════════════════════════════════════════════════════════════
# INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_semantic(ei, feat, labels, N, ratio):
    n_inject = int(ratio * N)
    if n_inject == 0: return ei
    fn = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), min(N, n_inject*5)):
        if added >= n_inject: break
        diff = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff.numel() == 0: continue
        cands = diff[torch.randperm(diff.numel())[:SAMPLE]]
        sims = (fn[vi]*fn[cands]).sum(-1)
        best = sims.argmax()
        if sims[best].item() >= 0.5:
            src.append(vi); dst.append(cands[best].item()); added += 1
    if not src: return ei
    new = torch.tensor([src+dst, dst+src], dtype=torch.long)
    return torch.cat([ei, new], 1)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════
def train_eval(model, x, ei, llm_e, labels, tr, va, te):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val, best_test = 0, 0
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, ei, llm_e=llm_e)[tr], labels[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei, llm_e=llm_e).argmax(-1)
        va_acc = (pred[va]==labels[va]).float().mean().item()
        te_acc = (pred[te]==labels[te]).float().mean().item()
        if va_acc > best_val:
            best_val, best_test = va_acc, te_acc
    return best_test

def run_method(name, x, ei, llm_e, labels, tr, va, te, in_d, nc):
    if   name == "none":       m = SAGEModel(in_d, HID, nc)
    elif name == "bl_gcn":     m = GCNModel(in_d, HID, nc)
    elif name == "bl_gat":     m = GATModel(in_d, HID, nc)
    elif name == "variance":   m = GatedSAGE(in_d, HID, nc, gate_variance)
    elif name == "spectral":   m = GatedSAGE(in_d, HID, nc, gate_spectral)
    elif name == "energy_dist":m = GatedSAGE(in_d, HID, nc, gate_energy_dist)
    elif name == "ensemble":   m = GatedSAGE(in_d, HID, nc, gate_ensemble)
    elif name == "llm_appnp":  m = LLMAPPNPModel(in_d, HID, nc)
    elif name == "gnn_guard":  m = GNNGuardModel(in_d, HID, nc)
    else: raise ValueError(name)
    return train_eval(m, x, ei, llm_e, labels, tr, va, te)

METHODS = ["none","bl_gcn","bl_gat","variance","spectral",
           "energy_dist","ensemble","llm_appnp","gnn_guard"]

def save_result(ds_key, method, accs):
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    if "heterophilic" not in data:
        data["heterophilic"] = {}
    if ds_key not in data["heterophilic"]:
        data["heterophilic"][ds_key] = {}
    data["heterophilic"][ds_key][method] = accs
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def already_done(ds_key, method):
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return method in data.get("heterophilic", {}).get(ds_key, {})

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("Encoding WebKB class prototypes …", flush=True)
class_emb = encode_texts(WEBKB_DESCS)   # [5, 768]

for ds_name in ["Wisconsin", "Cornell", "Texas"]:
    ds_key = ds_name.lower()
    print(f"\n{'='*55}\n  {ds_name}\n{'='*55}", flush=True)

    ds   = WebKB(DATA_ROOT, ds_name)
    data = ds[0]
    # WebKB has 10 fixed splits — use split 0
    tr = data.train_mask[:, 0]
    va = data.val_mask[:, 0]
    te = data.test_mask[:, 0]
    x      = data.x.float()
    labels = data.y
    ei     = to_undirected(data.edge_index)
    N      = x.size(0)
    in_d   = x.size(1)
    nc     = int(labels.max().item()) + 1
    llm_e  = class_emb[labels]            # [N, 768]

    # Homophily info
    row, col = ei
    homo = (labels[row]==labels[col]).float().mean().item()
    print(f"  N={N}, E={ei.size(1)//2}, classes={nc}, homophily={homo:.3f}", flush=True)

    for method in METHODS:
        if already_done(ds_key, method):
            print(f"  [skip] {method}"); continue
        print(f"\n  ── {method} ──", flush=True)
        accs = []
        for ratio in RATIOS:
            seed_accs = []
            for seed in SEEDS:
                torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                ei_atk = inject_semantic(ei, x, labels, N, ratio)
                acc = run_method(method, x, ei_atk, llm_e, labels, tr, va, te, in_d, nc)
                seed_accs.append(acc)
            mean_acc = float(np.mean(seed_accs))
            accs.append(mean_acc)
            print(f"  {method:<14} r={int(ratio*100):2d}%  {mean_acc*100:.2f}%", flush=True)
        save_result(ds_key, method, accs)
        print(f"  Saved {ds_key}/{method}  drop={(accs[-1]-accs[0])*100:+.1f}pp", flush=True)

print("\n=== HETEROPHILIC SWEEP COMPLETE ===")
