"""
Three new experiments:
  1. ensemble gate  — mean of variance + energy_dist + spectral (D+P+Q)
  2. random injection   — wrong-class edges with NO cosine filter
  3. adaptive injection — BOW cosine>=0.5 AND maximise LLM cosine to v's
                          neighbourhood centroid (hardest for uncertainty gates)

Methods tested under each injection type:
  none, bl_gcn, bl_gat, variance, spectral, energy_dist, ensemble, llm_appnp, gnn_guard

Results saved to results/llm_gates_results.json under key "ablation".
Skips (dataset, injection, method) triples already present.
"""
import os, json, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_softmax

SEEDS   = [0, 1]
EPOCHS  = 150
HID     = 256
LR      = 1e-3
WD      = 5e-4
RATIOS  = [0, 0.10, 0.20, 0.30, 0.50]
SAMPLE  = 2000          # candidates to scan per node for adaptive injection

RESULTS_PATH = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
DATA_ROOT    = "/tmp/planetoid"

# ── LLM encoder ───────────────────────────────────────────────────────────────
def encode_texts(texts, batch=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("multi-qa-distilbert-cos-v1"); m.eval()
    out = []
    for i in range(0, len(texts), batch):
        with torch.no_grad():
            out.append(m.encode(texts[i:i+batch], convert_to_tensor=True,
                                show_progress_bar=False).cpu())
    return torch.cat(out, 0)

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
# GATE SIGNAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def gate_variance(ei, e):
    row, col = ei; N = e.size(0)
    ef  = F.normalize(e, dim=-1)
    mu  = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)

def gate_energy_dist(ei, e):
    row, col = ei; N = e.size(0)
    ef      = F.normalize(e, dim=-1)
    diff    = (ef[row] - ef[col]).norm(dim=-1)
    mean_dv = scatter_mean(diff, row, dim=0, dim_size=N)
    mu_n    = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var_n   = scatter_mean((ef[col] - mu_n[row]).norm(dim=-1), row, dim=0, dim_size=N)
    return torch.sigmoid((2 * mean_dv - var_n).clamp(min=0) * 2)

def gate_spectral(ei, e):
    row, col = ei; N = e.size(0)
    ef      = F.normalize(e, dim=-1)
    cos     = (ef[row] * ef[col]).sum(-1)
    max_cos = scatter(cos, row, dim=0, dim_size=N, reduce="max")
    mu_cos  = scatter_mean(cos, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos - mu_cos[row])**2, row, dim=0, dim_size=N)
    return torch.sigmoid(((1 - max_cos) + var_cos.sqrt()) * 3)

def gate_ensemble(ei, e):
    return (gate_variance(ei, e) + gate_energy_dist(ei, e) + gate_spectral(ei, e)) / 3.0

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
        h = F.relu(self.l1(x, ei))
        h = F.relu(self.l2(h, ei))
        return self.clf(h)

class GCNModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1  = nn.Linear(in_d, hid, bias=False)
        self.l2  = nn.Linear(hid,  hid, bias=False)
        self.clf = nn.Linear(hid, nc)
    def _gcn(self, lin, x, ei):
        row, col = ei; N = x.size(0)
        deg  = scatter(torch.ones(ei.size(1), device=x.device),
                       row, dim=0, dim_size=N).clamp(min=1)
        norm = (deg[row] * deg[col]).rsqrt()
        agg  = scatter(x[col] * norm.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
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
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _gat(self, x, ei):
        row, col = ei; N = x.size(0)
        q = self.Wq(x).view(N, self.h, self.dh)
        k = self.Wk(x).view(N, self.h, self.dh)
        v = self.Wv(x).view(N, self.h, self.dh)
        score = (q[row] * k[col]).sum(-1) / self.dh**0.5   # [E, h]
        attn  = torch.zeros(N, self.h, device=x.device)
        for hh in range(self.h):
            attn[:, hh] = scatter_softmax(score[:, hh], row, dim=0).mean()
        # per-edge softmax
        w = torch.stack([scatter_softmax(score[:, hh], row, dim=0) for hh in range(self.h)], -1)  # [E,h]
        out = torch.zeros(N, self.h, self.dh, device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand(-1, self.h, self.dh),
                         v[col] * w.unsqueeze(-1))
        return out.view(N, self.h * self.dh)
    def forward(self, x, ei, **kw):
        h = F.relu(self._gat(x, ei))
        h = F.relu(self.l2(h, ei))
        return self.clf(h)

class GatedSAGE(nn.Module):
    def __init__(self, in_d, hid, nc, gate_fn):
        super().__init__()
        self.gfn = gate_fn
        self.l1, self.l2 = SAGELayer(in_d, hid), SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _apply(self, layer, x, ei, b):
        row, col = ei; N = x.size(0); b = b.unsqueeze(-1)
        return b * layer.Ws(x) + (1-b) * layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
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
        self.clf = nn.Linear(hid, nc)
        self.K   = K
    def forward(self, x, ei, llm_e, **kw):
        row, col = ei; N = x.size(0)
        h0   = F.dropout(self.enc(x), 0.5, self.training)
        beta = gate_variance(ei, llm_e).unsqueeze(-1)
        h    = h0
        for _ in range(self.K):
            h = (1-beta) * scatter_mean(h[col], row, dim=0, dim_size=N) + beta * h0
        return self.clf(h)

class GNNGuardModel(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1  = nn.Linear(in_d, hid, bias=False)
        self.l2  = nn.Linear(hid,  hid, bias=False)
        self.clf = nn.Linear(hid, nc)
    def _w(self, h, ei):
        row, col = ei; N = h.size(0)
        s = (F.normalize(h,dim=-1)[row] * F.normalize(h,dim=-1)[col]).sum(-1).clamp(min=0)
        return s / scatter(s, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
    def _conv(self, lin, x, ei):
        row, col = ei; N = x.size(0)
        w = self._w(x, ei)
        return F.relu(lin(x + scatter(x[col]*w.unsqueeze(-1), row, dim=0, dim_size=N)))
    def forward(self, x, ei, **kw):
        return self.clf(self._conv(self.l2, self._conv(self.l1, x, ei), ei))

# ══════════════════════════════════════════════════════════════════════════════
# INJECTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def inject_semantic(ei, feat, labels, N, ratio, sample=SAMPLE):
    """Existing attack: wrong-class edges with BOW cosine >= 0.5."""
    n_inject = int(ratio * N)
    if n_inject == 0: return ei
    fn = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), min(N, n_inject * 5)):
        if added >= n_inject: break
        diff = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff.numel() == 0: continue
        cands = diff[torch.randperm(diff.numel())[:sample]]
        sims  = (fn[vi] * fn[cands]).sum(-1)
        best  = sims.argmax()
        if sims[best].item() >= 0.5:
            src.append(vi); dst.append(cands[best].item()); added += 1
    if not src: return ei
    new = torch.tensor([src+dst, dst+src], dtype=torch.long)
    return torch.cat([ei, new], 1)

def inject_random(ei, labels, N, ratio):
    """Baseline: random wrong-class edges (no cosine filter)."""
    n_inject = int(ratio * N)
    if n_inject == 0: return ei
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), min(N, n_inject * 5)):
        if added >= n_inject: break
        diff = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff.numel() == 0: continue
        uj = diff[torch.randint(diff.numel(), (1,)).item()].item()
        src.append(vi); dst.append(uj); added += 1
    if not src: return ei
    new = torch.tensor([src+dst, dst+src], dtype=torch.long)
    return torch.cat([ei, new], 1)

def inject_adaptive(ei, feat, llm_emb, labels, N, ratio, sample=SAMPLE):
    """
    Adaptive attack against uncertainty gates.
    Constraint: BOW cosine >= 0.5 (same as semantic).
    Strategy: among valid candidates, pick the one whose LLM embedding is
    closest to v's current neighbourhood centroid in LLM space.
    This minimises the variance/entropy increase that uncertainty gates detect.
    """
    n_inject = int(ratio * N)
    if n_inject == 0: return ei
    fn = F.normalize(feat,    dim=-1)
    ln = F.normalize(llm_emb, dim=-1)
    row, col = ei
    nbr_cent = scatter_mean(ln[col], row, dim=0, dim_size=N)   # [N, d]
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), min(N, n_inject * 5)):
        if added >= n_inject: break
        diff = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff.numel() == 0: continue
        cands = diff[torch.randperm(diff.numel())[:sample]]
        # BOW gate first (same constraint as semantic attack)
        bow_ok = (fn[vi] * fn[cands]).sum(-1) >= 0.5
        valid  = cands[bow_ok]
        if valid.numel() == 0: continue
        # Among valid: pick the one closest to v's LLM neighbourhood centroid
        cent = nbr_cent[vi]
        if cent.norm() < 1e-6:          # isolated node: fall back to max BOW sim
            sims = (fn[vi] * fn[valid]).sum(-1)
            best_idx = sims.argmax().item()
        else:
            llm_sim = (ln[valid] * cent.unsqueeze(0)).sum(-1)
            best_idx = llm_sim.argmax().item()
        src.append(vi); dst.append(valid[best_idx].item()); added += 1
    if not src: return ei
    new = torch.tensor([src+dst, dst+src], dtype=torch.long)
    return torch.cat([ei, new], 1)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_eval(model, x, ei, llm_e, labels, train_m, val_m, test_m,
               epochs=EPOCHS, lr=LR, wd=WD, kl=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_test = 0, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out  = model(x, ei, llm_e=llm_e)
        loss = F.cross_entropy(out[train_m], labels[train_m])
        if kl:
            loss = loss + 1e-3 * model.kl_loss(x, ei)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei, llm_e=llm_e).argmax(-1)
        val_acc  = (pred[val_m]  == labels[val_m]).float().mean().item()
        test_acc = (pred[test_m] == labels[test_m]).float().mean().item()
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc
    return best_test

def run_method(name, x, ei, llm_e, labels, train_m, val_m, test_m,
               in_d, llm_d, nc):
    """Instantiate and train one method; return test accuracy."""
    if name == "none":
        m = SAGEModel(in_d, HID, nc)
    elif name == "bl_gcn":
        m = GCNModel(in_d, HID, nc)
    elif name == "bl_gat":
        m = GATModel(in_d, HID, nc)
    elif name == "variance":
        m = GatedSAGE(in_d, HID, nc, gate_fn=gate_variance)
    elif name == "spectral":
        m = GatedSAGE(in_d, HID, nc, gate_fn=gate_spectral)
    elif name == "energy_dist":
        m = GatedSAGE(in_d, HID, nc, gate_fn=gate_energy_dist)
    elif name == "ensemble":
        m = GatedSAGE(in_d, HID, nc, gate_fn=gate_ensemble)
    elif name == "llm_appnp":
        m = LLMAPPNPModel(in_d, HID, nc)
    elif name == "gnn_guard":
        m = GNNGuardModel(in_d, HID, nc)
    else:
        raise ValueError(name)
    return train_eval(m, x, ei, llm_e, labels, train_m, val_m, test_m)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(name):
    ds   = Planetoid(DATA_ROOT, name, split="public")
    data = ds[0]
    ei   = to_undirected(data.edge_index)
    return (data.x.float(), ei, data.y,
            data.train_mask, data.val_mask, data.test_mask)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════
METHODS = ["none", "bl_gcn", "bl_gat",
           "variance", "spectral", "energy_dist", "ensemble",
           "llm_appnp", "gnn_guard"]

# injection_type -> (label, which methods to run)
# We skip 'semantic' for most methods since those results already exist.
# We only run 'ensemble' under semantic to slot it into the existing comparison.
INJECT_PLAN = {
    "semantic": ["ensemble"],           # others already in llm_gates_results.json
    "random":   METHODS,
    "adaptive": METHODS,
}

DATASETS = {
    "cora":   (CORA_DESCS,   "Cora"),
    "pubmed": (PUBMED_DESCS, "PubMed"),
}

def main():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if "ablation" not in all_results:
        all_results["ablation"] = {}

    for ds_key, (class_descs, ds_name) in DATASETS.items():
        print(f"\n{'═'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'═'*60}")

        x, ei, labels, train_m, val_m, test_m = load_dataset(ds_name)
        N    = x.size(0)
        in_d = x.size(1)
        nc   = int(labels.max().item()) + 1

        # LLM embeddings: class-prototype lookup per node
        print("  Encoding LLM class prototypes …", flush=True)
        class_emb = encode_texts(class_descs)          # [nc, 768]
        llm_e     = class_emb[labels]                  # [N, 768]
        llm_d     = llm_e.size(1)

        if ds_key not in all_results["ablation"]:
            all_results["ablation"][ds_key] = {}
        ds_res = all_results["ablation"][ds_key]

        for inj_type, methods in INJECT_PLAN.items():
            if inj_type not in ds_res:
                ds_res[inj_type] = {}

            for method in methods:
                if method in ds_res[inj_type]:
                    print(f"  [skip] {inj_type}/{method}")
                    continue

                print(f"\n  ── {inj_type} / {method} ──")
                accs_per_ratio = []

                for ratio in RATIOS:
                    seed_accs = []
                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        random.seed(seed)
                        np.random.seed(seed)

                        # Inject edges
                        if inj_type == "semantic":
                            ei_atk = inject_semantic(ei, x, labels, N, ratio)
                        elif inj_type == "random":
                            ei_atk = inject_random(ei, labels, N, ratio)
                        elif inj_type == "adaptive":
                            ei_atk = inject_adaptive(ei, x, llm_e, labels, N, ratio)

                        acc = run_method(method, x, ei_atk, llm_e, labels,
                                         train_m, val_m, test_m, in_d, llm_d, nc)
                        seed_accs.append(acc)
                        print(f"  {method:12s}  r={int(ratio*100):2d}%  seed={seed}  {acc*100:.2f}%",
                              flush=True)

                    accs_per_ratio.append(float(np.mean(seed_accs)))

                ds_res[inj_type][method] = accs_per_ratio
                with open(RESULTS_PATH, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"  Saved {ds_key}/{inj_type}/{method}  "
                      f"drop={( accs_per_ratio[-1]-accs_per_ratio[0])*100:+.1f}pp")

    print("\n=== ABLATION SWEEP COMPLETE ===")

if __name__ == "__main__":
    main()
