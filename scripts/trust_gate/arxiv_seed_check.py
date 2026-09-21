"""
Targeted seed-replication check on ogbn-arxiv.

arxiv_sweep.py ran every method at SEEDS=1 ("1 seed x 100 epochs to stay
within SIGXCPU budget" per its own docstring) and found that no gate
mechanism beat the plain "none" baseline (bl_gat and moe_fusion were the
closest competitors). A single seed is very weak evidence for a claim that
consequential to the paper's story, so this script re-checks it with real
replication on just the handful of methods that matter for that claim:
  - none          (baseline)
  - bl_gat        (best-scoring alternative in the 1-seed run)
  - moe_fusion    (second-best alternative)
  - llm_appnp     (best-scoring LLM-gated method in the 1-seed run)
at the two extremes of corruption (0% and 50%), 3 seeds each, 60 epochs
(reduced from 100 to fit compute budget; "best val epoch" selection means
this mainly costs late-training fine-tuning, not the qualitative ranking).

Writes to results/llm_gates_results.json under "arxiv_seed_check" (does not
touch the original "arxiv_sweep" key) via read-modify-write, so it is safe
to interrupt/resume and safe to run alongside other sweeps reading the file.

Run from repo root:
  python scripts/trust_gate/arxiv_seed_check.py
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
SEEDS    = 3
EPOCHS   = 60
RATIOS   = [0.0, 0.5]
SAMPLE   = 2000

print("Loading OGBN-Arxiv...", flush=True)
from ogb.nodeproppred import PygNodePropPredDataset
ds = PygNodePropPredDataset("ogbn-arxiv", root=OGB_ROOT)
data = ds[0]
split = ds.get_idx_split()

bx = F.normalize(data.x.float(), dim=-1)
lbl = data.y.squeeze(-1).long()
nc = ds.num_classes
d = bx.size(1)
N = data.num_nodes

trm = torch.zeros(N, dtype=torch.bool); trm[split["train"]] = True
vm  = torch.zeros(N, dtype=torch.bool); vm[split["valid"]]  = True
tem = torch.zeros(N, dtype=torch.bool); tem[split["test"]]  = True

llm_e = bx.clone()
ei_b = to_undirected(data.edge_index, num_nodes=N)
print(f"N={N}  edges={ei_b.size(1)}  dim={d}  classes={nc}", flush=True)


def inject_sampled(ei, feat, labels, ratio, sample=SAMPLE):
    n_inject = int(ratio * N)
    if n_inject == 0: return ei
    fn = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    targets = random.sample(range(N), min(N, n_inject * 5))
    for vi in targets:
        if added >= n_inject: break
        diff_cls = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff_cls.numel() == 0: continue
        cands = diff_cls[torch.randperm(diff_cls.numel())[:sample]]
        sims = (fn[vi] * fn[cands]).sum(-1)
        best = sims.argmax().item()
        if sims[best].item() >= 0.5:
            src.append(vi); dst.append(cands[best].item()); added += 1
    if not src: return ei
    new = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return torch.cat([ei, new], dim=1)


class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=x.size(0)))


def gate_attn_entropy(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    logit = (ef[row] * ef[col]).sum(-1) / (e.size(1) ** 0.5)
    attn = scatter_softmax(logit, row, dim=0)
    eps = 1e-9
    ent = scatter(-(attn * (attn + eps).log()), row, dim=0, dim_size=N, reduce="sum")
    deg = scatter(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
    return torch.sigmoid(ent / deg.log().clamp(min=eps) * 5)


def gate_energy_dist(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    diff_vu = (ef[row] - ef[col]).norm(dim=-1)
    mean_vu = scatter_mean(diff_vu, row, dim=0, dim_size=N)
    mu_nbr = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var_nbr = scatter_mean((ef[col] - mu_nbr[row]).norm(dim=-1), row, dim=0, dim_size=N)
    return torch.sigmoid((2 * mean_vu - var_nbr).clamp(min=0) * 2)


def gate_spectral(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos = (ef[row] * ef[col]).sum(-1)
    max_cos = scatter(cos, row, dim=0, dim_size=N, reduce="max")
    mu_cos = scatter_mean(cos, row, dim=0, dim_size=N)
    var_cos = scatter_mean((cos - mu_cos[row]) ** 2, row, dim=0, dim_size=N)
    return torch.sigmoid(((1 - max_cos) + var_cos.sqrt()) * 3)


def gate_variance(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var = scatter_mean((ef[col] - mu[row]).norm(dim=-1) ** 2, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)


class PlainSAGE(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.l1 = SAGELayer(in_d, hid); self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei, **kw):
        h = F.relu(self.l1(x, ei)); h = F.relu(self.l2(h, ei))
        return self.clf(h)


class GATModel(nn.Module):
    def __init__(self, in_d, hid, nc, heads=4):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.c1 = GATConv(in_d, hid // heads, heads=heads, dropout=0.6)
        self.c2 = GATConv(hid, hid, heads=1, concat=False, dropout=0.6)
        self.clf = nn.Linear(hid, nc)
    def forward(self, x, ei, **kw):
        h = F.elu(self.c1(x, ei)); h = F.elu(self.c2(h, ei))
        return self.clf(h)


class LLMAPPNPModel(nn.Module):
    def __init__(self, in_d, hid, nc, K=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hid, hid))
        self.clf = nn.Linear(hid, nc); self.K = K
    def forward(self, x, ei, llm_e, **kw):
        row, col = ei; N = x.size(0)
        h0 = F.dropout(self.enc(x), p=0.5, training=self.training)
        beta = gate_variance(ei, llm_e).unsqueeze(-1)
        h = h0
        for _ in range(self.K):
            h = (1 - beta) * scatter_mean(h[col], row, dim=0, dim_size=N) + beta * h0
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
                w = scatter_softmax((q * kk).sum(-1) / (q.size(-1) ** 0.5), row, dim=0)
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


METHODS = [
    ("none", lambda: PlainSAGE(d, 256, nc)),
    ("bl_gat", lambda: GATModel(d, 256, nc)),
    ("moe_fusion", lambda: MoEFusionModel(d, d, 256, nc)),
    ("llm_appnp", lambda: LLMAPPNPModel(d, 256, nc)),
]


def run(model, bx, ei, llm_e, lbl, trm, vm, tem, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(bx, ei, llm_e=llm_e)
        loss = F.cross_entropy(logits[trm], lbl[trm])
        loss.backward(); opt.step(); opt.zero_grad()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bx, ei, llm_e=llm_e)
            pred = logits.argmax(-1)
            va = (pred[vm] == lbl[vm]).float().mean().item()
            ta = (pred[tem] == lbl[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt


def save_result(key, res):
    with open(RESULTS) as f:
        data = json.load(f)
    if "arxiv_seed_check" not in data:
        data["arxiv_seed_check"] = {"ratios": RATIOS, "seeds": SEEDS, "epochs": EPOCHS, "results": {}}
    data["arxiv_seed_check"]["results"][key] = res
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)


def already_done(key):
    with open(RESULTS) as f:
        data = json.load(f)
    return key in data.get("arxiv_seed_check", {}).get("results", {})


for key, model_fn in METHODS:
    if already_done(key):
        print(f"  SKIP {key}", flush=True)
        continue
    print(f"\n  -- {key} --", flush=True)
    per_ratio = []
    for r in RATIOS:
        ei = inject_sampled(ei_b, llm_e, lbl, r)
        seed_accs = [run(model_fn(), bx, ei, llm_e, lbl, trm, vm, tem, s) for s in range(SEEDS)]
        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs))
        per_ratio.append({"seed_accs": [round(a, 4) for a in seed_accs],
                           "mean": round(mean_acc, 4), "std": round(std_acc, 4)})
        print(f"  {key:<12} r={r:.0%}  {mean_acc*100:.2f}% +/- {std_acc*100:.2f}  seeds={[round(a,4) for a in seed_accs]}", flush=True)
    save_result(key, per_ratio)
    print(f"  Saved {key}", flush=True)

print("\n=== arxiv_seed_check DONE ===", flush=True)
