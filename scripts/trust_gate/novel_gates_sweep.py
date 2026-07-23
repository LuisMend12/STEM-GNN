"""
Novel LLM-guided gate signals — 4 new methods.

X1  proto_entropy   β_v = entropy of node's distribution over class prototypes
                    (approximates LLM CoT uncertainty without any prompting)

X2  proto_disagree  β_v = 1 - sim(neighbourhood centroid, v's predicted prototype)
                    (high when neighbourhood contradicts LLM class prediction for v)

X3  contrastive     β_v = 1 - mean learned helpfulness of neighbours
                    (MLP trained to distinguish same-class vs different-class edges
                     using ONLY training-set labels — frozen before main training)

X4  mi_approx       β_v = 1 - r²(z_v, z̄_N(v))
                    (1 minus coefficient of determination of neighbourhood centroid vs node;
                     measures how much neighbourhood "explains" the node in LLM space)

Comparison: all 4 run on Cora + PubMed with the same semantic injection attack
as existing methods. Results saved to "novel_gates" key in llm_gates_results.json.
"""
import json, math, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter, scatter_softmax

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS   = 2
EPOCHS  = 150
RATIOS  = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
RESULTS = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
CORA_PT = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

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

# ──────────────────────────────────────────────────────────────────────────────
# LLM encoder
# ──────────────────────────────────────────────────────────────────────────────
def encode_texts(texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("multi-qa-distilbert-cos-v1")
    m.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        with torch.no_grad():
            embs.append(m.encode(texts[i:i+batch_size],
                                 convert_to_tensor=True,
                                 show_progress_bar=False).cpu())
    return torch.cat(embs, 0)

# ──────────────────────────────────────────────────────────────────────────────
# Shared backbone
# ──────────────────────────────────────────────────────────────────────────────
class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))

class GatedSAGE(nn.Module):
    """GraphSAGE with a pluggable gate function β_v ∈ [0,1]."""
    def __init__(self, bow_dim, hid, nc, gate_fn):
        super().__init__()
        self.gate_fn = gate_fn
        self.l1 = SAGELayer(bow_dim, hid)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _apply_gate(self, layer, x, ei, b):
        row, col = ei; N = x.size(0)
        b = b.unsqueeze(-1)
        return b * layer.Ws(x) + (1 - b) * layer.Wn(
            scatter_mean(x[col], row, dim=0, dim_size=N))
    def forward(self, bow_x, ei, llm_e, **kw):
        b = self.gate_fn(ei, llm_e)
        h = F.relu(self._apply_gate(self.l1, bow_x, ei, b))
        b = self.gate_fn(ei, llm_e)
        h = F.relu(self._apply_gate(self.l2, h, ei, b))
        return self.clf(h)

# ──────────────────────────────────────────────────────────────────────────────
# Novel gate signal functions
# ──────────────────────────────────────────────────────────────────────────────

def make_gate_proto_entropy(llm_e_full, labels, train_mask, nc, temp=0.5):
    """
    X1 — Proto-Entropy gate.
    Precomputes class prototypes from training nodes in LLM space.
    At runtime: β_v = H(softmax(z_v @ P.T / τ)) / log(C).
    High entropy → v is on a class boundary in LLM space → distrust its neighbourhood.
    """
    zn = F.normalize(llm_e_full, dim=-1)
    d = zn.size(1)
    P = torch.zeros(nc, d)
    for c in range(nc):
        mask = train_mask & (labels == c)
        if mask.sum() > 0:
            P[c] = zn[mask].mean(0)
    P = F.normalize(P, dim=-1)

    def gate(ei, e):
        ef = F.normalize(e, dim=-1)
        sims = (ef @ P.to(ef.device).T) / temp      # [N, C]
        probs = F.softmax(sims, dim=-1)              # [N, C]
        H = -(probs * probs.clamp(min=1e-8).log()).sum(-1)  # [N]
        return (H / math.log(nc)).clamp(0, 1)
    return gate


def make_gate_proto_disagree(llm_e_full, labels, train_mask, nc):
    """
    X2 — Proto-Disagree gate.
    For each node v: (a) predict v's class from LLM prototypes,
    (b) compute the centroid of v's LLM neighbourhood,
    (c) β_v = 1 - sim(neighbourhood centroid, predicted prototype).
    High when the graph neighbourhood contradicts the LLM class prediction.
    """
    zn = F.normalize(llm_e_full, dim=-1)
    d = zn.size(1)
    P = torch.zeros(nc, d)
    for c in range(nc):
        mask = train_mask & (labels == c)
        if mask.sum() > 0:
            P[c] = zn[mask].mean(0)
    P = F.normalize(P, dim=-1)

    def gate(ei, e):
        row, col = ei; N = e.size(0)
        Pd = P.to(e.device)
        ef = F.normalize(e, dim=-1)
        # Predicted class for each node
        pred_c = (ef @ Pd.T).argmax(-1)             # [N]
        pred_proto = Pd[pred_c]                      # [N, d]
        # Neighbourhood centroid
        nbr_mean = scatter_mean(ef[col], row, dim=0, dim_size=N)  # [N, d]
        nbr_mean = F.normalize(nbr_mean, dim=-1)
        # Cosine agreement in [-1, 1] → map to [0, 1] for β
        agree = (nbr_mean * pred_proto).sum(-1)      # [N]
        return ((1 - agree) / 2).clamp(0, 1)
    return gate


def train_contrastive_mlp(llm_e, ei_clean, labels, train_mask, epochs=100):
    """
    Train a 2-layer MLP f(z_v, z_u) → [0,1] (1=same-class, helpful).
    Uses ONLY labeled training-node edges from the CLEAN graph.
    The MLP is then frozen and used as a gate signal.
    """
    zn = F.normalize(llm_e, dim=-1)
    d = zn.size(1)

    row, col = ei_clean
    # Only labeled edges (both endpoints in train_mask)
    labeled = train_mask[row] & train_mask[col]
    src = row[labeled]; dst = col[labeled]
    y = (labels[src] == labels[dst]).float()

    if len(y) < 20 or y.sum() < 5 or (1 - y).sum() < 5:
        return None  # degenerate — fall back to 0.5

    pairs = torch.cat([zn[src], zn[dst]], dim=-1).detach()

    mlp = nn.Sequential(
        nn.Linear(2 * d, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64),   nn.ReLU(),
        nn.Linear(64, 1),     nn.Sigmoid()
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        mlp.train()
        loss = F.binary_cross_entropy(mlp(pairs).squeeze(-1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return mlp.eval()


def make_gate_contrastive(mlp, N_nodes):
    """
    X3 — Contrastive gate.
    β_v = 1 - mean helpfulness score of v's neighbours (from frozen MLP).
    """
    if mlp is None:
        # Fallback: use variance-style gate
        def gate_fallback(ei, e):
            row, col = ei; N = e.size(0)
            ef = F.normalize(e, dim=-1)
            mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
            var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2, row, dim=0, dim_size=N)
            return torch.sigmoid(var * 10)
        return gate_fallback

    def gate(ei, e):
        row, col = ei; N = e.size(0)
        ef = F.normalize(e, dim=-1)
        with torch.no_grad():
            pairs = torch.cat([ef[row], ef[col]], dim=-1)
            helpfulness = mlp(pairs).squeeze(-1)       # [E]
        help_sum = scatter(helpfulness, row, dim=0, dim_size=N, reduce="sum")
        cnt = scatter(torch.ones(col.size(0), device=e.device),
                      row, dim=0, dim_size=N, reduce="sum").clamp(min=1)
        mean_help = help_sum / cnt                      # [N]
        return (1 - mean_help).clamp(0, 1)
    return gate


def gate_mi_approx(ei, e):
    """
    X4 — MI-Approx gate.
    β_v = 1 - cosine(z_v, z̄_{N(v)})²

    Intuition: the squared cosine of node v to its neighbourhood centroid
    is the coefficient of determination r² — how well the centroid "explains" v.
    Low r² → neighbourhood centroid has drifted from v (injection of wrong-class nodes
    pulls the centroid away from v's cluster) → distrust neighbourhood.

    Differs from gate A (mean per-edge cosine) because the centroid of a MIXED
    neighbourhood is penalised even when individual edge cosines are all ≥ 0.5.
    """
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    nbr_mean = scatter_mean(ef[col], row, dim=0, dim_size=N)  # [N, d]
    nbr_mean_n = F.normalize(nbr_mean, dim=-1)
    r = (ef * nbr_mean_n).sum(-1)                    # cosine(z_v, z̄_N) ∈ [-1,1]
    r2 = r ** 2                                       # coefficient of determination
    return (1 - r2).clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Injection (semantic: cosine ≥ 0.5, wrong class)
# ──────────────────────────────────────────────────────────────────────────────
def inject(ei, llm_emb, labels, ratio, sample=500):
    N = llm_emb.size(0); n = int(ratio * N)
    if n == 0: return ei
    fn = F.normalize(llm_emb, dim=-1)
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), N):
        if added >= n: break
        # Wrong-class candidates
        diff = (labels != labels[vi]).nonzero(as_tuple=True)[0]
        if diff.numel() == 0: continue
        cands = diff[torch.randperm(diff.numel())[:sample]]
        sims = (fn[vi] * fn[cands]).sum(-1)
        valid = cands[sims >= 0.5]
        if valid.numel() == 0: continue
        j = valid[0].item()
        src.append(vi); dst.append(j); added += 1
    if not src: return ei
    new = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return torch.cat([ei, new], 1)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def run(model, bow_x, ei, llm_e, labels, trm, vm, tem, seed, epochs=150):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = model.to(DEVICE)
    bow_x, ei = bow_x.to(DEVICE), ei.to(DEVICE)
    llm_e, labels = llm_e.to(DEVICE), labels.to(DEVICE)
    trm, vm, tem = trm.to(DEVICE), vm.to(DEVICE), tem.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(bow_x, ei, llm_e=llm_e)
        F.cross_entropy(logits[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bow_x, ei, llm_e=llm_e)
            pred = logits.argmax(-1)
            va = (pred[vm]  == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt


# ──────────────────────────────────────────────────────────────────────────────
# Read-modify-write helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_result(ds_key, method_key, res):
    with open(RESULTS) as f:
        data = json.load(f)
    if "novel_gates" not in data:
        data["novel_gates"] = {}
    if ds_key not in data["novel_gates"]:
        data["novel_gates"][ds_key] = {}
    data["novel_gates"][ds_key][method_key] = res
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)

def already_done(ds_key, method_key):
    with open(RESULTS) as f:
        data = json.load(f)
    return method_key in data.get("novel_gates", {}).get(ds_key, {})


# ──────────────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────────────
for ds_name, bow_name, raw_pt, descs, ds_key in [
    ("Cora",   "Cora",   CORA_PT, CORA_DESCS,   "cora"),
    ("PubMed", "PubMed", PUB_PT,  PUBMED_DESCS,  "pubmed"),
]:
    print(f"\n{'='*60}\n  {ds_name}\n{'='*60}", flush=True)

    pl  = Planetoid(PL_ROOT, bow_name)[0]
    bx  = F.normalize(pl.x.float(), dim=-1)
    lbl = pl.y.long()
    nc  = int(lbl.max().item()) + 1
    d   = bx.size(1)
    N   = bx.size(0)
    ei_clean = to_undirected(pl.edge_index, num_nodes=N)
    trm, vm, tem = pl.train_mask, pl.val_mask, pl.test_mask

    print("  Encoding node texts...", flush=True)
    llm_e = encode_texts(torch.load(raw_pt).raw_texts)
    print(f"  BOW={bx.shape}  LLM={llm_e.shape}  C={nc}", flush=True)

    # ── Precompute gate closures (once per dataset, before any injection) ──────
    print("  Building prototype gates...", flush=True)
    gate_pe = make_gate_proto_entropy(llm_e, lbl, trm, nc)
    gate_pd = make_gate_proto_disagree(llm_e, lbl, trm, nc)

    print("  Training contrastive MLP...", flush=True)
    cont_mlp = train_contrastive_mlp(llm_e, ei_clean, lbl, trm)
    gate_cont = make_gate_contrastive(cont_mlp, N)
    if cont_mlp is None:
        print("  WARNING: contrastive MLP degenerate, using variance fallback", flush=True)

    # Methods: (key, gate_fn, hid)
    METHODS = [
        ("proto_entropy",  gate_pe,   128),
        ("proto_disagree", gate_pd,   128),
        ("contrastive",    gate_cont, 128),
        ("mi_approx",      gate_mi_approx, 128),
    ]

    for key, gate_fn, hid in METHODS:
        if already_done(ds_key, key):
            print(f"  SKIP {key} (already done)", flush=True)
            continue
        print(f"\n  ── {key} ──", flush=True)
        res = []
        for r in RATIOS:
            ei_atk = inject(ei_clean, llm_e, lbl, r)
            accs = [
                run(GatedSAGE(d, hid, nc, gate_fn),
                    bx, ei_atk, llm_e, lbl, trm, vm, tem, s, EPOCHS)
                for s in range(SEEDS)
            ]
            acc = float(np.mean(accs))
            res.append(round(acc, 4))
            print(f"  {key:<20} r={r:.0%}  {acc*100:.2f}%", flush=True)
        save_result(ds_key, key, res)
        drop = (res[-1] - res[0]) * 100
        print(f"  Saved {ds_key}/{key}  drop={drop:+.1f}pp", flush=True)

print("\n=== NOVEL GATES SWEEP COMPLETE ===")
