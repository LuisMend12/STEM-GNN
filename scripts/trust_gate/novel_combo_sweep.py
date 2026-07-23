"""
Novel combination methods drawn from prior robustness literature.

Y1. svd_vargate   [GCNSVD × Variance Gate]
    Entezari et al. WSDM 2020 + our variance gate.
    Step 1: Sparse truncated-SVD scores every edge in the attacked graph.
            Adversarial edges violate the low-rank class structure → low score → pruned.
    Step 2: Variance gate on the SVD-filtered graph catches residual semantic noise.
    Two independent, complementary filters: spectral structure × LLM semantics.

Y2. elastic_gate  [ElasticGNN × Variance Gate]
    Liu et al. ICML 2021 + our variance gate.
    Replace mean aggregation with L1-proximal (soft-threshold) aggregation:
        elastic(x̄) = sign(x̄) · max(|x̄| − λ, 0)
    L1 shrinks outlier coordinate contributions from adversarial neighbours;
    variance gate provides the second layer of semantic protection.

Y3. proto_clean   [Pro-GNN × LLM Prototypes]
    Jin et al. KDD 2020 + our proto_entropy gate.
    One-shot, no-optimisation structure learning: prune edges where the LLM
    prototype-predicted classes of both endpoints disagree, then apply a
    variance gate on the cleaned graph. Faster than Pro-GNN; immune to
    training-time poisoning because prototypes come from frozen LLM embeddings.

Y4. rgcn_llm      [RobustGCN × LLM Variance]
    Zhu et al. KDD 2019 + our variance gate.
    RobustGCN's learned uncertainty attention × frozen LLM variance trust:
        w(u→v) = exp(−σ²_u) · (1 − β_u)
    Both sources must agree for an edge to be trusted.
    Learned sig² can eventually be poisoned at training time; the frozen LLM
    gate provides an attack-immune second veto.
"""
import os, json, random, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter

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

# ── Gate signals ───────────────────────────────────────────────────────────────
def gate_variance(ei, e):
    row, col = ei; N = e.size(0)
    ef  = F.normalize(e, dim=-1)
    mu  = scatter_mean(ef[col], row, dim=0, dim_size=N)
    var = scatter_mean((ef[col] - mu[row]).norm(dim=-1)**2, row, dim=0, dim_size=N)
    return torch.sigmoid(var * 10)

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE
# ══════════════════════════════════════════════════════════════════════════════
class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))

class GatedSAGE(nn.Module):
    """Standard variance-gated GraphSAGE (used by svd_vargate and proto_clean)."""
    def __init__(self, bow_dim, hid, nc):
        super().__init__()
        self.l1 = SAGELayer(bow_dim, hid)
        self.l2 = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
    def _apply(self, layer, x, ei, b):
        row, col = ei; N = x.size(0)
        b = b.unsqueeze(-1)
        return b * layer.Ws(x) + (1-b) * layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
    def forward(self, bow_x, ei, llm_e, **kw):
        b = gate_variance(ei, llm_e)
        h = F.relu(self._apply(self.l1, bow_x, ei, b))
        b = gate_variance(ei, llm_e)
        h = F.relu(self._apply(self.l2, h, ei, b))
        return self.clf(h)

# ══════════════════════════════════════════════════════════════════════════════
# Y1. SVD-VarGate  ── spectral pruning then LLM uncertainty gate
# ══════════════════════════════════════════════════════════════════════════════
def svd_filter(ei, N, rank=20):
    """Score every edge by its reconstruction value in the top-rank SVD.
    Adversarial edges violate the low-rank class structure → negative/near-zero
    scores → pruned.  Returns a filtered edge_index."""
    row_np = ei[0].numpy(); col_np = ei[1].numpy()
    data = np.ones(len(row_np), dtype=np.float32)
    A = sp.csr_matrix((data, (row_np, col_np)), shape=(N, N))
    A = (A + A.T) / 2  # symmetrize (handles undirected)
    k = min(rank, N - 2)
    U, S, Vt = svds(A, k=k)           # U: [N,k]  S: [k]  Vt: [k,N]
    US   = U * S[None, :]              # [N, k]
    VtT  = Vt.T                        # [N, k]
    scores = (US[row_np] * VtT[col_np]).sum(-1)   # [E]
    keep   = scores > 0.0
    if keep.sum() == 0:
        return ei                      # fallback: no pruning if degenerate
    return torch.stack([
        torch.tensor(row_np[keep], dtype=torch.long),
        torch.tensor(col_np[keep], dtype=torch.long),
    ])

# Model = GatedSAGE (defined above); edge_index is pre-filtered externally.

# ══════════════════════════════════════════════════════════════════════════════
# Y2. ElasticGate  ── L1-proximal aggregation + LLM uncertainty gate
# ══════════════════════════════════════════════════════════════════════════════
class ElasticSAGELayer(nn.Module):
    def __init__(self, in_d, out_d, lam=0.05):
        super().__init__()
        self.Ws  = nn.Linear(in_d, out_d, bias=False)
        self.Wn  = nn.Linear(in_d, out_d, bias=False)
        self.lam = lam
    def forward(self, x, ei, beta):
        row, col = ei; N = x.size(0)
        nbr_mean = scatter_mean(x[col], row, dim=0, dim_size=N)
        # L1 proximal operator: soft-threshold each coordinate of the mean
        # Adversarial outlier dimensions are clipped; clean dimensions survive.
        elastic  = nbr_mean.sign() * (nbr_mean.abs() - self.lam).clamp(min=0)
        b = beta.unsqueeze(-1)
        return b * self.Ws(x) + (1 - b) * self.Wn(elastic)

class ElasticGate(nn.Module):
    def __init__(self, bow_dim, hid, nc, lam=0.05):
        super().__init__()
        self.l1  = ElasticSAGELayer(bow_dim, hid, lam)
        self.l2  = ElasticSAGELayer(hid, hid, lam)
        self.clf = nn.Linear(hid, nc)
    def forward(self, bow_x, ei, llm_e, **kw):
        b = gate_variance(ei, llm_e)
        h = F.relu(self.l1(bow_x, ei, b))
        b = gate_variance(ei, llm_e)
        h = F.relu(self.l2(h, ei, b))
        return self.clf(h)

# ══════════════════════════════════════════════════════════════════════════════
# Y3. ProtoClean  ── one-shot LLM prototype pruning + variance gate
# ══════════════════════════════════════════════════════════════════════════════
def proto_prune(ei, llm_e, labels, train_mask, nc):
    """Remove edges whose endpoints disagree on LLM prototype class prediction.
    Isolated nodes (all edges removed) get their original edges restored."""
    ef = F.normalize(llm_e, dim=-1)
    d  = ef.shape[1]
    P  = torch.zeros(nc, d)
    for c in range(nc):
        m = train_mask & (labels == c)
        if m.sum() > 0:
            P[c] = ef[m].mean(0)
    P    = F.normalize(P, dim=-1)
    pred = (ef @ P.T).argmax(-1)          # [N] — LLM prototype prediction
    row, col = ei
    keep = pred[row] == pred[col]          # keep only same-predicted-class edges
    # Restore edges for isolated nodes so no node goes dark
    N = llm_e.size(0)
    kept_deg = scatter(keep.long(), row, dim=0, dim_size=N, reduce="sum")
    isolated = kept_deg == 0
    if isolated.any():
        keep = keep | isolated[row]
    return ei[:, keep]

# Model = GatedSAGE (defined above); edge_index is pre-pruned externally.

# ══════════════════════════════════════════════════════════════════════════════
# Y4. RobustGCN-LLM  ── RobustGCN learned attention × frozen LLM gate
# ══════════════════════════════════════════════════════════════════════════════
class RobustGCNLLMConv(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W_mu  = nn.Linear(in_d, out_d, bias=False)
        self.W_sig = nn.Linear(in_d, out_d, bias=False)
    def forward(self, mu, sig2, ei, N, llm_beta):
        row, col = ei
        # RobustGCN: high learned-variance neighbor → low trust
        learned = torch.exp(-sig2[col].mean(-1).clamp(max=10))
        # LLM gate: high LLM uncertainty at neighbor → low trust
        llm     = (1.0 - llm_beta[col]).clamp(min=0)
        w_raw   = learned * llm
        w       = w_raw / scatter(w_raw, row, dim=0, dim_size=N, reduce="sum")[row].clamp(min=1e-9)
        agg_mu  = scatter(mu[col]   * w.unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        agg_sig = scatter(sig2[col] * (w**2).unsqueeze(-1), row, dim=0, dim_size=N, reduce="sum")
        return F.relu(self.W_mu(mu + agg_mu)), F.relu(self.W_sig(sig2 + agg_sig))

class RobustGCNLLM(nn.Module):
    def __init__(self, in_d, hid, nc):
        super().__init__()
        self.sig_init = nn.Linear(in_d, in_d, bias=False)
        self.c1  = RobustGCNLLMConv(in_d, hid)
        self.c2  = RobustGCNLLMConv(hid,  hid)
        self.clf = nn.Linear(hid, nc)
    def _encode(self, bow_x, ei, llm_e):
        N    = bow_x.size(0)
        beta = gate_variance(ei, llm_e)          # frozen LLM signal
        mu   = bow_x
        sig2 = F.relu(self.sig_init(bow_x)) + 1e-4
        mu, sig2 = self.c1(mu, sig2, ei, N, beta)
        mu, sig2 = self.c2(mu, sig2, ei, N, beta)
        return mu, sig2
    def forward(self, bow_x, ei, llm_e, **kw):
        mu, _ = self._encode(bow_x, ei, llm_e)
        return self.clf(mu)
    def kl_loss(self, bow_x, ei, llm_e):
        mu, sig2 = self._encode(bow_x, ei, llm_e)
        return 0.5 * (sig2 + mu**2 - sig2.clamp(min=1e-9).log() - 1).mean()

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
        if sim[j].item() >= 0.5:
            src.append(vi); dst.append(j); added += 1
    if not src: return ei
    return torch.cat([ei, torch.tensor([src+dst, dst+src], dtype=torch.long)], 1)

def run(model, bow_x, ei, llm_e, labels, trm, vm, tem, seed,
        epochs=150, is_rgcn=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(bow_x, ei, llm_e=llm_e)
        loss   = F.cross_entropy(logits[trm], labels[trm])
        if is_rgcn:
            loss = loss + 5e-4 * model.kl_loss(bow_x, ei, llm_e)
        loss.backward(); opt.step(); opt.zero_grad()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bow_x, ei, llm_e=llm_e)
            pred = logits.argmax(-1)
            va   = (pred[vm]  == labels[vm] ).float().mean().item()
            ta   = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv: bv, bt = va, ta
    return bt

# ══════════════════════════════════════════════════════════════════════════════
# SAVE / SKIP HELPERS
# ══════════════════════════════════════════════════════════════════════════════
EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"

def load_results():
    with open(EXISTING) as f:
        return json.load(f)

def save_result(skey, method_key, res):
    data = load_results()
    data[skey]["results"][method_key] = res
    with open(EXISTING, "w") as f:
        json.dump(data, f, indent=2)

def already_done(skey, method_key):
    data = load_results()
    return method_key in data[skey]["results"]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SEEDS  = 2
EPOCHS = 150
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]

PL_ROOT  = "/tmp/planetoid"
CORA_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT   = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

SVD_RANK = 20   # top singular vectors to retain

DATASETS = [
    ("Cora",   "Cora",   CORA_PT,  "cora_sweep",  CORA_DESCS),
    ("PubMed", "PubMed", PUB_PT,   "pubmed_sweep", PUBMED_DESCS),
]

for ds_name, bow_name, raw_pt, skey, descs in DATASETS:
    print(f"\n{'='*60}\n  {ds_name}\n{'='*60}", flush=True)
    pl   = Planetoid(PL_ROOT, bow_name)[0]
    bx   = F.normalize(pl.x.float(), dim=-1)
    lbl  = pl.y.long()
    nc   = int(lbl.max().item()) + 1
    d    = bx.size(1)
    N    = bx.size(0)
    ei_b = to_undirected(pl.edge_index, num_nodes=N)
    trm, vm, tem = pl.train_mask, pl.val_mask, pl.test_mask

    print("  Encoding node texts …", flush=True)
    llm_e = encode_texts(torch.load(raw_pt).raw_texts)
    print(f"  LLM embeddings: {llm_e.shape}", flush=True)

    # ── method definitions ────────────────────────────────────────────────────
    # Each entry: (key, model_factory, uses_rgcn_kl, pre_fn)
    # pre_fn(ei_attacked) → ei_to_use  (None = use attacked graph directly)
    METHODS = [
        ("svd_vargate",
         lambda: GatedSAGE(d, 128, nc),
         False,
         lambda ei_a: svd_filter(ei_a, N, rank=SVD_RANK)),

        ("elastic_gate",
         lambda: ElasticGate(d, 128, nc, lam=0.05),
         False,
         None),

        ("proto_clean",
         lambda: GatedSAGE(d, 128, nc),
         False,
         lambda ei_a: proto_prune(ei_a, llm_e, lbl, trm, nc)),

        ("rgcn_llm",
         lambda: RobustGCNLLM(d, 128, nc),
         True,
         None),
    ]

    for key, model_fn, is_rgcn, pre_fn in METHODS:
        if already_done(skey, key):
            print(f"  SKIP {key} (already done)", flush=True)
            continue
        print(f"\n  ── {key} ──", flush=True)
        res = []
        for r in RATIOS:
            ei_attacked = inject(ei_b, llm_e, lbl, r)
            ei_use = pre_fn(ei_attacked) if pre_fn else ei_attacked

            if pre_fn and r > 0:
                orig_e = ei_attacked.shape[1]
                filt_e = ei_use.shape[1]
                print(f"    [{key}] edges: {orig_e} → {filt_e} "
                      f"(−{orig_e-filt_e} = "
                      f"{100*(orig_e-filt_e)/orig_e:.1f}% removed)", flush=True)

            accs = [run(model_fn(), bx, ei_use, llm_e, lbl, trm, vm, tem,
                        s, EPOCHS, is_rgcn=is_rgcn)
                    for s in range(SEEDS)]
            acc  = float(np.mean(accs))
            res.append(round(acc, 4))
            print(f"  {key:<16} r={r:.0%}  {acc*100:.2f}%", flush=True)

        save_result(skey, key, res)
        print(f"  Saved {key}", flush=True)

print("\n=== ALL DONE ===")
