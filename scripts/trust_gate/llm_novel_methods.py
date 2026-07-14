"""
Novel LLM paradigms for graph trust — beyond text embedding.

A. cross_attn  — LLM pairwise cross-attention per edge (LLM-as-Judge)
B. verifier    — test-time LLM prototype score blended with GNN logits
C. rewired     — prototype-guided edge pruning before GNN runs
D. curriculum  — LLM-confidence curriculum training order
E. proto_gate  — class prototype alignment as gating signal beta_v
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── LLM encoder ──────────────────────────────────────────────────────────────
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

# ── Class descriptions ────────────────────────────────────────────────────────
# Cora: 7 classes (IDs 0-6 from Planetoid ordering)
CORA_DESCS = [
    "case based reasoning and analogical retrieval systems",
    "genetic algorithms evolutionary computation and optimization",
    "artificial neural networks deep learning and connectionist models",
    "probabilistic graphical models Bayesian inference and belief networks",
    "reinforcement learning reward-based agents and policy optimization",
    "rule learning inductive logic programming and symbolic methods",
    "computational learning theory formal analysis and complexity",
]
# PubMed: 3 classes
PUBMED_DESCS = [
    "experimental diabetes mellitus animal models laboratory research",
    "type 1 juvenile insulin-dependent diabetes mellitus treatment",
    "type 2 adult-onset non-insulin-dependent diabetes mellitus",
]

# ── Graph rewiring (C) ────────────────────────────────────────────────────────
def rewire_proto(ei, e, proto_embs):
    """Keep only edges where LLM proto prediction matches at both endpoints."""
    row, col = ei
    ef = F.normalize(e, dim=-1)
    pf = F.normalize(proto_embs, dim=-1)
    pred = (ef @ pf.T).argmax(-1)          # [N] pseudo-labels from LLM
    keep = pred[row] == pred[col]
    ei_new = torch.stack([row[keep], col[keep]])
    # Safety: ensure no node is completely isolated
    # (isolated nodes stay connected to themselves implicitly via self-transforms)
    return ei_new

# ── Backbone ──────────────────────────────────────────────────────────────────
class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))

# ── Model (handles all 5 methods) ─────────────────────────────────────────────
class NovelModel(nn.Module):
    def __init__(self, bow_dim, llm_dim, hid, nc, method):
        super().__init__()
        self.method = method
        self.l1  = SAGELayer(bow_dim, hid)
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
        if method == "cross_attn":
            # Low-rank cross-attention for edge trust scoring
            self.Wq = nn.Linear(llm_dim, 32, bias=False)
            self.Wk = nn.Linear(llm_dim, 32, bias=False)

    def _beta(self, ei, e, protos):
        """Compute gate value beta_v in [0,1]. Returns None for pure SAGE."""
        if self.method == "cross_attn":
            row, col = ei; N = e.size(0)
            q = self.Wq(e[row]); k = self.Wk(e[col])
            # Per-edge trust: how much should v trust edge (v,u)?
            score = (q * k).sum(-1) / (q.size(-1) ** 0.5)
            # Aggregate: mean trust from all neighbours
            mean_trust = scatter_mean(torch.sigmoid(score), row, dim=0, dim_size=N)
            return 1.0 - mean_trust   # high trust in nbrs → low β

        if self.method == "proto_gate":
            ef = F.normalize(e, dim=-1)
            pf = F.normalize(protos, dim=-1)
            # Confidence: max similarity to any class prototype
            max_sim = (ef @ pf.T).max(-1).values   # [N]
            # High confidence → trust self (resist potentially bad nbrs) → β high
            return torch.sigmoid((max_sim - 0.4) * 8)

        return None   # curriculum / verifier / rewired use pure SAGE

    def _gated(self, layer, x, ei, beta):
        row, col = ei; N = x.size(0)
        hs = layer.Ws(x)
        hn = layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
        b  = beta.unsqueeze(-1)
        return b * hs + (1 - b) * hn

    def forward(self, bow_x, ei, llm_e, protos=None):
        beta = self._beta(ei, llm_e, protos)
        if beta is None:
            h = F.relu(self.l1(bow_x, ei))
            h = F.relu(self.l2(h, ei))
        else:
            h = F.relu(self._gated(self.l1, bow_x, ei, beta))
            beta = self._beta(ei, llm_e, protos)
            h = F.relu(self._gated(self.l2, h,     ei, beta))
        return self.clf(h)

# ── Training / evaluation ─────────────────────────────────────────────────────
def train_eval(method, bow_x, ei, llm_e, labels, protos,
               trm, vm, tem, nc, seed, epochs=150, curriculum_scores=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = NovelModel(bow_x.size(1), llm_e.size(1), 128, nc, method)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # Curriculum: identify high-confidence (easy) training nodes
    easy_mask = None
    if method == "curriculum" and curriculum_scores is not None:
        tidx   = trm.nonzero(as_tuple=True)[0]
        scores = curriculum_scores[tidx]
        easy_tidx = tidx[scores.argsort(descending=True)[:len(tidx)//2]]
        easy_mask = torch.zeros(bow_x.size(0), dtype=torch.bool)
        easy_mask[easy_tidx] = True

    bv = bt = 0.0
    for ep in range(epochs):
        model.train()
        # Select training nodes
        if method == "curriculum" and easy_mask is not None and ep < epochs // 2:
            cur_trm = easy_mask
        else:
            cur_trm = trm

        logits = model(bow_x, ei, llm_e, protos)
        F.cross_entropy(logits[cur_trm], labels[cur_trm]).backward()
        opt.step(); opt.zero_grad()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(bow_x, ei, llm_e, protos)
                # Verifier: blend GNN logits with LLM prototype similarities
                if method == "verifier":
                    ef = F.normalize(llm_e, dim=-1)
                    pf = F.normalize(protos, dim=-1)
                    proto_logits = ef @ pf.T        # [N, C]
                    logits = 0.7 * logits + 0.3 * proto_logits

            pred = logits.argmax(-1)
            va = (pred[vm]  == labels[vm]).float().mean().item()
            ta = (pred[tem] == labels[tem]).float().mean().item()
            if va > bv:
                bv, bt = va, ta
    return bt

# ── Edge injection (same as existing sweep) ───────────────────────────────────
def inject(ei, feat, labels, ratio):
    N = feat.size(0); n = int(ratio * N)
    if n == 0: return ei
    fn = F.normalize(feat, dim=-1)
    src, dst, added = [], [], 0
    for vi in random.sample(range(N), N):
        if added >= n: break
        sim = fn[vi] @ fn.T
        sim[vi] = -1
        sim[labels == labels[vi]] = -1
        j = sim.argmax().item()
        if sim[j].item() >= 0.5:
            src.append(vi); dst.append(j); added += 1
    if not src: return ei
    ne = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return torch.cat([ei, ne], 1)

# ── Config ────────────────────────────────────────────────────────────────────
NOVEL_METHODS = ["cross_attn", "proto_gate", "rewired", "curriculum", "verifier"]
SEEDS   = 2
EPOCHS  = 150
RATIOS  = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"

CORA_PT = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Cora/cora.pt"
PUB_PT  = "/home/lam23005/STEM-GNN/STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt"

EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(EXISTING) as f:
    all_results = json.load(f)

# ── Dataset loop ──────────────────────────────────────────────────────────────
for ds_name, bow_name, raw_pt, sweep_key, class_descs in [
    ("Cora",   "Cora",   CORA_PT, "cora_sweep",   CORA_DESCS),
    ("PubMed", "PubMed", PUB_PT,  "pubmed_sweep",  PUBMED_DESCS),
]:
    print(f"\n{'='*60}")
    print(f"  {ds_name}  —  novel LLM paradigms")
    print(f"{'='*60}", flush=True)

    pl_data = Planetoid(PL_ROOT, bow_name)[0]
    bow_x   = F.normalize(pl_data.x.float(), dim=-1)
    labels  = pl_data.y.long()
    nc      = int(labels.max().item()) + 1
    ei_base = to_undirected(pl_data.edge_index, num_nodes=pl_data.num_nodes)
    trm, vm, tem = pl_data.train_mask, pl_data.val_mask, pl_data.test_mask

    print(f"  Encoding {ds_name} texts ...", flush=True)
    raw_data = torch.load(raw_pt)
    llm_e    = encode_texts(raw_data.raw_texts)

    print(f"  Encoding {len(class_descs)} class descriptions ...", flush=True)
    protos   = encode_texts(class_descs)          # [C, 768]

    # Proto accuracy check (sanity)
    ef   = F.normalize(llm_e, dim=-1)
    pf   = F.normalize(protos, dim=-1)
    pred = (ef @ pf.T).argmax(-1)
    proto_acc = (pred == labels).float().mean().item()
    print(f"  LLM proto zero-shot acc: {proto_acc*100:.1f}%  "
          f"(tells us how good class descriptions are)", flush=True)

    # Curriculum scores: cos(e_v, proto[y_v]) for each node
    curriculum_scores = (ef * pf[labels]).sum(-1)
    print(f"  Curriculum score range: {curriculum_scores.min():.3f} – "
          f"{curriculum_scores.max():.3f}", flush=True)

    for method in NOVEL_METHODS:
        print(f"\n  ── {method} ──", flush=True)
        res_method = []

        for r in RATIOS:
            ei_inj = inject(ei_base, llm_e, labels, r)

            # Rewired: apply prototype-guided pruning AFTER injection
            if method == "rewired":
                ei_use = rewire_proto(ei_inj, llm_e, protos)
                n_kept = ei_use.size(1)
                n_orig = ei_inj.size(1)
                print(f"    rewire: kept {n_kept}/{n_orig} edges "
                      f"({100*n_kept/max(n_orig,1):.0f}%)", flush=True)
            else:
                ei_use = ei_inj

            accs = []
            for s in range(SEEDS):
                a = train_eval(
                    method, bow_x, ei_use, llm_e, labels, protos,
                    trm, vm, tem, nc, s, EPOCHS,
                    curriculum_scores=curriculum_scores,
                )
                accs.append(a)
            acc = float(np.mean(accs))
            res_method.append(round(acc, 4))
            print(f"  {method:<14} ratio={r:.0%}  {acc*100:.2f}%", flush=True)

        all_results[sweep_key]["results"][method] = res_method

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== Saved → {OUT} ===")
