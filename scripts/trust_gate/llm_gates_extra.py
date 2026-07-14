"""
Run only the 5 new gate variants (E-I) and merge into existing results.
Uses 2 seeds + 150 epochs to stay within CPU budget.
"""
import os, sys, random, json
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "STEM-GNN"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean, scatter

# ── LLM encoder ───────────────────────────────────────────────────────────────
def encode_texts(texts, batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("multi-qa-distilbert-cos-v1")
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        with torch.no_grad():
            embs = model.encode(texts[i:i+batch_size], convert_to_tensor=True,
                                show_progress_bar=False)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, 0)

# ── New gate functions ────────────────────────────────────────────────────────
def gate_max_cosine(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos_e = (ef[row] * ef[col]).sum(-1)
    return 1 - torch.sigmoid(scatter(cos_e, row, dim=0, dim_size=N, reduce="max"))

def gate_nbr_agree(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    mu = scatter_mean(ef[col], row, dim=0, dim_size=N)
    return 1 - (mu * mu).sum(-1)

def gate_topk(ei, e, k=3):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos_e    = (ef[row] * ef[col]).sum(-1)
    sort_idx = torch.argsort(row.float() * 1e9 - cos_e.detach())
    row_s    = row[sort_idx]; cos_s = cos_e[sort_idx]
    positions = torch.arange(row_s.size(0), device=e.device)
    first_pos = scatter(positions.float(), row_s, dim=0, dim_size=N, reduce="min").long()
    rank      = positions - first_pos[row_s]
    mask      = rank < k
    return 1 - torch.sigmoid(scatter_mean(cos_s[mask], row_s[mask], dim=0, dim_size=N))

def gate_cos_std(ei, e):
    row, col = ei; N = e.size(0)
    ef = F.normalize(e, dim=-1)
    cos_e  = (ef[row] * ef[col]).sum(-1)
    mu_c   = scatter_mean(cos_e, row, dim=0, dim_size=N)
    var_c  = scatter_mean((cos_e - mu_c[row])**2, row, dim=0, dim_size=N)
    return torch.sigmoid(var_c.sqrt() * 10)

# ── Model ─────────────────────────────────────────────────────────────────────
class SAGELayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.Ws = nn.Linear(in_d, out_d, bias=False)
        self.Wn = nn.Linear(in_d, out_d, bias=False)
    def forward(self, x, ei):
        row, col = ei; N = x.size(0)
        return self.Ws(x) + self.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))

class LLMGatedSAGE(nn.Module):
    def __init__(self, bow_dim, llm_dim, hid, nc, gate):
        super().__init__()
        self.gate_type = gate
        self.l1  = SAGELayer(bow_dim, hid)
        self.l2  = SAGELayer(hid, hid)
        self.clf = nn.Linear(hid, nc)
        if gate == "bilinear":
            self.U = nn.Linear(llm_dim, 32, bias=False)
            self.V = nn.Linear(llm_dim, 32, bias=False)

    def _beta(self, ei, llm_e):
        g = self.gate_type
        if g == "max_cosine": return gate_max_cosine(ei, llm_e)
        if g == "nbr_agree":  return gate_nbr_agree(ei, llm_e)
        if g == "topk":       return gate_topk(ei, llm_e)
        if g == "cos_std":    return gate_cos_std(ei, llm_e)
        if g == "bilinear":
            row, col = ei; N = llm_e.size(0)
            mu = scatter_mean(llm_e[col], row, dim=0, dim_size=N)
            return torch.sigmoid((self.U(llm_e) * self.V(mu)).sum(-1))

    def _apply(self, layer, x, ei, beta):
        row, col = ei; N = x.size(0)
        hs = layer.Ws(x)
        hn = layer.Wn(scatter_mean(x[col], row, dim=0, dim_size=N))
        return beta.unsqueeze(-1)*hs + (1-beta.unsqueeze(-1))*hn

    def forward(self, bow_x, ei, llm_e):
        b = self._beta(ei, llm_e)
        h = F.relu(self._apply(self.l1, bow_x, ei, b))
        b = self._beta(ei, llm_e)
        h = F.relu(self._apply(self.l2, h, ei, b))
        return self.clf(h)

# ── Training ──────────────────────────────────────────────────────────────────
def train_eval(gate, bow_x, ei, llm_e, labels, trm, vm, tem, nc, seed, epochs=150):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LLMGatedSAGE(bow_x.size(1), llm_e.size(1), 128, nc, gate)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=5e-4)
    bv = bt = 0.0
    for ep in range(epochs):
        m.train()
        F.cross_entropy(m(bow_x, ei, llm_e)[trm], labels[trm]).backward()
        opt.step(); opt.zero_grad()
        if (ep+1) % 10 == 0:
            m.eval()
            with torch.no_grad(): lg = m(bow_x, ei, llm_e)
            pred = lg.argmax(-1)
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
    ne = torch.tensor([src+dst, dst+src], dtype=torch.long)
    return torch.cat([ei, ne], 1)

# ── Config ────────────────────────────────────────────────────────────────────
NEW_GATES = ["max_cosine", "nbr_agree", "bilinear", "topk", "cos_std"]
LABEL = {
    "max_cosine": "E — LLM max cosine",
    "nbr_agree":  "F — neighbour agreement",
    "bilinear":   "G — LLM bilinear",
    "topk":       "H — LLM top-k cosine",
    "cos_std":    "I — cosine std",
}
SEEDS  = 2
EPOCHS = 150
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
PL_ROOT = "/tmp/planetoid"
CORA_PT = os.path.join(_root, "STEM-GNN/dataset/data/single_graph/Cora/cora.pt")
PUB_PT  = os.path.join(_root, "STEM-GNN/dataset/data/single_graph/Pubmed/pubmed.pt")

# Load existing results
EXISTING = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(EXISTING) as f:
    all_results = json.load(f)

# ── Dataset loop ──────────────────────────────────────────────────────────────
for ds_name, bow_name, raw_pt, sweep_key in [
    ("Cora",   "Cora",   CORA_PT, "cora_sweep"),
    ("PubMed", "PubMed", PUB_PT,  "pubmed_sweep"),
]:
    print(f"\n{'='*55}")
    print(f"  {ds_name}  —  new gates E-I")
    print(f"{'='*55}")

    pl_ds   = Planetoid(PL_ROOT, bow_name)
    pl_data = pl_ds[0]
    bow_x   = F.normalize(pl_data.x.float(), dim=-1)
    labels  = pl_data.y.long()
    nc      = int(labels.max().item()) + 1
    ei_base = to_undirected(pl_data.edge_index, num_nodes=pl_data.num_nodes)
    trm     = pl_data.train_mask
    vm      = pl_data.val_mask
    tem     = pl_data.test_mask

    print(f"  Encoding {ds_name} texts...", flush=True)
    raw_data = torch.load(raw_pt)
    llm_e    = encode_texts(raw_data.raw_texts)
    print(f"  BOW {bow_x.shape}  LLM {llm_e.shape}", flush=True)

    for gate in NEW_GATES:
        res_gate = []
        for r in RATIOS:
            ei  = inject(ei_base, llm_e, labels, r)
            acc = float(np.mean([
                train_eval(gate, bow_x, ei, llm_e, labels,
                           trm, vm, tem, nc, s, EPOCHS)
                for s in range(SEEDS)
            ]))
            res_gate.append(round(acc, 4))
            print(f"  {LABEL[gate]:<28} ratio={r:.0%}  {acc*100:.2f}%", flush=True)
        all_results[sweep_key]["results"][gate] = res_gate

# ── Save merged results ───────────────────────────────────────────────────────
OUT = "/home/lam23005/STEM-GNN/results/llm_gates_results.json"
with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n=== Done. Saved → {OUT} ===")
