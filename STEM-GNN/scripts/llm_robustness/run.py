"""LLM-as-robustness-signal experiment (experiments/llm-robustness/protocol.md).

The LLM never predicts: it supplies (L1) per-edge class-agreement trust and
(L2) structure-free pseudo-labels. Prediction is always a GNN.

Usage (from repo root):
  python STEM-GNN/scripts/llm_robustness/run.py --splits 0 --threads 2
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(SCRIPTS, "adaptive_robustness"))
sys.path.insert(0, os.path.join(SCRIPTS, "stem_vq_robustness"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load("probe", os.path.join(SCRIPTS, "adaptive_robustness", "run.py"))
stem = _load("stem_run", os.path.join(SCRIPTS, "stem_vq_robustness", "run.py"))
from models import SAGE  # noqa: E402
from stem_lite import STEMLite, commitment_loss  # noqa: E402

log = logging.getLogger("llm")
ROOT = os.path.join(SCRIPTS, "..", "..")
OUT_DIR = os.path.join(ROOT, "experiments", "llm-robustness", "results")
CATS = os.path.join(SCRIPTS, "..", "dataset", "data", "single_graph", "Cora", "categories.csv")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
N_PSEUDO, PSEUDO_W = 30, 0.5
SHARP_T = 0.02  # exploratory sharpened trust prior (diagnostic: separates camouflage edges)
COMMIT_W = 0.0  # repo's node finetune discards the VQ loss (ft_model.py:142, task/node.py:74)


# ------------------------------------------------------------ LLM signals
def class_embeddings(label_names):
    cache = os.path.join(OUT_DIR, "class_emb_llm_desc.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    enc = AutoModel.from_pretrained(ENCODER, local_files_only=True).eval()
    cats = pd.read_csv(CATS).values
    texts = [next(r[1] for r in cats if r[0] == ln) for ln in label_names]
    b = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    emb = F.normalize((h * m).sum(1) / m.sum(1), dim=-1)
    torch.save(emb, cache)
    return emb


def zero_shot_prior(x, class_emb, y, va):
    """softmax(cos/T), T picked by validation NLL (labels used only for T)."""
    sims = F.normalize(x, dim=-1) @ class_emb.t()
    best = min((F.nll_loss(F.log_softmax(sims[va] / t, -1), y[va]).item(), t) for t in (0.02, 0.05, 0.1))
    return F.softmax(sims / best[1], -1), best[1]


def edge_trust(prior, edge_index):
    """Bhattacharyya coefficient of endpoint class priors, in [0, 1]."""
    root = prior.sqrt()
    return (root[edge_index[0]] * root[edge_index[1]]).sum(-1)


def pseudo_labels(prior, tr):
    top2 = prior.topk(2, -1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).masked_fill(tr, -1.0)
    pred = top2.indices[:, 0]
    idx = torch.cat([torch.nonzero(pred == k).view(-1)[margin[pred == k].argsort(descending=True)[:N_PSEUDO]]
                     for k in range(prior.size(1))])
    return idx, pred[idx]


# ---------------------------------------------------------------- models
class LLMTrustSAGE(SAGE):
    def __init__(self, *a, prior=None, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("prior", prior)

    def _coeff_mass(self, x, edge_index, edge_weight):
        w = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1))
        return w * edge_trust(self.prior, edge_index), None


class LLMTrustSTEM(STEMLite):
    def __init__(self, *a, prior=None, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("prior", prior)

    def encode(self, x, edge_index, edge_weight=None):
        w = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1))
        return super().encode(x, edge_index, w * edge_trust(self.prior, edge_index))


# -------------------------------------------------------------- training
def sup_loss(logits, y, tr, pseudo):
    loss = F.cross_entropy(logits[tr], y[tr])
    if pseudo is not None:
        loss = loss + PSEUDO_W * F.cross_entropy(logits[pseudo[0]], pseudo[1])
    return loss


def train_sage(trust, pseudo, x, ei, y, tr, va, split, prior):
    torch.manual_seed(split)
    if trust == "textgate":
        model = SAGE(x.size(1), probe.HID, int(y.max()) + 1, mode="text_gate",
                     tau=probe.edge_cos_quantile(x, ei, 0.1))
    elif trust:
        model = LLMTrustSAGE(x.size(1), probe.HID, int(y.max()) + 1, prior=prior[trust == "sharp"])
    else:
        model = SAGE(x.size(1), probe.HID, int(y.max()) + 1)
    stem.fit(model, lambda: sup_loss(model(x, ei), y, tr, pseudo), x, ei, y, tr, va, probe.EPOCHS)
    return model


def train_stem(trust, pseudo, cons, x, ei, y, tr, va, split, prior):
    torch.manual_seed(split)
    model = (LLMTrustSTEM(x.size(1), int(y.max()) + 1, prior=prior[trust == "sharp"]) if trust
             else STEMLite(x.size(1), int(y.max()) + 1))
    model.vq_mode = "none"
    stem.fit(model, lambda: sup_loss(model(x, ei), y, tr, pseudo), x, ei, y, tr, va, stem.STAGE1_EPOCHS)
    model.init_codebook(x, ei, seed=split)
    model.vq_mode = "ste"

    def loss():
        logits, zh, sims, idx = model.forward_full(x, ei)
        total = sup_loss(logits, y, tr, pseudo) + COMMIT_W * commitment_loss(zh, sims, idx, model)
        return total + (cons * stem.dropedge_cons(model, x, ei) if cons > 0 else 0.0)

    stem.fit(model, loss, x, ei, y, tr, va, stem.STAGE3_EPOCHS)
    model.vq_mode = "hard"
    return model


VARIANTS = {  # name: (backbone, L1 trust, L2 pseudo, DropEdgeCons coef)
    "SAGE": ("sage", False, False, 0.0),
    "SAGE+TextGate(raw-cos)": ("sage", "textgate", False, 0.0),
    "SAGE+L1-LLMTrust": ("sage", True, False, 0.0),
    "SAGE+L1-LLMTrust-sharp[EXPL]": ("sage", "sharp", False, 0.0),
    "SAGE+L2-LLMPseudo": ("sage", False, True, 0.0),
    "SAGE+L3-Both": ("sage", True, True, 0.0),
    "STEM-lite": ("stem", False, False, 0.0),
    "STEM-lite+L1-LLMTrust": ("stem", True, False, 0.0),
    "STEM-lite+L1-LLMTrust-sharp[EXPL]": ("stem", "sharp", False, 0.0),
    "STEM-lite+L2-LLMPseudo": ("stem", False, True, 0.0),
    "STEM-lite+L3-Both": ("stem", True, True, 0.0),
    "STEM-lite+L4-Both+DropEdgeCons": ("stem", True, True, 1.0),
    "STEM-lite+L4-sharp+DropEdgeCons[EXPL]": ("stem", "sharp", True, 1.0),
}


# ------------------------------------------------------------ evaluation
def attack_row(model, is_stem, x, ei, y, te, test_idx, budgets, gen):
    row = {"clean": probe.accuracy(model, x, ei, y, te)}
    for r, b in budgets.items():
        row[f"random_add@{r}"] = probe.accuracy(model, x, probe.random_add(ei, x.size(0), b, gen), y, te)
        cam = probe.camouflage(x, ei, y, test_idx, b, gen)
        row[f"camouflage@{r}"] = probe.accuracy(model, x, cam, y, te)
        row[f"camouflage_edits@{r}"] = (cam.size(1) - ei.size(1)) // 2
        if is_stem:
            model.vq_mode = "ste"
        pert = probe.prbcd(model, x, ei, y, test_idx, b)
        if is_stem:
            model.vq_mode = "hard"
        row[f"prbcd@{r}"] = probe.accuracy(model, x, pert, y, te)
    return row


def run_split(split, rates, names):
    x, ei, y, trm, vam, tem = probe.load_cora()
    tr, va, te = trm[split], vam[split], tem[split]
    test_idx = te.nonzero().view(-1)
    d = torch.load(probe.CORA, weights_only=False)
    class_emb = class_embeddings(list(d.label_names))
    prior, temp = zero_shot_prior(x, class_emb, y, va)
    sharp = F.softmax(F.normalize(x, dim=-1) @ class_emb.t() / SHARP_T, -1)
    trust_priors = {False: prior, True: sharp}  # indexed by (trust == "sharp")
    pidx, plab = pseudo_labels(prior, tr)
    meta = {"split": split, "T": temp, "prior_acc_test": (prior.argmax(-1)[te] == y[te]).float().mean().item(),
            "pseudo_n": int(pidx.numel()), "pseudo_acc": (plab == y[pidx]).float().mean().item(),
            "edge_trust_clean_mean": edge_trust(prior, ei).mean().item()}
    log.info(json.dumps(meta))
    budgets = {r: int(r * ei.size(1) // 2) for r in rates}
    rows = []
    for name in names:
        backbone, trust, use_pseudo, cons = VARIANTS[name]
        pseudo = (pidx, plab) if use_pseudo else None
        t0 = time.time()
        if backbone == "sage":
            model = train_sage(trust, pseudo, x, ei, y, tr, va, split, trust_priors)
        else:
            model = train_stem(trust, pseudo, cons, x, ei, y, tr, va, split, trust_priors)
        gen = torch.Generator().manual_seed(1000 + split)
        row = {"model": name, "split": split}
        row.update(attack_row(model, backbone == "stem", x, ei, y, te, test_idx, budgets, gen))
        row["seconds"] = round(time.time() - t0, 1)
        log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        rows.append(row)
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--models", nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no_attack", action="store_true", help="clean accuracy only (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    rates = [] if args.no_attack else args.rates
    for s in args.splits:
        meta, rows = run_split(s, rates, args.models)
        if not args.no_attack:
            with open(os.path.join(OUT_DIR, f"split{s}.json"), "w") as f:
                json.dump({"args": vars(args), "meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
