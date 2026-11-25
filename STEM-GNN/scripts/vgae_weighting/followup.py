"""Follow-up to the structure-only VGAE weighting experiment (experiments/vgae-followup/protocol.md).

Four questions; the added scorer-derived edge weights use structure only. RUNG's existing propagation
coefficients also depend on predictions from node features:
  1. Does S-VG's adaptive gain survive a stronger, weight-aware attack?   (Cora, reuses Part A checkpoints)
       longer PRBCD with restarts on every attacked arm, plus three attacks on the weights of weighted arms:
       bpda_flat (forward = true weights, backward = uniform weights), detach (no gradient through the
       scorer), twin (attack the same trained model with uniform weights, transfer the graph).
  2. Do VGAE weights add robustness inside RUNG-lite (RUNG-VG)?
  3. Does a noise-aware VGAE help under random injection (S-VGm masked-edge VGAE, S-VGp self-pruned VGAE)?
  4. Does it replicate on Pubmed?

Usage (from repo root):
  python STEM-GNN/scripts/vgae_weighting/followup.py --selfcheck
  python STEM-GNN/scripts/vgae_weighting/followup.py --dataset cora --part A --split 0 --seed 0
  python STEM-GNN/scripts/vgae_weighting/followup.py --dataset pubmed --part R --level_idx 4 --graph 3
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
import os
import platform
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, negative_sampling, scatter, to_undirected

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vgae_weighting", os.path.join(HERE, "run.py"))
vw = importlib.util.module_from_spec(_spec)
sys.modules["vgae_weighting"] = vw
_spec.loader.exec_module(vw)
s2, ls, cr, fx, arm_models = vw.s2, vw.ls, vw.cr, vw.fx, vw.arm_models
sys.modules.setdefault("vgae_followup", sys.modules[__name__])
from torch_geometric.contrib.nn import GRBCDAttack, PRBCDAttack  # noqa: E402

log = logging.getLogger("vgae_followup")
ROOT = os.path.join(HERE, "..", "..", "..")
OUT = os.path.join(ROOT, "experiments", "vgae-followup", "results")
OLD = os.path.join(ROOT, "experiments", "vgae-weighting", "results")
OLD_TAG = "3405b67aa1"
DATA = {"cora": fx.CORA, "pubmed": os.path.join(os.path.dirname(fx.CORA), "..", "Pubmed", "pubmed.pt")}
CORA_UNDIRECTED = 5278
A_CELLS = {"cora": (3, 5), "pubmed": (3, 3)}  # (splits, seeds)
TAU, EPS = vw.TAU, vw.EPS
MASK_P = 0.5  # S-VGm: fraction of undirected edges hidden from the encoder each epoch (MaskGAE-style)
PRUNE = 0.3  # S-VGp: fraction of lowest-scoring undirected edges removed before the second VGAE fit
RUNG_KW = {"lam": 0.3, "delta": 0.3, "gamma": 1.2}  # RUNG-lite settings of the original experiment
# strong PRBCD on every attacked arm: name, seed offset, backward mode, loss, learning-rate multiplier
STRONG = (("s_lr1a", 61, "ste", "prob_margin", 1.0), ("s_lr1b", 62, "ste", "prob_margin", 1.0),
          ("s_lr01", 63, "ste", "prob_margin", 0.1), ("s_lr10a", 64, "ste", "prob_margin", 10.0),
          ("s_lr10b", 65, "ste", "prob_margin", 10.0), ("s_soft", 67, "soft", "prob_margin", 1.0),
          ("s_tanh", 71, "ste", "tanh_margin", 1.0), ("greedy", 89, "ste", "prob_margin", None))
WEIGHT_ATTACKS = (("bpda_flat", 73), ("detach", 79), ("twin", 83))
STRONG_EPOCHS, STRONG_RESAMPLE, STRONG_BLOCK, PRBCD_LR, GREEDY_EPOCHS = 400, 300, 200_000, 1_000, 125
STD_NAMES = tuple(a[0] for a in ls.ATTACKS)
ATTACKED = ("S0", "S-VG", "SAGE", "SAGE-VG", "RUNG-lite", "RUNG-VG")
OLD_ARMS = ("S0", "S-VG", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "MLP-M")  # shared with the original Part A
WEIGHTED = ("S-VG", "SAGE-VG", "RUNG-VG")
A_ARMS = ("S0", "S-VG", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "RUNG-VG", "MLP-M")
R_ARMS = ("S0", "S-VG", "S-VGm", "S-VGp", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "RUNG-VG", "MLP-M")


# ------------------------------------------------------------------ data
def load_data(ds):
    if ds == "cora":
        return fx.load()
    d = torch.load(DATA[ds], weights_only=False)
    ei = coalesce(to_undirected(d.edge_index))
    return d.x.float(), ei[:, ei[0] != ei[1]], d.y.long(), d.train_masks, d.val_masks, d.test_masks


def level_counts(ei):
    """Cora's milestone counts; other graphs get the same fractions of their undirected edge count."""
    e = len(cr.undirected(ei))
    if e == CORA_UNDIRECTED:
        return vw.LEVELS
    return tuple(int(round(lvl / CORA_UNDIRECTED * e)) for lvl in vw.LEVELS)


# ------------------------------------------------------------------ structure-only scorers
def sparse_jaccard(ei, n):
    """Exact unweighted structural Jaccard per edge, in edge order, without n x n matrices. Common-neighbor
    counts and degrees are exact integers, so the float32 result equals vw.JaccardScorer's bit for bit."""
    import numpy as np
    import scipy.sparse as sp
    e = ei.cpu().numpy()
    adj = sp.csr_matrix((np.ones(e.shape[1], dtype=np.float64), (e[0], e[1])), shape=(n, n))
    common = torch.from_numpy(np.asarray((adj @ adj)[e[0], e[1]]).ravel()).float()
    deg = scatter(torch.ones(ei.size(1)), ei[0], dim=0, dim_size=n, reduce="sum")
    return common / (deg[ei[0]] + deg[ei[1]] - common).clamp_min(1e-6)


class CachedJaccard(vw.JaccardScorer):
    """vw.JaccardScorer computed sparsely and memoised for unweighted calls (no parameters). Weighted calls
    (an attack through S-SJ, not planned) fall back to the dense original. CPU tensors assumed."""

    def __init__(self, n):
        super().__init__(n)
        self._key, self._val = None, None

    def scores(self, ei, w=None):
        if w is not None:
            return super().scores(ei, w)
        key = hashlib.sha1(ei.contiguous().cpu().numpy().tobytes()).hexdigest()
        if key != self._key:
            self._val, self._key = sparse_jaccard(ei, self.n), key
        return self._val


class FixedScorer(nn.Module):
    """Cosine of fixed posterior means (S-VGp): the score of an edge does not depend on the graph."""

    def __init__(self, mu):
        super().__init__()
        self.register_buffer("mu", mu)

    def scores(self, ei, w=None):
        return F.cosine_similarity(self.mu[ei[0]], self.mu[ei[1]], dim=-1)


def halves(ei):
    return ei[:, ei[0] < ei[1]]


def both(und):
    return torch.cat([und, und.flip(0)], 1)


def train_vgae_masked(n, ei, seed, p=MASK_P, epochs=vw.VGAE_EPOCHS):
    """The encoder sees a random (1 - p) of the edges and must reconstruct the hidden ones, so an edge the
    rest of the structure does not predict (e.g. a random one) scores low. Inference encodes the full graph."""
    fx.seed_all(seed)
    model = vw.StructVGAE(n)
    opt = torch.optim.Adam(model.parameters(), lr=vw.VGAE_LR)
    und = halves(ei)
    for epoch in range(epochs):
        model.train()
        fx.seed_all(seed * 1_009 + epoch)
        hide = torch.rand(und.size(1)) < p
        if not hide.any() or hide.all():
            continue
        vis, tgt = both(und[:, ~hide]), both(und[:, hide])
        mu, logstd = model.encode(vis)
        z = mu + torch.randn_like(mu) * logstd.exp()
        neg = negative_sampling(ei, n, num_neg_samples=tgt.size(1))
        pos_logit = (z[tgt[0]] * z[tgt[1]]).sum(-1)
        neg_logit = (z[neg[0]] * z[neg[1]]).sum(-1)
        recon = F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit)) + \
            F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
        kl = -0.5 / n * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))
        opt.zero_grad()
        (recon + kl).backward()
        opt.step()
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


def prune_graph(ei, vg, frac=PRUNE):
    """Drop the lowest-scoring frac of undirected edges under vg (computed on the full graph)."""
    und = halves(ei)
    with torch.no_grad():
        mu, _ = vg.encode(ei)
        s = F.cosine_similarity(mu[und[0]], mu[und[1]], dim=-1)
    order = torch.sort(s, descending=True, stable=True).indices
    keep = order[: und.size(1) - int(frac * und.size(1))].sort().values
    return both(und[:, keep])


def pruned_scorer(n, ei, vg, seed):
    pruned = prune_graph(ei, vg)
    vg2 = vw.train_vgae(n, pruned, seed)
    with torch.no_grad():
        mu, _ = vg2.encode(pruned)
    return FixedScorer(mu)


# ------------------------------------------------------------------ models
class WeightedRUNG(arm_models.RUNGLite):
    """RUNG-lite whose input edge weights are the VGAE neighbor weights, rescaled so each target's weights
    keep their total (uniform scores give back RUNG-lite exactly). Target = edge_index[1], as in RUNG-lite."""

    def __init__(self, in_dim, hid, out_dim, scorer, tau, eps, **kw):
        super().__init__(in_dim, hid, out_dim, **kw)
        self.scorer, self.tau, self.eps = scorer, tau, eps

    def forward(self, x, edge_index, edge_weight=None):
        w = torch.ones(edge_index.size(1)) if edge_weight is None else edge_weight
        s = self.scorer.scores(edge_index, edge_weight)
        a = vw.neighbor_weights(s, edge_index, w, self.tau, self.eps, tgt=1)
        total = scatter(w, edge_index[1], dim=0, dim_size=x.size(0), reduce="sum")[edge_index[1]]
        return super().forward(x, edge_index, a * total)


for _cls in (CachedJaccard, FixedScorer, WeightedRUNG):
    _cls.__module__ = "vgae_followup"


def twin(model):
    """The same trained weighted model with uniform neighbor weights (tau -> infinity)."""
    t = copy.deepcopy(model)
    t.tau = 1e6
    return t


@contextlib.contextmanager
def weight_grad(kind):
    """Change only the backward pass of the neighbor weights; the forward values stay the true weights."""
    orig = vw.neighbor_weights

    def bpda_flat(s, ei, w, tau, eps, tgt):
        true = orig(s.detach(), ei, w, tau, eps, tgt)
        flat = orig(torch.zeros_like(s), ei, w, tau, eps, tgt)  # = w / sum(w): the unweighted mean
        return flat + (true - flat).detach()

    def detach(s, ei, w, tau, eps, tgt):
        return orig(s.detach(), ei, w, tau, eps, tgt)

    vw.neighbor_weights = {"bpda_flat": bpda_flat, "detach": detach}[kind]
    try:
        yield
    finally:
        vw.neighbor_weights = orig


def prbcd_strong(model, x, ei, y, test_idx, budget, seed, mode, loss, lr_mult=1.0, epochs=None):
    """Strong PRBCD (lr_mult None = greedy GRBCD). The model's mode and train/eval state are restored."""
    was_training = model.training
    prev_mode = getattr(model, "attack_mode", None)
    model.eval()
    if prev_mode is not None:
        model.attack_mode = mode
    try:
        fx.seed_all(seed)
        if lr_mult is None:
            atk = GRBCDAttack(model, block_size=STRONG_BLOCK, epochs=epochs or GREEDY_EPOCHS, loss=loss, log=False)
        else:
            atk = PRBCDAttack(model, block_size=STRONG_BLOCK, epochs=epochs or STRONG_EPOCHS,
                              epochs_resampling=(epochs or STRONG_EPOCHS) * STRONG_RESAMPLE // STRONG_EPOCHS,
                              loss=loss, lr=PRBCD_LR * lr_mult, log=False)
        pert, _ = atk.attack(x, ei, y, budget=budget, idx_attack=test_idx)
    finally:
        if prev_mode is not None:
            model.attack_mode = prev_mode
        model.train(was_training)
    return pert


def weight_attack(attack, model, x, ei, y, test_idx, budget, seed, epochs=None):
    if attack == "twin":
        return prbcd_strong(twin(model), x, ei, y, test_idx, budget, seed, "ste", "prob_margin", epochs=epochs)
    with weight_grad(attack):
        return prbcd_strong(model, x, ei, y, test_idx, budget, seed, "ste", "prob_margin", epochs=epochs)


# ------------------------------------------------------------------ training
def train_ext(x, ei, y, tr, va, base, arms):
    """vw.train_all's recipe and seeds, plus S-VGm, S-VGp and RUNG-VG. Every step seeds itself, so shared
    arms reproduce vw.train_all exactly."""
    n, n_cls = x.size(0), int(y.max()) + 1
    enc, vq = s2.pretrain(x, ei, ls.PT_EPOCHS, base)
    vg = vw.train_vgae(n, ei, base + 900)
    scorers = {"S-VG": vg}
    if "S-SJ" in arms:
        scorers["S-SJ"] = CachedJaccard(n)
    if "S-VGm" in arms:
        scorers["S-VGm"] = train_vgae_masked(n, ei, base + 902)
    if "S-VGp" in arms:
        scorers["S-VGp"] = pruned_scorer(n, ei, vg, base + 901)
    models, meta = {}, {}
    models["S0"], val = vw.finetune(lambda: vw.stem_base(enc, vq, n_cls), x, ei, y, tr, va, base)
    meta["S0"] = {"val": val}
    for arm in ("S-VG", "S-SJ", "S-VGm", "S-VGp"):
        if arm in arms:
            models[arm], val = vw.finetune(lambda: vw.WeightedSTEM(vw.stem_base(enc, vq, n_cls), scorers[arm], TAU, EPS),
                                           x, ei, y, tr, va, base)
            meta[arm] = {"val": val, "tau": TAU, "eps": EPS}
    plain = {"SAGE": lambda: arm_models.SAGE(x.size(1), vw.BASE_HID, n_cls),
             "GCN": lambda: arm_models.GCN(x.size(1), vw.BASE_HID, n_cls),
             "RUNG-lite": lambda: arm_models.RUNGLite(x.size(1), vw.BASE_HID, n_cls, **RUNG_KW)}
    for k, (arm, build) in enumerate(plain.items()):
        if arm in arms:
            models[arm], val = vw.fit_plain(build, x, ei, y, tr, va, base + 700 + k)
            meta[arm] = {"val": val}
    if "SAGE-VG" in arms:
        models["SAGE-VG"], val = vw.fit_plain(lambda: vw.WeightedSAGE(x.size(1), vw.BASE_HID, n_cls, vg, TAU, EPS),
                                              x, ei, y, tr, va, base + 700)
        meta["SAGE-VG"] = {"val": val, "tau": TAU, "eps": EPS}
    if "RUNG-VG" in arms:
        models["RUNG-VG"], meta["RUNG-VG"] = fit_rung_vg(x, ei, y, tr, va, base, vg)
    if "MLP-M" in arms:
        models["MLP-M"] = s2.fit_mlp(x, y, tr, va, base)
        with torch.no_grad():
            meta["MLP-M"] = {"val": fx.acc(models["MLP-M"](x), y, va)}
    return models, meta, scorers


def fit_rung_vg(x, ei, y, tr, va, base, vg):
    """Same seed as RUNG-lite (base + 702), so the two differ only in the neighbor weights."""
    n_cls = int(y.max()) + 1
    model, val = vw.fit_plain(lambda: WeightedRUNG(x.size(1), vw.BASE_HID, n_cls, vg, TAU, EPS, **RUNG_KW),
                              x, ei, y, tr, va, base + 702)
    return model, {"val": val, "tau": TAU, "eps": EPS}


# ------------------------------------------------------------------ part R: random injection, poisoning
def run_r(li, graph, x, ei, y, trm, vam, tem):
    split = graph % 3
    level = level_counts(ei)[li]
    base = 100_000 * (li + 1) + 1000 * graph + split  # = vw.run_r's seed on Cora
    tr, va, te = trm[split], vam[split], tem[split]
    noisy, injected = vw.inject(ei, x.size(0), level, 50_000 + 1000 * graph + level)
    models, meta, scorers = train_ext(x, noisy, y, tr, va, base, R_ARMS)
    n_cls, n_orig = int(y.max()) + 1, len(cr.undirected(ei))
    rows = []
    with torch.no_grad():
        edge_auc = {}
        for k, sc in scorers.items():
            s = sc.scores(noisy)
            edge_auc[k] = vw.auc(s[~injected], s[injected])
        for arm, model in models.items():
            pred = (model(x) if arm == "MLP-M" else model(x, noisy)).argmax(-1)
            rows.append({"part": "R", "level_idx": li, "level": level, "graph": graph, "split": split, "arm": arm,
                         **meta[arm], "orig_undirected_edges": n_orig, "injected_pct": 100 * level / n_orig,
                         "acc": (pred == y)[te].float().mean().item(), "macro_f1": vw.macro_f1(pred, y, te, n_cls)})
            log.info(json.dumps(rows[-1]))
    return rows, edge_auc


# ------------------------------------------------------------------ part A: adaptive evasion
def old_tag(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


def load_old(split, seed):
    """An original Part A checkpoint, verified against its tag, the current Cora file and the current code."""
    path = os.path.join(OLD, f"A_split{split}_seed{seed}_{OLD_TAG}_ckpt.pt")
    state = torch.load(path, weights_only=False)
    cfg = state["config"]
    if old_tag(cfg) != OLD_TAG:
        raise RuntimeError(f"{path}: configuration tag {old_tag(cfg)} != {OLD_TAG}")
    if cfg["data_sha"] != hashlib.sha256(open(fx.CORA, "rb").read()).hexdigest():
        raise RuntimeError(f"{path}: the Cora file changed since the original run")
    if cfg["code_hash"] != vw.code_hash():
        raise RuntimeError(f"{path}: the original code or its dependencies changed since the original run")
    return state


def jobs_for(models, sur):
    """(target, attack name, kind, spec). 'gcn' is a strong transfer attack from the independently trained GCN."""
    jobs = [("sage", "sage", "std", (sur, 23, "ste", "prob_margin"))]
    jobs += [(arm, name, "std", (models[arm], off, mode, loss)) for arm in ATTACKED for name, off, mode, loss in ls.ATTACKS]
    jobs += [("gcn", "gcn", "strong", (models["GCN"], 59, "ste", "prob_margin", 1.0))]
    jobs += [(arm, name, "strong", (models[arm], off, mode, loss, lr))
             for arm in ATTACKED for name, off, mode, loss, lr in STRONG]
    jobs += [(arm, name, "weight", (models[arm], off)) for arm in WEIGHTED for name, off in WEIGHT_ATTACKS]
    return jobs


def run_one(kind, attack, spec, x, ei, y, test_idx, budget, seed_base):
    if kind == "std":
        model, off, mode, loss = spec
        return ls.prbcd(model, x, ei, y, test_idx, budget, seed_base + off, mode, loss)
    if kind == "strong":
        model, off, mode, loss, lr = spec
        return prbcd_strong(model, x, ei, y, test_idx, budget, seed_base + off, mode, loss, lr)
    model, off = spec
    return weight_attack(attack, model, x, ei, y, test_idx, budget, seed_base + off)


def key_name(k):
    return "|".join(map(str, k))


def save_atomic(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run_a(ds, split, seed, x, ei, y, trm, vam, tem, ckpt, config):
    """Models are saved once; each attacked graph is validated, then saved atomically in its own file."""
    base = 1000 * seed + split
    tr, va, te = trm[split], vam[split], tem[split]
    test_idx = te.nonzero().view(-1)
    n = x.size(0)
    budgets = {r: int(r * len(cr.undirected(ei))) for r in ls.RATES}
    gdir = ckpt[:-3] + "_graphs"
    gpath = lambda k: os.path.join(gdir, key_name(k).replace("|", "__") + ".pt")
    state = torch.load(ckpt, weights_only=False) if os.path.exists(ckpt) else None
    if state is not None and state["config"] != config:
        raise RuntimeError(f"{ckpt} was made with a different configuration; refusing to resume")
    if state is None:
        t0 = time.time()
        old_graphs, old_cfg = {}, None
        if ds == "cora":
            old = load_old(split, seed)
            models = {k: old["models"][k] for k in OLD_ARMS}
            meta = {k: old["meta"][k] for k in OLD_ARMS}
            sur, old_graphs, old_cfg = old["sur"], dict(old["graphs"]), old["config"]
            models["RUNG-VG"], meta["RUNG-VG"] = fit_rung_vg(x, ei, y, tr, va, base, models["S-VG"].scorer)
        else:
            models, meta, _ = train_ext(x, ei, y, tr, va, base, A_ARMS)
            fx.seed_all(base + 11)
            sur = cr.Surrogate(x.size(1), int(y.max()) + 1)
            fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
        os.makedirs(gdir, exist_ok=True)
        stale = [f for f in os.listdir(gdir) if f.endswith(".pt")]
        if stale:
            raise RuntimeError(f"{gdir} has graph files but no checkpoint; refusing to reuse them: {stale[:3]}")
        state = {"config": config, "models": models, "meta": meta, "sur": sur, "old_keys": sorted(old_graphs),
                 "old_config": old_cfg, "train_seconds": round(time.time() - t0, 1),
                 "checkpoint_id": f"{time.time_ns()}-{os.getpid()}"}
        save_atomic(ckpt, state)  # the identity exists before any graph file is written
        log.info(json.dumps({"trained": sorted(models), "seconds": state["train_seconds"]}))
    models, meta, sur, old_keys = state["models"], state["meta"], state["sur"], set(state["old_keys"])
    missing_old = [k for k in sorted(old_keys) if not os.path.exists(gpath(k))]
    if missing_old:  # first run, or an interruption during the import: original graphs are imported, never re-attacked
        old_graphs = load_old(split, seed)["graphs"]
        for k in missing_old:
            g = old_graphs[k]
            save_atomic(gpath(k), {"key": k, "config": config, "checkpoint_id": state["checkpoint_id"], "g": g,
                                   "seconds": None, "edits": cr.validate(ei, g, n, budgets[k[2]])})
    planned = {(tgt, name, r) for r in budgets for tgt, name, _, _ in jobs_for(models, sur)}
    unexpected = {f for f in os.listdir(gdir) if f.endswith(".pt")} - {os.path.basename(gpath(k)) for k in planned}
    if unexpected:
        raise RuntimeError(f"unexpected graph files in {gdir}: {sorted(unexpected)[:3]}")
    graphs, kinds, secs, edits = {}, {}, {}, {}
    for r, b in budgets.items():
        for tgt, name, kind, spec in jobs_for(models, sur):
            k = (tgt, name, r)
            kinds[k] = kind
            if os.path.exists(gpath(k)):
                d = torch.load(gpath(k), weights_only=False)
                ok = (d.get("key") == k and d.get("config") == config and d.get("checkpoint_id") == state["checkpoint_id"]
                      and cr.validate(ei, d["g"], n, b) == d["edits"])
                if not ok:
                    raise RuntimeError(f"{gpath(k)} does not belong to this checkpoint or fails validation")
            else:
                t0 = time.time()
                g = run_one(kind, name, spec, x, ei, y, test_idx, b, base + int(100 * r))
                d = {"key": k, "config": config, "checkpoint_id": state["checkpoint_id"], "g": g,
                     "edits": cr.validate(ei, g, n, b), "seconds": round(time.time() - t0, 1)}
                save_atomic(gpath(k), d)
                log.info(json.dumps({"attack": [tgt, name, r], "kind": kind, "seconds": d["seconds"]}))
            graphs[k], secs[key_name(k)], edits[key_name(k)] = d["g"], d["seconds"], d["edits"]
    assert old_keys <= set(graphs), "an original graph is missing from the planned attack set"
    t0 = time.time()
    rows = [score_arm(arm, models[arm], meta, graphs, kinds, old_keys, x, ei, y, te, split, seed) for arm in A_ARMS]
    log.info(json.dumps({"scoring_seconds": round(time.time() - t0, 1)}))
    total = state["train_seconds"] + sum(v for v in secs.values() if v is not None)
    return rows, edits, secs, round(total, 1), state["old_config"]


def score_arm(arm, model, meta, graphs, kinds, old_keys, x, ei, y, te, split, seed):
    """own_* are None for arms with no attack targeted at them. old_* use only the original graphs and the
    original semantics (an empty own set gives clean accuracy), so they can be compared with the first run."""
    fwd = (lambda g: model(x)) if arm == "MLP-M" else (lambda g: model(x, g))
    with torch.no_grad():
        clean = fwd(ei).argmax(-1) == y
        corr = {k: fwd(g).argmax(-1) == y for k, g in graphs.items()}
    worst = lambda keys: torch.stack([clean] + [corr[k] for k in keys]).all(0)[te].float().mean().item()
    acc = {key_name(k): v[te].float().mean().item() for k, v in corr.items()}
    row = {"part": "A", "split": split, "seed": seed, "arm": arm, **meta[arm], "clean": clean[te].float().mean().item(),
           "acc_by_graph": acc}
    for r in ls.RATES:
        keys = [k for k in corr if k[2] <= r]
        own = [k for k in keys if k[0] == arm]
        row[f"worst_node@{r}"] = worst(keys)
        row[f"worst_graph@{r}"] = torch.stack([clean] + [corr[k] for k in keys])[:, te].float().mean(1).min().item()
        row[f"std_worst_node@{r}"] = worst([k for k in keys if kinds[k] == "std"])
        for label, sel in (("own_std", [k for k in own if kinds[k] == "std"]),
                           ("own_strong", [k for k in own if kinds[k] in ("std", "strong")]), ("own_all", own)):
            row[f"{label}@{r}"] = worst(sel) if sel else None
        row[f"transfer_sage@{r}"] = acc[f"sage|sage|{r}"]
        row[f"transfer_gcn@{r}"] = acc.get(f"gcn|gcn|{r}")
        if old_keys:
            ok = [k for k in keys if k in old_keys]
            row[f"old_worst_node@{r}"] = worst(ok)
            row[f"old_worst_graph@{r}"] = torch.stack([clean] + [corr[k] for k in ok])[:, te].float().mean(1).min().item()
            row[f"old_own_worst_node@{r}"] = worst([k for k in ok if k[0] == arm])
    log.info(json.dumps({k: v for k, v in row.items() if k != "acc_by_graph"}))
    return row


# ------------------------------------------------------------------ selfcheck
ORIG_NW = vw.neighbor_weights
SHORT = 10  # attack epochs in the selfcheck only (validity, not strength)


def check(name, ok, detail=""):
    print(name, "OK" if ok else "FAILED", detail, flush=True)
    assert ok, name


def edge_grad(model, x, ei, y, idx):
    ww = torch.ones(ei.size(1), requires_grad=True)
    p = F.softmax(model(x, ei, ww), -1)
    m = p[idx].gather(1, y[idx, None]).sum() - p[idx].max(-1).values.sum()
    return torch.autograd.grad(m, ww)[0]


def derivative_checks():
    """Hand-computed derivatives of target 0's first weight c1 on a 2-neighbor star, including a zero weight."""
    ei = torch.tensor([[0, 0], [1, 2]])
    s = torch.tensor([0.4, -0.3])
    tau, eps = TAU, EPS
    for w0 in ([0.7, 1.3], [0.0, 1.0]):
        w = torch.tensor(w0, requires_grad=True)
        e = torch.exp(s / tau)
        dd, ww_ = (w0[0] * e[0] + w0[1] * e[1]).item(), w0[0] + w0[1]
        flat = torch.tensor([w0[1] / ww_ ** 2, -w0[0] / ww_ ** 2])
        soft = torch.tensor([e[0].item() * w0[1] * e[1].item() / dd ** 2, -w0[0] * e[0].item() * e[1].item() / dd ** 2])
        expect = {"bpda_flat": flat, "detach": (1 - eps) * soft + eps * flat}
        true = ORIG_NW(s, ei, w, tau, eps, 0)
        for kind, want in expect.items():
            with weight_grad(kind):
                c = vw.neighbor_weights(s.clone().requires_grad_(True), ei, w, tau, eps, 0)
            got = torch.autograd.grad(c[0], w)[0]
            check(f"{kind} derivative matches hand computation (w={w0})", torch.allclose(got, want, atol=1e-5),
                  (got.tolist(), want.tolist()))
            check(f"{kind} forward equals the true weights (w={w0})", torch.allclose(c, true, atol=1e-6))
    check("weight function restored (identity)", vw.neighbor_weights is ORIG_NW)
    try:
        with weight_grad("detach"):
            raise ValueError
    except ValueError:
        pass
    check("weight function restored after an exception (identity)", vw.neighbor_weights is ORIG_NW)


def selfcheck():
    derivative_checks()
    px, pei, py, ptr, pva, pte = load_data("pubmed")
    und = cr.undirected(pei)
    check("pubmed: symmetric, no self-loops", bool((pei[0] != pei[1]).all()) and len(und) * 2 == pei.size(1), len(und))
    check("pubmed: 10 splits, disjoint masks", len(ptr) == 10 and all(
        not (a & b).any() and not (a & c).any() and not (b & c).any() for a, b, c in zip(ptr, pva, pte)))
    check("pubmed levels are Cora's fractions", level_counts(pei)[-1] == round(5000 / CORA_UNDIRECTED * len(und)),
          level_counts(pei))
    t0 = time.time()
    sj = CachedJaccard(px.size(0)).scores(pei)
    check("pubmed sparse Jaccard finite, in [0, 1]", bool(torch.isfinite(sj).all() and (sj >= 0).all() and (sj <= 1).all()),
          f"{time.time() - t0:.1f}s")
    del px, pei, py, sj
    x, ei, y, trm, vam, tem = load_data("cora")
    check("cora levels are the milestone counts", level_counts(ei) == vw.LEVELS)
    n, n_cls = x.size(0), int(y.max()) + 1
    tr, va, te = trm[0], vam[0], tem[0]
    noisy, injected = vw.inject(ei, n, 2000, 7)
    cj = CachedJaccard(n)
    with torch.no_grad():
        check("sparse Jaccard equals the dense original bit for bit, on two graphs",
              torch.equal(cj.scores(ei), vw.JaccardScorer(n).scores(ei))
              and torch.equal(cj.scores(noisy), vw.JaccardScorer(n).scores(noisy)))
    # the original checkpoint: provenance and exact reproduction of every reported metric
    old = load_old(0, 0)
    check("old checkpoint: tag, data and code provenance verified", True)
    with open(os.path.join(OLD, f"A_split0_seed0_{OLD_TAG}.json")) as f:
        old_rows = {r["arm"]: r for r in json.load(f)["rows"]}
    kinds = {k: "std" for k in old["graphs"]}
    worst_diff = 0.0
    for arm in OLD_ARMS:
        row = score_arm(arm, old["models"][arm], old["meta"], old["graphs"], kinds, set(old["graphs"]),
                        x, ei, y, te, 0, 0)
        o = old_rows[arm]
        for r in ls.RATES:
            for new_k, old_k in ((f"old_worst_node@{r}", f"worst_node@{r}"), (f"old_worst_graph@{r}", f"worst_graph@{r}"),
                                 (f"old_own_worst_node@{r}", f"own_worst_node@{r}"),
                                 (f"transfer_sage@{r}", f"transfer_sage@{r}")):
                worst_diff = max(worst_diff, abs(row[new_k] - o[old_k]))
        worst_diff = max(worst_diff, abs(row["clean"] - o["clean"]))
    check("old checkpoint reproduces every original Part A metric exactly (split 0, seed 0)", worst_diff == 0, worst_diff)
    vg = old["models"]["S-VG"].scorer
    # RUNG-VG
    rung = arm_models.RUNGLite(x.size(1), vw.BASE_HID, n_cls, **RUNG_KW).eval()
    flat = WeightedRUNG(x.size(1), vw.BASE_HID, n_cls, vg, 1e6, 0.0, **RUNG_KW).eval()
    flat.mlp.load_state_dict(rung.mlp.state_dict())
    with torch.no_grad():
        check("RUNG-VG with tau -> infinity matches RUNG-lite (tolerance)", torch.allclose(flat(x, ei), rung(x, ei), atol=1e-5))
    rvg = WeightedRUNG(x.size(1), vw.BASE_HID, n_cls, vg, TAU, EPS, **RUNG_KW).eval()
    seen = []
    real = arm_models.RUNGLite.forward
    arm_models.RUNGLite.forward = lambda self, xx, g, w=None: (seen.append(w.detach().clone()), real(self, xx, g, w))[1]
    try:
        with torch.no_grad():
            out_a = rvg(x, ei)
            out_b = rvg(x[torch.randperm(n)] + torch.randn_like(x), ei)
    finally:
        arm_models.RUNGLite.forward = real
    check("RUNG-VG input edge weights identical when the text changes", torch.equal(seen[0], seen[1]))
    check("RUNG-VG: text still reaches the model", not torch.equal(out_a, out_b))
    tot = scatter(seen[0], ei[1], dim=0, dim_size=n, reduce="sum")
    check("RUNG-VG weights keep each target's total",
          torch.allclose(tot, scatter(torch.ones(ei.size(1)), ei[1], dim=0, dim_size=n, reduce="sum"), atol=1e-4))
    g = edge_grad(rvg, x, ei, y, va)
    check("RUNG-VG edge gradient finite and nonzero", bool(torch.isfinite(g).all()) and g.norm() > 0)
    # weight-aware variants on every weighted model
    weighted = {"S-VG": old["models"]["S-VG"], "SAGE-VG": old["models"]["SAGE-VG"], "RUNG-VG": rvg}
    for arm, model in weighted.items():
        g_true = edge_grad(model, x, ei, y, va)
        with torch.no_grad():
            ref = model(x, ei)
        for kind in ("bpda_flat", "detach"):
            with weight_grad(kind):
                with torch.no_grad():
                    diff = (model(x, ei) - ref).abs().max().item()
                gk = edge_grad(model, x, ei, y, va)
            # bpda_flat's flat + (true - flat).detach() reorders float ops, so allow rounding error
            check(f"{arm} {kind}: forward unchanged (up to rounding)", diff < 1e-5, f"max |diff| = {diff:.2e}")
            check(f"{arm} {kind}: gradient finite, nonzero, different from the true gradient",
                  bool(torch.isfinite(gk).all()) and gk.norm() > 0 and not torch.allclose(gk, g_true),
                  f"|g| = {gk.norm().item():.3e}, |g_true| = {g_true.norm().item():.3e}, "
                  f"max |g - g_true| = {(gk - g_true).abs().max().item():.3e}")
        check(f"{arm}: weight function restored (identity)", vw.neighbor_weights is ORIG_NW)
        t = twin(model)
        with torch.no_grad():
            check(f"{arm}: twin is a separate copy; the model is unchanged",
                  not torch.equal(t(x, ei), ref) and torch.equal(model(x, ei), ref) and model.tau == TAU
                  and all(a.data_ptr() != b.data_ptr() for a, b in zip(t.parameters(), model.parameters())))
    s_vg = old["models"]["S-VG"]
    with torch.no_grad():
        check("S-VG twin matches its own base with uniform weights (tolerance)",
              torch.allclose(twin(s_vg)(x, ei), s_vg.base(x, ei), atol=1e-4))
    # noise-aware scorers
    vm = train_vgae_masked(n, noisy, 5, epochs=30)
    check("S-VGm takes no text input, frozen", not any(p.requires_grad for p in vm.parameters())
          and not any(x.size(1) in p.shape for p in vm.parameters()))
    with torch.no_grad():
        sm = vm.scores(noisy)
    check("S-VGm scores finite, AUC computed", bool(torch.isfinite(sm).all()), round(vw.auc(sm[~injected], sm[injected]), 3))
    vg_noisy = vw.train_vgae(n, noisy, 5, epochs=30)
    pr = prune_graph(noisy, vg_noisy)
    check("pruning removes 30% of undirected edges, stays symmetric",
          len(cr.undirected(pr)) == len(cr.undirected(noisy)) - int(PRUNE * len(cr.undirected(noisy)))
          and pr.size(1) == 2 * len(cr.undirected(pr)))
    vp = pruned_scorer(n, noisy, vg_noisy, 6)
    with torch.no_grad():
        sp = vp.scores(noisy)
    check("S-VGp scores finite, latent-only (no text)", bool(torch.isfinite(sp).all()) and vp.mu.size(1) == vw.VGAE_LAT,
          round(vw.auc(sp[~injected], sp[injected]), 3))
    # every attack type returns a valid graph (short settings) and restores the model's mode and state
    budget = int(0.1 * len(cr.undirected(ei)))
    te_idx = te.nonzero().view(-1)
    s_vg.attack_mode = "soft"
    pert = prbcd_strong(s_vg, x, ei, y, te_idx, budget, 4, "ste", "prob_margin", 10.0, epochs=SHORT)
    check("strong PRBCD (lr x10) valid; restores the previous mode and eval state",
          cr.validate(ei, pert, n, budget) is not None and s_vg.attack_mode == "soft" and not s_vg.training)
    s_vg.attack_mode = "ste"
    pert = prbcd_strong(rvg, x, ei, y, te_idx, budget, 4, "ste", "prob_margin", None, epochs=SHORT)
    check("greedy GRBCD on RUNG-VG returns a valid graph", cr.validate(ei, pert, n, budget) is not None,
          cr.validate(ei, pert, n, budget))
    for kind in ("bpda_flat", "detach", "twin"):
        pert = weight_attack(kind, s_vg, x, ei, y, te_idx, budget, 3, epochs=SHORT)
        check(f"{kind} attack on S-VG returns a valid graph", cr.validate(ei, pert, n, budget) is not None)
    check("neighbor weights untouched after attacks (identity)", vw.neighbor_weights is ORIG_NW)
    # pickling in a fresh process
    path = os.path.join(OUT, "_selfcheck.pt")
    os.makedirs(OUT, exist_ok=True)
    with torch.no_grad():
        torch.save({"m": rvg, "p": vp, "j": cj, "x": x, "ei": ei, "out": rvg(x, ei)}, path)
    import subprocess
    probe = ("import importlib.util,sys,torch;"
             f"spec=importlib.util.spec_from_file_location('vgae_followup',{os.path.abspath(__file__)!r});"
             "m=importlib.util.module_from_spec(spec);sys.modules['vgae_followup']=m;spec.loader.exec_module(m);"
             f"d=torch.load({path!r},weights_only=False);"
             "print(torch.allclose(d['m'](d['x'],d['ei']),d['out'],atol=1e-5))")
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    ok = res.stdout.strip().endswith("True")
    check("new models load in a fresh process", ok, "" if ok else res.stderr[-300:])
    os.remove(path)


# ------------------------------------------------------------------ main
def code_hash():
    files = [os.path.join(HERE, "followup.py"), os.path.join(HERE, "run.py")] + [os.path.join(vw.SCRIPTS, p) for p in (
        "llm_stem/stage2.py", "llm_stem/run.py", "code_robustness/run.py", "stem_fixes/run.py",
        "adaptive_robustness/models.py", "../model/encoder.py", "../model/vq.py")]
    return hashlib.sha256(b"".join(open(f, "rb").read() for f in files)).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(DATA), default="cora")
    ap.add_argument("--part", choices=("R", "A"), default="R")
    ap.add_argument("--level_idx", type=int, default=0)
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.selfcheck:
        return selfcheck()
    x, ei, y, trm, vam, tem = load_data(args.dataset)
    os.makedirs(OUT, exist_ok=True)
    import torch_geometric
    config = {"code_hash": code_hash(), "dataset": args.dataset, "tau": TAU, "eps": EPS, "pt_epochs": ls.PT_EPOCHS,
              "vgae": [vw.VGAE_EMB, vw.VGAE_HID, vw.VGAE_LAT, vw.VGAE_EPOCHS, vw.VGAE_LR], "mask_p": MASK_P,
              "prune": PRUNE, "rung": RUNG_KW, "strong": [STRONG_EPOCHS, STRONG_RESAMPLE, STRONG_BLOCK, PRBCD_LR, GREEDY_EPOCHS],
              "strong_suite": STRONG, "weight_attacks": WEIGHT_ATTACKS,
              "threads": args.threads, "torch": torch.__version__, "pyg": torch_geometric.__version__,
              "python": platform.python_version(),
              "data_sha": hashlib.sha256(open(DATA[args.dataset], "rb").read()).hexdigest()}
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    t0 = time.time()
    if args.part == "R":
        if not 0 <= args.level_idx < len(vw.LEVELS) or not 0 <= args.graph < vw.GRAPHS:
            raise SystemExit(f"level_idx must be in [0, {len(vw.LEVELS)}) and graph in [0, {vw.GRAPHS})")
        rows, edge_auc = run_r(args.level_idx, args.graph, x, ei, y, trm, vam, tem)
        name, extra = f"{args.dataset}_R_l{args.level_idx}_g{args.graph}_{tag}", {"edge_auc": edge_auc}
    else:
        splits, seeds = A_CELLS[args.dataset]
        if not (0 <= args.split < splits and 0 <= args.seed < seeds):
            raise SystemExit(f"{args.dataset}: split in [0, {splits}), seed in [0, {seeds})")
        name = f"{args.dataset}_A_s{args.split}_e{args.seed}_{tag}"
        rows, edits, secs, total, old_cfg = run_a(args.dataset, args.split, args.seed, x, ei, y, trm, vam, tem,
                                                  ckpt=os.path.join(OUT, f"{name}_ckpt.pt"), config=config)
        extra = {"edits": edits, "attack_seconds": secs, "seconds_total": total, "old_config": old_cfg}
    env = {**config, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, **extra}, f, indent=1)


if __name__ == "__main__":
    main()
