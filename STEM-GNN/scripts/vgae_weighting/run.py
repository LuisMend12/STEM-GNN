"""Structure-only VGAE neighbor weighting for STEM-GNN (experiments/vgae-weighting/protocol.md).

A VGAE whose input is a learned per-node embedding (no text) is trained on the observed graph. Each edge
gets the cosine similarity of its endpoints' posterior means, recomputed on whatever graph the model
sees, so an attacker's edges are scored too. Neighbor weights follow the milestone formula:
  a_vu = (1 - eps) * softmax_u(s_vu / tau) + eps / |N(v)|   (PRBCD edge weights folded in).
Text never weights, selects, adds or drops edges.

Parts:
  R  random edge injection, poisoning: 0/500/1000/2000/5000 added edges, every model trained on the noisy graph
  A  adaptive evasion: the repo's PRBCD suite on the clean graph (as in experiments/capped-correction)

Usage (from repo root):
  python STEM-GNN/scripts/vgae_weighting/run.py --selfcheck
  python STEM-GNN/scripts/vgae_weighting/run.py --part R --level 2000 --graph 3
  python STEM-GNN/scripts/vgae_weighting/run.py --part A --split 0 --seed 0
"""
import argparse
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
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, scatter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s2 = _load("s2", os.path.join(SCRIPTS, "llm_stem", "stage2.py"))
ls, cr, fx = s2.ls, s2.cr, s2.fx
sys.modules.update({"ls": ls, "cr": cr, "fx": fx})
arm_models = _load("ar_models", os.path.join(SCRIPTS, "adaptive_robustness", "models.py"))
sys.modules.setdefault("vgae_weighting", sys.modules[__name__])

log = logging.getLogger("vgae_weighting")
ROOT = os.path.join(SCRIPTS, "..", "..")
OUT = os.path.join(ROOT, "experiments", "vgae-weighting", "results")
LEVELS = (0, 500, 1000, 2000, 5000)
GRAPHS = 10
TAU, EPS = 0.3, 0.2  # fixed before any run; never tuned on a text-reading model's validation accuracy
VGAE_EMB, VGAE_HID, VGAE_LAT, VGAE_EPOCHS, VGAE_LR = 64, 64, 32, 200, 0.01
BASE_HID = 64  # adaptive_robustness baseline width
STEM_ARMS = ("S0", "S-VG", "S-VGr", "S-SJ")
PLAIN_ARMS = ("SAGE", "SAGE-VG", "GCN", "RUNG-lite", "MLP-M")
ARMS = STEM_ARMS + PLAIN_ARMS
ATTACKED = ("S0", "S-VG", "SAGE", "SAGE-VG", "RUNG-lite")  # S-SJ is too costly inside PRBCD; scored on the union only


# ------------------------------------------------------------------ structure-only scorers
class StructVGAE(nn.Module):
    """VGAE encoder on structure only: learned node embedding -> GCN -> (mu, logstd)."""

    def __init__(self, n):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n, VGAE_EMB) * 0.1)
        self.c1 = GCNConv(VGAE_EMB, VGAE_HID)
        self.c_mu, self.c_ls = GCNConv(VGAE_HID, VGAE_LAT), GCNConv(VGAE_HID, VGAE_LAT)

    def encode(self, ei, w=None):
        h = F.relu(self.c1(self.emb, ei, w))
        return self.c_mu(h, ei, w), self.c_ls(h, ei, w).clamp(max=10)

    def scores(self, ei, w=None):
        mu, _ = self.encode(ei, w)
        return F.cosine_similarity(mu[ei[0]], mu[ei[1]], dim=-1)


def train_vgae(n, ei, seed, epochs=VGAE_EPOCHS):
    """ELBO: inner-product edge reconstruction with negative sampling + KL / n. epochs=0 gives the random control."""
    fx.seed_all(seed)
    model = StructVGAE(n)
    opt = torch.optim.Adam(model.parameters(), lr=VGAE_LR)
    for epoch in range(epochs):
        model.train()
        fx.seed_all(seed * 1_009 + epoch)
        mu, logstd = model.encode(ei)
        z = mu + torch.randn_like(mu) * logstd.exp()
        neg = negative_sampling(ei, n)
        pos_logit = (z[ei[0]] * z[ei[1]]).sum(-1)
        neg_logit = (z[neg[0]] * z[neg[1]]).sum(-1)
        recon = F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit)) + \
            F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
        kl = -0.5 / n * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))
        loss = recon + kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class JaccardScorer(nn.Module):
    """Structural Jaccard of the endpoints' (weighted) neighbor sets; differentiable in the edge weights."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def scores(self, ei, w=None):
        w = torch.ones(ei.size(1)) if w is None else w
        dense = torch.zeros(self.n, self.n).index_put((ei[0], ei[1]), w, accumulate=True)
        sparse = torch.sparse_coo_tensor(ei, w, (self.n, self.n))
        common = torch.sparse.mm(sparse, dense)[ei[0], ei[1]]
        deg = scatter(w, ei[0], dim=0, dim_size=self.n, reduce="sum")
        return common / (deg[ei[0]] + deg[ei[1]] - common).clamp_min(1e-6)


def neighbor_weights(s, ei, w, tau, eps, tgt):
    """(1 - eps) * softmax over each target's neighbors of s / tau + eps / degree, with edge weights w."""
    w = torch.ones(ei.size(1)) if w is None else w
    t = ei[tgt]
    n = int(ei.max()) + 1 if ei.numel() else 0
    logits = s / tau
    shift = scatter(logits.detach(), t, dim=0, dim_size=n, reduce="max")[t]
    e = w * torch.exp(logits - shift)
    total_e = scatter(e, t, dim=0, dim_size=n, reduce="sum")[t].clamp_min(1e-12)
    total_w = scatter(w, t, dim=0, dim_size=n, reduce="sum")[t].clamp_min(1e-12)
    return (1 - eps) * e / total_e + eps * w / total_w


# ------------------------------------------------------------------ models
class WeightedSTEM(nn.Module):
    """STEM's RouterCodeModel with scorer-derived neighbor weights (target = ei[0], as in moe_layer)."""

    def __init__(self, base, scorer, tau, eps):
        super().__init__()
        self.base, self.scorer, self.tau, self.eps = base, scorer, tau, eps

    @property
    def attack_mode(self):
        return self.base.attack_mode

    @attack_mode.setter
    def attack_mode(self, mode):
        self.base.attack_mode = mode

    def forward(self, x, ei, w=None):
        c = neighbor_weights(self.scorer.scores(ei, w), ei, w, self.tau, self.eps, tgt=0)
        return self.base(x, ei, c)

    def penalty(self, coeff):
        return self.base.penalty(coeff)


class WeightedSAGE(arm_models.SAGE):
    """The milestone's weighted update h_v = W_self h_v + sum_u a_vu W_nbr h_u (target = edge_index[1])."""

    def __init__(self, in_dim, hid, out_dim, scorer, tau, eps):
        super().__init__(in_dim, hid, out_dim)
        self.scorer, self.tau, self.eps = scorer, tau, eps

    def _coeff_mass(self, x, edge_index, edge_weight):
        s = self.scorer.scores(edge_index, edge_weight)
        return neighbor_weights(s, edge_index, edge_weight, self.tau, self.eps, tgt=1), None


for _cls in (StructVGAE, JaccardScorer, WeightedSTEM, WeightedSAGE):
    _cls.__module__ = "vgae_weighting"


def finetune(build_model, x, ei, y, tr, va, base):
    """Stage 2 finetune recipe for any STEM-backed model (same seeds, penalty, stopping)."""
    fx.seed_all(base + 500)
    model = build_model()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    best, state, wait = -1.0, None, 0
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)
        loss = F.cross_entropy(model(x, ei)[tr], y[tr]) + model.penalty(s2.COEFF)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = fx.acc(model(x, ei), y, va)
        if val > best:
            best, wait = val, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= fx.PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, best


def fit_plain(build_model, x, ei, y, tr, va, seed):
    fx.seed_all(seed)
    model = build_model()
    val, _ = fx.fit([p for p in model.parameters() if p.requires_grad], lambda: model(x, ei), y, tr, va, modules=(model,))
    model.eval()
    return model, val


def stem_base(enc, vq, n_cls):
    return ls.RouterCodeModel(copy.deepcopy(enc), copy.deepcopy(vq), n_cls, None)


def train_all(x, ei, y, tr, va, base, arms=ARMS):
    """tau and eps are fixed in advance (no validation search): no text-reading model picks the weights,
    and weighted arms get one fit each, like S0."""
    n, n_cls = x.size(0), int(y.max()) + 1
    enc, vq = s2.pretrain(x, ei, ls.PT_EPOCHS, base)
    scorers = {"S-VG": train_vgae(n, ei, base + 900), "S-VGr": train_vgae(n, ei, base + 900, epochs=0),
               "S-SJ": JaccardScorer(n)}
    models, meta = {}, {}
    models["S0"], val = finetune(lambda: stem_base(enc, vq, n_cls), x, ei, y, tr, va, base)
    meta["S0"] = {"val": val}
    for arm in ("S-VG", "S-VGr", "S-SJ"):
        if arm not in arms:
            continue
        models[arm], val = finetune(lambda: WeightedSTEM(stem_base(enc, vq, n_cls), scorers[arm], TAU, EPS),
                                    x, ei, y, tr, va, base)
        meta[arm] = {"val": val, "tau": TAU, "eps": EPS}
    plain = {"SAGE": lambda: arm_models.SAGE(x.size(1), BASE_HID, n_cls),
             "GCN": lambda: arm_models.GCN(x.size(1), BASE_HID, n_cls),
             "RUNG-lite": lambda: arm_models.RUNGLite(x.size(1), BASE_HID, n_cls, lam=0.3, delta=0.3, gamma=1.2)}
    for k, (arm, build) in enumerate(plain.items()):
        if arm in arms:
            models[arm], val = fit_plain(build, x, ei, y, tr, va, base + 700 + k)
            meta[arm] = {"val": val}
    if "SAGE-VG" in arms:
        models["SAGE-VG"], val = fit_plain(lambda: WeightedSAGE(x.size(1), BASE_HID, n_cls, scorers["S-VG"], TAU, EPS),
                                           x, ei, y, tr, va, base + 700)
        meta["SAGE-VG"] = {"val": val, "tau": TAU, "eps": EPS}
    if "MLP-M" in arms:
        models["MLP-M"] = s2.fit_mlp(x, y, tr, va, base)
        with torch.no_grad():
            meta["MLP-M"] = {"val": fx.acc(models["MLP-M"](x), y, va)}
    return models, meta, scorers


# ------------------------------------------------------------------ part R: random injection, poisoning
def inject(ei, n, count, seed):
    """count new undirected edges between distinct, previously unconnected nodes; both directions added."""
    gen = torch.Generator().manual_seed(seed)
    existing = {(min(a, b), max(a, b)) for a, b in ei.t().tolist()}
    new = set()
    while len(new) < count:
        a, b = torch.randint(0, n, (2,), generator=gen).tolist()
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in new or key in existing:
            continue
        new.add(key)
    added = torch.tensor(sorted(new), dtype=torch.long).t() if count else torch.empty((2, 0), dtype=torch.long)
    noisy = torch.cat([ei, added, added.flip(0)], 1)
    injected = torch.cat([torch.zeros(ei.size(1), dtype=torch.bool), torch.ones(2 * added.size(1), dtype=torch.bool)])
    return noisy, injected


def macro_f1(pred, y, mask, n_cls):
    f1 = []
    for c in range(n_cls):
        tp = ((pred == c) & (y == c) & mask).sum().item()
        fp = ((pred == c) & (y != c) & mask).sum().item()
        fn = ((pred != c) & (y == c) & mask).sum().item()
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1) / n_cls


def auc(pos, neg):
    """P(score of an original edge > score of an injected edge), ties count half."""
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    allv = torch.cat([pos, neg])
    ranks = allv.argsort().argsort().float() + 1
    # average ranks for ties
    uniq, inv = torch.unique(allv, return_inverse=True)
    mean_rank = scatter(ranks, inv, dim=0, dim_size=uniq.numel(), reduce="mean")[inv]
    r_pos = mean_rank[: pos.numel()].sum().item()
    return (r_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())


def run_r(level, graph, x, ei, y, trm, vam, tem):
    split = graph % 3
    base = 100_000 * (LEVELS.index(level) + 1) + 1000 * graph + split
    tr, va, te = trm[split], vam[split], tem[split]
    noisy, injected = inject(ei, x.size(0), level, 50_000 + 1000 * graph + level)
    models, meta, scorers = train_all(x, noisy, y, tr, va, base)
    n_cls = int(y.max()) + 1
    rows = []
    with torch.no_grad():
        edge_auc = {k: auc(sc.scores(noisy)[~injected], sc.scores(noisy)[injected]) for k, sc in scorers.items()}
        for arm, model in models.items():
            pred = (model(x) if arm == "MLP-M" else model(x, noisy)).argmax(-1)
            n_orig = len(cr.undirected(ei))
            rows.append({"part": "R", "level": level, "graph": graph, "split": split, "arm": arm, **meta[arm],
                         "orig_undirected_edges": n_orig, "injected_pct": 100 * level / n_orig,
                         "acc": (pred == y)[te].float().mean().item(), "macro_f1": macro_f1(pred, y, te, n_cls)})
            log.info(json.dumps(rows[-1]))
    return rows, edge_auc


# ------------------------------------------------------------------ part A: adaptive evasion
def run_a(split, seed, x, ei, y, trm, vam, tem, ckpt=None, config=None):
    base = 1000 * seed + split
    tr, va, te = trm[split], vam[split], tem[split]
    test_idx = te.nonzero().view(-1)
    budgets = {r: int(r * len(cr.undirected(ei))) for r in ls.RATES}
    state = torch.load(ckpt, weights_only=False) if ckpt and os.path.exists(ckpt) else None
    if state is not None and state["config"] != config:
        raise RuntimeError(f"{ckpt} was made with a different configuration; refusing to resume")
    started = time.time()
    if state is not None:
        models, meta, sur, graphs = state["models"], state["meta"], state["sur"], state["graphs"]
        spent = state.get("seconds", 0.0)
    else:
        models, meta, _ = train_all(x, ei, y, tr, va, base, arms=ARMS)
        fx.seed_all(base + 11)
        sur = cr.Surrogate(x.size(1), int(y.max()) + 1)
        fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
        graphs, spent = {}, 0.0
    payload = lambda: {"config": config, "models": models, "meta": meta, "sur": sur, "graphs": graphs,
                       "seconds": spent + time.time() - started}
    if ckpt:
        s2_save(ckpt, payload())
    for r, b in budgets.items():
        jobs = [("sage", sur, "sage", 23, "ste", "prob_margin")] + [
            (arm, models[arm], name, off, mode, loss) for arm in ATTACKED for name, off, mode, loss in ls.ATTACKS]
        for arm, model, name, off, mode, loss in jobs:
            if (arm, name, r) in graphs:
                continue
            t0 = time.time()
            graphs[(arm, name, r)] = ls.prbcd(model, x, ei, y, test_idx, b, base + off + int(100 * r), mode, loss)
            log.info(json.dumps({"attack": [arm, name, r], "seconds": round(time.time() - t0, 1)}))
            if ckpt:
                s2_save(ckpt, payload())
    edits = {"|".join(map(str, k)): cr.validate(ei, g, x.size(0), budgets[k[2]]) for k, g in graphs.items()}
    rows = []
    for arm, model in models.items():
        fwd = (lambda g, m=model: m(x)) if arm == "MLP-M" else (lambda g, m=model: m(x, g))
        with torch.no_grad():
            clean = fwd(ei).argmax(-1) == y
            corr = {k: fwd(g).argmax(-1) == y for k, g in graphs.items()}
        row = {"part": "A", "split": split, "seed": seed, "arm": arm, **meta[arm], "clean": clean[te].float().mean().item()}
        for r in ls.RATES:
            keys = [k for k in corr if k[2] <= r]
            suite = torch.stack([clean] + [corr[k] for k in keys])
            row[f"worst_node@{r}"] = suite.all(0)[te].float().mean().item()
            row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
            row[f"own_worst_node@{r}"] = torch.stack(
                [clean] + [corr[k] for k in keys if k[0] == arm]).all(0)[te].float().mean().item()
            row[f"transfer_sage@{r}"] = corr[("sage", "sage", r)][te].float().mean().item()
        rows.append(row)
        log.info(json.dumps(row))
    return rows, edits, round(spent + time.time() - started, 1)


def s2_save(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ selfcheck
def check(name, ok, detail=""):
    print(name, "OK" if ok else "FAILED", detail, flush=True)
    assert ok, name


def selfcheck(x, ei, y, trm, vam, tem):
    n, n_cls = x.size(0), int(y.max()) + 1
    tr, va, te = trm[0], vam[0], tem[0]
    noisy, injected = inject(ei, n, 500, 1)
    und = cr.undirected(noisy)
    check("injection adds 500 new undirected edges", len(und) == len(cr.undirected(ei)) + 500)
    check("no self-loops, symmetric", bool((noisy[0] != noisy[1]).all())
          and all((b, a) in set(map(tuple, noisy.t().tolist())) for a, b in noisy[:, injected].t().tolist()[:200]))
    check("injection repeatable", torch.equal(noisy, inject(ei, n, 500, 1)[0]))
    vg = train_vgae(n, noisy, 3, epochs=50)
    check("VGAE takes no text input", not any(p.size(-1) == x.size(1) or p.size(0) == x.size(1) for p in vg.parameters()))
    with torch.no_grad():
        s = vg.scores(noisy)
    a = auc(s[~injected], s[injected])
    check("VGAE scores finite, AUC computed", bool(torch.isfinite(s).all()) and a is not None, round(a, 3))
    check("AUC sanity (identical scores give 0.5)", abs(auc(torch.ones(5), torch.ones(7)) - 0.5) < 1e-9)
    w = torch.rand(noisy.size(1))
    for tgt in (0, 1):
        c = neighbor_weights(s, noisy, w, 0.3, 0.2, tgt)
        tot = scatter(c, noisy[tgt], dim=0, dim_size=n, reduce="sum")
        has = scatter(torch.ones_like(c), noisy[tgt], dim=0, dim_size=n, reduce="sum") > 0
        check(f"weights sum to 1 per target (tgt={tgt})", torch.allclose(tot[has], torch.ones(int(has.sum())), atol=1e-5))
    js = JaccardScorer(n)
    with torch.no_grad():
        j = js.scores(ei)
    check("Jaccard in [0, 1]", bool((j >= -1e-6).all() and (j <= 1 + 1e-6).all()))
    enc, vq = s2.pretrain(x, ei, 5, 0)
    ours = finetune(lambda: stem_base(enc, vq, n_cls), x, ei, y, tr, va, 0)[0]
    theirs = s2.finetune(enc, vq, x, ei, y, tr, va, 0)[0]
    with torch.no_grad():
        check("S0 finetune equals Stage 2's finetune", torch.equal(ours(x, ei).argmax(-1), theirs(x, ei).argmax(-1))
              and torch.allclose(ours(x, ei), theirs(x, ei), atol=1e-5))
        weighted = WeightedSTEM(stem_base(enc, vq, n_cls), vg, TAU, EPS).eval()
        seen = []
        real_forward = weighted.base.forward
        weighted.base.forward = lambda xx, g, w=None: (seen.append(w.detach().clone()), real_forward(xx, g, w))[1]
        out_a = weighted(x, ei)
        out_b = weighted(x[torch.randperm(n)] + torch.randn_like(x), ei)
        del weighted.base.forward
        check("edge weights passed to STEM are identical when the text changes", torch.equal(seen[0], seen[1]))
        check("the text still reaches the encoder (outputs differ)", not torch.equal(out_a, out_b))
    for name, scorer in (("VG", vg), ("SJ", js)):
        model = WeightedSTEM(stem_base(enc, vq, n_cls), scorer, 0.3, 0.2).eval()
        for mode in ("ste", "soft"):
            model.attack_mode = mode
            ww = torch.ones(ei.size(1), requires_grad=True)
            p = F.softmax(model(x, ei, ww), -1)
            m = p[va].gather(1, y[va, None]).sum() - p[va].max(-1).values.sum()
            g = torch.autograd.grad(m, ww)[0]
            check(f"S-{name} edge gradient finite and nonzero ({mode})", bool(torch.isfinite(g).all()) and g.norm() > 0)
        model.attack_mode = "ste"
        check(f"S-{name} attack mode reset", model.base.attack_mode == "ste")
    sage_vg = WeightedSAGE(x.size(1), BASE_HID, n_cls, vg, 0.3, 0.0).eval()
    ww = torch.ones(ei.size(1), requires_grad=True)
    g = torch.autograd.grad(sage_vg(x, ei, ww).logsumexp(-1).sum(), ww)[0]
    check("SAGE-VG edge gradient finite and nonzero", bool(torch.isfinite(g).all()) and g.norm() > 0)
    check("VGAE frozen", not any(p.requires_grad for p in vg.parameters()))
    base_model = stem_base(enc, vq, n_cls).eval()
    flat = WeightedSTEM(stem_base(enc, vq, n_cls), vg, 1e6, 0.0).eval()
    flat.base.load_state_dict(base_model.state_dict())
    with torch.no_grad():
        check("tau -> infinity reduces S-VG to S0", torch.allclose(flat(x, ei), base_model(x, ei), atol=1e-4))
    budget = int(0.1 * len(cr.undirected(ei)))
    model = WeightedSTEM(stem_base(enc, vq, n_cls), vg, 0.3, 0.2).eval()
    pert = ls.prbcd(model, x, ei, y, te.nonzero().view(-1), budget, 1, "ste", "prob_margin")
    check("PRBCD on S-VG returns a valid graph", cr.validate(ei, pert, n, budget) is not None)
    import io
    buf = io.BytesIO()
    torch.save({"m": model, "s": sage_vg}, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=False)
    with torch.no_grad():
        check("checkpoint round trip", torch.equal(back["m"](x, ei), model(x, ei)))
    path = os.path.join(OUT, "_selfcheck.pt")
    os.makedirs(OUT, exist_ok=True)
    with torch.no_grad():
        torch.save({"m": model, "x": x, "ei": ei, "out": model(x, ei)}, path)
    import subprocess
    probe = ("import importlib.util,sys,torch;"
             f"spec=importlib.util.spec_from_file_location('vgae_weighting',{os.path.abspath(__file__)!r});"
             "m=importlib.util.module_from_spec(spec);sys.modules['vgae_weighting']=m;spec.loader.exec_module(m);"
             f"d=torch.load({path!r},weights_only=False);"
             "print(torch.allclose(d['m'](d['x'],d['ei']),d['out'],atol=1e-5))")
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    ok = res.stdout.strip().endswith("True")
    check("checkpoint loads in a fresh process", ok, "" if ok else res.stderr[-300:])
    os.remove(path)


def code_hash():
    files = [os.path.join(HERE, "run.py")] + [os.path.join(SCRIPTS, p) for p in (
        "llm_stem/stage2.py", "llm_stem/run.py", "code_robustness/run.py", "stem_fixes/run.py",
        "adaptive_robustness/models.py", "../model/encoder.py", "../model/vq.py")]
    return hashlib.sha256(b"".join(open(f, "rb").read() for f in files)).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("R", "A"), default="R")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x, ei, y, trm, vam, tem = fx.load()
    if args.selfcheck:
        return selfcheck(x, ei, y, trm, vam, tem)
    os.makedirs(OUT, exist_ok=True)
    import torch_geometric
    config = {"code_hash": code_hash(), "tau": TAU, "eps": EPS, "pt_epochs": ls.PT_EPOCHS,
              "vgae": [VGAE_EMB, VGAE_HID, VGAE_LAT, VGAE_EPOCHS, VGAE_LR], "threads": args.threads,
              "torch": torch.__version__, "pyg": torch_geometric.__version__, "python": platform.python_version(),
              "data_sha": hashlib.sha256(open(fx.CORA, "rb").read()).hexdigest()}
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    t0 = time.time()
    if args.part == "R":
        if args.level not in LEVELS or not 0 <= args.graph < GRAPHS:
            raise SystemExit(f"level must be in {LEVELS} and graph in [0, {GRAPHS})")
        rows, edge_auc = run_r(args.level, args.graph, x, ei, y, trm, vam, tem)
        name, extra = f"R_level{args.level}_graph{args.graph}_{tag}", {"edge_auc": edge_auc}
    else:
        name = f"A_split{args.split}_seed{args.seed}_{tag}"
        rows, edits, total = run_a(args.split, args.seed, x, ei, y, trm, vam, tem,
                                   ckpt=os.path.join(OUT, f"{name}_ckpt.pt"), config=config)
        extra = {"edits": edits, "seconds_total": total}
    env = {**config, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, **extra}, f, indent=1)


if __name__ == "__main__":
    main()
