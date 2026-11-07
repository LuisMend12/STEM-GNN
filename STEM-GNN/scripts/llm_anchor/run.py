"""LLM prior anchoring of STEM-GNN's prediction head (experiments/llm-anchor/protocol.md, Amendment 1).

SAGE-VQ component pilot (real Encoder without MoE + real VQ, frozen EMA codebook).
Graph models: B0, A-qwen, A-minilm, MIX (validation-selected B0/Qwen probability mixture).
Structure-free controls: Qwen-alone, MLP, MLP+A-qwen.
Every model in a cell is evaluated on the shared union of candidate attack graphs.

Usage (from repo root):
  python STEM-GNN/scripts/llm_anchor/run.py --splits 0 1 2 --seeds 0 1 2 3 4 --threads 2
"""
import argparse
import copy
import importlib.util
import json
import logging
import os
import platform
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("cr", os.path.join(SCRIPTS, "code_robustness", "run.py"))
llm = _load("llm", os.path.join(SCRIPTS, "llm_robustness", "run.py"))
fx = cr.fx
log = logging.getLogger("anchor")
ROOT = os.path.join(SCRIPTS, "..", "..")
OUT = os.path.join(ROOT, "experiments", "llm-anchor", "results")
QWEN = os.path.join(ROOT, "experiments", "llm-robustness", "results", "qwen_prior_Qwen2.5-1.5B-Instruct_v2_perm0.pt")
LAM, QWEN_TS, MINILM_TS, ALPHAS = 1.0, (0.5, 1.0, 2.0, 4.0), (0.02, 0.05, 0.1), (0.0, 0.25, 0.5, 0.75, 1.0)
ATTACKS = (("ste", 37, "ste"), ("ste_r2", 41, "ste"), ("soft", 43, "soft"))


# ------------------------------------------------------------------ teachers
def calibrate(scores, y, va, temps):
    best = min((F.nll_loss(F.log_softmax(scores[va] / t, -1), y[va]).item(), t) for t in temps)
    return F.softmax(scores / best[1], -1), best[1]


def load_qwen(n, n_cls, y):
    store = torch.load(QWEN)
    meta, logits = store["meta"], store["logits"]
    assert meta["version"] == "v2" and meta["perm"] == 0 and meta["n_nodes"] == n, meta
    assert logits.shape == (n, n_cls) and torch.isfinite(logits).all()
    return logits, meta


# ------------------------------------------------------------------ models
class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, cr.HID), nn.ReLU(), nn.Dropout(fx.DROPOUT), nn.Linear(cr.HID, n_cls))

    def forward(self, x, ei=None, w=None):
        return self.net(x)


class Mix(nn.Module):
    """log((1-a) softmax(B0) + a p_qwen); attack_mode is forwarded to B0."""

    def __init__(self, b0, prior, alpha):
        super().__init__()
        self.b0, self.alpha = b0, alpha
        self.register_buffer("prior", prior)

    @property
    def attack_mode(self):
        return self.b0.attack_mode

    @attack_mode.setter
    def attack_mode(self, m):
        self.b0.attack_mode = m

    def forward(self, x, ei, w=None):
        p = (1 - self.alpha) * F.softmax(self.b0(x, ei, w), -1) + self.alpha * self.prior
        return torch.log(p.clamp_min(1e-12))


def train_model(model, x, ei, y, tr, va, base, prior=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    unl = ~tr
    best, state, wait, best_epoch, kl_at_best = -1.0, None, 0, 0, 0.0
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)
        loss = F.cross_entropy(model(x, ei)[tr], y[tr])
        kl = torch.zeros(())
        if prior is not None:
            model.eval()  # deterministic, gradient-enabled anchor forward
            kl = F.kl_div(F.log_softmax(model(x, ei)[unl], -1), prior[unl], reduction="batchmean")
            model.train()
        opt.zero_grad()
        (loss + LAM * kl).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = fx.acc(model(x, ei), y, va)
        if val > best:
            best, wait, best_epoch, kl_at_best = val, 0, epoch, kl.item()
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= fx.PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return best, best_epoch, kl_at_best


def low_label_mask(tr, va, te, y, per_class, seed):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros_like(tr)
    for c in range(int(y.max()) + 1):
        idx = (tr & (y == c)).nonzero().view(-1)
        assert idx.numel() >= per_class, f"class {c} has only {idx.numel()} training nodes"
        mask[idx[torch.randperm(idx.numel(), generator=g)[:per_class]]] = True
    assert not (mask & va).any() and not (mask & te).any() and not (va & te).any()
    return mask


# ------------------------------------------------------------------ cell
def run_setting(setting, enc, vq, x, ei, y, tr, va, te, priors, base, budgets, rates):
    n_cls = int(y.max()) + 1
    test_idx = te.nonzero().view(-1)
    models = {}
    for name, prior in (("B0", None), ("A-qwen", priors["qwen"]), ("A-minilm", priors["minilm"])):
        fx.seed_all(base + 500)
        m = cr.CodeModel(copy.deepcopy(enc), copy.deepcopy(vq), n_cls)
        models[name] = (m, *train_model(m, x, ei, y, tr, va, base, prior))
    for name, prior in (("MLP", None), ("MLP+A-qwen", priors["qwen"])):
        fx.seed_all(base + 600)
        m = MLP(x.size(1), n_cls)
        models[name] = (m, *train_model(m, x, ei, y, tr, va, base, prior))
    b0 = models["B0"][0]
    with torch.no_grad():
        p_b0 = F.softmax(b0(x, ei), -1)
    alpha = max(ALPHAS, key=lambda a: (fx.acc(torch.log((1 - a) * p_b0 + a * priors["qwen"]), y, va), -a))
    models["MIX"] = (Mix(b0, priors["qwen"], alpha).eval(), None, None, None)
    # surrogate trained on the same labels as the learners in this setting
    fx.seed_all(base + 11)
    sur = cr.Surrogate(x.size(1), n_cls)
    fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
    cands, edits = {}, {}
    for r, b in budgets.items():
        cands[("sage", r)] = cr.prbcd(sur, x, ei, y, test_idx, b, seed=base + 23 + int(100 * r))
        edits[f"sage@{r}"] = cr.validate(ei, cands[("sage", r)], x.size(0), b)
        for name in ("B0", "A-qwen", "A-minilm", "MIX"):
            for atk, off, mode in ATTACKS:
                g = cr.prbcd(models[name][0], x, ei, y, test_idx, b, seed=base + off + int(100 * r), mode=mode)
                cands[(f"{name}:{atk}", r)] = g
                edits[f"{name}:{atk}@{r}"] = cr.validate(ei, g, x.size(0), b)
    q_correct = priors["qwen"].argmax(-1) == y
    rows, pernode = [], {}
    for name, (m, val, best_epoch, kl) in list(models.items()) + [("Qwen-alone", (None, None, None, None))]:
        if m is None:
            pred = lambda g: priors["qwen"].argmax(-1)  # noqa: E731  structure-free
        else:
            pred = lambda g, m=m: m(x, g).argmax(-1).detach()  # noqa: E731
        with torch.no_grad():
            clean_pred = pred(ei)
            corr = {k: pred(g) == y for k, g in cands.items()}
        clean_c = clean_pred == y
        row = {"setting": setting, "method": name, "n_train": int(tr.sum()), "val": val, "best_epoch": best_epoch,
               "kl_at_best": kl, "clean": clean_c[te].float().mean().item(),
               "agree_qwen_clean": (clean_pred == priors["qwen"].argmax(-1))[te].float().mean().item(),
               "alpha": alpha if name == "MIX" else None}
        for r in rates:
            keys = [k for k in corr if k[1] <= r]
            suite = torch.stack([clean_c] + [corr[k] for k in keys])
            worst = suite.all(0)
            row[f"worst_node@{r}"] = worst[te].float().mean().item()
            row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
            row[f"worst_node_on_qwen_correct@{r}"] = worst[te & q_correct].float().mean().item()
            row[f"worst_node_on_qwen_wrong@{r}"] = worst[te & ~q_correct].float().mean().item()
            if r == max(rates):
                pernode[name] = worst
        row["clean_on_qwen_correct"] = clean_c[te & q_correct].float().mean().item()
        row["clean_on_qwen_wrong"] = clean_c[te & ~q_correct].float().mean().item()
        row["qwen_errors_corrected_clean"] = int((clean_c & ~q_correct & te).sum())
        row["qwen_errors_inherited_clean"] = int((~clean_c & ~q_correct & te & (clean_pred == priors["qwen"].argmax(-1))).sum())
        rows.append(row)
    return rows, pernode, edits, alpha


def run_cell(split, seed, x, ei, y, tr_std, va, te, priors, pt_epochs, rates, settings):
    base = 1000 * seed + split
    budgets = {r: int(r * len(cr.undirected(ei))) for r in rates}
    enc, vq, _ = fx.pretrain_x2("C-ema", x, ei, cr.HID, pt_epochs, pt_seed=base)
    rows, extra = [], {}
    for setting in settings:
        tr = tr_std if setting == "standard" else low_label_mask(tr_std, va, te, y, 3, base)
        t0 = time.time()
        srows, pernode, edits, alpha = run_setting(setting, enc, vq, x, ei, y, tr, va, te, priors, base, budgets, rates)
        for r in srows:
            r.update(split=split, seed=seed, seconds_setting=round(time.time() - t0, 1))
            log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
        rows += srows
        extra[setting] = {"pernode_worst": {k: v.tolist() for k, v in pernode.items()}, "edits": edits,
                          "train_mask": tr.nonzero().view(-1).tolist(), "alpha": alpha}
    return rows, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--settings", nargs="+", default=["standard", "low"])
    ap.add_argument("--pt_epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(OUT, exist_ok=True)
    x, ei, y, trm, vam, tem = fx.load()
    n_cls = int(y.max()) + 1
    qwen_logits, qmeta = load_qwen(x.size(0), n_cls, y)
    d = torch.load(llm.probe.CORA, weights_only=False)
    cos = F.normalize(x, dim=-1) @ llm.class_embeddings(list(d.label_names)).t()
    for s in args.splits:
        q, tq = calibrate(qwen_logits, y, vam[s], QWEN_TS)
        mlm, tm = calibrate(cos, y, vam[s], MINILM_TS)
        priors = {"qwen": q.float(), "minilm": mlm.float()}
        meta = {"qwen_meta": qmeta, "qwen_T": tq, "minilm_T": tm,
                "qwen_test_acc": (q.argmax(-1)[tem[s]] == y[tem[s]]).float().mean().item(),
                "minilm_test_acc": (mlm.argmax(-1)[tem[s]] == y[tem[s]]).float().mean().item()}
        rows, extras = [], {}
        for seed in args.seeds:
            r, e = run_cell(s, seed, x, ei, y, trm[s], vam[s], tem[s], priors, args.pt_epochs, args.rates, args.settings)
            rows += r
            extras[seed] = e
            with open(os.path.join(OUT, f"split{s}_{args.tag}.json"), "w") as f:
                json.dump({"args": vars(args), "meta": meta, "env": {"torch": torch.__version__,
                           "python": platform.python_version(), "threads": args.threads}, "rows": rows}, f, indent=1)
            torch.save(extras, os.path.join(OUT, f"split{s}_{args.tag}_pernode.pt"))


if __name__ == "__main__":
    main()
