"""Baselines for the capped-correction experiment (experiments/capped-correction/baselines_protocol.md).

For one cell, this loads the capped run's checkpoint (S0, RES, CAP arms, MLP-M, SAGE surrogate and every
attacked graph), trains standard baselines on the same split, attacks each baseline with the same PRBCD
suite, and re-scores every arm (old and new) on the union of all graphs built in the cell.

Baselines:
  GCN, SAGE                     plain message passing on MiniLM features
  APPNP-lite, RUNG-lite         repo's robust-propagation baselines (adaptive_robustness/models.py)
  SAGE+TextGate                 GNNGuard-style text-weighted edges; reference only (our method never
                                lets text touch edges)
  ENS{a}, a in ALPHAS           log-probability mix (1-a) log p_MLP-M + a log p_S0, no training: an
                                untrained shared-component control between the two models CAP combines

Usage (from repo root):
  python STEM-GNN/scripts/capped/baselines.py --selfcheck
  python STEM-GNN/scripts/capped/baselines.py --split 0 --seed 0
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("capped_run", os.path.join(HERE, "run.py"))
cap = importlib.util.module_from_spec(spec)
sys.modules["capped_run"] = cap
spec.loader.exec_module(cap)
ls, cr, fx = cap.ls, cap.cr, cap.fx

spec = importlib.util.spec_from_file_location("ar_models", os.path.join(HERE, "..", "adaptive_robustness", "models.py"))
arm_models = importlib.util.module_from_spec(spec)
sys.modules["ar_models"] = arm_models  # baseline checkpoints unpickle by this module name
spec.loader.exec_module(arm_models)

sys.modules.setdefault("capped_baselines", sys.modules[__name__])
log = logging.getLogger("capped_baselines")
OUT = os.path.join(cap.ROOT, "experiments", "capped-correction", "baselines")
HID = 64  # the adaptive_robustness baselines' width
TEXT_TAU_Q = 0.1  # the adaptive_robustness TextGate threshold: 10% quantile of edge cosine
ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
TRAINED = ("GCN", "SAGE", "APPNP-lite", "RUNG-lite", "SAGE+TextGate")
ENSEMBLES = tuple(f"ENS{a}" for a in ALPHAS)
BASELINES = TRAINED + ENSEMBLES
TOL = 1e-9


class LogMix(nn.Module):
    """(1 - a) log softmax(MLP-M) + a log softmax(S0). attack_mode reaches S0."""

    def __init__(self, mlp, s0, alpha):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.mlp, self.s0, self.alpha = mlp, s0, alpha

    @property
    def attack_mode(self):
        return self.s0.attack_mode

    @attack_mode.setter
    def attack_mode(self, mode):
        self.s0.attack_mode = mode

    def forward(self, x, ei, w=None):
        return (1 - self.alpha) * F.log_softmax(self.mlp(x), -1) + self.alpha * F.log_softmax(self.s0(x, ei, w), -1)


LogMix.__module__ = "capped_baselines"


def ensembles(models):
    return {n: LogMix(models["MLP-M"], models["S0"], a) for a, n in zip(ALPHAS, ENSEMBLES)}


def edge_cos_quantile(x, ei, q):
    src, dst = ei
    return torch.quantile(F.cosine_similarity(x[src], x[dst], dim=-1), q).item()


def build(name, in_dim, n_cls, tau):
    m = arm_models
    return {"GCN": lambda: m.GCN(in_dim, HID, n_cls),
            "SAGE": lambda: m.SAGE(in_dim, HID, n_cls),
            "APPNP-lite": lambda: m.RUNGLite(in_dim, HID, n_cls, penalty="l2", lam=0.3),
            "RUNG-lite": lambda: m.RUNGLite(in_dim, HID, n_cls, lam=0.3, delta=0.3, gamma=1.2),
            "SAGE+TextGate": lambda: m.SAGE(in_dim, HID, n_cls, mode="text_gate", tau=tau)}[name]()


def train_baselines(models, x, ei, y, tr, va, base):
    tau = edge_cos_quantile(x, ei, TEXT_TAU_Q)
    out, meta = {}, {}
    for k, name in enumerate(TRAINED):
        fx.seed_all(base + 700 + k)
        model = build(name, x.size(1), int(y.max()) + 1, tau)
        val, epochs = fx.fit(list(model.parameters()), lambda: model(x, ei), y, tr, va, modules=(model,))
        if not torch.isfinite(model(x, ei)).all():
            raise RuntimeError(f"non-finite logits from {name}")
        out[name], meta[name] = model, {"val": val, "epochs": epochs, "cap": None}
    for name, model in ensembles(models).items():
        out[name] = model
        with torch.no_grad():
            meta[name] = {"val": fx.acc(model(x, ei), y, va), "cap": None, "alpha": model.alpha}
    return out, meta, tau


def score(models, meta, graphs, orig_keys, x, ei, y, te, rates, split, seed):
    """orig_keys: the graph keys loaded from the capped checkpoint (the protocol's union)."""
    mlp = models["MLP-M"]
    rows = []
    for arm, model in models.items():
        clean = cr.correct(model, x, ei, y)
        corr = {k: cr.correct(model, x, g, y) for k, g in graphs.items()}
        c = meta[arm].get("cap")
        row = {"arm": arm, "split": split, "seed": seed, **meta[arm], "clean": clean[te].float().mean().item(),
               "cert_test": cap.certified(mlp, x, y, c)[te].float().mean().item() if c is not None else None}
        for r in rates:
            keys = [k for k in corr if k[2] <= r]
            suite = torch.stack([clean] + [corr[k] for k in keys])
            row[f"worst_node@{r}"] = suite.all(0)[te].float().mean().item()
            row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
            row[f"own_worst_node@{r}"] = torch.stack(
                [clean] + [corr[k] for k in keys if k[0] == arm]).all(0)[te].float().mean().item()
            row[f"transfer_sage@{r}"] = corr[("sage", "sage", r)][te].float().mean().item()
            orig = [corr[k] for k in keys if k in orig_keys]
            row[f"orig_worst_node@{r}"] = torch.stack([clean] + orig).all(0)[te].float().mean().item()
            assert row[f"worst_node@{r}"] <= row[f"orig_worst_node@{r}"] + TOL, (arm, r)
        assert all(not isinstance(v, float) or v == v for v in row.values()), row
        rows.append(row)
        log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    return rows


def verify_reproduction(rows, split, seed, rates):
    """Our arms' clean, certified and protocol-union worst-node must equal the saved capped results."""
    ref = {r["arm"]: r for r in saved_rows(split, seed)}
    for row in rows:
        if row["arm"] not in ref:
            continue
        want = ref[row["arm"]]
        for k in ["clean", "cert_test"] + [f"worst_node@{r}" for r in rates]:
            got = row["orig_" + k if k.startswith("worst") else k]
            if got is None or want[k] is None:
                same = got is None and want[k] is None
            else:
                same = abs(got - want[k]) < TOL
            if not same:
                raise AssertionError(f"split {split} seed {seed} {row['arm']} {k}: {got} != saved {want[k]}")


def saved_rows(split, seed):
    paths = glob.glob(os.path.join(cap.OUT, f"cap_split{split}_seed{seed}_*[0-9a-f].json"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one capped result for split {split} seed {seed}, found {paths}")
    with open(paths[0]) as f:
        return json.load(f)["rows"]


def cap_ckpt(split, seed):
    paths = glob.glob(os.path.join(cap.OUT, f"cap_split{split}_seed{seed}_*_ckpt.pt"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one capped checkpoint for split {split} seed {seed}, found {paths}")
    return paths[0]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_cell(split, seed, x, ei, y, tr, va, te, config, rates=ls.RATES, ckpt=None):
    base = 1000 * seed + split
    test_idx = te.nonzero().view(-1)
    budgets = {r: int(r * len(cr.undirected(ei))) for r in rates}
    state = torch.load(cap_ckpt(split, seed), weights_only=False)
    models, meta, graphs = state["models"], state["meta"], state["graphs"]
    orig_keys = frozenset(graphs)
    resumed = torch.load(ckpt, weights_only=False) if ckpt and os.path.exists(ckpt) else None
    if resumed is not None and resumed["config"] != config:
        raise RuntimeError(f"{ckpt} was made with a different configuration or source; refusing to resume")
    if resumed is not None:
        new, new_meta, tau, done = resumed["models"], resumed["meta"], resumed["tau"], resumed["graphs"]
        new.update(ensembles(models))
        log.info(json.dumps({"resumed": ckpt, "graphs": len(done)}))
    else:
        new, new_meta, tau = train_baselines(models, x, ei, y, tr, va, base)
        done = {}
    payload = lambda: {"config": config, "models": {k: v for k, v in new.items() if k in TRAINED},
                       "meta": new_meta, "tau": tau, "graphs": done}
    if ckpt:
        cap.save(ckpt, payload())
    for r, b in budgets.items():
        for arm in BASELINES:
            for name, offset, mode, loss in ls.ATTACKS:
                if (arm, name, r) in done:
                    continue
                t0 = time.time()
                done[(arm, name, r)] = ls.prbcd(new[arm], x, ei, y, test_idx, b, base + offset + int(100 * r), mode, loss)
                log.info(json.dumps({"attack": [arm, name, r], "seconds": round(time.time() - t0, 1)}))
                if ckpt:
                    cap.save(ckpt, payload())
    edits = {"|".join(map(str, k)): cr.validate(ei, g, x.size(0), budgets[k[2]]) for k, g in done.items()}
    all_graphs = {**graphs, **done}
    assert len(all_graphs) == len(orig_keys) + len(done), "graph keys collide"
    rows = score({**models, **new}, {**meta, **new_meta}, all_graphs, orig_keys, x, ei, y, te, rates, split, seed)
    verify_reproduction(rows, split, seed, rates)
    chosen = cap.select(meta)
    rows.append({**next(r for r in rows if r["arm"] == chosen), "arm": "CAP*", "chosen": chosen})
    return rows, edits, tau


def check(name, ok, detail=""):
    print(name, "OK" if ok else "FAILED", detail, flush=True)
    assert ok, name


def selfcheck(x, ei, y, trm, vam, tem):
    split, seed = 0, 0
    te, tr, va = tem[split], trm[split], vam[split]
    state = torch.load(cap_ckpt(split, seed), weights_only=False)
    models, meta, graphs = state["models"], state["meta"], state["graphs"]
    orig = frozenset(graphs)
    rows = score(models, meta, graphs, orig, x, ei, y, te, ls.RATES, split, seed)
    verify_reproduction(rows, split, seed, ls.RATES)
    check("rescoring reproduces the saved capped results (all arms, both budgets)", True)
    with torch.no_grad():
        p_m, p_s = models["MLP-M"](x).argmax(-1), models["S0"](x, ei).argmax(-1)
        check("ENS a=0 predicts as MLP-M", torch.equal(LogMix(models["MLP-M"], models["S0"], 0.0)(x, ei).argmax(-1), p_m))
        check("ENS a=1 predicts as S0", torch.equal(LogMix(models["MLP-M"], models["S0"], 1.0)(x, ei).argmax(-1), p_s))
    try:
        LogMix(models["MLP-M"], models["S0"], 1.5)
        check("alpha outside [0, 1] rejected", False)
    except ValueError:
        check("alpha outside [0, 1] rejected", True)
    new, new_meta, tau = train_baselines(models, x, ei, y, tr, va, 0)
    check("TextGate tau is a finite cosine", -1 < tau < 1, round(tau, 3))
    for name, model in new.items():
        check(f"{name} validation accuracy above chance", new_meta[name]["val"] > 0.3, round(new_meta[name]["val"], 3))
        for mode in ("ste", "soft"):
            model.eval()
            if hasattr(model, "attack_mode"):
                model.attack_mode = mode
            w = torch.ones(ei.size(1), requires_grad=True)
            p = F.softmax(model(x, ei, w), -1)
            m = p[va].gather(1, y[va, None]).sum() - p[va].max(-1).values.sum()
            g = torch.autograd.grad(m, w)[0]
            if hasattr(model, "attack_mode"):
                model.attack_mode = "ste"
            check(f"{name} edge gradient finite and nonzero ({mode})", bool(torch.isfinite(g).all()) and g.norm() > 0)
    check("ENS leaves S0 in ste after attack-mode use", models["S0"].attack_mode == "ste")
    budget = int(0.2 * len(cr.undirected(ei)))
    pert = ls.prbcd(new["GCN"], x, ei, y, te.nonzero().view(-1), budget, 1, "ste", "prob_margin")
    check("baseline PRBCD graph is valid", cr.validate(ei, pert, x.size(0), budget) is not None)
    with torch.no_grad():
        before = cr.correct(new["GCN"], x, ei, y)[te].float().mean().item()
        after = cr.correct(new["GCN"], x, pert, y)[te].float().mean().item()
    check("PRBCD lowers GCN test accuracy", after < before, (round(before, 4), round(after, 4)))
    extended = {**graphs, ("GCN", "ste", 0.2): pert}
    rows2 = score({**models, "GCN": new["GCN"]}, {**meta, "GCN": new_meta["GCN"]}, extended, orig, x, ei, y, te,
                  ls.RATES, split, seed)
    verify_reproduction(rows2, split, seed, ls.RATES)
    check("extended union keeps protocol-union numbers and never raises worst-node", True)
    path = os.path.join(OUT, "_selfcheck_ckpt.pt")
    os.makedirs(OUT, exist_ok=True)
    trained = {k: v for k, v in new.items() if k in TRAINED}
    torch.save({"models": trained, "x": x, "ei": ei,
                "out": {k: v(x, ei).detach() for k, v in trained.items()}}, path)
    probe = ("import importlib.util,sys,torch;"
             f"spec=importlib.util.spec_from_file_location('capped_baselines',{os.path.abspath(__file__)!r});"
             "m=importlib.util.module_from_spec(spec);sys.modules['capped_baselines']=m;spec.loader.exec_module(m);"
             f"d=torch.load({path!r},weights_only=False);"
             "print(all(torch.allclose(v.eval()(d['x'],d['ei']),d['out'][k],atol=1e-5) for k,v in d['models'].items()))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    ok = result.stdout.strip().endswith("True")
    check("baseline checkpoint loads in a fresh process", ok, "" if ok else result.stderr[-300:])
    os.remove(path)


def code_hash():
    paths = [os.path.join(HERE, "baselines.py"), os.path.join(HERE, "..", "adaptive_robustness", "models.py")]
    return hashlib.sha256(b"".join(open(p, "rb").read() for p in paths) + cap.code_hash().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
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
    s, t0 = args.split, time.time()
    import torch_geometric
    config = {"split": s, "seed": args.seed, "rates": list(ls.RATES), "alphas": list(ALPHAS),
              "code_hash": code_hash(), "source_sha": sha(cap_ckpt(s, args.seed)), "data_sha": sha(fx.CORA),
              "threads": args.threads, "torch": torch.__version__, "pyg": torch_geometric.__version__,
              "python": platform.python_version()}
    tag = config["code_hash"][:10]
    name = f"base_split{s}_seed{args.seed}_{tag}"
    rows, edits, tau = run_cell(s, args.seed, x, ei, y, trm[s], vam[s], tem[s], config,
                                ckpt=os.path.join(OUT, f"{name}_ckpt.pt"))
    env = {**config, "seconds": round(time.time() - t0, 1), "text_tau": tau}
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, "edits": edits}, f, indent=1)


if __name__ == "__main__":
    main()
