"""Capped graph correction (experiments/capped-correction/protocol.md).

CAP(B): logits = m(x) + B * tanh(g(x, A) / B), with m the frozen graph-free MLP-M and g a STEM-GNN
model finetuned through the combined output. Edges enter only through g, so under any graph each
logit stays within B of m's (two graphs can differ by up to 2B). Nodes where m's margin exceeds 2B are
certified for every graph. A BPDA attack (exact capped forward, identity backward through tanh) is
added for capped arms so saturation can't mask gradients.

Usage (from repo root):
  python STEM-GNN/scripts/capped/run.py --selfcheck
  python STEM-GNN/scripts/capped/run.py --split 0 --seed 0
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

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s2", os.path.join(HERE, "..", "llm_stem", "stage2.py"))
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)
ls, cr, fx = s2.ls, s2.cr, s2.fx
sys.modules.update({"s2": s2, "ls": ls, "cr": cr, "fx": fx})

sys.modules.setdefault("capped_run", sys.modules[__name__])
log = logging.getLogger("capped")
ROOT = os.path.join(HERE, "..", "..", "..")
OUT = os.path.join(ROOT, "experiments", "capped-correction", "results")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
CAP_ARMS = tuple(f"CAP{q}" for q in QUANTILES)
ATTACKED = ("S0", "RES") + CAP_ARMS
BPDA_ATTACKS = (("bpda", 59, "bpda", "prob_margin"), ("bpda_tanh", 61, "bpda", "tanh_margin"))


class CapModel(nn.Module):
    """m(x) + B * tanh(g / B); B None means the uncapped residual RES. m stays frozen and in eval mode."""

    def __init__(self, mlp, g, cap):
        super().__init__()
        if cap is not None and not (cap > 0 and cap < float("inf")):
            raise ValueError(f"cap must be positive and finite, got {cap}")
        self.mlp, self.g, self.cap, self.bpda = mlp, g, cap, False
        for p in self.mlp.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.mlp.eval()
        return self

    @property
    def attack_mode(self):
        return "bpda" if self.bpda else self.g.attack_mode

    @attack_mode.setter
    def attack_mode(self, mode):
        """bpda: exact capped forward, identity backward through tanh, STE through the VQ."""
        self.bpda = mode == "bpda"
        self.g.attack_mode = "ste" if self.bpda else mode

    def correction(self, x, ei, w=None):
        g = self.g(x, ei, w)
        if self.cap is None:
            return g
        capped = self.cap * torch.tanh(g / self.cap)
        return capped.detach() + (g - g.detach()) if self.bpda else capped

    def forward(self, x, ei, w=None):
        out = self.mlp(x) + self.correction(x, ei, w)
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite logits")
        return out

    def penalty(self, coeff):
        return self.g.penalty(coeff)


CapModel.__module__ = "capped_run"  # checkpoints load from any entry point that imports this file


def margin(logits):
    top = logits.topk(2, -1).values
    return top[:, 0] - top[:, 1]


@torch.no_grad()
def certified(mlp, x, y, cap):
    """m correct and m's margin above 2B: correct under every graph."""
    logits = mlp(x)
    return (logits.argmax(-1) == y) & (margin(logits) > 2 * cap)


def finetune_cap(enc, vq, mlp, x, ei, y, tr, va, base, cap):
    """Stage 2 finetune recipe, trained through the combined output."""
    fx.seed_all(base + 500)
    g = ls.RouterCodeModel(copy.deepcopy(enc), copy.deepcopy(vq), int(y.max()) + 1, None)
    model = CapModel(mlp, g, cap)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    best, state, wait, best_epoch = -1.0, None, 0, 0
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)
        loss = F.cross_entropy(model(x, ei)[tr], y[tr]) + model.penalty(s2.COEFF)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch} (cap={cap})")
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = fx.acc(model(x, ei), y, va)
        if val > best:
            best, wait, best_epoch = val, 0, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= fx.PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, best, best_epoch


def caps_from_val(mlp, x, va):
    with torch.no_grad():
        m = margin(mlp(x))[va]
    return {f"CAP{q}": 0.5 * torch.quantile(m, q).item() for q in QUANTILES}


def train_all(x, ei, y, tr, va, pt_epochs, base):
    enc, vq = s2.pretrain(x, ei, pt_epochs, base)
    models, meta = {}, {}
    models["S0"], val, epoch = s2.finetune(enc, vq, x, ei, y, tr, va, base)
    meta["S0"] = {"val": val, "best_epoch": epoch, "cap": None}
    mlp = s2.fit_mlp(x, y, tr, va, base)
    models["MLP-M"] = mlp
    with torch.no_grad():
        meta["MLP-M"] = {"val": fx.acc(mlp(x), y, va), "cap": None}
    caps = {"RES": None, **caps_from_val(mlp, x, va)}
    for arm, cap in caps.items():
        models[arm], val, epoch = finetune_cap(enc, vq, mlp, x, ei, y, tr, va, base, cap)
        meta[arm] = {"val": val, "best_epoch": epoch, "cap": cap}
        if cap is not None:
            meta[arm]["cert_val"] = certified(mlp, x, y, cap)[va].float().mean().item()
    return models, meta


def select(meta):
    """B* maximizes mean(clean val, certified val); ties go to the smaller B."""
    scored = [((meta[a]["val"] + meta[a]["cert_val"]) / 2, -meta[a]["cap"], a) for a in CAP_ARMS]
    return max(scored)[2]


DEPENDENCIES = ("capped/run.py", "llm_stem/stage2.py", "llm_stem/run.py", "code_robustness/run.py",
                "stem_fixes/run.py", "../model/encoder.py", "../model/vq.py")


def code_hash():
    paths = [os.path.join(HERE, "..", p) for p in DEPENDENCIES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(missing)
    return hashlib.sha256(b"".join(open(p, "rb").read() for p in paths)).hexdigest()


def attacks_for(arm):
    return ls.ATTACKS + (BPDA_ATTACKS if arm in CAP_ARMS else ())


def save(ckpt, payload):
    tmp = ckpt + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt)


def run_cell(split, seed, x, ei, y, tr, va, te, pt_epochs, rates=ls.RATES, ckpt=None, config=None):
    base = 1000 * seed + split
    test_idx = te.nonzero().view(-1)
    budgets = {r: int(r * len(cr.undirected(ei))) for r in rates}
    state = torch.load(ckpt, weights_only=False) if ckpt and os.path.exists(ckpt) else None
    if state is not None and state["config"] != config:
        raise RuntimeError(f"checkpoint {ckpt} was made with a different configuration; refusing to resume")
    if state is not None:
        models, meta, sur, graphs = state["models"], state["meta"], state["sur"], state["graphs"]
        log.info(json.dumps({"resumed": ckpt, "graphs": len(graphs)}))
    else:
        models, meta = train_all(x, ei, y, tr, va, pt_epochs, base)
        fx.seed_all(base + 11)
        sur = cr.Surrogate(x.size(1), int(y.max()) + 1)
        fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
        graphs = {}
    payload = lambda: {"config": config, "models": models, "meta": meta, "sur": sur, "graphs": graphs}
    if ckpt:
        save(ckpt, payload())
    for r, b in budgets.items():
        jobs = [("sage", sur, "sage", 23, "ste", "prob_margin")] + [
            (arm, models[arm], name, offset, mode, loss) for arm in ATTACKED for name, offset, mode, loss in attacks_for(arm)]
        for arm, model, name, offset, mode, loss in jobs:
            if (arm, name, r) in graphs:
                continue
            t0 = time.time()
            graphs[(arm, name, r)] = ls.prbcd(model, x, ei, y, test_idx, b, base + offset + int(100 * r), mode, loss)
            log.info(json.dumps({"attack": [arm, name, r], "seconds": round(time.time() - t0, 1)}))
            if ckpt:
                save(ckpt, payload())
    edits = {"|".join(map(str, k)): cr.validate(ei, g, x.size(0), budgets[k[2]]) for k, g in graphs.items()}
    chosen = select(meta)
    mlp = models["MLP-M"]
    rows, pernode = [], {}
    for arm, model in models.items():
        clean = cr.correct(model, x, ei, y)
        corr = {k: cr.correct(model, x, g, y) for k, g in graphs.items()}
        cap = meta[arm]["cap"]
        row = {"arm": arm, "split": split, "seed": seed, **meta[arm], "clean": clean[te].float().mean().item(),
               "cert_test": certified(mlp, x, y, cap)[te].float().mean().item() if cap is not None else None}
        for r in rates:
            keys = [k for k in corr if k[2] <= r]
            suite = torch.stack([clean] + [corr[k] for k in keys])
            row[f"worst_node@{r}"] = suite.all(0)[te].float().mean().item()
            row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
            row[f"own_worst_node@{r}"] = torch.stack(
                [clean] + [corr[k] for k in keys if k[0] == arm]).all(0)[te].float().mean().item()
            row[f"transfer_sage@{r}"] = corr[("sage", "sage", r)][te].float().mean().item()
        rows.append(row)
        pernode[arm] = {"clean": clean, **{"|".join(map(str, k)): v for k, v in corr.items()}}
        log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    for row in rows:
        assert all(not isinstance(v, float) or v == v for v in row.values()), row  # no NaN
    rows.append({**next(r for r in rows if r["arm"] == chosen), "arm": "CAP*", "chosen": chosen})
    return rows, edits, pernode


def check(name, ok, detail=""):
    print(name, "OK" if ok else "FAILED", detail, flush=True)
    assert ok, name


def selfcheck(x, ei, y, tr, va):
    base = 0
    enc, vq = s2.pretrain(x, ei, 5, base)
    mlp = s2.fit_mlp(x, y, tr, va, base)
    caps = caps_from_val(mlp, x, va)
    print("caps", {k: round(v, 3) for k, v in caps.items()}, flush=True)
    check("caps increase with q", all(a < b for a, b in zip(list(caps.values()), list(caps.values())[1:])))
    cap = caps["CAP0.5"]
    frozen = {k: v.clone() for k, v in mlp.state_dict().items()}
    model = finetune_cap(enc, vq, mlp, x, ei, y, tr, va, base, cap)[0]
    check("MLP frozen through finetune", all(not p.requires_grad for p in mlp.parameters()))
    gen = torch.Generator().manual_seed(3)
    n = x.size(0)
    rand = torch.randint(0, n, (2, 4000), generator=gen)
    rand = rand[:, rand[0] != rand[1]]
    graphs = {"clean": ei, "empty": torch.empty((2, 0), dtype=torch.long),
              "random": torch.cat([ei, rand, rand.flip(0)], 1)}
    cert = certified(mlp, x, y, cap)
    with torch.no_grad():
        base_logits = mlp(x)
        for name, g in graphs.items():
            out = model(x, g)
            check(f"correction bounded by B ({name})", bool(((out - base_logits).abs() <= cap).all()))
            check(f"certified nodes keep m's prediction ({name})",
                  torch.equal(out.argmax(-1)[cert], base_logits.argmax(-1)[cert]), int(cert.sum()))
    check("m unchanged after training and evaluation",
          all(torch.equal(v, mlp.state_dict()[k]) for k, v in frozen.items()))
    check("certified fraction in (0, 1)", 0 < cert[va].float().mean().item() < 1, round(cert[va].float().mean().item(), 3))
    for mode in ("ste", "soft", "soft_t1"):
        model.attack_mode = mode
        check(f"attack mode {mode} reaches g", model.g.attack_mode == mode)
        w = torch.ones(ei.size(1), requires_grad=True)
        p = F.softmax(model(x, ei, w), -1)
        m = p[va].gather(1, y[va, None]).sum() - p[va].max(-1).values.sum()
        grad = torch.autograd.grad(m, w)[0]
        check(f"edge gradient finite and nonzero ({mode})", bool(torch.isfinite(grad).all()) and grad.norm() > 0)
    model.attack_mode = "bpda"
    check("bpda sets ste on g", model.g.attack_mode == "ste" and model.attack_mode == "bpda")
    w = torch.ones(ei.size(1), requires_grad=True)
    with torch.no_grad():
        exact = model.correction(x, ei)
    out = model.correction(x, ei, w)
    check("bpda forward is the exact capped forward", torch.allclose(out, exact, atol=1e-6))
    gb = torch.autograd.grad(out.sum(), w)[0]
    model.attack_mode = "ste"
    check("bpda edge gradient finite and nonzero", bool(torch.isfinite(gb).all()) and gb.norm() > 0)
    check("reset leaves bpda off", not model.bpda)
    try:
        CapModel(mlp, model.g, 0.0)
        check("zero cap rejected", False)
    except ValueError:
        check("zero cap rejected", True)
    check("code hash covers every dependency", len(code_hash()) == 64)
    res = finetune_cap(enc, vq, mlp, x, ei, y, tr, va, base, None)[0]
    again = finetune_cap(enc, vq, mlp, x, ei, y, tr, va, base, cap)[0]
    with torch.no_grad():
        check("finetune repeatable", torch.allclose(model(x, ei), again(x, ei), atol=1e-4))
        check("RES is uncapped", bool(((res(x, ei) - base_logits).abs() >= cap).any()))
    s0 = s2.finetune(enc, vq, x, ei, y, tr, va, base)[0]
    meta = {a: {"val": 0.7, "cert_val": 0.5, "cap": c} for a, c in caps.items()}
    check("selection ties go to the smallest B", select(meta) == "CAP0.1")
    import io
    buf = io.BytesIO()
    torch.save({"models": {"CAP": model, "S0": s0}}, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=False)["models"]
    with torch.no_grad():
        check("checkpoint round trip", torch.equal(back["CAP"](x, ei), model(x, ei)))
    path = os.path.join(OUT, "_selfcheck_ckpt.pt")
    os.makedirs(OUT, exist_ok=True)
    torch.save({"CAP": model, "x": x, "ei": ei, "out": model(x, ei).detach()}, path)
    import subprocess
    probe = ("import importlib.util,sys,torch;"
             f"spec=importlib.util.spec_from_file_location('capped_run',{os.path.abspath(__file__)!r});"
             "m=importlib.util.module_from_spec(spec);sys.modules['capped_run']=m;spec.loader.exec_module(m);"
             f"d=torch.load({path!r},weights_only=False);d['CAP'].eval();"
             "print(torch.allclose(d['CAP'](d['x'],d['ei']),d['out'],atol=1e-5))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    ok = result.stdout.strip().endswith("True")
    check("checkpoint loads in a fresh process", ok, "" if ok else result.stderr[-300:])
    os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pt_epochs", type=int, default=ls.PT_EPOCHS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x, ei, y, trm, vam, tem = fx.load()
    s = args.split
    if args.selfcheck:
        return selfcheck(x, ei, y, trm[s], vam[s])
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    import torch_geometric
    config = {"pt_epochs": args.pt_epochs, "quantiles": list(QUANTILES), "code_hash": code_hash(),
              "pyg": torch_geometric.__version__, "torch": torch.__version__,
              "python": platform.python_version(), "threads": args.threads,
              "data_sha": hashlib.sha256(open(fx.CORA, "rb").read()).hexdigest()}
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f"cap_split{s}_seed{args.seed}_{tag}"
    rows, edits, pernode = run_cell(s, args.seed, x, ei, y, trm[s], vam[s], tem[s], args.pt_epochs,
                                    ckpt=os.path.join(OUT, f"{name}_ckpt.pt"), config=config)
    env = {"python": platform.python_version(), "torch": torch.__version__, "threads": args.threads,
           "seconds": round(time.time() - t0, 1), **config}
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, "edits": edits}, f, indent=1)
    torch.save({"pernode": pernode, "test_mask": tem[s], "y": y}, os.path.join(OUT, f"{name}_pernode.pt"))


if __name__ == "__main__":
    main()
