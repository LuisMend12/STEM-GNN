"""Controls for the Stage 2 low-label Qwen-prior result (experiments/llm-stem/lowlabel_controls_protocol.md).

Per cell, reuses the label-free S0 encoder and VQ from the Stage 2 checkpoint (tag STAGE2_TAG) and finetunes,
with 3 labels per class, in two validation regimes (V500: Stage 2's validation set; V21: 3 validation labels
per class):
  S0    no KL target
  S-A   KL to the calibrated Qwen prior, rotations {0, 2, 4} (Stage 2's S-A; the rotation choice used an audit)
  S-A0  KL to the calibrated Qwen prior, rotation 0 only (no audit: the strict-budget arm)
  S-U   KL to the uniform distribution (uniform-target cross-entropy regularization)
  S-T   KL to a calibrated MiniLM-MLP teacher trained on the same labels (no Qwen)
  MIX   S0 mixed with the S-A prior; QWEN, QWEN0 and MLP-M are the teachers alone
No attacks. Text never touches edges.

Usage (from repo root):
  python STEM-GNN/scripts/llm_stem/lowlabel_controls.py --selfcheck
  python STEM-GNN/scripts/llm_stem/lowlabel_controls.py --split 0 --seed 0
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s2", os.path.join(HERE, "stage2.py"))
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)
ls, cr, fx = s2.ls, s2.cr, s2.fx
sys.modules.update({"s2": s2, "ls": ls, "cr": cr, "fx": fx})
# Stage 2 ran as a script, so its own classes were pickled as __main__.<name>.
for _name in ("MLP", "Mix", "PriorOnly"):
    setattr(sys.modules["__main__"], _name, getattr(s2, _name))

log = logging.getLogger("lowlabel_controls")
OUT = os.path.join(s2.ROOT, "experiments", "llm-stem", "results_lowlabel_controls")
STAGE2_TAG = "ce407cf8cd"
STAGE2_CONFIG = {"arms": ["MIX", "S-A", "S-AP", "S-F", "S-V", "S-VA", "S-VP", "S0"], "pt_epochs": 200,
                 "prior_perms": [0, 2, 4],
                 "audit_hash": "8738e1785c858f7651e07a9018f1fc4462529c34b21dd36191aeaacb0c293db4",
                 "code_hash": "5db06d7d34d53019939837eeec1a079eefd36f482bb19236062efff2846be644"}
SMALL_VAL_PER_CLASS = 3
REGIMES = ("V500", "V21")
TRAINED = ("S0", "S-A", "S-A0", "S-U", "S-T")
ARMS = TRAINED + ("MIX", "QWEN", "QWEN0", "MLP-M")


def small_val_mask(y, va, base):
    """3 validation nodes per class, drawn from the regular validation set."""
    gen = torch.Generator().manual_seed(base + 80_001)
    mask = torch.zeros_like(va)
    for c in range(int(y.max()) + 1):
        pool = (va & (y == c)).nonzero().view(-1)
        if pool.numel() < SMALL_VAL_PER_CLASS:
            raise ValueError(f"class {c} has only {pool.numel()} validation nodes")
        mask[pool[torch.randperm(pool.numel(), generator=gen)[:SMALL_VAL_PER_CLASS]]] = True
    return mask


def stage2_paths(split, seed):
    stem = os.path.join(s2.OUT, f"stage2_split{split}_seed{seed}_{STAGE2_TAG}")
    for p in (stem + "_ckpt.pt", stem + ".json"):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return stem + "_ckpt.pt", stem + ".json"


def load_stage2(split, seed):
    """Encoder and VQ from the Stage 2 checkpoint, plus its saved rows; both must carry STAGE2_CONFIG."""
    ckpt_path, json_path = stage2_paths(split, seed)
    state = torch.load(ckpt_path, weights_only=False)
    with open(json_path) as f:
        blob = json.load(f)
    if state["config"] != STAGE2_CONFIG or {k: blob["env"][k] for k in STAGE2_CONFIG} != STAGE2_CONFIG:
        raise RuntimeError(f"Stage 2 checkpoint or result for split {split} seed {seed} has a different configuration")
    enc, vq = state["encoders"]["S0"]
    return enc, vq, {r["arm"]: r for r in blob["rows"]}


def teachers(x, y, low, va, priors, base):
    """Qwen priors and the MiniLM-MLP teacher, all calibrated on this regime's validation only."""
    qwen, t_qwen = s2.calibrate(priors["S-A"], y, va)
    qwen0, t_qwen0 = s2.calibrate(priors["S-A0"], y, va)
    mlp = s2.fit_mlp(x, y, low, va, base)
    with torch.no_grad():
        mlp_prior, t_mlp = s2.calibrate(mlp(x), y, va)
    n_cls = int(y.max()) + 1
    uniform = torch.full((x.size(0), n_cls), 1.0 / n_cls)
    return ({"S-A": qwen, "S-A0": qwen0, "S-U": uniform, "S-T": mlp_prior}, mlp,
            {"qwen": t_qwen, "qwen0": t_qwen0, "mlp": t_mlp})


def train_regime(enc, vq, x, ei, y, low, va, priors, base):
    targets, mlp, temps = teachers(x, y, low, va, priors, base)
    anchor = ~(low | va)
    models, meta = {}, {}
    for arm in TRAINED:
        prior = targets.get(arm)
        models[arm], val, epoch = s2.finetune(enc, vq, x, ei, y, low, va, base, prior,
                                              anchor if prior is not None else None)
        meta[arm] = {"val": val, "best_epoch": epoch}
    models["MIX"] = s2.fit_mix(models["S0"], targets["S-A"], x, ei, y, va)
    models["QWEN"] = s2.PriorOnly(targets["S-A"]).eval()
    models["QWEN0"] = s2.PriorOnly(targets["S-A0"]).eval()
    models["MLP-M"] = mlp
    with torch.no_grad():
        for arm in ("MIX", "QWEN", "QWEN0", "MLP-M"):
            meta[arm] = {"val": fx.acc(models[arm](x, ei), y, va)}
    meta["MIX"]["alpha"] = models["MIX"].alpha
    return models, meta, temps, targets


@torch.no_grad()
def predictions(models, x, ei):
    return {arm: m(x, ei).argmax(-1) for arm, m in models.items()}


def run_cell(split, seed, x, ei, y, tr, va, te, priors):
    base = 1000 * seed + split
    enc, vq, ref = load_stage2(split, seed)
    low = s2.low_label_mask(y, tr, base)
    masks = {"V500": va, "V21": small_val_mask(y, va, base)}
    rows, preds = [], {}
    for regime in REGIMES:
        models, meta, temps, _ = train_regime(enc, vq, x, ei, y, low, masks[regime], priors, base)
        preds[regime] = predictions(models, x, ei)
        for arm in ARMS:
            acc = (preds[regime][arm] == y)[te].float().mean().item()
            row = {"arm": arm, "regime": regime, "split": split, "seed": seed, "low_clean": acc,
                   "n_val": int(masks[regime].sum()), **{f"t_{k}": v for k, v in temps.items()}, **meta[arm]}
            rows.append(row)
            log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    drift = {arm: abs(next(r["low_clean"] for r in rows if r["arm"] == arm and r["regime"] == "V500")
                      - ref[arm]["low_clean"]) for arm in ("S0", "S-A", "MLP-M", "QWEN")}
    log.info(json.dumps({"stage2_drift": drift}))
    return rows, preds, drift, low, masks["V21"]


def check(name, ok, detail=""):
    print(name, "OK" if ok else "FAILED", detail, flush=True)
    assert ok, name


def selfcheck(x, ei, y, trm, vam, tem, priors):
    split, seed = 0, 0
    base = 1000 * seed + split
    tr, va, te = trm[split], vam[split], tem[split]
    enc, vq, _ = load_stage2(split, seed)
    check("Stage 2 checkpoint loads with the expected configuration", enc is not None and vq is not None)
    low = s2.low_label_mask(y, tr, base)
    small = small_val_mask(y, va, base)
    n_cls = int(y.max()) + 1
    check("V21 has 3 per class inside validation", int(small.sum()) == SMALL_VAL_PER_CLASS * n_cls
          and not bool((small & ~va).any())
          and all(int((small & (y == c)).sum()) == SMALL_VAL_PER_CLASS for c in range(n_cls)))
    check("V21 mask repeatable", torch.equal(small, small_val_mask(y, va, base)))
    check("low-label mask disjoint from V21 and test", not bool((low & (small | te)).any()))
    print("S-A0 and S-A prior logits differ:", not torch.equal(priors["S-A"], priors["S-A0"]), flush=True)
    for name, v in (("V500", va), ("V21", small)):
        targets, _, temps = teachers(x, y, low, v, priors, base)
        anchor = ~(low | v)
        check(f"{name} anchor excludes its labels", not bool((anchor & (low | v)).any()), int(anchor.sum()))
        check(f"{name} every target is a distribution",
              all(torch.allclose(t.sum(-1), torch.ones(t.size(0))) and bool((t >= 0).all()) for t in targets.values()),
              temps)
    scrambled = y.clone()
    hidden = ~(low | small)
    scrambled[hidden] = (y[hidden] + 1) % n_cls
    t_a, _, _ = teachers(x, y, low, small, priors, base)
    t_b, _, _ = teachers(x, scrambled, low, small, priors, base)
    check("V21 teachers ignore labels outside train and V21 (given fixed masks)", all(torch.equal(t_a[k], t_b[k]) for k in t_a))
    # The KL term is live: finite, nonzero gradient into the model from every target (a fixture, not an outcome).
    probe = ls.RouterCodeModel(copy.deepcopy(enc), copy.deepcopy(vq), n_cls, None).eval()
    anchor = ~(low | small)
    for arm in ("S-A", "S-A0", "S-U", "S-T"):
        log_p = F.log_softmax(probe(x, ei)[anchor], -1)
        kl = F.kl_div(log_p, t_a[arm][anchor], reduction="batchmean")
        grads = [g for g in torch.autograd.grad(kl, [q for q in probe.parameters() if q.requires_grad]) if g is not None]
        check(f"{arm} KL gradient finite and nonzero", bool(torch.isfinite(kl)) and len(grads) > 0
              and all(torch.isfinite(g).all() for g in grads) and sum(g.norm() for g in grads) > 0)
    models, _, _, _ = train_regime(enc, vq, x, ei, y, low, small, priors, base)
    with torch.no_grad():
        base_logits = models["S0"](x, ei)
        print("max |logit - S0| per KL arm (V21, descriptive):",
              {a: round((models[a](x, ei) - base_logits).abs().max().item(), 4) for a in ("S-A", "S-A0", "S-U", "S-T")},
              flush=True)
    p = predictions(models, x, ei)
    rows, preds, drift, _, _ = run_cell(split, seed, x, ei, y, tr, va, te, priors)
    check("V21 training repeatable", all(torch.equal(p[a], preds["V21"][a]) for a in ARMS))
    check("rows for every arm and regime", len(rows) == len(ARMS) * len(REGIMES))
    print("Stage 2 drift (points):", {k: round(100 * v, 3) for k, v in drift.items()}, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    blob = b"".join(open(os.path.join(HERE, f), "rb").read() for f in ("lowlabel_controls.py", "stage2.py", "run.py"))
    return hashlib.sha256(blob).hexdigest()


def load_priors(texts_hash):
    """S-A: the audited rotations, verified as Stage 2 did. S-A0: rotation 0 only, no audit."""
    with open(s2.AUDIT_FILE, "rb") as f:
        raw = f.read()
    decision = json.loads(raw)
    expected = [0, 2, 4] if decision["gain_over_identity"] >= s2.AUDIT_GAIN else [0]
    if decision.get("text_hash") != texts_hash or decision["prior_perms"] != expected:
        raise RuntimeError("prior audit manifest doesn't match these texts or its own rule")
    audit_hash = hashlib.sha256(raw).hexdigest()
    if audit_hash != STAGE2_CONFIG["audit_hash"] or decision["prior_perms"] != STAGE2_CONFIG["prior_perms"]:
        raise RuntimeError("prior audit differs from the one Stage 2 used")
    priors = {"S-A": s2.load_prior_logits(texts_hash, decision["prior_perms"]),
              "S-A0": s2.load_prior_logits(texts_hash, [0])}
    for name, p in priors.items():
        if p.dim() != 2 or not bool(torch.isfinite(p).all()):
            raise RuntimeError(f"{name} prior logits must be a finite 2-D tensor, got shape {tuple(p.shape)}")
    prior_hash = hashlib.sha256(b"".join(priors[k].numpy().tobytes() for k in sorted(priors))).hexdigest()
    return priors, audit_hash, prior_hash


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
    texts = list(torch.load(fx.CORA, weights_only=False).raw_texts)
    texts_hash = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    priors, audit_hash, prior_hash = load_priors(texts_hash)
    if priors["S-A"].shape != (x.size(0), int(y.max()) + 1):
        raise RuntimeError(f"prior shape {tuple(priors['S-A'].shape)} doesn't match nodes x classes")
    if args.selfcheck:
        return selfcheck(x, ei, y, trm, vam, tem, priors)
    os.makedirs(OUT, exist_ok=True)
    s, t0 = args.split, time.time()
    rows, preds, drift, low, small = run_cell(s, args.seed, x, ei, y, trm[s], vam[s], tem[s], priors)
    tag = code_hash()[:10]
    name = f"llc_split{s}_seed{args.seed}_{tag}"
    env = {"python": platform.python_version(), "torch": torch.__version__, "threads": args.threads,
           "seconds": round(time.time() - t0, 1), "code_hash": tag, "stage2_tag": STAGE2_TAG,
           "audit_hash": audit_hash, "texts_hash": texts_hash, "prior_hash": prior_hash,
           "stage2_ckpt_sha": sha(stage2_paths(s, args.seed)[0])}
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, "stage2_drift": drift,
                   "low_label_nodes": low.nonzero().view(-1).tolist(),
                   "v21_nodes": small.nonzero().view(-1).tolist()}, f, indent=1)
    torch.save({"preds": preds, "y": y, "test_mask": tem[s]}, os.path.join(OUT, f"{name}_preds.pt"))


if __name__ == "__main__":
    main()
