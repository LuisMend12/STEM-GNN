"""Stage 2: LLM at the input, VQ and head of STEM-GNN (experiments/llm-stem/protocol.md, Amendment 2).

Attacked arms: S0, S-F (Qwen input), S-V (Qwen distillation at vq.project_in), S-VP (wrong-node target),
S-A (Qwen prior anchor), S-AP (wrong-node prior), S-VA (V + A), MIX (S0 mixed with the prior).
Graph-free baselines, immune to edge attacks by construction: MLP-M, MLP-Q, QWEN.

The router reads message-passed z in every MoE arm. Text never weights, selects, adds or drops edges.

Usage (from repo root):
  python STEM-GNN/scripts/llm_stem/stage2.py --audit
  python STEM-GNN/scripts/llm_stem/stage2.py --selfcheck
  python STEM-GNN/scripts/llm_stem/stage2.py --split 0 --seed 0
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import logging
import os
import platform
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature, negative_sampling

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ls", os.path.join(HERE, "run.py"))
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)
cr, fx = ls.cr, ls.fx

log = logging.getLogger("llm_stem2")
ROOT = os.path.join(HERE, "..", "..", "..")
OUT = os.path.join(ROOT, "experiments", "llm-stem", "results_stage2")
PRIORS = os.path.join(ROOT, "experiments", "llm-robustness", "results")
PRIOR_NAME = "qwen_prior_Qwen2.5-1.5B-Instruct_v2_perm{}{}.pt"
COEFF = 0.001  # frozen in Stage 1 on S0 clean validation
ANCHOR_LAMBDA, SEMANTIC_WEIGHT = 1.0, 1.0
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LOW_LABEL_PER_CLASS = 3
AUDIT_GAIN = 0.02
STUDY_SPLITS = (0, 1, 2)
AUDIT_FILE = os.path.join(OUT, "prior_audit.json")
ATTACKED = ("S0", "S-F", "S-V", "S-VP", "S-A", "S-AP", "S-VA", "MIX")
FREE = ("MLP-M", "MLP-Q", "QWEN")


def permuted(tensor, base):
    """The same values on the wrong nodes: same distribution, independent of each node's own text."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base + 90_001)
        return tensor[torch.randperm(tensor.size(0))]


def load_prior_logits(texts_hash, perms):
    """Letter logits in class order, averaged over the option rotations in perms."""
    logits = []
    for k in perms:
        blob = torch.load(os.path.join(PRIORS, PRIOR_NAME.format(k, "")), weights_only=False)
        if blob["meta"]["text_hash"] != texts_hash:
            raise RuntimeError("Qwen prior was extracted from different texts or a different node order")
        if torch.isnan(blob["logits"]).any():
            raise RuntimeError(f"Qwen prior for rotation {k} is incomplete")
        logits.append(blob["logits"].float())
    return torch.stack(logits).mean(0)


def calibrate(logits, y, va):
    """Temperature chosen on validation NLL only."""
    nll = {t: F.cross_entropy(logits[va] / t, y[va]).item() for t in TEMPERATURES}
    t = min(nll, key=nll.get)
    return F.softmax(logits / t, -1), t


def seeded(seed, fn, *args, **kwargs):
    """Run fn on its own RNG stream, so input size can't shift other streams."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return fn(*args, **kwargs)


def make_predictor(vq, out_dim, seed):
    in_dim = vq.project_in(torch.zeros(1, ls.HID)).size(1)  # heads x codebook_dim, 512 here
    return seeded(seed, nn.Linear, in_dim, out_dim)


def semantic_loss(predictor, vq, z, target):
    return (1 - F.cosine_similarity(predictor(vq.project_in(z)), target, dim=-1)).mean()


def pretrain(x, ei, epochs, pt_seed, target=None):
    """Stage 1 recipe for S0. target adds a cosine distillation loss on a linear map of
    vq.project_in(z); the predictor is discarded. Each module and each random step has its own
    seed, so S-F's input size and S-V's predictor leave every other stream unchanged."""
    enc = ls.build_encoder("S0", x.size(1), None, init_seed=pt_seed)
    feat_dec = seeded(pt_seed + 3, nn.Linear, ls.HID, x.size(1))
    topo_dec = seeded(pt_seed + 4, fx.InnerProductDecoder, hidden_dim=ls.HID, output_dim=ls.HID)
    vq = seeded(pt_seed + 5, fx.make_vq, ls.HID, True, fx.X2_ARMS["C-ema"][2])
    mods = [enc, feat_dec, topo_dec, vq]
    predictor = None
    if target is not None:
        predictor = make_predictor(vq, target.size(1), pt_seed + 6)
        mods.append(predictor)
    opt = torch.optim.AdamW([p for m in mods for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        for m in mods:
            m.train()
        s = (pt_seed * 100_003 + epoch) * 10
        aug_x = seeded(s + 1, mask_feature, x, p=0.2)[0]
        aug_ei = seeded(s + 2, dropout_adj, ei, p=0.2, force_undirected=True, num_nodes=x.size(0))[0]
        neg = seeded(s + 3, negative_sampling, ei, x.size(0))
        z = seeded(s + 4, enc, aug_x, aug_ei)
        q, _, vq_loss, _ = seeded(s + 5, vq, z)
        loss = F.mse_loss(feat_dec(q), x) + fx.topo_loss_fn(topo_dec, q, ei, neg) + vq_loss.sum()
        if predictor is not None:
            loss = loss + SEMANTIC_WEIGHT * semantic_loss(predictor, vq, z, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for m in mods:
        m.eval()
    enc._last_env_reg = None
    return enc, vq


def finetune(enc, vq, x, ei, y, tr, va, base, prior=None, anchor_mask=None):
    """Stage 1 finetune. prior adds KL(prior || model) on anchor_mask through a deterministic forward."""
    fx.seed_all(base + 500)
    model = ls.RouterCodeModel(copy.deepcopy(enc), copy.deepcopy(vq), int(y.max()) + 1, None)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    best, state, wait, best_epoch = -1.0, None, 0, 0
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)
        loss = F.cross_entropy(model(x, ei)[tr], y[tr]) + model.penalty(COEFF)
        if prior is not None:
            model.eval()
            log_p = F.log_softmax(model(x, ei)[anchor_mask], -1)
            model.train()
            loss = loss + ANCHOR_LAMBDA * F.kl_div(log_p, prior[anchor_mask], reduction="batchmean")
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


class Mix(nn.Module):
    """Prediction-level mixture; edges reach it only through the base model."""

    def __init__(self, base, prior, alpha):
        super().__init__()
        self.base, self.alpha = base, alpha
        self.register_buffer("prior", prior)

    @property
    def attack_mode(self):
        return self.base.attack_mode

    @attack_mode.setter
    def attack_mode(self, mode):
        self.base.attack_mode = mode

    def forward(self, x, ei, w=None):
        p = (1 - self.alpha) * F.softmax(self.base(x, ei, w), -1) + self.alpha * self.prior
        return p.clamp_min(1e-12).log()


def fit_mix(base, prior, x, ei, y, va):
    """alpha chosen on clean validation accuracy; ties go to the smallest alpha."""
    scored = []
    for a in ALPHAS:
        mix = Mix(base, prior, a).eval()
        with torch.no_grad():
            scored.append((fx.acc(mix(x, ei), y, va), -a))
    return Mix(base, prior, -max(scored)[1]).eval()


class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.lin1, self.norm, self.lin2 = nn.Linear(in_dim, ls.HID), nn.BatchNorm1d(ls.HID), nn.Linear(ls.HID, n_cls)

    def forward(self, x, ei=None, w=None):
        return self.lin2(F.dropout(F.relu(self.norm(self.lin1(x))), fx.DROPOUT, self.training))


class PriorOnly(nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.register_buffer("prior", prior)

    def forward(self, x, ei=None, w=None):
        return self.prior.clamp_min(1e-12).log()


def fit_mlp(x, y, tr, va, base):
    fx.seed_all(base + 13)
    model = MLP(x.size(1), int(y.max()) + 1)
    fx.fit(list(model.parameters()), lambda: model(x), y, tr, va, modules=(model,))
    model.eval()
    return model


def low_label_mask(y, tr, base):
    """3 training nodes per class; validation and test masks stay the same."""
    gen = torch.Generator().manual_seed(base + 70_001)
    mask = torch.zeros_like(tr)
    for c in range(int(y.max()) + 1):
        pool = (tr & (y == c)).nonzero().view(-1)
        assert pool.numel() >= LOW_LABEL_PER_CLASS, (c, pool.numel())
        mask[pool[torch.randperm(pool.numel(), generator=gen)[:LOW_LABEL_PER_CLASS]]] = True
    return mask


def train_all(x, qwen, ei, y, tr, va, prior, pt_epochs, base, encoders=None):
    """Every model for one label setting. Encoders are pretrained once, without labels, and reused."""
    wrong_q, wrong_p = permuted(qwen, base), permuted(prior, base)
    if encoders is None:
        encoders = {"S0": pretrain(x, ei, pt_epochs, base), "S-F": pretrain(qwen, ei, pt_epochs, base),
                    "S-V": pretrain(x, ei, pt_epochs, base, qwen), "S-VP": pretrain(x, ei, pt_epochs, base, wrong_q)}
    anchor_mask = ~(tr | va)  # validation stays for calibration and checkpoint selection only
    plan = {"S0": ("S0", x, None), "S-F": ("S-F", qwen, None), "S-V": ("S-V", x, None), "S-VP": ("S-VP", x, None),
            "S-A": ("S0", x, prior), "S-AP": ("S0", x, wrong_p), "S-VA": ("S-V", x, prior)}
    models, meta = {}, {}
    for arm, (key, feats, teacher) in plan.items():
        enc, vq = encoders[key]
        models[arm], val, best_epoch = finetune(enc, vq, feats, ei, y, tr, va, base, teacher, anchor_mask)
        meta[arm] = {"val": val, "best_epoch": best_epoch, "x": feats}
    models["MIX"] = fit_mix(models["S0"], prior, x, ei, y, va)
    models["QWEN"] = PriorOnly(prior).eval()
    for arm, feats in (("MLP-M", x), ("MLP-Q", qwen)):
        models[arm] = fit_mlp(feats, y, tr, va, base)
        meta[arm] = {"x": feats}
    meta["MIX"], meta["QWEN"] = {"alpha": models["MIX"].alpha, "x": x}, {"x": x}
    with torch.no_grad():
        for arm in ("MIX", "QWEN", "MLP-M", "MLP-Q"):
            meta[arm]["val"] = fx.acc(models[arm](meta[arm]["x"], ei), y, va)
    return models, meta, encoders


def attack_suite(models, meta, sur, x, ei, y, test_idx, budgets, base, arms):
    graphs = {}
    for r, b in budgets.items():
        graphs[("sage", "sage", r)] = ls.prbcd(sur, x, ei, y, test_idx, b, base + 23 + int(100 * r), "ste", "prob_margin")
        for arm in arms:
            if arm == "MIX" and models[arm].alpha == 1.0:
                continue  # all prior: edges can't reach it
            for name, offset, mode, loss in ls.ATTACKS:
                t0 = time.time()
                graphs[(arm, name, r)] = ls.prbcd(models[arm], meta[arm]["x"], ei, y, test_idx, b,
                                                  base + offset + int(100 * r), mode, loss)
                log.info(json.dumps({"attack": [arm, name, r], "seconds": round(time.time() - t0, 1)}))
    return graphs


def run_cell(split, seed, x, qwen, ei, y, tr, va, te, logits, pt_epochs, arms=ATTACKED, rates=ls.RATES, ckpt=None):
    """ckpt holds trained models and candidate graphs, so a rerun resumes on the same models."""
    base = 1000 * seed + split
    test_idx = te.nonzero().view(-1)
    budgets = {r: int(r * len(cr.undirected(ei))) for r in rates}
    prior, temperature = calibrate(logits, y, va)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, weights_only=False)
        models, meta, graphs, encoders = state["models"], state["meta"], state["graphs"], state["encoders"]
        log.info(json.dumps({"resumed": ckpt}))
    else:
        models, meta, encoders = train_all(x, qwen, ei, y, tr, va, prior, pt_epochs, base)
        fx.seed_all(base + 11)
        sur = cr.Surrogate(x.size(1), int(y.max()) + 1)
        fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
        graphs = attack_suite(models, meta, sur, x, ei, y, test_idx, budgets, base, arms)
        if ckpt:
            torch.save({"models": models, "meta": meta, "graphs": graphs, "encoders": encoders}, ckpt)
    edits = {"|".join(map(str, k)): cr.validate(ei, g, x.size(0), budgets[k[2]]) for k, g in graphs.items()}
    low = low_label_mask(y, tr, base)
    low_models, low_meta, _ = train_all(x, qwen, ei, y, low, va, prior, pt_epochs, base, encoders)
    rows, pernode = [], {}
    for arm, model in models.items():
        feats = meta[arm]["x"]
        clean = cr.correct(model, feats, ei, y)
        corr = {k: cr.correct(model, feats, g, y) for k, g in graphs.items()}
        row = {"arm": arm, "split": split, "seed": seed, "temperature": temperature,
               **{k: v for k, v in meta[arm].items() if k != "x"}, "clean": clean[te].float().mean().item(),
               "low_clean": cr.correct(low_models[arm], low_meta[arm]["x"], ei, y)[te].float().mean().item(),
               "low_val": low_meta[arm]["val"], "low_alpha": low_meta[arm].get("alpha"),
               "low_best_epoch": low_meta[arm].get("best_epoch")}
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
    return rows, edits, pernode, prior.argmax(-1), low.nonzero().view(-1)


def audit(y, vam, texts_hash):
    """Amendment 1 rule of ../llm-anchor on the audit sample's validation nodes, pooled over the
    study splits {0, 1, 2}. A pooled node may be a test node in another split; declared. The
    decision is frozen to AUDIT_FILE, which every cell job reads."""
    logits = []
    for k in range(7):
        blob = torch.load(os.path.join(PRIORS, PRIOR_NAME.format(k, "_audit")), weights_only=False)
        assert blob["meta"]["text_hash"] == texts_hash
        logits.append(blob["logits"].float())
    logits = torch.stack(logits)
    have = ~torch.isnan(logits).any(-1).any(0)
    val = torch.stack([vam[s] for s in STUDY_SPLITS])
    assert val.dtype == torch.bool and val.shape == (len(STUDY_SPLITS), y.size(0))
    nodes = (have & val.any(0)).nonzero().view(-1)
    assert nodes.numel() > 0
    per_rotation = [(logits[k, nodes].argmax(-1) == y[nodes]).float().mean().item() for k in range(7)]
    average = (logits[:, nodes].mean(0).argmax(-1) == y[nodes]).float().mean().item()
    gain = average - per_rotation[0]
    decision = {"audit_val_nodes": nodes.tolist(), "per_rotation": per_rotation, "average": average,
                "gain_over_identity": gain, "prior_perms": [0, 2, 4] if gain >= AUDIT_GAIN else [0]}
    with open(AUDIT_FILE, "w") as f:
        json.dump(decision, f, indent=1)
    print(json.dumps({k: v for k, v in decision.items() if k != "audit_val_nodes"} | {"n": nodes.numel()}), flush=True)


def check(name, ok, detail=""):
    print(name, "OK" if ok else "FAILED", detail, flush=True)
    assert ok, name


def selfcheck(x, qwen, ei, y, tr, va, logits):
    prior, temperature = calibrate(logits, y, va)
    check("prior is a distribution", torch.allclose(prior.sum(-1), torch.ones(prior.size(0))), f"T={temperature}")
    base = 0
    enc, vq = pretrain(x, ei, 5, base)
    # Distillation gradient reaches project_in and the encoder layers.
    predictor = make_predictor(vq, qwen.size(1), 6)
    enc.train()
    loss = semantic_loss(predictor, vq, enc(x, ei), qwen)
    grads = torch.autograd.grad(loss, [vq.project_in.weight] + list(enc.layers.parameters()), allow_unused=True)
    check("semantic gradient reaches project_in and every encoder layer",
          all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in grads))
    enc.eval()
    enc._last_env_reg = None  # the training forward above stored a non-leaf tensor
    # S-F's input size leaves shared module inits and graph corruptions unchanged.
    a, b = pretrain(x, ei, 0, base), pretrain(qwen, ei, 0, base)
    check("S-F shares the VQ init", all(torch.equal(p, q) for p, q in zip(a[1].state_dict().values(), b[1].state_dict().values())))
    torch.manual_seed(1)
    seeded(10, mask_feature, x, p=0.2)
    after_minilm = torch.rand(3)
    torch.manual_seed(1)
    seeded(10, mask_feature, qwen, p=0.2)
    check("feature masking leaves the outer RNG stream unchanged", torch.equal(after_minilm, torch.rand(3)))
    enc_f, vq_f = pretrain(qwen, ei, 5, base)
    with torch.no_grad():
        model = ls.RouterCodeModel(enc_f, vq_f, int(y.max()) + 1, None).eval()
        _, idx_real, _, _ = copy.deepcopy(vq_f).eval()(enc_f(qwen, ei))
        check("S-F codes match real VQ", torch.equal(idx_real, model.forward_full(qwen, ei)[3]))
    m0 = finetune(enc, vq, x, ei, y, tr, va, base)[0]
    m1 = finetune(enc, vq, x, ei, y, tr, va, base)[0]
    with torch.no_grad():
        check("finetune repeatable", torch.allclose(m0(x, ei), m1(x, ei), atol=1e-4))
    anchor_mask = ~(tr | va)
    check("anchor mask excludes train and val", not bool((anchor_mask & (tr | va)).any()), int(anchor_mask.sum()))
    ma = finetune(enc, vq, x, ei, y, tr, va, base, prior, anchor_mask)[0]
    with torch.no_grad():
        check("anchor changes predictions", not torch.allclose(m0(x, ei), ma(x, ei), atol=1e-4))
        check("MIX alpha 0 == base", torch.equal(Mix(m0, prior, 0.0)(x, ei).argmax(-1), m0(x, ei).argmax(-1)))
        check("MIX alpha 1 == prior", torch.equal(Mix(m0, prior, 1.0)(x, ei).argmax(-1), prior.argmax(-1)))
    wrong = permuted(prior, base)
    check("placebo prior moved, same values", not torch.equal(wrong, prior)
          and torch.equal(wrong.sort(0).values, prior.sort(0).values))
    mix = Mix(m0, prior, 0.5).eval()
    forwards = []
    for mode in ("ste", "soft", "soft_t1"):
        mix.attack_mode = mode
        check(f"MIX passes attack mode {mode}", m0.attack_mode == mode)
        w = torch.ones(ei.size(1), requires_grad=True)
        out = mix(x, ei, w)
        forwards.append(out.detach())
        p = out.exp()
        margin = p[va].gather(1, y[va, None]).sum() - p[va].max(-1).values.sum()
        grad = torch.autograd.grad(margin, w)[0]
        check(f"MIX edge gradient finite and nonzero ({mode})", bool(torch.isfinite(grad).all()) and grad.norm() > 0)
    mix.attack_mode = "ste"
    check("MIX hard forward in every mode", all(torch.allclose(forwards[0], f) for f in forwards))
    mlp = fit_mlp(x, y, tr, va, base)
    empty = torch.empty((2, 0), dtype=torch.long)
    check("MLP ignores edges", torch.equal(cr.correct(mlp, x, ei, y), cr.correct(mlp, x, empty, y)))
    low = low_label_mask(y, tr, base)
    check("low-label mask", int(low.sum()) == LOW_LABEL_PER_CLASS * (int(y.max()) + 1) and not bool((low & ~tr).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=list(ATTACKED))
    ap.add_argument("--pt_epochs", type=int, default=ls.PT_EPOCHS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x, ei, y, trm, vam, tem = fx.load()
    texts = list(torch.load(fx.CORA, weights_only=False).raw_texts)
    texts_hash = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    os.makedirs(OUT, exist_ok=True)
    if args.audit:
        return audit(y, vam, texts_hash)
    with open(AUDIT_FILE) as f:
        prior_perms = json.load(f)["prior_perms"]
    qwen = ls.load_qwen(texts_hash)
    logits = load_prior_logits(texts_hash, prior_perms)
    s = args.split
    if args.selfcheck:
        return selfcheck(x, qwen, ei, y, trm[s], vam[s], logits)
    t0 = time.time()
    name = f"stage2_split{s}_seed{args.seed}_pt{args.pt_epochs}"
    rows, edits, pernode, qwen_pred, low_ids = run_cell(
        s, args.seed, x, qwen, ei, y, trm[s], vam[s], tem[s], logits, args.pt_epochs, args.arms,
        ckpt=os.path.join(OUT, f"{name}_ckpt.pt"))
    env = {"python": platform.python_version(), "torch": torch.__version__, "threads": args.threads,
           "seconds": round(time.time() - t0, 1), "prior_perms": prior_perms}
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, "edits": edits, "low_label_nodes": low_ids.tolist()}, f, indent=1)
    torch.save({"pernode": pernode, "test_mask": tem[s], "qwen_pred": qwen_pred, "y": y}, os.path.join(OUT, f"{name}_pernode.pt"))


if __name__ == "__main__":
    main()
