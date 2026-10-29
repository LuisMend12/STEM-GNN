"""LLM prior anchoring of STEM-GNN's prediction head (experiments/llm-anchor/protocol.md).

Methods: B0, A-qwen, A-minilm. Label settings: standard, low (3/class).
Reuses the code-robustness harness (real Encoder + real VQ, frozen codebook, strong attacks).

Usage (from repo root):
  python STEM-GNN/scripts/llm_anchor/run.py --splits 0 --seeds 0 1 --threads 2
"""
import argparse
import importlib.util
import json
import logging
import os
import platform
import time

import torch
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
QWEN = os.path.join(ROOT, "experiments", "llm-robustness", "results", "qwen_prior_Qwen2.5-1.5B-Instruct.pt")
LAM = 1.0
METHODS = ("B0", "A-qwen", "A-minilm")


def low_label_mask(tr, y, per_class, seed):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros_like(tr)
    for c in range(int(y.max()) + 1):
        idx = (tr & (y == c)).nonzero().view(-1)
        mask[idx[torch.randperm(idx.numel(), generator=g)[:per_class]]] = True
    return mask


def finetune_anchor(method, enc, vq, x, ei, y, tr, va, base, priors):
    import copy
    fx.seed_all(base + 500)
    model = cr.CodeModel(copy.deepcopy(enc), copy.deepcopy(vq), int(y.max()) + 1)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    unl = ~tr
    prior = priors.get(method)
    best, state, wait, best_epoch = -1.0, None, 0, 0
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)
        logits = model(x, ei)
        loss = F.cross_entropy(logits[tr], y[tr])
        if prior is not None:
            loss = loss + LAM * F.kl_div(F.log_softmax(logits[unl], -1), prior[unl], reduction="batchmean")
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


def run_cell(split, seed, x, ei, y, tr_std, va, te, priors, pt_epochs, rates, settings):
    base = 1000 * seed + split
    test_idx = te.nonzero().view(-1)
    budgets = {r: int(r * len(cr.undirected(ei))) for r in rates}
    enc, vq, _ = fx.pretrain_x2("C-ema", x, ei, cr.HID, pt_epochs, pt_seed=base)
    fx.seed_all(base + 11)
    sur = cr.Surrogate(x.size(1), int(y.max()) + 1)
    fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr_std, va, modules=(sur,))
    transfer = {}
    for r, b in budgets.items():
        transfer[("sage", r)] = cr.prbcd(sur, x, ei, y, test_idx, b, seed=base + 23 + int(100 * r))
        cr.validate(ei, transfer[("sage", r)], x.size(0), b)
    rows = []
    for setting in settings:
        tr = tr_std if setting == "standard" else low_label_mask(tr_std, y, 3, base)
        b0_graphs = None
        for method in METHODS:
            t0 = time.time()
            model, val, best_epoch = finetune_anchor(method, enc, vq, x, ei, y, tr, va, base, priors)
            own = {}
            for r, b in budgets.items():
                for name, s_off, mode in (("ste", 37, "ste"), ("soft", 43, "soft")):
                    own[(name, r)] = cr.prbcd(model, x, ei, y, test_idx, b, seed=base + s_off + int(100 * r), mode=mode)
                    cr.validate(ei, own[(name, r)], x.size(0), b)
            if method == "B0":
                b0_graphs = {("b0_" + k[0], k[1]): g for k, g in own.items()}
            cands = {**own, **transfer, **(b0_graphs if method != "B0" else {})}
            corr = {k: cr.correct(model, x, g, y) for k, g in cands.items()}
            clean_c = cr.correct(model, x, ei, y)
            row = {"setting": setting, "method": method, "split": split, "seed": seed, "n_train": int(tr.sum()),
                   "val": val, "best_epoch": best_epoch, "clean": clean_c[te].float().mean().item()}
            for k, c in corr.items():
                row[f"{k[0]}@{k[1]}"] = c[te].float().mean().item()
            for r in rates:
                suite = torch.stack([clean_c] + [corr[k] for k in corr if k[1] <= r])
                row[f"worst_node@{r}"] = suite.all(0)[te].float().mean().item()
                row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
            row["seconds"] = round(time.time() - t0, 1)
            log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
            rows.append(row)
    return rows


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
    qwen = torch.load(QWEN)
    assert torch.isfinite(qwen).all() and qwen.shape == (x.size(0), int(y.max()) + 1)
    d = torch.load(llm.probe.CORA, weights_only=False)
    meta = {"qwen_zero_shot_acc_all": (qwen.argmax(-1) == y).float().mean().item()}
    for s in args.splits:
        minilm, temp = llm.zero_shot_prior(x, llm.class_embeddings(list(d.label_names)), y, vam[s])
        priors = {"A-qwen": qwen.float(), "A-minilm": minilm.float()}
        meta[f"split{s}"] = {"minilm_T": temp,
                             "qwen_test_acc": (qwen.argmax(-1)[tem[s]] == y[tem[s]]).float().mean().item(),
                             "minilm_test_acc": (minilm.argmax(-1)[tem[s]] == y[tem[s]]).float().mean().item()}
        rows = []
        for seed in args.seeds:
            rows += run_cell(s, seed, x, ei, y, trm[s], vam[s], tem[s], priors, args.pt_epochs, args.rates, args.settings)
            with open(os.path.join(OUT, f"split{s}_{args.tag}.json"), "w") as f:
                json.dump({"args": vars(args), "meta": meta, "env": {"torch": torch.__version__,
                           "python": platform.python_version(), "threads": args.threads}, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
