"""Why does STEM-GNN's MoE encoder degrade with depth? (experiments/moe-depth/protocol.md)

Variants: SAGE; MoE V0 (Gumbel train / softmax eval, repo default); V1 (V0 model,
eval with Gumbel-sampled routing averaged over 16 draws); V2 (softmax train, no
Gumbel noise); V3 (uniform frozen routing).

Usage (from repo root):
  python STEM-GNN/scripts/moe_depth/run.py --splits 0 --threads 4
"""
import argparse
import importlib.util
import json
import logging
import os
import time
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mc", os.path.join(HERE, "..", "message_collapse", "run.py"))
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)  # load(), Model (real STEM Encoder), train(), accuracy()
enc_mod = __import__("model.encoder", fromlist=["F"])  # same module object mc.Model uses

log = logging.getLogger("moe_depth")
OUT = os.path.join(HERE, "..", "..", "..", "experiments", "moe-depth", "results")
N_MC = 16
REAL_F = enc_mod.F
NO_NOISE_F = types.SimpleNamespace(**{k: getattr(REAL_F, k) for k in dir(REAL_F) if not k.startswith("__")})
NO_NOISE_F.gumbel_softmax = lambda logits, tau=1.0, dim=-1, **_: F.softmax(logits / tau, dim=dim)


def set_router_noise(enabled):
    enc_mod.F = REAL_F if enabled else NO_NOISE_F


def freeze_uniform_router(model):
    for lin in model.enc.env_encoders:
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        lin.weight.requires_grad_(False)
        lin.bias.requires_grad_(False)


def gumbel_hooks(model, store):
    """Eval-time Gumbel routing: softmax((logits + g)/tau) == a Gumbel-softmax draw."""
    def hook(_m, _i, out):
        g = -torch.log(-torch.log(torch.rand_like(out).clamp_min(1e-10)).clamp_min(1e-10))
        store.append((F.softmax(out, -1), F.softmax(out + g, -1)))
        return (out + g) / model.enc.tau
    return [lin.register_forward_hook(hook) for lin in model.enc.env_encoders]


@torch.no_grad()
def mc_gumbel_eval(model, x, ei, y, te):
    """V1: same weights, eval mode (dropout/BN off), Gumbel-sampled routing averaged."""
    model.eval()
    store = []
    handles = gumbel_hooks(model, store)
    try:
        logits = torch.stack([model(x, ei) for _ in range(N_MC)]).mean(0)
    finally:
        for h in handles:
            h.remove()
    acc = (logits.argmax(-1)[te] == y[te]).float().mean().item()
    tv = torch.stack([0.5 * (a - b).abs().sum(-1).mean() for a, b in store]).mean().item()
    return acc, tv


def run_one(variant, depth, x, ei, y, tr, va, te, seed_val):
    torch.manual_seed(seed_val)
    moe = variant != "SAGE"
    set_router_noise(variant in ("SAGE", "V0"))
    try:
        model = mc.Model(x.size(1), int(y.max()) + 1, depth, moe)
        if variant == "V3":
            freeze_uniform_router(model)
        val = mc.train(model, "baseline", x, ei, y, tr, va)
        row = {"val": val, "test": mc.accuracy(model, x, ei, y, te)}
    finally:
        set_router_noise(True)
    if variant == "V0":
        torch.manual_seed(seed_val + 7)
        row["V1_test"], row["routing_tv"] = mc_gumbel_eval(model, x, ei, y, te)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    x, ei, y, trm, vam, tem = mc.load()
    for s in args.splits:
        rows = []
        for seed in args.seeds:
            for depth in args.depths:
                for variant in ("SAGE", "V0", "V2", "V3"):
                    t0 = time.time()
                    row = {"variant": variant, "depth": depth, "split": s, "seed": seed, "threads": args.threads,
                           **run_one(variant, depth, x, ei, y, trm[s], vam[s], tem[s], 1000 * seed + s)}
                    row["seconds"] = round(time.time() - t0, 1)
                    log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
                    rows.append(row)
                    with open(os.path.join(OUT, f"split{s}.json"), "w") as f:
                        json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
