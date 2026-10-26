"""X1 (MoE per-expert init) and X2 (STEM-component VQ pilot: codebook update rules)
on STEM-GNN's real Encoder and VectorQuantize. See experiments/stem-fixes/protocol.md
(including Amendment 1, from the Codex design review).

Usage (from repo root):
  python STEM-GNN/scripts/stem_fixes/run.py x1 --splits 0 --threads 4
  python STEM-GNN/scripts/stem_fixes/run.py x2 --splits 0 --threads 4
"""
import argparse
import json
import logging
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import coalesce, dropout_adj, mask_feature, negative_sampling, to_undirected

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, PKG)
from model.encoder import Encoder, InnerProductDecoder, MixtureSageLayer  # noqa: E402
from model.vq import VectorQuantize  # noqa: E402

log = logging.getLogger("fixes")
CORA = os.path.join(PKG, "dataset", "data", "single_graph", "Cora", "cora.pt")
OUT = os.path.join(PKG, "..", "experiments", "stem-fixes", "results")
LR, WD, EPOCHS, PATIENCE, DROPOUT = 0.01, 5e-4, 300, 50, 0.15
PROBE_SEEDS = (0, 1, 2)
X2_ARMS = {  # name: (use_vq, ema_update, orthogonal_reg_weight)
    "C-repo": (True, False, 1.0),
    "C-random": (True, False, 0.0),
    "C-ema": (True, True, 0.0),
    "C-none": (False, None, None),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load():
    d = torch.load(CORA, weights_only=False)
    ei = coalesce(to_undirected(d.edge_index))
    return d.x.float(), ei[:, ei[0] != ei[1]], d.y.long(), d.train_masks, d.val_masks, d.test_masks


@torch.no_grad()
def acc(logits, y, mask):
    return (logits.argmax(-1)[mask] == y[mask]).float().mean().item()


def fit(params, forward, y, tr, va, modules, lr=LR, wd=WD):
    """Train forward() -> logits; early-stop on validation and restore the best state."""
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    best, best_state, wait, epochs = -1.0, None, 0, 0
    for epochs in range(1, EPOCHS + 1):
        for m in modules:
            m.train()
        opt.zero_grad()
        F.cross_entropy(forward()[tr], y[tr]).backward()
        opt.step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            val = acc(forward(), y, va)
        if val > best:
            best, wait = val, 0
            best_state = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in modules]
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    for m, s in zip(modules, best_state):
        m.load_state_dict(s)
        m.eval()
    return best, epochs


# ---------------------------------------------------------------- X1
def per_expert_init(encoder):
    for layer in encoder.layers:
        if isinstance(layer, MixtureSageLayer):
            with torch.no_grad():
                for k in range(layer.num_experts):
                    nn.init.xavier_uniform_(layer.weights[k])


@torch.no_grad()
def layer_ratios(encoder, x, ei):
    """Per residual MoE layer: RMS(expert transform)/RMS(input) averaged over experts,
    and RMS(router-mixed transform)/RMS(input). Eval mode; inputs via pre-hooks."""
    inputs, handles = {}, []
    for i, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_pre_hook(lambda m, a, i=i: inputs.__setitem__(i, a[0])))
    encoder.eval()
    encoder.enable_router_cache(True)
    encoder(x, ei)
    routers = encoder.get_router_cache(reset=True) or []
    encoder.enable_router_cache(False)
    for h in handles:
        h.remove()
    indiv, mixed, env = [], [], 0
    for i, layer in enumerate(encoder.layers):
        if not isinstance(layer, MixtureSageLayer):
            continue
        w = routers[env]
        env += 1
        if not layer.residual:
            continue
        z = inputs[i]
        row, col = ei
        deg = torch.bincount(row, minlength=z.size(0)).clamp_min(1).unsqueeze(-1)
        agg = torch.zeros_like(z).index_add_(0, row, z[col]) / deg
        t = torch.einsum("nd,kdo->nko", torch.cat([agg, z], -1), layer.weights)
        zr = z.pow(2).mean().sqrt().clamp_min(1e-12)
        indiv.append((t.pow(2).mean().sqrt() / zr).item())
        mixed.append(((w.unsqueeze(-1) * t).sum(1).pow(2).mean().sqrt() / zr).item())
    return indiv, mixed


def run_x1(variant, depth, x, ei, y, tr, va, te, hid, init_seed, train_seed):
    seed_all(init_seed)
    enc = Encoder(x.size(1), hid, nn.ReLU, depth, backbone="sage", normalize="batch", dropout=DROPOUT,
                  moe=variant != "SAGE", num_experts=3, tau=1.0, moe_layers="all")
    head = nn.Linear(hid, int(y.max()) + 1)
    if variant == "M-expert":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed + 7919)
            per_expert_init(enc)
    row = {}
    if variant != "SAGE":
        ind, mix = layer_ratios(enc, x, ei)
        row.update(ratio_init_indiv=ind, ratio_init_mixed=mix,
                   ratio_init_mixed_mean=float(np.mean(mix)) if mix else float("nan"))
    seed_all(train_seed)
    val, epochs = fit(list(enc.parameters()) + list(head.parameters()), lambda: head(enc(x, ei)),
                      y, tr, va, modules=(enc, head))
    with torch.no_grad():
        logits = head(enc(x, ei))
    row.update(val=val, test=acc(logits, y, te), epochs_trained=epochs)
    if variant != "SAGE":
        ind, mix = layer_ratios(enc, x, ei)
        enc.enable_router_cache(True)
        with torch.no_grad():
            enc(x, ei)
        routers = enc.get_router_cache(reset=True) or []
        enc.enable_router_cache(False)
        lk = float(np.log(3))
        ent_mean_routing = [(-(w.mean(0) * w.mean(0).clamp_min(1e-12).log()).sum() / lk).item() for w in routers]
        ent_per_node = [(-(w * w.clamp_min(1e-12).log()).sum(-1).mean() / lk).item() for w in routers]
        row.update(ratio_trained_mixed=mix, ratio_trained_mixed_mean=float(np.mean(mix)) if mix else float("nan"),
                   entropy_of_mean_routing=float(np.mean(ent_mean_routing)),
                   mean_per_node_entropy=float(np.mean(ent_per_node)))
    return row


# ---------------------------------------------------------------- X2
def make_vq(hid, ema, ortho):
    return VectorQuantize(dim=hid, codebook_size=128, codebook_dim=hid, heads=4, separate_codebook_per_head=True,
                          decay=0.8, commitment_weight=0.25, use_cosine_sim=True, orthogonal_reg_weight=ortho,
                          orthogonal_reg_max_codes=32, orthogonal_reg_active_codes_only=False,
                          kmeans_init=False, ema_update=ema)


def topo_loss_fn(topo_dec, q, ei, neg):
    return (-torch.log(topo_dec(q, ei, sigmoid=True) + 1e-15).mean()
            - torch.log(1 - topo_dec(q, neg, sigmoid=True) + 1e-15).mean())


def pretrain_x2(arm, x, ei, hid, epochs, pt_seed):
    use_vq, ema, ortho = X2_ARMS[arm]
    seed_all(pt_seed)
    enc = Encoder(x.size(1), hid, nn.ReLU, 2, backbone="sage", normalize="batch", dropout=DROPOUT)
    feat_dec = nn.Linear(hid, x.size(1))
    topo_dec = InnerProductDecoder(hidden_dim=hid, output_dim=hid)
    vq = make_vq(hid, ema, ortho) if use_vq else None  # built last: shared modules initialize identically
    mods = [m for m in (enc, feat_dec, topo_dec, vq) if m is not None]
    opt = torch.optim.AdamW([p for m in mods for p in m.parameters()], lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        for m in mods:
            m.train()
        seed_all(pt_seed * 100_003 + epoch)  # identical corruptions/negatives across arms
        aug_x, _ = mask_feature(x, p=0.2)
        aug_ei, _ = dropout_adj(ei, p=0.2, force_undirected=True, num_nodes=x.size(0))
        neg = negative_sampling(ei, x.size(0))
        z = enc(aug_x, aug_ei)
        if vq is not None:
            q, _, vq_loss, _ = vq(z)
            vq_loss = vq_loss.sum()
        else:
            q, vq_loss = z, torch.zeros(())
        loss = F.mse_loss(feat_dec(q), x) + topo_loss_fn(topo_dec, q, ei, neg) + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    for m in mods:
        m.eval()
    with torch.no_grad():
        seed_all(pt_seed * 100_003 + 999_983)
        neg = negative_sampling(ei, x.size(0))
        z = enc(x, ei)
        q = vq(z)[0] if vq is not None else z
        diag = {"final_feat_loss": F.mse_loss(feat_dec(q), x).item(),
                "final_topo_loss": topo_loss_fn(topo_dec, q, ei, neg).item()}
    return enc, vq, diag


@torch.no_grad()
def features_and_usage(enc, vq, x, ei):
    z = enc(x, ei)
    if vq is None:
        return z, None, {}
    _, idx, _, codes = vq(z)
    assert idx.dim() == 2 and idx.size(1) == vq.heads, idx.shape
    assert codes.size(0) == z.size(0) and codes.size(1) % vq.heads == 0, codes.shape
    per_head_perp, per_head_dead = [], []
    for h in range(idx.size(1)):
        cnt = torch.bincount(idx[:, h], minlength=vq.codebook_size).float()
        p = cnt[cnt > 0] / cnt.sum()
        per_head_perp.append(torch.exp(-(p * p.log()).sum()).item())
        per_head_dead.append((cnt == 0).float().mean().item())
    return z, codes, {"code_perplexity_per_head": per_head_perp, "code_perplexity": float(np.mean(per_head_perp)),
                      "dead_code_frac_per_head": per_head_dead, "dead_code_frac": float(np.mean(per_head_dead))}


def probe(feats, y, tr, va, te):
    tests, vals = [], []
    for ps in PROBE_SEEDS:
        seed_all(50_000 + ps)
        lin = nn.Linear(feats.size(1), int(y.max()) + 1)
        v, _ = fit(lin.parameters(), lambda: lin(feats), y, tr, va, modules=(lin,))
        with torch.no_grad():
            tests.append(acc(lin(feats), y, te))
        vals.append(v)
    return float(np.mean(vals)), float(np.mean(tests))


def run_x2(arm, x, ei, y, tr, va, te, hid, epochs, pt_seed):
    enc, vq, diag = pretrain_x2(arm, x, ei, hid, epochs, pt_seed)
    z, codes, usage = features_and_usage(enc, vq, x, ei)
    row = {**diag, **usage, "feat_dim_z": z.size(1)}
    row["z_probe_val"], row["z_probe_test"] = probe(z.detach(), y, tr, va, te)
    if codes is not None:
        row["feat_dim_codes"] = codes.size(1)
        row["val"], row["test"] = probe(codes.detach(), y, tr, va, te)
    else:
        row["val"], row["test"] = row["z_probe_val"], row["z_probe_test"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["x1", "x2"])
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--pt_epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(OUT, exist_ok=True)
    env = {"python": platform.python_version(), "torch": torch.__version__, "pyg": torch_geometric.__version__,
           "threads": args.threads, "device": "cpu"}
    x, ei, y, trm, vam, tem = load()
    for s in args.splits:
        rows = []
        for seed in args.seeds:
            base = 1000 * seed + s
            jobs = ([(v, d) for d in args.depths for v in ("SAGE", "M-stack", "M-expert")] if args.exp == "x1"
                    else [(a, None) for a in X2_ARMS])
            for variant, depth in jobs:
                t0 = time.time()
                if args.exp == "x1":
                    r = run_x1(variant, depth, x, ei, y, trm[s], vam[s], tem[s], args.hid,
                               init_seed=base, train_seed=base + 500)
                else:
                    r = run_x2(variant, x, ei, y, trm[s], vam[s], tem[s], args.hid, args.pt_epochs, pt_seed=base)
                row = {"exp": args.exp, "variant": variant, "depth": depth, "split": s, "seed": seed, **r,
                       "seconds": round(time.time() - t0, 1)}
                log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                     if not isinstance(v, list)}))
                rows.append(row)
                with open(os.path.join(OUT, f"{args.exp}_split{s}_{args.tag}.json"), "w") as f:
                    json.dump({"args": vars(args), "env": env, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
