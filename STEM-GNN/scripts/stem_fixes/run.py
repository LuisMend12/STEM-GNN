"""X1 (MoE per-expert init) and X2 (EMA vs fixed VQ codebook) on STEM-GNN's real
Encoder and VectorQuantize. See experiments/stem-fixes/protocol.md.

Usage (from repo root):
  python STEM-GNN/scripts/stem_fixes/run.py x1 --splits 0 --threads 4
  python STEM-GNN/scripts/stem_fixes/run.py x2 --splits 0 --threads 4
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load():
    d = torch.load(CORA, weights_only=False)
    ei = coalesce(to_undirected(d.edge_index))
    return d.x.float(), ei[:, ei[0] != ei[1]], d.y.long(), d.train_masks, d.val_masks, d.test_masks


@torch.no_grad()
def acc(logits, y, mask):
    return (logits.argmax(-1)[mask] == y[mask]).float().mean().item()


def fit(params, forward, y, tr, va, lr=LR, wd=WD, modules=()):
    """Train `forward()` -> logits with early stopping on validation; returns (val, test_logits)."""
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    best, best_state, wait = -1.0, None, 0
    for _ in range(EPOCHS):
        for m in modules:
            m.train()
        opt.zero_grad()
        F.cross_entropy(forward()[tr], y[tr]).backward()
        opt.step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            logits = forward()
        val = acc(logits, y, va)
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
    return best


# ---------------------------------------------------------------- X1
def per_expert_init(encoder):
    for layer in encoder.layers:
        if isinstance(layer, MixtureSageLayer):
            with torch.no_grad():
                for k in range(layer.num_experts):
                    nn.init.xavier_uniform_(layer.weights[k])


@torch.no_grad()
def transform_residual_ratio(encoder, x, ei):
    """Mean over equal-width MoE layers of RMS(combined @ W_k) / RMS(x) at that layer's input."""
    ratios, z = [], x
    states = []
    h = encoder.activation.register_forward_hook(lambda m, i, o: states.append(o))
    encoder.eval()
    encoder(x, ei)
    h.remove()
    inputs = [x] + states  # input to layer i is the post-activation output of layer i-1
    for layer, z in zip(encoder.layers, inputs):
        if isinstance(layer, MixtureSageLayer) and layer.residual:
            row, col = ei
            agg = torch.zeros_like(z).index_add_(0, row, z[col]) / torch.bincount(row, minlength=z.size(0)).clamp_min(1).unsqueeze(-1)
            t = torch.einsum("nd,kdo->nko", torch.cat([agg, z], -1), layer.weights)
            ratios.append((t.pow(2).mean().sqrt() / z.pow(2).mean().sqrt().clamp_min(1e-12)).item())
    return sum(ratios) / len(ratios) if ratios else float("nan")


def run_x1(variant, depth, x, ei, y, tr, va, te, hid):
    enc = Encoder(x.size(1), hid, nn.ReLU, depth, backbone="sage", normalize="batch", dropout=DROPOUT,
                  moe=variant != "SAGE", num_experts=3, tau=1.0, moe_layers="all")
    if variant == "M-expert":
        per_expert_init(enc)
    head = nn.Linear(hid, int(y.max()) + 1)
    row = {"ratio_init": transform_residual_ratio(enc, x, ei) if variant != "SAGE" else float("nan")}
    val = fit(list(enc.parameters()) + list(head.parameters()), lambda: head(enc(x, ei)), y, tr, va, modules=(enc, head))
    with torch.no_grad():
        logits = head(enc(x, ei))
        enc.enable_router_cache(True)
        enc(x, ei)
        routers = enc.get_router_cache(reset=True) or []
        enc.enable_router_cache(False)
    ent = [(-(w.mean(0) * w.mean(0).clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(3.0))).item() for w in routers]
    row.update(val=val, test=acc(logits, y, te),
               ratio_trained=transform_residual_ratio(enc, x, ei) if variant != "SAGE" else float("nan"),
               router_entropy=sum(ent) / len(ent) if ent else float("nan"))
    return row


# ---------------------------------------------------------------- X2
def make_vq(hid, ema):
    return VectorQuantize(dim=hid, codebook_size=128, codebook_dim=hid, heads=4, separate_codebook_per_head=True,
                          decay=0.8, commitment_weight=0.25, use_cosine_sim=True, orthogonal_reg_weight=1,
                          orthogonal_reg_max_codes=32, orthogonal_reg_active_codes_only=False,
                          kmeans_init=False, ema_update=ema)


def pretrain_x2(variant, x, ei, hid, epochs):
    enc = Encoder(x.size(1), hid, nn.ReLU, 2, backbone="sage", normalize="batch", dropout=DROPOUT)
    vq = None if variant == "C-none" else make_vq(hid, ema=variant == "C-ema")
    feat_dec = nn.Linear(hid, x.size(1))
    topo_dec = InnerProductDecoder(hidden_dim=hid, output_dim=hid)
    mods = [m for m in (enc, vq, feat_dec, topo_dec) if m is not None]
    opt = torch.optim.AdamW([p for m in mods for p in m.parameters()], lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for m in mods:
            m.train()
        aug_x, _ = mask_feature(x, p=0.2)
        aug_ei, _ = dropout_adj(ei, p=0.2, force_undirected=True, num_nodes=x.size(0))
        z = enc(aug_x, aug_ei)
        if vq is not None:
            q, _, vq_loss, _ = vq(z)
            vq_loss = vq_loss.sum()
        else:
            q, vq_loss = z, torch.zeros(())
        feat_loss = F.mse_loss(feat_dec(q), x)
        neg = negative_sampling(ei, x.size(0))
        topo_loss = (-torch.log(topo_dec(q, ei, sigmoid=True) + 1e-15).mean()
                     - torch.log(1 - topo_dec(q, neg, sigmoid=True) + 1e-15).mean())
        loss = feat_loss + topo_loss + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    for m in mods:
        m.eval()
    return enc, vq, {"feat_loss": feat_loss.item(), "topo_loss": topo_loss.item(), "vq_loss": vq_loss.item()}


@torch.no_grad()
def codes_and_usage(enc, vq, x, ei):
    z = enc(x, ei)
    if vq is None:
        return z, {}
    _, idx, _, codes = vq(z)
    perps, dead = [], []
    for h in range(idx.size(-1)):
        cnt = torch.bincount(idx[:, h], minlength=128).float()
        p = cnt / cnt.sum()
        perps.append(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item())
        dead.append((cnt == 0).float().mean().item())
    return codes, {"code_perplexity": sum(perps) / len(perps), "dead_code_frac": sum(dead) / len(dead)}


def run_x2(variant, x, ei, y, tr, va, te, hid, epochs):
    enc, vq, losses = pretrain_x2(variant, x, ei, hid, epochs)
    feats, usage = codes_and_usage(enc, vq, x, ei)
    feats = feats.detach()
    probe = nn.Linear(feats.size(1), int(y.max()) + 1)
    val = fit(probe.parameters(), lambda: probe(feats), y, tr, va, modules=(probe,))
    with torch.no_grad():
        test = acc(probe(feats), y, te)
    return {"val": val, "test": test, **losses, **usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["x1", "x2"])
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--pt_epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(OUT, exist_ok=True)
    x, ei, y, trm, vam, tem = load()
    for s in args.splits:
        rows = []
        for seed in args.seeds:
            jobs = ([(v, d) for d in args.depths for v in ("SAGE", "M-stack", "M-expert")] if args.exp == "x1"
                    else [(v, None) for v in ("C-fixed", "C-ema", "C-none")])
            for variant, depth in jobs:
                torch.manual_seed(1000 * seed + s)
                t0 = time.time()
                if args.exp == "x1":
                    r = run_x1(variant, depth, x, ei, y, trm[s], vam[s], tem[s], args.hid)
                else:
                    r = run_x2(variant, x, ei, y, trm[s], vam[s], tem[s], args.hid, args.pt_epochs)
                row = {"exp": args.exp, "variant": variant, "depth": depth, "split": s, "seed": seed, **r,
                       "seconds": round(time.time() - t0, 1)}
                log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
                rows.append(row)
                with open(os.path.join(OUT, f"{args.exp}_split{s}.json"), "w") as f:
                    json.dump({"args": vars(args), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
