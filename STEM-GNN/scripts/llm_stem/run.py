"""Stage 1: LLM at the MoE router of STEM-GNN (experiments/llm-stem/protocol.md, Amendment 1).

Arms differ only in what the real MoE router reads:
  S0 message-passed z (repo default); S-R0 MiniLM text; S-R Qwen embedding;
  S-U frozen uniform logits; S-RF fixed random features; S-PQ node-permuted Qwen embedding.

Real Encoder (MoE, depth 2, width 128) + real VectorQuantize (EMA), pretrained with the MoE
router in place, frozen codebook at finetune. For differentiable edge weights (PRBCD) the MoE
forward is re-expressed with the real layers' parameters; --selfcheck compares it to the real
Encoder. Text never weights, selects, adds or drops edges.

Usage (from repo root):
  python STEM-GNN/scripts/llm_stem/run.py --selfcheck
  python STEM-GNN/scripts/llm_stem/run.py --select_coeff
  python STEM-GNN/scripts/llm_stem/run.py --split 0 --seed 0 --coeff 1e-4
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
from torch_geometric.contrib.nn import PRBCDAttack
from torch_geometric.utils import dropout_adj, mask_feature, negative_sampling

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..")
spec = importlib.util.spec_from_file_location("cr", os.path.join(SCRIPTS, "code_robustness", "run.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
fx = cr.fx

log = logging.getLogger("llm_stem")
OUT = os.path.join(SCRIPTS, "..", "..", "experiments", "llm-stem", "results")
QWEN_FEATURES = os.path.join(SCRIPTS, "..", "..", "experiments", "llm-robustness", "results",
                             "qwen_features_Qwen2.5-1.5B-Instruct.pt")
HID, DEPTH, EXPERTS, ROUTER_TAU, PT_EPOCHS, COMMIT = 128, 2, 3, 1.0, 200, 0.25
ARMS = ("S0", "S-R0", "S-R", "S-U", "S-RF", "S-PQ")
COEFF_GRID = (0.0, 1e-5, 1e-4, 1e-3)
COEFF_TIE = 0.005
RATES = (0.1, 0.2)
# name, seed offset, backward mode, PRBCD loss
ATTACKS = (("ste", 37, "ste", "prob_margin"), ("ste_r2", 41, "ste", "prob_margin"),
           ("soft", 43, "soft", "prob_margin"), ("soft_t1", 47, "soft_t1", "prob_margin"),
           ("ste_tanh", 53, "ste", "tanh_margin"))
SOFT_TAU = {"soft": 0.1, "soft_t1": 1.0}


def load_qwen(texts_hash):
    blob = torch.load(QWEN_FEATURES, weights_only=False)
    if blob["meta"]["text_hash"] != texts_hash:
        raise RuntimeError("Qwen features were extracted from different texts or a different node order")
    return F.normalize(blob["features"].float(), dim=-1)


def router_features(arm, x, qwen, base):
    """Edge-independent router input per arm; None means the router reads message-passed z (S0)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base + 90_001)
        if arm == "S0":
            return None
        if arm == "S-R0":
            return x
        if arm == "S-R":
            return qwen
        if arm == "S-U":
            return torch.zeros(x.size(0), 1)
        if arm == "S-RF":
            return F.normalize(torch.randn_like(qwen), dim=-1)
        if arm == "S-PQ":
            return qwen[torch.randperm(x.size(0))]
    raise ValueError(arm)


def build_encoder(arm, in_dim, rf, init_seed=0):
    """Experts get the same initial weights in every arm: router size must not shift the RNG stream."""
    enc = fx.Encoder(in_dim, HID, nn.ReLU, DEPTH, backbone="sage", normalize="batch", dropout=fx.DROPOUT,
                     moe=True, num_experts=EXPERTS, tau=ROUTER_TAU, moe_layers="all",
                     use_llm_router=rf is not None, router_input_dim=None if rf is None else rf.size(1))
    fx.seed_all(init_seed)
    for layer in enc.layers:
        layer.reset_parameters()
    fx.seed_all(init_seed + 1)
    for router in enc.env_encoders:
        router.reset_parameters()
    fx.seed_all(init_seed + 2)
    if arm == "S-U":
        for router in enc.env_encoders:
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)
            router.requires_grad_(False)
    return enc


def pretrain(arm, x, ei, rf, epochs, pt_seed):
    """X2 C-ema recipe (feature + topology reconstruction + VQ) with the MoE encoder in place.
    The router reads clean text features even when expert inputs are masked. The repo's router
    regularizer is excluded, as in X2."""
    fx.seed_all(pt_seed)
    enc = build_encoder(arm, x.size(1), rf, init_seed=pt_seed)
    feat_dec = nn.Linear(HID, x.size(1))
    topo_dec = fx.InnerProductDecoder(hidden_dim=HID, output_dim=HID)
    vq = fx.make_vq(HID, True, fx.X2_ARMS["C-ema"][2])
    mods = (enc, feat_dec, topo_dec, vq)
    opt = torch.optim.AdamW([p for m in mods for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        for m in mods:
            m.train()
        fx.seed_all(pt_seed * 100_003 + epoch)
        aug_x, _ = mask_feature(x, p=0.2)
        aug_ei, _ = dropout_adj(ei, p=0.2, force_undirected=True, num_nodes=x.size(0))
        neg = negative_sampling(ei, x.size(0))
        q, _, vq_loss, _ = vq(enc(aug_x, aug_ei, router_features=rf))
        loss = F.mse_loss(feat_dec(q), x) + fx.topo_loss_fn(topo_dec, q, ei, neg) + vq_loss.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for m in mods:
        m.eval()
    enc._last_env_reg = None  # unused non-leaf tensor; it would break deepcopy at finetune
    return enc, vq


def moe_layer(layer, z, ei, w):
    """MixtureSageLayer with edge weights: weighted mean of z[col] into row, zero for isolated nodes."""
    row, col = ei
    total = torch.zeros(z.size(0)).index_add_(0, row, w)
    agg = torch.zeros_like(z).index_add_(0, row, z[col] * w.unsqueeze(-1)) / total.clamp_min(1e-6).unsqueeze(-1)
    out = torch.einsum("nd,kdo->nko", torch.cat([agg, z], -1), layer.weights)
    return out + z.unsqueeze(1) if layer.residual else out


def moe_encode(enc, x, ei, rf, w=None, routes=None):
    """Real Encoder.encode re-expressed with edge weights. Appends router probabilities to routes."""
    w = torch.ones(ei.size(1)) if w is None else w
    z = x
    for i, layer in enumerate(enc.layers):
        logits = enc.env_encoders[i](z if rf is None else rf)
        probs = F.gumbel_softmax(logits, tau=enc.tau, dim=-1) if enc.training else F.softmax(logits, dim=-1)
        if routes is not None:
            routes.append(probs)
        z = (probs.unsqueeze(-1) * moe_layer(layer, z, ei, w)).sum(1)
        if enc.normalize != "none":
            z = enc.norms[i](z)
        if i < len(enc.layers) - 1:
            z = enc.dropout(enc.activation(z))
    return z


class RouterCodeModel(cr.CodeModel):
    def __init__(self, enc, vq, n_cls, rf):
        super().__init__(enc, vq, n_cls)
        self.register_buffer("rf", rf)

    def forward_full(self, x, ei, w=None):
        zh = self.heads_view(moe_encode(self.enc, x, ei, self.rf, w))
        sims, idx, c = self.assign(zh)
        if self.attack_mode in SOFT_TAU:
            soft = torch.einsum("nhk,hkd->nhd", F.softmax(sims / SOFT_TAU[self.attack_mode], -1), self.codebook)
            codes = soft + (c - soft).detach()
        else:
            codes = zh + (c - zh).detach()
        logits = self.decoder(codes.flatten(1)).view(-1, self.heads, self.n_cls).mean(1)
        return logits, zh, sims, idx, c

    def penalty(self, coeff):
        return coeff * self.decoder.weight.pow(2).sum() + self.enc.lipschitz_penalty(coeff)


def finetune(enc, vq, x, ei, y, tr, va, rf, coeff, base):
    fx.seed_all(base + 500)
    model = RouterCodeModel(copy.deepcopy(enc), copy.deepcopy(vq), int(y.max()) + 1, rf)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    best, state, wait, best_epoch = -1.0, None, 0, 0
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)
        logits = model(x, ei)
        loss = F.cross_entropy(logits[tr], y[tr]) + model.penalty(coeff)
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


def prbcd(model, x, ei, y, test_idx, budget, seed, mode, loss):
    """PRBCD with test labels and test indices: a label-informed evaluation attacker."""
    model.eval()
    if hasattr(model, "attack_mode"):
        model.attack_mode = mode
    try:
        fx.seed_all(seed)
        atk = PRBCDAttack(model, block_size=100_000, epochs=100, epochs_resampling=60, loss=loss, log=False)
        pert, _ = atk.attack(x, ei, y, budget=budget, idx_attack=test_idx)
    finally:
        if hasattr(model, "attack_mode"):
            model.attack_mode = "ste"
    return pert


@torch.no_grad()
def routing(model, x, ei):
    routes = []
    moe_encode(model.enc, x, ei, model.rf, routes=routes)
    return routes


def perplexity(probs):
    mean = probs.mean(0).clamp_min(1e-12)
    return torch.exp(-(mean * mean.log()).sum()).item()


def run_cell(split, seed, x, ei, y, tr, va, te, qwen, coeff, pt_epochs, arms=ARMS, rates=RATES):
    base = 1000 * seed + split
    test_idx = te.nonzero().view(-1)
    n_undirected = len(cr.undirected(ei))
    budgets = {r: int(r * n_undirected) for r in rates}
    models, rows = {}, {}
    for arm in arms:
        rf = router_features(arm, x, qwen, base)
        enc, vq = pretrain(arm, x, ei, rf, pt_epochs, pt_seed=base)
        model, val, best_epoch = finetune(enc, vq, x, ei, y, tr, va, rf, coeff, base)
        models[arm] = model
        rows[arm] = {"arm": arm, "split": split, "seed": seed, "coeff": coeff, "val": val, "best_epoch": best_epoch}
    fx.seed_all(base + 11)
    sur = cr.Surrogate(x.size(1), int(y.max()) + 1)
    fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
    graphs = {}
    for r, b in budgets.items():
        graphs[("sage", "sage", r)] = prbcd(sur, x, ei, y, test_idx, b, base + 23 + int(100 * r), "ste", "prob_margin")
        for arm, model in models.items():
            for name, offset, mode, loss in ATTACKS:
                t0 = time.time()
                graphs[(arm, name, r)] = prbcd(model, x, ei, y, test_idx, b, base + offset + int(100 * r), mode, loss)
                log.info(json.dumps({"attack": [arm, name, r], "seconds": round(time.time() - t0, 1)}))
    edits = {f"{k[0]}|{k[1]}|{k[2]}": cr.validate(ei, g, x.size(0), budgets[k[2]]) for k, g in graphs.items()}
    pernode = {}
    for arm, model in models.items():
        clean = cr.correct(model, x, ei, y)
        corr = {k: cr.correct(model, x, g, y) for k, g in graphs.items()}
        row = rows[arm]
        row["clean"] = clean[te].float().mean().item()
        for r in rates:
            keys = [k for k in corr if k[2] <= r]
            suite = torch.stack([clean] + [corr[k] for k in keys])
            row[f"worst_node@{r}"] = suite.all(0)[te].float().mean().item()
            row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
            row[f"own_worst_node@{r}"] = torch.stack(
                [clean] + [corr[k] for k in keys if k[0] == arm]).all(0)[te].float().mean().item()
            row[f"transfer_sage@{r}"] = corr[("sage", "sage", r)][te].float().mean().item()
        top = max(rates)
        clean_routes = routing(model, x, ei)
        attacked_routes = routing(model, x, graphs[(arm, "ste", top)])
        row["route_tv_own"] = [0.5 * (a - b).abs().sum(-1)[te].mean().item() for a, b in zip(clean_routes, attacked_routes)]
        row["expert_perplexity"] = [perplexity(p) for p in clean_routes]
        row["code_change_own"] = cr.code_change(model, x, ei, graphs[(arm, "ste", top)], test_idx)
        pernode[arm] = {"clean": clean, **{"|".join(map(str, k)): v for k, v in corr.items()}}
        log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                             if not isinstance(v, list)}))
    return list(rows.values()), edits, pernode


def select_coeff(x, ei, y, trm, vam, qwen, pt_epochs):
    """Shared penalty coefficient chosen on S0 clean validation only (seed 0, all splits), then frozen."""
    scores = {c: [] for c in COEFF_GRID}
    for split in range(3):
        rf = router_features("S0", x, qwen, split)
        enc, vq = pretrain("S0", x, ei, rf, pt_epochs, pt_seed=split)
        for c in COEFF_GRID:
            scores[c].append(finetune(enc, vq, x, ei, y, trm[split], vam[split], rf, c, split)[1])
    means = {c: sum(v) / len(v) for c, v in scores.items()}
    top = max(means.values())
    chosen = max(c for c, m in means.items() if m >= top - COEFF_TIE)  # ties favour the strongest penalty
    print(json.dumps({"val_by_coeff": {str(c): v for c, v in scores.items()},
                      "mean": {str(c): m for c, m in means.items()}, "chosen": chosen}), flush=True)


def selfcheck(x, ei, y, qwen):
    fx.seed_all(0)
    for arm in ARMS:
        rf = router_features(arm, x, qwen, 0)
        enc = build_encoder(arm, x.size(1), rf).eval()
        with torch.no_grad():
            real, mine = enc(x, ei, router_features=rf), moe_encode(enc, x, ei, rf)
        print(arm, "eval == real Encoder:", torch.allclose(real, mine, atol=1e-5), (real - mine).abs().max().item())
        enc.train()
        fx.seed_all(5)
        real = enc(x, ei, router_features=rf)
        fx.seed_all(5)
        print(arm, "train == real Encoder:", torch.allclose(real, moe_encode(enc, x, ei, rf), atol=1e-4))
    # Asymmetric toy graph with an isolated node: catches a flipped aggregation direction.
    toy_x = torch.randn(6, 16)
    toy_ei = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 3, 0]])
    enc = build_encoder("S0", 16, None).eval()
    with torch.no_grad():
        print("toy asymmetric/isolated == real:", torch.allclose(enc(toy_x, toy_ei), moe_encode(enc, toy_x, toy_ei, None), atol=1e-5))
    keep = torch.tensor([True, False, True, True, True])
    w = keep.float()
    with torch.no_grad():
        print("weight-0 == removed:", torch.allclose(moe_encode(enc, toy_x, toy_ei, None, w), enc(toy_x, toy_ei[:, keep]), atol=1e-5))
    with torch.no_grad():
        scale = torch.full((5,), 0.5)
        print("uniform weights == unit weights:", torch.allclose(moe_encode(enc, toy_x, toy_ei, None, scale),
                                                                  enc(toy_x, toy_ei), atol=1e-5))
    # Edge-to-router gradient path: the second-layer router must depend on edges only in S0.
    for arm in ("S0", "S-R"):
        rf = router_features(arm, x, qwen, 0)
        enc = build_encoder(arm, x.size(1), rf).eval()
        w = torch.ones(ei.size(1), requires_grad=True)
        routes = []
        moe_encode(enc, x, ei, rf, w, routes)
        path = routes[1].requires_grad and torch.autograd.grad(routes[1].sum(0)[0], w, allow_unused=True)[0]
        norm = 0.0 if path is False or path is None else path.norm().item()
        print(arm, "|d(layer-2 router)/d(edges)|:", norm)
    # Shared expert, decoder and VQ initial weights must not depend on the router's input size.
    inits = {}
    for arm in ARMS:
        rf = router_features(arm, x, qwen, 0)
        fx.seed_all(3)
        enc = build_encoder(arm, x.size(1), rf, init_seed=3)
        dec, vq = nn.Linear(HID, x.size(1)), fx.make_vq(HID, True, fx.X2_ARMS["C-ema"][2])
        inits[arm] = [p.detach().clone() for m in (enc.layers, dec, vq) for p in m.parameters()]
    print("shared init identical across arms:", all(
        len(v) == len(inits["S0"]) and all(torch.equal(a, b) for a, b in zip(v, inits["S0"])) for v in inits.values()))
    # Edge gradients stay finite when a node's incoming weight nearly vanishes.
    enc = build_encoder("S0", 16, None).eval()
    for floor in (1e-3, 1e-5, 1e-7, 0.0):
        w = torch.full((5,), floor, requires_grad=True)
        grad = torch.autograd.grad(moe_encode(enc, toy_x, toy_ei, None, w).sum(), w)[0]
        print(f"edge grad at weight {floor}: finite={torch.isfinite(grad).all().item()} max={grad.abs().max().item():.3g}")
    enc, vq = pretrain("S-R", x, ei, qwen, 5, pt_seed=0)
    copy.deepcopy(enc)
    print("pretrained encoder deep-copies: True")
    model = RouterCodeModel(enc, vq, int(y.max()) + 1, qwen).eval()
    with torch.no_grad():
        _, idx_real, _, codes_real = vq.eval()(enc(x, ei, router_features=qwen))
        _, _, _, idx_mine, c_mine = model.forward_full(x, ei)
    print("codes match real VQ:", torch.equal(idx_real, idx_mine), torch.allclose(codes_real, c_mine.flatten(1), atol=1e-5))
    with torch.no_grad():
        forwards = []
        for mode in ("ste", "soft", "soft_t1"):
            model.attack_mode = mode
            forwards.append(model(x, ei))
        model.attack_mode = "ste"
    print("hard forward in every backward mode:", all(torch.allclose(forwards[0], f, atol=1e-5) for f in forwards))
    print("penalty finite:", torch.isfinite(model.penalty(1e-4)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coeff", type=float, default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--pt_epochs", type=int, default=PT_EPOCHS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--select_coeff", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x, ei, y, trm, vam, tem = fx.load()
    texts = list(torch.load(fx.CORA, weights_only=False).raw_texts)
    qwen = load_qwen(hashlib.sha256("\x00".join(texts).encode()).hexdigest())
    if args.selfcheck:
        return selfcheck(x, ei, y, qwen)
    if args.select_coeff:
        return select_coeff(x, ei, y, trm, vam, qwen, args.pt_epochs)
    if args.coeff is None:
        raise SystemExit("--coeff is required: use the value frozen by --select_coeff")
    os.makedirs(OUT, exist_ok=True)
    s = args.split
    t0 = time.time()
    rows, edits, pernode = run_cell(s, args.seed, x, ei, y, trm[s], vam[s], tem[s], qwen, args.coeff, args.pt_epochs, args.arms)
    env = {"python": platform.python_version(), "torch": torch.__version__, "threads": args.threads,
           "seconds": round(time.time() - t0, 1)}
    name = f"split{s}_seed{args.seed}"
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "env": env, "rows": rows, "edits": edits}, f, indent=1)
    torch.save({"pernode": pernode, "test_mask": tem[s]}, os.path.join(OUT, f"{name}_pernode.pt"))


if __name__ == "__main__":
    main()
