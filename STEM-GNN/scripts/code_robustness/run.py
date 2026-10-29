"""Code-level robustness for STEM-GNN (experiments/code-robustness/protocol.md, Amendment 1).

Methods: B0, B1n (native commitment), B1s (strong commitment), CAC (code-assignment
consistency), PredCons (matched prediction-consistency control), VBM (all-competitor
normalized boundary margin).

Real Encoder + real VectorQuantize (EMA codebook, pretrained as in stem-fixes X2),
frozen codebook at finetune. For differentiable edge weights (PRBCD) the encoder forward
is re-expressed with the real layers' own parameters and weighted-mean aggregation; at
unit weights it equals the real Encoder (--selfcheck).

Usage (from repo root):
  python STEM-GNN/scripts/code_robustness/run.py --splits 0 1 2 --seeds 0 1 2 3 4 --threads 2
  python STEM-GNN/scripts/code_robustness/run.py --selfcheck
"""
import argparse
import copy
import importlib.util
import json
import logging
import os
import platform
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import dropout_edge, scatter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _load("fx", os.path.join(SCRIPTS, "stem_fixes", "run.py"))  # seed_all, load, pretrain_x2, fit, acc
warnings.filterwarnings("ignore", category=UserWarning)
from torch_geometric.contrib.nn import PRBCDAttack  # noqa: E402

log = logging.getLogger("coderob")
OUT = os.path.join(SCRIPTS, "..", "..", "experiments", "code-robustness", "results")
HID, LAM, TAU_SOFT, VIEW_P, VBM_M, ATTACK_TAU = 128, 1.0, 0.1, 0.3, 0.05, 0.1
METHODS = ("B0", "B1n", "B1s", "CAC", "PredCons", "VBM")


# ------------------------------------------------- weighted re-expression of the real encoder
def sage_layer(layer, x, ei, w):
    src, dst = ei
    msg = F.relu(x[src]) * w.unsqueeze(-1)  # MySAGEConv.message without edge_attr: relu(x_j)
    agg = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="sum")
    agg = agg / scatter(w, dst, dim=0, dim_size=x.size(0), reduce="sum").clamp_min(1e-12).unsqueeze(-1)
    return layer.lin_l(agg) + layer.lin_r(x)


def encode(enc, x, ei, w=None):
    w = torch.ones(ei.size(1)) if w is None else w
    z = x
    for i, layer in enumerate(enc.layers):
        z = enc.norms[i](sage_layer(layer, z, ei, w))
        if i < len(enc.layers) - 1:
            z = enc.dropout(enc.activation(z))
    return z


class CodeModel(nn.Module):
    """Encoder -> frozen VQ (project_in, per-head cosine lookup) -> decoder on concatenated codes.
    attack_mode: "ste" (identity backward) or "soft" (expected-code backward); forward is hard in both."""

    def __init__(self, enc, vq, n_cls):
        super().__init__()
        self.enc, self.vq = enc, vq
        self.heads, self.n_cls = vq.heads, n_cls
        self.register_buffer("codebook", vq._codebook.embed.detach().clone())  # [H, K, dh], frozen
        self.decoder = nn.Linear(self.codebook.size(0) * self.codebook.size(2), n_cls * self.heads)
        self.attack_mode = "ste"
        for p in self.vq.parameters():
            p.requires_grad_(False)

    def heads_view(self, z):
        return F.normalize(self.vq.project_in(z).view(z.size(0), self.heads, -1), dim=-1)

    def assign(self, zh):
        sims = torch.einsum("nhd,hkd->nhk", zh, self.codebook)
        idx = sims.argmax(-1)
        h = torch.arange(self.heads)[None, :]
        return sims, idx, self.codebook[h, idx]

    def forward_full(self, x, ei, w=None):
        zh = self.heads_view(encode(self.enc, x, ei, w))
        sims, idx, c = self.assign(zh)
        if self.attack_mode == "soft":
            soft = torch.einsum("nhk,hkd->nhd", F.softmax(sims / ATTACK_TAU, -1), self.codebook)
            codes = soft + (c - soft).detach()
        else:
            codes = zh + (c - zh).detach()
        logits = self.decoder(codes.flatten(1)).view(-1, self.heads, self.n_cls).mean(1)
        return logits, zh, sims, idx, c

    def forward(self, x, ei, w=None):
        return self.forward_full(x, ei, w)[0]


def deterministic_forward(model, x, ei):
    """Forward with dropout off and BatchNorm on running stats, keeping gradients."""
    was = model.training
    model.eval()
    try:
        return model.forward_full(x, ei)
    finally:
        model.train(was)


def view_graph(ei, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return dropout_edge(ei, p=VIEW_P, force_undirected=True, training=True)[0]


def regularizer(method, model, x, ei, zh, sims, c, logits, view_seed):
    if method == "B0":
        return torch.zeros(())
    if method == "B1n":
        return model.vq.commitment_weight * F.mse_loss(zh, c.detach())
    if method == "B1s":
        return LAM * (zh - c.detach()).pow(2).sum(-1).mean()
    if method in ("CAC", "PredCons"):
        view = view_graph(ei, view_seed)
        lg_c, _, s_c, _, _ = deterministic_forward(model, x, ei)
        lg_v, _, s_v, _, _ = deterministic_forward(model, x, view)
        if method == "CAC":
            kl = F.kl_div(F.log_softmax(s_v / TAU_SOFT, -1), F.softmax(s_c.detach() / TAU_SOFT, -1),
                          reduction="none").sum(-1).mean()
        else:
            kl = F.kl_div(F.log_softmax(lg_v, -1), F.softmax(lg_c.detach(), -1), reduction="none").sum(-1).mean()
        return LAM * kl
    if method == "VBM":
        a = sims.argmax(-1)
        h = torch.arange(model.heads)[None, :]
        ca, sa = model.codebook[h, a], sims.gather(-1, a.unsqueeze(-1))
        dist = (ca.unsqueeze(2) - model.codebook.unsqueeze(0)).norm(dim=-1)  # [N, H, K]
        m = (sa - sims) / dist.clamp_min(1e-6)
        m = m.masked_fill(F.one_hot(a, sims.size(-1)).bool(), float("inf")).min(-1).values
        return LAM * F.relu(VBM_M - m).mean()
    raise ValueError(method)


def finetune(method, enc, vq, x, ei, y, tr, va, base):
    fx.seed_all(base + 500)
    model = CodeModel(copy.deepcopy(enc), copy.deepcopy(vq), int(y.max()) + 1)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=fx.LR, weight_decay=fx.WD)
    best, state, wait, best_epoch, mags = -1.0, None, 0, 0, None
    for epoch in range(fx.EPOCHS):
        model.train()
        fx.seed_all(base * 7_919 + epoch)  # identical CE dropout stream across methods
        logits, zh, sims, _, c = model.forward_full(x, ei)
        ce = F.cross_entropy(logits[tr], y[tr])
        reg = regularizer(method, model, x, ei, zh, sims, c, logits, view_seed=base * 104_729 + epoch)
        opt.zero_grad()
        (ce + reg).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = fx.acc(model(x, ei), y, va)
        if val > best:
            best, wait, best_epoch, mags = val, 0, epoch, (ce.item(), float(reg))
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= fx.PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, best, best_epoch, mags


# ------------------------------------------------------------------ attacks
class Surrogate(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.enc = fx.Encoder(in_dim, HID, nn.ReLU, 2, backbone="sage", normalize="batch", dropout=fx.DROPOUT)
        self.head = nn.Linear(HID, n_cls)

    def forward(self, x, ei, w=None):
        return self.head(encode(self.enc, x, ei, w))


def undirected(ei):
    return {tuple(sorted(e)) for e in ei.t().tolist()}


def validate(ei, pert, n, budget):
    assert pert.min() >= 0 and pert.max() < n, "invalid node index"
    assert bool((pert[0] != pert[1]).all()), "self-loop"
    directed = set(map(tuple, pert.t().tolist()))
    assert len(directed) == pert.size(1), "duplicate directed edge"
    assert all((b, a) in directed for a, b in directed), "asymmetric graph"
    a, b = undirected(ei), undirected(pert)
    adds, dels = len(b - a), len(a - b)
    assert 0 <= adds + dels <= budget, (adds, dels, budget)
    return adds, dels


def prbcd(model, x, ei, y, test_idx, budget, seed, mode="ste"):
    model.eval()
    if hasattr(model, "attack_mode"):
        model.attack_mode = mode
    try:
        fx.seed_all(seed)
        atk = PRBCDAttack(model, block_size=100_000, epochs=100, epochs_resampling=60, log=False)
        pert, _ = atk.attack(x, ei, y, budget=budget, idx_attack=test_idx)
    finally:
        if hasattr(model, "attack_mode"):
            model.attack_mode = "ste"
    return pert


@torch.no_grad()
def correct(model, x, ei, y):
    return model(x, ei).argmax(-1) == y


@torch.no_grad()
def code_change(model, x, ei, pert, test_idx):
    a = model.forward_full(x, ei)[3][test_idx]
    b = model.forward_full(x, pert)[3][test_idx]
    return (a != b).float().mean(0).tolist()  # per head


def own_attacks(model, x, ei, y, test_idx, budgets, base):
    graphs = {}
    for r, b in budgets.items():
        for name, seed, mode in (("ste", 37, "ste"), ("ste_r2", 41, "ste"), ("soft", 43, "soft")):
            graphs[(name, r)] = prbcd(model, x, ei, y, test_idx, b, seed=base + seed + int(100 * r), mode=mode)
            validate(ei, graphs[(name, r)], x.size(0), b)
    return graphs


def run_cell(split, seed, x, ei, y, tr, va, te, pt_epochs, rates):
    base = 1000 * seed + split
    test_idx = te.nonzero().view(-1)
    m_undirected = len(undirected(ei))
    budgets = {r: int(r * m_undirected) for r in rates}
    enc, vq, pt_diag = fx.pretrain_x2("C-ema", x, ei, HID, pt_epochs, pt_seed=base)
    cb = vq._codebook.embed.detach()
    min_code_dist = min(torch.cdist(cb[h], cb[h]).add(torch.eye(cb.size(1)) * 9).min().item() for h in range(cb.size(0)))
    fx.seed_all(base + 11)
    sur = Surrogate(x.size(1), int(y.max()) + 1)
    fx.fit(list(sur.parameters()), lambda: sur(x, ei), y, tr, va, modules=(sur,))
    transfer = {}
    for r, b in budgets.items():
        transfer[("sage", r)] = prbcd(sur, x, ei, y, test_idx, b, seed=base + 23 + int(100 * r))
        validate(ei, transfer[("sage", r)], x.size(0), b)
    rows, b0_graphs = [], None
    for method in METHODS:
        t0 = time.time()
        model, val, best_epoch, mags = finetune(method, enc, vq, x, ei, y, tr, va, base)
        own = own_attacks(model, x, ei, y, test_idx, budgets, base)
        if method == "B0":
            b0_graphs = {("b0_" + k[0], k[1]): g for k, g in own.items()}
        cands = {**own, **transfer, **(b0_graphs if method != "B0" else {})}
        corr = {k: correct(model, x, g, y) for k, g in cands.items()}
        clean_c = correct(model, x, ei, y)
        row = {"method": method, "split": split, "seed": seed, "val": val, "best_epoch": best_epoch,
               "ce_at_best": mags[0], "reg_at_best": mags[1], "clean": clean_c[te].float().mean().item(),
               "min_code_dist": min_code_dist, **{f"pt_{k}": v for k, v in pt_diag.items()}}
        for k, c in corr.items():
            row[f"{k[0]}@{k[1]}"] = c[te].float().mean().item()
        for r in rates:
            keys = [k for k in corr if k[1] <= r]
            suite = torch.stack([clean_c] + [corr[k] for k in keys])
            row[f"worst_node@{r}"] = suite.all(0)[te].float().mean().item()
            row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
        row["code_change_sage@0.1"] = code_change(model, x, ei, transfer[("sage", 0.1)], test_idx)
        if b0_graphs is not None:
            row["code_change_b0ste@0.1"] = code_change(model, x, ei, b0_graphs[("b0_ste", 0.1)], test_idx)
        row["code_change_own@0.1"] = code_change(model, x, ei, own[("ste", 0.1)], test_idx)
        row["seconds"] = round(time.time() - t0, 1)
        log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                             if not isinstance(v, list)}))
        rows.append(row)
    return rows


def selfcheck():
    x, ei, y, *_ = fx.load()
    fx.seed_all(0)
    enc = fx.Encoder(x.size(1), HID, nn.ReLU, 2, backbone="sage", normalize="batch", dropout=fx.DROPOUT).eval()
    with torch.no_grad():
        a, b = enc(x, ei), encode(enc, x, ei)
    print("re-expressed encoder == real Encoder:", torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item())
    # binary edit equivalence: dropping an edge by weight 0 == physically removing it
    w = torch.ones(ei.size(1))
    pair = ei[:, 0]
    mask = ~(((ei[0] == pair[0]) & (ei[1] == pair[1])) | ((ei[0] == pair[1]) & (ei[1] == pair[0])))
    w[~mask] = 0.0
    with torch.no_grad():
        c1, c2 = encode(enc, x, ei, w), enc(x, ei[:, mask])
    print("weight-0 edge == removed edge:", torch.allclose(c1, c2, atol=1e-5))
    enc2, vq, _ = fx.pretrain_x2("C-ema", x, ei, HID, 5, pt_seed=0)
    m = CodeModel(enc2, vq, int(y.max()) + 1).eval()
    with torch.no_grad():
        _, idx_real, _, codes_real = vq.eval()(encode(m.enc, x, ei))
        _, _, _, idx_mine, c_mine = m.forward_full(x, ei)
        m.attack_mode = "soft"
        lg_soft = m(x, ei)
        m.attack_mode = "ste"
        lg_ste = m(x, ei)
    print("codes match real VQ:", torch.equal(idx_real, idx_mine), torch.allclose(codes_real, c_mine.flatten(1), atol=1e-5),
          "| soft-BPDA forward == hard forward:", torch.allclose(lg_soft, lg_ste, atol=1e-5))
    lg, zh, sims, _, c = m.train().forward_full(x, ei)
    for meth in METHODS:
        r = regularizer(meth, m, x, ei, zh, sims, c, lg, view_seed=1)
        print(f"  reg {meth}: {float(r):.4f}", "finite" if torch.isfinite(torch.as_tensor(r)).all() else "NOT FINITE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--pt_epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.selfcheck:
        return selfcheck()
    os.makedirs(OUT, exist_ok=True)
    env = {"python": platform.python_version(), "torch": torch.__version__, "pyg": torch_geometric.__version__,
           "threads": args.threads}
    x, ei, y, trm, vam, tem = fx.load()
    for s in args.splits:
        rows = []
        for seed in args.seeds:
            rows += run_cell(s, seed, x, ei, y, trm[s], vam[s], tem[s], args.pt_epochs, args.rates)
            with open(os.path.join(OUT, f"split{s}_{args.tag}.json"), "w") as f:
                json.dump({"args": vars(args), "env": env, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
