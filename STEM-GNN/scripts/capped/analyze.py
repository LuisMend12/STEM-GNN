"""Capped-correction analysis: H80-H82, the S0 gate and descriptive tables from the per-cell JSON files."""
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RESULTS = os.path.join(ROOT, "experiments", "capped-correction", "results")
ARMS = ("S0", "RES", "CAP0.1", "CAP0.25", "CAP0.5", "CAP0.75", "CAP0.9", "CAP*", "MLP-M")
WN, CL = "worst_node@0.2", "clean"


def load() -> dict:
    cells, configs = {}, set()
    for path in sorted(glob.glob(os.path.join(RESULTS, "cap_split*_seed*_*.json"))):
        with open(path) as f:
            blob = json.load(f)
        configs.add(json.dumps({k: v for k, v in blob["env"].items() if k != "seconds"}, sort_keys=True))
        for row in blob["rows"]:
            key = (row["arm"], row["split"], row["seed"])
            assert key not in cells, f"duplicate result for {key} in {path}"
            cells[key] = row
    assert len(configs) == 1, f"{len(configs)} different configurations"
    return cells


def paired(diff: np.ndarray) -> dict:
    d = 100 * diff
    assert np.isfinite(d).all() and len(d) >= 2, "paired: need >= 2 finite differences"
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, n - 1) * se
    if se > 0:
        p = 2 * stats.t.sf(abs(m / se), n - 1)
    else:  # constant differences: identical arms (p = 1) or a deterministic shift (p = 0)
        p = 1.0 if m == 0 else 0.0
    return {"mean": m, "lo": m - half, "hi": m + half, "p": p, "wins": int((d > 0).sum()), "n": n}


def holm(ps: dict) -> dict:
    assert all(np.isfinite(p) for p in ps.values()), f"non-finite p-value: {ps}"
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


def fmt(r: dict) -> str:
    return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f} wins {r['wins']}/{r['n']}"


def main() -> None:
    cells = load()
    keys = sorted({(s, e) for _, s, e in cells})
    assert keys == [(s, e) for s in range(3) for e in range(5)], keys
    assert len(cells) == 15 * len(ARMS), len(cells)
    get = lambda arm, m: np.array([cells[(arm, s, e)][m] for s, e in keys], dtype=float)
    splits = np.array([s for s, _ in keys])

    print("## Arm means (points), sd over cells")
    for m in ("val", "clean", "cert_test", "worst_node@0.1", WN, "worst_graph@0.2", "own_worst_node@0.2",
              "transfer_sage@0.2"):
        out = {}
        for a in ARMS:
            v = [cells[(a, s, e)].get(m) for s, e in keys]
            if all(x is not None for x in v):
                v = 100 * np.array(v, dtype=float)
                out[a] = f"{v.mean():.2f}±{v.std(ddof=1):.2f}"
        print(m, out)

    print("\n## H80-H82 (one Holm family)")
    tests = {"H80 CAP*-MLP-M worst-node@20%": (get("CAP*", WN) - get("MLP-M", WN), 1, 0),
             "H81 CAP*-MLP-M clean": (get("CAP*", CL) - get("MLP-M", CL), 2, 1),
             "H82 CAP*-RES worst-node@20%": (get("CAP*", WN) - get("RES", WN), 2, 1)}
    res = {k: paired(d) for k, (d, _, _) in tests.items()}
    ph = holm({k: r["p"] for k, r in res.items()})
    verdicts = {}
    for k, (_, support, against) in tests.items():
        r = res[k]
        v = "SUPPORTED" if r["mean"] >= support and ph[k] < 0.05 else ("AGAINST" if r["hi"] < against else "INCONCLUSIVE")
        verdicts[k[:3]] = v
        print(f"{k}: {fmt(r)} holm={ph[k]:.4f} -> {v}")

    gate = paired(get("CAP*", WN) - get("S0", WN))
    gate_ok = gate["mean"] >= 2 and gate["lo"] > 0
    print(f"\nGate CAP*-S0 worst-node@20%: {fmt(gate)} -> {'MET' if gate_ok else 'NOT MET'}")
    claim = verdicts["H81"] == "SUPPORTED" and verdicts["H82"] == "SUPPORTED" and gate_ok
    print("Claim (H81 and H82 and gate):", "MET" if claim else "NOT MET",
          "| beats graph-free under tested attacks (H80):", verdicts["H80"])

    print("\n## Descriptive")
    for k, d in {"CAP*-S0 clean": get("CAP*", CL) - get("S0", CL),
                 "CAP*-MLP-M cert_test (cert <= MLP acc)": get("CAP*", "cert_test") - get("MLP-M", CL),
                 "CAP* worst-node - cert (attack slack)": get("CAP*", WN) - get("CAP*", "cert_test"),
                 "RES-S0 worst-node@20%": get("RES", WN) - get("S0", WN),
                 "RES-S0 clean": get("RES", CL) - get("S0", CL)}.items():
        print(f"{k}: {fmt(paired(d))}")
    gain = (get("CAP*", CL) - get("MLP-M", CL)).mean() / (get("S0", CL) - get("MLP-M", CL)).mean()
    print(f"Retained clean gain (CAP*-MLP)/(S0-MLP), ratio of means: {gain:.2f}")

    print("\n## Trade-off curve: arm clean / worst-node@20% / certified (points)")
    for a in ARMS:
        cert = [cells[(a, s, e)].get("cert_test") for s, e in keys]
        c = f"{100 * np.mean(cert):.1f}" if all(x is not None for x in cert) else "-"
        print(f"{a}: {100 * get(a, CL).mean():.1f} / {100 * get(a, WN).mean():.1f} / {c}")

    print("\n## Per-split CAP*-MLP-M worst-node@20% and CAP*-S0 clean")
    for s in np.unique(splits):
        m = splits == s
        print(int(s), f"{100 * (get('CAP*', WN) - get('MLP-M', WN))[m].mean():+.2f}",
              f"{100 * (get('CAP*', CL) - get('S0', CL))[m].mean():+.2f}")
    print("\n## Chosen cap per cell")
    print({f"{s},{e}": cells[("CAP*", s, e)]["chosen"] for s, e in keys})


if __name__ == "__main__":
    main()
