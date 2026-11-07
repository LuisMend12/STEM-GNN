"""Baseline comparison for the capped correction (experiments/capped-correction/baselines_protocol.md)."""
import glob
import importlib.util
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cap_an", os.path.join(HERE, "analyze.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

RESULTS = os.path.join(an.ROOT, "experiments", "capped-correction", "baselines")
CAPS = ("CAP0.1", "CAP0.25", "CAP0.5", "CAP0.75", "CAP0.9")
OURS = ("S0", "RES") + CAPS + ("CAP*",)
ENS = ("ENS0.1", "ENS0.25", "ENS0.5", "ENS0.75", "ENS0.9")
BASE = ("MLP-M", "GCN", "SAGE", "APPNP-lite", "RUNG-lite", "SAGE+TextGate") + ENS
ARMS = OURS + BASE
WN, CL = "worst_node@0.2", "clean"
CLEAN_TOL = 0.005  # a baseline counts as "at least as clean" if within 0.5 points of the CAP arm, per cell
TOL = 1e-9


def load() -> dict:
    cells, hashes = {}, set()
    for path in sorted(glob.glob(os.path.join(RESULTS, "base_split*_seed*_*[0-9a-f].json"))):
        with open(path) as f:
            blob = json.load(f)
        hashes.add(blob["env"]["code_hash"])
        for row in blob["rows"]:
            key = (row["arm"], row["split"], row["seed"])
            assert key not in cells, f"duplicate {key} in {path}"
            cells[key] = row
    assert len(hashes) == 1, hashes
    return cells


def envelope_gap(get, keys, c: str) -> tuple:
    """Per cell: WN(CAP c) - max WN over baselines whose clean >= CAP c's clean - CLEAN_TOL."""
    wn, cl = get(c, WN), get(c, CL)
    base_wn = np.stack([get(x, WN) for x in BASE])
    base_cl = np.stack([get(x, CL) for x in BASE])
    eligible = base_cl >= cl - CLEAN_TOL
    ok = eligible.any(0)
    best = np.where(eligible, base_wn, -np.inf).max(0)
    return wn[ok] - best[ok], int((~ok).sum())


def main() -> None:
    cells = load()
    keys = sorted({(s, e) for _, s, e in cells})
    assert keys == [(s, e) for s in range(3) for e in range(5)], keys
    assert set(cells) == {(a, s, e) for a in ARMS for s, e in keys}, "arm/cell set differs from the protocol"
    for (a, s, e), r in cells.items():
        for k, v in r.items():
            if k.startswith(("worst", "orig", "own", "transfer", "clean", "val")):
                assert v is not None and np.isfinite(v), (a, s, e, k, v)
        cert = r.get("cert_test")
        assert (cert is not None and np.isfinite(cert)) == (a in CAPS + ("CAP*",)), (a, s, e, "cert_test", cert)
    get = lambda arm, m: np.array([cells[(arm, s, e)][m] for s, e in keys], dtype=float)

    protocol = an.load()
    drift = 0.0
    for a in OURS + ("MLP-M",):
        for s, e in keys:
            new, old = cells[(a, s, e)], protocol[(a, s, e)]
            for r in ("0.1", "0.2"):
                drift = max(drift, abs(new[f"orig_worst_node@{r}"] - old[f"worst_node@{r}"]))
                assert new[f"worst_node@{r}"] <= new[f"orig_worst_node@{r}"] + TOL, (a, s, e, r)
            drift = max(drift, abs(new["clean"] - old["clean"]))
            if old["cert_test"] is not None:
                drift = max(drift, abs(new["cert_test"] - old["cert_test"]))
    assert drift < TOL, f"capped-run results not reproduced: drift {drift}"
    print(f"Check: clean, certified and protocol-union worst-node (both budgets) reproduced for all original "
          f"arms (max drift {drift:.1e}); extended union never raises worst-node")

    print("\n## Arm means over 15 cells (points, extended union): clean / worst-node@10% / worst-node@20% / "
          "worst-graph@20% / own-attack worst-node@20% / SAGE transfer@20% / certified")
    for a in ARMS:
        vals = [100 * get(a, m).mean() for m in (CL, "worst_node@0.1", WN, "worst_graph@0.2", "own_worst_node@0.2",
                                                   "transfer_sage@0.2")]
        cert = [cells[(a, s, e)].get("cert_test") for s, e in keys]
        c = f"{100 * np.mean(cert):.1f}" if all(v is not None for v in cert) else "-"
        print(f"{a:14s} " + " / ".join(f"{v:.1f}" for v in vals) + f" / {c}")

    print("\n## Family B (one Holm family): CAP* - X, worst-node@20%, extended union")
    print("(robustness of the fixed selected arm only; not a frontier test)")
    tests = {f"CAP*-{x}": get("CAP*", WN) - get(x, WN) for x in BASE if x != "MLP-M"}
    res = {k: an.paired(d) for k, d in tests.items()}
    ph = an.holm({k: r["p"] for k, r in res.items()})
    for k, r in res.items():
        print(f"{k}: {an.fmt(r)} holm={ph[k]:.4f}")
    print("\nClean, CAP* - X (descriptive):")
    for x in BASE:
        print(f"CAP*-{x}: {an.fmt(an.paired(get('CAP*', CL) - get(x, CL)))}")

    print(f"\n## Family F (one Holm family): matched-clean frontier, CAP(q) - baseline envelope, worst-node@20%")
    print(f"envelope = best baseline worst-node among baselines with clean >= CAP(q) clean - {100 * CLEAN_TOL:.1f} pt, per cell")
    fam = {}
    for c in CAPS:
        d, missing = envelope_gap(get, keys, c)
        fam[c] = (an.paired(d) if len(d) >= 3 else None, missing)
    ph = an.holm({c: (r["p"] if r is not None else 1.0) for c, (r, _) in fam.items()})  # untestable keep p = 1
    for c, (r, missing) in fam.items():
        if r is None:
            print(f"{c}: fewer than 3 cells with an eligible baseline ({missing} missing)")
            continue
        verdict = "CAP above envelope" if r["mean"] > 0 and ph[c] < 0.05 else (
            "CAP below envelope" if r["mean"] < 0 and ph[c] < 0.05 else "no difference shown")
        print(f"{c}: {an.fmt(r)} holm={ph[c]:.4f} cells without eligible baseline={missing} -> {verdict}")

    print("\n## Pareto set of arm means (clean, worst-node@20%), descriptive")
    pts = {a: (get(a, CL).mean(), get(a, WN).mean()) for a in ARMS if a != "CAP*"}
    front = [a for a in pts if not any(pts[b][0] >= pts[a][0] and pts[b][1] >= pts[a][1] and pts[b] != pts[a]
                                       for b in pts)]
    for a in sorted(front, key=lambda a: pts[a][0]):
        print(f"{a}: clean {100 * pts[a][0]:.1f}, worst-node {100 * pts[a][1]:.1f}")

    print("\n## Pairwise (exploratory, uncorrected): CAP(q)/RES vs baseline X, delta clean and delta worst-node@20%")
    print("'dominates' = both lower 95% CI bounds > 0; 'dominated' = both upper bounds < 0")
    for c, x in itertools.product(CAPS + ("RES",), BASE):
        dc, dw = an.paired(get(c, CL) - get(x, CL)), an.paired(get(c, WN) - get(x, WN))
        tag = "dominates" if dc["lo"] > 0 and dw["lo"] > 0 else ("dominated" if dc["hi"] < 0 and dw["hi"] < 0 else "")
        print(f"{c} vs {x}: clean {dc['mean']:+.2f} [{dc['lo']:+.2f},{dc['hi']:+.2f}]  "
              f"worst {dw['mean']:+.2f} [{dw['lo']:+.2f},{dw['hi']:+.2f}]  {tag}")

    print("\n## Per-split CAP* - best-baseline worst-node@20% (split sensitivity)")
    splits = np.array([s for s, _ in keys])
    for s in range(3):
        m = splits == s
        print(s, {x: f"{100 * (get('CAP*', WN) - get(x, WN))[m].mean():+.2f}" for x in BASE})


if __name__ == "__main__":
    main()
