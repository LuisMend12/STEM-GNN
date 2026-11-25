"""Stage 2 analysis: H70-H75, staged claims and descriptive tables from the per-cell JSON files."""
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RESULTS = os.path.join(ROOT, "experiments", "llm-stem", "results_stage2")
ATTACKED = ("S0", "S-F", "S-V", "S-VP", "S-A", "S-AP", "S-VA", "MIX")
FREE = ("MLP-M", "MLP-Q", "QWEN")
METRICS = ("val", "clean", "worst_node@0.1", "worst_node@0.2", "worst_graph@0.2",
           "own_worst_node@0.2", "transfer_sage@0.2", "low_clean")
WN, CL, LOW = "worst_node@0.2", "clean", "low_clean"
PLACEBO = {"S-V": "S-VP", "S-A": "S-AP"}


def load() -> dict:
    cells = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "stage2_split*_seed*_*.json"))):
        with open(path) as f:
            for row in json.load(f)["rows"]:
                key = (row["arm"], row["split"], row["seed"])
                assert key not in cells, f"duplicate result for {key} in {path}"
                cells[key] = row
    return cells


def paired(diff: np.ndarray) -> dict:
    """Mean, 95% t interval and two-sided p for a paired difference, in points."""
    d = 100 * diff
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else float("nan")
    return {"mean": m, "lo": m - half, "hi": m + half, "p": p, "wins": int((d > 0).sum()), "n": n}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


def fmt(r: dict) -> str:
    return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f} wins {r['wins']}/{r['n']}"


def verdict(r: dict, p_holm: float, clean: dict = None) -> str:
    ok = r["mean"] >= 2 and p_holm < 0.05
    if clean is not None:
        ok = ok and clean["mean"] >= -1 and clean["lo"] > -2
    if ok:
        return "SUPPORTED"
    return "AGAINST" if r["hi"] < 1 else "INCONCLUSIVE"


def family(name: str, tests: dict, cleans: dict = None) -> list:
    print(f"\n## {name}")
    res = {k: paired(v) for k, v in tests.items()}
    ph = holm({k: r["p"] for k, r in res.items()})
    supported = []
    for k, r in res.items():
        c = paired(cleans[k]) if cleans else None
        v = verdict(r, ph[k], c)
        extra = f" | clean {fmt(c)}" if c else ""
        print(f"{k}: {fmt(r)} holm={ph[k]:.4f} -> {v}{extra}")
        if v == "SUPPORTED":
            supported.append(k.split()[1].split("-S0")[0])
    return supported


def staged(get, arms_by_endpoint: dict) -> None:
    print("\n## Staged claims (one Holm family, two-sided)")
    tests = {}
    for ep, arms in arms_by_endpoint.items():
        free = np.max([get(a, ep) for a in FREE], axis=0)
        for arm in arms:
            if arm in PLACEBO:
                tests[f"{ep} {arm}-{PLACEBO[arm]}"] = get(arm, ep) - get(PLACEBO[arm], ep)
            tests[f"{ep} {arm}-MIX"] = get(arm, ep) - get("MIX", ep)
            tests[f"{ep} {arm}-max(free)"] = get(arm, ep) - free
    if not tests:
        print("No arm met support; no staged claims tested.")
        return
    res = {k: paired(v) for k, v in tests.items()}
    ph = holm({k: r["p"] for k, r in res.items()})
    for k, r in res.items():
        ok = r["mean"] >= 1 and r["lo"] > 0 and ph[k] < 0.05
        print(f"{k}: {fmt(r)} holm={ph[k]:.4f} -> {'MET' if ok else 'NOT MET'}")


def main() -> None:
    cells = load()
    keys = sorted({(s, e) for _, s, e in cells})
    assert keys == [(s, e) for s in range(3) for e in range(5)], keys
    assert len(cells) == 15 * (len(ATTACKED) + len(FREE)), len(cells)
    get = lambda arm, m: np.array([cells[(arm, s, e)][m] for s, e in keys])
    splits = np.array([s for s, _ in keys])

    print("## Arm means (points), sd over cells")
    for m in METRICS:
        print(m, {a: f"{100 * get(a, m).mean():.2f}±{100 * get(a, m).std(ddof=1):.2f}"
                  for a in ATTACKED + FREE})

    wn_ok = family("H70-H72 worst-node@20%",
                   {f"H{70 + i} {a}-S0": get(a, WN) - get("S0", WN) for i, a in enumerate(("S-F", "S-V", "S-A"))},
                   {f"H{70 + i} {a}-S0": get(a, CL) - get("S0", CL) for i, a in enumerate(("S-F", "S-V", "S-A"))})
    low_ok = family("H73-H75 low-label clean",
                    {f"H{73 + i} {a}-S0": get(a, LOW) - get("S0", LOW) for i, a in enumerate(("S-F", "S-V", "S-A"))})
    staged(get, {WN: wn_ok, LOW: low_ok})

    print("\n## Descriptive contrasts")
    desc = {f"{a}-S0 {m}": get(a, m) - get("S0", m) for a in ("S-VA", "MIX", "S-VP", "S-AP")
            for m in (WN, CL, LOW)}
    desc.update({f"S-VA-{b} {LOW}": get("S-VA", LOW) - get(b, LOW) for b in ("S-A", "S-V", "MIX", "S-AP")})
    desc[f"S-VA-max(free) {LOW}"] = get("S-VA", LOW) - np.max([get(a, LOW) for a in FREE], axis=0)
    desc[f"S0-max(free) {CL}"] = get("S0", CL) - np.max([get(a, CL) for a in FREE], axis=0)
    desc[f"S0-max(free) {WN}"] = get("S0", WN) - np.max([get(a, WN) for a in FREE], axis=0)
    for k, v in desc.items():
        print(f"{k}: {fmt(paired(v))}")

    print("\n## Per-split means of key low-label effects (points)")
    for a in ("S-A", "S-VA", "MIX"):
        d = 100 * (get(a, LOW) - get("S0", LOW))
        print(a, "-S0", {int(s): f"{d[splits == s].mean():+.2f}" for s in np.unique(splits)})
    print("\n## MIX alpha (clean, low-label) and prior temperature per cell")
    for s, e in keys:
        r = cells[("MIX", s, e)]
        print((s, e), r.get("alpha"), r.get("low_alpha"), r["temperature"])


if __name__ == "__main__":
    main()
