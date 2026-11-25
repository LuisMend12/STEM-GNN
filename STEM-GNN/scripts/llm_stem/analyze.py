"""Stage 1 analysis: H60-H62 and descriptive tables from the per-cell JSON files."""
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
RESULTS = os.path.join(ROOT, "experiments", "llm-stem", "results")
ARMS = ("S0", "S-R0", "S-R", "S-U", "S-RF", "S-PQ")
METRICS = ("val", "clean", "worst_node@0.1", "worst_node@0.2", "worst_graph@0.2",
           "own_worst_node@0.2", "transfer_sage@0.2")


def load() -> dict:
    cells = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "split*_seed*.json"))):
        if path.endswith("_pernode.json"):
            continue
        with open(path) as f:
            for row in json.load(f)["rows"]:
                cells[(row["arm"], row["split"], row["seed"])] = row
    return cells


def paired(diff: np.ndarray) -> dict:
    """Mean, 95% t interval and two-sided p for a paired difference, in points."""
    d = 100 * diff
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else float("nan")
    return {"mean": m, "lo": m - half, "hi": m + half, "p": p, "wins": int((d > 0).sum()), "n": n}


def fmt(r: dict) -> str:
    return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f} wins {r['wins']}/{r['n']}"


def main() -> None:
    cells = load()
    keys = sorted({(s, e) for _, s, e in cells})
    assert len(keys) == 15 and len(cells) == 90, (len(keys), len(cells))
    get = lambda arm, m: np.array([cells[(arm, s, e)][m] for s, e in keys])
    splits = np.array([s for s, _ in keys])

    print("## Arm means (points), sd over cells")
    for m in METRICS:
        print(m, {a: f"{100 * get(a, m).mean():.2f}±{100 * get(a, m).std(ddof=1):.2f}" for a in ARMS})

    wn, cl = "worst_node@0.2", "clean"
    trivial = np.max([get(a, wn) for a in ("S-U", "S-RF", "S-PQ")], axis=0)
    tests = {"H60 S-R-S-R0": get("S-R", wn) - get("S-R0", wn),
             "H61 S-R-S0": get("S-R", wn) - get("S0", wn),
             "H62 S-R-max(trivial)": get("S-R", wn) - trivial}
    clean = {"H60 S-R-S-R0": get("S-R", cl) - get("S-R0", cl), "H61 S-R-S0": get("S-R", cl) - get("S0", cl)}

    print("\n## Hypotheses at worst-node@20%")
    res = {}
    for name, diff in tests.items():
        res[name] = paired(diff)
        print(name, fmt(res[name]))
        print("  per split", {int(s): f"{100 * diff[splits == s].mean():+.2f}" for s in np.unique(splits)})
    for name, diff in clean.items():
        print("clean", name, fmt(paired(diff)))

    fam = sorted(["H61 S-R-S0", "H62 S-R-max(trivial)"], key=lambda k: res[k]["p"])
    print("Holm:", {k: min(1.0, res[k]["p"] * (2 - i)) for i, k in enumerate(fam)})

    print("\n## Descriptive, each arm minus S0 and minus S-U")
    for a in ARMS[1:]:
        for m in (wn, "worst_node@0.1", "worst_graph@0.2", "own_worst_node@0.2", "transfer_sage@0.2", cl):
            print(a, "-S0", m, fmt(paired(get(a, m) - get("S0", m))))
    for a in ("S-R", "S-R0", "S-RF", "S-PQ"):
        print(a, "-S-U", wn, fmt(paired(get(a, wn) - get("S-U", wn))))
        print(a, "-S-U clean", fmt(paired(get(a, cl) - get("S-U", cl))))

    print("S-R-S-U worst_graph@0.2", fmt(paired(get("S-R", "worst_graph@0.2") - get("S-U", "worst_graph@0.2"))))
    for a in ("S-RF", "S-PQ"):
        print("S-R -", a, wn, fmt(paired(get("S-R", wn) - get(a, wn))))

    print("\n## Diagnostics, means per layer")
    for a in ARMS:
        for m in ("route_tv_own", "expert_perplexity", "code_change_own"):
            print(a, m, np.round(np.mean([cells[(a, s, e)][m] for s, e in keys], axis=0), 4).tolist())
    print("best_epoch mean", {a: round(float(get(a, "best_epoch").mean()), 1) for a in ARMS})


if __name__ == "__main__":
    main()
