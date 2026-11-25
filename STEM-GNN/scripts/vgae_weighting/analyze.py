"""Analysis for experiments/vgae-weighting/protocol.md: H100-H104, guardrails and descriptives."""
import glob
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cap_an", os.path.join(HERE, "..", "capped", "analyze.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)  # paired t, Holm, formatting

RESULTS = os.path.join(HERE, "..", "..", "..", "experiments", "vgae-weighting", "results")
LEVELS = (0, 500, 1000, 2000, 5000)
GRAPHS = 10
ARMS = ("S0", "S-VG", "S-VGr", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "MLP-M")
SCORERS = ("S-VG", "S-VGr", "S-SJ")


def load(prefix):
    blobs, configs = [], set()
    for path in sorted(glob.glob(os.path.join(RESULTS, f"{prefix}_*[0-9a-f].json"))):
        with open(path) as f:
            blob = json.load(f)
        configs.add(json.dumps({k: v for k, v in blob["env"].items() if k != "seconds"}, sort_keys=True))
        blobs.append(blob)
    return blobs, configs


def guard(r):
    return r["mean"] >= -1 and r["lo"] > -2


def main():
    rb, rc = load("R")
    ab, ac = load("A")
    assert len(rc | ac) == 1, f"{len(rc | ac)} configurations"
    R, A, aucs = {}, {}, {}
    for b in rb:
        for row in b["rows"]:
            key = (row["arm"], row["level"], row["graph"])
            assert key not in R, key
            R[key] = row
        cell = (b["rows"][0]["level"], b["rows"][0]["graph"])
        assert cell not in aucs, f"duplicate Part R cell {cell}"
        for sc, v in b["edge_auc"].items():
            assert (v is None) == (cell[0] == 0), (cell, sc, v)
            assert v is None or 0 <= v <= 1, (cell, sc, v)
        aucs[cell] = b["edge_auc"]
    for b in ab:
        for row in b["rows"]:
            key = (row["arm"], row["split"], row["seed"])
            assert key not in A, key
            A[key] = row
    assert set(R) == {(a, l, g) for a in ARMS for l in LEVELS for g in range(GRAPHS)}, "Part R grid incomplete"
    akeys = [(s, e) for s in range(3) for e in range(5)]
    assert set(A) == {(a, s, e) for a in ARMS for s, e in akeys}, "Part A grid incomplete"
    metrics = ("acc", "macro_f1", "clean", "val") + tuple(f"{m}@{r}" for m in ("worst_node", "worst_graph",
                                                                                "own_worst_node", "transfer_sage") for r in (0.1, 0.2))
    need_r = ("acc", "macro_f1", "val")
    need_a = ("clean", "val") + tuple(m for m in metrics if "@" in m)
    for part, rows, need in (("R", R.values(), need_r), ("A", A.values(), need_a)):
        for v in rows:
            for k in need:
                assert k in v and v[k] is not None and np.isfinite(v[k]) and 0 <= v[k] <= 1, (part, v["arm"], k, v.get(k))
            if v["arm"] in ("S-VG", "S-VGr", "S-SJ", "SAGE-VG"):
                assert (v.get("tau"), v.get("eps")) == (0.3, 0.2), (part, v["arm"], v.get("tau"), v.get("eps"))
    assert all(set(x) == set(SCORERS) for x in aucs.values()), "edge AUC scorer keys differ"
    print("Original undirected edges:", sorted({v["orig_undirected_edges"] for v in R.values()}),
          "| injected % per level:", {l: round(R[("S0", l, 0)]["injected_pct"], 1) for l in LEVELS})

    r = lambda arm, lvl, m="acc": np.array([R[(arm, lvl, g)][m] for g in range(GRAPHS)], dtype=float)
    a = lambda arm, m: np.array([A[(arm, s, e)][m] for s, e in akeys], dtype=float)

    print("## Part R: test accuracy by added edges (mean ± sd over 10 graphs, points)")
    print("arm".ljust(10), *[str(l).rjust(12) for l in LEVELS])
    for arm in ARMS:
        print(arm.ljust(10), *[f"{100 * r(arm, l).mean():6.2f}±{100 * r(arm, l).std(ddof=1):4.2f}" for l in LEVELS])
    print("\n## Part R: macro-F1 (mean, points)")
    for arm in ARMS:
        print(arm.ljust(10), *[f"{100 * r(arm, l, 'macro_f1').mean():12.2f}" for l in LEVELS])
    print("\n## Edge AUC, original vs injected edges (mean over graphs; 0.5 = no separation)")
    for sc in SCORERS:
        print(sc.ljust(8), *[f"{np.mean([aucs[(l, g)][sc] for g in range(GRAPHS)]):.3f}" for l in LEVELS if l > 0])

    print("\n## Part A: clean / worst-node@10% / worst-node@20% / own-attack worst-node@20% / SAGE transfer@20% (points)")
    for arm in ARMS:
        print(arm.ljust(10), " / ".join(f"{100 * a(arm, m).mean():.1f}" for m in
                                         ("clean", "worst_node@0.1", "worst_node@0.2", "own_worst_node@0.2", "transfer_sage@0.2")))

    print("\n## H100-H104 (one Holm family)")
    tests = {"H100 S-VG - S0, R 2000": r("S-VG", 2000) - r("S0", 2000),
             "H101 S-VG - S0, R 5000": r("S-VG", 5000) - r("S0", 5000),
             "H102 S-VG - S-VGr, R 5000": r("S-VG", 5000) - r("S-VGr", 5000),
             "H103 S-VG - S0, A worst-node@20%": a("S-VG", "worst_node@0.2") - a("S0", "worst_node@0.2"),
             "H104 SAGE-VG - SAGE, R 5000": r("SAGE-VG", 5000) - r("SAGE", 5000)}
    res = {k: an.paired(d) for k, d in tests.items()}
    ph = an.holm({k: v["p"] for k, v in res.items()})
    verdict = {}
    for k, v in res.items():
        verdict[k[:4]] = "SUPPORTED" if v["mean"] >= 2 and ph[k] < 0.05 else ("AGAINST" if v["hi"] < 1 else "INCONCLUSIVE")
        print(f"{k}: {an.fmt(v)} holm={ph[k]:.4f} -> {verdict[k[:4]]}")
    g_r = an.paired(r("S-VG", 0) - r("S0", 0))
    g_a = an.paired(a("S-VG", "clean") - a("S0", "clean"))
    print(f"Guardrail R clean (0 edges): {an.fmt(g_r)} -> {'MET' if guard(g_r) else 'NOT MET'}")
    print(f"Guardrail A clean: {an.fmt(g_a)} -> {'MET' if guard(g_a) else 'NOT MET'}")
    claim_r = (verdict["H100"] == "SUPPORTED" or verdict["H101"] == "SUPPORTED") and verdict["H102"] == "SUPPORTED" and guard(g_r)
    claim_a = verdict["H103"] == "SUPPORTED" and guard(g_a)
    print("Claim, random injection:", "MET" if claim_r else "NOT MET", "| Claim, adaptive:", "MET" if claim_a else "NOT MET",
          "| Milestone method on SAGE (H104):", verdict["H104"])

    print("\n## Descriptive paired differences (uncorrected)")
    for lvl in LEVELS:
        for x, y in (("S-VG", "S0"), ("S-SJ", "S0"), ("S-VG", "S-SJ"), ("S-VGr", "S0"), ("SAGE-VG", "SAGE"),
                     ("S-VG", "RUNG-lite"), ("S-VG", "MLP-M")):
            print(f"R {lvl} {x} - {y}: {an.fmt(an.paired(r(x, lvl) - r(y, lvl)))}")
    for x, y in (("S-SJ", "S0"), ("S-VG", "S-SJ"), ("SAGE-VG", "SAGE"), ("S-VG", "RUNG-lite"), ("S-VG", "MLP-M")):
        print(f"A worst-node@20% {x} - {y}: {an.fmt(an.paired(a(x, 'worst_node@0.2') - a(y, 'worst_node@0.2')))}")
        print(f"A clean {x} - {y}: {an.fmt(an.paired(a(x, 'clean') - a(y, 'clean')))}")
    print("\n## Per-split effects (split sensitivity; Part R graphs g use split g mod 3: 4/3/3 graphs)")
    gsplit = np.array([g % 3 for g in range(GRAPHS)])
    for k, d in tests.items():
        sp = gsplit if " R " in k else np.array([s for s, _ in akeys])
        print(k, [f"{100 * d[sp == s].mean():+.2f}" for s in range(3)])


if __name__ == "__main__":
    main()
