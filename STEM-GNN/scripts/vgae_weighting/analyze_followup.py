"""Analysis for experiments/vgae-followup/protocol.md: H110-H115, guardrails, reproduction checks, descriptives."""
import glob
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cap_an", os.path.join(HERE, "..", "capped", "analyze.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)  # paired t, Holm, formatting

ROOT = os.path.join(HERE, "..", "..", "..", "experiments")
RESULTS = os.path.join(ROOT, "vgae-followup", "results")
OLD = os.path.join(ROOT, "vgae-weighting", "results")
LEVELS, GRAPHS = 5, 10
A_CELLS = {"cora": [(s, e) for s in range(3) for e in range(5)], "pubmed": [(s, e) for s in range(3) for e in range(3)],
           "wikics": [(s, e) for s in range(3) for e in range(3)]}
DESCRIPTIVE_ONLY = ("wikics",)  # added after the protocol was fixed: reported, never in the Holm family
A_ARMS = ("S0", "S-VG", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "RUNG-VG", "MLP-M")
R_ARMS = ("S0", "S-VG", "S-VGm", "S-VGp", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "RUNG-VG", "MLP-M")
ATTACKED = ("S0", "S-VG", "SAGE", "SAGE-VG", "RUNG-lite", "RUNG-VG")
SCORERS = ("S-VG", "S-VGm", "S-VGp", "S-SJ")
OLD_ARMS = ("S0", "S-VG", "S-SJ", "SAGE", "SAGE-VG", "GCN", "RUNG-lite", "MLP-M")
OLD_R_ARMS = OLD_ARMS
OLD_TAG = "3405b67aa1"
N_GRAPHS = 2 * (1 + 6 * 5 + 1 + 6 * 8 + 3 * 3)  # per rate: sage + std + gcn transfer + strong + weight-aware
WEIGHTED = {"S-VG", "S-VGm", "S-VGp", "S-SJ", "SAGE-VG", "RUNG-VG"}


def load(ds, part):
    blobs, configs = [], set()
    for path in sorted(glob.glob(os.path.join(RESULTS, f"{ds}_{part}_*[0-9a-f].json"))):
        with open(path) as f:
            blob = json.load(f)
        configs.add(json.dumps({k: v for k, v in blob["env"].items() if k != "seconds"}, sort_keys=True))
        blobs.append(blob)
    return blobs, configs


def guard(r):
    return r["mean"] >= -1 and r["lo"] > -2


def collect(ds):
    rb, rc = load(ds, "R")
    ab, ac = load(ds, "A")
    assert len(rc) == 1 and len(ac) == 1, (ds, len(rc), len(ac))
    assert {json.loads(c)["code_hash"] for c in rc | ac} and len({json.loads(c)["code_hash"] for c in rc | ac}) == 1
    R, A, aucs = {}, {}, {}
    for b in rb:
        for row in b["rows"]:
            key = (row["arm"], row["level_idx"], row["graph"])
            assert key not in R, key
            R[key] = row
        cell = (b["rows"][0]["level_idx"], b["rows"][0]["graph"])
        assert cell not in aucs, cell
        for sc, v in b["edge_auc"].items():
            assert (v is None) == (cell[0] == 0) and (v is None or 0 <= v <= 1), (cell, sc, v)
        aucs[cell] = b["edge_auc"]
    for b in ab:
        for row in b["rows"]:
            key = (row["arm"], row["split"], row["seed"])
            assert key not in A, key
            A[key] = row
    assert set(R) == {(a, l, g) for a in R_ARMS for l in range(LEVELS) for g in range(GRAPHS)}, f"{ds} R grid incomplete"
    assert set(A) == {(a, s, e) for a in A_ARMS for s, e in A_CELLS[ds]}, f"{ds} A grid incomplete"
    assert all(set(v) == set(SCORERS) for v in aucs.values()), "edge AUC scorer keys differ"
    for v in R.values():
        for k in ("acc", "macro_f1", "val"):
            assert np.isfinite(v[k]) and 0 <= v[k] <= 1, (ds, "R", v["arm"], k)
    for v in A.values():
        for k, val in v.items():
            if "@" in k and val is not None and not k.startswith("acc"):
                assert np.isfinite(val) and 0 <= val <= 1, (ds, "A", v["arm"], k)
    for v in list(R.values()) + list(A.values()):
        if v["arm"] in WEIGHTED:
            assert (v.get("tau"), v.get("eps")) == (0.3, 0.2), (ds, v["arm"])
    for b in ab:
        assert len(b["edits"]) == N_GRAPHS, (ds, len(b["edits"]))
    for v in A.values():
        for r in (0.1, 0.2):
            for label in ("own_std", "own_strong", "own_all"):
                value = v[f"{label}@{r}"]
                assert (value is not None) == (v["arm"] in ATTACKED), (ds, v["arm"], r, label, value)
    return R, A, aucs


def reproduction(R, A):
    """Cora: every shared arm and metric must reproduce the original experiment exactly, for every cell."""
    fail = []
    for s_, e in A_CELLS["cora"]:
        with open(os.path.join(OLD, f"A_split{s_}_seed{e}_{OLD_TAG}.json")) as f:
            old = {r["arm"]: r for r in json.load(f)["rows"]}
        for arm in OLD_ARMS:
            new, o = A[(arm, s_, e)], old[arm]
            pairs = [("clean", "clean")] + [(f"{a}@{r}", f"{b}@{r}") for r in (0.1, 0.2) for a, b in (
                ("old_worst_node", "worst_node"), ("old_worst_graph", "worst_graph"),
                ("old_own_worst_node", "own_worst_node"), ("transfer_sage", "transfer_sage"))]
            fail += [(s_, e, arm, a, new[a], o[b]) for a, b in pairs if new[a] != o[b]]
    assert not fail, f"Part A does not reproduce the original: {fail[:5]} ({len(fail)} mismatches)"
    print(f"Cora A: all 15 cells x {len(OLD_ARMS)} shared arms reproduce the original metrics exactly")
    for li, lvl in enumerate((0, 500, 1000, 2000, 5000)):
        for g in range(GRAPHS):
            with open(os.path.join(OLD, f"R_level{lvl}_graph{g}_{OLD_TAG}.json")) as f:
                old = {r["arm"]: r for r in json.load(f)["rows"]}
            for arm in OLD_R_ARMS:
                for m in ("acc", "macro_f1"):
                    if R[(arm, li, g)][m] != old[arm][m]:
                        fail.append((lvl, g, arm, m, R[(arm, li, g)][m], old[arm][m]))
    assert not fail, f"Part R does not reproduce the original: {fail[:5]} ({len(fail)} mismatches)"
    print(f"Cora R: all 50 cells x {len(OLD_R_ARMS)} shared arms reproduce the original accuracy and macro-F1 exactly")


def main():
    data = {ds: collect(ds) for ds in ("cora", "pubmed")}
    for ds in DESCRIPTIVE_ONLY:
        a_files = glob.glob(os.path.join(RESULTS, f"{ds}_A_*[0-9a-f].json"))
        r_files = glob.glob(os.path.join(RESULTS, f"{ds}_R_*[0-9a-f].json"))
        if len(a_files) == len(A_CELLS[ds]) and len(r_files) == LEVELS * GRAPHS:
            data[ds] = collect(ds)
        elif a_files or r_files:
            print(f"{ds}: descriptive grid incomplete; skipping ({len(a_files)} A files, {len(r_files)} R files)")
    reproduction(data["cora"][0], data["cora"][1])
    tests, guards = {}, {}
    for ds, (R, A, aucs) in data.items():
        r = lambda arm, li, m="acc": np.array([R[(arm, li, g)][m] for g in range(GRAPHS)], dtype=float)
        a = lambda arm, m: np.array([A[(arm, s, e)][m] for s, e in A_CELLS[ds]], dtype=float)
        lv = sorted({(v["level_idx"], v["level"], round(v["injected_pct"], 1)) for v in R.values()})
        tag = "  [DESCRIPTIVE ONLY: added after the protocol was fixed]" if ds in DESCRIPTIVE_ONLY else ""
        print(f"\n# {ds.upper()}{tag}  (levels: {lv}; original undirected edges {R[('S0', 0, 0)]['orig_undirected_edges']})")
        print("## Part R: test accuracy by level (mean ± sd over 10 graphs, points)")
        for arm in R_ARMS:
            print(arm.ljust(10), *[f"{100 * r(arm, l).mean():6.2f}±{100 * r(arm, l).std(ddof=1):4.2f}" for l in range(LEVELS)])
        print("## Part R macro-F1 (mean, points)")
        for arm in R_ARMS:
            print(arm.ljust(10), *[f"{100 * r(arm, l, 'macro_f1').mean():8.2f}" for l in range(LEVELS)])
        print("## Edge AUC, original vs injected (mean over graphs)")
        for sc in SCORERS:
            print(sc.ljust(8), *[f"{np.mean([aucs[(l, g)][sc] for g in range(GRAPHS)]):.3f}" for l in range(1, LEVELS)])
        print("## Part R paired differences at each level (descriptive)")
        for x, y in (("S-VG", "S0"), ("S-VGm", "S0"), ("S-VGp", "S0"), ("S-SJ", "S0"), ("S-VGm", "S-VG"),
                     ("S-VGp", "S-VG"), ("SAGE-VG", "SAGE"), ("RUNG-VG", "RUNG-lite")):
            print(f"{x} - {y}:", " | ".join(an.fmt(an.paired(r(x, l) - r(y, l))) for l in range(LEVELS)))

        print("## Part A (points): clean / worst-node@10% / worst-node@20% [expanded union] / old-union@20% / "
              "std-suite@20% / worst-graph@20% / SAGE transfer@20% / GCN transfer@20%")
        for arm in A_ARMS:
            old = "—" if A[(arm, 0, 0)].get("old_worst_node@0.2") is None else f"{100 * a(arm, 'old_worst_node@0.2').mean():.1f}"
            old += " (transfer only)" if arm == "RUNG-VG" and old != "—" else ""
            print(arm.ljust(10), " / ".join([f"{100 * a(arm, m).mean():.1f}" for m in ("clean", "worst_node@0.1", "worst_node@0.2")]
                                            + [old] + [f"{100 * a(arm, m).mean():.1f}" for m in
                                                       ("std_worst_node@0.2", "worst_graph@0.2", "transfer_sage@0.2",
                                                        "transfer_gcn@0.2")]))
        print("## Own-attack worst-node@20% (attacked arms): standard suite -> + strong PRBCD -> + weight-aware attacks")
        for arm in ATTACKED:
            st, sg, al = (100 * a(arm, f"{m}@0.2").mean() for m in ("own_std", "own_strong", "own_all"))
            print(f"{arm.ljust(10)} {st:.1f} -> {sg:.1f} -> {al:.1f}   (strong adds {sg - st:+.1f}, weight-aware adds {al - sg:+.1f})")
        print("## Mean accuracy on each own attack graph at 20% (attacked arms)")
        names = ("ste", "ste_r2", "soft", "soft_t1", "ste_tanh", "s_lr1a", "s_lr1b", "s_lr01", "s_lr10a", "s_lr10b",
                 "s_soft", "s_tanh", "greedy", "bpda_flat", "detach", "twin")
        for arm in ATTACKED:
            vals = []
            for nm in names:
                k = f"{arm}|{nm}|0.2"
                vals.append(f"{nm}={100 * np.mean([A[(arm, s, e)]['acc_by_graph'][k] for s, e in A_CELLS[ds]]):.1f}"
                            if k in A[(arm, 0, 0)]["acc_by_graph"] else None)
            print(arm.ljust(10), " ".join(v for v in vals if v))
        print("## Part A paired differences (descriptive)")
        for x, y in (("S-VG", "S0"), ("SAGE-VG", "SAGE"), ("RUNG-VG", "RUNG-lite"), ("S-VG", "S-SJ"), ("RUNG-VG", "S-VG"),
                     ("RUNG-VG", "MLP-M"), ("RUNG-lite", "MLP-M")):
            print(f"{x} - {y}: worst-node@20% {an.fmt(an.paired(a(x, 'worst_node@0.2') - a(y, 'worst_node@0.2')))}; "
                  f"clean {an.fmt(an.paired(a(x, 'clean') - a(y, 'clean')))}")
        if ds == "cora":
            d_old = an.paired(a("S-VG", "old_worst_node@0.2") - a("S0", "old_worst_node@0.2"))
            print(f"S-VG - S0 on the original union (reproduces H103): {an.fmt(d_old)}")
            tests["H110 cora S-VG - S0, A worst-node@20% (expanded union)"] = a("S-VG", "worst_node@0.2") - a("S0", "worst_node@0.2")
            tests["H111 cora RUNG-VG - RUNG-lite, A worst-node@20%"] = a("RUNG-VG", "worst_node@0.2") - a("RUNG-lite", "worst_node@0.2")
            tests["H112 cora S-VGm - S0, R accuracy, 5000 edges"] = r("S-VGm", 4) - r("S0", 4)
            tests["H113 cora S-VGp - S0, R accuracy, 5000 edges"] = r("S-VGp", 4) - r("S0", 4)
            tests["H116 cora S-VGm - S-VG, R accuracy, 5000 edges"] = r("S-VGm", 4) - r("S-VG", 4)
            tests["H117 cora S-VGp - S-VG, R accuracy, 5000 edges"] = r("S-VGp", 4) - r("S-VG", 4)
            guards["G-H110 cora S-VG - S0 A clean"] = a("S-VG", "clean") - a("S0", "clean")
            guards["G-H111 cora RUNG-VG - RUNG-lite A clean"] = a("RUNG-VG", "clean") - a("RUNG-lite", "clean")
            guards["G-H112 cora S-VGm - S0 R 0 edges"] = r("S-VGm", 0) - r("S0", 0)
            guards["G-H113 cora S-VGp - S0 R 0 edges"] = r("S-VGp", 0) - r("S0", 0)
            masking = a("S-VG", "own_all@0.2") - a("S-VG", "own_strong@0.2")
        elif ds == "pubmed":
            tests["H114 pubmed S-VG - S0, A worst-node@20% (expanded union)"] = a("S-VG", "worst_node@0.2") - a("S0", "worst_node@0.2")
            tests["H115 pubmed RUNG-VG - RUNG-lite, A worst-node@20%"] = a("RUNG-VG", "worst_node@0.2") - a("RUNG-lite", "worst_node@0.2")
            guards["G-H114 pubmed S-VG - S0 A clean"] = a("S-VG", "clean") - a("S0", "clean")
            guards["G-H115 pubmed RUNG-VG - RUNG-lite A clean"] = a("RUNG-VG", "clean") - a("RUNG-lite", "clean")
        print("## Per-split S-VG - S0 worst-node@20% and RUNG-VG - RUNG-lite worst-node@20%")
        sp = np.array([s for s, _ in A_CELLS[ds]])
        for x, y in (("S-VG", "S0"), ("RUNG-VG", "RUNG-lite")):
            d = a(x, "worst_node@0.2") - a(y, "worst_node@0.2")
            print(f"{x} - {y}:", [f"{100 * d[sp == s].mean():+.2f}" for s in range(3)])

    print("\n# H110-H117 (one Holm family; operational rules conditional on these fixed graphs and splits)")
    print("Paired cell differences (points), every hypothesis and guardrail:")
    for k, d in {**tests, **guards}.items():
        print(f"  {k}: {[round(100 * v, 2) for v in d]}")
    res = {k: an.paired(d) for k, d in tests.items()}
    ph = an.holm({k: v["p"] for k, v in res.items()})
    verdict = {}
    for k, v in res.items():
        verdict[k[:4]] = "SUPPORTED" if v["mean"] >= 2 and ph[k] < 0.05 else ("AGAINST" if v["hi"] < 1 else "INCONCLUSIVE")
        print(f"{k}: {an.fmt(v)} holm={ph[k]:.4f} -> {verdict[k[:4]]}")
    gres = {}
    for k, d in guards.items():
        gres[k[2:6]] = guard(an.paired(d))
        print(f"Guardrail {k}: {an.fmt(an.paired(d))} -> {'MET' if gres[k[2:6]] else 'NOT MET'}")
    ok = lambda h: verdict[h] == "SUPPORTED" and gres[h]
    print("\n# Claims")
    print("C1 gain survives stronger, weight-aware attacks (Cora):", "MET" if ok("H110") else "NOT MET")
    print("C2 gain replicates on Pubmed:", "MET" if ok("H114") else "NOT MET")
    print("C3 VGAE weights add robustness inside RUNG-lite: Cora", "MET" if ok("H111") else "NOT MET",
          "| Pubmed", "MET" if ok("H115") else "NOT MET")
    c4m = ok("H112") and verdict["H116"] == "SUPPORTED"
    c4p = ok("H113") and verdict["H117"] == "SUPPORTED"
    print("C4 a noise-aware VGAE improves on S0 and on the original S-VG under random injection (Cora): S-VGm",
          "MET" if c4m else "NOT MET", "| S-VGp", "MET" if c4p else "NOT MET")
    print("Replication signal on two fixed graphs (C1 and C2):", "MET" if ok("H110") and ok("H114") else "NOT MET",
          "(a signal, not by itself a new frontier)")
    print(f"Sensitivity diagnostic (Cora): weight-aware attacks change S-VG own worst-node@20% by "
          f"{100 * masking.mean():+.2f} beyond standard+strong; per cell {[round(100 * v, 1) for v in masking]}; "
          f"{'FLAG (sensitive to the added attacks; cause not identified)' if 100 * masking.mean() <= -3 else 'no flag (does not rule out masking)'}")


if __name__ == "__main__":
    main()
