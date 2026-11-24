"""Analysis for experiments/llm-stem/lowlabel_controls_protocol.md: H90-H94 and descriptives."""
import glob
import importlib.util
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cap_an", os.path.join(HERE, "..", "capped", "analyze.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)  # paired t, Holm and formatting shared with the capped analysis

RESULTS = os.path.join(HERE, "..", "..", "..", "experiments", "llm-stem", "results_lowlabel_controls")
ARMS = ("S0", "S-A", "S-A0", "S-U", "S-T", "MIX", "QWEN", "QWEN0", "MLP-M")
REGIMES = ("V500", "V21")
M = "low_clean"
DRIFT_FLAG = 0.005
IDENTITY = ("code_hash", "stage2_tag", "audit_hash", "texts_hash", "prior_hash", "torch", "python", "threads")


def load():
    cells, envs, drifts, preds = {}, set(), [], {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "llc_split*_seed*_*[0-9a-f].json"))):
        with open(path) as f:
            blob = json.load(f)
        envs.add(json.dumps({k: blob["env"][k] for k in IDENTITY}, sort_keys=True))
        drifts.append(blob["stage2_drift"])
        for row in blob["rows"]:
            key = (row["arm"], row["regime"], row["split"], row["seed"])
            assert key not in cells, f"duplicate {key}"
            assert row[M] is not None and np.isfinite(row[M]), key
            cells[key] = row
        s, e = blob["rows"][0]["split"], blob["rows"][0]["seed"]
        pr = torch.load(path.replace(".json", "_preds.pt"), weights_only=False)
        assert set(pr["preds"]) == set(REGIMES) and all(set(pr["preds"][r]) == set(ARMS) for r in REGIMES), path
        for row in blob["rows"]:  # the saved predictions must reproduce every reported accuracy
            p = pr["preds"][row["regime"]][row["arm"]]
            acc = (p == pr["y"])[pr["test_mask"]].float().mean().item()
            assert abs(acc - row[M]) < 1e-9, (path, row["arm"], row["regime"], acc, row[M])
        preds[(s, e)] = pr
    assert len(envs) == 1, f"{len(envs)} different code/data/audit/environment identities"
    return cells, drifts, preds


def main():
    cells, drifts, preds = load()
    keys = sorted({(s, e) for _, _, s, e in cells})
    assert keys == [(s, e) for s in range(3) for e in range(5)], keys
    assert set(cells) == {(a, r, s, e) for a in ARMS for r in REGIMES for s, e in keys}, "arm/regime/cell set differs"
    get = lambda arm, reg: np.array([cells[(arm, reg, s, e)][M] for s, e in keys], dtype=float)

    worst = {arm: max(d[arm] for d in drifts) for arm in drifts[0]}
    print("Stage 2 reproduction, max |V500 - Stage 2 low_clean| (points):",
          {k: round(100 * v, 3) for k, v in worst.items()},
          "FLAG: above 0.5 points" if max(worst.values()) > DRIFT_FLAG else "(within 0.5 points)")

    print("\n## Low-label test accuracy, mean ± sd over 15 cells (points)")
    for reg in REGIMES:
        print(reg, {a: f"{100 * get(a, reg).mean():.2f}±{100 * get(a, reg).std(ddof=1):.2f}" for a in ARMS})

    print("\n## H90-H94 (one Holm family; run-level intervals conditional on Cora)")
    tests = {"H90 S-A - S-U (V500)": get("S-A", "V500") - get("S-U", "V500"),
             "H91 S-A - S-T (V500)": get("S-A", "V500") - get("S-T", "V500"),
             "H92 S-A0 - S0 (V21)": get("S-A0", "V21") - get("S0", "V21"),
             "H93 S-A0 - S-U (V21)": get("S-A0", "V21") - get("S-U", "V21"),
             "H94 S-A0 - S-T (V21)": get("S-A0", "V21") - get("S-T", "V21")}
    res = {k: an.paired(d) for k, d in tests.items()}
    ph = an.holm({k: r["p"] for k, r in res.items()})
    verdict = {}
    for k, r in res.items():
        v = "SUPPORTED" if r["mean"] >= 2 and ph[k] < 0.05 else ("AGAINST" if r["hi"] < 1 else "INCONCLUSIVE")
        verdict[k[:3]] = v
        print(f"{k}: {an.fmt(r)} holm={ph[k]:.4f} -> {v}")
    print("Claim A, Qwen-prior gain beyond the uniform target with the full validation set (H90):", verdict["H90"])
    strict = verdict["H92"] == "SUPPORTED" and verdict["H93"] == "SUPPORTED"
    print("Claim B, gain survives a strict 21 + 21 label budget (H92 and H93):", "MET" if strict else "NOT MET")
    print("Superiority over the MiniLM-MLP teacher: V500 (H91)", verdict["H91"], "| V21 (H94)", verdict["H94"])

    print("\n## Descriptive (paired, uncorrected)")
    for reg in REGIMES:
        for a in ("S-A", "S-A0", "S-U", "S-T", "MIX", "QWEN", "QWEN0", "MLP-M"):
            print(f"{a} - S0 ({reg}): {an.fmt(an.paired(get(a, reg) - get('S0', reg)))}")
    print(f"S-A - S-A0 (V21): {an.fmt(an.paired(get('S-A', 'V21') - get('S-A0', 'V21')))}")
    print(f"S-A - S0 (V21), Stage 2 prior under the small validation: {an.fmt(an.paired(get('S-A', 'V21') - get('S0', 'V21')))}")
    for a in ARMS:
        print(f"{a} V21 - V500: {an.fmt(an.paired(get(a, 'V21') - get(a, 'V500')))}")
    for reg in REGIMES:
        temps = {t: sorted({cells[("S0", reg, s, e)][f"t_{t}"] for s, e in keys}) for t in ("qwen", "qwen0", "mlp")}
        alpha = sorted({cells[("MIX", reg, s, e)]["alpha"] for s, e in keys})
        print(f"{reg}: temperatures {temps}, MIX alpha {alpha}")

    print("\n## Accuracy on teacher-correct vs teacher-wrong test nodes (points; pooled counts over cells)")
    for reg in REGIMES:
        for student, teacher in (("S-A", "QWEN"), ("S-A0", "QWEN0"), ("S-T", "MLP-M"), ("S0", "QWEN0"), ("S0", "MLP-M")):
            num = {"right": 0, "wrong": 0}
            den = {"right": 0, "wrong": 0}
            for s, e in keys:
                p, y, te = preds[(s, e)]["preds"][reg], preds[(s, e)]["y"], preds[(s, e)]["test_mask"]
                hit = p[student] == y
                for k, m in (("right", (p[teacher] == y) & te), ("wrong", (p[teacher] != y) & te)):
                    num[k] += int(hit[m].sum())
                    den[k] += int(m.sum())
            fmt = lambda k: f"{100 * num[k] / den[k]:.1f} (n={den[k]} node-cell obs.)" if den[k] else "empty"
            print(f"{reg} {student} on {teacher}-correct {fmt('right')} / {teacher}-wrong {fmt('wrong')}")

    print("\n## Per-split means (points), split-level sensitivity")
    splits = np.array([s for s, _ in keys])
    for reg, a, b in (("V500", "S-A", "S-U"), ("V500", "S-A", "S-T"), ("V21", "S-A0", "S0"), ("V21", "S-A0", "S-U"),
                      ("V21", "S-A0", "S-T")):
        d = 100 * (get(a, reg) - get(b, reg))
        print(f"{reg} {a}-{b}:", [f"{d[splits == s].mean():+.2f}" for s in range(3)])


if __name__ == "__main__":
    main()
