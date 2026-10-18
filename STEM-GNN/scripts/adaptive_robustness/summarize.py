"""Aggregate results_split*.json into a mean±std markdown table (results.md)."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "results_split*.json"))):
        with open(path) as f:
            rows += json.load(f)["rows"]
    return rows


def fmt(vals):
    m = 100 * st.mean(vals)
    s = 100 * st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.1f}±{s:.1f}"


def main():
    rows = load_rows()
    cols = [k for k in rows[0] if k not in ("model", "split", "val", "seconds")]
    models = list(dict.fromkeys(r["model"] for r in rows))
    lines = ["| model | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        lines.append(f"| {m} | {len(rs)} | " + " | ".join(fmt([r[c] for r in rs]) for c in cols) + " |")
    table = "\n".join(lines)
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
