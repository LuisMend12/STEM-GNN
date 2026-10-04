import json
from pathlib import Path

OUT = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")
NEEDLES = [
    "trustworthy graph neural",
    "heterophily: a survey",
    "heterophily: a survey".lower(),
    "distribution shift",
    "distribution shifts",
    "test-time",
    "test time",
    "oversmoothing",
    "incomplete feature",
    "incomplete features",
    "missing attribute",
    "feature noise",
    "degree bias",
    "degree shift",
    "homophily",
    "heterophily",
    "out-of-distribution",
    "temperature scaling",
    "feature reconstruction",
    "batch norm",
    "covariate",
    "structural distribution",
    "calibration",
    "domain adaptation",
]
# keep only reasonably on-topic by requiring graph in title
seen = set()
lines = []
files = list(OUT.glob("title_*.json")) + list(OUT.glob("gap_*.json"))
for f in files:
    try:
        rows = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    for r in rows:
        title = r.get("title") or ""
        key = title.lower().strip()
        if key in seen:
            continue
        tl = key
        if "graph" not in tl and "gnn" not in tl:
            continue
        if not any(n in tl for n in NEEDLES):
            continue
        # skip obvious junk
        if "gensi" in tl or "zenodo" in (r.get("venue") or "").lower() and "adversarial robustness of graph neural networks under feature" not in tl:
            if "gensi" in tl:
                continue
        seen.add(key)
        lines.append("=" * 70)
        lines.append(f"FILE: {f.name}")
        lines.append(f"TITLE: {title}")
        lines.append(f"YEAR: {r.get('year')} | CITES: {r.get('cited')} | VENUE: {r.get('venue')} | TYPE: {r.get('type')}")
        lines.append("AUTHORS: " + "; ".join(r.get("authors") or []))
        lines.append(f"DOI: {r.get('doi')}")
        lines.append(f"OA: {r.get('oa_url')}")
        lines.append("ABS: " + (r.get("abstract") or "[no abstract]"))
        lines.append("")

text = "\n".join(lines)
(OUT / "digest.txt").write_text(text, encoding="utf-8")
print("papers", len(seen), "chars", len(text))
