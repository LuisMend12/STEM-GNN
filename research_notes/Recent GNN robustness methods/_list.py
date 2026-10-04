import json
from pathlib import Path

p = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")
out = []
for f in sorted(p.glob("oa_*.json")):
    out.append("=" * 80)
    out.append(f.name)
    data = json.loads(f.read_text(encoding="utf-8"))
    for i, w in enumerate(data, 1):
        au = ", ".join([a for a in (w.get("authors") or [])[:5] if a])
        out.append(
            f"{i:02d}. [{w.get('year')}] cites={w.get('cited')} | {w.get('venue')} | {w.get('title')}"
        )
        out.append(f"    {au}")
        out.append(f"    doi={w.get('doi')}")
(p / "_titles.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote", len(out), "lines")
