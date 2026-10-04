import json
import re

src = r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_oa_raw.json"
dst = r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_abs_dump.txt"
data = json.load(open(src, encoding="utf-8"))
want = [
    "BDP:",
    "A Dual Robust",
    "Improve Robustness of Graph Neural Networks: Multi-hop",
    "Adversarial Contrastive Graph Masked",
    "Pure-GNN",
    "Refine then Classify",
    "Adversarial Attack Defense in Graph Neural Networks via Multiview",
    "ADEdgeDrop",
    "Towards Inductive Robustness",
    "ADGAT",
    "A Robust Defense Model with Graph Purification",
    "An Adversarial Attack Defense Method Based on Graph Purification",
    "Self-supervised Adversarial Purification",
    "Evolutionary graph structure learning",
    "DSGNN",
    "IDEA:",
    "Robust Mid-Pass Filtering",
    "Are Defenses for Graph Neural Networks Robust",
    "Pruning Graphs by Adversarial",
    "Enhancing Graph Classification Robustness",
    "GJDNet",
    "Topology-Aware Gaussian",
    "Robust Graph Neural Networks via Unbiased",
    "Defense Against Graph Injection",
]
seen = set()
lines = []
for q, rows in data.items():
    if not isinstance(rows, list):
        continue
    for r in rows:
        title = r.get("title") or ""
        if title in seen:
            continue
        if any(title.startswith(w) or w in title for w in want):
            seen.add(title)
            lines.append("=" * 80)
            lines.append(f"{r.get('year')} | {title}")
            lines.append(f"venue={r.get('venue')} doi={r.get('doi')} cited={r.get('cited')} oa={r.get('oa')}")
            lines.append("authors=" + "; ".join(r.get("authors") or []))
            lines.append(r.get("abs") or "(no abstract)")
            lines.append("")
open(dst, "w", encoding="utf-8").write("\n".join(lines))
print("papers", len(seen), "chars", len("\n".join(lines)))
