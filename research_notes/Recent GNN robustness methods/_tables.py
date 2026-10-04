import urllib.request
import pymupdf

UA = {"User-Agent": "STEM-GNN-research/1.0 (mailto:research@example.com)"}
outp = r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_tables.txt"


def open_pdf(i):
    req = urllib.request.Request(f"https://arxiv.org/pdf/{i}.pdf", headers=UA)
    data = urllib.request.urlopen(req, timeout=90).read()
    return pymupdf.open(stream=data, filetype="pdf")


parts = []
jobs = [
    ("2311.14934", [8, 9]),
    ("2605.23239", [7, 8]),
    ("2606.01560", [9]),
    ("2606.03462", [15]),
]
for i, pages in jobs:
    doc = open_pdf(i)
    for p in pages:
        parts.append(f"\n===== {i} page {p} =====\n")
        parts.append(doc[p - 1].get_text("text"))
    doc.close()
open(outp, "w", encoding="utf-8").write("".join(parts))
print("ok", sum(len(x) for x in parts))
