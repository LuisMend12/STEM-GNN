import re
import urllib.request

import pymupdf

UA = {"User-Agent": "STEM-GNN-research/1.0 (mailto:research@example.com)"}
outp = r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_pdf_hits.txt"
ids = ["2311.14934", "2301.13694", "2403.09171", "2605.23239", "2606.03462", "2606.01560"]
keys = re.compile(
    r"Cora|CiteSeer|Citeseer|PubMed|Pubmed|Nettack|Metattack|PRBCD|adaptive|accuracy|Accuracy|"
    r"limitation|heterophily|scalab|GNNGuard|Jaccard|ProGNN|RGCN|SoftMedian|failure|weaker|"
    r"does not|cannot|NeurIPS|ICLR|ICML|AAAI",
    re.I,
)


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


chunks = []
for i in ids:
    chunks.append("=" * 80)
    chunks.append(i)
    try:
        data = fetch(f"https://arxiv.org/pdf/{i}.pdf")
        doc = pymupdf.open(stream=data, filetype="pdf")
        chunks.append(f"pages {doc.page_count}")
        # first page for title/venue comments
        first = doc[0].get_text("text")
        chunks.append("--- PAGE0 ---")
        chunks.append(first[:1800])
        hits = []
        for pi, page in enumerate(doc):
            if pi == 0:
                continue
            text = page.get_text("text")
            for para in re.split(r"\n\s*\n", text):
                if keys.search(para) and re.search(r"\d", para):
                    hits.append(f"[p{pi+1}] " + re.sub(r"\s+", " ", para).strip())
        chunks.append(f"hit_paras {len(hits)}")
        for h in hits[:25]:
            chunks.append(h[:500])
        doc.close()
    except Exception as e:
        chunks.append(f"ERR {type(e).__name__}: {e}")

open(outp, "w", encoding="utf-8").write("\n".join(chunks))
print("chars", sum(len(c) for c in chunks))
