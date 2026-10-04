import json
import re
import urllib.request

UA = {"User-Agent": "STEM-GNN-research/1.0 (mailto:research@example.com)"}
out = r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_details.txt"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def pdf_strings(data, limit=400000):
    # Extract parenthesized PDF text strings and uncompressed text runs.
    text = data.decode("latin1", errors="ignore")
    chunks = re.findall(r"\((?:\\.|[^\\)]){4,200}\)", text)
    parts = []
    for c in chunks[:8000]:
        s = c[1:-1]
        s = s.replace(r"\n", " ").replace(r"\r", " ")
        s = re.sub(r"\\[0-9]{3}", " ", s)
        s = s.replace("\\", "")
        if re.search(r"[A-Za-z]{4}", s):
            parts.append(s)
    blob = " ".join(parts)
    return blob[:limit]


ids = [
    "2311.14934",  # RUNG
    "2301.13694",  # Are defenses robust
    "2403.09171",  # ADEdgeDrop
    "2605.23239",  # GPR-GAE
    "2606.03462",  # TAGR
    "2606.01560",  # GJDNet
]
lines = []
for i in ids:
    url = f"https://arxiv.org/pdf/{i}.pdf"
    lines.append("=" * 80)
    lines.append(i)
    try:
        data = get(url)
        lines.append(f"bytes {len(data)}")
        blob = pdf_strings(data)
        # keep sentences with numbers and robustness keywords
        bits = re.findall(
            r".{0,80}(?:Cora|CiteSeer|Citeseer|PubMed|Nettack|Metattack|Metattack|adaptive|accuracy|Accuracy|limitation|heterophily|scalab).{0,120}",
            blob,
            flags=re.I,
        )
        lines.append(f"hits {len(bits)}")
        for b in bits[:40]:
            lines.append(re.sub(r"\s+", " ", b))
        if len(bits) < 5:
            lines.append(blob[:2500])
    except Exception as e:
        lines.append(f"ERR {type(e).__name__}: {e}")

open(out, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", out, "chars", sum(len(x) for x in lines))
