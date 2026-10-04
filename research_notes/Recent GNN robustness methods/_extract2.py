import re
from pathlib import Path

OUT = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")


def visible(html):
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"</p>", "\n", html, flags=re.I)
    html = re.sub(r"</h[1-6]>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s+", "\n", html)
    return html


def around(text, pat, before=200, after=1800, n=3):
    out = []
    for m in re.finditer(pat, text, flags=re.I):
        out.append(text[max(0, m.start() - before): m.start() + after])
        if len(out) >= n:
            break
    return "\n---\n".join(out) if out else "[none]"


hetero = visible((OUT / "hetero_surv.html").read_text(encoding="utf-8", errors="replace"))
trust = visible((OUT / "trust_ieee.html").read_text(encoding="utf-8", errors="replace"))
shift = visible((OUT / "shift_surv.html").read_text(encoding="utf-8", errors="replace"))
allev = visible((OUT / "over_allev.html").read_text(encoding="utf-8", errors="replace"))

# find GTrans
idx = shift.lower().find("gtrans")
gtrans = shift[max(0, idx - 400): idx + 700] if idx >= 0 else "[no gtrans]"

# hetero future remainder: from Theoretical
i = hetero.find("Theoretical Heterophily")
hetero_rest = hetero[i: i + 4500] if i >= 0 else "[no]"

# trust six directions: from "six trending"
j = trust.find("six trending")
trust_rest = trust[j: j + 5500] if j >= 0 else "[no]"

# batch norm context in allev
k = allev.lower().find("batch norm")
bn = allev[max(0, k - 250): k + 400] if k >= 0 else "[no bn]"

# OOD subcats one paragraph
p = shift.find("Data Augmentation")
oodsub = shift[p: p + 1600] if p >= 0 else "[no]"

text = "\n\n==== GTRANS ====\n" + gtrans + "\n\n==== HETERO REST ====\n" + hetero_rest + "\n\n==== TRUST REST ====\n" + trust_rest + "\n\n==== BN ====\n" + bn + "\n\n==== OOD SUB ====\n" + oodsub
(OUT / "extracts2.txt").write_text(text, encoding="utf-8")
print(len(text))
