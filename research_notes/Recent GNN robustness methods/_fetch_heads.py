"""Extract survey headings and a few missing abstracts. One gap search."""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")
UA = "STEM-GNN-research/1.0 (literature notes)"


def get_bytes(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def headings_from_html(html: str):
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    heads = re.findall(r"<h([2-4])[^>]*>([\s\S]*?)</h\1>", text, flags=re.I)
    clean = []
    for lvl, inner in heads:
        t = re.sub(r"<[^>]+>", " ", inner)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            clean.append(f"H{lvl}: {t}")
    return clean


def grab(url, tag):
    raw = get_bytes(url)
    html = raw.decode("utf-8", errors="replace")
    (OUT / f"{tag}.html").write_text(html, encoding="utf-8")
    heads = headings_from_html(html)
    (OUT / f"{tag}_heads.txt").write_text("\n".join(heads), encoding="utf-8")
    print(f"\n==== {tag} heads {len(heads)} ====")
    for h in heads:
        if any(k in h.lower() for k in ["future", "open", "challenge", "taxonom", "robust", "conclusion", "direction", "limitation"]):
            print(h)
    print("--- all h2 ---")
    for h in heads:
        if h.startswith("H2"):
            print(h)
    time.sleep(1.0)


def find_abs(substr):
    hits = []
    for f in list(OUT.glob("title_*.json")) + list(OUT.glob("gap_*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        for r in rows:
            if substr.lower() in (r.get("title") or "").lower():
                hits.append(r)
    # dedupe
    seen = set()
    out = []
    for r in hits:
        if r.get("title") in seen:
            continue
        seen.add(r.get("title"))
        out.append(r)
    return out


if __name__ == "__main__":
    pages = [
        ("shift_surv", "https://ar5iv.labs.arxiv.org/html/2402.16374"),
        ("hetero_surv", "https://ar5iv.labs.arxiv.org/html/2202.07082"),
        ("trust_ieee", "https://ar5iv.labs.arxiv.org/html/2205.07424"),
        ("over_smooth", "https://ar5iv.labs.arxiv.org/html/2303.10993"),
        ("over_allev", "https://ar5iv.labs.arxiv.org/html/2405.01663"),
    ]
    for tag, url in pages:
        try:
            grab(url, tag)
        except Exception as exc:
            print("FAIL", tag, type(exc).__name__, exc)

    want = [
        "FR-GNN",
        "FRGNN",
        "GraphPatcher",
        "Origins of Degree Bias",
        "ES-GNN",
        "PC-Conv",
        "Generalizing Graph Neural Networks on Out-of-Distribution",
        "Brief Survey of Distribution Robust",
        "Energy-based Out-of-Distribution",
        "SSGNN",
        "When does distribution shift break",
        "MaxEnt",
        "COIN-GNN",
        "Distribution Matching for Graph Quantification",
        "Evaluating Robustness and Uncertainty of Graph Models",
        "What Is Missing For Graph Homophily",
        "Triple Filter",
        "Robust graph structure learning under heterophily",
        "DCGNN",
        "Grace: Graph Self-Distillation",
        "Learnable Structural Augmentation",
        "ADEdgeDrop",
    ]
    lines = []
    for s in want:
        for r in find_abs(s):
            lines.append("=" * 60)
            lines.append(r.get("title"))
            lines.append(f"{r.get('year')} c={r.get('cited')} {r.get('venue')}")
            lines.append("; ".join(r.get("authors") or []))
            lines.append(str(r.get("doi")))
            lines.append(str(r.get("oa_url")))
            lines.append(r.get("abstract") or "[none]")
    (OUT / "picked.txt").write_text("\n".join(lines), encoding="utf-8")
    print("picked chars", sum(len(x) for x in lines))
