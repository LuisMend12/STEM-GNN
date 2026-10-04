"""Dump abstracts of prior hits and run gap-filling searches."""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")
UA = "STEM-GNN-research/1.0 (literature notes)"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def reconstruct(inv):
    if not inv:
        return ""
    pairs = []
    for word, idxs in inv.items():
        for i in idxs:
            pairs.append((i, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def slim(w):
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    authors = []
    for a in (w.get("authorships") or [])[:18]:
        au = (a.get("author") or {}).get("display_name")
        if au:
            authors.append(au)
    return {
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "cited": w.get("cited_by_count"),
        "doi": w.get("doi"),
        "type": w.get("type"),
        "venue": src.get("display_name"),
        "oa_url": (w.get("open_access") or {}).get("oa_url"),
        "authors": authors,
        "abstract": reconstruct(w.get("abstract_inverted_index")),
    }


def dump_existing():
    lines = []
    for f in sorted(OUT.glob("title_*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        lines.append("\n" + "=" * 20 + " " + f.name + " " + "=" * 20)
        for r in rows:
            t = (r.get("title") or "")
            lines.append("-" * 60)
            lines.append(f"TITLE: {t}")
            lines.append(f"YEAR: {r.get('year')} CITES: {r.get('cited')} VENUE: {r.get('venue')}")
            lines.append(f"AUTHORS: {'; '.join(r.get('authors') or [])}")
            lines.append(f"DOI: {r.get('doi')}")
            lines.append(f"OA: {r.get('oa_url')}")
            lines.append(f"ABS: {r.get('abstract')}")
    (OUT / "abstracts_round1.txt").write_text("\n".join(lines), encoding="utf-8")
    print("dumped", len(lines))


def oa(filt, tag, per_page=12, sort="cited_by_count:desc"):
    params = urllib.parse.urlencode(
        {
            "filter": filt,
            "per-page": per_page,
            "sort": sort,
            "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index,type,open_access",
        }
    )
    data = get("https://api.openalex.org/works?" + params)
    rows = [slim(w) for w in data.get("results", [])]
    (OUT / f"gap_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n== {tag} n={data.get('meta', {}).get('count')} ==")
    for r in rows:
        print(f"  [{r['year']}] c={r['cited']} | {r['venue']} | {r['title']}")
    time.sleep(0.3)
    return rows


if __name__ == "__main__":
    dump_existing()
    queries = [
        ("homophily_t", "title.search:homophily graph neural,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("degree_bias", "title.search:degree bias graph,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("struct_shift", "title.search:structural distribution shift,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("impute", "title.search:imputation graph neural,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("missing_attr", "title.search:missing attribute graph,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("ood_gen", "title.search:out-of-distribution graph neural,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("robust_gcn", "title.search:robust graph neural,from_publication_date:2024-01-01,to_publication_date:2026-10-04"),
        ("feature_shift", "title.search:feature shift graph neural,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("bn_abs", "title.search:graph neural,abstract.search:batch normalization,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("tta_bn", "abstract.search:test-time batch,title.search:graph,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("gtrans", "title.search:graph transformation test-time,from_publication_date:2023-01-01,to_publication_date:2026-10-04"),
        ("hetero_method", "title.search:heterophily,from_publication_date:2024-01-01,to_publication_date:2026-10-04"),
    ]
    for tag, filt in queries:
        try:
            oa(filt, tag)
        except Exception as exc:
            print("FAIL", tag, type(exc).__name__, exc)
    print("DONE")
