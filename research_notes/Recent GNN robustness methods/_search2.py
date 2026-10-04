"""Precise title and phrase searches. Relevance + citation sorts."""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")
UA = "STEM-GNN-research/1.0 (literature notes)"


def get(url, headers=None, timeout=60):
    h = {"User-Agent": UA, "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def slim_oa(w):
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    authors = []
    for a in (w.get("authorships") or [])[:15]:
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
        "abstract": reconstruct(w.get("abstract_inverted_index"))[:2200],
    }


def oa_title(phrase, tag, per_page=15):
    # title.search is a filter; quote spaces via urlencode
    filt = f"title.search:{phrase},from_publication_date:2023-01-01,to_publication_date:2026-10-04"
    params = urllib.parse.urlencode(
        {
            "filter": filt,
            "per-page": per_page,
            "sort": "cited_by_count:desc",
            "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index,type,open_access",
        }
    )
    url = "https://api.openalex.org/works?" + params
    data = get(url)
    rows = [slim_oa(w) for w in data.get("results", [])]
    (OUT / f"title_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n== OA title {tag} count={data.get('meta', {}).get('count')} ==")
    for r in rows:
        print(f"  [{r['year']}] c={r['cited']} | {r['venue']} | {r['title']}")
    time.sleep(0.35)


def s2(query, tag, limit=15, year="2023-2026"):
    params = urllib.parse.urlencode(
        {
            "query": query,
            "year": year,
            "limit": limit,
            "fields": "title,year,authors,venue,citationCount,abstract,externalIds,url,publicationDate",
        }
    )
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + params
    try:
        data = get(url)
    except Exception as exc:
        print("S2 FAIL", tag, exc)
        return
    rows = []
    for p in data.get("data", []):
        authors = [a.get("name") for a in (p.get("authors") or [])[:12]]
        ext = p.get("externalIds") or {}
        rows.append(
            {
                "title": p.get("title"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "cited": p.get("citationCount"),
                "date": p.get("publicationDate"),
                "doi": ext.get("DOI"),
                "arxiv": ext.get("ArXiv"),
                "url": p.get("url"),
                "authors": authors,
                "abstract": (p.get("abstract") or "")[:2200],
            }
        )
    (OUT / f"s2_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n== S2 {tag} total={data.get('total')} ==")
    for r in rows:
        print(f"  [{r['year']}] c={r['cited']} | {r['venue']} | {r['title']}")
    time.sleep(1.1)


if __name__ == "__main__":
    titles = [
        ("robust_surv", "graph neural network robustness survey"),
        ("ood_surv", "out-of-distribution graph neural survey"),
        ("dist_shift", "distribution shift graph neural"),
        ("hetero_surv", "heterophily graph neural survey"),
        ("tta", "test-time adaptation graph"),
        ("ttt", "test-time training graph neural"),
        ("missing", "missing features graph neural"),
        ("degree", "degree shift graph neural"),
        ("homo", "homophily shift graph"),
        ("covariate", "covariate shift graph neural"),
        ("trust", "trustworthy graph neural networks"),
        ("domain", "graph domain adaptation survey"),
        ("oversmooth", "oversmoothing graph neural survey"),
        ("bn", "batch normalization graph neural"),
        ("temp", "temperature scaling graph"),
        ("incomplete", "incomplete graph neural network"),
        ("feature_noise", "feature noise graph neural"),
    ]
    for tag, phrase in titles:
        try:
            oa_title(phrase, tag)
        except Exception as exc:
            print("OA FAIL", tag, exc)
    s2_queries = [
        ("robust_surv", "graph neural network robustness survey"),
        ("ood", "graph out-of-distribution generalization survey"),
        ("tta", "test-time adaptation graph neural networks"),
        ("hetero", "heterophily robust graph neural networks"),
        ("missing", "missing node features graph neural network"),
        ("degree", "degree shift graph neural network"),
        ("bn", "test-time batch normalization graph neural network"),
        ("temp", "temperature scaling graph neural network calibration"),
        ("cited_robust", "graph neural network adversarial robustness defense"),
    ]
    for tag, q in s2_queries:
        s2(q, tag)
    print("DONE")
