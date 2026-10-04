"""One-shot literature search for GNN robustness shift/survey notes. Do not invent."""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_raw")
OUT.mkdir(parents=True, exist_ok=True)
UA = "STEM-GNN-research/1.0 (literature notes; mailto:research@local)"

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def arxiv(query, max_results=20, tag="q"):
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = "https://export.arxiv.org/api/query?" + params
    raw = get(url)
    (OUT / f"arxiv_{tag}.xml").write_bytes(raw)
    root = ET.fromstring(raw)
    rows = []
    for e in root.findall("a:entry", NS):
        title = " ".join((e.findtext("a:title", default="", namespaces=NS) or "").split())
        summary = " ".join((e.findtext("a:summary", default="", namespaces=NS) or "").split())
        authors = [a.findtext("a:name", default="", namespaces=NS) for a in e.findall("a:author", NS)]
        published = (e.findtext("a:published", default="", namespaces=NS) or "")[:10]
        aid = (e.findtext("a:id", default="", namespaces=NS) or "").strip()
        comment = e.findtext("arxiv:comment", default="", namespaces=NS) or ""
        journal = e.findtext("arxiv:journal_ref", default="", namespaces=NS) or ""
        doi = e.findtext("arxiv:doi", default="", namespaces=NS) or ""
        cats = [c.get("term") for c in e.findall("a:category", NS)]
        rows.append(
            {
                "title": title,
                "authors": authors,
                "published": published,
                "id": aid,
                "comment": comment,
                "journal": journal,
                "doi": doi,
                "cats": cats,
                "summary": summary[:1800],
            }
        )
    (OUT / f"arxiv_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"arxiv {tag}: {len(rows)}")
    return rows


def reconstruct_abstract(inv):
    if not inv:
        return ""
    pairs = []
    for word, idxs in inv.items():
        for i in idxs:
            pairs.append((i, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def openalex(search, tag, per_page=20, extra_filter=""):
    filt = "from_publication_date:2023-01-01,to_publication_date:2026-10-04"
    if extra_filter:
        filt = filt + "," + extra_filter
    params = urllib.parse.urlencode(
        {
            "search": search,
            "filter": filt,
            "per-page": per_page,
            "sort": "cited_by_count:desc",
            "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index,type,open_access",
        }
    )
    url = "https://api.openalex.org/works?" + params
    raw = get(url)
    data = json.loads(raw)
    slim = []
    for w in data.get("results", []):
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        authors = []
        for a in (w.get("authorships") or [])[:12]:
            au = a.get("author") or {}
            authors.append(au.get("display_name"))
        slim.append(
            {
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "cited": w.get("cited_by_count"),
                "doi": w.get("doi"),
                "type": w.get("type"),
                "venue": src.get("display_name"),
                "venue_type": src.get("type"),
                "oa_url": (w.get("open_access") or {}).get("oa_url"),
                "authors": authors,
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index"))[:1600],
            }
        )
    (OUT / f"oa_{tag}.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"openalex {tag}: {len(slim)} meta={data.get('meta', {}).get('count')}")
    time.sleep(0.4)


QUERIES = [
    ('surv_robust', 'ti:"survey" AND all:"graph neural" AND (all:robustness OR all:trustworthy OR all:heterophily OR all:"out-of-distribution")'),
    ('ood', 'all:"out-of-distribution" AND all:"graph neural" AND (ti:survey OR ti:generalization OR ti:robust)'),
    ('tta', 'all:"test-time" AND (all:adaptation OR all:training) AND all:"graph" AND all:neural'),
    ('hetero', 'all:heterophily AND all:"graph neural" AND (ti:survey OR ti:robust OR all:homophily)'),
    ('missing', 'all:"missing" AND all:feature AND all:"graph neural" AND (all:robust OR all:imputation OR all:incomplete)'),
    ('degree', 'all:"degree" AND (all:shift OR all:bias) AND all:"graph neural"'),
    ('bn_temp', '(all:"batch normalization" OR all:"temperature scaling") AND all:"graph neural" AND (all:shift OR all:"test-time" OR all:calibration)'),
]

if __name__ == "__main__":
    for tag, q in QUERIES:
        try:
            arxiv(q, 12, tag)
        except Exception as exc:
            print("ARXIV FAIL", tag, exc)
        time.sleep(3.2)
    oa = [
        ("surv", "graph neural network robustness survey"),
        ("oodsurv", "graph neural network out-of-distribution generalization survey"),
        ("trust", "trustworthy graph neural networks survey"),
        ("tta", "test-time adaptation graph neural networks"),
        ("hetero", "heterophily graph neural network survey"),
        ("missing", "graph neural network missing node features"),
        ("degree", "graph neural network degree shift robustness"),
        ("homo", "graph neural network homophily shift"),
        ("covariate", "graph neural network covariate shift"),
        ("bn", "batch normalization test-time adaptation graph neural network"),
        ("temp", "temperature scaling graph neural network calibration"),
        ("cited", "increasing robustness graph neural networks adversarial"),
    ]
    for tag, q in oa:
        try:
            openalex(q, tag)
        except Exception as exc:
            print("OA FAIL", tag, exc)
    print("DONE")
