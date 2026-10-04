import json
import os
import time
import urllib.parse
import urllib.request

outp = r"G:\broswer download\STEM-GNN\research_notes\Recent GNN robustness methods\_oa_raw.json"
UA = {"User-Agent": "STEM-GNN-research/1.0 (mailto:research@example.com)"}


def recon(idx):
    if not idx:
        return ""
    pairs = []
    for w, ps in idx.items():
        for p in ps:
            pairs.append((p, w))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def search(q, year_from=2023):
    filt = f"from_publication_date:{year_from}-01-01"
    url = (
        "https://api.openalex.org/works?search="
        + urllib.parse.quote(q)
        + "&filter="
        + urllib.parse.quote(filt)
        + "&per-page=6&mailto=research@example.com"
    )
    req = urllib.request.Request(url, headers=UA)
    data = json.loads(urllib.request.urlopen(req, timeout=45).read().decode())
    rows = []
    for w in data.get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        authors = [
            (a.get("author") or {}).get("display_name")
            for a in (w.get("authorships") or [])[:12]
        ]
        rows.append(
            {
                "year": w.get("publication_year"),
                "title": w.get("title"),
                "venue": loc.get("display_name"),
                "authors": authors,
                "doi": w.get("doi"),
                "cited": w.get("cited_by_count"),
                "oa": (w.get("open_access") or {}).get("oa_url"),
                "abs": recon(w.get("abstract_inverted_index")),
            }
        )
    return rows


queries = [
    "Robust Graph Neural Networks via Unbiased Aggregation",
    "Defense Against Graph Injection Attack Graph Neural Networks",
    "Bipartite Graph Adversarial Defense Graph Purification",
    "Dual Robust Graph Neural Network Against Graph Adversarial Attacks",
    "Homophily-based Truncation Defense graph neural",
    "soft median aggregation graph neural network",
    "Elastic Graph Neural Networks robust",
    "EvenNet Ignoring Odd-Hop Neighbors",
    "TWIRLS graph neural network iteratively reweighted",
    "robust message passing adversarial edges node classification",
    "graph purification adversarial edges node classification",
    "GNNGuard robust graph defense",
    "Topology-Aware Gaussian Graph Repair",
    "Joint Disentangled Learning adversarial graph neural",
    "Pruning Graphs by Adversarial Robustness Evaluation",
    "Robust Singular Pooling graph classification",
    "unbiased aggregation graph neural network Hou",
]

allr = {}
for q in queries:
    print("Q", q[:80])
    try:
        allr[q] = search(q)
        top = allr[q][0]["title"] if allr[q] else None
        print("  n", len(allr[q]), "top:", top)
    except Exception as e:
        allr[q] = {"error": str(e)}
        print("  ERR", type(e).__name__, e)
    time.sleep(0.35)

with open(outp, "w", encoding="utf-8") as f:
    json.dump(allr, f, indent=2)
print("WROTE", outp)
