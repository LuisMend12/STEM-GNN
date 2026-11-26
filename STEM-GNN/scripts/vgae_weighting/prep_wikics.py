"""Build Wiki-CS in the same .pt format as the repo's Cora/Pubmed (x = 384-d sentence embeddings).

Graph, labels and splits come from PyG's WikiCS (the 20 official train/val splits; one shared test set).
Node text follows the repo's OFA recipe (dataset/data/single_graph/wikics/gen_data.py). The text is
encoded with sentence-transformers/all-MiniLM-L6-v2 (384-d). The encoder of Cora's and Pubmed's x is not
recorded in the repo, so feature comparability with them is assumed, not verified.

Usage (from repo root, on a node with internet):
  python STEM-GNN/scripts/vgae_weighting/prep_wikics.py
"""
import functools
import hashlib
import json
import os

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WikiCS

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "..", "..", "dataset", "data", "single_graph", "wikics")
OUT = os.path.join(WDIR, "wikics.pt")
RAW = os.path.join(HERE, "..", "..", "..", "cache_data", "wikics_pyg")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def node_texts():
    with open(os.path.join(WDIR, "metadata.json")) as f:
        meta = json.load(f)
    texts = [("feature node. wikipedia entry name: " + n["title"] + ". entry content: "
              + functools.reduce(lambda a, b: a + " " + b, n["tokens"])).lower().strip() for n in meta["nodes"]]
    return texts, [n["label"] for n in meta["nodes"]], meta


def main():
    pyg = WikiCS(RAW, is_undirected=True)[0]
    texts, meta_labels, meta = node_texts()
    n = pyg.num_nodes
    if len(texts) != n:
        raise SystemExit(f"metadata has {len(texts)} nodes, PyG WikiCS has {n}")
    label_to_id = {name: int(key) for key, name in meta["labels"].items()}
    if len(label_to_id) != len(meta["labels"]):
        raise SystemExit("metadata label names are not unique")
    try:
        meta_y = torch.tensor([label_to_id[label] for label in meta_labels], dtype=pyg.y.dtype)
    except KeyError as exc:
        raise SystemExit(f"unknown metadata node label: {exc}") from exc
    # detects label-sequence mismatches, not permutations within a class (upstream documents a shared order)
    if not torch.equal(meta_y, pyg.y.cpu()):
        raise SystemExit("metadata label sequence does not match PyG WikiCS labels")
    if pyg.train_mask.shape != (n, 20) or pyg.val_mask.shape != (n, 20) or pyg.test_mask.shape != (n,):
        raise SystemExit("unexpected Wiki-CS train/validation/test mask shapes")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER, device="cpu")
    x = torch.from_numpy(enc.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)).float()
    if x.shape != (n, 384) or not torch.isfinite(x).all():
        raise SystemExit(f"bad embeddings {tuple(x.shape)}")
    splits = pyg.train_mask.size(1)
    train = [pyg.train_mask[:, i].clone() for i in range(splits)]
    val = [pyg.val_mask[:, i].clone() for i in range(splits)]
    test = [pyg.test_mask.clone() for _ in range(splits)]
    for a, b, c in zip(train, val, test):
        if (a & b).any() or (a & c).any() or (b & c).any():
            raise SystemExit("overlapping split masks")
    data = Data(x=x, edge_index=pyg.edge_index, y=pyg.y, train_masks=train, val_masks=val, test_masks=test,
                raw_texts=texts, label_names=list(meta["labels"].values()))
    torch.save(data, OUT)
    info = {"nodes": n, "directed_edges": int(pyg.edge_index.size(1)), "classes": int(pyg.y.max()) + 1,
            "splits": splits, "train": int(train[0].sum()), "val": int(val[0].sum()), "test": int(test[0].sum()),
            "encoder": ENCODER, "sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest()}
    print(json.dumps(info))


if __name__ == "__main__":
    main()
