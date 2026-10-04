"""Compare weight-norm Lipschitz regularization with a propagation penalty.

The propagation penalty is a finite-difference estimate of how much node
states move when edges are deleted. Evaluation drops edges that touch test
nodes and reports accuracy plus that same relative state change.

Run from the repo root with the interpreter that has torch and torch_geometric:

    python results/prop_lip/run_prop_lip.py
"""
import json
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PKG = osp.join(osp.dirname(__file__), "..", "..", "STEM-GNN")
sys.path.insert(0, osp.abspath(PKG))

from model.encoder import Encoder, drop_undirected_edges

CORA_PATH = osp.abspath(osp.join(PKG, "dataset", "data", "single_graph", "Cora", "cora.pt"))
OUT_PATH = osp.join(osp.dirname(__file__), "prop_lip_results.json")

N_SPLITS = int(os.environ.get("N_SPLITS", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
PATIENCE = int(os.environ.get("PATIENCE", "12"))
HIDDEN = int(os.environ.get("HIDDEN", "256"))
LR = float(os.environ.get("LR", "1e-3"))
DROP_PROBS = (0.0, 0.2, 0.4, 0.6, 1.0)
EVAL_SEEDS = (0, 1, 2)
PROP_DROP = 0.3

CONFIGS = [
    {"name": "baseline", "encoder_lip_coeff": 0.0, "propagation_lip_coeff": 0.0},
    {"name": "weight_lip", "encoder_lip_coeff": 1e-3, "propagation_lip_coeff": 0.0},
    {"name": "prop_lip", "encoder_lip_coeff": 0.0, "propagation_lip_coeff": 1.0},
    {"name": "prop_lip_0.3", "encoder_lip_coeff": 0.0, "propagation_lip_coeff": 0.3},
]


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encoder(x, edge_index, edge_attr)
        return z, self.head(z)


def accuracy(logits, y, mask):
    pred = logits.argmax(dim=-1)
    correct = pred[mask] == y[mask]
    return 100.0 * correct.float().mean().item()


def weight_sq(encoder):
    total = torch.zeros((), device=next(encoder.parameters()).device)
    for layer in encoder.layers:
        for name, param in layer.named_parameters():
            if param.dim() >= 2 and "weight" in name.lower():
                total = total + param.detach().pow(2).sum()
    return float(total.item())


def evaluate_drop(model, x, edge_index, y, test_mask, drop_prob, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dropped_index, _ = drop_undirected_edges(
        edge_index, None, drop_prob=drop_prob, node_scope=test_mask, generator=generator,
    )
    with torch.no_grad():
        clean_z, _ = model(x, edge_index)
        z, logits = model(x, dropped_index)
        acc = accuracy(logits, y, test_mask)
        change = (clean_z - z).norm(dim=-1)
        scale = clean_z.norm(dim=-1).clamp_min(1e-6)
        sensitivity = (change / scale)[test_mask].mean().item()
    return acc, sensitivity


def train_one(data, split_idx, config):
    torch.manual_seed(split_idx)
    x = data.x
    edge_index = data.edge_index
    y = data.y
    train_mask = data.train_masks[split_idx]
    val_mask = data.val_masks[split_idx]
    test_mask = data.test_masks[split_idx]
    num_classes = int(y.max().item()) + 1

    encoder = Encoder(
        input_dim=x.size(1),
        hidden_dim=HIDDEN,
        activation=nn.ReLU,
        num_layers=2,
        backbone="sage",
        normalize="none",
        dropout=0.15,
        moe=False,
    )
    model = Classifier(encoder, num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    stale = 0
    last_parts = {}

    for epoch in range(EPOCHS):
        model.train()
        _, logits = model(x, edge_index)
        ce = F.cross_entropy(logits[train_mask], y[train_mask])
        lip = model.encoder.lipschitz_penalty(config["encoder_lip_coeff"])
        prop = model.encoder.propagation_sensitivity_penalty(
            x, edge_index, None, drop_prob=PROP_DROP, coeff=config["propagation_lip_coeff"],
        )
        loss = ce + lip + prop
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_parts = {"ce": float(ce.item()), "lip": float(lip.item()), "prop": float(prop.item())}

        model.eval()
        with torch.no_grad():
            _, logits = model(x, edge_index)
            val_acc = accuracy(logits, y, val_mask)
            test_acc = accuracy(logits, y, test_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"  [{config['name']} split {split_idx}] epoch {epoch} "
                f"ce {last_parts['ce']:.3f} lip {last_parts['lip']:.3f} prop {last_parts['prop']:.3f} "
                f"val {val_acc:.2f} test {test_acc:.2f}",
                flush=True,
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, logits = model(x, edge_index)
        clean = {
            "train": accuracy(logits, y, train_mask),
            "val": accuracy(logits, y, val_mask),
            "test": accuracy(logits, y, test_mask),
        }

    dropped = {}
    for drop_prob in DROP_PROBS:
        accs, senses = [], []
        for seed in EVAL_SEEDS:
            acc, sense = evaluate_drop(model, x, edge_index, y, test_mask, drop_prob, seed)
            accs.append(acc)
            senses.append(sense)
        dropped[f"{drop_prob:.1f}"] = {
            "test": sum(accs) / len(accs),
            "sensitivity": sum(senses) / len(senses),
        }

    row = {
        "config": config["name"],
        "split": split_idx,
        "best_epoch": best_epoch,
        "best_val_during_train": best_val,
        "weight_sq": weight_sq(model.encoder),
        "clean": clean,
        "dropped": dropped,
        "last_parts": last_parts,
    }
    print(
        f"  [{config['name']} split {split_idx}] best epoch {best_epoch} "
        f"clean test {clean['test']:.2f} | 40% drop {dropped['0.4']['test']:.2f} "
        f"sens {dropped['0.4']['sensitivity']:.3f} | weight_sq {row['weight_sq']:.1f}",
        flush=True,
    )
    return row


def summarize(rows):
    configs = []
    for row in rows:
        if row["config"] not in configs:
            configs.append(row["config"])
    summary = {}
    for name in configs:
        group = [row for row in rows if row["config"] == name]
        def mean_std(values):
            tensor = torch.tensor(values, dtype=torch.float32)
            return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False))}

        entry = {
            "n": len(group),
            "clean_test": mean_std([row["clean"]["test"] for row in group]),
            "clean_val": mean_std([row["clean"]["val"] for row in group]),
            "weight_sq": mean_std([row["weight_sq"] for row in group]),
        }
        for key in group[0]["dropped"]:
            entry[f"drop_{key}_test"] = mean_std([row["dropped"][key]["test"] for row in group])
            entry[f"drop_{key}_sensitivity"] = mean_std([row["dropped"][key]["sensitivity"] for row in group])
        summary[name] = entry
    return summary


def main():
    only = os.environ.get("ONLY_CONFIG", "").strip()
    configs = [cfg for cfg in CONFIGS if not only or cfg["name"] == only]
    data = torch.load(CORA_PATH, map_location="cpu", weights_only=False)
    n_splits = min(N_SPLITS, len(data.train_masks))
    print(
        f"Cora x={tuple(data.x.shape)} edges={data.edge_index.size(1)} "
        f"splits={n_splits} epochs={EPOCHS} hidden={HIDDEN} configs={[c['name'] for c in configs]}",
        flush=True,
    )

    rows = []
    if osp.exists(OUT_PATH) and os.environ.get("RESUME", "") == "1":
        with open(OUT_PATH) as handle:
            rows = json.load(handle).get("rows", [])
        print(f"resumed {len(rows)} rows", flush=True)

    done = {(row["config"], row["split"]) for row in rows}
    for config in configs:
        for split_idx in range(n_splits):
            if (config["name"], split_idx) in done:
                continue
            rows.append(train_one(data, split_idx, config))
            payload = {"rows": rows, "summary": summarize(rows)}
            with open(OUT_PATH, "w") as handle:
                json.dump(payload, handle, indent=2)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
