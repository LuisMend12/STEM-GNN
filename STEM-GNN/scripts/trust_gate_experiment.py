"""
Make-or-break experiment: LLM-Guided Semantic Trust Gate
Compares 4 gate signals under 2 stress conditions.

Gate signals:
  A - cosine of GNN hidden states (GNNGuard-style)
  B - model confidence on neighbours (Mowst-style)
  C - learned gate from hidden states (ACM-style)
  D - cosine of raw LLM text embeddings (ours)

Stress conditions:
  1 - feature-close but label-wrong edge injection (Cora)
  2 - heterophily graph (Texas / WebKB)

Run from STEM-GNN/ root:
  python STEM-GNN/scripts/trust_gate_experiment.py
"""
import os, sys, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.utils import add_self_loops, to_undirected
from torch_scatter import scatter_mean

from dataset.process_datasets import get_finetune_graph
from utils.preprocess import pre_node


# ---------------------------------------------------------------------------
# Trust-Gated SAGE layer
# ---------------------------------------------------------------------------
class TrustGatedSAGELayer(nn.Module):
    """
    h_v = beta_v * W_self(x_v) + (1 - beta_v) * W_neigh(mean_u(x_u))

    beta_v is driven by one of 4 gate signals.
    """

    def __init__(self, in_dim, out_dim, gate_type: str, hidden_for_gate=64):
        super().__init__()
        self.gate_type = gate_type
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)

        # Gate C needs a small MLP operating on concatenated features
        if gate_type == "C_learned":
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim * 2, hidden_for_gate),
                nn.ReLU(),
                nn.Linear(hidden_for_gate, 1),
            )

    def forward(self, x, edge_index, text_feat=None, prev_logits=None):
        """
        x          : [N, in_dim]   — current node features / hidden states
        text_feat  : [N, text_dim] — raw LLM embeddings (needed only for D)
        prev_logits: [N, C]        — soft class prediction (needed only for B)
        """
        row, col = edge_index           # row = destination, col = source
        N = x.size(0)

        # Neighbour aggregate
        x_agg = scatter_mean(x[col], row, dim=0, dim_size=N)   # [N, in_dim]

        # Self and neigh projections
        self_out  = self.W_self(x)      # [N, out_dim]
        neigh_out = self.W_neigh(x_agg) # [N, out_dim]

        # --- Gate signal ---
        if self.gate_type == "A_cosine":
            # Cosine similarity of hidden state vs neighbour aggregate
            cos = F.cosine_similarity(x, x_agg, dim=-1)  # [N]
            beta = 1.0 - torch.sigmoid(cos)               # high similarity → trust neighbours more → lower beta

        elif self.gate_type == "B_confidence":
            # Mean max-probability of neighbours
            if prev_logits is None:
                beta = torch.full((N,), 0.5, device=x.device)
            else:
                prob = F.softmax(prev_logits, dim=-1)          # [N, C]
                conf = prob.max(dim=-1).values                  # [N]
                neigh_conf = scatter_mean(conf[col], row, dim=0, dim_size=N)  # [N]
                beta = 1.0 - neigh_conf                        # low confidence → trust self more

        elif self.gate_type == "C_learned":
            gate_in = torch.cat([x, x_agg], dim=-1)       # [N, 2*in_dim]
            beta = self.gate_mlp(gate_in).squeeze(-1).sigmoid()  # [N]

        elif self.gate_type == "D_llm":
            if text_feat is None:
                raise ValueError("gate_type D_llm requires text_feat")
            text_agg = scatter_mean(text_feat[col], row, dim=0, dim_size=N)
            cos = F.cosine_similarity(text_feat, text_agg, dim=-1)
            beta = 1.0 - torch.sigmoid(cos)

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        beta = beta.unsqueeze(-1)                          # [N, 1]
        return beta * self_out + (1.0 - beta) * neigh_out  # [N, out_dim]


# ---------------------------------------------------------------------------
# Full GNN model
# ---------------------------------------------------------------------------
class TrustGatedGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, gate_type, text_dim=None):
        super().__init__()
        self.gate_type = gate_type
        self.layer1 = TrustGatedSAGELayer(in_dim, hidden_dim, gate_type)
        self.layer2 = TrustGatedSAGELayer(hidden_dim, hidden_dim, gate_type)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # For gate B we need a quick preliminary predictor
        if gate_type == "B_confidence":
            self.pre_clf = nn.Linear(in_dim, num_classes)

    def forward(self, x, edge_index, text_feat=None):
        prev_logits = None
        if self.gate_type == "B_confidence":
            prev_logits = self.pre_clf(x).detach()

        h = self.layer1(x, edge_index, text_feat=text_feat, prev_logits=prev_logits)
        h = F.relu(h)

        # For layer 2 gate B, update prev_logits with layer-1 output
        if self.gate_type == "B_confidence":
            prev_logits = self.classifier(h).detach()

        h = self.layer2(h, edge_index, text_feat=text_feat, prev_logits=prev_logits)
        h = F.relu(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Stress condition 1: inject confusing edges
# ---------------------------------------------------------------------------
def inject_confusing_edges(edge_index, node_feat, labels, injection_ratio=0.3, sim_threshold=0.75):
    """
    For each node, find top-k embedding-similar but different-class nodes.
    Add injection_ratio * N new edges.
    """
    N = node_feat.size(0)
    feat_norm = F.normalize(node_feat, dim=-1)  # [N, D]
    num_inject = int(injection_ratio * N)

    new_src, new_dst = [], []
    indices = list(range(N))
    random.shuffle(indices)
    added = 0
    batch_size = 256

    for start in range(0, N, batch_size):
        if added >= num_inject:
            break
        batch_idx = torch.arange(start, min(start + batch_size, N))
        sim_matrix = feat_norm[batch_idx] @ feat_norm.T  # [B, N]
        for i, vi in enumerate(batch_idx.tolist()):
            if added >= num_inject:
                break
            row_sim = sim_matrix[i]                        # [N]
            diff_class = labels != labels[vi]              # [N] bool
            row_sim[vi] = -1                               # exclude self
            row_sim[~diff_class] = -1                      # exclude same-class
            best_j = row_sim.argmax().item()
            if row_sim[best_j].item() >= sim_threshold:
                new_src.append(vi)
                new_dst.append(best_j)
                added += 1

    if len(new_src) == 0:
        return edge_index

    new_edges = torch.tensor([new_src + new_dst, new_dst + new_src],
                              dtype=torch.long, device=edge_index.device)
    combined = torch.cat([edge_index, new_edges], dim=1)
    return combined


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_eval(gate_type, x, edge_index, text_feat, labels,
               train_mask, val_mask, test_mask,
               in_dim, hidden_dim=128, num_classes=7,
               epochs=300, lr=5e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = x.device
    model = TrustGatedGNN(in_dim, hidden_dim, num_classes, gate_type,
                          text_dim=text_feat.size(1) if text_feat is not None else None
                          ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val, best_test = 0.0, 0.0
    for epoch in range(epochs):
        model.train()
        logits = model(x, edge_index, text_feat)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, text_feat)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()
            if val_acc > best_val:
                best_val  = val_acc
                best_test = test_acc

    return best_test


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_stress1_cora(data_dir, seeds=3, injection_ratio=0.3, hidden_dim=128, epochs=300):
    """Stress 1: inject confusing edges into Cora."""
    print("\n" + "="*60)
    print("STRESS 1 — Feature-close but label-wrong edge injection (Cora)")
    print("="*60)

    dataset, split_list, labels, num_classes, _ = get_finetune_graph(data_dir, "cora")
    dataset = pre_node(dataset)
    data    = dataset[0]
    labels  = labels.long()
    data.y  = labels

    # node features = node_text_feat indexed by data.x
    node_text_feat = data.node_text_feat           # [N, 768]
    x = node_text_feat                             # use as GNN input too

    # Use split 0
    sp = split_list[0]
    train_idx = sp["train"]
    val_idx   = sp["valid"]
    test_idx  = sp["test"]

    # Boolean masks
    N = data.num_nodes
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    edge_index_clean = to_undirected(data.edge_index, num_nodes=N)
    edge_index_stress = inject_confusing_edges(
        edge_index_clean, node_text_feat, labels,
        injection_ratio=injection_ratio
    )
    n_new = edge_index_stress.size(1) - edge_index_clean.size(1)
    print(f"Cora: {N} nodes | {edge_index_clean.size(1)} edges → +{n_new} injected")

    gates = ["A_cosine", "B_confidence", "C_learned", "D_llm"]
    results_clean  = {g: [] for g in gates}
    results_stress = {g: [] for g in gates}

    for seed in range(seeds):
        for gate in gates:
            tf = node_text_feat if gate == "D_llm" else None
            acc_c = train_eval(gate, x, edge_index_clean, tf, labels,
                               train_mask, val_mask, test_mask,
                               in_dim=768, hidden_dim=hidden_dim,
                               num_classes=num_classes, epochs=epochs, seed=seed)
            acc_s = train_eval(gate, x, edge_index_stress, tf, labels,
                               train_mask, val_mask, test_mask,
                               in_dim=768, hidden_dim=hidden_dim,
                               num_classes=num_classes, epochs=epochs, seed=seed)
            results_clean[gate].append(acc_c)
            results_stress[gate].append(acc_s)
            print(f"  seed={seed} | gate={gate} | clean={acc_c*100:.1f}% | stress={acc_s*100:.1f}%")

    print("\n--- Stress 1 Summary ---")
    print(f"{'Gate':<14} {'Clean':>8} {'Stressed':>10} {'Drop':>8}")
    for gate in gates:
        c = np.mean(results_clean[gate]) * 100
        s = np.mean(results_stress[gate]) * 100
        print(f"{gate:<14} {c:>8.1f} {s:>10.1f} {s-c:>+8.1f}")
    return results_clean, results_stress


def run_stress2_texas(data_root, seeds=3, hidden_dim=128, epochs=300):
    """Stress 2: heterophily graph (Texas)."""
    print("\n" + "="*60)
    print("STRESS 2 — Heterophily graph (Texas / WebKB)")
    print("="*60)

    ds   = WebKB(data_root, name="Texas")
    data = ds[0]
    N    = data.num_nodes
    x    = data.x.float()          # [183, 1703] bag-of-words
    # Normalise so features are unit vectors (improves cosine gates)
    x = F.normalize(x, dim=-1)
    labels = data.y.long()
    num_classes = int(labels.max().item()) + 1

    # For D_llm in Texas, use the raw node features as proxy text embedding
    # (no real LLM embeddings available; this tests whether the signal degrades)
    text_feat_proxy = x   # same as input; D will behave like A on Texas

    edge_index = to_undirected(data.edge_index, num_nodes=N)
    # Add self-loops so every node has itself as a "neighbour"
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    # Compute homophily ratio for info
    src, dst = edge_index
    same = (labels[src] == labels[dst]).float().mean().item()
    print(f"Texas: {N} nodes | {edge_index.size(1)} edges | homophily={same:.3f}")

    in_dim = x.size(1)

    gates = ["A_cosine", "B_confidence", "C_learned", "D_llm"]
    results = {g: [] for g in gates}

    for split_idx in range(min(seeds, data.train_mask.size(1))):
        train_mask = data.train_mask[:, split_idx]
        val_mask   = data.val_mask[:, split_idx]
        test_mask  = data.test_mask[:, split_idx]
        for gate in gates:
            tf = text_feat_proxy if gate == "D_llm" else None
            acc = train_eval(gate, x, edge_index, tf, labels,
                             train_mask, val_mask, test_mask,
                             in_dim=in_dim, hidden_dim=hidden_dim,
                             num_classes=num_classes, epochs=epochs, seed=split_idx)
            results[gate].append(acc)
            print(f"  split={split_idx} | gate={gate} | acc={acc*100:.1f}%")

    print("\n--- Stress 2 Summary ---")
    print(f"{'Gate':<14} {'Acc (%)':>10}")
    for gate in gates:
        print(f"{gate:<14} {np.mean(results[gate])*100:>10.1f}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="STEM-GNN/dataset", help="STEM-GNN data path")
    parser.add_argument("--webkb_dir",  default="/tmp/webkb",       help="WebKB download cache")
    parser.add_argument("--seeds",      type=int, default=3)
    parser.add_argument("--epochs",     type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--injection",  type=float, default=0.3)
    parser.add_argument("--stress",     choices=["1", "2", "both"], default="both")
    args = parser.parse_args()

    if args.stress in ("1", "both"):
        run_stress1_cora(
            args.data_dir, seeds=args.seeds,
            injection_ratio=args.injection,
            hidden_dim=args.hidden_dim, epochs=args.epochs,
        )

    if args.stress in ("2", "both"):
        run_stress2_texas(
            args.webkb_dir, seeds=args.seeds,
            hidden_dim=args.hidden_dim, epochs=args.epochs,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
