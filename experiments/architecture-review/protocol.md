# Protocol E1: does STEM-GNN's raw edge-text injection help or hurt? (finding F1a)

Written 2026-10-22, before any result.

## Setting

- **Encoder:** STEM-GNN's real `Encoder`, non-MoE SAGE (the default path), `normalize='batch'`, dropout 0.15, ReLU.
- **Width:** hidden width **384 = input dim**. This is needed so the repo's raw `x_j + xe` addition is shape-valid at every layer.
- **Head:** a linear classification head.
- **Data:** OFA Cora with MiniLM-384 node features.
- **Edge feature:** c = MiniLM(`"feature edge. connected papers are cited together by other papers."`), the exact Cora edge text (`single_graph/Cora/gen_data.py:56`), broadcast to every edge. This is what the repo feeds the encoder on Cora.
- **Training:** Adam (lr 0.01, wd 5e-4), up to 300 epochs, patience 50 on validation accuracy.
- **Runs:** 5 seeds × 3 splits (n = 15, paired). Threads = 4, with `torch.use_deterministic_algorithms(True, warn_only=True)`. Depths {2 (repo default), 4}.

## Variants

| ID | Edge handling |
|---|---|
| **S0** | no edge features (`edge_attr=None`) |
| **S1** | **repo behavior:** raw c added to every layer's messages, `ReLU(x_j + c)` |
| **S2** | raw c added at layer 0 only, where it shares the text space; no edge term deeper |
| **S3** | **proposed EdgeProj:** per-layer learned projection, `ReLU(x_j + W_e^ℓ c + b_e^ℓ)` |

## Hypotheses (paired by split and seed, n = 15)

| ID | Hypothesis | Support | Refute / other outcomes |
|---|---|---|---|
| **H24** | Raw injection effect at depth 2: S1 − S0 | — | ≤ −1 → harmful; ≥ +1 → helpful; otherwise no effect |
| **H25** | EdgeProj beats raw injection: S3 − S1 ≥ +1 at depth 2 or depth 4 | as stated | S3 − S1 ≤ 0 at both depths |
| **H26** | Raw injection hurts more with depth: (S2 − S1) at depth 4 > (S2 − S1) at depth 2 + 1 | as stated | — |

**Scope.** Cora's edge text is constant, so this tests only whether *how* the repo injects edge features (raw, unprojected, at every layer) is sound. It does not test the value of informative edge features (molecules, F1b), which needs `rdkit` and the HuggingFace molecule dataset.
