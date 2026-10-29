# Protocol: Text-Anchored Edge Floor (TAEF) against message collapse in STEM-GNN's encoder

Written 2026-10-21, before any result. Novelty has **not** been checked. The closest known work is EGNN, "Dirichlet Energy Constrained Learning for Deep GNNs" (Zhou et al., NeurIPS 2021), which bounds the *graph-level* Dirichlet energy relative to the input features. Run the `prior-art-check` skill before any claim.

## Problem

As message passing gets deeper, node states converge and neighbors become indistinguishable. This is *representation collapse* (oversmoothing). In the MoE encoder, the router can also send most nodes to one expert, which is *expert collapse*.

STEM-GNN has a resource that generic GNNs lack: a frozen, structure-free text embedding `x` for every node, in a space shared across domains.

## The new addition: Text-Anchored Edge Floor (TAEF)

**Idea.** Message passing may pull neighbors together, but not past a fraction ρ of how different their *texts* are. For each layer ℓ, with cosine distance d(a,b) = 1 − cos(a,b), the loss is

L_TAEF = λ · mean_ℓ mean_{(i,j)∈E} relu( ρ · d(x_i, x_j) − d(h_i^ℓ, h_j^ℓ) )

- The floor is **per edge and anchored to the LLM text geometry.** Neighbors whose papers are textually different must stay at least partly distinguishable, while textually similar neighbors are free to merge.
- **It applies unchanged to both the MoE and non-MoE encoders**, as a training regularizer with no architecture change.
- **Fixed in advance:** ρ = 0.5 and λ = 1. No tuning.

**Nearest-prior-art comparator: Graph-level Energy Floor (GEF), EGNN-style.**

L_GEF = λ · mean_ℓ relu( ρ · Ē(x) − Ē(h^ℓ) )

Here Ē is the mean edge cosine distance over the whole graph. GEF is a single *scalar* floor, where TAEF puts a floor on *each edge*. GEF uses the same ρ and λ. Any TAEF gain must come from the edge-wise text anchoring, not from "having an energy floor at all".

## Setting

- **Encoder:** STEM-GNN's real `Encoder` from `model/encoder.py`, with SAGE backbone, ReLU, `normalize='batch'` (the repo default), dropout 0.15, hidden width 64, and a linear classification head.
- **No VQ, no pretraining.** The collapse question concerns the encoder.
- **MoE settings:** `moe=False`, or `moe=True` with `num_experts=3`, `moe_layers='all'`, τ = 1 and `lamda_env = 0` (the repo default).
- **Depth:** L ∈ {2, 4, 8, 16} layers.
- **Data:** OFA Cora with frozen MiniLM-384 text features, splits 0/1/2.
- **Training:** Adam (lr 0.01, wd 5e-4), up to 300 epochs, early stopping on validation accuracy with patience 50.

## Metrics (eval mode, clean graph)

| Metric | What it measures |
|---|---|
| Test accuracy | Task performance. |
| **Energy ratio** R = Ē(h^L) / Ē(x) | Final-layer mean edge cosine distance relative to the text. Collapse means R → 0. |
| **MAD** | Mean cosine distance over 20k random node pairs at the final layer. Global collapse means MAD → 0. |
| **Effective rank** | exp(entropy of normalized singular values) of the final states. Rank collapse means it falls toward 1. |
| **Expert entropy** (MoE only) | Normalized entropy of mean router usage per layer, averaged over layers. Expert collapse means it falls toward 0. |

## Hypotheses (confirmatory)

- **H15 (collapse exists).** The baseline's R, MAD and effective rank fall as L goes from 2 to 16, and test accuracy at L=16 is ≥ 5 points below L=2, for at least the non-MoE encoder.
- **H16 (MoE residuals resist collapse).** The MoE encoder adds a residual connection in every hidden layer (`MixtureSageLayer`, `residual=True`), so its L=16 accuracy drop is smaller than the non-MoE drop.
- **H17 (TAEF prevents collapse).** At L=16, TAEF raises R and effective rank above the baseline and recovers ≥ 3 points of test accuracy, for both encoders.
- **H18 (edge-wise text anchoring matters).** At L=16, TAEF beats GEF by ≥ 1.5 points of accuracy, averaged over the two encoders. *Refuted if* TAEF ≤ GEF.
- **H19 (no cost when shallow).** At L=2, TAEF changes accuracy by at most 1 point.

**Decision basis:** mean over 3 splits. A difference within one std is "no detectable difference".

## Threats to validity

- One dataset, which is homophilous.
- Hidden width 64 rather than 768.
- No pretraining. Collapse in the pretrained model may differ.
- BatchNorm partly counteracts collapse. A `normalize='none'` run would be exploratory.
- Robustness to attacks is out of scope here.

## Amendment (2026-10-21, after the split-0 smoke test, before the full run)

**Smoke test** (split 0, baseline and TAEF only, depths 2 and 16):

| Encoder | Depth | Baseline | TAEF |
|---|---|---|---|
| SAGE | 2 | 81.0 | 69.8 |
| SAGE | 16 | 68.9 | 78.0 |
| MoE | 2 | 82.5 | — |
| MoE | 16 | 78.8 | 78.0 |

Because split-0 test results were seen, split 0's contribution to H17 and H19 is not blind.

**Change:** added `TAEF-rho0.25[EXPL]` as an exploratory variant. The confirmatory settings (ρ = 0.5, λ = 1) are unchanged.

## Amendment 2 (2026-10-21, after the single-seed full run, before any multi-seed result)

**Finding: training is nondeterministic across CPU thread counts.** SAGE, depth 16, split 0, seed 0 gives test 76.4% with 4 threads and 68.9% with 8. Scatter reduction order changes, and deep training amplifies the difference. The single-seed full run (`results/split{0,1,2}.json`) is therefore noise-dominated: per-cell std is up to ±5.8. **The split-0 smoke test's "9-point TAEF recovery" was noise.**

**Change:**

- Rerun at depths {2, 16} with **5 seeds per split** (15 runs per cell) and **threads fixed at 4**. Seed formula: 1000·seed + split.
- H15–H19 are decided on the mean over 15 runs, with paired differences matched by (split, seed).
- The single-seed run is kept as exploratory.
- Methods and ρ/λ are unchanged.
