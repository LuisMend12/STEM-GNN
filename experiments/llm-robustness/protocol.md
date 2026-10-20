# Protocol: using the LLM to increase robustness, not to decode messages

Written 2026-10-20, before any result for these hypotheses existed.

**Constraint from the user.** The LLM must not be the predictor or the message decoder. It may only supply signals that make the GNN harder to attack. Every model below still predicts with a GNN (SAGE, or STEM-lite with its VQ and per-head decoders) reading messages over graph edges.

## LLM signals (all frozen, all label-free, all computed once)

- **Node text embedding.** `x_i` comes from all-MiniLM-L6-v2 and is the same as Cora's 384-d features. Re-encoding the raw text gives cosine 1.000.
- **Class text embedding.** `c_k` is MiniLM applied to the LLM-written class descriptions in `categories.csv`.
- **Zero-shot class prior.** p_i = softmax(cos(x_i, c_k) / T). T is chosen on the validation split from {0.02, 0.05, 0.1} by validation NLL. The argmax alone scores 59.9% on all nodes.

## Mechanisms

**L1. LLM edge trust (semantic-class agreement).** Each edge gets the weight t_ij = Σ_k √(p_ik · p_jk), the Bhattacharyya coefficient of the two endpoints' class priors, inside the weighted-mean aggregation.

- *Contrast with the raw-cosine TextGate:* a camouflaged edge that joins two wrong-label nodes with high raw cosine still gets a low weight when their class priors disagree.

**L2. LLM pseudo-label supervision.** For each class, take the n = 30 non-training nodes with the highest prior margin (p_top1 − p_top2) as extra training labels. They get a cross-entropy weight of 0.5, and their labels are the prior's argmax, not the true labels.

- *Rationale:* more structure-free supervision ties the decision boundary to text evidence.

**L3. L1 + L2 together.**

**L4. L3 + DropEdge consistency** on STEM-lite. This combines the best LLM variant with the best non-LLM variant from the earlier evidence.

## Models

- SAGE and STEM-lite, as baselines.
- {SAGE, STEM-lite} × {L1, L2, L3}.
- STEM-lite + L4.

STEM-lite is defined in `experiments/stem-vq-robustness/protocol.md`. It is attacked with an STE surrogate, and accuracy is measured with hard VQ.

## Threats

Evasion on OFA Cora splits 0/1/2. Budgets are 5%, 10% and 20% of the undirected edges.

- **PRBCD:** white-box and adaptive. Gradients flow through t_ij, because PRBCD perturbs `edge_weight` and t_ij multiplies it. New edges between disagreeing nodes therefore have small influence, which is real robustness and not masking: those edges do little.
- **Camouflage:** a PGD-Guard-style heuristic. For each attacked test node, add an edge to the wrong-label node with the highest raw text cosine.
- **Random insertion.**

## Hypotheses and decision rules (confirmatory)

| Hypothesis | Support if | Refuted if |
|---|---|---|
| **H5.** L1 improves PRBCD@10% over its backbone with a clean drop of at most 1.5 points | gain ≥ 3 points | gain < 1 point |
| **H6.** L1 beats raw-cosine TextGate under camouflage@20% (SAGE backbone, compared with the earlier partial results) | gain ≥ 2 points | — |
| **H7.** L2 improves PRBCD@10% | gain ≥ 2 points with clean accuracy not lower | — |
| **H8.** L3 is additive | L3 ≥ max(L1, L2) + 1 point at PRBCD@10% | — |
| **H9.** L4 is the best STEM-lite variant | L4 is best at PRBCD@20% | — |

Report mean±std over 3 splits. A difference within one std is "no detectable difference".

## Threats to validity

- The LLM signal here is a 22M-parameter sentence encoder plus LLM-written class descriptions. No generative LLM is queried per node, because Mistral-7B on this CPU is too slow.
- The zero-shot prior is only about 60% accurate, so L2's pseudo-labels are noisy.
- A single dataset, at CPU-scale PRBCD.
- The adversary knows the defense, but cannot change node text. Text attacks would be a separate experiment.

## Amendments (2026-10-20, before any LLM-run result; prompted by the Codex design review)

- **No commitment loss in STEM-lite** (`COMMIT_W = 0`). The repo's node finetune discards the VQ loss (`ft_model.py:142`, `task/node.py:74`).
- **Camouflage now spends its full budget on unique, non-existing, wrong-label pairs.** Each row logs the actual number of edits added. Earlier camouflage numbers in `partial_results_6models.json` used the old version, which could add fewer edges.
- **H6 is evaluated on matched graphs.** SAGE+TextGate (raw cosine, τ = the 10th-percentile clean-edge cosine) runs in this same experiment instead of being compared against the earlier partial results.
- **Exploratory variants added before any result:** `*-sharp[EXPL]`, which use T = 0.02 for the edge-trust prior only. A pre-run diagnostic on the clean graph and split 0 test targets motivated them:

  | Edge type | Trust at T=0.1 | Trust at T=0.02 | Raw cosine |
  |---|---|---|---|
  | Same-label | 0.932 | 0.767 | 0.505 |
  | Different-label | 0.884 | 0.540 | 0.422 |
  | Random | 0.756 | 0.274 | 0.190 |
  | Camouflage | 0.923 | 0.707 | 0.552 |

  Raw cosine ranks camouflage edges above true edges. These results are exploratory, not confirmatory.
- **Budgets reduced to 10% and 20%** (5% dropped) to fit CPU time. All decision rules use @10% or @20%.
