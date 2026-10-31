# Protocol: Integrating an LLM across the full STEM-GNN architecture (Qwen2.5-1.5B)

Written 2026-10-31, before any result from these arms. DRAFT for Codex design review; nothing here has been run.

This supersedes the SAGE–VQ pilot in `../llm-anchor/protocol.md` as the main test, because the question is now about the whole architecture: input → MoE encoder → router → experts → VQ → Lipschitz head.

**Constraint from the user:** text or LLM output must never weight, select, add or drop edges.

## Signals already extracted (cluster, A100, float32)

- **Class prior:** letter-logit zero-shot distribution per node. Test-free description: 53.5% accuracy over all 2,708 nodes, unreviewed. The earlier 62–66% figures came from only 16–32 nodes.
- **Node embedding:** mean of Qwen's last hidden layer over the title and abstract, 1,536 dimensions, no prompt, no labels.
- **Position-bias audit:** 70 nodes × 7 rotations, decision rule as in `../llm-anchor` Amendment 1.

## Baseline S0: the real architecture

- Real `Encoder` with `moe=True`, `moe_layers='all'`, `K=3`, SAGE backbone, Gumbel-softmax router in training and softmax at evaluation.
- Real `VectorQuantize`: 4 heads × 128 codes, cosine, EMA codebook (the X2 C-ema fix, which is the only variant with a live codebook), frozen at finetune.
- Linear head with the repo's `decoder_jacobian_penalty` and `encoder_lipschitz_penalty` active, using the repo's Cora values where a config exists. If none exists, the coefficients are fixed on S0 clean validation only, before any arm is run, and then shared by all arms.
- Input: the repo's existing MiniLM node features.
- Known defects stay in place in S0 unless listed here. Per-expert init is not changed (X1 showed it harmful at depth 16).

## Arms: one LLM placement at each stage

| Arm | Stage | Change from S0 |
|---|---|---|
| **S-F** | Input | Qwen embedding replaces MiniLM as `x`, for pretraining and finetuning. |
| **S-R0** | Router (control) | The repo's `use_llm_router`: router reads the original MiniLM `x`, not `z`. Tests a structure-free router without Qwen. |
| **S-R** | Router | Router reads the Qwen embedding at every MoE layer. `x` stays MiniLM. |
| **S-V** | VQ | During pretraining, a linear map from the assigned codes predicts the Qwen embedding (cosine loss, weight 1). Discarded afterwards. Shapes the codes toward LLM semantics. |
| **S-A** | Head | KL(calibrated Qwen prior ‖ model) on non-train nodes, λ = 1, finetune only. |
| **S-ALL** | All | S-R + S-V + S-A, with MiniLM input. |
| **S-ALL+F** | All | S-ALL with S-F's input. |

**Why S-R is the main mechanism.** Today the router reads `z`, which already contains message-passed neighbours from the second layer on, so an edge attacker can steer which expert handles a node. A router fed only by the node's own text can't be steered by edges. It closes one attack path; the experts themselves still aggregate over perturbed edges.

**Router init.** The repo's `--router_init` stays at `random` for all arms.

## Attack harness requirement (must be built and self-checked first)

`code_robustness/run.py` re-expresses only the SAGE path with edge weights. A new MoE path is needed:

1. `MixtureSageLayer` re-expressed with edge weights: weighted scatter mean over `x[col]` by `row`, matching the real layer's direction.
2. **Self-checks, on real weights:** with all weights 1 it equals the real `Encoder` at evaluation; weight 0 equals removing that edge.
3. The router is run at evaluation (softmax) during attacks. For S-R arms the router input is a constant with respect to edges. For S0 the attack differentiates through the router, which must be asserted by a nonzero gradient through router weights.

## Threats

- PRBCD at 10% and 20% of undirected edges: `ste`, `ste_r2`, `soft`, plus SAGE surrogate transfer.
- **Shared candidate union:** every arm in a cell is evaluated on every candidate graph, plus clean and lower budgets. Validity asserts on each realized graph (symmetric, no duplicates, no self-loops, edits ≤ budget).
- **Adaptive check:** S-R's attack must be run against the S-R model itself, since its router input is fixed.
- Reported as empirical accuracy over a finite suite; not a certificate.

## Label settings

Standard OFA split, and low-label (3 per class, as in `../llm-anchor` Amendment 1).

## Hypotheses

n = 15 cells (splits {0, 1, 2} × seeds {0..4}), paired t 95% CI, conditional on this graph.

| ID | Role | Comparison | Support | Evidence against |
|---|---|---|---|---|
| **H50** | PRIMARY | S-ALL − S0, worst-node@20%, standard labels | ≥ +3, CI excludes 0, clean change ≥ −1 | upper CI < +1.5 |
| **H51** | Router | S-R − S0, worst-node@20% | ≥ +2, CI excludes 0 | upper CI < +1 |
| **H52** | Qwen vs MiniLM at the router | S-R − S-R0, worst-node@20% | ≥ +1.5, CI excludes 0 | — |
| **H53** | Input | S-F − S0, worst-node@20% and clean | ≥ +2 | upper CI < +1 |
| **H54** | VQ | S-V − S0, worst-node@20% | ≥ +2 | upper CI < +1 |
| **H55** | Head | S-A − S0, worst-node@20% | ≥ +2 | upper CI < +1 |
| **H56** | Generalization | S-ALL − S0, clean accuracy, low-label | ≥ +3, CI excludes 0 | upper CI < +1 |
| **H57** | Full stack | S-ALL+F − S-ALL, worst-node@20% | descriptive | — |

Secondary hypotheses use Holm adjustment.

## Controls and diagnostics

- **Controls:** Qwen-alone; MLP on Qwen features; MLP+anchor; MIX (S0 with Qwen prior, α selected on clean validation, attacked directly).
- **Claim staging:** S-ALL > S0 alone shows only that the LLM helps. S-ALL > MIX and > MLP-on-Qwen-features is needed to claim added value from integration over simply trusting the LLM.
- **Router:** routing shift between clean and attacked graphs (mean total variation), expert usage and perplexity, for S0, S-R0 and S-R. Mechanism test: S-R's routing shift must be 0, and any robustness gain should come with lower expert-output shift.
- **Teacher:** accuracy on Qwen-correct vs Qwen-wrong nodes; errors corrected and inherited. Qwen is weak here (53.5%), so inherited errors are a main risk for S-A.
- **Codebook:** perplexity and code change rate, since MoE may interact with the EMA codebook differently from the SAGE pilot.

## Compute

Attack and training jobs run on cluster CPU nodes as a Slurm array, with one job per cell, unique tags, and fixed thread counts. The harness has no GPU path. Earlier work showed deep MoE is not deterministic even with fixed threads, so identical reruns are not expected to match exactly.

## Threats to validity

- Cora only, and pretraining is Cora-only rather than multi-domain, so domain-generalization claims can't be made.
- One LLM and one λ.
- Text attacks are out of scope, and a text attacker could move all Qwen-derived signals. The fixed router trades edge-attack exposure for text-attack exposure.
- Transductive use of unlabeled nodes' text is declared.
- 21.8% of texts are truncated at 1,200 characters.
- Possible Qwen pretraining exposure to Cora can't be ruled out.
- Novelty is unverified. A prior-art check on structure-free expert routing is needed before calling S-R novel.

## Amendment 1 (2026-10-31, from the Codex design review, BEFORE any result from these arms)

The review is in `../reviews/codex_llm_stem_design_review_2026-10-31.md`. It found the draft not ready to launch. The study is now staged, and the arm table and hypotheses above are replaced as follows.

**Stage 1: the router. Stage 2 (R × V × A, input swap, anchor) only runs after Stage 1 is analyzed and reviewed.** S-V will be redefined as semantic distillation at `vq.project_in(z)` with a no-VQ control, because a detached code predictor can't shape the encoder or codebook.

**S0, fully specified.** "STEM components with the EMA repair and the X2 pretraining recipe", not the full STEM objective.

- MoE on all layers, depth 2, width 128, K = 3, router tau 1, BatchNorm, ReLU, dropout 0.15.
- VQ: 4 heads × 128 codes, cosine, EMA decay 0.8, commitment 0.25.
- Pretraining: 200 epochs, AdamW 1e-3, feature + topology reconstruction + VQ loss. The semantic loss and router regularizer are excluded, and that is declared.
- Finetune: Adam 0.01, weight decay 5e-4, 300 epochs, patience 50, frozen codebook, decoder on concatenated per-head codes.
- Depth 16 is not part of Stage 1.

**Penalties.** The repo has no nonzero Cora values (both coefficients default to 0), so nonzero values are study-selected. One shared coefficient for the decoder and encoder penalties is chosen from {0, 1e-5, 1e-4, 1e-3} on S0 clean validation only (seed 0, all splits), and ties within 0.5 points go to the larger value. It is frozen before any arm comparison. These are Frobenius penalties, not an end-to-end guarantee.

**Stage 1 arms** (only the router input differs; pretraining and finetuning use the same router input):

| Arm | Router reads | Separates |
|---|---|---|
| S0 | message-passed z | Baseline |
| S-R0 | MiniLM text | Structure-free routing without Qwen |
| S-R | Qwen embedding | The mechanism under test |
| S-U | Frozen uniform logits | Whether removing edge-sensitive routing is all that matters |
| S-RF | Fixed random features | Random node partitions as regularization |
| S-PQ | Node-permuted Qwen embedding | Same distribution, wrong node |

Gumbel noise stays on in all arms during training. S-R's router weights still learn; only its input is fixed.

**The user constraint, defined operationally.** Adjacency, edge existence, scalar aggregation coefficients and sampling probabilities must be independent of text. Node-conditioned expert transforms are allowed. Under a literal ban on any text-dependent change to how neighbours are combined, S-R and its controls would violate the constraint, so this is flagged to the user.

**Attack harness.**

- MoE re-expression uses a true weighted mean, aggregating `z[col]` into `row`, with zero output for isolated nodes.
- Self-checks: evaluation and training-mode equality with the real Encoder; an asymmetric toy graph with an isolated node; weight 0 equals a removed edge; uniform non-unit weights equal unit weights; code assignment matches real VQ; forward is hard in every backward mode; the layer-2 router has an edge gradient path in S0 and none in S-R.
- Suite: `ste`, `ste_r2`, `soft` (τ 0.1), `soft_t1` (τ 1.0), `ste_tanh` (margin objective), plus SAGE transfer. All models are evaluated on the shared union.
- The attacker uses test labels and indices, declared as a label-informed evaluation attacker.
- Discrete local refinement is not implemented, so own-attack results are upper bounds on robust accuracy.

**Stage 1 hypotheses** (n = 15 run-level cells; intervals are conditional on this graph, with per-split effects reported; 3 graph splits are not independent replicates):

| ID | Role | Comparison | Support | Evidence against |
|---|---|---|---|---|
| H60 | PRIMARY, Qwen-specific routing | S-R − S-R0, worst-node@20%, standard labels | ≥ +2, CI excludes 0, clean change ≥ −1 | upper CI < +1 |
| H61 | Structure-free routing | S-R − S0, worst-node@20% | ≥ +2, CI excludes 0, clean change ≥ −1 | upper CI < +1 |
| H62 | Beats trivial routers | S-R − max(S-U, S-RF, S-PQ) per cell, worst-node@20% | ≥ +1, CI excludes 0 | — |

H61 and H62 form the Holm family. Anything else is descriptive. A positive H61 with H60 and H62 null means the benefit isn't specific to Qwen. Routing shift in S-R is an implementation check, not evidence of cause.

**Leakage disclosures.** The 53.5% zero-shot figure and the rotation accuracies on the audit sample were printed on all scored nodes, so aggregate test performance has been seen, though neither is used by Stage 1. The Qwen prior isn't used in Stage 1. The feature loader checks the text hash, so node order mismatches fail.

**Not in Stage 1.** Low-label runs, MIX/MLP controls, the anchor and the learned-global router all wait for Stage 2. So Stage 1 can't show integration beats trusting the LLM directly.
