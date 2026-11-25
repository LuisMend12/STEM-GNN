# Protocol: LLM prior anchoring of STEM-GNN's prediction head (Qwen2.5-1.5B)

Written 2026-10-29, before any result. Codex will review the design before any full run.

**Where the LLM sits.** At the *prediction head* (see the user's architecture diagram). Qwen never weights, drops or adds edges; the graph structure is untouched (user constraint). Qwen supplies a structure-free class distribution per node, and finetuning adds an anchoring term pulling STEM's predictions toward it on unlabeled nodes.

**Rationale.**

- **Robustness:** an edge attacker can change STEM's message passing but not Qwen's belief, which is computed from the node's own text alone.
- **Generalization:** extra supervision from an independent signal helps most when labels are scarce.

## LLM signal

- **Model:** Qwen2.5-1.5B-Instruct, float32, CPU (`STEM-GNN/scripts/llm_robustness/qwen_prior.py`).
- **Input:** the paper's title and abstract (first 1,200 characters) plus the 7 category names.
- **Output:** a softmax over the next-token logits of letters A–G. One forward pass per node, no generation, no labels.
- **Smoke check:** 65.6% zero-shot accuracy on the first 32 nodes.

## Model and pipeline

Identical to `../code-robustness` (Amendment 1):

1. Real `Encoder` + real `VectorQuantize` (EMA codebook) pretrained per (split, seed) with the X2 pilot objective.
2. Finetuning with a frozen codebook, using deterministic regularizer forwards and a per-epoch CE seed.
3. Width 128.

## Methods (λ = 1, fixed; no tuning on test)

| ID | Anchor |
|---|---|
| **B0** | none |
| **A-qwen** | KL(p_qwen ‖ softmax(logits)) averaged over all nodes outside the training mask |
| **A-minilm** (control) | the same anchor with the MiniLM zero-shot prior from `../llm-robustness` (T chosen on validation NLL). Tests whether LLM quality matters. |

Using unlabeled test nodes' *text* is transductive but label-free, and is declared.

## Label settings

- **Standard:** the OFA split's training mask.
- **Low-label:** 3 nodes per class, sampled from the training mask with seed (split, seed). The same validation and test masks are used.

## Threats (at 10% and 20% of undirected edges)

- **Own-model attacks:** STE-PRBCD and soft-backward/hard-forward BPDA.
- **Transfer graphs:** from a SAGE surrogate and from the paired B0 model.
- **Report:** worst-node (intersection including the clean graph and lower budgets), worst-graph, and clean accuracy.
- **Graph validity:** asserts on every candidate.

## Hypotheses

n = 15 cells (splits {0, 1, 2} × seeds {0..4}); paired t 95% CI, conditional on this graph.

| ID | Role | Comparison |
|---|---|---|
| **H39** | **PRIMARY**, robustness | A-qwen − B0 on worst-node@20%, standard labels |
| **H40** | generalization | A-qwen − B0 clean accuracy, low-label |
| **H41** | LLM quality matters | A-qwen − A-minilm on worst-node@20%, standard |
| **H42** | generalization | A-qwen − B0 clean accuracy, standard labels |

| ID | Support | Evidence against | Otherwise |
|---|---|---|---|
| **H39** | ≥ +3, CI excludes 0, and clean change ≥ −1 | upper CI < +1.5 | inconclusive |
| **H40** | ≥ +3, CI excludes 0 | upper CI < +1 | inconclusive |
| **H41** | ≥ +1.5, CI excludes 0 | — | — |
| **H42** | descriptive only | — | — |

Secondary hypotheses use Holm adjustment.

**Ceiling check (descriptive).** Report Qwen-alone and MiniLM-alone zero-shot test accuracy. If the anchored model's worst-node accuracy merely approaches the prior-alone accuracy, the gain is "falling back to the LLM", not a GNN improvement. Say which.

## Amendment 1 (2026-10-29, from the Codex design review, BEFORE any anchor result)

**Scope.** The model is a **SAGE–VQ component pilot**: the real `Encoder` without MoE, the real `VectorQuantize`, and a linear decoder with no explicit Lipschitz/Jacobian penalty. It is not the full MoE → router → VQ → Lipschitz-head architecture. Any claim about the user's architecture requires a faithful MoE follow-up, which will only be run if this pilot holds up.

**Qwen extraction fixed (v2).** v1 ended the prompt with a trailing space and scored bare letters. A tokenizer check showed that was context-inconsistent: Qwen tokenizes " A" as a single space-prefixed token. v2 ends with "The answer is" and scores " A"–" G", which are asserted single-token and context-consistent. On 16 nodes the letter-mass diagnostic is 0.90 and accuracy is 62.5%.

- v2 caches raw letter logits plus metadata (model, dtype, label order, permutation, prompt hash, text hash), and refuses to resume a cache with mismatched metadata.
- 21.8% of texts are truncated at 1,200 characters.
- Texts are confirmed to be title + abstract.
- Possible pretraining exposure of Qwen to Cora cannot be ruled out, and is declared.

**Position-bias audit.** On a fixed, label-independent sample of 70 nodes (every 38th node id), all 7 cyclic option rotations are scored and mapped back to class order. Reported: argmax disagreement across rotations, class frequencies, and the effect of rotation-averaging on validation NLL and accuracy.

- **Decision rule, fixed now:** if rotation-averaging improves validation accuracy by ≥ 2 points on the audit sample, the full prior will use a declared 3-rotation ensemble ({0, 2, 4}). Otherwise the identity order is used.

**Calibration.** Both teachers get the same validation-only rule: p = softmax(scores/T), with T ∈ {0.5, 1, 2, 4} (Qwen, on raw letter logits) or {0.02, 0.05, 0.1} (MiniLM, on cosines), chosen by validation NLL. H41 is renamed **"Qwen teacher vs MiniLM teacher"**: it is not a test of LLM quality as a cause.

**Mandatory controls (all on the same splits and seeds):**

| Control | Purpose |
|---|---|
| **Qwen-alone** | argmax of the calibrated prior |
| **MLP** and **MLP+A-qwen** | the same optimization, checkpoint rule and KL coefficient, on `x`, with no graph |
| **MIX** = (1−α)·p_B0 + α·p_Qwen | α ∈ {0, .25, .5, .75, 1}, selected on **clean validation accuracy** only. MIX is attacked directly, using log-mixture probabilities as logits. |

**Claims are staged:**

1. A-qwen > B0 shows that distillation helps this student.
2. A-qwen > MIX and A-qwen > MLP+A-qwen shows graph-specific added value.
3. Retained gains on Qwen-wrong nodes are evidence against simple teacher imitation.

**Diagnostics:**

- Agreement with Qwen on the clean and the worst graphs.
- Accuracy on Qwen-correct vs Qwen-wrong test nodes.
- Teacher errors corrected and inherited.
- Per-node correctness saved for the full candidate union.

**Attacks.**

- Own-model attacks are **ste, ste_r2 and soft** (Amendment 1 of code-robustness).
- **Shared candidate union:** every model in a cell, including MIX, is evaluated on all candidate graphs generated in that cell, plus the clean graph and lower budgets.
- Realized additions and deletions are saved.
- Results are reported as *empirical accuracy over a finite attack suite*.

**Anchor forward.** The KL term uses a separate gradient-enabled deterministic forward (dropout off, BatchNorm on running statistics). CE keeps the seeded training forward.

**Low-label setting.**

- 3 per class, with at least 3 eligible nodes per class asserted. Masks are saved, and train/val/test disjointness is asserted.
- The validation-label budget is unchanged; the setting is described as "21 training labels with the original validation budget".
- The SAGE attack surrogate is trained on the **same labels as the learner** in each setting.

**Inference.** Per-split paired effects are reported, and the 15-cell intervals are described as conditional run-level summaries. The explicit invocation uses `--splits 0 1 2`.

## Threats to validity

- Cora only.
- Width 128.
- Pilot pretraining.
- One λ.
- A small LLM (1.5B, about 66% zero-shot).
- CPU-scale attacks, which may be weak. The suite includes soft-BPDA and transfer to reduce that risk.
- **Text attacks are out of scope.** A text attacker could move Qwen's prior.
- Novelty is unverified. LLM-to-GNN distillation exists (e.g., GLEM-style, TAPE-style LLM signals); what is under test is its effect on STEM's structural robustness.
