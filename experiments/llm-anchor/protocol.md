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

## Threats to validity

- Cora only.
- Width 128.
- Pilot pretraining.
- One λ.
- A small LLM (1.5B, about 66% zero-shot).
- CPU-scale attacks, which may be weak. The suite includes soft-BPDA and transfer to reduce that risk.
- **Text attacks are out of scope.** A text attacker could move Qwen's prior.
- Novelty is unverified. LLM-to-GNN distillation exists (e.g., GLEM-style, TAPE-style LLM signals); what is under test is its effect on STEM's structural robustness.
