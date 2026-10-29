# Codex design review: STEM-VQ robustness probe (2026-10-20)

Codex reviewed the experiment design read-only, through the codex plugin. Its sandbox could not write files, so the review was returned inline. This file is a faithful condensation. The **Verified** and **Action** columns are Claude's.

| # | Sev | Finding | Verified | Action |
|---|---|---|---|---|
| 1 | HIGH | STEM-lite adds 0.25·commitment loss (summed over dimensions, so about 32× the repo's mean-MSE scale). The repo's node finetune discards the VQ loss. | Yes: `ft_model.py:142` keeps only `quantize, _, _, codes`, and `task/node.py:74` has no VQ term. | LLM run uses `COMMIT_W=0`. The STEM-VQ run carries this caveat (protocol amendment). |
| 2 | HIGH | The proxy omits the real encoder's ReLU(x_j + edge_attr) messages, BatchNorm, MoE experts and residuals. | Partly checked (`encoder.py` loop). | All claims are labeled STEM-lite. A replica using the real `Encoder`/`VectorQuantize` is the next step. |
| 3 | MED | "Hard-VQ PRBCD" means zero-gradient PRBCD with forward candidate selection, not exactly what the real eval VQ does (the real one may raise a disconnected-gradient error). | Consistent with `vq.py` (STE only when `self.training`). | Renamed in the H1 amendment. |
| 4 | HIGH | H1's "<2 points ⇒ refuted" rule is unsound. A small gap can mean robust codes or a weak surrogate. | Agreed. | The 2–5 point band is now inconclusive (amendment). |
| 5 | MED | H2/H3 rely on one surrogate. No worst-case-across-attacks metric. | Agreed. | Report the minimum over {STE, transfer} in the analysis. |
| 6 | HIGH | The H4 endpoint diagnostic can't test the dynamics (early stopping restores epoch 0). | Yes. A per-epoch trace confirmed the collapse by about epoch 6. | Amended. Per-epoch evidence goes in the analysis. |
| 7 | MED | The preact repair penalizes layer 1, not the repo's final encoder output. The repo also leaves Gumbel routing and BatchNorm in train mode during the penalty. | Yes (`encoder.py:491-523`). | Noted as a limitation. Test the final-output variant on the real encoder. |
| 8 | MED | Camouflage can underspend: existing edges and duplicate reciprocal pairs collapse when coalesced. | Yes. | Fixed in `adaptive_robustness/run.py`; edit counts are logged in the LLM run. |
| 9 | MED | Code-change rate alone doesn't establish H3's mechanism. | Agreed. | Add occupancy and harmful-transition metrics in follow-up work. |
| 10 | MED | Decoder geometry differs: the repo maps the concatenated codes densely (`decoder(codes)`), with width 768. | Yes. | Limitation. |
| 11 | LOW | Core attack plumbing (budgets, undirected handling, idx_attack, eval mode, transfer) looks sound. | — | Add assertions later. |
| 12 | LOW | No leakage. Transductive use of unlabeled nodes is legitimate and should be declared. | — | Declared. |

## Codex's ranked next tests (hypotheses, not results)

1. Prediction consistency on the VQ finetune path, with and without the commitment loss.
2. Teacher-anchored code consistency: perturbed soft codes matched to detached clean assignments.
3. Frozen-text edge trust before propagation. Being tested now as the LLM-trust variants.
4. Final-output PropSens with a stabilized denominator, a norm floor and coefficient warmup.
5. A normalized distance-to-code-boundary margin.
6. A decoder-aware code-transition margin.
7. Router consistency across graph views (needs MoE).
8. LLM-router invariance ablation (`--use_llm_router`). Prediction: router weights are exactly unchanged under structure-only attacks.
9. Code-aware adversarial training with short BPDA attacks.
10. Randomized smoothing around VQ, evaluated with EOT-BPDA.
