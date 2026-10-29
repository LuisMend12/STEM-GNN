# Analysis: STEM-GNN's VQ finetune path under adaptive attack (Cora, 3 splits)

**Status: complete.** H1 is supported. H2 and H3 are not supported. H4 was amended, and its repairs train stably but add at most a small gain.

## Results (mean ± std over OFA splits 0/1/2, accuracy %)

| Model | Clean | Attack | 5% | 10% | 20% | Code change @10% |
|---|---|---|---|---|---|---|
| SAGE | 81.3±0.6 | PRBCD | 70.7 | 64.2 | 55.0 | — |
| SAGE + PropSens (c=0.1, post-ReLU) | 81.0 | PRBCD | 70.0 | 64.2 | 54.8 | — |
| SAGE + PropSens (c=1, pre-activation) | 81.2 | PRBCD | 71.4 | 64.8 | 56.2 | — |
| STEM-lite | 82.3±1.6 | **hard-VQ PRBCD (eval mode)** | 81.8 | **81.6** | 81.1 | — |
| STEM-lite | | STE-PRBCD | 74.2 | **69.6** | 62.6 | 16.9% |
| STEM-lite | | transfer from SAGE | — | 71.4 | 64.6 | — |
| STEM-lite + DropEdgeCons | 82.8±1.9 | STE-PRBCD | 75.2 | 70.7±2.9 | 64.1 | 15.6% |
| STEM-lite + CodeMargin | 81.4±2.2 | STE-PRBCD | 73.6 | 68.0 | 60.7 | 16.5% |
| STEM-lite + Both | 81.9±2.0 | STE-PRBCD | 73.9 | 69.2 | 62.3 | 15.4% |

Paired per-split differences in STE-PRBCD@10% against STEM-lite:

| Variant | Split 0 | Split 1 | Split 2 | Mean |
|---|---|---|---|---|
| DropEdgeCons | +3.3 | −0.8 | +0.8 | +1.1 |
| CodeMargin | +0.3 | −2.1 | −3.0 | −1.6 |
| Both | +2.6 | −2.9 | −1.0 | −0.4 |

## Hypothesis outcomes

**H1 (gradient masking): supported.** Hard-VQ PRBCD@10% minus min(STE, transfer)@10% is 12.9, 11.9 and 11.1 points on the three splits, against a bar of ≥ 5. Attacked in eval mode, STEM-lite loses only 0.7 points at a 10% budget. That isn't robustness. `model/vq.py` applies the straight-through estimator only when `self.training`, so a white-box attacker sees zero gradient through the quantizer and is limited to PRBCD's forward candidate selection.

> **Implication for the repo:** any adversarial evaluation of STEM-GNN must use an STE/BPDA surrogate, or a transfer attack. Do not switch the whole model to training mode for this, because that also changes BatchNorm and routing. Use an attack-only wrapper that applies straight-through to the VQ alone.
>
> **Scope (per the Codex analysis review):** the ~12-point gap is measured on this STEM-lite proxy and harness. The hard mode here is `q.detach() + 0*zh`, a connected graph with zero gradient. The real eval VQ may instead raise a disconnected-gradient error, and full STEM was not evaluated. The mechanism is grounded in `model/vq.py`. The magnitude for full STEM is unverified.

**H2 (DropEdge consistency transfers to STEM): not supported.** The mean paired gain is +1.1 ± 2.1, below the +3 bar. Split 0 alone crosses it. The transfer-attack gain is larger (+3.2 at 20%) but highly variable (std 6.8), so it is exploratory only.

> Consistency was the best addition on SAGE (+6 at 20%) but adds little on the VQ path.

**H3 (code margin): not supported.** It does slightly worse than STEM-lite (−1.6 at 10%), costs 0.9 clean, and barely changes the code-change rate (16.5% vs 16.9%). As Codex warned, the raw top-two cosine gap is not a distance to the cell boundary. A normalized boundary-distance margin remains untested.

**H4 (PropSens collapse): original endpoint diagnostic inconclusive. The mechanism diagnosis is EXPLORATORY,** because it was written after the collapse was observed, and the per-epoch trace was an interactive diagnostic not archived in `results/`.

- **Mechanism:** with coef 1 on post-ReLU states, the per-node relative penalty explodes at initialization (penalty around 400 at epoch 6, while ‖h‖ ≈ 0.15). Every ReLU unit dies within about 6 epochs, and early stopping restores the epoch-0 weights, which score about 32% clean.
- **Repairs:** both train stably. c=0.1 shows no detectable gain. The pre-activation variant gains about +1.2 at 20% (56.2 vs 55.0), roughly one std.
- **Relevance to the repo:** the repo's penalty acts on the final encoder output, which has no activation, so it should not hit this collapse. Its robustness benefit is likely small.

## Exploratory cross-run observation (not pre-registered)

The `llm-robustness` run trains STEM-lite **without** the commitment loss (repo-faithful) on the same splits, seeds and attack. It reaches 80.9 clean and 58.4 at STE-PRBCD@20%. This run, with 0.25·commitment summed over dimensions, reaches 82.3 clean and 62.6.

**Hypothesis to test:** a commitment term during finetuning pulls encoder states toward their codes and adds about 4 points of robustness. Codex verified the two runs share training code, seeds and PRBCD settings. They are **not** matched on attack random streams: this run also attacks at 5% before 10% and 20%, with no per-budget reseeding. So the +4.1 is an observation, not a causal estimate. The repo currently discards the VQ loss in node finetuning (`ft_model.py:142`, `task/node.py:74`). That makes this the cheapest candidate change found so far. It needs a direct, pre-registered ablation (commitment coefficient ∈ {0, 0.05, 0.25}) before it is acted on.

## Threats to validity

- **STEM-lite is not pretrained STEM-GNN:** it has no MoE router, no BatchNorm, no edge-attribute messages, and a 128-d per-head decoder.
- **One dataset.**
- **CPU-scale PRBCD,** which can overestimate robustness. On CTV it was off by 21 points (`../certified-token-voting/analysis.md`), so treat every STE-PRBCD number as an upper bound.
- **Evasion only.**
