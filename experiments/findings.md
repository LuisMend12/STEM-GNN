# Findings: increasing STEM-GNN's structural robustness (as of 2026-10-21)

**Setup.** CPU-scale evidence on OFA Cora (frozen MiniLM-384 text embeddings), 3 splits, evasion attacks, plus a STEM-lite proxy of STEM-GNN's finetune path. Pretrained STEM checkpoints are not available here.

**Experiments:**

- `stem-vq-robustness/`: the VQ path, consistency, code margin and PropSens.
- `llm-robustness/`: LLM-derived edge trust and pseudo-labels.
- `certified-token-voting/`: a new certified head.

Two Codex reviews (design and analysis) are in `reviews/`.

## Current understanding

1. **STEM-GNN's eval-mode VQ masks gradients.** A naive white-box attack on STEM-lite in eval mode sees 81.6% at a 10% budget, while a straight-through (STE) attack sees 69.6%. The mechanism is in `model/vq.py`: STE is applied only when `self.training`. The ~12-point magnitude is measured on the proxy only. *Any adversarial evaluation of STEM-GNN must use an attack-only STE/BPDA wrapper or a transfer attack.*
2. **No tested addition beats the baseline by a reliable margin under adaptive attack.**
   - DropEdge consistency: +6 on SAGE, +1.1 on STEM-lite.
   - LLM edge trust: +0.5 to +0.9.
   - Code margin: −1.6.
   - PropSens: up to +1.2.
   - LLM pseudo-labels: −1.
3. **LLM signals help against semantically camouflaged edges, not gradient attacks.** LLM class-agreement trust keeps clean accuracy and beats a raw-cosine gate by 3.2 points under camouflage. The raw-cosine gate is itself exploited by those edges.
4. **Certified Token Voting gives a sound, free certificate (0 violations) but is not robust under a strong attack.** PRBCD reported 68.0 at 20%; a realized graph-constructing greedy attack gives 46.6. Sharper votes (α=4, exploratory) reach 58.7 under the realized attack, at 75.1 clean.

## Patterns

- **PRBCD is an optimistic attack** and was off by about 21 points on a linear voting head. Every PRBCD-only gain in this repo is an upper bound.
- **The text-only MLP (72.5%, structure-free) is the bar a structural defense must clear under heavy attack.** Above about a 10% budget, every graph model in this study falls below it.
- **What helped on SAGE often vanished on the VQ path.** The frozen codebook already provides some invariance.

## Lessons and constraints

- **An attack number counts only if the attacked graph was built and evaluated,** with the edit count asserted to be within budget. Per-node cost accounting is an estimate.
- **Amendments written after seeing outcomes are exploratory,** even if they predate *other* results.
- **Post-ReLU relative sensitivity penalties collapse at initialization.** Apply them to pre-activation or output states.
- **bf16 LLM inference on this CPU is unusable** (0.001 nodes/s). Use fp32, or the GPU.
- **Background jobs can be suspended for hours** (machine sleep). Allow for this in run-time estimates.

## Open questions and ranked next tests

1. **Strong-attack parity:** PRBCD with restarts and more epochs, or another stronger attack, on SAGE and STEM-lite. This is needed for any fair cross-model claim.
2. **Commitment-loss ablation** (coefficient {0, 0.05, 0.25}, reseeded attacks). An unmatched cross-run comparison showed +4.1 at 20% with commitment. The repo discards this loss in node finetuning, so this is the cheapest candidate change.
3. **Text-confidence-conditioned cap on graph corrections** (Codex's proposal). Confident text predictions get a small structural budget. Target: ≥ +3 worst-over-attacks at 10%, with ≤ 1 point clean loss.
4. **Qwen prior** for CTV votes and LLM trust. It needs the GPU environment.
5. **Prior-art check** on CTV and the text-confidence cap before any novelty claim.
