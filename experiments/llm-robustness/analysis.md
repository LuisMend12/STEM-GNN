# Analysis: using the LLM to increase robustness (not to decode messages)

Cora, OFA splits 0/1/2, evasion. The LLM signal comes from MiniLM-384 text embeddings plus the LLM-written class descriptions. A zero-shot prior built from them is 60.3% accurate on test nodes; its 210 pseudo-labels are 76.8% correct.

**Status: complete.**

- **Supported:** H6, LLM trust resists camouflage where a raw-cosine gate fails.
- **Refuted:** H5, H7, H8, H9.
- **Caveat:** every PRBCD number here is an upper bound. On CTV, PRBCD overestimated robustness by about 21 points (`../certified-token-voting/analysis.md`).

## Results (mean ± std, accuracy %)

| Model | Clean | Camouflage 20% | Random-add 20% | PRBCD 10% | PRBCD 20% |
|---|---|---|---|---|---|
| SAGE | 81.3±0.6 | 77.3±0.5 | 79.9 | 64.3±0.3 | 55.1±0.9 |
| SAGE + TextGate (raw cosine, non-LLM baseline) | 79.1±0.4 | **74.1±1.3** | 78.9 | 66.0±0.6 | 60.0±0.4 |
| SAGE + L1 LLM trust | **81.2±0.1** | **77.3±0.3** | 80.2 | 65.2±0.5 | 57.4±0.2 |
| SAGE + L1 sharp (T=0.02) [EXPL] | 80.4 | 76.8 | 79.5 | 65.2 | 58.2 |
| SAGE + L2 LLM pseudo-labels | 79.1 | 75.7 | 78.0 | 63.4 | 54.2 |
| SAGE + L3 (L1 + L2) | 79.1 | 75.4 | 77.7 | 63.8 | 55.6 |
| STEM-lite (no commitment loss, repo-faithful) | 80.9±0.7 | 76.8±1.2 | 79.0 | 67.0±1.0 | 58.4±1.6 |
| STEM-lite + L1 | 80.8±2.1 | 77.0 | 79.6 | 67.5±2.6 | 60.0±2.3 |
| STEM-lite + L1 sharp [EXPL] | 80.7 | 76.6 | 80.2 | 67.3 | 60.6 |
| STEM-lite + L2 | 79.2±2.9 | 75.1 | 77.5 | 65.7 | 57.5 |
| STEM-lite + L3 | 80.3 | 76.6 | 79.1 | 67.8 | 60.7 |
| STEM-lite + L4 (L3 + DropEdgeCons) | 79.3 | 76.0 | 78.4 | 67.2 | 60.4 |
| STEM-lite + L4 sharp [EXPL] | 79.4 | 76.0 | 78.7 | 67.7 | 60.9±0.2 |

Every camouflage attack spent its full budget: 527 unique edges at 10% and 1055 at 20%.

## Hypothesis outcomes

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H5.** L1 improves PRBCD@10% by ≥ 3 with a clean drop ≤ 1.5 | **Refuted** (gain < 1) | Paired gains: SAGE +1.7 / +0.7 / +0.4 (mean +0.9); STEM-lite +1.3 / +1.8 / −1.7 (mean +0.5). Clean change −0.1 on both. |
| **H6.** L1 beats raw-cosine TextGate under camouflage@20% by ≥ 2 | **Supported** | Paired +3.7 / +2.1 / +3.8 (mean +3.2). TextGate is *below* SAGE under camouflage (74.1 vs 77.3). L1 matches SAGE. |
| **H7.** L2 improves PRBCD@10% by ≥ 2 without a clean loss | **Refuted** | SAGE −0.9 with clean −2.1. STEM-lite −1.3 with clean −1.7. |
| **H8.** L3 ≥ max(L1, L2) + 1 at PRBCD@10% | **Refuted** | SAGE: 63.8 vs L1's 65.2. STEM-lite: 67.8 vs L1's 67.5 (+0.3). |
| **H9.** L4 is the best STEM-lite variant at PRBCD@20% | **Refuted** | 60.4 vs 60.7 for L3. All LLM-trust variants lie within 60.0–60.9, inside noise. |

## What we learned

1. **The one real contribution is resistance to camouflage edges, which a raw-cosine gate lacks.**
   - A raw-cosine gate (GNNGuard-style) is *exploited* by edges that join wrong-label nodes with similar text. Those edges have *higher* raw cosine (0.552) than true edges (0.505), per the pre-run diagnostic in `protocol.md`.
   - Weighting edges by LLM class-agreement keeps clean accuracy and is not fooled by camouflage edges.
2. **LLM signals barely help against adaptive gradient attacks.** On both backbones, L1 adds at most about 1–2 points under PRBCD, within noise. PRBCD does not need semantically plausible edges, so a semantic trust weight has little to stop.
3. **Noisy LLM pseudo-labels hurt.** At 77% correct, the zero-shot pseudo-labels add noise, not useful structure-free supervision. A stronger LLM prior might change this; Qwen2.5-1.5B scored 75% zero-shot on a 16-node sample, versus about 60% for MiniLM. That remains untested at scale.
4. **Stacking does not add up.** L3 and L4 do not beat their best component.

## Threats to validity

- The "LLM" here is a 22M-parameter sentence encoder plus LLM-written class descriptions. No generative LLM was queried per node; a Qwen prior needs the GPU environment.
- One dataset.
- CPU-scale PRBCD, an optimistic attack. Under a strong attack, the modest PRBCD gains may shrink further.
- STEM-lite is a proxy (no pretraining, no MoE).
- Text attacks are out of scope. The defense assumes node text is not rewritten.
