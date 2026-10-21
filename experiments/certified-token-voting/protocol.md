# Protocol: Certified Token Voting (CTV), a new robustness head for STEM-GNN

Written 2026-10-20, before any CTV result existed. Novelty has **not** been checked yet; run the `prior-art-check` skill before any claim.

## Idea

STEM-GNN represents each node as discrete VQ tokens. If the tokens are computed from the node's **own text only** (the encoder applied with no edges), a structure attack cannot change any token. It can only rewire which tokens a node hears from.

**CTV head:**

- **Votes.** v_j = softmax(decoder(tokens_j) / T), a point on the probability simplex, computed with no edges and frozen at inference. Variants mix in an LLM prior: v_j = (1 − β)·v_j + β·p_LLM(j).
- **Score.** score_c(i) = α·v_ic + Σ_{j ∈ N(i)} v_jc, with prediction argmax_c score_c(i). This is a one-hop sum, not a mean.
- **Certificate.** An insertion or deletion of an edge incident to i changes each score_c(i) by at most v_jc ≤ 1, so it changes the margin by at most 1. With m_i = score_y(i) − max_{c≠y} score_c(i), node i is **certified against any k ≤ ⌈m_i⌉ − 1 incident edge edits**. Margins are computed on the predicted class at test time, so no labels are needed.
- **Global budgets.** The certificate is local (edits incident to i). For global PRBCD, report empirical accuracy.
- **No edge features in the votes.** The certificate holds as long as v_j does not depend on edges, so the token tower must use no message passing.

## Tower (STEM-lite style, no edges)

- MLP on x (frozen MiniLM text embedding), 4 heads × 32 dims.
- Cosine VQ with 32 codes per head, initialized by k-means and then frozen.
- Per-head linear decoders.
- Trained with cross-entropy on train labels, with straight-through gradients through the VQ.
- **Variant:** add 210 LLM pseudo-labels (the same MiniLM prior as `llm-robustness`), and add the Qwen2.5-1.5B prior once it is computed.

## Tuning (validation only, then frozen)

- α ∈ {0.5, 1, 2, 4}, T ∈ {0.5, 1, 2}, β ∈ {0, 0.5}.
- Choose by clean validation accuracy, with ties broken by validation certified accuracy at k=1.

## Hypotheses (confirmatory)

- **H10 (clean accuracy).** CTV clean test accuracy ≥ 78%, within about 3 points of SAGE (81.3). *Refuted if* < 76%.
- **H11 (adaptive robustness).** CTV accuracy under adaptive PRBCD@10% and @20% exceeds SAGE and SAGE+DropEdgeCons (61.0 @20% in the earlier evidence) by ≥ 5 points at 20%. PRBCD differentiates through `edge_weight`, which scales each vote, so the attack is fully adaptive. There is no gradient masking: the score is linear in the edge weights.
- **H12 (certificate is non-trivial).** Certified test accuracy at k=1 is ≥ 60%, and at k=3 is ≥ 40%. Report the full curve of certified accuracy versus k.
- **H13 (degree).** Certified radius grows with degree, because the sum aggregation gives high-degree nodes larger margins. Report the radius for each degree bucket.
- **H14 (sanity).** Every attacked prediction respects the certificate: no node certified at k is flipped by an attack that makes ≤ k edits incident to it. This is checked empirically on PRBCD graphs. Any violation means a bug.

## Threats

The same as in `llm-robustness`: random insertion, camouflage (fixed version), and adaptive PRBCD at 10% and 20%, evasion only, on OFA Cora splits 0/1/2.

## Threats to validity

- The tower is not the pretrained STEM encoder. In the real model, the edge-free pass of the STEM encoder with `--use_llm_router` would produce the tokens.
- One-hop only. A two-hop version needs a different certificate.
- Feature or text attacks are out of scope: they change v_j directly.

## Amendment (2026-10-20, after the split-0 clean smoke test, before any attack result)

- **Missing comparison: MLP is trivially certified.** An MLP that ignores structure is "certified" at every k, with 72.5% accuracy. A structural certificate is useful only where certified accuracy at k exceeds about 72.5%, so H12 is re-stated against that baseline.
- **Split-0 smoke test** (tuned on validation):
  - clean 79.4%;
  - certified accuracy 31.4% at k=1 and 5.9% at k=2.

  H12 is failing so far.
- **Exploratory frontier added:** a sweep over α ∈ {0.5, 1, 2, 4, 8} and T ∈ {0.1, 0.25, 0.5, 1, 2}, reporting clean accuracy and certified accuracy at k = 1, 2, 3. Labeled EXPLORATORY.
