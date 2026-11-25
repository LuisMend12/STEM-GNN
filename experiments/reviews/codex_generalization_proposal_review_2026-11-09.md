The proposal has a plausible robustness hypothesis, but its evidence does not yet justify ranking it as the best generalization frontier. The main problems are an incorrect TTA summary, an infeasible-as-written pilot, and a go/no-go rule that tests correlation rather than the proposed mechanism.

All proposal line references below refer to [proposal.md](experiments/generalization-frontiers/proposal.md).

1. **BLOCKER — proposal.md:48–57: the “free pilot” assumes artifacts that are not present and baseline outputs that are not saved.**

   The local `experiments/capped-correction/results/` contains 15 aggregate JSONs; I found no capped `_pernode.pt` or checkpoint files there. Per `STEM-GNN/scripts/capped/run.py:235,361`, the intended per-node payload is:
   
   ```text
   {"pernode": {arm: {"clean": correctness_vector,
                     attack_key: correctness_vector, ...}},
    "test_mask": ..., "y": ...}
   ```
   
   It saves correctness booleans for all nodes, not logits, margins, features, degrees, or homophily. The capped checkpoint saves `config`, `models`, `meta`, `sur`, and attack `graphs`; features and masks must be loaded separately. `baselines.py:119–143` computes per-node correctness transiently but returns aggregate rows, while its checkpoint stores trained baseline models, metadata, threshold, and newly attacked graphs. ENS models must be reconstructed from the capped models.

   **Replacement text:** “This is a no-training pilot conditional on retrieving the cluster checkpoints and capped per-node artifacts. Reload the matching dataset and splits, compute frozen MLP margins and clean-graph subgroups, reconstruct ENS, and regenerate correctness for every arm on the same extended attack union. Baseline per-node correctness was not saved.”

2. **BLOCKER — proposal.md:55–57,86–89: concentration of graph benefit is insufficient to decide whether adaptive caps work.**

   Low-margin nodes could contain both most graph rescues and most graph-induced errors. Giving them larger budgets may simply recreate the vulnerable residual model. Clean S0/MLP disagreement does not show that a bounded, retrained correction can exploit those rescues.

   **Replacement text:** “Report both rescue rates (`MLP wrong, graph correct`) and damage rates (`MLP correct, graph wrong`) by frozen MLP margin, including attack-induced damage. Proceed to a small training pilot only if an uncertainty-based allocation offers a useful rescue/damage trade-off across splits. Proceed to full evaluation only if a validation-selected adaptive cap beats validation-selected global caps at a prespecified matched robustness tolerance, then confirms that gain on untouched runs. Include fine-grid calibrated ENS, a constant-cap control, and a shuffled-budget control.”

   Test-label subgroup analysis is legitimate exploration, but those same results cannot subsequently count as untouched confirmation.

3. **SHOULD-FIX — proposal.md:13,72–76: the TTA evidence summary is wrong and understates protocol differences.**

   The first three gains are correctly rounded: **+3.63, +3.58, +2.85 points**. However, `tta_result.json` contains **10 rows, indexed 0–9**, and the overall gain is **+4.09 points**, from **62.40 to 66.48**. The mean clean reference is **69.85**.

   This is exploratory evidence, not protocol-grade confirmation: `eval_tta.py` deliberately enables BatchNorm although the default Cora configuration uses `normalize="none"`; evaluates final training weights rather than necessarily the best-validation checkpoint; uses one missing-feature probability and perturbation seed; and perturbs validation plus test nodes. The implementation temporarily uses batch statistics and restores BN state, so persistent contamination between evaluation arms is not the issue. It is also not evidence for VQ adaptation.

   **Replacement text:** “Exploratory BN-enabled STEM experiment: over 10 saved runs, BN adaptation raises accuracy under 50% missing features from 62.40 to 66.48 (+4.09 points). This uses a modified normalization configuration, final training weights, one corruption seed, and transductive full-graph statistics; replicate under a frozen protocol before proposing VQ-specific adaptation.”

4. **SHOULD-FIX — proposal.md:15,95: “every experiment” and “all existing evidence” being Cora-only are false.**

   `results/sweep_results.json:86` contains a populated PubMed sweep. `results/llm_gates_results.json:251,784` contains populated PubMed and `ogbn-arxiv` experiments. These do not establish held-out-domain transfer or validate their methods, but they disprove the blanket claim.

   **Replacement text:** “The cited recent controlled experiment series is Cora-only. Older results include PubMed and arXiv experiments, but the cited evidence does not establish held-out-domain transfer through shared tokens.”

5. **SHOULD-FIX — proposal.md:12,35–38: the degree gap is real, but its interpretation needs narrowing.**

   The log matches: **83.76 ID, 78.99 low-degree OOD**, a **4.77-point** gap; encoder-only regularization lowers ID by **2.90 points**. These are comparable evaluations within the same runs: `degree_shift_ood.py` evaluates the same validation-selected model on ID test nodes and OOD buckets. They are **not** a causal degradation from changing the graph, nor directly comparable with capped S0’s 78.3 clean accuracy.

   The degree script trains and validates on the middle **70%** of degree-ranked nodes and evaluates the bottom/top **15%**, using different splits and a pretrained full-model recipe. Ordinary degree quartiles within the capped test set do not reproduce that protocol. Class composition and other node properties can also explain part of the gap.

   **Replacement text:** “Within 10 full-model runs, low-degree OOD accuracy is 4.77 points below ID accuracy under a degree-selected training protocol. This motivates a controlled shift evaluation; it does not isolate degree as the cause. The tested Lipschitz settings did not improve the reported means.”

6. **SHOULD-FIX — proposal.md:21–25,97–98: the certificate is sound only with explicit definitions and implementation conditions.**

   For fixed features and frozen models, if
   \[
   f_i(A)=m(x_i)+r_i(A),\qquad \|r_i(A)\|_\infty\le B_i,
   \]
   then the top-versus-runner-up margin of **MLP logits** exceeding \(2B_i\) guarantees that its prediction survives every graph. Certified correctness additionally requires the MLP prediction to be correct. This is a **sufficient certificate**, not an iff characterization of actual robustness: nodes failing it can still be robust.

   `run.py:91–100` uses an unbounded logit margin. Consequently, `1-margin_i` can be negative; validation normalization alone does not fix this. Zero budgets also need special handling in `B_i*tanh(g/B_i)`. Large budgets on uncertain nodes can leave virtually none of those nodes certified.

   Validation-fitted calibration is not inherently leakage and does not invalidate the deterministic bound. It must be frozen before evaluation; using test labels, retuning after outcomes, or recomputing graph-dependent normalization would undermine the claimed evaluation or invariance.

   **Replacement text:** “Choose a finite nonnegative uncertainty function, freeze its calibration and scale before testing, and enforce the componentwise correction bound with explicit zero-budget handling. Certify correctness when the frozen MLP is correct and its logit margin exceeds \(2B_i\). Report certificate coverage, especially among nodes receiving larger budgets.”

7. **SHOULD-FIX — proposal.md:30–38: the premise is unverified, and the likely failure is stronger than poor calibration.**

   The saved evidence does not establish that useful graph corrections concentrate where MLP confidence is low. Confidently wrong text predictions receive tiny budgets and become difficult to repair; uncertain nodes receive large budgets and become attack targets. The certificate can never certify a correction that rescues an incorrect MLP prediction.

   **Replacement text:** “Hypothesis: frozen text uncertainty predicts where additional graph correction produces more rescues than errors. The likely failure is protecting confidently wrong text predictions while exposing uncertain predictions to graph attacks. Edge-attack improvement establishes robustness; natural-shift generalization requires a separate controlled evaluation against MLP-M and global-cap controls.”

8. **SHOULD-FIX — proposal.md:11: the cap numbers match, but the frontier description overstates the comparison.**

   **+9.62 and +8.96** match the baseline analysis. Eligibility means baseline clean accuracy is **at least CAP clean minus 0.5 points**, not exact clean matching. The analysis explicitly identifies a sparse baseline grid around 71–73 clean accuracy. CAP0.5 is inconclusive against the eligible envelope; CAP0.75/0.9 have negative envelope gaps. “Higher caps fall below RUNG-lite” loses those distinctions.

   **Replacement text:** “CAP0.1 and CAP0.25 exceed the tested eligible baseline envelope by +9.62 and +8.96 worst-node points; eligibility permits cleaner baselines and a 0.5-point clean deficit. CAP0.5 is inconclusive; CAP0.75/0.9 fall below the envelope. A finer calibrated blend grid could change this result.”

   The **12%** clean-gain retention at lines 28–29 is correctly reported from `analysis.md`.

9. **SHOULD-FIX — proposal.md:3–4,88: the inherited baseline suite includes a constraint violation.**

   The five proposed mechanisms need not change edges. However, `baselines.py:51,87–102` constructs SAGE+TextGate using an edge-cosine threshold from text features. That comparator violates “text never weights, selects, adds or drops edges.” Degree and true-label homophily diagnostics do not violate the constraint when used only for analysis.

   **Replacement text:** “All proposed interventions preserve edge construction and weights. The historical envelope includes SAGE+TextGate, which violates the current constraint; identify it as an excluded historical reference and recompute the compliant comparator envelope.”

10. **SHOULD-FIX — proposal.md:14,80–82: the EMA numbers match, but “better tokens” is too broad.**

    **+2.15 [0.98,3.31]** is EMA versus **C-random**, not versus C-repo; the latter difference is **+2.39**. **6.3×** refers to perplexity, whereas occupied-code count rises **2.81×**. This is a component reconstruction pilot with linear probes, not full STEM transfer; the separately trained continuous reference scores **2.36 points above EMA**.

    **Replacement text:** “In a component pilot, EMA improves code-probe accuracy over the fixed random-codebook control by +2.15 points and increases perplexity 6.27×. Occupied-code count increases 2.81×. Full-model and transfer benefits remain untested, and a continuous reference performs better.”

11. **SHOULD-FIX — proposal.md:19,40–44: neither ‘new’ nor the runtime estimate is established.**

    `experiments/findings.md:43` already proposes a text-confidence-conditioned correction cap. The code makes the bounded forward pass easy to extend, but currently assumes a scalar cap: constructor validation, tensor broadcasting, certificate calculation, metadata, selection, serialization, and BPDA/selfchecks all need updating. New adaptive arms also need their own attacks and common-union rescoring.

    **Replacement text:** “Previously proposed, untested extension of capped correction; novelty unestablished. Forward-pass implementation is small, but attack integration and evaluation require additional work. Benchmark one complete cell before estimating total cluster cost.”

12. **SHOULD-FIX — proposal.md:61–82: direction 3 is stronger than its ranking suggests; directions 4–5 need decisive controls.**

    All S-A numbers at line 10 match the cited analysis, including **15/15 positive cells** and **−1.77 full-label clean**. Its inference remains conditional on Cora and dependent split/seed cells. “Only supported generalization gain in the repo” is broader than the reviewed evidence supports. Existing Cora priors make the neutral-prior and small-validation controls cheaper than acquiring a new dataset’s priors.

    **Replacement text:** “S-A is the clearest positive low-label result in the cited controlled series. First run neutral-prior and small-validation controls using existing Cora priors; then replicate elsewhere. For TTA, distinguish assignment monitoring from an adaptation rule and compare against BN alone, with clean-data regression and collapse checks. For transfer, start with one held-out target and matched no-pretraining, single-domain, EMA/random-codebook, and continuous controls before scaling up.”

    Also include the controlled **commitment-loss ablation** already proposed in `experiments/findings.md:42`: its approximately **+4.1-point** lead is attack-stream-confounded, but its implementation cost is low and it respects the edge constraint.

13. **NIT — proposal.md:43–44,76: tighten prior-art language.**

    GLNN-style distillation is not itself a confidence-gating mechanism, and the saved BN-statistics intervention should not be equated with entropy-minimizing adaptation.

    **Replacement text:** “Prior-art review pending: separately examine prediction-level confidence gating, bounded residual corrections, graph distillation, BN-statistics adaptation, and entropy-minimizing adaptation.”

My ranking:

1. **Direction 3 — low-label controls and replication:** strongest direct evidence for improved generalization, with inexpensive unresolved controls.
2. **Direction 2 — corrected diagnostic pilot:** best immediate decision aid once artifacts are retrieved, provided it measures rescues and damage on a common attack union.
3. **Direction 1 — uncertainty-scaled caps:** plausible, cheap robustness hypothesis with a sound sufficient certificate, but substantial risk of reallocating vulnerability.
4. **Direction 4 — feature-shift adaptation:** a measurable exploratory lead, requiring a controlled BN baseline before adding VQ/router adaptation.
5. **Direction 5 — held-out-domain transfer:** most directly addresses STEM’s central promise, but should begin with a small discriminating experiment rather than broad pretraining.

I would place a matched commitment-loss ablation alongside direction 1 as another inexpensive robustness experiment.

Yes, there is a useful frontier: learning when graph information helps a node and limiting how much it can override the text prediction, without changing edges. The repo supports bounded corrections as a robustness trade-off, but it does not yet show that text confidence allocates those corrections well. First measure where graph predictions rescue text errors and where they introduce errors, then test whether confidence predicts that trade-off on fresh runs. For generalization specifically, the stronger existing lead is the low-label Qwen-prior result, which still needs neutral-prior, realistic-validation-budget, and second-dataset controls. Shared-token transfer to a held-out domain remains an important research question, rather than an established benefit of healthier codebooks.