# Where STEM-GNN's generalization could improve: candidate directions

Written 2026-11-09. A ranked proposal, not a result. Revised after Codex review
(`../reviews/codex_generalization_proposal_review_2026-11-09.md`); all findings applied.

**Constraint.** All proposed interventions leave edge construction and weights alone: text never weights,
selects, adds or drops edges. The earlier baseline envelope includes SAGE+TextGate, which breaks this
constraint. It is kept only as an excluded historical reference, and any new comparator envelope is
computed without it.

## Evidence already in the repo

| Evidence | Source | Reading |
|---|---|---|
| Qwen head prior (S-A): +6.98 [+4.34, +9.61] low-label accuracy vs S0, 15/15 cells positive; −1.77 full-label clean | `experiments/llm-stem/analysis_stage2.md` | The clearest positive low-label result in the recent controlled series. Conditional on Cora; cells share splits. Missing: neutral-prior and realistic-validation-budget controls, and a second dataset |
| Capped correction: CAP0.1 and CAP0.25 exceed the tested eligible baseline envelope by +9.62 and +8.96 worst-node@20% points | `experiments/capped-correction/analysis_baselines.md` | Eligibility allows cleaner baselines and a 0.5-point clean deficit. CAP0.5 is inconclusive; CAP0.75 and CAP0.9 fall below the envelope. A finer, calibrated blend grid could change this. CAP\* kept 12% of S0's clean gain over MLP-M |
| Degree OOD, 10 full-model runs: ID 83.76, low-degree OOD 78.99 (−4.77) | `results/degree_ood/degree_ood_sweep.log` | Same validation-selected models, evaluated on ID test nodes and degree buckets. Training uses the middle 70% of degree-ranked nodes. This motivates a controlled shift evaluation but doesn't isolate degree as the cause. The tested Lipschitz settings didn't improve the means (encoder penalty −2.90 ID) |
| Test-time BN adaptation, 50% missing features, 10 saved runs: 62.40 → 66.48 (+4.09); clean reference 69.85 | `results/tta/tta_result.json`, `eval_tta.py` | Exploratory. BN was enabled on purpose (the default Cora config has none), final training weights were used, one corruption seed, transductive full-graph statistics. Not evidence for VQ adaptation |
| EMA codebook, component pilot: +2.15 [+0.98, +3.31] code-probe accuracy over the fixed random-codebook control (+2.39 over C-repo); perplexity 6.27×; occupied codes 2.81× | `experiments/stem-fixes/analysis.md` | Linear probes on a reconstruction pilot, not full STEM. A continuous reference scores 2.36 above EMA. Full-model and transfer benefits are untested |
| Datasets | `results/sweep_results.json`, `results/llm_gates_results.json` | The recent controlled series is Cora-only. Older results include PubMed and ogbn-arxiv runs, but none tests held-out-domain transfer through shared tokens |

## Candidate directions, ranked

### 1. Close out the low-label result (best evidence for generalization, cheapest controls)

S-A is the clearest positive generalization result. It needs, in order:
1. **A neutral prior** (uniform, or the training-label marginal) in place of Qwen.
2. **A validation budget matched to the label budget** (for example about 21 validation labels, not 500),
   for temperature, α and checkpoint selection.
3. **A second dataset.** PubMed is the obvious candidate; older PubMed results exist, but Qwen priors for
   it would need the GPU.

Steps 1 and 2 reuse the existing Cora Qwen priors, so they're cheap (CPU jobs on the cluster, no new
LLM inference). Novelty is low: distilling LLM pseudo-labels or annotations for graphs with scarce labels
is an existing line, so this would be a validated improvement to STEM, not a new method.

### 2. Diagnostic pilot: where does the graph rescue the text, and where does it damage it?

Conditional on the cluster artifacts:
- the capped checkpoints and `_pernode.pt` files (correctness booleans only, no logits);
- the baseline checkpoints. Baseline per-node correctness was not saved.

Steps:
- Reload the dataset and splits, then compute the frozen MLP-M margins and clean-graph subgroups: degree,
  and local homophily from true labels (analysis only).
- Rebuild ENS, and regenerate correctness for every arm on the same extended attack union.
- By MLP margin bin, report rescue rates (MLP wrong, graph arm correct) and damage rates (MLP correct,
  graph arm wrong), clean and attack-induced.

These test-label subgroup results are exploratory and can't later count as confirmation. Cost: one short
cluster job (no training, just reloading and rescoring).

**Go/no-go for direction 3.** Go to a small training pilot only if allocating budget by uncertainty gives a
useful rescue-versus-damage trade-off in every split.

### 3. Text-confidence-scaled cap (previously proposed in `findings.md`; untested; novelty unestablished)

**Idea.** A per-node budget B_i = B · u(m(x_i)):
- u is a finite, nonnegative uncertainty function of the frozen text model only. The raw logit margin is
  unbounded, so u must be bounded, for example a calibrated softmax margin.
- Its calibration and scale are fitted on validation and frozen before testing.
- Zero budgets are handled explicitly; B_i·tanh(g/B_i) is undefined at B_i = 0.

**Certificate (sufficient, not iff).** With f_i(A) = m(x_i) + r_i(A) and |r_i(A)| ≤ B_i per coordinate, a
node is certified correct if the frozen MLP is correct and its logit margin exceeds 2B_i. Nodes failing
this can still be robust. Report certificate coverage, especially among nodes with large budgets.

**Hypothesis.** Frozen text uncertainty predicts where extra graph correction produces more rescues than
errors.

**Likely failure.** It protects confidently *wrong* text predictions (tiny budget, so they can't be
repaired; the certificate can never certify a rescue) while exposing uncertain nodes to graph attacks.

**Evaluation.** Proceed to full evaluation only if:
- a validation-selected adaptive cap beats validation-selected global caps at a prespecified matched
  robustness tolerance;
- and that gain holds up on untouched runs.

Controls:
- a constant-cap control;
- a shuffled-budget control (same budgets, randomly reassigned to nodes);
- a fine-grid, calibrated ENS;
- the compliant baselines.

Edge-attack gains establish robustness only; natural-shift generalization needs a separate controlled
shift evaluation against MLP-M and the global caps.

**Cost.** The forward pass is small, but `capped/run.py` assumes a scalar cap throughout. Validation,
broadcasting, certificate, metadata, selection, serialization, and the BPDA attacks and selfchecks all
need updating. The new arms also need their own attacks and common-union rescoring. Benchmark one full
cell before estimating cluster time.

**Alongside it: a commitment-loss ablation** (coefficient {0, 0.05, 0.25}, reseeded attacks). Proposed in
`findings.md`. Its about +4.1-point lead came from an unmatched comparison with confounded attack streams,
but it's cheap and respects the constraint.

### 4. Test-time adaptation for feature shift

First replicate BN adaptation under a frozen protocol:
- the default normalization or a declared change;
- best-validation checkpoints;
- several corruption seeds and missing rates.

Only then try a VQ- or router-specific rule. Distinguish monitoring code assignment from an actual
adaptation rule. Compare against BN alone, with a clean-data regression check and a collapse check.

Prior-art review pending: BN-statistics adaptation and entropy-minimizing adaptation (TENT-style) are
distinct, known families.

### 5. Held-out-domain transfer (STEM's central promise; most expensive)

Start small with one held-out target domain, few-shot, and matched controls:
- no pretraining;
- single-domain pretraining;
- multi-domain pretraining;
- EMA versus random codebook;
- a continuous (no-VQ) reference.

Scale up only if a difference appears. This is the direction that most directly tests STEM's claim, but the
codebook-health gain so far is a component-pilot result, not evidence of transfer.

## Recommendation

1. Run **direction 1's neutral-prior and small-validation controls** on Cora first: the strongest
   generalization lead, cheap, and decisive about whether the +7.0 is real.
2. In parallel, run the **diagnostic pilot (direction 2)**. It's no-training and decides whether
   direction 3 is worth building.
3. Build **direction 3** (with the commitment-loss ablation alongside) only if the pilot shows a useful
   rescue-versus-damage trade-off.

## Risks

- The recent controlled evidence is Cora-only. Claiming generalization in general needs a held-out dataset.
- The adaptive cap could reallocate vulnerability rather than reduce it (see the likely failure in
  direction 3).
- Low-homophily subgroups use true labels: fine for analysis, never as a model input.
- Prior-art review is pending for each direction: prediction-level confidence gating, bounded residual
  corrections, graph distillation, BN-statistics adaptation, entropy-minimizing adaptation.
