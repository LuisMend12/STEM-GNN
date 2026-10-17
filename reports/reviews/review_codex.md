# Independent review: Recent GNN robustness methods

Reviewed 17 October 2026. Target: `reports/Recent GNN robustness methods.md`. No repository files changed.

## Assessment and evidence conventions

The brief is generally faithful to its notes, but some notes contain the same imprecision as the brief. The most consequential issues are a gap in SpecSphere's supplied certificate proof, an incorrect threat model in the comparison table, overbroad exclusivity claims, an incomplete TAGR weighting formula, and failure to distinguish the published STEM-GNN description from the current checkout.

I read the brief and four notes, inspected all three supplied PDFs for relevance, examined SpecSphere's certificate/proof, checked relevant implementation code, and consulted primary arXiv pages. This is not an experimental reproduction or verification of every numerical table cell. Numbers traced only to notes remain secondary evidence.

Paths below are repository-relative. Brief quotations refer to original line numbers. Notes are under `research_notes/Recent GNN robustness methods/`. PDF locators such as `p16:L97` mean page 16, line 97 of PyMuPDF `page.get_text().splitlines()`, one-based including blank lines. Equation numbers provide stable navigation. Findings distinguish demonstrated errors, missing implications, and hypotheses.

## Findings

### F01 — HIGH — SpecSphere's certificate has an unresolved proof gap

**Exact quoted text:** “**SpecSphere** is the paper that turns a Lipschitz constant of the propagator itself into a joint edge-and-feature certificate” (brief:3); “A gate composes the branches.” (brief:23).

**Problem:** The brief presents the proposed certificate without examining a crucial invalid inference. Appendix F bounds clean branch disagreement by a perturbation-dependent quantity. At zero perturbation, Eq. (62) requires the clean branches to agree exactly. The preceding triangle inequality does not establish this, and branch specialization can require disagreement. The next step drops mixed/higher-order terms without a quantified budget condition. This is a concrete objection to the supplied proof, not proof that no corrected certificate is possible.

**Evidence:** `research_notes/2505.08320v3.pdf:p16:L73-L107`, Appendix F, Eqs. (58)-(63), especially L95-L98 and L105-L107. Eq. (59) correctly retains `||Z_spec-Z_spat||∞`; Eq. (62) incorrectly makes that clean discrepancy vanish at zero perturbation. `certified-lipschitz.md:11,86` reproduces the claimed result without auditing this step.

**Proposed replacement:** “SpecSphere proposes a joint edge/feature certificate by composing branch and gate bounds. Its supplied Appendix F proof has an unresolved gap: Eq. (62) bounds clean branch disagreement by perturbation size. Treat the certificate as an author claim pending a corrected proof; a valid fusion bound must retain or independently bound branch disagreement.”

### F02 — HIGH — Wrong threat model in the comparison table

**Exact quoted text:** “AuditVotes, AGNNCert, and message-interception smoothing certify counted edge edits.” (brief:52).

**Problem:** The cited message-interception work certifies arbitrary feature rewrites of a bounded number of nodes. Edge deletion is defender noise. The table contradicts the correct body explanation.

**Evidence:** Brief:31; `certified-lipschitz.md:39,44,57` explicitly separates attack from smoothing noise.

**Proposed replacement:** “AuditVotes certifies edge-edit budgets; AGNNCert certifies combined counted perturbations; message-interception smoothing certifies arbitrary feature rewrites of a bounded number of nodes, using edge deletion and node-feature ablation as noise.”

### F03 — HIGH — Current STEM-GNN already implements the sensitivity penalty

**Exact quoted text:** “The enforced quantity is the prediction head” and “Node-state movement under edge deletion lies outside that penalty” (brief:27).

**Problem:** These can describe the paper's head penalty, but are incomplete guidance for this checkout. It also has optional encoder weight regularization and a direct edge-deletion sensitivity loss, invoked in full-batch and sampled node training. Presence does not mean enabled or formally certified.

**Evidence:** `STEM-GNN/model/ft_model.py:37-65`; `STEM-GNN/model/encoder.py:474-524`; `STEM-GNN/task/node.py:58-74,112-120`. The sensitivity loss compares clean and edge-dropped encoder states and normalizes by detached clean-state norms.

**Proposed replacement:** “The published analysis discussed here controls the head. This checkout additionally exposes optional `encoder_lip_coeff` and `propagation_lip_coeff`; the latter already penalizes relative encoder-state displacement after random undirected edge deletion. These are empirical regularizers, not established end-to-end certificates. Evaluate them before proposing the same mechanism as new implementation work.”

### F04 — HIGH — The exclusion of continuous semantic weighting is unsupported

**Exact quoted text:** “No paper in the set that was read multiplies a structural edge weight or an attention coefficient by a text similarity” and “Semantic scores spend the entire edge or none of it.” (brief:37).

**Problem:** This conflicts with the brief's uncertainty about inherited GNNGuard coefficients, its account of embedding gating/rescaling, and TAGR's continuous feature/topology combination. Hard thresholds and continuous coefficients can coexist. Not finding a newly printed formula does not establish that an inherited coefficient is absent. Distinguish raw text cosine from cosine of learned states.

**Evidence:** Brief:11,17,41,53; `llm-semantic.md:9,78,90`; `edge-structure.md:118,121`.

**Proposed replacement:** “The inspected text-defense sections emphasize thresholding. We did not verify whether Guardual retains continuous GNNGuard similarity coefficients, or establish the absence of continuous semantic weighting. TAGR combines feature cosine with structural scores in preprocessing. The narrower unverified question is whether frozen text scores allocate an explicitly constrained structural budget during propagation.”

### F05 — MEDIUM — RUNG is not the only edge-weighting method in the brief

**Exact quoted text:** “The 2024–2026 edge defenses decide trust in a few recurring ways, and only one of them reweights inside aggregation.” (brief:7).

**Problem:** Later passages describe GNNGuard's embedding gate during propagation and Pure-GNN's importance weights. Exclusivity is unsupported. The heading also generalizes a selected sample to propagation papers broadly.

**Evidence:** Brief:15,17; `edge-structure.md:46,78,109,121`.

**Proposed replacement:** “RUNG is the most fully specified robust aggregation method in this retrieval, with detailed adaptive-attack evidence. Other methods also weight messages or edges; their mechanisms and evaluations were inspected less completely.” Heading: “RUNG provides the clearest adaptive-attack evidence in this sample.”

### F06 — MEDIUM — RUNG's derivative variable matters

**Exact quoted text:** “The edge weight is the derivative of that penalty” (brief:7).

**Problem:** QN-IRLS differentiates with respect to squared residual. The ordinary derivative with respect to residual changes nonzero weights, although the cutoff stays the same. The notes share this shorthand.

**Evidence:** `edge-structure.md:16`; [RUNG §3.2 and Appendix G.1](https://arxiv.org/html/2311.14934), weight definition and Eq. (54).

**Proposed replacement:** “For nonself edges and positive residual y, the IRLS weight is `dρ(y)/d(y²)=max(0,1/(2y)−1/(2γ))`; y is the degree-normalized state difference. The zero cutoff alone does not specify the weighting rule.”

### F07 — MEDIUM — TAGR's score is not the actual edge multiplier

**Exact quoted text:** “reweights each observed edge by a fixed mix, q_ij = 1.50 z(cosine) + 0.50 z(Jaccard) + 0.25 z(log(1+M_ij)) + 0.10 z(κ_i+κ_j) − 0.10 z(|d_i−d_j|)” (brief:11).

**Problem:** The score undergoes another standardization, tanh residual transform, and clipping. Applying q directly would produce a different graph, potentially with negative weights. “Gaussian-only graph” is ambiguous: disabling residual weighting retains the original graph plus Gaussian repair edges.

**Evidence:** `edge-structure.md:32,105` gives only part of the transform; [TAGR §3, residual weighting and final repaired graph](https://arxiv.org/html/2606.03462), equations for s, r, repaired adjacency, and λ=0 discussion.

**Proposed replacement:** “TAGR standardizes q again and uses `r_ij=clip(1+λ tanh(stdz(q)_ij),0.5,1.5)`. It freezes `A⊙R+A_GFN`. At λ=0 it retains observed edges and adds Gaussian repair edges.”

### F08 — MEDIUM — Frozen preprocessing does not prove adaptive-attack failure

**Exact quoted text:** “The repaired graph cannot adapt to an attacker who sees the repair” (brief:11).

**Problem:** Freezing before downstream training is an architectural fact; vulnerability to a defense-aware attacker is a separate empirical claim. A deterministic cleaner can react to changed inputs when recomputed. The inspected experiments supply random-corruption results.

**Evidence:** `edge-structure.md:68,134`; [TAGR experimental setting](https://arxiv.org/html/2606.03462), clean/random addition/deletion protocol.

**Proposed replacement:** “TAGR freezes each repaired graph before downstream training. The inspected experiments do not establish defense-aware attack robustness; specify whether repair is recomputed after an attack before testing that setting.”

### F09 — MEDIUM — Local/global baseline rankings do change

**Exact quoted text:** “On a global 40% edge budget the gap shrinks and the order holds” (brief:9).

**Problem:** ProGNN is above GNNGuard in the listed global results but below it in the local results. Only RUNG's lead persists.

**Evidence:** Brief:9; `edge-structure.md:19,80`.

**Proposed replacement:** “RUNG remains ahead at the global 40% budget, but margins shrink and baseline rankings change.” Replace “The same columns are” with “At the 200% local budget, baseline accuracies are,” because only one baseline column is supplied.

### F10 — MEDIUM — Missing RUNG architecture ablation

**Exact quoted text:** “unroll a quasi-Newton iteratively reweighted least-squares solver as aggregation layers on an APPNP-style stack” (brief:7).

**Problem:** The notes also discuss a weaker GCN variant. The brief omits this directly relevant caution against assuming RUNG's gain transfers unchanged to GraphSAGE/MoE.

**Evidence:** `edge-structure.md:104`; `STEM-GNN/model/encoder.py:139-142,247,446-450`.

**Proposed replacement:** “The notes also report a weaker GCN variant. Compare MCP-weighted SAGE/MoE with decoupled propagation in STEM-GNN; do not attribute the full published gain to the edge-weight formula.”

### F11 — HIGH — Missing VQ-boundary and routing implications

**Exact quoted text:** “The encoder is a mixture-of-experts message passer plus vector quantization” (brief:27); “penalizes node-state movement under edge deletion” (brief:69).

**Problem:** Small encoder-state changes need not preserve discrete code assignments or routes. A smooth head and empirical encoder consistency alone do not establish prediction invariance through quantization. This is a mathematical/engineering concern, not a measured failure of STEM-GNN.

**Evidence:** `STEM-GNN/model/ft_model.py:140-152` applies VQ before the decoder; `:89-117` implements routing consistency; `certified-lipschitz.md:90` flags route instability. The sensitivity term measures pre-VQ states (`encoder.py:513-524`).

**Proposed replacement:** “Measure pre-VQ displacement, per-head code-change rate, router-distribution change, and prediction flips under identical perturbations. A future certificate needs distances to quantizer boundaries and control of routing, together with a classification margin.”

### F12 — MEDIUM — A certified radius is not a Lipschitz constant

**Exact quoted text:** “No opened 2024–2026 paper states a certified radius equal to the Lipschitz constant of a GraphSAGE or GCN message function under an ℓ2 feature ball alone” (brief:33).

**Problem:** A radius is normally derived from a margin divided by a sensitivity bound. A larger constant generally reduces the radius. Absence of this unusual equality is not a meaningful research gap.

**Evidence:** `certified-lipschitz.md:25` repeats the wording; `research_notes/2505.08320v3.pdf:p5`, Theorem 4/Eq. (33), combines perturbation bounds with a margin.

**Proposed replacement:** “This retrieval did not identify a GraphSAGE/GCN feature-ℓ2 certificate derived from an end-to-end sensitivity bound and decision margin. A local message-function constant alone would not certify the full classifier.”

### F13 — MEDIUM — The combinatorial-only framing contradicts SpecSphere

**Exact quoted text:** “What the opened defenses actually certify is combinatorial.” (brief:29).

**Problem:** SpecSphere, already discussed in brief:23, claims continuous ℓ∞ feature perturbations jointly with edge edits.

**Evidence:** Brief:23; `certified-lipschitz.md:31,38`.

**Proposed replacement:** “Most other certificates here use counted edge, node, or feature-replacement budgets. SpecSphere claims a joint edge-edit/continuous-ℓ∞ feature guarantee, subject to the proof concerns above.”

### F14 — MEDIUM — Omitted structural-shift evaluation is directly relevant

**Exact quoted text:** “Degree bias, which is the retrieved neighbor of ‘degree shift,’ is patched at test time” (brief:59).

**Problem:** This omits Bazhenov et al.'s direct structural distribution-shift evaluation from the notes. Degree-conditioned difficulty and train/test degree shift are different questions.

**Evidence:** `shift-surveys.md:23` describes popularity/locality/density splits and strong simple baselines; `:28-30` separately describes degree bias; `CLAUDE.md:39-45` lists the OOD scripts.

**Proposed replacement:** “Add Bazhenov et al. (arXiv:2302.13875) as a structural-shift evaluation reference. Report train/test property distributions and feature-only/standard-GNN baselines for degree/homophily OOD. Degree-bias mitigation alone does not establish shift robustness.”

### F15 — MEDIUM — Missing robust-filter and pretraining precedents

**Exact quoted text:** “The rest of the slice is real and thinner.” (brief:15).

**Problem:** Abstract-only methods receive space while Mid-GCN, explicitly in the notes, is omitted. STABLE is another relevant pretraining/refinement comparator identified by a primary source. These additions are related work, not established exact matches to the proposed combination.

**Evidence:** `edge-structure.md:56,85,137`; [LLM4RGNN Appendix A.2](https://arxiv.org/html/2408.08685), description of STABLE.

**Proposed replacement:** “Include Mid-GCN (WWW 2023; DOI 10.1145/3543507.3583335) as a robust-filter precedent. Add STABLE as a pretraining/refinement comparator, marking its original mechanism and attack protocol for primary-paper verification before claiming exact overlap.”

### F16 — MEDIUM — Missing certification-directed immunization

**Exact quoted text:** “When they want a guarantee, they change the input distribution or the aggregator.” (brief:21).

**Problem:** The notes identify graph immunization as another intervention, which this framing omits.

**Evidence:** `certified-lipschitz.md:99`; [Graph Adversarial Immunization for Certifiable Robustness](https://arxiv.org/abs/2302.08051), abstract describes selected edge/node immunization.

**Proposed replacement:** “Certification-oriented interventions also include graph immunization (Tao et al., arXiv:2302.08051). Its certificate construction was not inspected in the notes; this taxonomy is not exhaustive.”

### F17 — MEDIUM — Supplied oversmoothing PDF adds an omitted limitation

**Exact quoted text:** “Oversmoothing is treated as related and not identical to adversarial robustness.” (brief:63).

**Problem:** Correct, but the supplied Zhao et al. PDF adds an operator-consistency distinction useful for evaluating DropEdge and changing propagation weights. Retain the paper's Markovian assumptions rather than generalizing to arbitrary encoders.

**Evidence:** `research_notes/2211.06605v2.pdf:p1:L14-L28`; p2 discussion of residuals, PPNP, and DropEdge. Authors are Weichen Zhao, Chenguang Wang, Congying Han, Tiande Guo (p1:L4).

**Proposed replacement:** “Zhao et al. (arXiv:2211.06605v2, March 2023 revision) analyze oversmoothing via Markov chains: operator consistency and convergence assumptions matter, and common mitigations need not prevent asymptotic smoothing. This is depth analysis, not an adversarial certificate.”

### F18 — LOW — Explicitly exclude the unrelated supplied AMP PDF

**Exact quoted text:** “The record behind these claims is OpenAlex, arXiv HTML, and PDFs retrieved on 4 October 2026.” (brief:3).

**Problem:** A source inventory should clarify the supplied PDFs' relevance. The 2512 PDF is about spiked-Wigner inference, not GNN robustness. Its exclusion is appropriate; this is a documentation gap, not a detected misattribution.

**Evidence:** `research_notes/2512.02384v3.pdf:p1:L1-L20`: “Markov Chains Approximate Message Passing,” Amit Rajaraman and David X. Wu, Bayesian matrix inference and Glauber dynamics.

**Proposed replacement:** “SpecSphere supports the robustness discussion; Zhao et al. supplies adjacent oversmoothing analysis; arXiv:2512.02384v3 was excluded because AMP inference analysis is not a GNN robustness defense.”

### F19 — HIGH — Missing actionable matched evaluation for STEM-GNN

**Exact quoted text:** “A design that reweights from structural discrepancy during propagation, penalizes node-state movement under edge deletion, and spends a structural trust budget from frozen text scores is absent from the papers that were opened.” (brief:69).

**Problem:** Limited-search absence is neither a defensible novelty claim nor an experiment plan. Isolate components and distinguish current corruption scripts from adaptive-attack evidence.

**Evidence:** `edge-structure.md:17,68,127-146`; `CLAUDE.md:39-45`; `STEM-GNN/task/node.py:58-74` already separates regularizers.

**Proposed replacement:** “Treat the combination as a candidate hypothesis. Run baseline, structural weighting only, existing sensitivity loss only, frozen semantic gating only, and combinations under matched seeds/checkpoints. Evaluate degree/homophily OOD, missing features, and random deletion as separate corruption regimes; add defense-aware insertion/deletion and joint text/structure attacks. Report clean accuracy, robustness curves, variability, and cost.”

### F20 — MEDIUM — ‘Trust budget’ is undefined

**Exact quoted text:** “Frozen text scores allocate a structural trust budget” (brief:53).

**Problem:** A product of scores is not automatically a budget. The brief never specifies whether it means incoming message mass, retained edges, or adjacency changes.

**Evidence:** Brief:37,49-53,69; `edge-structure.md:100-118` distinguishes membership, weights, and hop mixing; `STEM-GNN/model/encoder.py:139-142` adds edge attributes before aggregation.

**Proposed replacement:** “Define a proposed budget explicitly, for example a constraint on total normalized incoming weight per node, and how semantic scores allocate it. Specify whether weights apply to complete messages including edge attributes, and how self/residual contributions are handled. This is a design proposal, not a literature finding.”

### F21 — MEDIUM — Missing features and attacked text need different tests

**Exact quoted text:** “Own text is the reliable evidence under a structure attack, and it is not trusted under a text attack.” (brief:43).

**Problem:** This is a defense intuition, not an unconditional premise. Cached text embeddings can be missing or shifted; coordinate masking does not simulate adversarial text rewriting.

**Evidence:** `llm-semantic.md:100-104` describes joint attacks; `STEM-GNN/scripts/missing_feature.py:163-174` masks coordinates and `:191-212` adds embedding noise.

**Proposed replacement:** “SFT-auto prefers own text for detected structure-only attacks. Test STEM-GNN semantic gates separately under embedding masking/noise, raw-text rewrites followed by re-encoding, and combined topology attacks.”

### F22 — LOW — Resolve the recoverable author-list gap

**Exact quoted text:** “with Yongchao Liu named in a footnote; ... full author list and venue were not recovered” (brief:41).

**Problem:** The primary HTML now exposes additional authors. Preserve uncertainty about venue, not a recoverable author-list omission.

**Evidence:** `llm-semantic.md:9`; [TGRB HTML byline](https://arxiv.org/html/2510.17185), author block includes Yongchao Liu and Chuntao Hong.

**Proposed replacement:** “TGRB (Lei, Yi, He, Qiu, Wei, Liu, and Hong; arXiv:2510.17185, 2025; venue unverified).”

### F23 — LOW — Retrieval history overwhelms synthesis

**Exact quoted text:** “Semantic Scholar returned HTTP 429 on every query, a later arXiv batch also returned HTTP 429, several abstracts were missing” (brief:3); “The 2024–2026 papers that were opened keep three designs in different rooms” (brief:67).

**Problem:** Repeated caveats, citation counts, and metaphors obscure the scientific distinctions and proposed work. Failed literal title searches are weak evidence of field-wide gaps even with disclaimers.

**Evidence:** Brief:3,21,27,33,57,59,63,69; `shift-surveys.md:3,53-55,189-191` already records retrieval limits.

**Proposed replacement:** Open with: “This selected 2023–2026 review distinguishes adaptive attack defenses, certified guarantees, and distribution-shift methods. Their metrics are not directly comparable. For STEM-GNN, priorities are matched corruption/adaptive-attack evaluation and ablation of existing sensitivity regularization.” Move retrieval failures and citation snapshots to one appendix. Use a compact matrix of intervention stage, threat, guarantee, benchmark, evidence quality, and repository relevance.

## Retained claims and verification limits

The body usually distinguishes random deletion from adaptive attack evaluation, poisoning from evasion, calibration from accuracy recovery, and RUNG feature residuals from structural discrepancy correctly. RUNG's quoted hyperparameter grids agree with its primary source. The checked notes support its numerical lead, but not an unchanged full ranking. TAGR's displayed coefficients match its source; its omitted transformation is the issue. The SpecSphere PDF confirms its authors, version date, bounds, and assumptions as statements made by the paper; this does not validate its theorem.

Several exact values, including AuditVotes relative improvements, GJDNet coverage, and recent calibration/TTA figures, were traceable to notes but not independently rederived from tables. Retain attribution rather than treating these as verified cross-paper comparisons. No unsupported alternative values are proposed. The AAAI copyright footer in SpecSphere does not itself establish conference acceptance. Nothing here claims that an uninspected paper lacks a mechanism merely because retrieval failed.

## Prioritized top 10

1. **F01 (HIGH):** Qualify SpecSphere's certificate and explain its Appendix F proof gap.
2. **F02 (HIGH):** Correct message-interception smoothing's threat model in the table.
3. **F03 (HIGH):** Distinguish published STEM-GNN claims from penalties implemented in this checkout.
4. **F04 (HIGH):** Retract the categorical exclusion of continuous semantic weighting.
5. **F11 (HIGH):** Add VQ-code and routing stability measurements.
6. **F19 (HIGH):** Supply matched ablations and separate corruption/adaptive-attack evaluations.
7. **F07 (MEDIUM):** Give TAGR's actual bounded multiplier and adjacency construction.
8. **F06 (MEDIUM):** Specify RUNG's derivative with respect to squared residual.
9. **F14 (MEDIUM):** Restore structural-shift evaluation and separate degree bias from shift.
10. **F05 (MEDIUM):** Remove aggregation-reweighting exclusivity contradicted by the brief itself.
