# Review: `reports/Recent GNN robustness methods.md`

Reviewer: Claude (independent of the Codex/GPT review). Date: 2026-10-17. No repo files were edited.

Sources checked:
- Brief: `reports/Recent GNN robustness methods.md` (cited as `brief:L<n>`).
- Notes: `research_notes/Recent GNN robustness methods/{edge-structure,certified-lipschitz,llm-semantic,shift-surveys}.md` (cited as `ES`, `CL`, `LS`, `SS` plus line).
- Local PDFs: `research_notes/2505.08320v3.pdf` (SpecSphere), extracted with pdftotext and cited as `SpecSphere.txt:L<n>`. `2211.06605v2.pdf` is Zhao et al., an oversmoothing paper that takes a Markov-chain view. `2512.02384v3.pdf` is Rajaraman and Wu on spiked-Wigner AMP and is not relevant.
- Web: the arXiv abstracts for 2510.17185, 2405.16405, 2602.09258, 2006.08149 (the GNNGuard PDF), and 2402.13987. Search results for Jia et al. (KDD 2023), Juvina et al. (TMLR 2024), and Scholten et al. (NeurIPS 2023).
- Repo code: `STEM-GNN/model/encoder.py:474-523`, `STEM-GNN/utils/args.py:152-177`, `STEM-GNN/utils/others.py:189-196`, `STEM-GNN/LLM4GNN/ICLR/iclr2026_conference.tex` (the internal memo "Semantic-Structural Robust Propagation for GNNs").

Overall: the brief follows its notes closely, and nearly every number traces to a note line. Its weaknesses are a different kind:
- (a) Several "not found" claims are contradicted by well-known 2023–2024 work that the search missed.
- (b) Two key mechanisms are described too weakly or too strongly: GNNGuard too weakly, SpecSphere's certificate too strongly.
- (c) The brief never says it is a prior-art check for the memo's three design ideas, so its negative findings read as unexplained novelty claims.
- (d) Provenance caveats are interleaved into nearly every sentence, which makes it hard to read.

---

## 1. Factual accuracy

### 1.1 HIGH: GNNGuard is described as a gate. It is a continuous cosine reweighting inside message passing.
Quoted (brief:L17): "GNNGuard, as restated in Mujkanovic et al.'s appendix, keeps edges whose layer-wise embedding cosine is at least 0.1 and rescales by a degree-like normalizer. That is an embedding gate during propagation…"
Also (brief:L37, L53): "No paper in the set that was read multiplies a structural edge weight or an attention coefficient by a text similarity…"

Problem:
- The original GNNGuard (Zhang and Zitnik, NeurIPS 2020) computes cosine similarity s_uv of the endpoint embeddings at every layer. It normalizes s_uv within u's neighborhood to get importance weights α_uv. It prunes edges through a learned transform of [α_uv, α_vu] against a threshold, and then uses the surviving α_uv as continuous message weights, with layer-wise memory β.
- The paper says it "assign[s] higher weights to edges connecting similar nodes while pruning edges between unrelated nodes" and "assigns less weight to suspicious edges".
- So GNNGuard multiplies the observed adjacency by a normalized cosine. On a TAG whose input features are sentence or LLM embeddings, which is the setting where TGRB finds GNNGuard "top-tier" (brief:L41), the first-layer weight is a normalized text-embedding cosine times A_ij.
- That is the nearest published prior art to "structural weight × text similarity". It weakens the brief's headline negative more than the brief admits. The brief also flags the formula as unread (brief:L41: "whether TGRB inherits a cosine edge coefficient… was not on the extracted pages"), but the formula is easy to read.

Evidence: GNNGuard PDF (arXiv 2006.08149), Sec. 4.1–4.2: "we use cosine similarity to calculate d… We normalize s_uv^k at the node-level within u's neighborhood… prunes likely fake edges, and assigns less weight to suspicious edges". Also LS:L52 and LS:L64, where the notes already list this as a gap.

Proposed replacement for L17's GNNGuard sentence:
> "GNNGuard (Zhang and Zitnik, NeurIPS 2020) computes the cosine of endpoint embeddings at each layer, normalizes it within each neighborhood, prunes edges through a learned threshold on those normalized scores, and uses the surviving normalized cosines as continuous message weights with a layer-wise memory. When the input features are frozen text embeddings, as in TGRB's re-evaluation, its first-layer edge weight is a neighborhood-normalized text-embedding cosine multiplied into the adjacency. That is the closest published relative of a structural-weight-times-text-score rule, although GNNGuard has no separate structural discrepancy term and no per-node budget."

Proposed replacement for L37's first two sentences:
> "No paper read here multiplies a *separately computed* structural or propagation weight (for example RUNG's W_ij) by a frozen text score, or allocates a per-node trust budget from such a score. The nearest relative is GNNGuard on text-embedding features, whose edge weight is itself a normalized embedding cosine (see the edge section). Other semantic defenses use hard keep-or-drop tests."

### 1.2 HIGH: The "Lipschitz training of GNN layers not found" claims miss 2023–2024 work.
Quoted (brief:L21): "Orthogonal weight constraints and spectral normalization of GNN layers were not found as training methods for certificates."
Quoted (brief:L33): "No opened 2024–2026 paper states a certified radius equal to the Lipschitz constant of a GraphSAGE or GCN message function under an ℓ2 feature ball alone…"

Problem: these sentences are true only of the opened sample, and the brief's stated window is 2023–2026 (brief:L3). Directly relevant papers exist:
- Jia, Zou, Wang, Jin, "Enhancing Node-Level Adversarial Defenses by Lipschitz Regularization of Graph Neural Networks," KDD 2023, DOI 10.1145/3580305.3599335. It derives Lipschitz upper bounds for GCN, GraphSAGE and GAT and uses them as a training regularizer.
- Juvina, Neacșu, Rony, Pesquet, Burileanu, Ben Ayed, "Training Graph Neural Networks Subject to a Tight Lipschitz Constraint," TMLR 2024. It enforces a tight spectral-theoretic Lipschitz bound by projected subgradient training, aimed at feature attacks.
- Ennadir et al., "A Simple and Yet Fairly Effective Defense for GNNs" (NoisyGNN), AAAI 2024, arXiv 2402.13987. It injects noise into the architecture with a theoretical robustness link and appears in the notes only as a DSGNN baseline (ES:L47).

The arXiv phrase query `"spectral normalization" AND "graph neural"` returning 0 hits says more about how narrow the query was than about the literature.

Evidence: Duke Scholars record for Jia et al. (https://scholars.duke.edu/publication/1591315); MLAnthology TMLR 2024 entry (https://mlanthology.org/tmlr/2024/juvina2024tmlr-training); arXiv 2402.13987.

Proposed replacement for L21's second sentence onward:
> "The opened 2024–2026 certificate papers do not spectrally normalize GNN layers. Lipschitz-constrained GNN training does exist outside that sample: Jia et al. (KDD 2023) derive Lipschitz bounds for GCN, GraphSAGE and GAT and regularize them, and Juvina et al. (TMLR 2024) train under a tight Lipschitz constraint against feature attacks. Both bound sensitivity to features, not to edge deletion. The zero-hit arXiv phrase queries are therefore a limit of the query, not evidence of absence."

### 1.3 HIGH: Continuous feature certification for graphs does exist.
Quoted (brief:L33): "Continuous feature-ℓ2 certification of the image-domain kind was not found for graphs in the opened papers; AuditVotes mentions Gaussian smoothing for images."

Problem: Hierarchical Randomized Smoothing (Scholten, Schuchardt, Bojchevski, Günnemann; NeurIPS 2023; arXiv 2310.16221) adds noise to a random subset of nodes. Its continuous noise variants certify continuous feature perturbations, and it is evaluated on node classification. These are the same authors as message-interception smoothing, which the brief does include (brief:L31).

Evidence: arXiv 2310.16221 abstract ("certificates for both discrete and continuous domains"); TUM project page https://cs.cit.tum.de/daml/hierarchical-smoothing.

Proposed replacement:
> "Among the opened papers, none certifies a continuous ℓ2 feature ball. Outside the opened sample, hierarchical randomized smoothing (Scholten et al., NeurIPS 2023) does certify continuous feature perturbations on a bounded number of nodes for node classification, so the gap is in this pass and not in the field."

### 1.4 HIGH: SpecSphere's certificate is presented more strongly than the paper supports.
Quoted (brief:L3): "SpecSphere is the paper that turns a Lipschitz constant of the propagator itself into a joint edge-and-feature certificate."
Quoted (brief:L23): "Whether ∥W∥₂ ≤ β is enforced by spectral normalization, by projection, or only assumed in the proof was not visible in the extracted sections."

Problems found in the local PDF:
1. **The proof is first-order, not exact.** The Theorem 4 proof (Appendix F, eqs. 58–63) bounds the gate term and then states: "we keep only linear terms in ∥ΔA∥₂, ∥ΔX∥∞ since higher-order terms are dominated for small budgets, and constants can again be absorbed." The cross term is bounded with an unspecified constant C ≥ 1 that is "absorbed". The bound is therefore a small-budget linearization with hidden constants, not a sound worst-case certificate. (SpecSphere.txt:L1100-1130)
2. **β and γ_ℓ are assumptions, not enforced.** The text reads "Assume each attention layer and weight spectral norm ≤ β" (SpecSphere.txt:L338-340). The Implementation paragraph lists only lr 1e-2, dropout 0.5, weight decay 5e-4, 8 heads, and 300 epochs, with no spectral normalization or projection (SpecSphere.txt:L455-460, L487-490). This answers the brief's open question as far as the paper goes: β is only assumed.
3. **No certified accuracy is reported.** Table 2 and Table 7 report empirical accuracy under DropEdge 20%, Metattack 5%, and Feature-PGD ε=0.1 on Cora and Chameleon. No certified-accuracy curve or radius is reported, even though the abstract claims "tighter certified robustness" (SpecSphere.txt:L31, L378-395).
4. **The certificate is likely vacuous at the training budget.** It scales as (2^{K+1}−1)‖c‖₁ and β^{L_p}, and training uses p = 0.1|E| (SpecSphere.txt:L457). That gives √(2p) ≈ 30 on Cora, so the right-hand side of eq. 33 is almost certainly larger than any softmax margin. The brief should say this, labeled as an inference.
5. **Training objective detail.** One weight λ_cons multiplies all four of R_LP, R_HP, L_cons and a margin-based complementarity loss L_comp (eq. 21, SpecSphere.txt:L237-245). The brief omits L_comp and implies that separate weights exist.
6. **Hyperparameters.** The reported optima are λ_adv=0.5, λ_cons=10⁻² on Cora and λ_adv=1.0, λ_cons=10⁻¹ on Chameleon. Performance drops at λ_adv ≥ 1.0 on Cora (SpecSphere.txt:L474-490). The brief gives only the Chameleon setting.
7. **There is a head-to-head with RUNG and GNNGuard.** It is relevant because the brief says no shared leaderboard exists (brief:L15). Appendix Table 7 on Chameleon: SpecSphere+ 46.4 / 41.9 / 45.1, RUNG 45.6 / 42.8 / 44.1, GNNGuard 45.0 / 41.3 / 43.7 (DropEdge / Metattack / Feature-PGD). RUNG beats SpecSphere under Metattack. (SpecSphere.txt:L1245-1258)
8. **Internal inconsistency in SpecSphere.** Clean Chameleon accuracy is 61.4% in Appendix Table 5, which is trained without adversarial terms, but 46–48% in Tables 2 and 7 (SpecSphere.txt:L1270-1280 and L378-395). Readers should not compare these numbers across tables.
9. **Venue.** The PDF carries "Copyright © 2025, Association for the Advancement of Artificial Intelligence" and an anonymous code link (SpecSphere.txt:L56-60). That is an AAAI template, which suggests an AAAI submission. Acceptance is still unconfirmed.

Proposed replacement for L23, from "Training described on the page…" to the end of the paragraph:
> "The training loss is cross-entropy plus λ_adv times three PGD inner maximizations (edges on the spectral branch, features on the spatial branch, both on the fused output, run every T=10 rounds), plus a single λ_cons weighting low-pass, high-pass, consistency and margin-complementarity terms. Reported optima are λ_adv=0.5, λ_cons=10⁻² on Cora and λ_adv=1.0, λ_cons=10⁻¹ on Chameleon. The spectral-norm bounds β and γ_ℓ are assumptions of the theorem. The implementation section reports only weight decay, with no spectral normalization or projection. The proof of Theorem 4 keeps only first-order terms 'for small budgets' and absorbs unspecified constants, so it is a linearized bound rather than a sound certificate. The experiments report empirical accuracy (DropEdge 20%, Metattack 5%, Feature-PGD ε=0.1 on Cora and Chameleon) and no certified accuracy. With the training budget p=0.1|E|, √(2p) is about 30 on Cora, so the bound is likely vacuous there (our inference). Appendix Table 7 does compare against RUNG and GNNGuard on Chameleon, and RUNG is ahead under Metattack (42.8 versus 41.9)."

Also soften L3:
> "SpecSphere is the one opened paper that derives a joint edge-and-feature bound from Lipschitz constants of the propagators themselves, though the bound is a first-order, assumption-dependent result with no reported certified accuracy."

### 1.5 MEDIUM: RUNG numbers are selected in a way that overstates the margin, and "the order holds" is wrong.
Quoted (brief:L9): "The same columns are 2.0±2.7 for GNNGuard at 200%, 0.0 for ProGNN and RGCN, 0.7±1.3 for Jaccard-GCN, 8.7±3.4 for SoftMedian, and 12.0±6.9 for TWIRLS. On a global 40% edge budget the gap shrinks and the order holds: RUNG 65.1±1.2, SoftMedian 58.1±0.7, GNNGuard 51.0±1.2, ProGNN 51.3±0.6…"

Problems:
- The strongest non-RUNG baseline in Table 1 is **TWIRLS-T, 40.7±11.8 at 200%**, and it is omitted. The real local margin is about 29 points, not about 57.
- GARNET (9.6±3.5), EvenNet (1.1±1.3) and the undefended GCN (0.0) are also missing, so the reader has no undefended reference.
- At the global 40% budget, TWIRLS is 58.7±1.1, which is *above* SoftMedian 58.1, and it is omitted.
- ProGNN (51.3) is above GNNGuard (51.0) but listed after it. RGCN (46.1) is *below* undefended GCN (48.4).
- "Same columns" promises clean / 100% / 200% values but gives only 200%.

Evidence: ES:L18-19, ES:L80.

Proposed replacement:
> "At 200% of target degree, RUNG keeps 69.3±9.0. The strongest baseline is TWIRLS-T at 40.7±11.8. Next come GARNET 9.6±3.5, SoftMedian 8.7±3.4, TWIRLS 12.0±6.9, GNNGuard 2.0±2.7, Jaccard-GCN 0.7±1.3, and EvenNet 1.1±1.3. ProGNN, RGCN and undefended GCN are all at 0.0. Standard deviations near ±10 reflect a small set of target nodes. Under a global 40% budget the gap narrows: RUNG 65.1±1.2, TWIRLS 58.7±1.1, SoftMedian 58.1±0.7, ProGNN 51.3±0.6, GNNGuard 51.0±1.2, Jaccard-GCN 49.1±2.4, GCN 48.4±0.5, RGCN 46.1±0.7. RUNG still leads and the ℓ1 family is second, but RGCN falls below undefended GCN."

### 1.6 MEDIUM: Bibliographic gaps the brief leaves open can be resolved.

| Brief text | Resolution | Evidence |
|---|---|---|
| L41: "HTML byline Runlin Lei, Lu Yi, Mingguo He, Pengyu Qiu, and Zhewei Wei, with Yongchao Liu named in a footnote… full author list and venue were not recovered" | Authors are Runlin Lei, Lu Yi, Mingguo He, Pengyu Qiu, Zhewei Wei, Yongchao Liu, Chuntao Hong. There is no venue comment, so it is an arXiv preprint. | https://arxiv.org/abs/2510.17185 |
| L45: "The 2024 text-level graph-injection paper… had its title and arXiv id truncated" | "Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level," Lei, Hu, Ren, Wei, arXiv:2405.16405, NeurIPS 2024 | https://arxiv.org/abs/2405.16405 |
| L27: "The arXiv API abstract says 'nine' benchmarks; the HTML introduction says 'eight.'" | The current abstract (v2, revised 12 June 2026) says "nine node, link, and graph benchmarks". Cite v2 and drop the eight/nine aside, or move it to a footnote. | https://arxiv.org/abs/2602.09258 |
| L15: "ADEdgeDrop (IEEE TKDE typesetting; volume and issue year not extracted)" | The notes already give IEEE TKDE 2025, DOI 10.1109/tkde.2025.3586369, with 16 OpenAlex citations. This is an internal inconsistency between note files. | SS:L172 vs ES:L43 |
| L15: "the NeurIPS 2022 proceedings page was not opened" | Lee and Park's bibliography lists NeurIPS 2022 (ES:L5). It is a NeurIPS 2022 paper, so state that. | ES:L5 |
| L23: "venue not stated on the pages read" (SpecSphere) | The PDF carries an AAAI copyright line and an anonymous code link. Say "AAAI-format preprint; acceptance unconfirmed". | SpecSphere.txt:L56-60 |

### 1.7 MEDIUM: RUNG's discrepancy is computed on node estimates, not raw features. The brief alternates between three names for it.
Quoted (brief:L3) "keyed to a feature residual"; (L7) "normalized feature difference… a feature gap, recovered from the node states being estimated"; (L17, L51) "normalized feature residual".

Problem:
- RUNG's y_ij is the normalized difference between the *current propagated estimates* F^(k) of i and j. It is not the raw-feature difference.
- "Residual" already names λ's role, which is to anchor the estimate to F^(0) (ES:L103). Using "residual" for y_ij conflates the two quantities.
- The memo calls RUNG's W_ij^t the "dynamic structural weight" (iclr2026_conference.tex, around L190). The brief calls the same quantity a "feature" signal, as opposed to a "structural" one. A reader of both documents will be confused.

Proposed fix: use one term everywhere, for example "endpoint-state discrepancy y_ij = ‖f_i/√d_i − f_j/√d_j‖ on the current propagation estimates", and keep "residual" for λ. Add one sentence in the edge section:
> "In a text-attributed graph with sentence-embedding inputs, F^(0) is a learned projection of the text embedding, so RUNG's discrepancy is partly a text signal at the first iteration. This is why BadGraph's semantically aligned edits can move it."

That sentence also makes L43 and L67 less awkward ("The excerpt does not say RUNG's weight is a text similarity").

### 1.8 LOW: Smaller accuracy points
- L11, TAGR, "The repaired graph cannot adapt to an attacker who sees the repair." This reads as our inference (ES:L134 frames it as a consequence of freezing), not an author statement. Rewrite as: "Because the repaired adjacency is fixed before training, it cannot adapt to an attacker who sees the repair (our inference)."
- L11, TAGR Cornell, "0.2270±0.0308 for GCN". This is unusually low for GCN on Cornell; typical reports on Geom-GCN splits are about 0.45–0.55. Flag it as "as reported; low relative to common baselines, so the split or protocol may differ."
- L29, AuditVotes, "clean accuracy improves by 437.1%". Clarify that this is the *relative* change in the smoothed classifier's clean accuracy at the noise level needed for a 20-edge radius. Otherwise readers will read it as an accuracy above 100%.
- L13, GJDNet. The brief reports values on a 0–1 scale; other papers in the brief use percentages. Normalize the units.
- L23. The brief writes ‖ΔA‖₂ ≤ √(2p) as the threat. The paper's threat set is ‖A′−A₀‖₀ ≤ p (ℓ0 flips, eq. 15), and √(2p) is the spectral-norm consequence. Say "p edge flips (ℓ0), which implies ‖ΔA‖₂ ≤ √(2p)". The certificate's Chebyshev bound is stated in terms of a perturbation of the rescaled Laplacian, and how a perturbation in A maps to one in L is not addressed in the paper. A one-clause caveat is enough.

---

## 2. Consistency

### 2.1 MEDIUM: The "no shared leaderboard" claim conflicts with evidence the brief already has.
Quoted (brief:L15): "no 2024–2026 paper in the pass reports one shared leaderboard of RUNG against GPR-GAE, GJDNet, and DSGNN."
That is literally true, but SpecSphere (in the brief) and BadGraph (in the brief) both evaluate RUNG directly. SpecSphere's Table 7 also includes GNNGuard and UnGSL. Add:
> "Two papers in this brief do put RUNG in their own tables: SpecSphere (Chameleon; DropEdge, Metattack, Feature-PGD) and BadGraph (joint text and structure attack)."

### 2.2 MEDIUM: The intro and the conclusion frame different questions.
- The intro (L3) lists findings framed as absences ("A training penalty on how far node states move when edges are deleted was not found").
- The conclusion (L69) says "A design that reweights from structural discrepancy during propagation, penalizes node-state movement under edge deletion, and spends a structural trust budget from frozen text scores is absent."
- Neither one says *why* these three ideas are the yardstick. They are the memo's design (RUNG backbone, v1 W×q, v3 trust budget, propagation penalty). The table at L49-53 uses the three as row labels, but the reader never learns where they come from.
- Fix: state the three design questions in the first paragraph (see 4.1).

### 2.3 LOW: The date window is used inconsistently.
- L3 says "Between 2023 and 2026".
- L21, L33 and L57 restrict to "opened 2024–2026 papers".
- L31 includes a NeurIPS 2022 paper, which the brief does justify.
- Pick one window, for example "2023–2026, with 2022 evaluation and certificate anchors", and use it everywhere. Several of the HIGH gaps above (Jia 2023, Scholten 2023) fall inside the stated window but outside the "opened 2024–2026" filter.

### 2.4 LOW: The "Chen et al." label is ambiguous.
Four different first authors appear as "Chen et al.": ADEdgeDrop (Zhaoliang Chen), DSGNN (Xiaohan Chen), BadGraph (Zihui Chen), and the IJCAI TTA survey (Jiayi Chen). Add a short tag on first mention, for example "Chen, Z. et al. (ADEdgeDrop)".

### 2.5 LOW: STEM-GNN self-reference.
- L27 discusses "STEM-GNN… (Guo et al.)" as an outside paper. It is the paper behind this repo.
- The repo code has gone beyond the paper: `encoder_lipschitz_penalty` and `propagation_sensitivity_penalty` exist (`STEM-GNN/model/encoder.py:474-523`, `STEM-GNN/utils/args.py:158-177`).
- The brief should separate "the published STEM-GNN paper" from "this repository's current code". Otherwise L27's sentence "Node-state movement under edge deletion lies outside that penalty" is out of date for the code.

---

## 3. Coverage gaps

### 3.1 HIGH: Nearest prior art for "penalty on node-state movement under edge perturbation" is missing.
The brief's claim at L3, L33 and L52 that no such training penalty exists needs the nearest neighbors stated, so that a reviewer of the memo cannot say it was missed:
- **GRAND** (Feng et al., NeurIPS 2020). Random propagation with DropNode plus a consistency loss across augmented views, which penalizes prediction movement under random node removal. It is one of the seven defenses Mujkanovic et al. evaluate adaptively (ES:L78), so it is already inside the brief's evidence base. This is the closest relative.
- **IDEA** (Tao et al., arXiv 2305.15792, 2023). Node-based and structure-based invariance objectives so predictions stay stable across attacks (ES:L58). This is in the notes and omitted from the brief.
- **Jia et al. KDD 2023**, **Juvina et al. TMLR 2024** and **NoisyGNN AAAI 2024** (see 1.2).
- Pre-window stability theory that a reviewer will expect: Gama, Bruna and Ribeiro (2020), and Kenlay, Thanou and Dong (ICML 2021), which give stability of graph filters to edge perturbations. A sentence noting these exist is enough.

Proposed added paragraph, at the end of the certificate section:
> "Closest relatives of a node-state-movement penalty: GRAND's consistency loss across DropNode views (2020; evaluated adaptively by Mujkanovic et al.), IDEA's structure-invariance objective (2023), Lipschitz regularizers on GCN/SAGE/GAT weights (Jia et al., KDD 2023; Juvina et al., TMLR 2024), and noise injection (NoisyGNN, AAAI 2024). None penalizes the change in encoder node states between the observed graph and an edge-deleted copy, normalized by a detached state norm, as the repo's `propagation_sensitivity_penalty` does. GRAND is the nearest and should be the baseline for that claim."

### 3.2 MEDIUM: Notes findings omitted from the brief that matter for STEM-GNN's scripts
- **Bazhenov et al. (2023)**, structural distributional shifts by popularity, locality and density splits, with the finding that simpler models often win (SS:L23). This is a ready-made protocol for `degree_shift_ood.py`, and it is absent from the brief.
- **GNNSafe** (Wu et al. 2023), energy-based OOD detection with up to 17.0% AUROC gain (SS:L24). Relevant if STEM-GNN reports OOD detection.
- **ACGMAE** (AAAI 2025), defense against joint structure and feature attacks with a kNN graph and adversarial feature masking (ES:L45). Relevant to `missing_feature.py` and joint corruption.
- **GPR-GAE on OGB-arXiv** (transfer): 69.1 / 49.2 / 42.3 versus GCN 70.8 / 43.1 / 33.1 (ES:L28). arxiv is a STEM-GNN finetune dataset, so this is the most transferable large-graph number in the notes.
- **GPR-GAE's costs**: at most 0.9 points clean drop from adding the purifier, and at most 8.9 points LRBCD drop (ES:L26). Also its point that non-differentiable purifiers (Jaccard, SVD) can only be compared under transfer attacks (ES:L132), which matters for any adaptive comparison.
- **RUNG appendix**: a small γ hurts clean accuracy more than it helps robustness, and γ→∞ recovers ℓ1 behavior that degrades at 20–40% budgets (ES:L103). This is directly relevant to the memo's γ choice.
- **RUNG-GCN variant**: the coupled variant is weaker than decoupled RUNG (ES:L104), which matters if RUNG is placed inside the MoE SAGE encoder rather than as APPNP-style decoupled propagation.
- **Mid-GCN, MHR-GCN, GRN**: 2023 homophily and spectral baselines (ES:L50-57). One sentence each is enough.

### 3.3 MEDIUM: Missing related work beyond the notes
- **Gosch et al., "Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions"** (NeurIPS 2023). This is the standard reference on adaptive adversarial training for GNNs; SpecSphere cites it as evidence that adversarial training overfits one threat model (SpecSphere.txt:L17-19). Absent from the brief.
- **Hierarchical randomized smoothing** (NeurIPS 2023): see 1.3.
- **"Intruding with Words"** (NeurIPS 2024): see 1.6. It is the text-level GIA reference the brief could not name.
- **Local PDF 2211.06605** (Zhao et al., an analysis of oversmoothing as Markov-chain convergence, with a sufficient condition for operator-inconsistent GNNs to avoid it). It could support the oversmoothing sentence at L63 if that thread is kept. It is optional.

---

## 4. Clarity and structure

### 4.1 HIGH: State the purpose and the three yardstick questions up front.
The first paragraph (L3) packs mechanism, numbers, absences and retrieval failures (HTTP 429) into one 300-word block.

Proposed opening:
> "**Purpose.** This brief checks prior art for three design choices in the STEM-GNN robustness memo: (Q1) reweighting edges during propagation from an endpoint-state discrepancy (RUNG-style); (Q2) penalizing how far encoder node states move when edges are deleted; (Q3) allocating a per-node structural trust budget from frozen text scores. **Short answers.** Q1: RUNG does this, keyed to node-estimate discrepancy; GNNGuard does a cosine version on embeddings. Q2: no paper read penalizes encoder state movement under edge deletion directly; GRAND (consistency under DropNode), IDEA (invariance) and Lipschitz regularizers (Jia et al. 2023; Juvina et al. 2024) are the nearest. Q3: no paper read allocates a per-node budget; semantic defenses use hard cosine or LLM gates, and GNNGuard on text embeddings is the nearest continuous relative. **Caveat.** These are findings from a bounded search (see Method and limits)."

### 4.2 MEDIUM: Move provenance caveats into one "Method and limits" section.
Sentences such as "was not extracted", "was not opened", "HTTP 429", "standard deviations… not in the extract", and "OpenAlex's author string 'Bang Ye Wu' does not match…" appear about 40 times (for example L9, L13, L15, L23, L27, L33, L41, L43, L45, L47, L57, L59, L61, L63, L69). Keep at most one inline caveat per paper, where it changes the interpretation of a number. Move the rest into a closing section with three short lists: databases and dates; papers not opened; resolved items (see 1.6).

### 4.3 MEDIUM: Add a protocol column to every number.
The brief rightly warns that "Adaptive local PGD, PRBCD, random deletion, and a min-max rate are different experiments" (L15), but the numbers are still embedded in prose. Proposed compact table:

| Method | Dataset | Threat (adaptive? budget) | Defense / undefended / best baseline |
|---|---|---|---|
| RUNG | Cora-ML | adaptive local PGD, 200% of degree | 69.3 / 0.0 (GCN) / 40.7 (TWIRLS-T) |
| RUNG | Cora-ML | adaptive global PGD, 40% | 65.1 / 48.4 / 58.7 (TWIRLS) |
| GPR-GAE+GCN | Cora | adaptive PRBCD ε=0.5 | 61.3 / – / 48.2 (SoftMedianGDC) |
| GPR-GAE+GCN | OGB-arXiv | transfer PRBCD ε=0.5 | 42.3 / 33.1 / – |
| TAGR-GCN | Cora-ML | random deletion 50% | 78.4 / – / 78.1 (Gaussian-only) |
| GJDNet | Cora | "min-max", rate 0.25 (adaptivity unverified) | 77.7 / 73.9 / 74.5 (RGCN) |
| SpecSphere+ | Chameleon | DropEdge 20% / Metattack 5% / Feature-PGD 0.1 | 46.4 / 41.9 / 45.1 vs RUNG 45.6 / 42.8 / 44.1 |
| AuditVotes | Citeseer | certified, ≤20 inserted edges | 72.6 certified vs 14.7 vanilla smoothing |

### 4.4 LOW: Other wording fixes
- The title "Papers harden GNNs with separate mechanisms" is vague. Suggested: "GNN robustness 2023–2026: prior-art check for state-discrepancy reweighting, edge-deletion stability, and text trust budgets".
- L17 is a paragraph of eight loosely linked claims (RUNG, TAGR, GNNGuard, heterophily survey, ES-GNN). Split it into "Q1 verdict" (RUNG, TAGR, GNNGuard) and "Heterophily-side membership rules" (Zheng taxonomy, ES-GNN).
- L27 is the longest paragraph (STEM-GNN, Sun and Lin, Liang et al., Schmidt and Mateos). Split STEM-GNN into its own paragraph, given its importance to the repo.
- L57 to L63 (the shift and test-time section) reads like a catalogue. Open it with: "For STEM-GNN's shift scripts, the usable protocols are Bazhenov et al.'s structural splits and degree-bucketed reporting (Subramonian et al.); the usable mechanisms are GraphPatcher and Grace (degree), T2-GNN and MDS-GNN (missing features), and SETTA and T3R (test-time, frozen backbone)."

---

## 5. Relevance to STEM-GNN: missing actionable implications

The brief contains no "what this means for the repo" section. Proposed additions, each tied to evidence:

1. **HIGH: State that the repo's current robustness evidence comes from random corruption.** `random_edge_drop.py`, `missing_feature.py`, `degree_shift_ood.py` and `homophily_shift_ood.py` are random or shift protocols. Mujkanovic et al. (brief:L15) and Hou et al. show that non-adaptive evaluations inflate robustness. Any adversarial claim needs an adaptive attack (PGD or PRBCD on the full finetune pipeline), or must be scoped explicitly to random corruption.

2. **HIGH: Watch for gradient masking from VQ.** STEM-GNN routes the encoder output through a vector-quantized codebook with frozen VQ at finetune (`--freeze_vq 1`, CLAUDE.md). The quantization is non-differentiable, so naive white-box PGD can under-report vulnerability. Adaptive attacks should use a straight-through or BPDA surrogate through the VQ, as the Mujkanovic et al. methodology requires. Otherwise robustness numbers may come from obfuscated gradients. This inference is not in the brief, but it follows from L15.

3. **HIGH: Baselines must share STEM-GNN's text encoder.** TGRB shows GNNGuard becomes "top-tier" once features are advanced text embeddings (brief:L41), and LLM4RGNN shows that encoder choice alone moves robustness (brief:L39). STEM-GNN uses 768-dim sentence embeddings, so GNNGuard, Guardual and RUNG should run on the *same* embeddings. Include PGD-Guard (cosine-constrained edits, brief:L45) as the defense-aware attack against any cosine or semantic gate.

4. **MEDIUM: Position the repo's `propagation_sensitivity_penalty` against GRAND and IDEA.** The penalty (`STEM-GNN/model/encoder.py:494-523`, `--propagation_lip_coeff`, `--propagation_drop_prob 0.3`) is exactly the Q2 mechanism the brief says is absent. The brief should name it, say it is novel relative to the opened papers, and recommend GRAND-style consistency regularization as the ablation baseline. Note that the repo's `encoder_lipschitz_penalty` is a Frobenius-squared sum over conv weights. That amounts to targeted weight decay: it bounds each layer's spectral norm only loosely and does not control the product across layers. Jia et al. (KDD 2023) give tighter GCN, SAGE and GAT bounds that could replace it.

5. **MEDIUM: Reuse published protocols in the existing scripts.**
   - `random_edge_drop.py`: SpecSphere uses DropEdge 20% as a perturbation (SpecSphere.txt:L378-395), and TAGR uses 25% and 50% random deletion on Cora-ML. Add those rates so STEM-GNN numbers line up with published tables.
   - `degree_shift_ood.py`: adopt Bazhenov et al.'s popularity, locality and density splits, and report accuracy by degree bucket. Subramonian et al. show high-degree nodes have lower error regardless of training (brief:L59), so aggregate accuracy hides the effect.
   - `homophily_shift_ood.py`: the brief notes there is no "homophily shift" literature (L59). Say plainly that the repo's protocol is novel and must be defined precisely. Cite GJDNet's assortativity-inversion attacks as the nearest adversarial analogue.
   - `missing_feature.py`: T2-GNN and MDS-GNN are the comparison mechanisms, and ACGMAE covers joint feature and structure attack.

6. **MEDIUM: Use arxiv as the scale test.** GPR-GAE's OGB-arXiv transfer numbers and LLM4RGNN's OGBN-Arxiv caching are the two large-graph references in the brief. STEM-GNN finetunes on arxiv, so report at least a transfer attack there. Lee and Park note adaptive attacks are "unscalable" on OGB-arXiv.

7. **LOW: Check γ scale against the RUNG grid.** The memo uses γ=36 with u_ij = √(d_ij+ε), where d_ij = √Z_ij (iclr2026_conference.tex, MCP paragraph). The RUNG paper tunes γ ∈ {0.5, 1, 2, 3, 5} on its own normalization (brief:L7). The brief should not imply the grids are comparable. The RUNG appendix finding that a small γ hurts clean accuracy and γ→∞ degrades to ℓ1 (ES:L103) is the relevant guidance.

8. **LOW: The BadGraph warning applies to v1 and v3.** BadGraph's over-50% drop on RUNG under SBERT (brief:L45) is the attack most aligned with a RUNG-plus-text-score design. The repo's BM25 candidate-pool attacker (memo L90-109) is a lexical version of PGD-Guard. The brief's L45 statement that "a strict threat model that chooses the injected edge because… a BM25 score is high, was not found" should be connected to the memo explicitly as a novelty point for the threat model.

---

## Prioritized top 10

1. **HIGH (1.1)**: GNNGuard is a continuous, neighborhood-normalized cosine reweighting inside message passing, not a gate. On text-embedding features it is the nearest prior art to "structural weight × text score". Rewrite L17, L37 and L53.
2. **HIGH (1.4)**: SpecSphere's certificate is a first-order linearization with absorbed constants. β is assumed, not enforced. No certified accuracy is reported. The bound is likely vacuous at p=0.1|E|. The paper does contain a RUNG and GNNGuard head-to-head. Soften L3 and rewrite L23.
3. **HIGH (1.2, 3.1)**: The "not found" claims on Lipschitz training and node-state penalties miss Jia et al. (KDD 2023), Juvina et al. (TMLR 2024), NoisyGNN (AAAI 2024), GRAND (consistency under DropNode, already in Mujkanovic's evaluated set) and IDEA (in the notes).
4. **HIGH (1.3)**: "Continuous feature certification not found for graphs" is contradicted by hierarchical randomized smoothing (NeurIPS 2023).
5. **HIGH (4.1, 2.2)**: State the purpose and the three yardstick questions (Q1–Q3) up front, with short answers. The negatives currently read as unexplained novelty claims.
6. **HIGH (5.1–5.3)**: Add a STEM-GNN implications section. Current scripts are random-corruption only. VQ can mask gradients in adaptive attacks. Baselines (GNNGuard, Guardual, RUNG) must use the same 768-dim text encoder, with a PGD-Guard-style attack.
7. **MEDIUM (1.5)**: RUNG numbers omit TWIRLS-T (40.7 at 200%) and TWIRLS (58.7 at global 40%). "The order holds" is wrong, and RGCN is below undefended GCN.
8. **MEDIUM (1.6)**: Resolve the open bibliographic items: TGRB's full authors, "Intruding with Words" (2405.16405, NeurIPS 2024), STEM-GNN v2 "nine", ADEdgeDrop TKDE 2025 (conflicts with SS:L172), Mujkanovic NeurIPS 2022, and SpecSphere's AAAI template.
9. **MEDIUM (1.7, 2.5)**: Use one term for RUNG's discrepancy (current-state, not raw-feature; not "residual"), and align it with the memo's "structural weight" wording. Separate the STEM-GNN paper (head-only penalty) from the repo code (encoder and propagation penalties already implemented).
10. **MEDIUM (3.2, 4.2, 4.3)**: Restore the omitted notes findings that matter for the scripts: Bazhenov et al.'s structural-shift splits, GPR-GAE's OGB-arXiv numbers and costs, the RUNG γ guidance, and ACGMAE. Move the about 40 provenance caveats into one "Method and limits" section. Add a protocol-labelled results table.
