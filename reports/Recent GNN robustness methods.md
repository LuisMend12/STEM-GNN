# GNN robustness 2023–2026: prior-art check for state-discrepancy reweighting, edge-deletion stability, and text trust budgets

**Purpose.** This brief checks prior art for the three design choices in the internal robustness memo, *Semantic-Structural Robust Propagation for GNNs* (`STEM-GNN/LLM4GNN/ICLR/iclr2026_conference.tex`):

- **Q1.** Reweight edges during propagation from an endpoint-state discrepancy (RUNG-style).
- **Q2.** Penalize how far encoder node states move when edges are deleted.
- **Q3.** Allocate a per-node structural trust budget from frozen text scores.

The negative findings below are the result of a bounded literature pass. They are not field-wide novelty claims. The sources and their limits are listed in [Method and limits](#method-and-limits).

**Short answers.**

- **Q1: published.** RUNG reweights edges inside unrolled aggregation from the discrepancy between the current node estimates of the two endpoints ([Hou et al.](https://arxiv.org/abs/2311.14934)). GNNGuard does the same with a neighborhood-normalized embedding cosine. TAGR puts explicitly structural scores (Jaccard, common neighbors, degree gap) into a *frozen preprocessing* step instead ([Goel and Kanniainen](https://arxiv.org/abs/2606.03462)).
- **Q2: no exact match among the papers opened, but close relatives exist.** These include GRAND's consistency loss across DropNode views, IDEA's structure-invariance objective, and Lipschitz regularizers for GCN, SAGE and GAT (Jia et al., KDD 2023; Juvina et al., TMLR 2024). **This repository already implements the Q2 mechanism** as `propagation_sensitivity_penalty` ([encoder.py:491-523](../STEM-GNN/model/encoder.py#L491-L523)). The open task is therefore an evaluation against GRAND and IDEA, not new implementation work.
- **Q3: no paper read allocates a constrained per-node budget from text scores.** Semantic defenses mostly threshold cached embedding cosines or LLM judgments ([Lei et al.](https://arxiv.org/html/2510.17185); [Ma, Wang, and Yan](https://arxiv.org/html/2603.26105)). GNNGuard on text-embedding features is the nearest continuous relative. Attacks already target both gates: PGD-Guard, and BadGraph's attack on RUNG.

The certified-robustness literature is a separate track. Most certificates count discrete edits (edges, injected nodes, or whole-node feature rewrites). **SpecSphere** claims a joint edge-flip and continuous ℓ∞ feature bound derived from the Lipschitz constants of its propagators ([Choi, Choi, and Kim](https://arxiv.org/abs/2505.08320)). Its supplied proof, however, is a first-order bound with a gap (see the certificate section), and it reports no certified accuracy.

## Q1: propagation-time edge reweighting

### RUNG: endpoint-state discrepancy inside unrolled aggregation

**RUNG** treats message passing as robust mean estimation. The authors describe an ℓ1 penalty as biased once outliers accumulate, so they replace it with a minimax concave penalty (MCP). They then unroll a quasi-Newton iteratively reweighted least-squares (QN-IRLS) solver as aggregation layers on an APPNP-style decoupled stack.

**Edge weight.** Let y_ij = ‖f_i/√d_i − f_j/√d_j‖ be the degree-normalized difference between the *current propagation estimates* of the two endpoints. It is not the raw-feature difference. For a non-self edge, the IRLS weight is the derivative of the penalty with respect to y², W_ij = dρ(y)/d(y²) = max(0, 1/(2y) − 1/(2γ)). The weight is therefore zero once y ≥ γ, which prunes the edge, and decays continuously below that threshold.

**Anchor term.** A separate weight λ controls how strongly the estimate stays anchored to F⁽⁰⁾. In this brief "residual" refers only to that anchor term. The notes record it through the reparameterization λ̂ = 1/(1+λ), tuned in {0.7, 0.8, 0.9}; γ is tuned in {0.5, 1, 2, 3, 5}. The γ/λ split is the nearest published version of "which neighbor" versus "how much influence."

**What RUNG's discrepancy measures on a text-attributed graph.** When the inputs are sentence embeddings, F⁽⁰⁾ is a learned projection of the text embedding. At early iterations the discrepancy is therefore partly a text signal. This is why BadGraph's semantically aligned edits can move it (see Q3).

**Hyperparameter guidance from the RUNG appendix.** A small γ costs more clean accuracy than it gains in robustness. As γ→∞ the method recovers ℓ1 behavior, which degrades at 20–40% budgets.

**Architecture caveat.** A coupled RUNG-GCN variant (robust weights W multiplying a learned feature map) is weaker than decoupled RUNG on Cora-ML. The published gain should therefore not be attributed to the edge-weight formula alone. It may not transfer unchanged to STEM-GNN's SAGE or MoE encoder.

Crossref indexes a proceedings version as NeurIPS 37 (2024) ([Hou et al.](https://arxiv.org/pdf/2311.14934)).

**Headline numbers** (Cora-ML; the protocol is given with each set):

- **Adaptive local evasion, accuracy at 200% of target degree.**
  - RUNG: **69.3±9.0** (clean 84.0±5.3; 100% of degree 70.7±10.6).
  - Strongest baseline: TWIRLS-T at **40.7±11.8**.
  - Other baselines: TWIRLS **12.0±6.9**, GARNET **9.6±3.5**, SoftMedian **8.7±3.4**, GNNGuard **2.0±2.7**, EvenNet **1.1±1.3**, Jaccard-GCN **0.7±1.3**. ProGNN, RGCN and undefended GCN are all at **0.0**.
  - Standard deviations near ±10 reflect a small set of target nodes.
- **Adaptive global attack, 40% edge budget.**
  - RUNG **65.1±1.2**, TWIRLS **58.7±1.1**, SoftMedian **58.1±0.7**, ProGNN **51.3±0.6**, GNNGuard **51.0±1.2**, Jaccard-GCN **49.1±2.4**, GCN **48.4±0.5**, RGCN **46.1±0.7**.
  - RUNG still leads, but the margin shrinks and the baselines change order: ProGNN moves above GNNGuard, and RGCN falls below undefended GCN.
- **Poisoning, 40% budget.** RUNG **64.6±1.3** versus GCN **48.7±1.0**.
- **Citeseer graph injection** (appendix). RUNG-MCP falls from **75.65** to **51.13**, and GNNGuard from **73.47** to **34.73**. Standard deviations were not extracted.

**Other details.**

- Hou et al. group SoftMedian, TWIRLS and ElasticGNN as the ℓ1 family whose bias RUNG is built to remove.
- RUNG's complexity is O((m+n)kd) with a diagonal Hessian approximation.
- The extracted experiments cover Cora-ML and Citeseer only.

### GNNGuard: a continuous cosine weight, not only a gate

**GNNGuard** (Zhang and Zitnik, NeurIPS 2020) works layer by layer:

1. It computes the cosine similarity of the two endpoint embeddings.
2. It normalizes those similarities within each node's neighborhood to get importance weights.
3. It prunes edges through a learned transform of the normalized pair scores against a threshold.
4. It uses the surviving normalized weights as **continuous message weights**, with a layer-wise memory term.

The paper describes this as pruning "likely fake edges" and assigning "less weight to suspicious edges" ([arXiv:2006.08149](https://arxiv.org/abs/2006.08149)). Mujkanovic et al. restate it in their appendix as a 0.1 cosine cutoff with a degree-like normalizer.

When the input features are frozen text embeddings, as in TGRB's re-evaluation, GNNGuard's first-layer edge weight is a neighborhood-normalized text-embedding cosine multiplied into the adjacency. That makes it the closest published relative of a structural-weight × text-score rule. It still has no separate structural-discrepancy term and no per-node budget.

### Frozen repair: TAGR, GPR-GAE, GJDNet

**TAGR** (Goel and Kanniainen, arXiv 2 June 2026; no venue on the PDF) is the method whose score is explicitly structural. Before training, it builds a sparse feature-Gaussian neighborhood and scores each observed edge:

q_ij = 1.50 z(cosine) + 0.50 z(Jaccard) + 0.25 z(log(1+M_ij)) + 0.10 z(κ_i+κ_j) − 0.10 z(|d_i−d_j|)

The edge multiplier is not q itself. TAGR standardizes q again and uses r_ij = clip(1 + λ·tanh(stdz(q)_ij), 0.5, 1.5). It then freezes A⊙R + A_GFN, where A_GFN holds the Gaussian repair edges, for an unmodified GNN.

**Results and limits.**

- Feature cosine has the largest coefficient.
- At λ=0 the method keeps the observed edges unweighted and adds the Gaussian repair edges. The authors call this repair the main gain. The residual weighting adds only **0.7 points** at 25% deletion and **0.3 points** at 50% deletion on Cora-ML.
- On clean Cornell, TAGR-GCN reaches **0.8203±0.0581** against **0.2270±0.0308** for GCN. That GCN figure is low relative to commonly reported Cornell baselines, so the split or protocol may differ.
- The extracted threat model is random edge addition and deletion. Because the repaired graph is fixed before training, an attacker who sees the repair may be able to exploit it. That is our inference; the inspected experiments neither test nor establish defense-aware robustness ([Goel and Kanniainen](https://arxiv.org/pdf/2606.03462)).

**GPR-GAE** (Lee and Park, arXiv 2026; no venue printed). A self-supervised graph autoencoder with several Generalized PageRank filters purifies the adjacency in multiple steps, and the classifier then trains on the cleaned graph. Trust is the reconstructed edge set, and the hop mix is a separately learned coefficient. Training corruption uses edge drops, additions and reweighting at a budget of (1.5, 0.2, 3), with K=7.

| Setting | Attack | GPR-GAE result | Comparison |
| --- | --- | --- | --- |
| Cora, GPR-GAE + vanilla GCN | adaptive PRBCD, ε=0.5 | **61.3±3.6** (clean **79.4±0.3**) | — |
| Cora, GPR-GAE + vanilla GPRGNN | adaptive PRBCD, ε=0.5 | **15.7 points** higher | adversarially trained GPRGNN at **48.5%** |
| Citeseer, clean / ε=0.25 / ε=0.5 | transfer | **71.2 / 68.7 / 66.4** | vanilla GCN **71.5 / 40.0 / 25.3** |
| OGB-arXiv, clean / ε=0.25 / ε=0.5 | transfer only | **69.1 / 49.2 / 42.3** | vanilla GCN **70.8 / 43.1 / 33.1** |

**Costs and caveats.**

- Adding the purifier costs at most **0.9 points** of clean accuracy, and the largest stated LRBCD drop is **8.9 points**.
- Adaptive gradients are described as unscalable on OGB-arXiv, so that graph is evaluated with transfer attacks only.
- Non-differentiable purifiers (Jaccard, SVD) can only be compared under transfer attacks ([Lee and Park](https://arxiv.org/pdf/2605.23239)).

**GJDNet** (Cui et al., arXiv 1 June 2026; peer-reviewed venue unverified) filters neighbors that are skewness outliers in feature similarity, and separately tightens class regions with a spherical decision boundary. On its Cora "min-max" table, accuracy at perturbation rates 0 / 0.10 / 0.25 is **85.6 / 83.3 / 77.7%**, against GCN-Jaccard **74.2%** and RGCN **74.5%** at 0.25. Whether that attack is adaptive in Mujkanovic et al.'s sense was not checked. Rejection-perturbation coverage under a 5% min-max attack is weakest on Actor (**45.91%**) ([Cui et al.](https://arxiv.org/pdf/2606.01560)).

### Thinner entries

- **ADEdgeDrop** (Chen, Z. et al.; IEEE TKDE 2025, DOI 10.1109/tkde.2025.3586369) trains an adversarial predictor on the line graph to choose edges to drop. Its table flags one-day timeouts and out-of-memory failures at 16 GB ([arXiv](https://arxiv.org/pdf/2403.09171)).
- **GRANCE** (AAAI 2025) refines neighborhoods with dual contrastive learning and a neighbor estimator ([Zhuang et al.](https://doi.org/10.1609/aaai.v39i12.33471)).
- **Pure-GNN** keeps every edge and learns importance weights plus a residual ([Yang et al.](https://doi.org/10.26599/tst.2025.9010034)).
- **DSGNN** (Chen, X. et al.) runs one channel for structural noise and one for feature noise. Its best abstract averages are **81.24%** on Cora, **71.94%** on CiteSeer and **81.66%** on Industry ([DOI](https://doi.org/10.32604/cmc.2025.067284)).
- **ACGMAE** (AAAI 2025) defends against joint structure and feature attacks with a kNN graph contrast and adversarial feature masking ([Shen, Shen, and Pan](https://doi.org/10.1609/aaai.v39i12.33364)).
- **Mid-GCN** (WWW 2023) replaces purification with a robust mid-pass spectral filter ([Huang et al.](https://doi.org/10.1145/3543507.3583335)).
- **BDP** prunes low-Jaccard edges on bipartite graphs ([Deng et al.](https://doi.org/10.1109/ijcnn60899.2024.10650844)).
- **Tong et al.** score injected edges with link prediction and a homogeneity filter ([Tong et al.](https://doi.org/10.1109/dsc63484.2024.00031)).
- **ES-GNN** (IEEE TPAMI 2024) splits edges into two complementary subgraphs ([Guo et al.](https://doi.org/10.1109/tpami.2024.3459932)).
- The heterophily survey's taxonomy has a slot for "identifiable message aggregation" (edge, feature, or hybrid weights) ([Zheng et al.](https://ar5iv.labs.arxiv.org/html/2202.07082)).

Edge splitting and hard Jaccard cutoffs are membership rules, not continuous reweighting.

### Evaluation protocols are not interchangeable

Adaptive local PGD, PRBCD, random deletion, and a "min-max" rate are different experiments. Mujkanovic, Geisler, Günnemann and Bojchevski (NeurIPS 2022) evaluated seven defenses: Jaccard-GCN, SVD-GCN, ProGNN, GRAND, GNNGuard, RGCN and Soft-Median-GDC. They report three findings:

- Essentially none of 49 robust-GNN papers at major venues used adaptive attacks.
- Non-adaptive attacks make the same defenses look robust.
- On Cora-ML, about half of the defenses do worse than an undefended GCN under the strongest global attack ([Mujkanovic et al.](https://arxiv.org/pdf/2301.13694)).

Gosch et al. (NeurIPS 2023), *Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions*, is the standard companion reference on adaptive adversarial training.

No paper reports a single leaderboard across RUNG, GPR-GAE, GJDNet and DSGNN. Two papers in this brief do, however, place RUNG in their own tables:

- **SpecSphere**, on Chameleon. SpecSphere+ scores **46.4 / 41.9 / 45.1** under DropEdge 20% / Metattack 5% / Feature-PGD 0.1, against RUNG **45.6 / 42.8 / 44.1** and GNNGuard **45.0 / 41.3 / 43.7**. RUNG leads under Metattack.
- **BadGraph**, under a joint text and structure attack (see Q3).

## Q2: stability of node states under edge deletion

### Nearest relatives of a node-state-movement penalty

No paper that was opened penalizes the change in encoder node states between the observed graph and an edge-deleted copy. The nearest relatives are below. The last four (Jia et al., Juvina et al., NoisyGNN, and the stability theory) were added during the 17 October review and have not been read in full.

- **GRAND** (Feng et al., NeurIPS 2020). Random propagation with DropNode, plus a consistency loss across augmented views. This penalizes *prediction* movement under random node removal. It is the closest relative, and it is already among the seven defenses Mujkanovic et al. evaluated adaptively.
- **IDEA** (Tao et al., arXiv:2305.15792, 2023). Node-based and structure-based invariance objectives that keep predictions stable across attacks.
- **Jia et al.** (KDD 2023, DOI 10.1145/3580305.3599335). Derive Lipschitz upper bounds for GCN, GraphSAGE and GAT, and use them as a training regularizer.
- **Juvina et al.** (TMLR 2024). Train under a tight Lipschitz constraint against feature attacks.
- **NoisyGNN** (Ennadir et al., AAAI 2024, arXiv:2402.13987). Noise injection inside the architecture, with a theoretical link to robustness.
- **Stability theory for graph filters.** Gama, Bruna and Ribeiro (2020) and Kenlay, Thanou and Dong (ICML 2021) bound how graph filters respond to edge perturbations. Both are analyses, not training penalties.

The two Lipschitz papers (Jia et al. and Juvina et al.) bound sensitivity to features, not to edge deletion. The zero-hit arXiv phrase queries in the original pass (`"spectral normalization" AND "graph neural"` and others) were therefore a limit of those queries, not evidence that such work is absent.

The closest training penalty on graph-perturbation sensitivity *outside* node classification is Schmidt and Mateos. They penalize the spatial Lipschitz constant of a GNN-parameterized flow so that relative graph perturbations move the generated distribution less ([arXiv 8 July 2026](https://arxiv.org/abs/2607.07510)).

### The STEM-GNN paper versus this repository

**The published paper.** STEM-GNN, "Generalizing GNNs with Tokenized Mixture of Experts" (Guo et al.), is v2 on arXiv, revised 12 June 2026, and was accepted to KDD 2026. The abstract reports nine node, link and graph benchmarks.

- It bounds the Lipschitz constant of the *prediction head* through a Frobenius penalty, written as the stability constraint L_h η_θ ρ ≤ ε.
- The theory's admissible set includes a feature budget ρ_x and a structure budget ρ_s.
- Theorem 3.1 says a Lipschitz stability constraint caps reliance on the high-frequency filter component and leaves a nonzero worst-environment floor.
- The paper does not issue a radius inside which the prediction is constant ([Guo et al.](https://arxiv.org/abs/2602.09258)).

**This repository has gone further than the paper.** It adds two optional encoder regularizers:

- **`encoder_lip_coeff`** ([encoder.py:474-489](../STEM-GNN/model/encoder.py#L474-L489)) applies a squared-Frobenius penalty to the conv and expert weights. This is targeted weight decay. It loosely bounds each layer's norm and does not control the product across layers. The bounds of Jia et al. for GCN, SAGE and GAT are a tighter alternative.
- **`propagation_lip_coeff` / `propagation_drop_prob`** ([encoder.py:491-523](../STEM-GNN/model/encoder.py#L491-L523)) control the sensitivity penalty. It drops a fraction of undirected edges (default 0.3) with dropout disabled. It then penalizes the change in node states, measured as ‖clean − perturbed‖ divided by the detached norm of the clean state. Node training calls it in both the full-batch and the sampled paths ([node.py:66](../STEM-GNN/task/node.py#L66), [node.py:117](../STEM-GNN/task/node.py#L117)).

That sensitivity penalty is exactly the Q2 mechanism. Both regularizers are empirical. Neither is an end-to-end certificate.

**Penalizing node states does not make the VQ codes or routes stable.** The sensitivity term measures encoder states *before* vector quantization, and the head reads the VQ codes ([ft_model.py:140-152](../STEM-GNN/model/ft_model.py#L140-L152)). A small move in encoder state can still flip a code assignment or an MoE route. Any guarantee would need three ingredients: the distance from each state to the nearest quantizer boundary, control of the routing, and a classification margin.

### Analytic and architectural Lipschitz work

- **FSW-GNN** (LoG 2025) makes message-passing embeddings bi-Lipschitz with respect to the doubly stochastic metric and Tree Mover's Distance, using a Fourier sliced-Wasserstein aggregator. This addresses separation stability (oversmoothing and oversquashing). It is not an adversarial certificate ([Sverdlov et al.](https://arxiv.org/abs/2410.09118)).
- **Davidson and Dym** (arXiv 2024) show uniform upper Lipschitzness for parametric MPNNs and introduce **SortMPNN** ([arXiv](https://arxiv.org/abs/2406.06984)).
- **mHC-GNN** (Mishra, arXiv 2026) projects stream-mixing matrices onto doubly stochastic matrices with Sinkhorn–Knopp. Without that projection the model collapses, for example to **18.20±0.00** on Chameleon ([Mishra](https://arxiv.org/abs/2601.02451)).
- **Sun and Lin** (arXiv 2024) put ‖P_G‖₂ inside a robust PAC-Bayesian generalization gap. They note that the bound has little effect on graph edge attacks and does not extend to node classification ([arXiv](https://arxiv.org/abs/2402.04038)).
- **Liang et al.** (arXiv 2026) tighten a robust PAC-Bayesian margin bound for MPNNs. This is an analysis bound, not a training constraint ([arXiv](https://arxiv.org/abs/2606.06293)).

## Certified robustness: mostly counted edits, plus one continuous claim

### SpecSphere: a joint bound with caveats

**SpecSphere** (Choi, Choi, and Kim; arXiv 13 May 2025, v3 1 August 2025; AAAI-format PDF, acceptance unconfirmed).

**Threat model.** p edge flips (ℓ0, which implies ‖ΔA‖₂ ≤ √(2p)) jointly with an ℓ∞ feature perturbation.

**Bound construction.**

- The spectral branch uses a Chebyshev expansion on a rescaled Laplacian, which gives a branch bound of (2^{K+1}−1)‖c‖₁.
- The spatial branch *assumes* that each attention layer and weight matrix has spectral norm at most β.
- A gate composes the two branches.

**Training.** The loss is cross-entropy plus λ_adv times three PGD inner maximizations: edges on the spectral branch, features on the spatial branch, and both on the fused output. A *single* weight λ_cons covers four further terms: the low-pass regularizer, the high-pass regularizer, the consistency loss, and a margin-based complementarity loss. Reported optima are λ_adv=0.5 and λ_cons=10⁻² on Cora, and λ_adv=1.0 and λ_cons=10⁻¹ on Chameleon. λ_cons below 10⁻³ fails to align the branches, and λ_cons ≥ 1.0 blocks adaptive fusion.

**Caveats found by the two reviews** (from the local PDF `research_notes/2505.08320v3.pdf`):

1. **The proof has a gap.** Appendix F, Eq. 62, bounds the *clean* disagreement between the branches, ‖Z_spec − Z_spat‖, by the size of the perturbation. That would force the two branches to agree exactly at zero perturbation, and the preceding triangle inequality does not establish this. The final step keeps "only linear terms… since higher-order terms are dominated for small budgets" and absorbs unspecified constants. The result is a first-order, small-budget bound, not a sound worst-case certificate. Treat it as an author claim pending a corrected proof.
2. **β is assumed, not enforced.** The implementation section lists weight decay and no spectral normalization or projection.
3. **No certified accuracy is reported.** Tables 2 and 7 report empirical accuracy under DropEdge 20%, Metattack 5% and Feature-PGD (ε=0.1) on Cora and Chameleon.
4. **The bound is likely vacuous at the training budget** (our inference). At p = 0.1|E|, √(2p) is about 30 on Cora, so the bound almost certainly exceeds any softmax margin.
5. **Clean accuracy differs across tables.** Clean Chameleon accuracy is 61.4% in Appendix Table 5 but 46–48% in Tables 2 and 7, so do not compare numbers across those tables.

### Counted-edit certificates

- **AuditVotes** (Lai et al.; CCS 2026) issues a randomized-smoothing radius in counted edge edits. It majority-votes over randomized graphs, rewires each draw with Jaccard, a feature autoencoder or multi-head feature similarity, and drops low-confidence votes.
  - On Citeseer, with edge insertions up to 20, SimAug lifts clean accuracy from **14.7%** to **70.9%**. With the confidence filter added, certified accuracy rises from **14.7%** to **72.6%**.
  - On Cora-ML, clean accuracy improves by **437.1%** and certified accuracy by **409.3%**. Both are *relative* changes of the smoothed classifier at the noise level needed for a 20-edge radius; they are not accuracies above 100%.
  - The authors say that raising edge noise to enlarge the radius destroys clean accuracy ([Lai et al.](https://arxiv.org/abs/2503.22998)).
- **Collective certificate** (Lai et al., arXiv 2024). Uses node-aware bi-smoothing and a linear-program relaxation. On Citeseer the certified ratio rises from **0.0%** to **81.2%** when injected nodes make up 5% of the graph ([arXiv](https://arxiv.org/abs/2403.01423)).
- **AGNNCert** (Li and Wang; USENIX Security 2025). Certifies a deterministic count of manipulated nodes, feature-manipulated nodes and edges, using hash partitioning and majority vote. A node-centric directed partition avoids certificates dominated by high-degree injected nodes ([arXiv](https://arxiv.org/abs/2502.00765)).
- **PGNNCert** (CVPR 2025). The same construction for training-time poisoning ([arXiv](https://arxiv.org/abs/2503.18503)).
- **Message-interception smoothing** (Scholten et al., NeurIPS 2022). The attacker arbitrarily rewrites the features of at most ρ whole nodes. The defender's smoothing noise deletes edges and ablates node features at random, so edge deletion is the defender's noise, not the threat. Certificates on Cora-ML take **17 s**, versus up to **25 min** for Bojchevski et al. (2020) ([arXiv](https://arxiv.org/abs/2301.02039)).
- **Hierarchical randomized smoothing** (Scholten et al., NeurIPS 2023; arXiv:2310.16221; added in review). Adds noise to a random subset of nodes. Its continuous-noise variants certify *continuous* feature perturbations for node classification. The original pass's "continuous feature certification not found for graphs" therefore reflects that pass, not the field.
- **Graph adversarial immunization** (Tao et al., arXiv:2302.08051). Another certification-oriented intervention: it immunizes selected edges or nodes. Its construction was not inspected.

**What would count as a feature certificate.** A certified radius normally comes from a decision margin divided by an end-to-end sensitivity bound. A local Lipschitz constant of the message function alone does not certify the full classifier.

**Not opened:** GNNCert (Xia et al. 2024) and RES-GRACE (Lin et al. 2023).

## Q3: text scores and trust budgets

### What "trust budget" would have to mean

A product of scores is not automatically a budget. A Q3 design must state what is constrained, for example the total normalized incoming weight per node, and how semantic scores allocate it. It must also say whether the weights apply to complete messages, including the edge attributes that STEM-GNN adds before aggregation ([encoder.py](../STEM-GNN/model/encoder.py)), and how self and residual contributions are handled. That definition belongs to the design, not to the literature.

### Semantic defenses mostly threshold

The inspected text-defense sections emphasize thresholding. Continuous coefficients can coexist with hard thresholds (GNNGuard), and this pass did not verify whether Guardual inherits GNNGuard's continuous similarity weights. The narrower open question is whether frozen text scores allocate an *explicitly constrained* structural budget during propagation. No paper read here does that.

**LLM4RGNN** (Zhang et al.; KDD 2025).

- **A stronger text encoder alone is not a topology defense.** Under Mettack at 20% on Cora and PubMed, six LLM or LM feature setups lose up to **37.9%** of accuracy (**23.1%** on average). Vanilla GCN loses up to **39.1%** (**35.5%** on average).
- **The repair pipeline.** GPT-4 labels **26,518** edges on a clean graph. A local LLM (Mistral-7B or Llama3-8B) is LoRA-tuned on those labels, and they are then distilled into an edge MLP. The MLP adds an edge when its probability exceeds γ ∈ {0.91,…,0.99}. Cosine similarity only builds a negative candidate pool, so it is not an adjacency multiplier.
- **Scale.** On OGBN-Arxiv the inferred edges are cached and only the top-K neighbors are scored.
- This is a fine-tuned pipeline, not a frozen text score ([Zhang et al.](https://arxiv.org/html/2408.08685)).

**Guardual**, from the TGRB paper (Lei, Yi, He, Qiu, Wei, Liu, and Hong; arXiv:2510.17185, 2025; venue unverified).

- **Two thresholds instead of one.** Guardual replaces GNNGuard's single validation-tuned threshold with two thresholds on a precomputed RoBERTa embedding. Training picks the threshold by maximizing 0.7·P_preserve + 0.3·P_filter. These weights select the threshold; they are not edge multipliers.
- **Thresholds** (training / test): Cora **0.250/0.503**, PubMed **0.360/0.633**, ArXiv **0.320/0.540**.
- **Largest gains over GNNGuard:** Computer **+8.23%** and Photo **+10.74%**.
- **GNNGuard on text embeddings.** TGRB reports that GNNGuard becomes "top-tier" against inductive and evasion structural attacks once node features are advanced text embeddings. Earlier robust-GNN papers often evaluated only shallow features ([Lei et al.](https://arxiv.org/html/2510.17185)).

**SFT-auto** (also from TGRB).

- **Structure attack.** It flags a node as structure-attacked when the node's embedding cosine is below **0.5** to more than half of its neighbors. It then keeps the node's own text and removes the low-similarity edges.
- **Text attack.** When the node's text is flagged instead, it discards that text and uses the neighbors' text.
- **Cost.** The LLM is supervised fine-tuned, at one forward pass per sample.
- Preferring a node's own text is therefore conditional on which attack was detected, not an unconditional premise.

**IEEE S&P 2026 purifier** (Ma, Wang, and Yan) drops an edge when the cosine between the two endpoint embeddings, taken from the feature enhancer, falls below a threshold. The paper does not state whether the eight enhancers are frozen ([arXiv](https://arxiv.org/html/2603.26105)).

**Clean-graph or non-attack work.**

- **GraphEdit** fine-tunes an LLM to judge whether an edge exists. Its reported numbers are clean accuracy only ([arXiv](https://arxiv.org/html/2402.15183)).
- **SDM-SCR** (ICME 2026) splits node text into a signal view and a noise view ([arXiv](https://arxiv.org/html/2604.14746)).
- **ZeroGAR** benchmarks 13 zero-shot graph models; only its abstract was read ([arXiv](https://arxiv.org/abs/2609.33314)).

### Attacks already aim at these gates

- **PGD-Guard**, as configured in TGRB, is an adaptive structural attack. It only perturbs node pairs whose cosine exceeds a threshold, so a cosine gate never fires. The thresholds are {0.0, 0.3, 0.5, 0.7}, the budget is 20% of edges, and the embedding is RoBERTa.
- **BadGraph** (Chen, Zh. et al.; TheWebConf 2026) jointly edits adjacency and node text under an L0 token budget, and cuts accuracy by up to **76.3%**. Under SBERT it lowers RUNG's accuracy by **over 50%**. The authors attribute this to RUNG's reweighting being made to "overemphasize adversarial edges" ([arXiv](https://arxiv.org/html/2603.21155)).
- **Intruding with Words** (Lei, Hu, Ren, and Wei; NeurIPS 2024; arXiv:2405.16405) studies graph-injection attacks at the text level. **GraphTextack** (AAAI 2026) searches structure and text jointly for injected nodes.

None of the attacks read here chooses edges by BM25 or lexical overlap. The memo's BM25 candidate-pool attacker, a lexical counterpart of PGD-Guard, is therefore a threat-model contribution relative to this pass.

| Idea | What the opened papers do | Where the match stops |
| --- | --- | --- |
| Q1: reweight edges during propagation from a discrepancy | RUNG: continuous MCP-IRLS weights on the endpoint-state discrepancy, with a cutoff at γ and an anchor λ. GNNGuard: neighborhood-normalized cosine weights. TAGR: structural and cosine scores in frozen preprocessing. | RUNG's discrepancy is computed on node estimates, not explicit structure. TAGR freezes its adjacency before training. |
| Q2: penalize node-state movement under edge deletion | GRAND consistency under DropNode; IDEA invariance; Lipschitz regularizers on weights (Jia et al.; Juvina et al.); SpecSphere's first-order joint bound; the STEM-GNN paper's head penalty. | None penalizes the relative movement of encoder states under deletion of observed edges. This repo's `propagation_sensitivity_penalty` does. |
| Q3: frozen text scores allocate a structural trust budget | Guardual and the S&P purifier threshold cached cosines. LLM4RGNN distills GPT-4 judgments. GNNGuard on text embeddings uses continuous cosine weights. | No paper constrains a per-node budget. PGD-Guard and BadGraph already attack cosine-based trust. |
| Certified robustness (separate track) | AuditVotes: counted edge edits. AGNNCert: counted nodes, features and edges. Message-interception smoothing: whole-node feature rewrites. Hierarchical smoothing: continuous features. SpecSphere: joint edge and ℓ∞ features (claimed). | SpecSphere's proof has a gap and reports no certified accuracy. |

## Distribution shift, missing features, and test time

**Structural shift.** **Bazhenov et al.** (arXiv:2302.13875) induce structural distributional shifts by splitting nodes on popularity, locality and density. They find these splits hard, and simpler models often win. This is a ready-made protocol for `degree_shift_ood.py` ([arXiv](https://arxiv.org/pdf/2302.13875)). The OpenAlex title queries for `homophily shift graph` and `degree shift graph neural` returned **0** works, so the repo's homophily-shift protocol must be defined precisely in its own right.

**Degree bias** is a different question from train/test degree shift.

- **GraphPatcher** adds virtual nodes at test time, with gains of up to **3.6%** overall and **6.5%** on low-degree nodes ([Ju et al.](https://arxiv.org/pdf/2310.00800)).
- **Grace** combines self-distillation, graph completion and label propagation ([Xu et al.](https://doi.org/10.1145/3580305.3599368)).
- **Subramonian, Kang, and Sun** (2024) surveyed 38 degree-bias papers and found their hypotheses often unvalidated. They also find that high-degree test nodes have lower error regardless of training, so aggregate accuracy hides the effect ([arXiv](https://arxiv.org/pdf/2404.03139)).

**Heterophily filters.** **PC-Conv** (AAAI 2024) uses a two-fold Poisson–Charlier filter ([DOI](https://doi.org/10.1609/aaai.v38i12.29246)). **SEC-GFD** (AAAI 2024) is a fraud detector built on hybrid spectral filtering ([DOI](https://doi.org/10.1609/aaai.v38i8.28773)).

**Missing features.** **T2-GNN** (AAAI 2023) distills separate feature-level and structure-level teachers ([DOI](https://doi.org/10.1609/aaai.v37i4.25553)). **MDS-GNN** (arXiv 2024) couples reconstruction with contrastive learning ([arXiv](https://arxiv.org/pdf/2408.04845)). Masking feature coordinates is not the same as adversarially rewriting text, so the two need separate tests.

**Distribution shift and OOD detection.**

- **StableGNN** (TPAMI 2023) learns sample weights over subgraphs ([DOI](https://doi.org/10.1109/tpami.2023.3321097)).
- **FR-GNN** (IEEE IoT-J 2024) reconstructs features for a frozen GNN ([DOI](https://doi.org/10.1109/jiot.2024.3385292)).
- **COIN-GNN** (TKDE 2025) targets spatial-temporal graphs ([DOI](https://doi.org/10.1109/tkde.2025.3606629)).
- **GNNSafe** (Wu et al. 2023) is an energy-based OOD detector, relevant if STEM-GNN reports OOD detection.
- Wu et al.'s survey places test-time graph transformation (GTRANS, GAPGC) as the third branch of graph domain adaptation ([ar5iv](https://ar5iv.labs.arxiv.org/html/2402.16374)).

**Test time.** A title search for TENT-style test-time batch-norm adaptation turned up only training-time uses of batch normalization.

- **Temperature calibration.** **STAC** uses one global temperature under homogeneous shift ([arXiv](https://arxiv.org/pdf/2607.10804)). **HoTS** sets per-node temperatures from entropy and local homophily, with a best mean ECE of **4.79%** across 18 benchmarks ([arXiv](https://arxiv.org/pdf/2609.32426)). **WATS** uses graph-wavelet features, with up to **42.3%** lower ECE ([arXiv](https://arxiv.org/pdf/2506.23782)).
- **Frozen-backbone adaptation.** **SETTA** reports gains of **+4.61 / +3.08 / +2.01** points on Cora, CiteSeer and PubMed for a frozen two-layer GCN ([DOI](https://doi.org/10.3390/bdcc10080260)). **T3R** reports at least **9.37%** relative improvement on cross-domain OGB classification ([arXiv](https://arxiv.org/pdf/2606.30011)). **GeoTTA** and **PromptDyG** are related methods.
- The IJCAI 2026 test-time adaptation survey was not parsed ([PDF](https://www.ijcai.org/proceedings/2026/0866.pdf)).

**Oversmoothing** is related to robustness but not the same thing. **Rusch, Bronstein, and Mishra** (2023) define it as exponential convergence of node-feature similarity. They call normalization, DropEdge and changes to the message-passing dynamics necessary but not sufficient mitigations ([ar5iv](https://ar5iv.labs.arxiv.org/html/2303.10993)). **Zhao, Wang, Han, and Guo** (arXiv:2211.06605v2, local PDF) analyze oversmoothing as Markov-chain convergence and give a sufficient condition for operator-inconsistent GNNs to avoid it. This is relevant when judging DropEdge-style perturbation and changing propagation weights. It is a depth analysis, not an adversarial certificate.

**Surveys.**

- **Dai et al.** (Machine Intelligence Research 2024, **152** citations) name robustness as one of four aspects of trustworthy GNNs ([DOI](https://doi.org/10.1007/s11633-024-1510-8)).
- **Zhang, Wu, Yuan, Pan, Tong, and Pei** (Proceedings of the IEEE, **145** citations) make three relevant points ([ar5iv](https://ar5iv.labs.arxiv.org/html/2205.07424)):
  - Attack and defense rates do not transfer to other attacks.
  - Certificates depend on the architecture and graph type.
  - Most robust GNNs were tested only on medium-sized graphs.

  They ask for a comparable metric, such as an upper bound on prediction divergence. That request remains open, which is why the headline accuracies in this brief cannot be ranked against each other.
- **Zheng et al.**'s heterophily survey lists heterophily–robustness links, but only as section headings ([ar5iv](https://ar5iv.labs.arxiv.org/html/2202.07082)).

## Implications for STEM-GNN

1. **Scope current claims to random corruption.** `random_edge_drop.py`, `missing_feature.py`, `degree_shift_ood.py` and `homophily_shift_ood.py` test random corruption or shift. Mujkanovic et al. and Hou et al. show that non-adaptive evaluation inflates robustness. Any adversarial claim needs an adaptive attack (PGD or PRBCD) on the full finetune pipeline.
2. **Guard against gradient masking from VQ.** Finetuning routes encoder states through a frozen VQ codebook (`--freeze_vq 1`). Quantization is non-differentiable, so naive white-box PGD can under-report vulnerability. Use a straight-through or BPDA surrogate through the VQ step when attacking.
3. **Measure stability past the quantizer.** Under identical perturbations, report:
   - pre-VQ state displacement (what `propagation_sensitivity_penalty` controls);
   - the per-head code-change rate;
   - the change in router distribution;
   - the rate of prediction flips.
4. **Run baselines on the same text encoder.** GNNGuard, Guardual and RUNG should use STEM-GNN's 768-dim embeddings, because TGRB shows the encoder alone changes GNNGuard's ranking. Include PGD-Guard as the defense-aware attack against any cosine or semantic gate.
5. **Ablate Q1–Q3 separately before claiming the combination.** Under matched seeds and checkpoints, run:
   - the baseline;
   - structural reweighting only;
   - the existing sensitivity penalty only;
   - frozen semantic gating only;
   - their combinations.

   Use GRAND-style consistency as the Q2 baseline. Report clean accuracy, robustness curves, variance and cost.
6. **Reuse published protocols in the existing scripts.**
   - **`random_edge_drop.py`:** add DropEdge 20% (SpecSphere) and 25% and 50% deletion (TAGR).
   - **`degree_shift_ood.py`:** add Bazhenov et al.'s popularity, locality and density splits, and report accuracy by degree bucket.
   - **`homophily_shift_ood.py`:** define the protocol precisely, with GJDNet's assortativity-inversion attacks as the nearest adversarial analogue.
   - **`missing_feature.py`:** compare against T2-GNN and MDS-GNN, and test embedding masking separately from raw-text rewrites followed by re-encoding.
7. **Use arxiv as the scale test.** GPR-GAE's OGB-arXiv transfer numbers are the most transferable large-graph reference here. Report at least a transfer attack on arxiv.
8. **Check RUNG's γ scale before borrowing it.** The memo's γ=36 uses its own normalization of the discrepancy, so it is not comparable with RUNG's {0.5,…,5} grid.

## Method and limits

**Sources.** OpenAlex, arXiv HTML and abstracts, and PDFs retrieved on 4 October 2026. Semantic Scholar returned HTTP 429 on every query, and a later arXiv batch also returned 429.

**Verification (17 October 2026).** Two independent reviews, one by Claude and one by Codex, rechecked the brief against the notes, the local PDFs and arXiv pages.

**Added in review, not yet read in full:** Jia et al. (KDD 2023), Juvina et al. (TMLR 2024), NoisyGNN, hierarchical randomized smoothing, Gosch et al., graph filter stability theory, and the original GNNGuard paper (its mechanism was read).

**Not opened or not extracted:**

- GNNCert and RES-GRACE.
- The IJCAI 2026 TTA survey.
- Dai et al.'s robustness sub-taxonomy.
- Main tables for ADEdgeDrop, GRANCE, ACGMAE, Pure-GNN, BDP, Mid-GCN and IDEA.
- LLM4RGNN's text-quality numbers.
- Numbers from the S&P 2026 purifier tables.
- Abstracts for DEGNN, the learnable-structural-augmentation degree-bias paper, and Damke and Hüllermeier (2025).

**Resolved bibliographic items:**

- The full TGRB author list.
- The title of the text-level graph-injection paper (*Intruding with Words*).
- STEM-GNN v2 reports "nine" benchmarks.
- ADEdgeDrop is TKDE 2025.
- Mujkanovic et al. is NeurIPS 2022.
- SpecSphere uses an AAAI template, but acceptance is unconfirmed.

**Citation counts** are OpenAlex snapshots from title-filtered queries. They are not a census, and they may double-count preprint and journal versions.

**Local PDFs.** SpecSphere (2505.08320v3) and Zhao et al. (2211.06605v2) are used above. arXiv:2512.02384v3 (Rajaraman and Wu, Markov-chain approximate message passing for spiked-Wigner inference) was excluded because it is not about GNN robustness.

**Numbers.** Values traced only to the notes, such as the AuditVotes relative changes, GJDNet coverage, and the calibration and TTA figures, are reported as the papers state them. They were not re-derived from the tables.
