These are proposed amendments for the researcher to decide, not changes to the existing confirmatory family. The new datasets and any new decision thresholds should be fixed before their relevant outcomes are inspected. Costs below are relative estimates; actual runtime and memory require pilot measurements.

### Q1. Attack weakness versus robustness

**H110 alone is not a deciding test.** It tests whether the S-VG advantage remains on the expanded Cora union, with the existing clean guardrail. A positive result supports a gain against that finite suite; neither outcome identifies gradient obstruction by itself.

| Option | Decision rule and tradeoff | Cost |
|---|---|---|
| **A. Existing H110 only** | Use its prespecified support and evidence-against rules. Cleanest confirmatory interpretation, but limited mechanism evidence. | Already planned. |
| **B. H110 plus prespecified attack diagnostics — recommended** | Apply the criteria below to Cora; report Pubmed and Wiki-CS separately. Distinguishes a suite-specific gain from signs of attack weakness, but remains observational. | Mostly analysis of planned graphs; modest additional scoring. |
| **C. Independent attack audit** | Add independently implemented attacks, more restarts and convergence traces before drawing a stronger conclusion. Stronger challenge, but substantially more compute and engineering. | High; scope and runtime need a pilot. |

For **B**, predefine these as *diagnostic classifications*, not replacements for H110 or proof of a mechanism:

- **Evidence consistent with attack weakness:** the expanded shared union reduces the old +3.09-point advantage to below +1 point, or H110’s upper CI is below +1; **and** at least one of GRBCD, GCN transfer, bpda_flat, detach or twin causes a substantial additional S-VG own-attack worst-node drop versus the standard-plus-strong true-gradient suite. The protocol’s **3-point drop** is a useful existing flag. Report which attack contributes newly broken nodes and the comparable change for S0. A large drop alone flags sensitivity; it does not establish obstruction.
- **Evidence consistent with suite-specific robustness:** H110 meets its existing support rule and clean guardrail; the gain remains positive across the three split means; and no diagnostic family produces a ≥3-point incremental S-VG own-attack drop. Report the x0.1/x10 sweep, GRBCD and transfer results individually, including whether true-gradient attacks converge comparably and whether S0 is also strengthened by the suite.
- **Otherwise inconclusive.** In particular, a learning-rate improvement suggests optimization sensitivity; transfer or altered-backward attacks outperforming the true-gradient attack can suggest obstruction. Neither proves it. Failure of these attacks to improve also does not rule it out.

The existing protocol does not record loss trajectories, so convergence cannot currently be established from its saved final graphs alone. Adding trajectories or independent attacks requires a separately documented follow-up. The shared union equalizes the graphs used for scoring, not attack effort or attack strength. The original gain and its conditional uncertainty are documented in [analysis.md](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md>); H110 and the diagnostic are in [protocol.md](<G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md>).

### Q2. Knowledge-graph link tasks

The repository performs **relation-type classification for a given endpoint pair**, scored by accuracy, rather than conventional missing-link ranking. Its generator concatenates train, validation and test triples into the graph, while using generic edge text; the link head averages endpoint embeddings. These choices need an explicit leakage and threat-model decision before an attack is meaningful. See [link.py](<G:/broswer download/STEM-GNN/STEM-GNN/task/link.py>) and [KG/gen_data.py](<G:/broswer download/STEM-GNN/STEM-GNN/dataset/data/KG/gen_data.py>).

| Option | Threat model and tradeoff | Cost |
|---|---|---|
| **A. Global context perturbation — recommended primary** | Build message-passing context from training triples only; hold evaluation query endpoints and labels fixed and exclude queried triples from context. Flip directed, typed context triples with a global budget, e.g. 10%/20% of unique training triples. Most analogous to the node experiment, but potentially expensive. | High: data split audit, typed attack implementation, leakage tests and scoring. |
| **B. Per-query local perturbation** | Independently edit up to a fixed integer number of context triples near each query. More interpretable local threat and potentially cheaper per attack; results do not describe one globally corrupted KG. | High engineering; potentially many attacks, mitigated by a fixed query subset. |
| **C. Reproduce existing all-split context** | Perturb the repository’s current context and label the result transductive and leakage-sensitive. Easiest comparison to the current implementation, but less persuasive as a held-out robustness evaluation. | Moderate, after auditing loader behavior. |

For A, define a valid edit as an addition or deletion of a **directed typed triple**; a relation substitution costs one deletion plus one addition. Preserve parallel relations between the same endpoints, reject duplicate triples, and declare whether inverse edges are materialized and counted as one logical edit. Restrict added relation IDs to the known vocabulary. Arbitrary typed triples may be semantically implausible, so present this as a structural stress test; a domain-constrained threat is a separate option.

Use **worst-query accuracy**: for every fixed test query, mark it correct only if correct on the clean graph and every valid graph in the shared union at that budget or below; average those indicators. Also report minimum whole-graph accuracy and clean accuracy. As with worst-node accuracy, the per-query worst cases need not coexist in one graph.

A first VGAE option is to score a **relation-collapsed structural projection**, then map its score to each typed message edge; this preserves the no-text rule but cannot distinguish relations or direction unless designed to do so. A typed/directed structural VGAE is more expressive but is a different, more costly defense. For poisoning, randomly inject valid typed triples into training context, retrain the VGAE and downstream model on each corrupted graph, and report injection rules and seed-matched comparisons. Do not infer relation-aware protection from a collapsed scorer.

### Q3. Molecular graph tasks

| Option | Threat model and tradeoff | Cost |
|---|---|---|
| **A. Structural stress test — recommended pilot** | Independently flip a small fixed number of undirected bonds per molecule, such as 1 and 2, with reciprocal message edges counted once. This is computationally defined but may create invalid molecules or change their true properties. | High: graph-specific attacks, bond-feature handling and validity checks. |
| **B. Chemistry-constrained edits** | Require valid valence and defined bond types, optionally connectedness. More chemically plausible, but validity does not guarantee the original property label remains correct. | Very high: chemical validation and constrained search. |
| **C. Random bond corruption only** | Sample edits under the same rules and retrain for poisoning. Cheap feasibility baseline, but does not test adaptive robustness. | Moderate. |

For adaptive evaluation, generate a shared candidate union **for each molecule** across models and attacks. Since ROC-AUC and AP depend on rankings across molecules, there is no literal per-graph “worst AUC.” A reproducible surrogate is to select, **separately for each model and labeled task**, the candidate with the worst true-label margin for that molecule, then calculate the dataset metric on the assembled scores. This is a label-aware, per-example worst-case construction, not necessarily the global minimum AUC/AP. Include clean candidates, preserve missing-label masks, report the task aggregation and class counts, and also report clean metrics and results on common attacked datasets. The repository maps graph tasks to AUC; verify the exact evaluator and the desired PCBA AP convention before fixing the primary endpoint. See [graph.py](<G:/broswer download/STEM-GNN/STEM-GNN/task/graph.py>) and [eval.py](<G:/broswer download/STEM-GNN/STEM-GNN/utils/eval.py>).

A separately fitted VGAE on each roughly 20-node molecule is likely data-poor; treat it as a baseline, not an assumed useful defense. Prefer testing a **shared structure-only VGAE trained across training molecules**, with no cross-molecule edges or text inputs, then freeze it for evasion. Decide explicitly whether topology alone is sufficient or whether discrete atom/bond types are permitted structural inputs; the latter changes the defense definition. For poisoning, refit using corrupted training molecules.

Start with HIV and a **fixed, seeded, label-aware stratified PCBA test subset**, chosen before results, with enough positives and negatives per reported task; disclose tasks excluded for insufficient labels. Pilot attack time and memory per molecule, then scale. A subset saves substantial compute but limits precision and can distort rare-task AP; full training and clean evaluation may still be feasible independently of full adaptive evaluation. The stated dataset sizes and typical molecule size are user-provided and were not independently verified here.

### Q4. Arxiv cost

| Option | Tradeoff | Cost |
|---|---|---|
| **A. Fewer cells, same global suite — recommended first choice** | Preserve the threat model and shared-union interpretation; fewer splits/seeds reduce precision and coverage. | High per cell; pilot a single complete cell on the available GPU first. |
| **B. Cheaper declared suite** | Keep representative true-gradient, greedy, transfer and scorer-aware attacks, but reduce restarts or iterations before inspecting accuracy. Saves attack time; weaker attacks limit comparison with Cora/Pubmed. | Moderate-to-high engineering and compute; benchmark each attack. |
| **C. Fixed target subset and local budgets** | Reduces attack objective and possibly subgraph cost; estimates robustness only for the sampled targets and defines a different adversary from graph-global attacks. A target subset alone does not guarantee cheap full-graph propagation. | High implementation and validation; potentially substantial runtime savings. |

Record graph size, integer budgets, candidate-block feasibility, GPU memory, per-attack time, scoring and storage in a gate before choosing the grid. If global attacks are infeasible, label B or C as a different suite/threat model and avoid direct claims that differences reflect dataset difficulty alone. The Arxiv size is the supplied estimate; available GPU capacity and actual runtime were not verified.

### Q5. “Avg.” column

| Option | Tradeoff | Cost |
|---|---|---|
| **A. Per-task-type averages — recommended** | Average percentage-point gains over the same baseline within node accuracy, link accuracy, and each graph metric separately. Retains interpretable units; still gives each dataset equal weight unless stated otherwise. | Low. |
| **B. Mean rank across datasets** | Avoids mixing metric scales and highlights consistency, but discards effect magnitude and depends on the set of methods compared. Report ties and individual results. | Low. |
| **C. Normalized gain over baseline** | Can summarize heterogeneous metrics, but the denominator and metric scale determine the answer; near-zero denominators can destabilize it. Requires a prespecified formula and sensitivity analysis. | Low-to-moderate analysis. |

Keep each dataset’s clean and robust score visible, including whether PCBA uses AP. A single raw average of accuracy, ROC-AUC and AP can be reported only as a clearly labeled descriptive convention if matching the paper is essential; it has no straightforward common performance interpretation. I did not verify the paper’s exact table convention, so reproducing its “Avg.” requires checking that source first.