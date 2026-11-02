The arithmetic checks out. The main issues are interpretation, missing diagnostic disclosures, and unsupported rerun provenance. **Two wording issues should block finalizing the analysis; neither requires new training.**

1. **HIGH — BLOCKER: “No robustness advantage” overstates the result.**  
   **`experiments/llm-stem/analysis.md:7`**  
   H60 and H61 are inconclusive under the preregistered evidence-against rule: their upper confidence limits are **+1.08275** and **+1.30820**, both above +1. There is also a descriptive worst-graph improvement.  
   **Replace with:** “Stage 1 did not establish a Qwen-specific robustness advantage on the preregistered worst-node endpoint. No hypothesis met its support threshold; H60 and H61 remain inconclusive under the evidence-against rule.”

2. **HIGH — BLOCKER: The worst-graph attribution uses the wrong contrast and claims too much.**  
   **`analysis.md:38`**  
   S-R − S-U **clean** accuracy cannot establish the source of a **worst-graph** gain. The relevant paired contrast is:
   
   **S-R − S-U worst-graph@20%: +0.02579 points, 95% CI [−1.11109, +1.16267], p = 0.96188, 7/15 wins.**
   
   **Replace with:** “S-R and S-U have similar worst-graph point estimates. Their paired difference is +0.03 [−1.11, +1.16], so these descriptive results do not establish a Qwen-specific benefit.”  
   Similar estimates support that limited conclusion; they do not prove the gain “isn’t attributable to Qwen” or establish equivalence.

3. **MEDIUM — The proposed explanation from routing shift is unsupported.**  
   **`analysis.md:53`; `protocol.md:79,144`; `STEM-GNN/scripts/llm_stem/run.py:270–275`**  
   The **0.044446** figure is mean test-node routing TV at **layer 2**, on S0’s **own `ste` attack at 20%**. It does not characterize the whole attack union, node-level extremes, or downstream sensitivity. Small average probability changes can still matter. The protocol explicitly treats S-R’s zero shift as an implementation check, not causal evidence.  
   **Replace with:** “S0’s average layer-2 routing TV is 0.044 on its own `ste` attack at 20%. This diagnostic alone does not explain the small worst-node effect.”  
   The preregistered **expert-output shift** diagnostic, needed to assess the proposed mechanism, is absent.

4. **MEDIUM — The union is symmetric in exposure, but its penalties are empirically unequal.**  
   **`analysis.md:15,27,39`; `run.py:247–269`**  
   At 20%, every arm faces the **same 62 attacked graphs plus clean**: five attacks against each of six arms, plus SAGE, at both budgets. Thus every arm faces attacks tuned to itself and transfers from the others. No arm is uniquely singled out by the construction.

   Nevertheless, equal exposure does not guarantee equal attack effectiveness. Removing each evaluated arm’s own-source attacks raises worst-node accuracy by these mean points:

   | Arm | Increase |
   |---|---:|
   | S0 | 2.56 |
   | S-R0 | 2.05 |
   | S-R | 2.09 |
   | S-U | 1.89 |
   | S-RF | 1.70 |
   | S-PQ | 2.31 |

   These differences do **not** prove bias; they show that own-source attacks contribute unequally. State that the comparison is conditional on this finite, model-generated candidate suite. Worst-node also permits a different candidate graph for each node; worst-graph uses one candidate graph.

   **Own-attack is computed correctly**, but should be defined explicitly: intersection of clean correctness and all **ten own-source attacks across 10% and 20%**, not just one attack or the 20% attacks. SAGE transfer, by contrast, is the single graph at the stated budget.

5. **MEDIUM — Rerun provenance is stronger than the available evidence supports.**  
   **`analysis.md:13`; `run.py:238–254,277–278,394–400`**  
   The supplied workspace contains no llm-stem execution logs establishing “13 of 60,” a slow host, unchanged invocation, or what was inspected. The code supports a narrower statement: training finishes before attacks; attack logs expose timing; result rows are logged after attack generation; JSON/PT output is written after `run_cell` returns. Consequently, an interrupted attack phase would produce no final result file through this code path.

   **Correct wording:** Attribute the cancellation account to operator records and cite/archive those records; distinguish “no final metrics emitted by this script” from “no results seen.”

   Speed-only cancellation avoids direct accuracy-based selection, but is not automatically unbiased: runtime can depend on model/attack state, and nondeterministic retraining can replace one realization with another. A host-only slowdown independent of outcomes would reduce that concern. The available files do not establish either explanation.

6. **MEDIUM — Diagnostic definitions and preregistered omissions need disclosure.**  
   **`analysis.md:42–57`; `protocol.md:79–81,91–94,146`; `run.py:227–229,270–275`**
   
   Add the following:
   
   - Routing TV is averaged over **test nodes**, separately for both layers.
   - Expert perplexity is computed from **clean, all-node mean soft routing probabilities**, separately per layer. It is not mean nodewise perplexity or hard expert assignment usage.
   - Code-change rates are **per VQ head**, on test nodes, under the own `ste` 20% graph. The reported approximate 0.26–0.27 range is reasonable.
   - **Expert usage, expert-output shift, and codebook perplexity** were preregistered but are not reported. Expert perplexity is not codebook perplexity. Mark these as uncollected/unreported rather than implying diagnostic completeness.
   - Carry forward relevant text/leakage limitations: previously viewed aggregate teacher test performance, transductive text use, **21.8% text truncation**, possible Qwen pretraining exposure, and exclusion of text attacks.
   
   Teacher-error inheritance diagnostics are less relevant here because Stage 1 does not use the teacher prior; that scope distinction should be explicit.

7. **LOW — H62’s caveat is substantially correct, but “biases” needs qualification.**  
   **`analysis.md:23`; `protocol.md:142`**  
   The per-cell maximum is the preregistered, deliberately demanding benchmark. It is at least as large as every constituent comparator in each cell; selection can also elevate it relative to a fixed comparator. This is not an implementation defect.

   **Prefer:** “H62 compares S-R with the best observed trivial router in each cell, rather than a single fixed router. Its negative estimate does not establish inferiority to every trivial router.”

   Indeed, S-R minus S-RF is **+0.05 [−0.78, +0.89]**, while S-R minus S-PQ is **−0.51 [−1.56, +0.53]**. H62 has **no preregistered evidence-against threshold**; “not supported” is the correct verdict.

All reported numerical summaries match independent JSON recomputation at displayed precision: paired t intervals with 14 degrees of freedom, p-values, split means, wins, clean changes, arm means/SDs, diagnostics, and descriptive contrasts. Holm-adjusted values are **H61 0.966206** and **H62 0.136656**. H60/H61 satisfy the clean-change ≥−1 condition on point estimates—and even their reported lower confidence limits—but fail the robustness support conditions.

All **900 reconstructed accuracy checks** from the 15 per-node files matched the JSON. There are **930 saved attack edit records**, with maximum total edits **527/1,055**, all within budget. Full structural validation cannot be independently repeated from correctness vectors and edit counts alone.

**No numerical or verdict blocker found. The blockers are the blanket conclusion and the categorical worst-graph attribution.**