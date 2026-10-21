**The arithmetic is mostly correct, but the strongest CTV conclusion is not established: `exact_attack.py` computes independent local costs, then reports a global accuracy without constructing or evaluating an attacked graph.** H1 is supported for STEM-lite’s attack harness; the claimed 12-point effect should not be generalized quantitatively to full STEM.

I performed only reads and short JSON calculations. No files were modified and no training or attacks were run.

Paths below use:

- `SV/` = `STEM-GNN/scripts/stem_vq_robustness/`
- `CTV/` = `STEM-GNN/scripts/certified_token_voting/`
- `AR/` = `STEM-GNN/scripts/adaptive_robustness/`
- `LLM/` = `STEM-GNN/scripts/llm_robustness/`

**Findings**

| Severity | Location | Problem and evidence | Fix |
|---|---|---|---|
| **HIGH** | `CTV/exact_attack.py:62–70`; CTV analysis:20,29,36–37 | “Exact-greedy accuracy” is an accounting estimate, not measured attack accuracy. The function subtracts a count of independently affordable flips from initially correct nodes. It never chooses edges, updates scores, handles interference, or counts initially wrong nodes becoming correct. Thus 46.5% and 59.0% are not verified achieved accuracies. | Construct a unique undirected edit set, update both endpoints after each edit, and evaluate the resulting graph. Rename existing columns “independent-cost allocation estimate.” |
| **HIGH** | CTV analysis:29; CTV protocol:33 | H11’s verdict silently changes the registered endpoint. Locked CTV **passes the original PRBCD rule**: 67.99% at 20%, versus 60.99% DropEdgeCons and approximately 55% SAGE. Rejecting broader robustness is reasonable only with valid stronger-attack evidence; the current allocator does not supply it. | Report “registered PRBCD criterion met; stronger-attack robustness unresolved.” Treat the additional attack assessment as exploratory. |
| **MEDIUM** | `CTV/exact_attack.py:49–57` | `cum >= margin` counts a tie as a successful flip for every competing class. With `argmax` choosing the lowest class index, equality flips only when the competitor has a lower index than the original winning class. Costs can therefore be underestimated. | Use `>=` for lower-index competitors and `>` for higher-index competitors; verify chosen edits through the actual model, preferably with float64 score calculations. |
| **MEDIUM** | CTV analysis:30,38 | “73% … does not beat … 72.5%” is numerically false. Exploratory α=4 certifies **73.211% at k=1**, versus the prior MLP’s **72.467%**. The broader assertion that certification is bounded above by text-only accuracy is also unsupported: neighbors can correct a text-only error. | Say “a small exploratory numerical improvement, without confirmatory evidence.” Distinguish a useful comparison baseline from a mathematical upper bound. |
| **MEDIUM** | STEM analysis:29–31; `SV/stem_lite.py:45–53`; real `model/vq.py:929–937` | H1’s proxy result is sound, but “otherwise [full STEM] will report about 12 points” overgeneralizes. Hard mode explicitly injects a connected zero-gradient path. Real eval VQ omits STE and may instead yield a disconnected gradient error. Full STEM was not evaluated. | Restrict the magnitude to this proxy and implementation. Test actual eval VQ with a frozen model and an attack-only BPDA wrapper. |
| **MEDIUM** | STEM analysis:39–43; protocol:77–81; `SV/run.py:100–105` | H4’s endpoint diagnostic was replaced after observing collapse. The completed JSONs contain only stable repairs and SAGE, all with **zero** endpoint zero-row fraction. The alleged epoch-6 trace and global-denominator collapse are not supplied as raw trace artifacts in the reviewed experiment directory. | Classify the mechanism diagnosis as exploratory; retain the original diagnostic as inconclusive. Archive per-epoch norms, zero fractions, losses, and checkpoint epochs. |
| **MEDIUM** | CTV protocol:48–56 | H12’s amendment follows observed certificate results: it explicitly quotes split-0 Cert@1=31.4% and Cert@2=5.9%. “Before any attack result” does not mean before the outcome relevant to H12. | Evaluate confirmatory H12 under its original 60%/40% thresholds. Treat the MLP-relative replacement as a post-result interpretation. |
| **MEDIUM** | STEM analysis:47–49; `SV/run.py:129–176`; `LLM/run.py:169–181` | The commitment comparison shares training seeds and code, but does not use matched attack random streams. STEM attacks 5%, 10%, 20%; LLM attacks 10%, 20%. PRBCD is not reseeded per budget. Changing commitment also changes training RNG consumption through early stopping. | Reset a dedicated attack seed before every model/budget/restart, and run a direct paired commitment ablation. |
| **LOW** | STEM analysis table:7–17 | The baseline/transfer rows have extra cells and do not align with the seven-column header. The requested mean±std and runtime reporting is incomplete. Most displayed means nevertheless match raw JSONs. | Repair table columns; report a consistent sample SD and runtime summary. |
| **LOW** | Prior review:11; STEM analysis table | The prior review records an action to report minimum-over-attacks, but the main table still emphasizes STE alone. Transfer is slightly stronger for CodeMargin on some model/budget/split combinations. | Report per-split minimum across attacks before averaging, plus the component results. |

**A. Numerical reconciliation**

I recomputed aggregates from all three STEM split JSONs, all three CTV split JSONs, and `exact_attack.json`. All twelve CTV log rows exactly match their corresponding JSON rows.

Selected reconciled results, in percentage points:

| Quantity | Recomputed | Analysis |
|---|---:|---:|
| STEM-lite clean | 82.318 ± 1.600 | 82.3 ± 1.6 |
| STEM-lite STE@10% | 69.616 | 69.6 |
| STEM-lite hard@10% | 81.560 | 81.6 |
| STEM-lite STE@20% | 62.589 | 62.6 |
| DropEdgeCons STE@10% | 70.745 ± 2.881 | 70.7 ± 2.9 |
| DropEdgeCons paired gain @10% | +1.128 ± 2.072 | +1.1 ± 2.1 |
| CodeMargin paired gain @10% | −1.580 | −1.6 |
| Both paired gain @10% | −0.419 | −0.4 |
| Locked CTV clean | 78.562 ± 1.893 | 78.6 ± 1.9 |
| Locked CTV PRBCD@20% | 67.988 ± 3.558 | 68.0 ± 3.6 |
| Locked CTV Cert@1 / @2 / @3 | 30.400 / 10.251 / 3.836 | 30.4 / 10.3 / 3.8 |
| α=4 exploratory Cert@1 | 73.211 | 73.2 |

Sources: each experiment’s `results/split*.json`, keyed by `model` and the metrics above. SDs here are sample SDs.

The “exact-greedy” means also reproduce the table: locked **59.236% / 46.470%** at 10%/20%; α=4 **65.152% / 58.978%**. Their arithmetic is correct; their interpretation as achieved attacks is not.

The LLM logs’ baseline rows give **80.913% clean, 66.973% PRBCD@10%, 58.447% @20%**. Thus the cross-run differences are **+1.405 clean, +2.643 @10%, +4.142 @20%** with commitment. These are observations, not an isolated causal estimate.

The clear reporting error is the statement that 73.2% does not exceed 72.5%.

**B. Verdicts and registration**

| Hypothesis | Codex verdict | Analysis verdict |
|---|---|---|
| **H1** gradient masking | **Supported for the registered proxy comparison.** Full-STEM magnitude unverified. | Supported |
| **H2** DropEdge consistency | **Not supported:** +1.128 < +3; clean condition passes. | Not supported |
| **H3** code margin | **Not supported:** −1.580 robust points despite slightly fewer changed codes. | Not supported |
| **H4** collapse/repair | **Original endpoint inconclusive; repair clean accuracy supported; mechanism exploratory.** Reduced zero rows versus the collapsed endpoint cannot be established from supplied results. | Mechanism found; amended |
| **H10** CTV clean | **Supported under the registered ≥78% mean rule:** 78.562%. Three splits do not establish precision. | Supported, borderline |
| **H11** CTV adaptive robustness | **Original PRBCD criterion met; broader robustness unresolved.** Current “exact” global estimate cannot overturn it as measured attack evidence. | Not supported |
| **H12** certificate | **Original locked criterion fails:** 30.400% <60% and 3.836% <40%. MLP-relative amendment is post-result. | Not supported |
| **H13** degree | **Descriptively supported.** Locked bucket means increase across all three splits; this is not a degree-only causal result. | Supported |
| **H14** sanity | **Passed on tested PRBCD graphs:** zero violations. Mathematical soundness follows from assumptions below, not from zero violations alone. | Supported |

For H11, locked CTV’s 74.533% at 10% also exceeds the registered comparison values. At 20%, its margin over prior DropEdgeCons is **6.995 points**, exceeding the required five.

The α=2 and α=4 sharp-vote settings **are correctly marked exploratory** in JSON model names, `CTV/run.py:155–156`, the analysis table, and analysis:37. Split-0 TEST-based selection contaminates the pooled three-split confirmatory interpretation; adding two splits does not erase that selection.

Timing evidence has limits. Git commit `e52fbce` at **14:24:26 −04:00** already contains the STEM amendments; the preceding `2edb790` version lacks them. CTV’s amended protocol is present in `730d315` at **20:12:11 −04:00**. These snapshots do not prove precise result-generation order. More decisively, both H4’s amendment and H12’s amendment explicitly incorporate observed outcomes. They cannot be described as pre-outcome amendments for those outcomes.

**C. Exact local costs, global allocation, and certificate**

The local optimization is correct **apart from tie handling and numerical boundary cases**, under the implemented simple-graph threat.

`CTV/run.py:53–57` computes:

\[
s_i=\alpha v_i+\sum_{j\to i}v_j.
\]

`AR/run.py:39–43` coalesces an undirected graph and removes self-loops. Therefore:

- Each undirected neighbor contributes once to node \(i\), through its incoming directed edge.
- Deleting \(\{i,j\}\) reduces the \(y-c\) margin by \(v_{jy}-v_{jc}\).
- Inserting it reduces that margin by \(v_{jc}-v_{jy}\).
- The reverse stored edge changes node \(j\), not node \(i\) again.
- The own-vote term is included in the initial margin and is unaffected by permitted edits.
- Existing neighbors and self are excluded from insertions.

Those are exactly the gains implemented at `exact_attack.py:49–53`. Since incident edits have equal cost and additive effects, sorting positive gains and taking the shortest successful prefix gives the minimum cost for each competitor. Minimizing over competitors is valid.

The equality test needs the class-index correction described in the findings. A zero initial margin also needs explicit tie-aware treatment rather than assuming a positive-gain edit is always required.

**The global allocator is unsound as an achieved-accuracy calculation.** Its documentation calls endpoint reuse “attacker-favourable,” but endpoint effects can also repair errors or undo other targets.

A concrete mathematical counterexample using the implemented score:

- Two nodes, votes \(v_0=(1,0)\), \(v_1=(0,1)\), α=0.5, one undirected edge.
- Both true labels are class 1.
- Clean scores are \((0.5,1)\) and \((1,0.5)\): accuracy 50%.
- The initially correct node has local flip cost one.
- `greedy_accuracy` predicts 0% accuracy after spending one edit.
- Deleting the edge flips that node **and repairs the other node**: actual accuracy remains 50%.

Thus the reported number is not guaranteed feasible and is not generally a bound in either direction.

**The certificate \(r=\lceil m\rceil-1\) is sound and conservative.** A simplex vote changes any fixed class-pair margin by at most one per incident edge edit. If \(k<m\), every original winning margin stays strictly positive.

- Integer margin \(m=2\): radius 1; two edits may produce a tie.
- Noninteger margin \(m=2.3\): radius 2.
- Zero margin: radius −1, conservatively certifying nothing.

A favorable deterministic tie rule could certify some equality cases, but the current strict-margin certificate safely avoids them. `radii()` uses `topk` whereas prediction uses `argmax`; tied winners can differ between these routines. Because zero-margin cases receive radius −1, this does not create a positive-radius false certificate.

Floating-point rounding near an integer remains an implementation caveat. A formal numerical guarantee needs conservative error handling. Zero observed violations do not resolve that boundary issue.

**D. H1: real mechanism, harness-specific magnitude**

The per-split registered gaps are **12.863, 11.896, and 11.073 points**, all above five.

`SV/stem_lite.py:49,53` gives essentially identical hard-code forward values in STE and hard modes, with different gradients. The model is placed in eval mode by `AR/run.py:153`, so dropout remains disabled. `SV/run.py:134–137` attacks through STE and evaluates the returned graph through hard VQ.

This establishes that the proxy’s apparently high hard-mode robustness is misleading. Transfer attacks provide additional evidence: baseline transfer@10% averages **71.373%**, substantially below hard-mode PRBCD’s 81.560%.

However:

- `q.detach()+0*zh` deliberately makes the edge gradient zero while allowing autograd to run.
- Real VQ’s STE is training-only (`model/vq.py:929–937`); its hard eval path may disconnect the edge gradient entirely.
- PRBCD’s behavior with all-zero gradients depends on its implementation and stochastic search. I did not verify the installed PRBCD source/version.
- The real decoder consumes concatenated codes through a dense linear map (`model/ft_model.py:43–44,142–144`), unlike the proxy’s independent head decoders.

**Conclusion:** gradient obstruction is grounded in the real code; the approximately 12-point gap is a property of this proxy/harness evaluation, not a measured full-STEM effect. Switching the entire model into training mode is an unsafe substitute for an attack-only BPDA wrapper because it changes other model behavior.

**E. Are the commitment runs matched?**

They are closely matched in **training implementation**, but not fully matched experiments.

Verified matches:

- Same `load_cora()`, split masks, and graph preprocessing.
- Same `STEMLite` class and architecture.
- Same initialization seed `split` and k-means seed `split`.
- Same stage-1 CE objective for the baseline.
- LLM training imports the original `stem.fit`, epoch limits, optimizer settings, patience, and commitment implementation.
- Same baseline setting: no trust, pseudo-labels, or consistency.
- Same PRBCD constructor: block 100,000, 100 epochs, 60 resampling epochs.
- Same test-node attack objective and undirected budget formula.
- Same STE attack / hard evaluation arrangement.

Differences found:

| Difference | Evidence | Consequence |
|---|---|---|
| Commitment 0.25 versus 0 | `SV/run.py:74`; `LLM/run.py:46,143` | Intended treatment; summed squared distance per head, not dimension-mean MSE. |
| Three attack budgets versus two | STEM JSON `args.rates`; LLM logs and protocol:77 | STEM’s 5% PRBCD consumes RNG before its 10% attack. |
| Training RNG after early stopping | `SV/run.py:42–54`; shared by both | Different stopping epochs can change the RNG state inherited by attacks despite identical initial seeds. |
| Additional evaluations | `LLM/run.py:172–175`; `SV/run.py:132–139,173–176` | LLM adds random insertion/camouflage; STEM adds transfer, code-change, and hard-mode attacks. Random insertion/camouflage use a separate generator, so they are not themselves the demonstrated global-RNG mismatch. |
| Surrogate-transfer availability | `SV/run.py:156–166`; absent from LLM attack rows | Cross-run robustness is compared using a single adaptive attack rather than matched worst-over-attacks results. |
| Persisted provenance | STEM JSON includes args/full precision; partial LLM logs contain rounded metrics | LLM thread count, exact invocation, package versions, and execution-time code snapshot are not established by the supplied logs. STEM used two threads. |

**Inference:** because both runs share the training implementation, commitment is a plausible explanation for the clean and robustness differences. The **+4.142-point @20% difference is still exploratory**, with attack randomness and checkpoint selection unresolved. No evidence here justifies claiming an isolated four-point causal gain.

**F. Most promising new addition and top-five experiments**

My proposed **new addition relative to the reviewed experiment set** is a **text-margin-conditioned cap on graph corrections**.

Use the edge-free text/VQ tower as an anchor. Let the graph branch contribute a residual, but constrain its maximum class-pair effect according to the anchor’s confidence. Confident text predictions receive a small structural correction budget; ambiguous nodes retain more graph influence. Cap total influence so inserting many neighbors cannot simply increase it.

This is an inference motivated by three observations: the structure-free baseline remains competitive under attack; increasing CTV’s fixed own-vote weight improves its certificates but costs clean accuracy; blanket code-margin regularization failed. Node-dependent structural influence may preserve useful graph corrections while protecting confident text decisions. **Literature novelty has not been established**, and no robustness gain is yet demonstrated.

A falsifiable target: **at least +3 points in paired worst-over-attacks accuracy at 10%, with at most one point of clean loss**, versus a matched STEM-lite baseline. Failure on either condition rejects this implementation.

The cheapest decisive pilot freezes both trained towers and fits only the correction head/cap, comparing against a fixed mixing weight selected on validation. It avoids repeating encoder pretraining. It must include adaptive BPDA through the cap and transfer attacks; success would warrant full-STEM replication.

| Rank | Next experiment | Cheapest decisive design |
|---|---|---|
| **1** | **Repair CTV attack validation** | Retain local edit witnesses, construct actual undirected graphs, update both endpoints, and evaluate accuracy and edit counts. Include the two-node counterexample and tie cases. No encoder retraining is needed if votes are retained. |
| **2** | **Direct commitment ablation** | Coefficients {0, 0.05, 0.25}; reuse the same stage-1/codebook state per split, reset stage-3 RNG, and independently reset attack seeds. Report paired clean and worst-over-attacks differences. |
| **3** | **Text-margin-conditioned graph correction cap** | Frozen towers; validation-selected cap and fixed-mixture control. Use the falsification rule above. This is the proposed new defense addition. |
| **4** | **Actual STEM VQ/BPDA parity test** | On a small frozen real-model instance, verify identical forward predictions, inspect edge gradients, and compare hard, BPDA, and transfer attacks without changing BatchNorm/router evaluation behavior. |
| **5** | **Confirmatory CTV selection** | Select α/T using validation and a graph-constructing attack, then lock settings before test evaluation. Compare certificates against both the prior MLP and the matched edge-free tower, reporting uncertainty. |

The immediate methodological priority is experiment 1: until actual edited graphs reproduce the claimed losses, CTV’s “21-point PRBCD overestimate” and “59% best verified robustness” should be withdrawn.

Codex session ID: 01a15c28-bc9b-7f12-85a4-b20f327e10ea
Resume in Codex: codex resume 01a15c28-bc9b-7f12-85a4-b20f327e10ea
