# Learning Progress

No learning events recorded yet.

## Pending decision: dataset extension
- Learner asked (2026-11-25) to also run on PubMed, Wiki-CS, Arxiv, WN18RR, FB15K237, HIV, PCBA, plus an "Avg." column.
- Stage: awaiting reasoning (Build checkpoint).
- Facts given: only Cora/Pubmed have ready .pt files; Wiki-CS and Arxiv need download and text encoding (Arxiv 169k nodes, very costly attacks); WN18RR/FB15K237 are link tasks and HIV/PCBA are graph tasks, needing new threat models and metrics; averaging mixes accuracy, AUC and AP.
- Awaiting: learner's scope, threat model and metric per task type, meaning of Avg., and Arxiv cost handling.
- Learner said "try our experimental setup on these datasets". Wiki-CS (node task, setup applies unchanged) was approved at an Implementation checkpoint and is being implemented as descriptive only (Amendment 1). Still awaiting: the threat model and metric for link tasks (WN18RR, FB15K237) and graph tasks (HIV, PCBA), Arxiv's cost, and the meaning of Avg.

## Pending decision
- 2026-11-25: learner asked for Codex to draft answers to all open questions; Codex draft saved at experiments/vgae-followup/design_options_codex_draft.md (proposals only, not learner decisions).
- 2026-11-26: Q1 resolved: learner adopted Codex's option B criteria verbatim (no own reasoning given), recorded as protocol Amendment 2 before Cora accuracy was inspected; AI chose the implementation details (per-node recompute from checkpoints; interim Bonferroni x8 bound). Q2-Q5 still awaiting the learner's choice.
- Decision: criteria for telling attack weakness (gradient obstruction) from real robustness in the VGAE follow-up, before any smoke or grid run.
- Stage: awaiting reasoning (Build checkpoint).
- Current AI-proposed plan: H110 (S-VG − S0 ≥ +2 on the expanded union) is the deciding test; the drop in S-VG's own-attack worst-node when weight-aware attacks are added is a descriptive flag (≤ −3 points).
- Awaiting: the learner's own criteria and how each new attack type would discriminate the two explanations.
- Learner chose (2026-11-25) to launch the smoke cells first and set the criteria later; still awaiting reasoning. Smoke cells 34128298-34128301 launched.
- Full grid queued 2026-11-25 (Rc 34146014, Rp 34146015, Ac 34146016/17, Ap 34146018/19/20, Pubmed smoke resume 34146021/22). Accuracy outcomes are not being inspected until the criteria are set; only reproduction equality and timing were checked.

Note: learning mode started 2026-11-25, after the VGAE follow-up protocol and code
(experiments/vgae-followup/) were already written by the AI and Codex-reviewed. Those
design choices are AI-proposed, not learner decisions.
