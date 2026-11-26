**NO-GO.** The Cora classification formulas match Amendment 2, but the diagnostic can silently classify incomplete graph sets, and the registered separate Pubmed and Wiki-CS diagnostics are not delivered. The interim path also omits an existing protocol diagnostic.

**BLOCKER — Validate the complete graph set before scoring.** In [q1_diagnostic.py](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/q1_diagnostic.py:41), missing graph files are silently accepted. A missing family is skipped; [q1_classification](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:134) then treats it as having no large drop. That can produce a false robustness classification. Duplicate payload keys can also overwrite one another.

Replace the graph-loading block in `cell`, from `graphs = {}` through `graphs[d["key"]] = d["g"]`, with:

```python
    budgets = {r: int(r * len(fu.cr.undirected(ei))) for r in fu.ls.RATES}
    planned = {(target, name, r)
               for r in budgets
               for target, name, _, _ in fu.jobs_for(state["models"], state["sur"])}
    graphs = {}
    for filename in sorted(os.listdir(gdir)):
        if not filename.endswith(".pt"):
            continue
        path = os.path.join(gdir, filename)
        d = torch.load(path, weights_only=False)
        key = d["key"]
        if (key in graphs or key not in planned
                or d.get("config") != state["config"]
                or d.get("checkpoint_id") != state["checkpoint_id"]
                or fu.cr.validate(ei, d["g"], x.size(0), budgets[key[2]]) != d["edits"]):
            raise RuntimeError(f"{path}: duplicate, unexpected, invalid, or mismatched graph")
        graphs[key] = d["g"]
    if set(graphs) != planned:
        raise RuntimeError(f"{gdir}: missing graph keys: {sorted(planned - set(graphs))}")
    graphs = {k: g for k, g in graphs.items() if k[2] <= RATE}
```

Also make the consumer fail closed if its JSON is missing or incomplete. Replace the missing-file branch in `q1_classification` with:

```python
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run q1_diagnostic.py before analysis")
```

After the cell-key assertion, add:

```python
    expected = {"S-VG": set(Q1_FAMILIES), "S0": {"greedy", "gcn"}}
    for cell, result in q1.items():
        for arm, families in expected.items():
            if set(result[arm]["families"]) != families:
                raise ValueError(f"{cell} {arm}: incomplete or unexpected Q1 families")
```

**BLOCKER — Complete the registered separate dataset diagnostics.** Amendment 2 says Pubmed and Wiki-CS receive the same diagnostics, reported separately. [q1_diagnostic.py](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/q1_diagnostic.py:78) accepts only Cora and Pubmed, while [analyze_followup.py](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:120) reads and reports only Cora’s Q1 JSON. Thus even the generated Pubmed summary is unused. Add Wiki-CS loading through the existing `followup_extra.py` registration, and generalize the diagnostic reporting to consume each completed dataset’s summary. Keep Wiki-CS descriptive and outside the Holm family; do not treat its result as H110.

**SHOULD-FIX — Preserve the original sensitivity diagnostic in interim mode.** The return at [analyze_followup.py:249](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:249) skips the protocol’s existing combined weight-aware sensitivity output, including its cell values. Move the existing print at lines 276–278 immediately before that return and remove its later copy. The new family-by-family Q1 outputs do not replace that registered combined diagnostic.

**SHOULD-FIX — Tie diagnostic summaries to the results being analyzed.** The summary JSON contains neither checkpoint identities nor configuration provenance, so an old summary with the same 15 cell keys can be combined with newer H110 results. Save each cell’s checkpoint ID and configuration identity in the summary, then verify them against the current checkpoints when loading it. Without that check, the corrected graph validation protects generation but does not protect subsequent analysis from a stale summary.

**NIT — Set evaluation mode explicitly.** After `model = state["models"][arm]` at [q1_diagnostic.py:50](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/q1_diagnostic.py:50), insert:

```python
        model.eval()
```

This makes the intended inference mode explicit; `torch.no_grad()` alone does not disable training-mode behavior.

The remaining requested checks are consistent on static inspection: graph keys use `(target, name, rate)` correctly; `<= 0.2` includes both 0.1 and 0.2; the base and family selectors match the amendment; incremental drops have the correct negative sign and are converted from fractions to points once. The H110 thresholds, clean guardrail, positive split-mean test, and interim `min(1, 8p)` bound are correctly applied. The non-interim path still constructs all eight hypotheses and passes them to Holm. This review was static; no analysis or tests were run.