**NO-GO.** Wiki-CS preparation currently fails before embeddings are generated. The extension also needs a feasibility gate before committing to its full attack and random-injection grids.

### BLOCKER

1. **Metadata labels are strings, so preparation fails.** In [prep_wikics.py](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/prep_wikics.py:41), `meta_labels` contains names such as `"Distributed computing architecture"`, while `pyg.y` contains integer class IDs. `torch.tensor(meta_labels)` raises an error.

   Replace lines 41–42 with:

   ```python
   label_to_id = {name: int(key) for key, name in meta["labels"].items()}
   if len(label_to_id) != len(meta["labels"]):
       raise SystemExit("metadata label names are not unique")
   try:
       meta_y = torch.tensor([label_to_id[label] for label in meta_labels],
                             dtype=pyg.y.dtype)
   except KeyError as exc:
       raise SystemExit(f"unknown metadata node label: {exc}") from exc
   if not torch.equal(meta_y, pyg.y.cpu()):
       raise SystemExit("metadata label sequence does not match PyG WikiCS labels")
   ```

   This checks that the **label sequence** agrees, but it cannot detect a permutation among nodes of the same class. The upstream dataset says its metadata and vectorized data share node ordering; that provenance supports the alignment, but the label comparison does not independently prove it. [Wiki-CS dataset repository](https://github.com/pmernyei/wiki-cs-dataset)

### SHOULD-FIX

1. **Add a Wiki-CS feasibility gate before the full run.** The amendment currently says the setup applies unchanged without addressing its much larger graph. For the commonly reported undirected PyG graph, approximately 215,863 undirected edges imply 10% and 20% budgets of about **21,586 and 43,172 edits**, subject to measuring the actual loaded graph. Both are below the standard 100,000 and strong 200,000 candidate block sizes; the 20% budget nevertheless occupies roughly 43% and 22% of those blocks. The PyG implementation requires `block_size > budget`, and the remaining margin and attack effectiveness deserve checking. [PyG attack implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/contrib/nn/models/rbcd_attack.html)

   Each Part A cell retains **178 attack graphs** for scoring. Their edge-index tensors alone are roughly 1.2–1.5 GB in total at this graph size, before Python overhead, models, intermediate activations, candidate blocks and working copies. Part R’s largest level adds about 94.7% of the original undirected edge count. Runtime and peak memory cannot be established from inspection.

   Add this text to Amendment 1:

   > **Wiki-CS feasibility gate.** Before launching the full grid, record the loaded graph’s undirected edge count, the two integer attack budgets, and the five injection counts. Verify that each attack budget is strictly below its candidate block size. Run one Part A cell and one Part R cell at the largest injection level, recording peak memory, wall time, per-attack timing, and graph-file size. Do not launch the remaining Wiki-CS cells unless these measurements fit the allocated resources. Any change to the attack suite or resource-driven stopping rule will be recorded as an amendment before inspecting Wiki-CS accuracy; results from an altered suite will be labeled accordingly.

2. **Describe the node-order check accurately.** Replace the data bullet at [protocol.md](/G:/broswer%20download/STEM-GNN/experiments/vgae-followup/protocol.md:205) with:

   > - node text from the repo’s OFA recipe; metadata and PyG label sequences are compared after mapping label names to integer IDs. This detects label-sequence mismatches but cannot detect permutations within a class; the upstream dataset documents shared metadata and vectorized-data ordering.

3. **Prevent incomplete Wiki-CS results from interrupting the registered analysis.** [analyze_followup.py](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:118) attempts full Wiki-CS collection as soon as any Part A result exists. An incomplete Wiki-CS grid then raises an assertion before Cora/Pubmed claims are printed. Replace lines 118–120 with:

   ```python
   for ds in DESCRIPTIVE_ONLY:
       a_files = glob.glob(os.path.join(RESULTS, f"{ds}_A_*[0-9a-f].json"))
       r_files = glob.glob(os.path.join(RESULTS, f"{ds}_R_*[0-9a-f].json"))
       if len(a_files) == len(A_CELLS[ds]) and len(r_files) == LEVELS * GRAPHS:
           data[ds] = collect(ds)
       elif a_files or r_files:
           print(f"{ds}: descriptive grid incomplete; skipping "
                 f"({len(a_files)} A files, {len(r_files)} R files)")
   ```

   `collect` still validates the complete grid and configurations once those file counts are reached.

### NIT

Add explicit mask-shape checks before indexing in [prep_wikics.py](/G:/broswer%20download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/prep_wikics.py:48), so an unexpected PyG input fails with a useful message:

```python
if (pyg.train_mask.shape != (n, 20) or pyg.val_mask.shape != (n, 20)
        or pyg.test_mask.shape != (n,)):
    raise SystemExit("unexpected Wiki-CS train/validation/test mask shapes")
```

The intended split construction is otherwise correct: 20 train and validation columns become lists, and the shared one-dimensional test mask is cloned for each split. The node text matches the repository’s OFA recipe, and the resulting `Data` fields match `load_data`. The loader symmetrizes and coalesces edges and removes self-loops.

The wrapper registers Wiki-CS before calling `followup.main()`, and `main()` resolves the patched `code_hash` from the module’s globals. Its registration under `vgae_followup` is consistent with the classes’ declared pickle module and the runner’s fresh-process loading pattern. It only adds Wiki-CS entries to the in-memory dictionaries; it does not alter Cora/Pubmed configurations. The current `followup.py` SHA-256 is `523bff077539c5ab0462b0a3b54311bd68d5ee261bca9ae125c5596ce1852d96`; I did not find a separate recorded baseline hash to independently establish historical identity.

The analysis adds hypotheses and guardrails only in the explicit Cora and Pubmed branches, so Wiki-CS stays out of the Holm family and printed claims. Its paired summaries remain labeled descriptive. I performed a static review only: no experiment, preparation, checkpoint round-trip, or resource measurement was run, and no files were changed.