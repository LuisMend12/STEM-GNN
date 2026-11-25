The fixes remove the original serialization blocker. I found no new launch-blocking bug, but several analysis safeguards remain incomplete.

File abbreviations below: `B` = `STEM-GNN/scripts/capped/baselines.py`; `AB` = the same directory’s `analyze_baselines.py`; `A` = `analyze.py`; `report` and `protocol` = `experiments/capped-correction/analysis.md` and `baselines_protocol.md`.

| # | Status | Evidence |
|---|---|---|
| 1 | **FIXED** | `B:44` registers `ar_models` before execution; `B:274–286` adds a fresh-process checkpoint probe. |
| 2 | **FIXED** | `protocol:55–66` defines a numerical matched-clean baseline envelope and limits the interpretation of one successful CAP point. |
| 3 | **FIXED** | `AB:92` applies Holm to frontier tests; `protocol:82` explicitly excludes exploratory pairwise searches from claims. |
| 4 | **FIXED** | `protocol:72–74` restricts Family B to robustness of the fixed selected arm. |
| 5 | **PARTIAL** | `protocol:30–38` expands ENS to five weights and explains scale dependence and training differences, but adds neither validation-selected weights nor calibration. |
| 6 | **FIXED** | `B:311–314` records source/data hashes and runtime configuration; `B:190–202` compares configuration and saves before attacks. |
| 7 | **PARTIAL** | `B:138,217` enforces extended-union monotonicity and per-cell reproduction, but `AB:61–64` independently checks only original-union worst-node@20%. |
| 8 | **FIXED** | `B:188,136` captures original graph membership explicitly and uses it for reproduction scoring. |
| 9 | **FIXED** | `A:34–42` defines constant-difference p-values and rejects nonfinite p-values before Holm. |
| 10 | **FIXED** | `report:87–90` states dataset dependence; `AB:116–120` reports split sensitivity. |
| 11 | **FIXED** | `report:55–57` acknowledges RES’s descriptive gain and CAP’s additional gain with lower clean accuracy. |
| 12 | **FIXED** | `report:63–77` uses validation scores and winning margins and removes the unsupported alternative-selection claim. |
| 13 | **FIXED** | `report:30–31,49–54` corrects dominance wording and bounds the interpretation of attack slack. |
| 14 | **FIXED** | `report:26` explicitly identifies H81’s upper CI and the +1 threshold. |
| 15 | **PARTIAL** | `AB:54–60` checks exact arm/cell membership and loads originals once, but its finiteness check omits certificates. |

New issues and remaining defects:

- **SHOULD-FIX — Family F shrinks the prespecified five-test family.** `AB:91–92` excludes caps with fewer than three eligible cells before Holm. For example, one available p=.02 becomes .02 rather than .10 with four unavailable tests retained as p=1. **Fix:** pass all five caps to Holm, assigning p=1 to untestable caps; continue reporting their results as unavailable.

- **SHOULD-FIX — Analysis reproduction checks are narrower than promised.** `AB:61–64` checks only @20% worst-node for `OURS`, excluding MLP-M, clean accuracy, certificates, and @10%. The cluster runner checks these, but analysis does not independently enforce the protocol’s promise. **Fix:** compare every original arm at both budgets plus clean and certificate values, and assert extended-union monotonicity again.

- **SHOULD-FIX — Certificate finiteness is unchecked.** `AB:56–57` accepts `cert_test=NaN`; I reproduced this. Valid `None` certificates do not crash this assertion because certificate keys are excluded altogether. **Fix:** explicitly validate required numeric metrics and require finite certificates for capped arms, allowing `None` only where expected.

- **NIT — Certificate mismatch produces a TypeError instead of the intended diagnostic.** `B:154` handles two `None` values correctly, but subtracts when exactly one side is `None`. **Fix:** branch on either value being `None` before numerical comparison.

Other requested checks:

- Module registration order is correct for both entry-point execution and the fresh-process probe. `LogMix` has a resolvable module alias (`B:47,80`), though checkpoint payloads exclude it (`B:199`).
- Resume reconstructs ensembles from the loaded original models. `source_sha` is computed before checkpoint loading and compared with the saved configuration; there is no ordinary stale-hash ordering error.
- The `orig_` mapping is correct. Copied CAP* rows satisfy the exact arm/cell assertion, and their copied original-union metrics support the drift check.
- Selfcheck models reach the serialization probe in eval mode: training ends in eval, the gradient loop calls eval, and PRBCD calls eval. The probe imports the entry point before unpickling.
- Envelope eligibility, masked `-inf`, and exclusion/counting of empty cells work as specified. I exercised fully eligible and entirely ineligible cases.

`analyze.py` reproduces `analysis_output.txt` exactly, excluding its trailing `ANALYSIS_EXIT=0` marker. The report’s numerical summaries match. All five validation-table rows round correctly; winning margins are **minimum 1.300001 points, mean 2.726667 points**.

I ran numerical and synthetic analysis checks. Full baseline import, fresh-process serialization, and SAGE/RUNGLite output comparisons could not execute here because `torch_geometric` is unavailable; those conclusions are based on code inspection.

**GO for launching baselines.py on the cluster.**