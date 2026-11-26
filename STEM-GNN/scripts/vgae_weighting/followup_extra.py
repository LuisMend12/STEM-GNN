"""followup.py on extra node-classification datasets (Wiki-CS), descriptive only.

followup.py itself is left unchanged: the Cora and Pubmed cells that are queued or resuming check its hash, and
editing it would split their configuration. This wrapper registers the extra datasets and folds its own source
into the code hash, then runs followup.main() with the same recipes, seeds, arms and attack suite.

Usage (from repo root):
  python STEM-GNN/scripts/vgae_weighting/followup_extra.py --dataset wikics --part A --split 0 --seed 0
"""
import hashlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vgae_followup", os.path.join(HERE, "followup.py"))
fu = importlib.util.module_from_spec(_spec)
sys.modules["vgae_followup"] = fu  # so checkpoints pickle under the same module name as followup.py's
_spec.loader.exec_module(fu)

EXTRA = {"wikics": (os.path.join(os.path.dirname(fu.fx.CORA), "..", "wikics", "wikics.pt"), (3, 3))}

for name, (path, cells) in EXTRA.items():
    fu.DATA[name] = path
    fu.A_CELLS[name] = cells

_base_hash = fu.code_hash


def code_hash():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256((_base_hash() + hashlib.sha256(f.read()).hexdigest()).encode()).hexdigest()


fu.code_hash = code_hash

if __name__ == "__main__":
    if "--dataset" not in sys.argv or sys.argv[sys.argv.index("--dataset") + 1] not in EXTRA:
        raise SystemExit(f"use followup.py for cora/pubmed; this wrapper runs only {sorted(EXTRA)}")
    fu.main()
