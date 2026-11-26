"""Q1 diagnostic (protocol Amendment 2): per-family incremental worst-node drops for S-VG and S0.

Per-node correctness is not stored in the result JSONs, so this reloads each finished Part A cell's saved
models and graph files and recomputes it. Every planned graph must be present, valid and bound to its
checkpoint. The summary records the SHA-256 of each cell's result JSON, so the analysis can refuse a stale
summary. Reads only; writes one summary JSON per dataset.

Usage (from repo root):  python STEM-GNN/scripts/vgae_weighting/q1_diagnostic.py --dataset cora
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# followup_extra loads followup.py as module 'vgae_followup' (for unpickling) and registers Wiki-CS
_spec = importlib.util.spec_from_file_location("vgae_followup_extra", os.path.join(HERE, "followup_extra.py"))
_extra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extra)
fu = _extra.fu

RATE = 0.2
ARMS = ("S-VG", "S0")
TRUE_GRADIENT = tuple(n for n, *_ in fu.STRONG if n != "greedy")
STD = tuple(a[0] for a in fu.ls.ATTACKS)
FAMILIES = ("greedy", "gcn", "bpda_flat", "detach", "twin")
LR_SWEEP = {"x0.1": ("s_lr01",), "x1": ("s_lr1a", "s_lr1b"), "x10": ("s_lr10a", "s_lr10b")}


def family_keys(arm, fam, keys):
    if fam == "gcn":
        return [k for k in keys if k[0] == "gcn"]
    return [k for k in keys if k[0] == arm and k[1] == fam]


def load_graphs(state, gdir, x, ei):
    budgets = {r: int(r * len(fu.cr.undirected(ei))) for r in fu.ls.RATES}
    planned = {(t, name, r) for r in budgets for t, name, _, _ in fu.jobs_for(state["models"], state["sur"])}
    graphs, fingerprint = {}, {}
    for filename in sorted(os.listdir(gdir)):
        if not filename.endswith(".pt"):
            continue
        path = os.path.join(gdir, filename)
        d = torch.load(path, weights_only=False)
        key = d["key"]
        if (key in graphs or key not in planned or d.get("config") != state["config"]
                or d.get("checkpoint_id") != state["checkpoint_id"]
                or fu.cr.validate(ei, d["g"], x.size(0), budgets[key[2]]) != d["edits"]):
            raise RuntimeError(f"{path}: duplicate, unexpected, invalid, or mismatched graph")
        graphs[key] = d["g"]
        fingerprint[fu.key_name(key)] = (list(d["edits"]), d["seconds"])
    if set(graphs) != planned:
        raise RuntimeError(f"{gdir}: missing graph keys: {sorted(planned - set(graphs))}")
    return {k: g for k, g in graphs.items() if k[2] <= RATE}, fingerprint


def cell(ds, split, seed, x, ei, y, te):
    (ckpt,) = glob.glob(os.path.join(fu.OUT, f"{ds}_A_s{split}_e{seed}_*_ckpt.pt"))
    result_json = ckpt[: -len("_ckpt.pt")] + ".json"
    with open(result_json, "rb") as f:
        result_bytes = f.read()
    result = json.loads(result_bytes)
    with open(ckpt, "rb") as f:
        checkpoint_sha = hashlib.sha256(f.read()).hexdigest()
    state = torch.load(ckpt, weights_only=False)
    graphs, fingerprint = load_graphs(state, ckpt[:-3] + "_graphs", x, ei)
    # followup.py's result JSON has no checkpoint id; its per-graph edit counts and attack timings equal the
    # values stored in the checkpoint-bound graph files only if it was produced from this checkpoint's graphs
    config = json.loads(json.dumps(state["config"]))  # tuples -> lists, as in the JSON
    stored = {k: (list(v), result["attack_seconds"][k]) for k, v in result["edits"].items()}
    if ({k: v for k, v in result["env"].items() if k != "seconds"} != config
            or any((row["split"], row["seed"]) != (split, seed) for row in result["rows"])
            or stored != fingerprint):
        raise ValueError(f"{result_json}: result does not match this checkpoint or cell")
    out = {"result_json": os.path.basename(result_json), "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
           "checkpoint_id": state["checkpoint_id"], "checkpoint_sha256": checkpoint_sha}
    for arm in ARMS:
        model = state["models"][arm]
        model.eval()
        with torch.no_grad():
            clean = (model(x, ei).argmax(-1) == y)[te]
            corr = {k: (model(x, graphs[k]).argmax(-1) == y)[te] for k in graphs if k[0] in (arm, "gcn")}
        base = [k for k in corr if k[0] == arm and k[1] in STD + TRUE_GRADIENT]
        ok_base = torch.stack([clean] + [corr[k] for k in base]).all(0)
        row = {"base_worst_node": ok_base.float().mean().item(), "base_graphs": len(base), "families": {}}
        for fam in FAMILIES:
            fk = family_keys(arm, fam, list(corr))
            if not fk:
                continue
            ok = ok_base & torch.stack([corr[k] for k in fk]).all(0)
            row["families"][fam] = {"drop": ok.float().mean().item() - row["base_worst_node"],
                                    "newly_broken": int((ok_base & ~ok).sum()), "graphs": len(fk)}
        std_only = torch.stack([clean] + [corr[k] for k in corr if k[0] == arm and k[1] in STD]).all(0)
        row["lr_sweep"] = {}
        for lr, names in LR_SWEEP.items():
            ok = std_only & torch.stack([corr[k] for k in corr if k[0] == arm and k[1] in names]).all(0)
            row["lr_sweep"][lr] = {"drop_vs_std": ok.float().mean().item() - std_only.float().mean().item(),
                                   "single_graph_acc": [corr[k].float().mean().item()
                                                        for k in corr if k[0] == arm and k[1] in names]}
        out[arm] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(fu.A_CELLS), default="cora")
    args = ap.parse_args()
    torch.set_num_threads(8)
    x, ei, y, trm, vam, tem = fu.load_data(args.dataset)
    splits, seeds = fu.A_CELLS[args.dataset]
    res = {}
    for s in range(splits):
        for e in range(seeds):
            res[f"{s}_{e}"] = cell(args.dataset, s, e, x, ei, y, tem[s])
            print(json.dumps({"cell": [s, e], **{a: {f: round(100 * v["drop"], 2) for f, v in res[f"{s}_{e}"][a]["families"].items()}
                                                   for a in ARMS}}), flush=True)
    path = os.path.join(fu.OUT, f"q1_diagnostic_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
