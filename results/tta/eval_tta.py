import os
import os.path as osp
import sys
import json

sys.argv = [sys.argv[0]]
sys.path.insert(0, "/home/lam23005/STEM-GNN/STEM-GNN")

import yaml
import wandb

from utils.args import get_args_finetune
from utils.others import ensure_finetune_lr
from utils.calibration import fit_temperature
from task.node import eval_node, get_logits_labels
from scripts.missing_feature import _apply_missing_features, _extract_masks_train_valid_test
import finetune
from finetune import dataset2task

REPO_ROOT = "/home/lam23005/STEM-GNN"
PKG_ROOT = osp.join(REPO_ROOT, "STEM-GNN")
OUT_DIR = osp.join(REPO_ROOT, "results", "tta")

os.environ["WANDB_MODE"] = "disabled"

RESULT_PATH = osp.join(OUT_DIR, "tta_result.json")
MISSING_PROB = float(os.environ.get("MISSING_PROB", "0.5"))


def build_params():
    dataset = "cora"
    task = dataset2task[dataset]
    with open(osp.join(REPO_ROOT, "config", "finetune.yaml"), "r") as f:
        default_params = yaml.safe_load(f)
    params = get_args_finetune(default_params=default_params[task][dataset])

    ensure_finetune_lr(params)
    params["data_path"] = osp.join(PKG_ROOT, "..", "data")
    params["pt_model_path"] = osp.join(PKG_ROOT, "..", "ckpts", "pretrain_model")
    params["task"] = task
    params["finetune_dataset"] = dataset
    params["use_params"] = True
    params["debug"] = True

    # Same untuned baseline as Section "Results" in results/encoder_lipschitz_reg.tex:
    # no head/encoder regularization -- this experiment is about TTA/calibration,
    # not about the Lipschitz penalties.
    params["decoder_jac_coeff"] = 0.0
    params["encoder_lip_coeff"] = 0.0

    params["pretrain_path"] = osp.join(PKG_ROOT, "..", "ckpts", "pretrain_model", "default")
    params["pretrain_model_epoch"] = 50

    params["finetune_epochs"] = 60
    params["early_stop"] = 30

    # config/finetune.yaml sets normalize="none" for Cora, which means the
    # encoder's BatchNorm1d layers are never invoked in the forward pass at
    # all (Encoder.encode only calls self.norms[i] when normalize != "none").
    # BatchNorm test-time adaptation is a no-op under that setting by
    # construction, so this experiment deliberately turns batch
    # normalization on to give it something to adapt.
    params["normalize"] = "batch"

    return params


all_rows = []


def on_split_end(idx, task_model, data, dataset, split, labels, params):
    device = next(task_model.parameters()).device

    # 1. Clean, no adaptation -- reference point for this specific trained model
    #    (the post-training weights, not necessarily the exact best-val-epoch
    #    snapshot the main results table reports, since EarlyStopping here only
    #    tracks metrics, not weights -- see utils/early_stop.py).
    clean = eval_node(task_model, data, None, split, labels, params, return_ece=True)

    # 2. Fit temperature once on the CLEAN validation logits.
    val_logits, val_y = get_logits_labels(task_model, data, None, split, labels, params)
    val_mask = split["valid"].to(val_logits.device)
    temperature = fit_temperature(val_logits[val_mask], val_y[val_mask])

    # 3. Apply a missing-feature shift to val+test nodes (same perturbation as
    #    scripts/missing_feature.py's default "valtest" mode).
    train_mask, valid_mask, test_mask = _extract_masks_train_valid_test(split, data, device)
    mask_dict = {"train": train_mask, "valid": valid_mask, "val": valid_mask, "test": test_mask}
    noisy_data = _apply_missing_features(
        data, mask_dict, missing_prob=MISSING_PROB, seed=1, perturb="valtest",
    )

    shifted_none = eval_node(task_model, noisy_data, None, split, labels, params, return_ece=True)
    shifted_bn = eval_node(task_model, noisy_data, None, split, labels, params,
                            bn_adapt=True, return_ece=True)
    shifted_temp = eval_node(task_model, noisy_data, None, split, labels, params,
                              temperature=temperature, return_ece=True)
    shifted_bn_temp = eval_node(task_model, noisy_data, None, split, labels, params,
                                 bn_adapt=True, temperature=temperature, return_ece=True)

    row = {
        "split": idx,
        "temperature": temperature,
        "clean_test": clean["test"], "clean_test_ece": clean["test_ece"],
        "shifted_none_test": shifted_none["test"], "shifted_none_test_ece": shifted_none["test_ece"],
        "shifted_bn_test": shifted_bn["test"], "shifted_bn_test_ece": shifted_bn["test_ece"],
        "shifted_temp_test": shifted_temp["test"], "shifted_temp_test_ece": shifted_temp["test_ece"],
        "shifted_bn_temp_test": shifted_bn_temp["test"], "shifted_bn_temp_test_ece": shifted_bn_temp["test_ece"],
    }
    all_rows.append(row)
    print(f"[TTA] split {idx}: {json.dumps(row)}", flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)


def summarize():
    import numpy as np
    keys = [k for k in all_rows[0].keys() if k not in ("split", "temperature")]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in all_rows])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    summary["temperature"] = {
        "mean": float(np.mean([r["temperature"] for r in all_rows])),
        "std": float(np.std([r["temperature"] for r in all_rows])),
    }
    return summary


def main():
    params = build_params()
    wandb.init(project="STEM-GNN-Finetune", name="CORA - TTA", config=params,
               mode="disabled", tags=[params["setting"]])
    params = dict(wandb.config)
    ensure_finetune_lr(params)

    finetune.run(params, on_split_end=on_split_end)

    summary = summarize()
    print("\n=== TTA SUMMARY (mean +/- std over splits) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump({"rows": all_rows, "summary": summary, "missing_prob": MISSING_PROB}, f, indent=2)

    print("TTA_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
