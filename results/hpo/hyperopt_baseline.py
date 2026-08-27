import os
import os.path as osp
import sys
import json
import pickle

sys.argv = [sys.argv[0]]  # keep argparse inside get_args_finetune from seeing our own CLI args
sys.path.insert(0, "/home/lam23005/STEM-GNN/STEM-GNN")

import numpy as np
import yaml
import wandb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

from utils.args import get_args_finetune
from utils.others import ensure_finetune_lr
from finetune import run, dataset2task

REPO_ROOT = "/home/lam23005/STEM-GNN"
PKG_ROOT = osp.join(REPO_ROOT, "STEM-GNN")
HPO_DIR = osp.join(REPO_ROOT, "results", "hpo")

os.environ["WANDB_MODE"] = "disabled"

TRIALS_PATH = osp.join(HPO_DIR, "hyperopt_baseline_trials.pkl")
RESULT_PATH = osp.join(HPO_DIR, "hyperopt_baseline_result.json")
N_SEARCH_TRIALS = int(os.environ.get("N_TRIALS", "12"))

# Same search space as results/hpo/optuna_baseline.py, so the two searches are
# directly comparable.
SPACE = {
    "finetune_lr": hp.loguniform("finetune_lr", np.log(1e-4), np.log(1e-2)),
    "dropout": hp.uniform("dropout", 0.0, 0.5),
    "normalize": hp.choice("normalize", ["none", "batch"]),
}


def build_base_params():
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

    # Baseline: no Lipschitz-style regularization on either head or encoder.
    params["decoder_jac_coeff"] = 0.0
    params["encoder_lip_coeff"] = 0.0

    # Non-MoE checkpoint, matches the plain SAGE encoder architecture.
    params["pretrain_path"] = osp.join(PKG_ROOT, "..", "ckpts", "pretrain_model", "default")
    params["pretrain_model_epoch"] = 50

    # Same reduced search-phase budget as the Optuna script: single split, short
    # training, for tractable trial cost on a single CPU core.
    params["finetune_epochs"] = 30
    params["early_stop"] = 15
    params["finetune_seed"] = 0

    return params


def run_trial(params):
    wandb.init(
        project="STEM-GNN-Finetune",
        name="{} - Finetune".format(str.upper(params["finetune_dataset"])),
        config=params,
        mode="disabled",
        tags=[params["setting"]],
    )
    params = dict(wandb.config)
    ensure_finetune_lr(params)
    best = run(params)
    wandb.finish()
    return best


def objective(hp_choice):
    params = build_base_params()
    params["finetune_lr"] = hp_choice["finetune_lr"]
    params["dropout"] = hp_choice["dropout"]
    params["normalize"] = hp_choice["normalize"]

    best = run_trial(params)
    return {
        "loss": -best["val"]["mean"],  # hyperopt minimizes
        "status": STATUS_OK,
        "val_mean": best["val"]["mean"],
        "test_mean": best["test"]["mean"],
        "test_std": best["test"]["std"],
        "train_mean": best["train"]["mean"],
        "params": hp_choice,
    }


def load_trials():
    if osp.exists(TRIALS_PATH):
        with open(TRIALS_PATH, "rb") as f:
            return pickle.load(f)
    return Trials()


def save_trials(trials):
    with open(TRIALS_PATH, "wb") as f:
        pickle.dump(trials, f)


def run_search():
    trials = load_trials()
    print(f"[HPO] {len(trials.trials)} trials already completed.", flush=True)
    # Advance one trial at a time and persist after each, so a mid-run kill
    # loses at most the trial in flight (mirrors the Optuna script's per-trial
    # sqlite durability). This was added after two earlier long-running
    # background experiments in this session were silently killed by the
    # sandbox environment recycling mid-run.
    while len(trials.trials) < N_SEARCH_TRIALS:
        target = len(trials.trials) + 1
        fmin(
            fn=objective,
            space=SPACE,
            algo=tpe.suggest,
            max_evals=target,
            trials=trials,
            rstate=np.random.default_rng(0),
            show_progressbar=False,
        )
        save_trials(trials)
        last = trials.trials[-1]["result"]
        print(f"[HPO] trial {target} done: val={last['val_mean']:.2f} "
              f"test={last['test_mean']:.2f} params={last['params']}", flush=True)
    return trials


def run_finalize(trials):
    best_result = min(trials.results, key=lambda r: r["loss"])
    best_params = best_result["params"]

    print("\n=== HYPEROPT SEARCH DONE ===", flush=True)
    print("Best val (single split):", best_result["val_mean"], flush=True)
    print("Best params:", best_params, flush=True)

    if osp.exists(RESULT_PATH):
        existing = json.load(open(RESULT_PATH))
        if "final_all_splits" in existing:
            print("[HPO] Final confirmation already recorded, skipping re-run.", flush=True)
            return

    params = build_base_params()
    params.update(best_params)
    params.pop("finetune_seed", None)  # use all splits, not just split 0
    params["finetune_epochs"] = 60
    params["early_stop"] = 30
    final = run_trial(params)

    print("\n=== FINAL CONFIRMATION (all 10 splits, best hyperparameters) ===", flush=True)
    print(json.dumps({"best_params": best_params, "final": final}, indent=2), flush=True)

    with open(RESULT_PATH, "w") as f:
        json.dump({
            "best_params": best_params,
            "search_val_single_split": best_result["val_mean"],
            "final_all_splits": final,
            "trials": [
                {"params": t["result"].get("params"), "loss": t["result"].get("loss"),
                 "val_mean": t["result"].get("val_mean"), "test_mean": t["result"].get("test_mean")}
                for t in trials.trials
            ],
        }, f, indent=2)

    print("HYPEROPT_ALL_DONE", flush=True)


def main():
    trials = run_search()
    if len(trials.trials) >= N_SEARCH_TRIALS:
        run_finalize(trials)


if __name__ == "__main__":
    main()
