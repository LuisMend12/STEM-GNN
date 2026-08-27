import os
import os.path as osp
import sys
import json

sys.argv = [sys.argv[0]]  # keep argparse inside get_args_finetune from seeing our own CLI args
sys.path.insert(0, "/home/lam23005/STEM-GNN/STEM-GNN")

import yaml
import wandb
import optuna

from utils.args import get_args_finetune
from utils.others import ensure_finetune_lr
from finetune import run, dataset2task

REPO_ROOT = "/home/lam23005/STEM-GNN"
PKG_ROOT = osp.join(REPO_ROOT, "STEM-GNN")
HPO_DIR = osp.join(REPO_ROOT, "results", "hpo")

os.environ["WANDB_MODE"] = "disabled"

STORAGE = f"sqlite:///{osp.join(HPO_DIR, 'optuna_baseline.db')}"
STUDY_NAME = "stemgnn_cora_baseline"
N_SEARCH_TRIALS = int(os.environ.get("N_TRIALS", "12"))


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

    # Reduced search-phase budget: single split, short training, for tractable trial cost
    # on a single CPU core. Best config is re-validated on the full 10-split protocol after.
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


def objective(trial):
    params = build_base_params()
    params["finetune_lr"] = trial.suggest_float("finetune_lr", 1e-4, 1e-2, log=True)
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    params["normalize"] = trial.suggest_categorical("normalize", ["none", "batch"])

    best = run_trial(params)
    trial.set_user_attr("test_mean", best["test"]["mean"])
    trial.set_user_attr("test_std", best["test"]["std"])
    trial.set_user_attr("train_mean", best["train"]["mean"])
    return best["val"]["mean"]


def run_search():
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_SEARCH_TRIALS - completed)
    print(f"[HPO] {completed} trials already completed, running {remaining} more "
          f"(target {N_SEARCH_TRIALS}).", flush=True)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


def run_finalize(study):
    print("\n=== OPTUNA SEARCH DONE ===", flush=True)
    print("Best val (single split):", study.best_value, flush=True)
    print("Best params:", study.best_trial.params, flush=True)
    print("Best trial extra:", study.best_trial.user_attrs, flush=True)

    result_path = osp.join(HPO_DIR, "optuna_baseline_result.json")
    if osp.exists(result_path) and "final_all_splits" in json.load(open(result_path)):
        print("[HPO] Final confirmation already recorded, skipping re-run.", flush=True)
        return

    # Final confirmation: re-run best hyperparameters across all 10 Cora splits
    # (the project's normal protocol) with a slightly larger epoch budget.
    params = build_base_params()
    params.update(study.best_trial.params)
    params.pop("finetune_seed", None)  # use all splits, not just split 0
    params["finetune_epochs"] = 60
    params["early_stop"] = 30
    final = run_trial(params)

    print("\n=== FINAL CONFIRMATION (all 10 splits, best hyperparameters) ===", flush=True)
    print(json.dumps({"best_params": study.best_trial.params, "final": final}, indent=2), flush=True)

    with open(result_path, "w") as f:
        json.dump({
            "best_params": study.best_trial.params,
            "search_val_single_split": study.best_value,
            "final_all_splits": final,
            "trials": [
                {"number": t.number, "params": t.params, "value": t.value, "user_attrs": t.user_attrs}
                for t in study.trials
            ],
        }, f, indent=2)

    print("OPTUNA_ALL_DONE", flush=True)


def main():
    study = run_search()
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed >= N_SEARCH_TRIALS:
        run_finalize(study)
    else:
        print(f"[HPO] Only {completed}/{N_SEARCH_TRIALS} trials completed; not finalizing yet.", flush=True)


if __name__ == "__main__":
    main()
