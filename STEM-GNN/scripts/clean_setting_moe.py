"""MoE routing diagnostics on the standard (clean, non-OOD) finetune setting.

Reuses STEM-GNN's normal train/val/test splits from get_finetune_graph (no
degree bucketing), trains with task/node.py's ft_node/eval_node exactly like
finetune.py, and reports the same per-layer router collapse diagnostics
(entropy_of_avg, max_usage, top1_frac) used in scripts/degree_shift_ood.py,
computed once on the held-out test set after early stopping.
"""

import os.path as osp
from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.ft_model import TaskModel
from model.vq import VectorQuantize
from task.node import ft_node, eval_node
from utils.args import get_args_finetune
from utils.others import freeze_params, load_params, seed_everything, ensure_finetune_lr, get_pretrain_run_id
from utils.preprocess import pre_node

import warnings
import wandb

def _entropy(p: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum().item())


def _compute_routing_cache(model, data) -> List[torch.Tensor]:
    was_training = model.training
    model.eval()
    encoder = model.encoder
    encoder.enable_router_cache(True)
    with torch.no_grad():
        x = data.node_text_feat
        edge_index = data.edge_index
        edge_attr = data.edge_text_feat[data.xe]
        encoder(x, edge_index, edge_attr)
    cache = encoder.get_router_cache(reset=True)
    encoder.enable_router_cache(False)
    if was_training:
        model.train()
    return cache


DATASET2TASK = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
}


def compute_test_routing_stats(
    router_cache: List[torch.Tensor],
    moe_layer_indices: List[int],
    test_mask: torch.Tensor,
) -> Dict[int, Dict[str, object]]:
    stats: Dict[int, Dict[str, object]] = {}
    for cache_idx, layer_idx in enumerate(moe_layer_indices):
        weights = router_cache[cache_idx][test_mask.to(router_cache[cache_idx].device)]
        num_experts = weights.size(1)
        avg_prob = weights.mean(dim=0)
        entropy_of_avg = _entropy(avg_prob)
        clamped = weights.clamp_min(1e-12)
        node_entropies = -(clamped * clamped.log()).sum(dim=1)
        mean_node_entropy = float(node_entropies.mean().item())
        top1 = weights.argmax(dim=1)
        top1_frac = torch.stack([(top1 == e).float().mean() for e in range(num_experts)])
        stats[layer_idx] = {
            "avg_prob": avg_prob.cpu().tolist(),
            "entropy_of_avg": entropy_of_avg,
            "mean_node_entropy": mean_node_entropy,
            "top1_frac": top1_frac.cpu().tolist(),
            "max_usage": float(top1_frac.max().item()),
        }
    return stats


def run(params):
    data_dir = params['data_path']
    dataset_name = params['finetune_dataset']
    if dataset_name not in DATASET2TASK:
        raise ValueError(f"Unsupported dataset: {dataset_name} (node datasets only).")
    params['task'] = 'node'

    dataset, splits, labels, num_classes, _ = get_finetune_graph(data_dir, dataset_name)
    params["num_classes"] = num_classes

    dataset = pre_node(dataset)
    data = dataset[0]
    labels = labels.long()
    data.y = labels

    print(
        f"dataset {dataset_name}: nodes {data.num_nodes} | edges {data.edge_index.size(1)} | "
        f"classes {num_classes} | feats {data.node_text_feat.shape[1]}"
    )

    activation_str = params["activation"].lower() if isinstance(params["activation"], str) else "relu"
    activation = nn.ReLU if activation_str == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params["activation"] = activation
    print(f"Using device: {device}")

    base_encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
        moe=params.get("moe", False),
        num_experts=params.get("moe_experts", params.get("K", 3)),
        tau=params.get("moe_tau", params.get("tau", 1.0)),
        moe_layers=params.get("moe_layers", "none"),
    )

    base_vq = VectorQuantize(
        dim=params["hidden_dim"],
        codebook_size=params["codebook_size"],
        codebook_dim=params["code_dim"],
        heads=params["codebook_head"],
        separate_codebook_per_head=True,
        decay=params["codebook_decay"],
        commitment_weight=params["commit_weight"],
        use_cosine_sim=True,
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=True,
        ema_update=False,
    )

    pretrain_run_id = get_pretrain_run_id(params)
    pretrain_path = str(params.get("pretrain_path", "") or "").strip()
    if pretrain_path.lower() in {"default", "auto"}:
        pretrain_path = ""
    if pretrain_path and not osp.isabs(pretrain_path):
        pretrain_path = osp.join(params['pt_model_path'], pretrain_path)

    if pretrain_path or params.get("pretrain_dataset", "na") != 'na':
        pretrain_task = params.get('pretrain_task', 'all')
        if pretrain_path:
            path = pretrain_path
        elif pretrain_task == 'all':
            path = osp.join(params['pt_model_path'], pretrain_run_id)
        else:
            raise ValueError("Invalid pretrain task configuration.")

        encoder_path = osp.join(path, f'encoder_{params["pretrain_model_epoch"]}.pt')
        vq_path = osp.join(path, f'vq_{params["pretrain_model_epoch"]}.pt')
        if not osp.exists(encoder_path):
            raise FileNotFoundError("Cannot find encoder checkpoint. Set --pretrain_path to a valid folder.")
        if not osp.exists(vq_path):
            raise FileNotFoundError("Cannot find vector-quantizer checkpoint. Set --pretrain_path to a valid folder.")

        base_encoder = load_params(base_encoder, encoder_path)
        base_vq = load_params(base_vq, vq_path)
        print("Loaded pretrained encoder and VQ.")

    if params.get("freeze_vq", 1):
        freeze_params(base_vq)
        print("Freeze VQ parameters during fine-tuning")

    if params["batch_size"] != 0:
        raise ValueError("This script only supports full-batch training (batch_size must be 0).")

    data = data.to(device)
    labels = labels.to(device)

    if isinstance(splits, dict):
        splits = [splits] * params["repeat"]
    num_runs = min(params.get("repeat", 1), len(splits))

    run_metrics = []
    run_routing_summaries = []

    for run_idx in range(num_runs):
        current_seed = run_idx
        seed_everything(current_seed)

        split = splits[run_idx]
        split = {k: v.to(device) for k, v in split.items()}

        task_model = TaskModel(
            encoder=deepcopy(base_encoder),
            vq=deepcopy(base_vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        print(f"[Run {run_idx + 1:02d}] MoE enabled: {getattr(task_model.encoder, 'moe', False)} | "
              f"MoE layers: {getattr(task_model.encoder, 'moe_layer_flags', [])}")

        optimizer = torch.optim.AdamW(task_model.parameters(), lr=params["finetune_lr"])

        best_val = float("-inf")
        best_state = None
        best_epoch = -1
        patience_counter = 0

        for epoch in range(params["finetune_epochs"]):
            loss_dict = ft_node(
                model=task_model,
                dataset=data,
                loader=None,
                optimizer=optimizer,
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[30] * params["num_layers"],
            )

            result = eval_node(
                model=task_model,
                dataset=data,
                loader=None,
                split=split,
                labels=labels,
                params=params,
            )

            if result["val"] > best_val:
                best_val = result["val"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % params.get("display_step", 10) == 0:
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss_dict.get('loss', float('nan')):.4f}, "
                    f"Train: {result['train']:.2f}%, Valid: {result['val']:.2f}%, Test: {result['test']:.2f}%"
                )

            try:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": loss_dict.get("loss", float("nan")),
                    "train/train_value": result["train"],
                    "train/val_value": result["val"],
                    "train/test_value": result["test"],
                })
            except Exception:
                pass

            if params["early_stop"] > 0 and patience_counter >= params["early_stop"]:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best valid: {best_val:.2f}%).")
                break

        if best_state is None:
            raise RuntimeError("Training finished without recording validation improvements.")

        task_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final_result = eval_node(
            model=task_model,
            dataset=data,
            loader=None,
            split=split,
            labels=labels,
            params=params,
        )
        print(f"Run {run_idx + 1:02d} (seed={current_seed}), best epoch {best_epoch}: "
              f"Train: {final_result['train']:.2f}%, Valid: {final_result['val']:.2f}%, Test: {final_result['test']:.2f}%")

        routing_summary = None
        if getattr(task_model.encoder, "moe", False):
            moe_layer_indices = [i for i, flag in enumerate(task_model.encoder.moe_layer_flags) if flag]
            router_cache = _compute_routing_cache(task_model, data)
            test_mask = split["test"]
            routing_stats = compute_test_routing_stats(router_cache, moe_layer_indices, test_mask)
            routing_summary = routing_stats

            print(f"Run {run_idx + 1:02d} routing stats (clean test set):")
            routing_wandb_payload = {}
            for layer_idx in moe_layer_indices:
                s = routing_stats[layer_idx]
                avg_str = ", ".join(f"{v:.3f}" for v in s["avg_prob"])
                top1_str = ", ".join(f"{v:.3f}" for v in s["top1_frac"])
                print(
                    f"  Layer {layer_idx}: avg_prob=[{avg_str}] top1_frac=[{top1_str}] "
                    f"entropy_of_avg={s['entropy_of_avg']:.3f} mean_node_entropy={s['mean_node_entropy']:.3f} "
                    f"max_usage={s['max_usage']:.3f}"
                )
                routing_wandb_payload[f"moe/layer{layer_idx}/test/entropy_of_avg"] = s["entropy_of_avg"]
                routing_wandb_payload[f"moe/layer{layer_idx}/test/mean_node_entropy"] = s["mean_node_entropy"]
                routing_wandb_payload[f"moe/layer{layer_idx}/test/max_usage"] = s["max_usage"]
            try:
                wandb.log(routing_wandb_payload)
            except Exception:
                pass

        run_metrics.append([final_result["train"], final_result["val"], final_result["test"]])
        if routing_summary is not None:
            run_routing_summaries.append(routing_summary)

    metrics_tensor = torch.tensor(run_metrics, dtype=torch.float32)
    mean_scores = metrics_tensor.mean(dim=0)
    std_scores = metrics_tensor.std(dim=0, unbiased=False)

    print(f"\nSummary over {num_runs} run(s) on {dataset_name} (clean setting):")
    for name, mean, std in zip(["train", "val", "test"], mean_scores, std_scores):
        print(f"  {name}: {mean:.2f}% ± {std:.2f}%")

    if run_routing_summaries:
        moe_layer_indices = list(run_routing_summaries[0].keys())
        print(f"\nMoE Routing Summary over {len(run_routing_summaries)} run(s) on {dataset_name} (clean test set):")
        agg_payload = {}
        for layer_idx in moe_layer_indices:
            entropies = torch.tensor([s[layer_idx]["entropy_of_avg"] for s in run_routing_summaries])
            node_entropies = torch.tensor([s[layer_idx]["mean_node_entropy"] for s in run_routing_summaries])
            max_usages = torch.tensor([s[layer_idx]["max_usage"] for s in run_routing_summaries])
            print(
                f"  Layer {layer_idx}: entropy_of_avg={entropies.mean():.3f}±{entropies.std(unbiased=False):.3f} "
                f"mean_node_entropy={node_entropies.mean():.3f}±{node_entropies.std(unbiased=False):.3f} "
                f"max_usage={max_usages.mean():.3f}±{max_usages.std(unbiased=False):.3f}"
            )
            agg_payload[f"moe_summary/layer{layer_idx}/test/entropy_of_avg_mean"] = float(entropies.mean())
            agg_payload[f"moe_summary/layer{layer_idx}/test/mean_node_entropy_mean"] = float(node_entropies.mean())
            agg_payload[f"moe_summary/layer{layer_idx}/test/max_usage_mean"] = float(max_usages.mean())
        try:
            wandb.log(agg_payload)
        except Exception:
            pass

    try:
        wandb.log({
            "final/test_mean": mean_scores[2].item(),
            "final/test_std": std_scores[2].item(),
        })
    except Exception:
        pass


def main():
    params = get_args_finetune()
    base_dir = osp.dirname(__file__)

    if params["use_params"]:
        config_path = osp.join(base_dir, '..', '..', 'config', 'finetune.yaml')
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        import yaml
        with open(config_path, 'r') as f:
            default_params = yaml.safe_load(f)
        dataset = params["finetune_dataset"]
        if dataset not in DATASET2TASK:
            raise ValueError(f"Unsupported dataset: {dataset} (node datasets only).")
        params = get_args_finetune(default_params=default_params["node"][dataset])

    ensure_finetune_lr(params)
    params['data_path'] = osp.join(base_dir, '..', '..', 'data')
    params['pt_model_path'] = osp.join(base_dir, '..', '..', 'ckpts', 'pretrain_model')
    params.setdefault("display_step", 10)

    explicit_pretrain_path = str(params.get("pretrain_path", "") or "").strip()
    if explicit_pretrain_path and explicit_pretrain_path.lower() not in {"default", "auto"}:
        params["pretrain_path"] = explicit_pretrain_path
    elif params.get("pretrain_dataset", "na") == "na":
        params["pretrain_path"] = ""

    warnings.filterwarnings("ignore")
    run_name = f"{str.upper(params['finetune_dataset'])} - Clean Setting MoE"
    wandb.init(
        project="STEM-GNN-Finetune",
        name=run_name,
        config=params,
        mode="disabled" if params.get("debug", False) else "online",
        tags=[params.get('setting', 'standard'), 'clean-setting'],
    )
    params = dict(wandb.config)
    ensure_finetune_lr(params)
    print("Params loaded.")

    try:
        run(params)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
