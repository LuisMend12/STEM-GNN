import math
import os
import os.path as osp
import re
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.utils import degree, to_undirected

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.ft_model import TaskModel
from model.vq import VectorQuantize
from task.node import ft_node
from utils.args import get_args_finetune
from utils.eval import evaluate
from utils.others import (
    freeze_params, load_params, seed_everything, ensure_finetune_lr, get_pretrain_run_id,
    compute_class_similarity_profile,
)
from utils.preprocess import pre_node

import warnings
import wandb


BUCKET_NAMES: List[str] = ["ID", "OOD-low", "OOD-high"]
PRIMARY_RATIOS: Tuple[float, float, float] = (0.5, 0.25, 0.25)
SECONDARY_RATIOS: Tuple[float, float, float] = (0.6, 0.2, 0.2)
DATASET2TASK = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
}


def compute_degree_buckets(data) -> Tuple[torch.Tensor, Tuple[float, float], Dict[str, torch.Tensor]]:
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    deg = degree(edge_index[0], num_nodes=data.num_nodes).to(torch.float32)
    n = data.num_nodes
    sorted_idx = torch.argsort(deg)

    low_count = max(1, int(math.floor(n * 0.15)))
    high_count = max(1, int(math.floor(n * 0.15)))
    if low_count + high_count >= n:
        overflow = low_count + high_count - (n - 1)
        if overflow > 0:
            reducible_high = max(0, high_count - 1)
            reduction = min(overflow, reducible_high)
            high_count -= reduction
            overflow -= reduction
        if overflow > 0:
            reducible_low = max(0, low_count - 1)
            reduction = min(overflow, reducible_low)
            low_count -= reduction
            overflow -= reduction

    id_count = n - low_count - high_count
    ood_low_idx = sorted_idx[:low_count]
    id_idx = sorted_idx[low_count:low_count + id_count]
    ood_high_idx = sorted_idx[low_count + id_count:]

    low_boundary = deg[ood_low_idx[-1]].item()
    high_boundary = deg[ood_high_idx[0]].item()

    bucket_indices = {
        "ID": id_idx,
        "OOD-low": ood_low_idx,
        "OOD-high": ood_high_idx,
    }

    return deg, (float(low_boundary), float(high_boundary)), bucket_indices


def determine_split_counts(class_size: int) -> Tuple[int, int]:
    if class_size < 3:
        raise RuntimeError(f"class size {class_size} too small for 3-way split")

    for ratios in (PRIMARY_RATIOS, SECONDARY_RATIOS):
        train = max(1, math.floor(class_size * ratios[0]))
        val = max(1, math.floor(class_size * ratios[1]))
        if train + val >= class_size:
            overflow = train + val - (class_size - 1)
            if overflow > 0:
                reducible_val = max(0, val - 1)
                reduction = min(overflow, reducible_val)
                val -= reduction
                overflow -= reduction
            if overflow > 0:
                reducible_train = max(0, train - 1)
                reduction = min(overflow, reducible_train)
                train -= reduction
                overflow -= reduction
        test = class_size - train - val
        if train >= 1 and val >= 1 and test >= 1:
            return train, val

    train = max(1, class_size - 2)
    val = 1
    return train, val


def stratified_split(id_indices: torch.Tensor, labels: torch.Tensor, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_parts, val_parts, test_parts = [], [], []
    id_labels = labels[id_indices]
    classes = torch.unique(id_labels)

    for cls in classes.tolist():
        cls_idx = id_indices[id_labels == cls]
        cls_size = cls_idx.numel()
        train_count, val_count = determine_split_counts(cls_size)

        perm = torch.randperm(cls_size, generator=generator)
        cls_idx = cls_idx[perm]

        train_parts.append(cls_idx[:train_count])
        val_parts.append(cls_idx[train_count:train_count + val_count])
        test_parts.append(cls_idx[train_count + val_count:])

    train_idx = torch.sort(torch.cat(train_parts))[0]
    val_idx = torch.sort(torch.cat(val_parts))[0]
    test_idx = torch.sort(torch.cat(test_parts))[0]
    return train_idx, val_idx, test_idx


def indices_to_mask(indices: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[indices.to(device=device)] = True
    return mask


def compute_predictions(model, data, labels, params, train_mask, aux_router_feat=None):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        x = data.node_text_feat
        edge_index = data.edge_index
        edge_attr = data.edge_text_feat[data.xe]
        z = model.encode(x, edge_index, edge_attr, aux_router_feat=aux_router_feat)
        pred = model.get_lin_logits(z).mean(1).softmax(dim=-1)

    if was_training:
        model.train()
    return pred


def compute_accuracy(pred, labels, mask, params):
    if mask.sum().item() == 0:
        return float("nan")
    value = evaluate(pred, labels, mask, params)
    return value / 100.0


def compute_routing_cache(model, data, aux_router_feat=None) -> List[torch.Tensor]:
    """Runs one eval-mode forward pass and returns the per-MoE-layer router
    weights [N, num_experts] for every node (softmax, no Gumbel noise)."""
    was_training = model.training
    model.eval()
    encoder = model.encoder
    encoder.enable_router_cache(True)
    with torch.no_grad():
        x = data.node_text_feat
        edge_index = data.edge_index
        edge_attr = data.edge_text_feat[data.xe]
        encoder(x, edge_index, edge_attr, aux_router_feat=aux_router_feat)
    cache = encoder.get_router_cache(reset=True)
    encoder.enable_router_cache(False)
    if was_training:
        model.train()
    return cache


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum().item())


def compute_bucket_routing_stats(
    router_cache: List[torch.Tensor],
    moe_layer_indices: List[int],
    bucket_masks: Dict[str, torch.Tensor],
) -> Dict[int, Dict[str, Dict[str, object]]]:
    """For each MoE layer and each node bucket, summarizes the router's expert
    distribution: avg_prob (mean softmax over the bucket), entropy_of_avg and
    max_usage (imbalance of that average), and mean_node_entropy (how decisive
    individual nodes' routing is)."""
    stats: Dict[int, Dict[str, Dict[str, object]]] = {}
    for cache_idx, layer_idx in enumerate(moe_layer_indices):
        weights = router_cache[cache_idx]
        num_experts = weights.size(1)
        layer_stats: Dict[str, Dict[str, object]] = {}
        for name, mask in bucket_masks.items():
            mask = mask.to(weights.device)
            if mask.sum().item() == 0:
                layer_stats[name] = {
                    "avg_prob": [float("nan")] * num_experts,
                    "entropy_of_avg": float("nan"),
                    "mean_node_entropy": float("nan"),
                    "top1_frac": [float("nan")] * num_experts,
                    "max_usage": float("nan"),
                }
                continue
            bucket_weights = weights[mask]
            avg_prob = bucket_weights.mean(dim=0)
            entropy_of_avg = _entropy(avg_prob)
            clamped = bucket_weights.clamp_min(1e-12)
            node_entropies = -(clamped * clamped.log()).sum(dim=1)
            mean_node_entropy = float(node_entropies.mean().item())
            top1 = bucket_weights.argmax(dim=1)
            top1_frac = torch.stack(
                [(top1 == e).float().mean() for e in range(num_experts)]
            )
            layer_stats[name] = {
                "avg_prob": avg_prob.cpu().tolist(),
                "entropy_of_avg": entropy_of_avg,
                "mean_node_entropy": mean_node_entropy,
                "top1_frac": top1_frac.cpu().tolist(),
                "max_usage": float(top1_frac.max().item()),
            }
        stats[layer_idx] = layer_stats
    return stats


def _total_variation(p: List[float], q: List[float]) -> float:
    if any(math.isnan(v) for v in p) or any(math.isnan(v) for v in q):
        return float("nan")
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _kl_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    if any(math.isnan(v) for v in p) or any(math.isnan(v) for v in q):
        return float("nan")
    return sum(a * math.log((a + eps) / (b + eps)) for a, b in zip(p, q) if a > 0)


def compute_routing_shift(
    stats: Dict[int, Dict[str, Dict[str, object]]],
    id_name: str,
    ood_names: List[str],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Distributional distance between each OOD bucket's average router
    distribution and the ID bucket's — the core 'routing shift' measurement."""
    shift: Dict[int, Dict[str, Dict[str, float]]] = {}
    for layer_idx, bucket_stats in stats.items():
        id_avg = bucket_stats[id_name]["avg_prob"]
        layer_shift: Dict[str, Dict[str, float]] = {}
        for name in ood_names:
            ood_avg = bucket_stats[name]["avg_prob"]
            layer_shift[name] = {
                "tv_distance": _total_variation(id_avg, ood_avg),
                "kl_id_to_ood": _kl_divergence(id_avg, ood_avg),
            }
        shift[layer_idx] = layer_shift
    return shift


def run(params):
    data_dir = params['data_path']
    dataset_name = params['finetune_dataset']

    if dataset_name not in DATASET2TASK:
        raise ValueError(f"Unsupported dataset: {dataset_name} (node datasets only).")

    params['task'] = 'node'

    dataset, _, labels, num_classes, _ = get_finetune_graph(data_dir, dataset_name)
    params["num_classes"] = num_classes

    dataset = pre_node(dataset)
    data = dataset[0]
    labels = labels.long()
    data.y = labels

    deg, boundaries, bucket_indices = compute_degree_buckets(data)
    print(
        f"dataset {dataset_name}: nodes {data.num_nodes} | edges {data.edge_index.size(1)} | "
        f"classes {num_classes} | feats {data.node_text_feat.shape[1]}"
    )
    print(f"Degree boundaries (15/85): {boundaries[0]:.4f}, {boundaries[1]:.4f}")
    for name in BUCKET_NAMES:
        print(f"Bucket {name}: {bucket_indices[name].numel()} nodes")

    activation_str = params["activation"].lower() if isinstance(params["activation"], str) else "relu"
    activation = nn.ReLU if activation_str == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params["activation"] = activation
    print(f"Using device: {device}")

    use_aux_router = bool(params.get("use_aux_router", False))
    if use_aux_router and not (hasattr(data, "class_node_text_feat") and data.class_node_text_feat is not None):
        raise AttributeError(
            "--use_aux_router requires data.class_node_text_feat, but it is missing for this dataset."
        )
    aux_router_dim = num_classes if use_aux_router else 0
    if use_aux_router:
        print(f"Aux router signal enabled: layer-0 router gets a {num_classes}-dim class-similarity profile.")

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
        aux_router_dim=aux_router_dim,
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

    aux_router_feat = None
    if use_aux_router:
        aux_router_feat = compute_class_similarity_profile(data.node_text_feat, data.class_node_text_feat)

    router_init = params.get("router_init", "random")
    if router_init == "kmeans" and base_encoder.moe and base_encoder.moe_layer_flags[0]:
        cluster_feat = data.node_text_feat
        if aux_router_feat is not None:
            cluster_feat = torch.cat([cluster_feat, aux_router_feat], dim=-1)
        base_encoder.init_router_kmeans(layer_idx=0, features=cluster_feat, seed=0)
        print("Layer-0 router initialized via k-means clustering on LLM text features (instead of random init).")

    num_runs = params.get("repeat", 1)
    display_step = params.get("display_step", 10)
    base_seed = 0

    run_metrics = []
    run_routing_summaries = []

    for run in range(num_runs):
        current_seed = base_seed + run
        seed_everything(current_seed)

        task_model = TaskModel(
            encoder=deepcopy(base_encoder),
            vq=deepcopy(base_vq),
            num_classes=num_classes,
            params=params,
        ).to(device)
        print(f"[Run {run + 1:02d}] MoE enabled: {getattr(task_model.encoder, 'moe', False)} | "
              f"MoE layers: {getattr(task_model.encoder, 'moe_layer_flags', [])}")

        optimizer = torch.optim.AdamW(task_model.parameters(), lr=params["finetune_lr"])

        id_indices = bucket_indices["ID"]
        try:
            train_idx, val_idx, id_test_idx = stratified_split(id_indices, labels.cpu(), current_seed)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to stratify ID bucket for dataset {dataset_name}. "
                f"Consider adjusting bucket ratios or ensuring each class has at least 3 nodes within the ID bucket."
            ) from exc

        train_mask = indices_to_mask(train_idx, data.num_nodes, device)
        val_mask = indices_to_mask(val_idx, data.num_nodes, device)
        id_test_mask = indices_to_mask(id_test_idx, data.num_nodes, device)

        ood_masks = []
        for name in BUCKET_NAMES[1:]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            mask[bucket_indices[name].to(device=device)] = True
            ood_masks.append(mask)

        print(
            f"[Run {run + 1:02d}] ID splits -> Train: {train_idx.numel()}, "
            f"Val: {val_idx.numel()}, ID-Test: {id_test_idx.numel()}"
        )
        for name, mask in zip(BUCKET_NAMES[1:], ood_masks):
            count = mask.sum().item()
            print(f"          Bucket {name}: {count} nodes")
            if count == 0:
                print(f"          [Warning] Bucket {name} is empty for this dataset/run.")

        split = {"train": train_mask, "valid": val_mask, "test": id_test_mask}

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
                aux_router_feat=aux_router_feat,
            )

            pred = compute_predictions(task_model, data, labels, params, train_mask, aux_router_feat=aux_router_feat)
            train_acc = compute_accuracy(pred, labels, train_mask, params)
            val_acc = compute_accuracy(pred, labels, val_mask, params)
            id_test_acc = compute_accuracy(pred, labels, id_test_mask, params)
            ood_accs = [compute_accuracy(pred, labels, mask, params) for mask in ood_masks]

            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % display_step == 0:
                bucket_msg = ", ".join(
                    f"{name}: {100.0 * score:.2f}%"
                    for name, score in zip(BUCKET_NAMES, [id_test_acc] + ood_accs)
                )
                total_loss = loss_dict.get('loss', float('nan'))
                print(
                    f"Epoch: {epoch:03d}, Loss: {total_loss:.4f}, "
                    f"Train(ID): {100.0 * train_acc:.2f}%, "
                    f"Valid(ID): {100.0 * val_acc:.2f}%, "
                    f"{bucket_msg}"
                )

            # wandb logging each epoch
            try:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/lin_loss": loss_dict.get("act_loss", float("nan")),
                        "train/jac_loss": loss_dict.get("jac_loss", float("nan")),
                        "train/env_loss": loss_dict.get("env_loss", float("nan")),
                        "train/loss": loss_dict.get("loss", float("nan")),
                        "train/train_value": 100.0 * train_acc,
                        "train/val_value": 100.0 * val_acc,
                        "train/test_value": 100.0 * id_test_acc,
                        "ood/OOD-low": 100.0 * (ood_accs[0] if len(ood_accs) > 0 else float("nan")),
                        "ood/OOD-high": 100.0 * (ood_accs[1] if len(ood_accs) > 1 else float("nan")),
                    }
                )
            except Exception:
                pass

            if params["early_stop"] > 0 and patience_counter >= params["early_stop"]:
                print(
                    f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, "
                    f"best valid(ID): {100.0 * best_val:.2f}%)."
                )
                break

        if best_state is None:
            raise RuntimeError("Training finished without recording validation improvements.")

        task_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        pred = compute_predictions(task_model, data, labels, params, train_mask, aux_router_feat=aux_router_feat)
        id_test_acc = compute_accuracy(pred, labels, id_test_mask, params)
        ood_accs = [compute_accuracy(pred, labels, mask, params) for mask in ood_masks]

        print(f"Run {run + 1:02d} (seed={current_seed}), best epoch {best_epoch}:")
        for name, score in zip(BUCKET_NAMES, [id_test_acc] + ood_accs):
            print(f"  {name}: {100.0 * score:.2f}%")

        routing_summary = None
        if getattr(task_model.encoder, "moe", False):
            moe_layer_indices = [i for i, flag in enumerate(task_model.encoder.moe_layer_flags) if flag]
            router_cache = compute_routing_cache(task_model, data, aux_router_feat=aux_router_feat)
            bucket_masks_for_routing = {"ID": id_test_mask}
            bucket_masks_for_routing.update(zip(BUCKET_NAMES[1:], ood_masks))
            routing_stats = compute_bucket_routing_stats(router_cache, moe_layer_indices, bucket_masks_for_routing)
            routing_shift = compute_routing_shift(routing_stats, "ID", BUCKET_NAMES[1:])
            routing_summary = {"stats": routing_stats, "shift": routing_shift}

            print(f"Run {run + 1:02d} routing stats:")
            routing_wandb_payload = {}
            for layer_idx in moe_layer_indices:
                for name in BUCKET_NAMES:
                    s = routing_stats[layer_idx][name]
                    avg_str = ", ".join(f"{v:.3f}" for v in s["avg_prob"])
                    top1_str = ", ".join(f"{v:.3f}" for v in s["top1_frac"])
                    print(
                        f"  Layer {layer_idx} [{name}]: avg_prob=[{avg_str}] top1_frac=[{top1_str}] "
                        f"entropy_of_avg={s['entropy_of_avg']:.3f} mean_node_entropy={s['mean_node_entropy']:.3f} "
                        f"max_usage={s['max_usage']:.3f}"
                    )
                    routing_wandb_payload[f"moe/layer{layer_idx}/{name}/entropy_of_avg"] = s["entropy_of_avg"]
                    routing_wandb_payload[f"moe/layer{layer_idx}/{name}/mean_node_entropy"] = s["mean_node_entropy"]
                    routing_wandb_payload[f"moe/layer{layer_idx}/{name}/max_usage"] = s["max_usage"]
                for name in BUCKET_NAMES[1:]:
                    sh = routing_shift[layer_idx][name]
                    print(
                        f"  Layer {layer_idx} routing shift ID->{name}: "
                        f"TV={sh['tv_distance']:.4f} KL={sh['kl_id_to_ood']:.4f}"
                    )
                    routing_wandb_payload[f"moe/layer{layer_idx}/shift_{name}/tv_distance"] = sh["tv_distance"]
                    routing_wandb_payload[f"moe/layer{layer_idx}/shift_{name}/kl_id_to_ood"] = sh["kl_id_to_ood"]
            try:
                wandb.log(routing_wandb_payload)
            except Exception:
                pass

        # log best-of-run to wandb
        try:
            wandb.log(
                {
                    "best/epoch": best_epoch,
                    "best/train": 100.0 * compute_accuracy(pred, labels, train_mask, params),
                    "best/val": 100.0 * best_val,
                    "best/test": 100.0 * id_test_acc,
                    "best/OOD-low": 100.0 * (ood_accs[0] if len(ood_accs) > 0 else float("nan")),
                    "best/OOD-high": 100.0 * (ood_accs[1] if len(ood_accs) > 1 else float("nan")),
                }
            )
        except Exception:
            pass

        run_metrics.append([id_test_acc] + ood_accs)
        if routing_summary is not None:
            run_routing_summaries.append(routing_summary)

    metrics_tensor = torch.tensor(run_metrics, dtype=torch.float32)
    mean_scores = metrics_tensor.mean(dim=0) * 100.0
    std_scores = metrics_tensor.std(dim=0, unbiased=False) * 100.0

    print(f"\nSummary over {num_runs} run(s) on {dataset_name}:")
    for name, mean, std in zip(BUCKET_NAMES, mean_scores, std_scores):
        print(f"  {name}: {mean:.2f}% ± {std:.2f}%")

    if run_routing_summaries:
        moe_layer_indices = list(run_routing_summaries[0]["stats"].keys())
        print(f"\nMoE Routing Summary over {len(run_routing_summaries)} run(s) on {dataset_name}:")
        agg_payload = {}
        for layer_idx in moe_layer_indices:
            for name in BUCKET_NAMES:
                entropies = torch.tensor([s["stats"][layer_idx][name]["entropy_of_avg"] for s in run_routing_summaries])
                node_entropies = torch.tensor([s["stats"][layer_idx][name]["mean_node_entropy"] for s in run_routing_summaries])
                max_usages = torch.tensor([s["stats"][layer_idx][name]["max_usage"] for s in run_routing_summaries])
                print(
                    f"  Layer {layer_idx} [{name}]: entropy_of_avg={entropies.mean():.3f}±{entropies.std(unbiased=False):.3f} "
                    f"mean_node_entropy={node_entropies.mean():.3f}±{node_entropies.std(unbiased=False):.3f} "
                    f"max_usage={max_usages.mean():.3f}±{max_usages.std(unbiased=False):.3f}"
                )
                agg_payload[f"moe_summary/layer{layer_idx}/{name}/entropy_of_avg_mean"] = float(entropies.mean())
                agg_payload[f"moe_summary/layer{layer_idx}/{name}/mean_node_entropy_mean"] = float(node_entropies.mean())
                agg_payload[f"moe_summary/layer{layer_idx}/{name}/max_usage_mean"] = float(max_usages.mean())
            for name in BUCKET_NAMES[1:]:
                tvs = torch.tensor([s["shift"][layer_idx][name]["tv_distance"] for s in run_routing_summaries])
                kls = torch.tensor([s["shift"][layer_idx][name]["kl_id_to_ood"] for s in run_routing_summaries])
                print(
                    f"  Layer {layer_idx} routing shift ID->{name}: "
                    f"TV={tvs.mean():.4f}±{tvs.std(unbiased=False):.4f} KL={kls.mean():.4f}±{kls.std(unbiased=False):.4f}"
                )
                agg_payload[f"moe_summary/layer{layer_idx}/shift_{name}/tv_distance_mean"] = float(tvs.mean())
                agg_payload[f"moe_summary/layer{layer_idx}/shift_{name}/kl_id_to_ood_mean"] = float(kls.mean())
        try:
            wandb.log(agg_payload)
        except Exception:
            pass

    # final summary to wandb (means + stds)
    try:
        final_payload = {
            "final/ID": f"{mean_scores[0].item():.2f} ± {std_scores[0].item():.2f}",
            "final/OOD-low": f"{mean_scores[1].item():.2f} ± {std_scores[1].item():.2f}",
            "final/OOD-high": f"{mean_scores[2].item():.2f} ± {std_scores[2].item():.2f}",
            "final/ID_mean": mean_scores[0].item(),
            "final/OOD-low_mean": mean_scores[1].item(),
            "final/OOD-high_mean": mean_scores[2].item(),
            "final/ID_std": std_scores[0].item(),
            "final/OOD-low_std": std_scores[1].item(),
            "final/OOD-high_std": std_scores[2].item(),
        }
        wandb.log(final_payload)
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

    pretrain_run_id = get_pretrain_run_id(params)
    default_pretrain_path = osp.join(base_dir, "..", "..", "ckpts", "pretrain_model", pretrain_run_id)
    DEFAULT_PRETRAIN_SEED = 42
    DEFAULT_PRETRAIN_EPOCH = 25

    explicit_pretrain_path = str(params.get("pretrain_path", "") or "").strip()
    if explicit_pretrain_path and explicit_pretrain_path.lower() not in {"default", "auto"}:
        params["pretrain_path"] = explicit_pretrain_path
    elif params.get("pretrain_dataset", "na") == "na":
        params["pretrain_path"] = ""
    else:
        params["pretrain_path"] = default_pretrain_path

    if "pretrain_model_epoch" not in params or params["pretrain_model_epoch"] in (None, ""):
        params["pretrain_model_epoch"] = DEFAULT_PRETRAIN_EPOCH
    if "pretrain_seed" not in params or params["pretrain_seed"] in (None, ""):
        params["pretrain_seed"] = DEFAULT_PRETRAIN_SEED

    def _maybe_infer_moe_settings(path: str, target: Dict):
        if not path:
            return
        name = osp.basename(path.rstrip("/"))
        moe_flag = re.search(r"moe_(\d+)", name)
        if moe_flag:
            target["moe"] = moe_flag.group(1) != "0"
        if target.get("moe", False):
            layers_match = re.search(r"layers_([A-Za-z]+)", name)
            if layers_match:
                target["moe_layers"] = layers_match.group(1).lower()
            experts_match = re.search(r"_K_(\d+)", name)
            if experts_match:
                target["moe_experts"] = int(experts_match.group(1))
            tau_match = re.search(r"_tau_([0-9.]+)", name)
            if tau_match:
                tau_val = float(tau_match.group(1))
                target["moe_tau"] = tau_val
                target["tau"] = tau_val
            lam_match = re.search(r"_lam_([0-9.]+)", name)
            if lam_match:
                target["lamda_env"] = float(lam_match.group(1))

    _maybe_infer_moe_settings(params.get("pretrain_path", ""), params)

    # wandb init (mimic finetune.py)
    warnings.filterwarnings("ignore")
    run_name = f"{str.upper(params['finetune_dataset'])} - Degree OOD"
    wandb.init(
        project="STEM-GNN-Finetune",
        name=run_name,
        config=params,
        mode="disabled" if params.get("debug", False) else "online",
        tags=[params.get('setting', 'standard'), 'degree-ood'],
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
