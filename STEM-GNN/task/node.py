import torch

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model
from utils.calibration import apply_temperature, expected_calibration_error


def _run_full_batch(model, dataset, labels, split, params, aux_router_feat=None, bn_adapt=False):
    device = get_device_from_model(model)
    x = dataset.node_text_feat.to(device)
    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_text_feat[dataset.xe].to(device)
    y = labels.to(device)

    if bn_adapt:
        z = model.encode_bn_adapted(x, edge_index, edge_attr, aux_router_feat=aux_router_feat)
    else:
        z = model.encode(x, edge_index, edge_attr, aux_router_feat=aux_router_feat)
    return z, y


def _accumulate_minibatch_logits(model, loader, device, bn_adapt=False):
    logits_list, gts = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat[batch.xe]
        y = batch.y[:bs]

        if bn_adapt:
            z = model.encode_bn_adapted(x, edge_index, edge_attr)[:bs]
        else:
            z = model.encode(x, edge_index, edge_attr)[:bs]
        logits = model.get_lin_logits(z).mean(1)

        logits_list.append(logits.detach())
        gts.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(gts, dim=0)


def ft_node(model, dataset, loader, optimizer, split, labels, params, scheduler=None, aux_router_feat=None, class_sim=None, **kwargs):
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.train()

    device = get_device_from_model(model)
    lamda_env = params.get("lamda_env", 0.0)
    lambda_act = 1.0

    if loader is None:
        z, y = _run_full_batch(model, dataset, labels, split, params, aux_router_feat=aux_router_feat)
        train_mask = split["train"].to(z.device)
        z_train, y_train = z[train_mask], y[train_mask]

        act_loss = model.compute_activation_loss(z_train, y_train) * lambda_act
        jac_loss = model.decoder_jacobian_penalty()
        lip_loss = model.encoder_lipschitz_penalty()
        env_loss = lamda_env * model.get_env_reg()
        llm_routing_reg = params.get("llm_routing_reg", 0.0)
        if llm_routing_reg > 0.0 and class_sim is not None:
            llm_loss = llm_routing_reg * model.get_llm_routing_consistency_loss(class_sim)
        else:
            llm_loss = torch.zeros(1, device=z.device)
        loss = act_loss + jac_loss + lip_loss + env_loss + llm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        return {
            "act_loss": act_loss.item(),
            "jac_loss": jac_loss.item(),
            "lip_loss": lip_loss.item(),
            "env_loss": env_loss.item(),
            "llm_loss": llm_loss.item(),
            "loss": loss.item(),
        }

    total_act_loss = 0.0
    total_jac_loss = 0.0
    total_lip_loss = 0.0
    total_env_loss = 0.0
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat[batch.xe]
        y = batch.y[:bs]

        z = model.encode(x, edge_index, edge_attr)[:bs]
        env_reg = model.get_env_reg()

        act_loss = model.compute_activation_loss(z, y) * lambda_act
        jac_loss = model.decoder_jacobian_penalty()
        lip_loss = model.encoder_lipschitz_penalty()
        env_loss = lamda_env * env_reg
        loss = act_loss + jac_loss + lip_loss + env_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_act_loss += act_loss.item()
        total_jac_loss += jac_loss.item()
        total_lip_loss += lip_loss.item()
        total_env_loss += env_loss.item()
        total_loss += loss.item()

    num_batches = len(loader)
    return {
        "act_loss": total_act_loss / num_batches,
        "jac_loss": total_jac_loss / num_batches,
        "lip_loss": total_lip_loss / num_batches,
        "env_loss": total_env_loss / num_batches,
        "loss": total_loss / num_batches,
    }


def get_logits_labels(model, dataset, loader, split, labels, params, aux_router_feat=None, bn_adapt=False):
    """Raw (pre-softmax) logits and labels for the whole graph/loader, for use
    by calibration.fit_temperature or other post-hoc analysis outside the
    standard train/val/test accuracy report below."""
    model.eval()
    device = get_device_from_model(model)
    with torch.no_grad():
        if loader is None:
            z, y = _run_full_batch(model, dataset, labels, split, params,
                                    aux_router_feat=aux_router_feat, bn_adapt=bn_adapt)
            logits = model.get_lin_logits(z).mean(1)
        else:
            logits, y = _accumulate_minibatch_logits(model, loader, device, bn_adapt=bn_adapt)
    return logits, y


def eval_node(model, dataset, loader, split, labels, params, aux_router_feat=None,
              bn_adapt=False, temperature=None, return_ece=False, **kwargs):
    """
    bn_adapt: use the current (possibly shifted) batch's own BatchNorm
        statistics instead of the stored running statistics (test-time
        adaptation for covariate/feature shift; see
        Encoder.encode_with_bn_adaptation).
    temperature: if given, divide logits by this scalar before softmax
        (post-hoc calibration; fit on clean validation logits via
        utils.calibration.fit_temperature, then reused unchanged here).
    return_ece: if True, also report expected calibration error on val/test.
    """
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.eval()
    device = get_device_from_model(model)

    with torch.no_grad():
        if loader is None:
            z, y = _run_full_batch(model, dataset, labels, split, params,
                                    aux_router_feat=aux_router_feat, bn_adapt=bn_adapt)
            logits = model.get_lin_logits(z).mean(1)
        else:
            logits, y = _accumulate_minibatch_logits(model, loader, device, bn_adapt=bn_adapt)

        if temperature is not None:
            logits = apply_temperature(logits, temperature)
        pred = logits.softmax(dim=-1)

        train_mask = split["train"].to(pred.device)
        val_mask = split["valid"].to(pred.device)
        test_mask = split["test"].to(pred.device)

        train_value = evaluate(pred, y, train_mask, params)
        val_value = evaluate(pred, y, val_mask, params)
        test_value = evaluate(pred, y, test_mask, params)

        result = {
            "train": train_value,
            "val": val_value,
            "test": test_value,
            "metric": task2metric[params["task"]],
        }
        if return_ece:
            result["val_ece"] = expected_calibration_error(pred[val_mask], y[val_mask])
            result["test_ece"] = expected_calibration_error(pred[test_mask], y[test_mask])

    return result
