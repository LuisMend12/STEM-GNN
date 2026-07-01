import torch

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model


def _run_full_batch(model, dataset, labels, split, params, aux_router_feat=None):
    device = get_device_from_model(model)
    x = dataset.node_text_feat.to(device)
    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_text_feat[dataset.xe].to(device)
    y = labels.to(device)

    z = model.encode(x, edge_index, edge_attr, aux_router_feat=aux_router_feat)
    return z, y


def _accumulate_minibatch_predictions(model, loader, device):
    preds, gts = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat[batch.xe]
        y = batch.y[:bs]

        z = model.encode(x, edge_index, edge_attr)[:bs]
        pred = model.get_lin_logits(z).mean(1).softmax(dim=-1)

        preds.append(pred.detach())
        gts.append(y)
    return torch.cat(preds, dim=0), torch.cat(gts, dim=0)


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
        env_loss = lamda_env * model.get_env_reg()
        llm_routing_reg = params.get("llm_routing_reg", 0.0)
        if llm_routing_reg > 0.0 and class_sim is not None:
            llm_loss = llm_routing_reg * model.get_llm_routing_consistency_loss(class_sim)
        else:
            llm_loss = torch.zeros(1, device=z.device)
        loss = act_loss + jac_loss + env_loss + llm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        return {
            "act_loss": act_loss.item(),
            "jac_loss": jac_loss.item(),
            "env_loss": env_loss.item(),
            "llm_loss": llm_loss.item(),
            "loss": loss.item(),
        }

    total_act_loss = 0.0
    total_jac_loss = 0.0
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
        env_loss = lamda_env * env_reg
        loss = act_loss + jac_loss + env_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_act_loss += act_loss.item()
        total_jac_loss += jac_loss.item()
        total_env_loss += env_loss.item()
        total_loss += loss.item()

    num_batches = len(loader)
    return {
        "act_loss": total_act_loss / num_batches,
        "jac_loss": total_jac_loss / num_batches,
        "env_loss": total_env_loss / num_batches,
        "loss": total_loss / num_batches,
    }


def eval_node(model, dataset, loader, split, labels, params, aux_router_feat=None, **kwargs):
    assert params["setting"] == "standard", "Only standard setting is supported"
    model.eval()
    device = get_device_from_model(model)

    with torch.no_grad():
        if loader is None:
            z, y = _run_full_batch(model, dataset, labels, split, params, aux_router_feat=aux_router_feat)
            pred = model.get_lin_logits(z).mean(1).softmax(dim=-1)
        else:
            pred, y = _accumulate_minibatch_predictions(model, loader, device)

        train_mask = split["train"].to(pred.device)
        val_mask = split["valid"].to(pred.device)
        test_mask = split["test"].to(pred.device)

        train_value = evaluate(pred, y, train_mask, params)
        val_value = evaluate(pred, y, val_mask, params)
        test_value = evaluate(pred, y, test_mask, params)

    return {
        "train": train_value,
        "val": val_value,
        "test": test_value,
        "metric": task2metric[params["task"]],
    }
