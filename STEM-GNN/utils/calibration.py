import torch


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Standard binned ECE: mean absolute gap between confidence and accuracy,
    weighted by how much probability mass falls in each confidence bin
    (Guo et al. 2017, "On Calibration of Modern Neural Networks")."""
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            ece = ece + (accuracies[in_bin].mean() - confidences[in_bin].mean()).abs() * prop_in_bin
    return ece.item()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """Post-hoc temperature scaling (Guo et al. 2017): find the scalar T > 0
    minimizing NLL of softmax(logits / T) on a held-out, in-distribution
    split. Fit once on clean validation logits, then applied unchanged to
    shifted/OOD logits at eval time -- it rescales confidence, it does not
    adapt to the shift itself (pair with BatchNorm adaptation for that).

    Parametrized as exp(log_T) so T stays positive without clamping.
    """
    log_temperature = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.exp().item()


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-3)
