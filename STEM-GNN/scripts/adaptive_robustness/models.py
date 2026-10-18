"""Candidate robust node classifiers for the adaptive-attack probe.

Every model has the signature ``forward(x, edge_index, edge_weight=None)`` so
PyG's PRBCD attack can differentiate through ``edge_weight``. ``x`` is the
frozen sentence-embedding matrix of a text-attributed graph, so any score
computed from ``x`` is a frozen text score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter


def _weights(edge_index, edge_weight):
    if edge_weight is None:
        return torch.ones(edge_index.size(1), device=edge_index.device)
    return edge_weight


def text_trust(x, edge_index, tau):
    """Frozen text score s_ij = relu(cos(x_i, x_j) - tau) / (1 - tau)."""
    src, dst = edge_index
    cos = F.cosine_similarity(x[src], x[dst], dim=-1)
    return F.relu(cos - tau) / (1.0 - tau)


class WeightedSAGELayer(nn.Module):
    """h_i = W_self x_i + m_i * W_nb * (sum_j c_ij x_j / sum_j c_ij).

    ``mass`` (m_i) defaults to 1, which gives a weighted-mean GraphSAGE layer.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nb = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, coeff, mass=None):
        src, dst = edge_index
        n = x.size(0)
        agg = scatter(coeff.unsqueeze(-1) * x[src], dst, dim=0, dim_size=n, reduce="sum")
        norm = scatter(coeff, dst, dim=0, dim_size=n, reduce="sum")
        mean = agg / norm.clamp_min(1e-9).unsqueeze(-1)
        nb = self.lin_nb(mean)
        if mass is not None:
            nb = mass.unsqueeze(-1) * nb
        return self.lin_self(x) + nb


class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.dropout = nn.Linear(in_dim, hid), nn.Linear(hid, out_dim), dropout

    def forward(self, x, edge_index=None, edge_weight=None):
        h = F.dropout(F.relu(self.l1(x)), self.dropout, self.training)
        return self.l2(h)


class GCN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        super().__init__()
        self.c1, self.c2, self.dropout = GCNConv(in_dim, hid), GCNConv(hid, out_dim), dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.c1(x, edge_index, edge_weight))
        h = F.dropout(h, self.dropout, self.training)
        return self.c2(h, edge_index, edge_weight)


class SAGE(nn.Module):
    """Weighted-mean GraphSAGE (STEM-GNN's default backbone family).

    ``mode``:
      - ``"plain"``: observed edges only.
      - ``"text_gate"``: GNNGuard-style; edges weighted by the frozen text score.
      - ``"trust_budget"``: text-weighted mean, scaled by a per-node mass
        m_i = S_i / (S_i + kappa) with S_i = sum_j w_ij s_ij. Adding edges can
        never raise the neighbor share above 1, and a node whose neighbors are
        all low-trust falls back to its own text.
    """

    def __init__(self, in_dim, hid, out_dim, dropout=0.5, mode="plain", tau=0.0, kappa=0.5):
        super().__init__()
        self.l1, self.l2 = WeightedSAGELayer(in_dim, hid), WeightedSAGELayer(hid, out_dim)
        self.dropout, self.mode, self.tau, self.kappa = dropout, mode, tau, kappa

    def _coeff_mass(self, x, edge_index, edge_weight):
        w = _weights(edge_index, edge_weight)
        if self.mode == "plain":
            return w, None
        c = w * text_trust(x, edge_index, self.tau)
        if self.mode == "text_gate":
            return c, None
        total = scatter(c, edge_index[1], dim=0, dim_size=x.size(0), reduce="sum")
        return c, total / (total + self.kappa)

    def hidden(self, x, edge_index, edge_weight=None):
        coeff, mass = self._coeff_mass(x, edge_index, edge_weight)
        return F.relu(self.l1(x, edge_index, coeff, mass)), coeff, mass

    def forward(self, x, edge_index, edge_weight=None):
        h, coeff, mass = self.hidden(x, edge_index, edge_weight)
        h = F.dropout(h, self.dropout, self.training)
        return self.l2(h, edge_index, coeff, mass)


class RUNGLite(nn.Module):
    """RUNG-style robust propagation (simplified; not the reference code).

    F0 = MLP(x). Then K Jacobi/IRLS steps on
        sum_ij w_ij rho(y_ij) + lam * sum_i ||F_i - F0_i||^2,
    F_i <- (sum_j c_ij F_j + lam F0_i) / (sum_j c_ij + lam).
    The discrepancy y_ij is computed on softmax(F) so gamma is scale-free.
    The weights follow the MCP-IRLS shape (proportional to 1/y, zero at
    y >= gamma), bounded by 1:
        W_ij = min(1, delta / y) * relu(1 - y / gamma).
    ``penalty="l2"`` sets W = 1, which gives APPNP-like smoothing (no pruning).
    ``text=True`` multiplies the weights by the frozen text score (Q1 x Q3).
    """

    def __init__(self, in_dim, hid, out_dim, dropout=0.5, k=10, lam=1.0,
                 gamma=0.8, delta=0.05, penalty="mcp", text=False, tau=0.0):
        super().__init__()
        self.mlp = MLP(in_dim, hid, out_dim, dropout)
        self.k, self.lam, self.gamma, self.delta = k, lam, gamma, delta
        self.penalty, self.text, self.tau = penalty, text, tau

    def forward(self, x, edge_index, edge_weight=None):
        w = _weights(edge_index, edge_weight)
        if self.text:
            w = w * text_trust(x, edge_index, self.tau)
        src, dst = edge_index
        n = x.size(0)
        f0 = self.mlp(x)
        f = f0
        for _ in range(self.k):
            if self.penalty == "mcp":
                p = F.softmax(f, dim=-1)
                y = (p[src] - p[dst]).norm(dim=-1).clamp_min(1e-6)
                c = w * torch.clamp(self.delta / y, max=1.0) * F.relu(1.0 - y / self.gamma)
            else:
                c = w
            num = scatter(c.unsqueeze(-1) * f[src], dst, dim=0, dim_size=n, reduce="sum")
            den = scatter(c, dst, dim=0, dim_size=n, reduce="sum")
            f = (num + self.lam * f0) / (den + self.lam).unsqueeze(-1)
        return f
