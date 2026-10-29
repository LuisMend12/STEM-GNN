"""STEM-lite: STEM-GNN's finetune path (SAGE encoder -> frozen cosine VQ ->
per-head linear decoders, logits averaged over heads) without pretraining or MoE.

``vq_mode`` selects what the quantizer does on the forward pass:
  "ste"  : straight-through (forward = code, gradient = identity). Training and
           the BPDA attack surrogate.
  "hard" : forward = code, no gradient to the encoder. This is what
           STEM-GNN's model/vq.py does in eval mode.
  "none" : no quantization (continuous per-head decoder, stage 1).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adaptive_robustness"))
from models import WeightedSAGELayer, _weights  # noqa: E402


class STEMLite(nn.Module):
    def __init__(self, in_dim, n_cls, hid=128, heads=4, codes=32, dropout=0.5):
        super().__init__()
        self.l1, self.l2 = WeightedSAGELayer(in_dim, hid), WeightedSAGELayer(hid, hid)
        self.heads, self.dh, self.dropout = heads, hid // heads, dropout
        self.register_buffer("codebook", F.normalize(torch.randn(heads, codes, hid // heads), dim=-1))
        self.decoders = nn.ModuleList(nn.Linear(self.dh, n_cls) for _ in range(heads))
        self.vq_mode = "none"

    def encode(self, x, edge_index, edge_weight=None):
        w = _weights(edge_index, edge_weight)
        h = F.relu(self.l1(x, edge_index, w))
        h = F.dropout(h, self.dropout, self.training)
        return self.l2(h, edge_index, w)

    def split_heads(self, z):
        return F.normalize(z.view(z.size(0), self.heads, self.dh), dim=-1)

    def assign(self, zh):
        """Cosine similarities [N, H, K] and nearest-code indices [N, H]."""
        sims = torch.einsum("nhd,hkd->nhk", zh, self.codebook)
        return sims, sims.argmax(-1)

    def quantize(self, zh):
        sims, idx = self.assign(zh)
        q = torch.stack([self.codebook[h][idx[:, h]] for h in range(self.heads)], dim=1)
        if self.vq_mode == "ste":
            return zh + (q - zh).detach(), sims, idx
        # Hard mode: value = code, gradient to the encoder = 0 (what an attacker
        # differentiating STEM-GNN's eval-mode VQ sees). The 0*zh term keeps the
        # autograd graph connected so the attack runs instead of erroring.
        return q.detach() + 0.0 * zh, sims, idx

    def head_logits(self, zh):
        return torch.stack([dec(zh[:, h]) for h, dec in enumerate(self.decoders)], 1).mean(1)

    def forward_full(self, x, edge_index, edge_weight=None):
        zh = self.split_heads(self.encode(x, edge_index, edge_weight))
        if self.vq_mode == "none":
            return self.head_logits(zh), zh, None, None
        q, sims, idx = self.quantize(zh)
        return self.head_logits(q), zh, sims, idx

    def forward(self, x, edge_index, edge_weight=None):
        return self.forward_full(x, edge_index, edge_weight)[0]

    @torch.no_grad()
    def init_codebook(self, x, edge_index, iters=30, seed=0):
        """Spherical k-means per head on clean-graph embeddings of all nodes."""
        self.eval()
        zh = self.split_heads(self.encode(x, edge_index))
        gen = torch.Generator().manual_seed(seed)
        k = self.codebook.size(1)
        for h in range(self.heads):
            data = zh[:, h]
            cent = data[torch.randperm(data.size(0), generator=gen)[:k]].clone()
            for _ in range(iters):
                a = (data @ cent.t()).argmax(-1)
                for c in range(k):
                    members = data[a == c]
                    cent[c] = members.mean(0) if members.numel() else data[torch.randint(0, data.size(0), (1,), generator=gen)][0]
                cent = F.normalize(cent, dim=-1)
            self.codebook[h] = cent


def commitment_loss(zh, sims, idx, model):
    q = torch.stack([model.codebook[h][idx[:, h]] for h in range(model.heads)], dim=1)
    return (zh - q.detach()).pow(2).sum(-1).mean()


def code_margin_loss(sims, margin=0.1):
    top2 = sims.topk(2, dim=-1).values
    return F.relu(margin - (top2[..., 0] - top2[..., 1])).mean()
