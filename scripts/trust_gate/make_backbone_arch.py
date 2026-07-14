"""
Architecture diagram: GCN/GraphSAGE backbone + LLM-only gate signal.
Highlights the decoupling: backbone handles aggregation, LLM handles trust.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
fig.patch.set_facecolor("white")

def box(ax, x, y, w, h, color, label, sub=None, badge=None, radius=0.35):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.1", linewidth=2,
        edgecolor=color, facecolor=color + "22", zorder=2)
    ax.add_patch(rect)
    ax.text(x, y + (0.18 if sub else 0), label,
            ha="center", va="center", fontsize=12, fontweight="bold", color=color, zorder=3)
    if sub:
        ax.text(x, y - 0.32, sub, ha="center", va="center",
                fontsize=9.5, color="#555555", style="italic", zorder=3)
    if badge:
        bx = mpatches.FancyBboxPatch((x - w/2 + 0.1, y + h/2 - 0.38), w - 0.2, 0.34,
            boxstyle="round,pad=0.05", linewidth=0, facecolor=color, zorder=4)
        ax.add_patch(bx)
        ax.text(x, y + h/2 - 0.21, badge, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=5)

def arrow(ax, x1, y1, x2, y2, color="#555555", label=None, style="arc3,rad=0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color,
                        connectionstyle=style))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom", fontsize=9, color=color)

# ── Top row: backbone path ──────────────────────────────────────────────────
# Input node features
box(ax, 1.5, 4.5, 2.2, 1.2, "#555555",
    "Node Features", "x_v  (LLM embeddings)")

# GCN / GraphSAGE backbone
box(ax, 5.0, 4.5, 2.6, 1.2, "#2563EB",
    "GNN Backbone",
    "GCN or GraphSAGE",
    badge="ENCODER")

# Hidden representation
box(ax, 8.6, 4.5, 2.2, 1.2, "#555555",
    "Node Repr.", "h_v  (128-dim)")

# Output
box(ax, 11.5, 4.5, 1.8, 1.2, "#D97706",
    "Classifier",
    "softmax(Wh_v)")

arrow(ax, 2.6, 4.5, 3.7, 4.5, "#555555", "x_v")
arrow(ax, 6.3, 4.5, 7.5, 4.5, "#2563EB", "h_v")
arrow(ax, 9.7, 4.5, 10.6, 4.5, "#555555")

# ── Bottom row: LLM gate signal ──────────────────────────────────────────────
box(ax, 1.5, 1.8, 2.2, 1.2, "#7C3AED",
    "LLM Embeddings", "e_v  (frozen, 768-dim)")

box(ax, 5.0, 1.8, 2.6, 1.4, "#7C3AED",
    "Semantic Trust Gate",
    "β_v = 1 − σ(cos(e_v, ē_u))",
    badge="OUR CONTRIBUTION")

# Arrow: LLM embeddings → gate
arrow(ax, 2.6, 1.8, 3.7, 1.8, "#7C3AED", "e_v")

# Arrow: gate signal β_v goes up to GNN backbone
arrow(ax, 5.0, 2.5, 5.0, 3.9, "#7C3AED", "β_v", style="arc3,rad=0.0")

# Dashed separator line
ax.plot([3.7, 3.7], [0.6, 5.7], color="#CCCCCC", lw=1.5, ls="--", zorder=1)
ax.text(3.7, 0.35, "DECOUPLE", ha="center", va="center",
        fontsize=8, color="#AAAAAA", fontweight="bold")

# ── Equation box ────────────────────────────────────────────────────────────
eq_box = mpatches.FancyBboxPatch((5.5, 0.3), 6.2, 1.05,
    boxstyle="round,pad=0.1", linewidth=1.5,
    edgecolor="#E2E8F0", facecolor="#F8FAFF", zorder=2)
ax.add_patch(eq_box)
ax.text(8.6, 1.0,
        r"h_v = β_v · W_self(x_v)  +  (1 − β_v) · W_neigh( mean x_u )",
        ha="center", va="center", fontsize=10.5,
        color="#1E293B", family="monospace", zorder=3)
ax.text(8.6, 0.52,
        "β_v ∈ [0, 1]  |  β_v → 1: trust self  |  β_v → 0: trust neighbours",
        ha="center", va="center", fontsize=8.5, color="#64748B", zorder=3)

# ── Labels ───────────────────────────────────────────────────────────────────
ax.text(0.15, 5.7, "Backbone path:", fontsize=9, color="#2563EB",
        fontweight="bold", va="top")
ax.text(0.15, 2.5, "Gate signal:", fontsize=9, color="#7C3AED",
        fontweight="bold", va="top")

# Title
ax.text(6.5, 5.85,
        "LLM-Guided Semantic Trust Gate  —  Backbone + Gate Decoupling",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1E293B")

out = "/home/lam23005/STEM-GNN/figures/arch_backbone.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
