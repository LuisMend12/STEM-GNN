"""
Minimal contribution diagram — 3 steps, contribution box highlighted.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── 3 boxes ───────────────────────────────────────────────────────────────────
# Box 1: Input (small, gray)
# Box 2: Trust Gate (large, purple — the contribution)
# Box 3: Output (small, orange)

def box(ax, x, y, w, h, fc, ec, lw=1.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)

def arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=2.2, mutation_scale=20), zorder=3)

# ── Box 1: Input ──────────────────────────────────────────────────────────────
box(ax, 0.3, 1.0, 2.8, 3.0, "#F7FAFC", "#A0AEC0", lw=1.2)
ax.text(1.7, 3.72, "Input", fontsize=12, fontweight="bold",
        color="#4A5568", ha="center")
ax.text(1.7, 3.15, "Node  v  +  Neighbours  N(v)", fontsize=10,
        color="#4A5568", ha="center")
ax.plot([0.55, 2.85], [2.9, 2.9], color="#CBD5E0", lw=0.8)
ax.text(1.7, 2.55, "LLM Encoder  φ", fontsize=10, color="#718096",
        ha="center", style="italic")
ax.text(1.7, 2.00, "t_v  →  e_v  ∈  ℝ⁷⁶⁸", fontsize=10,
        color="#4A5568", ha="center", fontfamily="DejaVu Sans Mono")
ax.text(1.7, 1.42, "{t_u}  →  {e_u}  ∈  ℝ⁷⁶⁸", fontsize=10,
        color="#4A5568", ha="center", fontfamily="DejaVu Sans Mono")

arrow(ax, 3.1, 3.6, 2.5)

# ── Box 2: Trust Gate — CONTRIBUTION ─────────────────────────────────────────
box(ax, 3.6, 0.5, 6.8, 4.0, "#FAF5FF", "#805AD5", lw=2.8)

# "Our Contribution" badge
badge = FancyBboxPatch((4.5, 4.08), 4.9, 0.52,
                        boxstyle="round,pad=0.06",
                        linewidth=0, facecolor="#805AD5", zorder=4)
ax.add_patch(badge)
ax.text(6.95, 4.34, "Our Contribution", fontsize=11, fontweight="bold",
        color="white", ha="center", va="center", zorder=5)

ax.text(6.95, 4.05, "Semantic Trust Gate", fontsize=14, fontweight="bold",
        color="#553C9A", ha="center", va="top")

ax.text(6.95, 3.30,
        "β_v  =  σ( f( e_v ,  mean{e_u} ) )",
        fontsize=14, color="#322659", ha="center",
        fontfamily="DejaVu Sans Mono", fontweight="bold")

ax.plot([4.0, 9.9], [2.88, 2.88], color="#B794F4", lw=0.9)

ax.text(6.95, 2.58, "Gate signal  f  —  4 variants explored:", fontsize=10,
        color="#6B46C1", ha="center", fontweight="bold")

ax.text(6.95, 2.05,
        "A  cosine( h_v, h̄_u )     B  confidence( logits_v )",
        fontsize=9.5, color="#4A5568", ha="center",
        fontfamily="DejaVu Sans Mono")
ax.text(6.95, 1.55,
        "C  MLP( e_v ‖ ē_u )       D  cosine( e_v, ē_u )",
        fontsize=9.5, color="#4A5568", ha="center",
        fontfamily="DejaVu Sans Mono")

arrow(ax, 10.4, 10.9, 2.5)

# ── Box 3: Output ─────────────────────────────────────────────────────────────
box(ax, 10.9, 1.0, 2.8, 3.0, "#FFFAF0", "#ED8936", lw=1.2)
ax.text(12.3, 3.72, "Output", fontsize=12, fontweight="bold",
        color="#744210", ha="center")
ax.text(12.3, 3.20, "h_v  =", fontsize=11, color="#4A5568",
        ha="center", fontfamily="DejaVu Sans Mono")
ax.text(12.3, 2.68, "β_v · W_self(e_v)", fontsize=10, color="#4A5568",
        ha="center", fontfamily="DejaVu Sans Mono")
ax.text(12.3, 2.16, "+ (1−β_v) · W_neigh(ē_u)", fontsize=10, color="#4A5568",
        ha="center", fontfamily="DejaVu Sans Mono")
ax.plot([11.15, 13.45], [1.9, 1.9], color="#FBD38D", lw=0.8)
ax.text(12.3, 1.60, "β_v → 1 : trust self", fontsize=9,
        color="#744210", ha="center")
ax.text(12.3, 1.20, "β_v → 0 : trust nbrs", fontsize=9,
        color="#744210", ha="center")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.0, 4.82, "LLM-Guided Semantic Trust Gate",
        fontsize=15, fontweight="bold", color="#1A202C", ha="center")

out = "/home/lam23005/STEM-GNN/figures/arch_simple.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
