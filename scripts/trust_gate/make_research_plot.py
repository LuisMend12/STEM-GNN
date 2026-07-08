"""
Comprehensive 3-panel research figure:
  (a) Design-space taxonomy scatter
  (b) Injection robustness — dumbbell (clean vs 50%)
  (c) Texas heterophily — bars + literature refs
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# ── Theme ──────────────────────────────────────────────────────────────────
BG   = "#12141A"; CARD = "#1A1E28"
GREY = "#8A93A8"; RED  = "#FF4D4D"; GREEN = "#2ECC71"; GOLD  = "#FFC107"
BLUE = "#4590FF"; ORNG = "#FF7F2A"; TEAL  = "#45C4FF"; PURP  = "#A06CFF"
WHITE= "#FFFFFF"; MUTED= "#6A7385"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "--",
    "grid.color": "#35404A", "figure.facecolor": BG,
    "axes.facecolor": CARD, "axes.labelcolor": "#C8D0E0",
    "axes.edgecolor": "#35404A", "xtick.color": GREY,
    "ytick.color": GREY, "text.color": WHITE,
    "legend.framealpha": 0.3, "legend.edgecolor": "#35404A",
    "legend.facecolor": "#1E222C",
})

with open("/home/lam23005/STEM-GNN/sweep_results.json") as f:
    data = json.load(f)

cora  = data["cora_sweep"]
texas = data["heterophily"]["Texas"]
res_c = cora["results"]
res_t = texas["results"]
GATES = ["none", "A_cosine", "B_confidence", "C_learned", "D_llm"]

GSTYLE = {
    "none":         ("GraphSAGE",          GREY,  "o"),
    "A_cosine":     ("A  Cosine",          RED,   "s"),
    "B_confidence": ("B  Confidence",      ORNG,  "^"),
    "C_learned":    ("C  Learned MLP",     PURP,  "D"),
    "D_llm":        ("D  LLM text cosine", TEAL,  "P"),
}

fig = plt.figure(figsize=(17, 5.6), facecolor=BG)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1],
                      left=0.05, right=0.98, top=0.88, bottom=0.16,
                      wspace=0.38)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): Design Space Taxonomy
# ═══════════════════════════════════════════════════════════════════════════
ax1.set_facecolor(CARD)
ax1.set_title("(a)  Method Design Space", fontsize=11.5, fontweight="bold",
              color=WHITE, pad=7, loc="left")
ax1.set_xlabel("LLM text utilization  →", fontsize=10.5, labelpad=5)
ax1.set_ylabel("Structural adaptivity  →", fontsize=10.5, labelpad=5)
ax1.set_xlim(-0.08, 1.12); ax1.set_ylim(-0.08, 1.12)
ax1.set_xticks([0.0, 0.5, 1.0])
ax1.set_xticklabels(["None", "Features", "Semantic"], fontsize=9)
ax1.set_yticks([0.0, 0.5, 1.0])
ax1.set_yticklabels(["Equal-wt", "Edge-level", "Node-gate"], fontsize=9)

# Quadrant shading
for (xr, yr, lbl) in [((0,0.5),(0,0.5),"Legacy"), ((0.5,1.1),(0,0.5),"Text-aware"),
                       ((0,0.5),(0.5,1.1),"Structurally\nadaptive"),
                       ((0.5,1.1),(0.5,1.1),"Ideal frontier")]:
    col = GREEN if lbl == "Ideal frontier" else CARD
    ax1.axhspan(yr[0], yr[1], xmin=xr[0]/1.2, xmax=xr[1]/1.2,
                alpha=0.06 if lbl=="Ideal frontier" else 0.0,
                color=col, zorder=0)
ax1.fill_between([0.5, 1.12], [0.5, 0.5], [1.12, 1.12],
                 alpha=0.07, color=GREEN, zorder=0)

# Dashed frontier lines
ax1.axvline(0.5, color="#35404A", lw=0.8, ls=":")
ax1.axhline(0.5, color="#35404A", lw=0.8, ls=":")

# Methods positioned in design space
methods = {
    "GraphSAGE\n(NeurIPS'17)": (0.03, 0.04, GREY,  "o",  75,  (0.07, -0.06)),
    "GNNGuard\n(NeurIPS'20)":  (0.06, 0.50, RED,   "s",  75,  (0.10,  0.03)),
    "H2GCN\n(NeurIPS'20)":     (0.07, 0.80, GREEN, "^",  75,  (0.11,  0.03)),
    "GPR-GNN\n(ICLR'21)":      (0.10, 0.88, GOLD,  "D",  75,  (0.14, -0.08)),
    "TAPE\n(ICLR'24)":         (0.55, 0.06, BLUE,  "P",  75,  (0.10,  0.04)),
    "STEM-GNN\n(KDD'26)":      (0.57, 0.55, ORNG,  "h",  90,  (0.10,  0.04)),
    "Ours\n(proposed)":         (0.94, 0.95, TEAL,  "*", 220, (-0.02, 0.04)),
}

for name, (x, y, col, mrk, sz, (dx, dy)) in methods.items():
    ax1.scatter(x, y, color=col, marker=mrk, s=sz, zorder=5,
                edgecolors=BG, linewidths=0.9)
    ax1.annotate(name, (x, y), xytext=(x + dx, y + dy),
                 fontsize=7.8, color=col, va="center",
                 arrowprops=dict(arrowstyle="-", color=col, lw=0.5)
                 if abs(dx) > 0.05 or abs(dy) > 0.05 else None)

ax1.text(0.76, 0.97, "Ideal\nfrontier", fontsize=8.5, color=GREEN,
         ha="center", va="top", style="italic",
         transform=ax1.transAxes)

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): Injection Robustness — dumbbell (clean vs 50%)
# ═══════════════════════════════════════════════════════════════════════════
ax2.set_facecolor(CARD)
ax2.set_title("(b)  Cora — Clean vs 50% Injection", fontsize=11.5,
              fontweight="bold", color=WHITE, pad=7, loc="left")
ax2.set_xlabel("Test accuracy  (%)", fontsize=10.5, labelpad=5)
ax2.set_xlim(60, 85)
ax2.grid(axis="x", alpha=0.2, linestyle="--", color="#35404A")
ax2.grid(axis="y", visible=False)

gate_labels = [GSTYLE[g][0] for g in GATES]
colors       = [GSTYLE[g][1] for g in GATES]
ys           = np.arange(len(GATES))[::-1]  # top=none, bottom=D_llm

for i, (gate, y) in enumerate(zip(GATES, ys)):
    clean = res_c[gate][0] * 100
    inj50 = res_c[gate][-1] * 100
    col   = GSTYLE[gate][1]
    mrk   = GSTYLE[gate][2]
    # connecting line
    ax2.plot([inj50, clean], [y, y], color=col, lw=2.2, alpha=0.55, zorder=2)
    # clean dot (open)
    ax2.scatter(clean, y, color=col, marker="o", s=90, zorder=4,
                edgecolors=col, linewidths=1.8, facecolors=BG)
    # 50% dot (filled)
    ax2.scatter(inj50, y, color=col, marker=mrk, s=80, zorder=4,
                edgecolors=BG, linewidths=0.8)
    # drop label
    drop = inj50 - clean
    ax2.text(inj50 - 0.4, y, f"{drop:+.1f} pp", fontsize=8.5,
             color=col, ha="right", va="center", fontweight="bold")
    ax2.text(clean + 0.4, y, f"{clean:.1f}", fontsize=8.5,
             color=col, ha="left", va="center")

ax2.set_yticks(ys)
ax2.set_yticklabels([GSTYLE[g][0] for g in GATES], fontsize=9.5)
for tick, gate in zip(ax2.get_yticklabels(), GATES):
    tick.set_color(GSTYLE[gate][1])

# Legend
leg_handles = [
    Line2D([0],[0], marker="o", color=WHITE, mfc=BG, mec=WHITE,
           ms=7, lw=0, label="Clean (0%)"),
    Line2D([0],[0], marker="s", color=WHITE, mfc=WHITE, mec=BG,
           ms=7, lw=0, label="50% injection"),
]
ax2.legend(handles=leg_handles, fontsize=9, loc="lower right")

# ═══════════════════════════════════════════════════════════════════════════
# Panel (c): Texas Heterophily — bars + literature refs
# ═══════════════════════════════════════════════════════════════════════════
ax3.set_facecolor(CARD)
ax3.set_title("(c)  Texas — Heterophily  (h = 0.31)", fontsize=11.5,
              fontweight="bold", color=WHITE, pad=7, loc="left")
ax3.set_xlabel("Test accuracy  (%)", fontsize=10.5, labelpad=5)
ax3.set_xlim(40, 100)
ax3.grid(axis="x", alpha=0.2, linestyle="--", color="#35404A")
ax3.grid(axis="y", visible=False)

all_methods = list(GATES) + ["H2GCN", "GPR-GNN"]
labels_r = [GSTYLE[g][0] if g in GSTYLE else g for g in all_methods]
accs = ([res_t[g] * 100 for g in GATES]
        + [84.86, 92.92])
cols_r = ([GSTYLE[g][1] for g in GATES]
          + [GREEN, GOLD])
is_lit = [False]*5 + [True, True]
ys3 = np.arange(len(all_methods))[::-1]

for i, (lbl, acc, col, lit, y) in enumerate(zip(labels_r, accs, cols_r, is_lit, ys3)):
    bar_col = col
    alpha   = 0.45 if lit else 0.80
    ax3.barh(y, acc, height=0.55, color=bar_col, alpha=alpha,
             edgecolor=BG, linewidth=0.6, zorder=3)
    ax3.text(acc + 0.5, y, f"{acc:.1f}%", fontsize=9, color=col,
             va="center", fontweight="bold" if not lit else "normal")
    if lit:
        ax3.text(acc - 2.5, y, "lit.", fontsize=7.5, color=col,
                 va="center", ha="right", style="italic", alpha=0.7)

# separator between ours/literature
ax3.axhline(ys3[4] - 0.5, color="#35404A", lw=0.8, ls="--", alpha=0.6)

ax3.set_yticks(ys3)
ax3.set_yticklabels(labels_r, fontsize=9.5)
for tick, (gate, lit) in zip(ax3.get_yticklabels(), zip(all_methods, is_lit)):
    tick.set_color(GSTYLE[gate][1] if gate in GSTYLE else
                   (GREEN if gate == "H2GCN" else GOLD))
    if lit:
        tick.set_fontstyle("italic")

# ── Supertitle + footnote ─────────────────────────────────────────────────
fig.suptitle("LLM-Guided Semantic Trust Gate  —  Research Summary",
             fontsize=14, fontweight="bold", color=WHITE, y=0.97)

fig.text(0.5, 0.005,
         "Our gates (A–D) run on Cora/Texas (3 seeds, 200 ep).  "
         "H2GCN & GPR-GNN Texas numbers from their original papers "
         "(different training protocol — not directly comparable).  "
         "Proposed method (Ours) in design-space panel is aspirational.",
         ha="center", fontsize=8, color=MUTED, style="italic")

out = "/home/lam23005/STEM-GNN/research_figure_v2.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
