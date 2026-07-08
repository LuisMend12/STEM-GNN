"""
3-panel contribution figure:
  (a) Gap table — what each method handles (contribution framing)
  (b) Cora injection sweep — noise robustness
  (c) Texas heterophily — comparison vs. published baselines
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("/home/lam23005/STEM-GNN/results/sweep_results.json") as f:
    data = json.load(f)

ratios = [r * 100 for r in data["cora_sweep"]["ratios"]]
cora   = data["cora_sweep"]["results"]
texas  = data["heterophily"]["Texas"]["results"]

GATES = ["none", "A_cosine", "B_confidence", "C_learned", "D_llm"]
GATE_COLOR = {
    "none":         "#888888",
    "A_cosine":     "#4C9BE8",
    "B_confidence": "#E86B4C",
    "C_learned":    "#7C4CE8",
    "D_llm":        "#4CC8A0",
}
GATE_LABEL = {
    "none":         "GraphSAGE (no gate)",
    "A_cosine":     "Gate A: cosine",
    "B_confidence": "Gate B: confidence",
    "C_learned":    "Gate C: learned MLP",
    "D_llm":        "Gate D: LLM cosine",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig = plt.figure(figsize=(17, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25],
                      left=0.03, right=0.98, top=0.88, bottom=0.12,
                      wspace=0.36)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): Contribution gap table
# ═══════════════════════════════════════════════════════════════════════════════
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.axis("off")
ax1.set_title("(a)  Research Gap", fontsize=12, fontweight="bold", pad=8, loc="left")

methods = [
    ("GraphSAGE",        False, False, False),
    ("GNNGuard (N'20)",  False, True,  False),
    ("H2GCN (N'20)",     False, False, True),
    ("GPR-GNN (I'21)",   False, False, True),
    ("TAPE (I'24)",      True,  False, False),
    ("STEM-GNN (K'26)",  True,  False, False),
    ("Ours (Trust Gate)",True,  True,  True),
]

cols   = ["Method", "LLM\nfeatures", "Injection\nrobust", "Hetero-\nphily"]
col_xs = [0.02, 0.48, 0.65, 0.83]
header_y = 0.93
row_h    = 0.108

# Column headers
for cx, ch in zip(col_xs, cols):
    ax1.text(cx, header_y, ch, fontsize=8.8, fontweight="bold",
             color="#333333", va="top", ha="left")

# Divider below header
ax1.plot([0.0, 1.0], [header_y - 0.045, header_y - 0.045],
         color="#CCCCCC", lw=1.0, transform=ax1.transAxes)

CHECK = "✓"; CROSS = "✗"
for i, (name, llm, inj, het) in enumerate(methods):
    y = header_y - 0.065 - i * row_h
    is_ours = i == len(methods) - 1

    # Row highlight for ours
    if is_ours:
        rect = mpatches.FancyBboxPatch((0.0, y - 0.045), 1.0, row_h - 0.008,
                                        boxstyle="round,pad=0.005",
                                        linewidth=1.2, edgecolor="#4C9BE8",
                                        facecolor="#EEF4FF",
                                        transform=ax1.transAxes, zorder=0)
        ax1.add_patch(rect)

    name_color = "#1A56DB" if is_ours else "#333333"
    name_weight = "bold" if is_ours else "normal"
    ax1.text(col_xs[0], y, name, fontsize=8.5, color=name_color,
             fontweight=name_weight, va="center", ha="left")

    for cx, val in zip(col_xs[1:], [llm, inj, het]):
        sym   = CHECK if val else CROSS
        color = "#16A34A" if val else "#DC2626"
        fw    = "bold" if (is_ours and val) else "normal"
        ax1.text(cx + 0.07, y, sym, fontsize=10, color=color,
                 fontweight=fw, va="center", ha="center")

# Gap label
ax1.text(0.5, 0.005,
         "No prior method handles all three simultaneously",
         fontsize=8, color="#888888", style="italic",
         ha="center", va="bottom", transform=ax1.transAxes)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Cora injection sweep
# ═══════════════════════════════════════════════════════════════════════════════
for gate in GATES:
    accs = [v * 100 for v in cora[gate]]
    lw = 2.2 if gate != "none" else 1.6
    ls = "--" if gate == "none" else "-"
    ax2.plot(ratios, accs, color=GATE_COLOR[gate], lw=lw, ls=ls,
             marker="o", markersize=4.5, label=GATE_LABEL[gate], zorder=3)

ax2.set_xlabel("Injected confusing edges (%)", fontsize=11)
ax2.set_ylabel("Test accuracy (%)", fontsize=11)
ax2.set_title("(b)  Noise robustness — Cora", fontsize=12, fontweight="bold", pad=8, loc="left")
ax2.set_xticks(ratios)
ax2.set_ylim(60, 84)
ax2.grid(axis="y", alpha=0.2, linestyle="--", color="#CCCCCC")
ax2.legend(fontsize=8.8, framealpha=0, loc="lower left")

# Annotate key takeaway

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (c): Texas heterophily — our gates + published baselines
# ═══════════════════════════════════════════════════════════════════════════════
TEXAS_LIT = {
    "GNNGuard (NeurIPS'20)": 52.2,
    "H2GCN (NeurIPS'20)":    84.9,
    "GPR-GNN (ICLR'21)":     92.9,
}

our_names  = [GATE_LABEL[g] for g in GATES]
our_accs   = [texas[g] * 100 for g in GATES]
our_colors = [GATE_COLOR[g] for g in GATES]

pub_names  = list(TEXAS_LIT.keys())
pub_accs   = list(TEXAS_LIT.values())

all_names  = our_names + [""] + pub_names
all_accs   = our_accs  + [0]  + pub_accs
all_colors = our_colors + ["white"] + ["#CCCCCC"] * len(pub_names)
all_hatch  = [""] * len(GATES) + [""] + ["////"] * len(pub_names)

ys = np.arange(len(all_names))

for i, (name, acc, col, hatch) in enumerate(
        zip(all_names, all_accs, all_colors, all_hatch)):
    if name == "":
        continue
    ax3.barh(ys[i], acc, height=0.6, color=col, hatch=hatch,
             edgecolor="#DDDDDD" if hatch else "white",
             linewidth=0.8, zorder=2)
    if acc > 0:
        is_pub = i > len(GATES)
        ax3.text(acc + 0.8, ys[i], f"{acc:.1f}%",
                 va="center", fontsize=9,
                 color="#888888" if is_pub else "#222222",
                 fontweight="normal" if is_pub else "bold")

# Separator
sep_y = len(GATES) + 0.5
ax3.axhline(sep_y, color="#CCCCCC", lw=1.0)
ax3.text(34, sep_y + 0.15, "published ↑", fontsize=7.5, color="#AAAAAA", style="italic")
ax3.text(34, sep_y - 0.15, "ours ↓",      fontsize=7.5, color="#555555", style="italic", va="top")

# Baseline dashed line
ax3.axvline(texas["none"] * 100, color=GATE_COLOR["none"], lw=1.2, ls="--", alpha=0.5)


tick_labels = [n for n in all_names]
ax3.set_yticks(ys)
ax3.set_yticklabels(tick_labels, fontsize=9)
for i, (tick, col) in enumerate(zip(ax3.get_yticklabels(), all_colors)):
    if col not in ("white", "#CCCCCC"):
        tick.set_color(col)

ax3.set_xlabel("Test accuracy (%)", fontsize=11)
ax3.set_title("(c)  Heterophily — Texas (h = 0.31)", fontsize=12, fontweight="bold", pad=8, loc="left")
ax3.set_xlim(30, 106)
ax3.grid(axis="x", alpha=0.2, linestyle="--", color="#CCCCCC")

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle(
    "LLM-Guided Semantic Trust Gate  —  Contribution & Empirical Results",
    fontsize=13, fontweight="bold", y=0.99)

out = "/home/lam23005/STEM-GNN/figures/contribution_results.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
