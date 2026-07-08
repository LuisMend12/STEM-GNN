"""
2-panel figure comparing our trust gate variants against published baselines.
Left:  Cora injection sweep (our setup — baselines shown as horizontal ref lines)
Right: Texas heterophily — our gates + published baselines side by side
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

# ── Colors ────────────────────────────────────────────────────────────────────
GATE_COLOR = {
    "none":         "#888888",
    "A_cosine":     "#4C9BE8",
    "B_confidence": "#E86B4C",
    "C_learned":    "#7C4CE8",
    "D_llm":        "#4CC8A0",
}
GATE_LABEL = {
    "none":         "GraphSAGE (ours, no gate)",
    "A_cosine":     "Gate A: cosine similarity",
    "B_confidence": "Gate B: model confidence",
    "C_learned":    "Gate C: learned MLP",
    "D_llm":        "Gate D: LLM text cosine",
}

# Published baseline numbers (from original papers, standard splits)
# Cora clean accuracy:
CORA_LIT = {
    "GNNGuard\n(NeurIPS'20)": (82.9,  "#AAAAAA"),
    "H2GCN\n(NeurIPS'20)":    (87.9,  "#AAAAAA"),
    "GPR-GNN\n(ICLR'21)":     (88.6,  "#AAAAAA"),
    "TAPE\n(ICLR'24)":         (88.6,  "#AAAAAA"),
}
# Texas accuracy (60/20/20 splits avg, from published papers):
TEXAS_LIT = {
    "GNNGuard\n(NeurIPS'20)": (52.2,  "#BBBBBB"),
    "H2GCN\n(NeurIPS'20)":    (84.9,  "#BBBBBB"),
    "GPR-GNN\n(ICLR'21)":     (92.9,  "#BBBBBB"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                gridspec_kw={"wspace": 0.44})

# ── Panel (a): Cora injection sweep ──────────────────────────────────────────
GATES = ["none", "A_cosine", "B_confidence", "C_learned", "D_llm"]

for gate in GATES:
    accs = [v * 100 for v in cora[gate]]
    lw = 2.2 if gate != "none" else 1.6
    ls = "--" if gate == "none" else "-"
    ax1.plot(ratios, accs,
             color=GATE_COLOR[gate], lw=lw, ls=ls,
             marker="o", markersize=4.5,
             label=GATE_LABEL[gate], zorder=3)

# Reference lines for published baselines (clean Cora accuracy)
# Slightly offset overlapping labels so they don't collide
ref_offsets = {"GNNGuard\n(NeurIPS'20)": 0.4,
               "H2GCN\n(NeurIPS'20)":    0.4,
               "GPR-GNN\n(ICLR'21)":     0.4,
               "TAPE\n(ICLR'24)":        -1.2}
ref_styles = [(":", 0.6), ("-.", 0.5), ("--", 0.6), (":", 0.4)]
for (name, (acc, col)), (ls, alpha) in zip(CORA_LIT.items(), ref_styles):
    ax1.axhline(acc, color="#AAAAAA", lw=1.0, ls=ls, alpha=alpha, zorder=1)
    dy = ref_offsets[name]
    ax1.text(50, acc + dy, name.replace("\n", " "), fontsize=7.5,
             color="#999999", va="bottom", ha="right")

ax1.set_xlabel("Injected confusing edges (%)", fontsize=11)
ax1.set_ylabel("Test accuracy (%)", fontsize=11)
ax1.set_title("(a)  Noise robustness — Cora\n(confusing edge injection)", fontsize=12, fontweight="bold", pad=8)
ax1.set_xticks(ratios)
ax1.set_ylim(60, 94)
ax1.grid(axis="y", alpha=0.2, linestyle="--", color="#CCCCCC")
ax1.legend(fontsize=9, framealpha=0, loc="lower left", ncol=1)

ax1.text(0.98, 0.97, "Grey lines = published\nclean-setting baselines",
         transform=ax1.transAxes, fontsize=7.5, color="#AAAAAA",
         ha="right", va="top", style="italic")

# ── Panel (b): Texas heterophily comparison ───────────────────────────────────
# Rows: published baselines (gray) then our variants (colored)
# Order: our gates on bottom (low y), published baselines on top (high y)
# so reading top-to-bottom goes: published → separator → ours
our_names  = [GATE_LABEL[g] for g in GATES]
our_accs   = [texas[g] * 100 for g in GATES]
our_colors = [GATE_COLOR[g] for g in GATES]
our_hatch  = [""] * len(GATES)

pub_names  = list(TEXAS_LIT.keys())
pub_accs   = [v for v, _ in TEXAS_LIT.values()]
pub_colors = ["#CCCCCC"] * len(pub_names)
pub_hatch  = ["////"] * len(pub_names)

# Stack: ours (indices 0..4), blank separator, published (indices 6..8)
all_names  = our_names  + [""] + pub_names
all_accs   = our_accs   + [0]  + pub_accs
all_colors = our_colors + ["white"] + pub_colors
all_hatch  = our_hatch  + [""]      + pub_hatch

ys = np.arange(len(all_names))

for i, (name, acc, col, hatch) in enumerate(
        zip(all_names, all_accs, all_colors, all_hatch)):
    if name == "":
        continue
    ax2.barh(ys[i], acc, height=0.6,
             color=col, hatch=hatch,
             edgecolor="#DDDDDD" if hatch else "white",
             linewidth=0.8, zorder=2)
    if acc > 0:
        is_pub = i > len(GATES)
        ax2.text(acc + 0.8, ys[i], f"{acc:.1f}%",
                 va="center", fontsize=9.5,
                 color="#888888" if is_pub else "#222222",
                 fontweight="normal" if is_pub else "bold")

# Separator line between ours and published
sep_y = len(GATES) + 0.5
ax2.axhline(sep_y, color="#CCCCCC", lw=1.2, ls="-")
ax2.text(36, sep_y + 0.18, "published baselines (hatched) ↑",
         fontsize=8, color="#999999", style="italic", va="bottom")
ax2.text(36, sep_y - 0.18, "our gate variants ↓",
         fontsize=8, color="#555555", style="italic", va="top")

# Dashed vertical line at our no-gate baseline
ax2.axvline(texas["none"] * 100, color=GATE_COLOR["none"], lw=1.2, ls="--", alpha=0.5, zorder=1)

tick_labels = [n.replace("\n", " ") for n in all_names]
ax2.set_yticks(ys)
ax2.set_yticklabels(tick_labels, fontsize=9.5)
for i, (tick, col) in enumerate(zip(ax2.get_yticklabels(), all_colors)):
    if col not in ("white", "#CCCCCC"):
        tick.set_color(col)

ax2.set_xlabel("Test accuracy (%)", fontsize=11)
ax2.set_title("(b)  Heterophily generalization — Texas (h = 0.31)\nvs. published baselines (hatched)",
              fontsize=12, fontweight="bold", pad=8)
ax2.set_xlim(35, 105)
ax2.grid(axis="x", alpha=0.2, linestyle="--", color="#CCCCCC")

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle("LLM-Guided Semantic Trust Gate  —  Comparison to Prior Work",
             fontsize=13, fontweight="bold", y=1.02)

out = "/home/lam23005/STEM-GNN/figures/simple_results.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
