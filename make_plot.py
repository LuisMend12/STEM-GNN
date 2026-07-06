"""
Generates research_figure.png — two-panel research figure.
Left: accuracy vs. injection ratio (Cora). Right: accuracy on Texas heterophily.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.2,
    "grid.linestyle":   "--",
    "grid.color":       "#35404A",
    "figure.facecolor": "#12141A",
    "axes.facecolor":   "#1A1E28",
    "axes.labelcolor":  "#C8D0E0",
    "axes.edgecolor":   "#35404A",
    "xtick.color":      "#8A93A8",
    "ytick.color":      "#8A93A8",
    "text.color":       "#FFFFFF",
    "legend.framealpha": 0.35,
    "legend.edgecolor": "#35404A",
    "legend.facecolor": "#1E222C",
})

GATE_STYLE = {
    "none":         ("GraphSAGE (no gate)",    "#8A93A8", "o", "--", 1.8, 7),
    "A_cosine":     ("A  Cosine (GNNGuard)",   "#FF4D4D", "s", "-",  2.2, 7),
    "B_confidence": ("B  Confidence (Mowst)",  "#FFA02A", "^", "-",  2.2, 8),
    "C_learned":    ("C  Learned (ACM)",       "#A06CFF", "D", "-",  2.2, 7),
    "D_llm":        ("D  LLM text cosine (Ours*)", "#45C4FF", "P", "-", 2.6, 9),
}

with open("/home/lam23005/STEM-GNN/sweep_results.json") as f:
    data = json.load(f)

cora   = data["cora_sweep"]
texas  = data["heterophily"]["Texas"]

ratios  = [r * 100 for r in cora["ratios"]]
results = cora["results"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                gridspec_kw={"width_ratios": [1.65, 1.0]})
fig.patch.set_facecolor("#12141A")
fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.18, wspace=0.38)

# ── Left panel: injection sweep ───────────────────────────────────────────
ax1.set_facecolor("#1A1E28")
ax1.set_title("(a)  Cora — Confusing-edge Injection Sweep", fontsize=12,
              fontweight="bold", color="white", pad=8, loc="left")
ax1.set_xlabel("Injection ratio  (%)", fontsize=11, labelpad=6)
ax1.set_ylabel("Test accuracy  (%)", fontsize=11, labelpad=6)
ax1.set_xlim(-3, 53)
ax1.set_ylim(62, 84)
ax1.set_xticks([0, 10, 20, 30, 40, 50])

drop_labels = {}
for gate, (label, color, marker, ls, lw, ms) in GATE_STYLE.items():
    accs = [v * 100 for v in results[gate]]
    ax1.plot(ratios, accs, color=color, marker=marker, linestyle=ls,
             linewidth=lw, markersize=ms, label=label, zorder=3,
             markeredgecolor="#12141A", markeredgewidth=0.6)
    # Drop annotation at 50%
    drop = accs[-1] - accs[0]
    drop_labels[gate] = (ratios[-1], accs[-1], drop, color)

# Annotate drops — stagger vertically to avoid overlap
y_order = sorted(drop_labels.items(), key=lambda kv: kv[1][1])  # sort by y
used_y = []
for gate, (x, y, drop, col) in y_order:
    # nudge if too close to an already-placed label
    label_y = y
    for uy in used_y:
        if abs(label_y - uy) < 0.9:
            label_y = uy - 1.1
    used_y.append(label_y)
    ax1.annotate(f"{drop:+.1f}pp", xy=(x, y),
                 xytext=(x + 0.8, label_y),
                 fontsize=8.5, color=col, va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", color=col, lw=0.6) if abs(label_y-y)>0.3 else None)

# Clean line
ax1.axvline(0, color="#454C5C", linewidth=1.0, linestyle=":")
ax1.text(0.8, 83.3, "Clean", fontsize=9, color="#8A93A8")

ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.4,
           title="Gate signal", title_fontsize=10)

# ── Right panel: Texas heterophily bars ──────────────────────────────────
ax2.set_facecolor("#1A1E28")
ax2.set_title("(b)  Texas — Heterophily  (h=0.31)", fontsize=12,
              fontweight="bold", color="white", pad=8, loc="left")
ax2.set_ylabel("Test accuracy  (%)", fontsize=11, labelpad=6)
ax2.set_ylim(48, 97)
ax2.set_xlim(-0.6, 4.6)

gates  = list(GATE_STYLE.keys())
tex_ac = [texas["results"][g] * 100 for g in gates]
colors = [GATE_STYLE[g][1] for g in gates]
xs     = np.arange(len(gates))

bars = ax2.bar(xs, tex_ac, width=0.58, color=colors, alpha=0.85,
               edgecolor="#12141A", linewidth=0.8, zorder=3)

for i, (acc, col) in enumerate(zip(tex_ac, colors)):
    ax2.text(i, acc + 0.6, f"{acc:.1f}", ha="center", va="bottom",
             fontsize=9.5, color=col, fontweight="bold")

# Literature reference lines
refs = [
    (84.86, "#2ECC71", "--", "H2GCN (NeurIPS'20)  84.9%"),
    (92.92, "#FFC107", "-.", "GPR-GNN (ICLR'21)  92.9%"),
]
for val, col, ls, lbl in refs:
    ax2.axhline(val, color=col, linestyle=ls, linewidth=1.3, alpha=0.75, zorder=2)
    ax2.text(4.55, val + 0.5, lbl, fontsize=8, color=col, va="bottom", ha="right")

short = ["SAGE\n(no gate)", "A  Cosine\n(GNNGuard)", "B  Conf.\n(Mowst)",
         "C  Learned\n(ACM)", "D  LLM\n(Ours*)"]
ax2.set_xticks(xs)
ax2.set_xticklabels(short, fontsize=9)
for tick, gate in zip(ax2.get_xticklabels(), gates):
    tick.set_color(GATE_STYLE[gate][1])

# ── Figure title + footnote ───────────────────────────────────────────────
fig.suptitle("LLM-Guided Semantic Trust Gate — Experimental Results",
             fontsize=13, fontweight="bold", color="white", y=0.99)

fig.text(0.5, 0.01,
         "* Gate D uses raw LLM text-embedding cosine similarity. "
         "Full label-support framing (proposed method) is expected to outperform both A and B/C.  "
         "Cora: 3 seeds, 200 epochs.  Texas: 3 of 10 official splits, 200 epochs.  "
         "Literature refs (H2GCN, GPR-GNN) use different training protocols — not directly comparable.",
         ha="center", fontsize=8, color="#8A93A8", style="italic")

out = "/home/lam23005/STEM-GNN/research_figure.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
