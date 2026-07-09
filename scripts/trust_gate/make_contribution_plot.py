"""
4-panel contribution figure:
  (a) Gap table
  (b) Injection sweep — Cora + PubMed lines
  (c) Accuracy drop at 50% injection
  (d) Heterophily — Texas / Wisconsin / Cornell grouped bars
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("/home/lam23005/STEM-GNN/results/sweep_results.json") as f:
    data = json.load(f)

ratios  = [r * 100 for r in data["cora_sweep"]["ratios"]]
cora    = data["cora_sweep"]["results"]
pubmed  = data["pubmed_sweep"]["results"]
texas   = data["heterophily"]["Texas"]["results"]
wisc    = data["heterophily"]["Wisconsin"]["results"]
cornell = data["heterophily"]["Cornell"]["results"]

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

fig = plt.figure(figsize=(22, 5.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.1, 0.95, 1.3],
                      left=0.02, right=0.99, top=0.88, bottom=0.13, wspace=0.34)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

# ── Panel (a): Contribution gap table ────────────────────────────────────────
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
ax1.set_title("(a)  Research Gap", fontsize=12, fontweight="bold", pad=8, loc="left")

methods = [
    ("GraphSAGE",          False, False, False),
    ("GNNGuard (N'20)",    False, True,  False),
    ("H2GCN (N'20)",       False, False, True),
    ("GPR-GNN (I'21)",     False, False, True),
    ("TAPE (I'24)",        True,  False, False),
    ("STEM-GNN (K'26)",    True,  False, False),
    ("Ours (Trust Gate)",  True,  True,  True),
]
cols   = ["Method", "LLM\nfeat.", "Inject.\nrobust", "Hetero-\nphily"]
col_xs = [0.02, 0.50, 0.67, 0.84]
header_y = 0.93; row_h = 0.108

for cx, ch in zip(col_xs, cols):
    ax1.text(cx, header_y, ch, fontsize=8.5, fontweight="bold",
             color="#333333", va="top", ha="left")
ax1.plot([0.0, 1.0], [header_y - 0.045, header_y - 0.045],
         color="#CCCCCC", lw=1.0, transform=ax1.transAxes)

for i, (name, llm, inj, het) in enumerate(methods):
    y = header_y - 0.065 - i * row_h
    is_ours = i == len(methods) - 1
    if is_ours:
        rect = mpatches.FancyBboxPatch((0.0, y - 0.045), 1.0, row_h - 0.008,
                                        boxstyle="round,pad=0.005",
                                        linewidth=1.2, edgecolor="#4C9BE8",
                                        facecolor="#EEF4FF",
                                        transform=ax1.transAxes, zorder=0)
        ax1.add_patch(rect)
    ax1.text(col_xs[0], y, name, fontsize=8.5, va="center", ha="left",
             color="#1A56DB" if is_ours else "#333333",
             fontweight="bold" if is_ours else "normal")
    for cx, val in zip(col_xs[1:], [llm, inj, het]):
        sym = "✓" if val else "✗"
        col = "#16A34A" if val else "#DC2626"
        ax1.text(cx + 0.07, y, sym, fontsize=10, color=col,
                 fontweight="bold" if (is_ours and val) else "normal",
                 va="center", ha="center")

ax1.text(0.5, 0.005, "No prior method handles all three simultaneously",
         fontsize=7.5, color="#888888", style="italic",
         ha="center", va="bottom", transform=ax1.transAxes)

# ── Panel (b): Injection sweep — Cora + PubMed ───────────────────────────────
for gate in GATES:
    c_accs = [v * 100 for v in cora[gate]]
    p_accs = [v * 100 for v in pubmed[gate]]
    lw = 2.2 if gate != "none" else 1.6
    ls = "--" if gate == "none" else "-"
    ax2.plot(ratios, c_accs, color=GATE_COLOR[gate], lw=lw, ls=ls,
             marker="o", markersize=4, label=GATE_LABEL[gate], zorder=3)
    ax2.plot(ratios, p_accs, color=GATE_COLOR[gate], lw=lw, ls=ls,
             marker="s", markersize=4, alpha=0.4, zorder=3)

ax2.set_xlabel("Injected confusing edges (%)", fontsize=11)
ax2.set_ylabel("Test accuracy (%)", fontsize=11)
ax2.set_title("(b)  Injection robustness\nCora (●)  PubMed (■, faded)",
              fontsize=11, fontweight="bold", pad=8, loc="left")
ax2.set_xticks(ratios); ax2.set_ylim(60, 84)
ax2.grid(axis="y", alpha=0.2, linestyle="--", color="#CCCCCC")
ax2.legend(fontsize=8.2, framealpha=0, loc="lower left")

# ── Panel (c): Accuracy drop at 50% injection (Cora + PubMed) ────────────────
drops_cora   = [(cora[g][-1]   - cora[g][0])   * 100 for g in GATES]
drops_pubmed = [(pubmed[g][-1] - pubmed[g][0]) * 100 for g in GATES]
xs = np.arange(len(GATES)); w = 0.36

bars1 = ax3.barh(xs + w/2, drops_cora,   height=w,
                 color=[GATE_COLOR[g] for g in GATES], alpha=0.85, label="Cora")
bars2 = ax3.barh(xs - w/2, drops_pubmed, height=w,
                 color=[GATE_COLOR[g] for g in GATES], alpha=0.38,
                 hatch="///", edgecolor="white", label="PubMed")

for i, (dc, dp) in enumerate(zip(drops_cora, drops_pubmed)):
    ax3.text(dc - 0.1, xs[i] + w/2, f"{dc:+.1f}", va="center", ha="right",
             fontsize=8.5, color=GATE_COLOR[GATES[i]], fontweight="bold")
    ax3.text(dp - 0.1, xs[i] - w/2, f"{dp:+.1f}", va="center", ha="right",
             fontsize=8.5, color=GATE_COLOR[GATES[i]])

ax3.set_yticks(xs)
ax3.set_yticklabels([GATE_LABEL[g] for g in GATES], fontsize=9)
for tick, g in zip(ax3.get_yticklabels(), GATES):
    tick.set_color(GATE_COLOR[g])
ax3.set_xlabel("Accuracy drop (pp)", fontsize=11)
ax3.set_title("(c)  Drop: clean → 50% injection",
              fontsize=11, fontweight="bold", pad=8, loc="left")
ax3.axvline(0, color="#AAAAAA", lw=0.8)
ax3.grid(axis="x", alpha=0.2, linestyle="--", color="#CCCCCC")
ax3.legend(fontsize=9, framealpha=0, loc="lower right")

# ── Panel (d): Heterophily grouped bars ───────────────────────────────────────
hetero_datasets = [
    ("Texas\nh=0.31",     texas),
    ("Wisconsin\nh=0.37", wisc),
    ("Cornell\nh=0.34",   cornell),
]
n_ds = len(hetero_datasets)
w = 0.13; group_gap = 0.85
xs_g = np.arange(n_ds) * group_gap

for gi, gate in enumerate(GATES):
    offsets = (gi - n_ds + 0.5) * w * 1.15
    accs = [ds[gate] * 100 for _, ds in hetero_datasets]
    ax4.bar(xs_g + offsets, accs, width=w,
            color=GATE_COLOR[gate], label=GATE_LABEL[gate], alpha=0.85)

# Dashed baseline line per dataset group
for i, (_, ds_res) in enumerate(hetero_datasets):
    x_left  = xs_g[i] - 0.38
    x_right = xs_g[i] + 0.38
    ax4.plot([x_left, x_right],
             [ds_res["none"] * 100, ds_res["none"] * 100],
             color="#888888", lw=1.5, ls="--", alpha=0.7, zorder=5)

ax4.set_xticks(xs_g)
ax4.set_xticklabels([lbl for lbl, _ in hetero_datasets], fontsize=10.5)
ax4.set_ylabel("Test accuracy (%)", fontsize=11)
ax4.set_title("(d)  Heterophily generalization\n(dashed = no-gate baseline)",
              fontsize=11, fontweight="bold", pad=8, loc="left")
ax4.set_ylim(35, 105)
ax4.grid(axis="y", alpha=0.2, linestyle="--", color="#CCCCCC")
ax4.legend(fontsize=8.2, framealpha=0, loc="upper right")

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle("LLM-Guided Semantic Trust Gate  —  Contribution & Empirical Results  "
             "(Cora, PubMed, Texas, Wisconsin, Cornell)",
             fontsize=13, fontweight="bold", y=1.01)

out = "/home/lam23005/STEM-GNN/figures/contribution_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
