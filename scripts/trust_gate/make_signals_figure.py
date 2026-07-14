"""
Signal gallery: one panel per LLM gate signal (A-I).
Each panel shows the formula, Cora curve, PubMed curve vs. no-gate baseline.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

with open("/home/lam23005/STEM-GNN/results/llm_gates_results.json") as f:
    data = json.load(f)

# ── Signal definitions ────────────────────────────────────────────────────────
SIGNALS = [
    ("cosine",    "A",  "LLM Cosine\nAlignment",
     r"$\beta_v = 1 - \sigma(\cos(e_v,\,\mu_{nbr}))$",
     "How similar is v to mean neighbour?\nHigh similarity → trust neighbours.",
     False),
    ("entropy",   "B",  "LLM Neighbour\nEntropy",
     r"$\beta_v = H(\mathrm{softmax}(e_v \cdot e_u))\ /\ \log d_v$",
     "Confusion over neighbour distribution.\nHigh entropy → distrust neighbours.",
     False),
    ("mlp",       "C",  "LLM MLP Gate\n(learned)",
     r"$\beta_v = \sigma(\mathrm{MLP}(e_v \,\|\, \mu_{nbr}))$",
     "Learned gate from concatenated\nnode + mean-neighbour LLM embs.",
     True),
    ("variance",  "D",  "LLM Neighbour\nVariance",
     r"$\beta_v = \sigma\!\left(10\cdot\overline{\|e_u - \mu_{nbr}\|^2}\right)$",
     "Spread of neighbour embeddings.\nHigh variance → mixed, risky neighbourhood.",
     True),
    ("max_cosine","E",  "LLM Max\nCosine",
     r"$\beta_v = 1 - \max_u \cos(e_v, e_u)$",
     "Similarity to single closest neighbour.\nHigh max sim → trust that neighbour.",
     False),
    ("nbr_agree", "F",  "Neighbour\nAgreement",
     r"$\beta_v = 1 - \|\mu_{nbr}\|^2$",
     "Do neighbours agree with each other?\nLow norm → spread out, disagreeing.",
     False),
    ("bilinear",  "G",  "LLM Bilinear\n(learned)",
     r"$\beta_v = \sigma(e_v^\top U^\top V\, \mu_{nbr})$",
     "Low-rank learned interaction between\nnode and mean-neighbour embeddings.",
     True),
    ("topk",      "H",  "LLM Top-k\nFiltered",
     r"$\beta_v = 1 - \sigma\!\left(\mathrm{mean}_{k\mathrm{-nn}} \cos(e_v,e_u)\right)$",
     "Average cosine to top-3 most similar\nneighbours only (filter noisy ones).",
     False),
    ("cos_std",   "I",  "Cosine Std\n(uncertainty)",
     r"$\beta_v = \sigma\!\left(10 \cdot \mathrm{std}\{\cos(e_v,e_u)\}\right)$",
     "Spread of per-edge cosine scores.\nHigh std = uncertain neighbourhood.",
     True),
]

RATIOS = [0, 10, 20, 30, 50]
C_ROBUST   = "#10B981"   # green
C_WEAK     = "#F59E0B"   # amber
C_HURT     = "#EF4444"   # red
C_BASELINE = "#9CA3AF"   # gray

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, axes = plt.subplots(3, 3, figsize=(17, 13))
fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.06,
                    hspace=0.55, wspace=0.32)

base_cora   = [v*100 for v in data["cora_sweep"]["results"]["none"]]
base_pubmed = [v*100 for v in data["pubmed_sweep"]["results"]["none"]]

for idx, (key, letter, name, formula, desc, robust) in enumerate(SIGNALS):
    ax  = axes[idx // 3][idx % 3]
    res_c = data["cora_sweep"]["results"].get(key)
    res_p = data["pubmed_sweep"]["results"].get(key)

    # Compute drop for color coding
    if res_c and res_p:
        drop_c = (res_c[-1] - res_c[0]) * 100
        drop_p = (res_p[-1] - res_p[0]) * 100
        avg_drop = (drop_c + drop_p) / 2
        base_drop = ((base_cora[-1]-base_cora[0]) + (base_pubmed[-1]-base_pubmed[0])) / 2
        better_than_base = avg_drop > base_drop   # less drop = more positive
    else:
        avg_drop = 0; better_than_base = False

    card_color = C_ROBUST if robust else (C_WEAK if better_than_base else C_HURT)

    # Background card
    bg = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                         boxstyle="round,pad=0.02",
                         facecolor="#F9FAFB" if robust else "#FEFCE8" if better_than_base else "#FFF5F5",
                         edgecolor=card_color, linewidth=2.0, clip_on=False, zorder=0)
    ax.add_patch(bg)

    # Plot baseline
    ax.plot(RATIOS, base_cora,   color=C_BASELINE, lw=1.2, ls="--", alpha=0.6,
            marker="o", markersize=3, label="Baseline (no gate)", zorder=2)
    ax.plot(RATIOS, base_pubmed, color=C_BASELINE, lw=1.2, ls=":",  alpha=0.6,
            marker="o", markersize=3, zorder=2)

    # Plot this signal
    if res_c:
        accs_c = [v*100 for v in res_c]
        ax.plot(RATIOS, accs_c, color=card_color, lw=2.2, ls="-",
                marker="o", markersize=5, label="Cora", zorder=4)
    if res_p:
        accs_p = [v*100 for v in res_p]
        ax.plot(RATIOS, accs_p, color=card_color, lw=2.2, ls="--",
                marker="s", markersize=5, label="PubMed", zorder=4, alpha=0.75)

    ax.set_xticks(RATIOS)
    ax.set_xticklabels([f"{r}%" for r in RATIOS], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", alpha=0.2, ls="--", color="#CCCCCC")
    ax.set_xlabel("Injection ratio", fontsize=8, labelpad=2)
    ax.set_ylabel("Acc (%)", fontsize=8, labelpad=2)

    # Title block
    star = "  ★ ROBUST" if robust else ""
    title_col = card_color
    ax.set_title(f"Signal {letter} — {name.replace(chr(10),' ')}{star}",
                 fontsize=9.5, fontweight="bold", color=title_col, pad=4)

    # Formula annotation below axes
    ax.text(0.5, -0.30, formula, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5,
            color="#1E293B", style="normal",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E2E8F0", lw=0.8))

    # Drop annotation top-right
    if res_c and res_p:
        drop_str = f"Δ {avg_drop:+.1f}pp avg"
        ax.text(0.97, 0.97, drop_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=card_color, fontweight="bold")

# Legend on last panel area
handles = [
    plt.Line2D([0],[0], color=C_BASELINE, lw=1.5, ls="--", label="Baseline Cora (no gate)"),
    plt.Line2D([0],[0], color=C_BASELINE, lw=1.5, ls=":",  label="Baseline PubMed (no gate)"),
    plt.Line2D([0],[0], color=C_ROBUST,  lw=2.5, ls="-",   marker="o", ms=5, label="Signal — Cora"),
    plt.Line2D([0],[0], color=C_ROBUST,  lw=2.5, ls="--",  marker="s", ms=5, label="Signal — PubMed"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
           framealpha=0.9, edgecolor="#E2E8F0", bbox_to_anchor=(0.5, 0.005))

# Color legend
for col, lbl in [(C_ROBUST,"★ Robust gate"), (C_WEAK,"Marginal gain"), (C_HURT,"Hurts vs baseline")]:
    pass  # embedded in titles

fig.suptitle(
    "9 LLM Gate Signals  ·  Formula + Robustness Under Edge Injection Attack\n"
    "Green border = robust  ·  Solid line = Cora  ·  Dashed = PubMed  ·  Gray = no-gate baseline",
    fontsize=12, fontweight="bold", y=0.975
)

out = "/home/lam23005/STEM-GNN/figures/signals_gallery.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
