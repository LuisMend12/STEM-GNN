"""
Clean comparison figure: our best LLM gates vs. published baselines.
Two panels (Cora, PubMed) + summary bar.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("/home/lam23005/STEM-GNN/results/llm_gates_results.json") as f:
    data = json.load(f)

# ── Methods to show ───────────────────────────────────────────────────────────
# Baselines (published)
BASELINES = {
    "bl_mlp":     ("MLP (no graph)",               "#78716C", "--", "^", 7),
    "bl_gcn":     ("GCN (Kipf 2017)",              "#9CA3AF", "-",  "s", 6),
    "bl_gat":     ("GAT (Veličković 2018)",         "#6B7280", "-",  "D", 6),
    "bl_appnp":   ("APPNP (Klicpera 2019)",        "#374151", "-",  "v", 6),
    "robust_gcn": ("RobustGCN (Zhu 2019) ▲",       "#DC2626", "-",  "P", 7),
    "gnn_guard":  ("GNNGuard (Zhang 2020) ▲",      "#B45309", "-",  "h", 7),
}
# Ours — existing best gates
OURS = {
    "none":         ("GraphSAGE + BOW (no gate)",        "#94A3B8", "--", "o", 5),
    "variance":     ("Ours: D — LLM Variance ★",         "#10B981", "-",  "o", 8),
    "mlp":          ("Ours: C — LLM MLP ★",              "#7C3AED", "-",  "o", 7),
    # New signals
    "attn_entropy": ("Ours: O — Attn Entropy",           "#F59E0B", "-",  "D", 7),
    "energy_dist":  ("Ours: P — Energy Distance",        "#EF4444", "-",  "D", 7),
    "spectral":     ("Ours: Q — Spectral Coherence",     "#8B5CF6", "-",  "D", 7),
    # New fusion (novel — subgraph robustness focus)
    "llm_gat":      ("Ours: R — LLM-GAT (per-edge) ◆",  "#0EA5E9", "-",  "*", 9),
    "llm_appnp":    ("Ours: S — LLM-APPNP ◆",           "#F97316", "-",  "*", 9),
    "moe_fusion":   ("Ours: T — MoE Fusion ◆",           "#22C55E", "-",  "*", 9),
    "llm_trimmed":  ("Ours: U — LLM Trimmed ◆",          "#06B6D4", "-",  "*", 9),
    "llm_consensus":("Ours: V — LLM Consensus ◆",        "#A855F7", "-",  "*", 9),
    "llm_multiscale":("Ours: W — LLM Multi-Scale ◆",     "#F43F5E", "-",  "*", 9),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig = plt.figure(figsize=(18, 6))
gs  = fig.add_gridspec(1, 3, wspace=0.35, left=0.05, right=0.99,
                        top=0.82, bottom=0.14)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

RATIOS_PCT = [0, 10, 20, 30, 50]

for ax, ds_key, title in [
    (ax1, "cora_sweep",   "(a) Cora"),
    (ax2, "pubmed_sweep", "(b) PubMed"),
]:
    res = data[ds_key]["results"]

    # Draw baselines first (behind)
    for key, (label, color, ls, mk, ms) in BASELINES.items():
        if key not in res: continue
        accs = [v*100 for v in res[key]]
        ax.plot(RATIOS_PCT, accs, color=color, lw=1.8, ls=ls,
                marker=mk, markersize=ms, alpha=0.85,
                label=label, zorder=2)

    # Draw ours on top
    for key, (label, color, ls, mk, ms) in OURS.items():
        if key not in res: continue
        accs = [v*100 for v in res[key]]
        lw   = 3.0 if key != "none" else 1.6
        ax.plot(RATIOS_PCT, accs, color=color, lw=lw, ls=ls,
                marker=mk, markersize=ms, alpha=1.0,
                label=label, zorder=4 if key != "none" else 3)

    ax.set_xlabel("Injected confusing edges (%)", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8, loc="left")
    ax.set_xticks(RATIOS_PCT)
    ax.grid(axis="y", alpha=0.25, ls="--", color="#CCCCCC")
    ax.grid(axis="x", alpha=0.1,  ls=":",  color="#CCCCCC")

    # Shade the injection region
    ax.axvspan(10, 50, alpha=0.04, color="#DC2626", zorder=0)
    ax.text(30, ax.get_ylim()[0] + 0.5, "← attack zone →",
            ha="center", va="bottom", fontsize=9, color="#DC2626", alpha=0.6)

    ax.legend(fontsize=9.5, framealpha=0.95, loc="lower left",
              edgecolor="#E2E8F0", fancybox=True)

# ── Summary bar: accuracy drop clean → 50% injection ─────────────────────────
SHOW = list(BASELINES.keys()) + list(OURS.keys())
SHOW_LABEL = {**{k: v[0] for k,v in BASELINES.items()},
              **{k: v[0] for k,v in OURS.items()}}
SHOW_COLOR = {**{k: v[1] for k,v in BASELINES.items()},
              **{k: v[1] for k,v in OURS.items()}}

drops = {}
for g in SHOW:
    vals = []
    for ds in ["cora_sweep","pubmed_sweep"]:
        r = data[ds]["results"]
        if g in r:
            vals.append((r[g][-1] - r[g][0]) * 100)
    if vals:
        drops[g] = np.mean(vals)

sorted_g = sorted(drops, key=lambda g: drops[g], reverse=True)
ys = np.arange(len(sorted_g))
NEW_FUSION = {"llm_gat", "llm_appnp", "moe_fusion"}
is_ours = {k for k in OURS if k != "none"}

for yi, g in enumerate(sorted_g):
    v   = drops[g]
    col = SHOW_COLOR[g]
    alp = 1.0 if g in is_ours else 0.65
    ax3.barh(yi, v, color=col, alpha=alp, height=0.65,
             edgecolor="white" if g in is_ours else "none", linewidth=0)
    ha  = "right" if v < 0 else "left"
    off = -0.2 if v < 0 else 0.2
    tag = " ★ OURS" if g in is_ours else ""
    ax3.text(v + off, yi, f"{v:+.1f}pp{tag}", va="center", ha=ha,
             fontsize=9, color=col,
             fontweight="bold" if g in is_ours else "normal")

ax3.set_yticks(ys)
ax3.set_yticklabels([SHOW_LABEL[g] for g in sorted_g], fontsize=9.5)
for tick, g in zip(ax3.get_yticklabels(), sorted_g):
    tick.set_color(SHOW_COLOR[g])
    if g in is_ours:
        tick.set_fontweight("bold")

ax3.axvline(0, color="#888", lw=1.0)
ax3.set_xlabel("Mean accuracy drop  clean → 50% injection (pp)", fontsize=11)
ax3.set_title("(c) Robustness summary\n(higher = less drop = more robust)",
              fontsize=13, fontweight="bold", pad=8, loc="left")
ax3.grid(axis="x", alpha=0.2, ls="--", color="#CCCCCC")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Divider line between baselines and ours in bar chart
n_bl = sum(1 for g in sorted_g if g in BASELINES)
boundary = next(i for i, g in enumerate(sorted_g) if g in is_ours or g == "none")
ax3.axhline(boundary - 0.5, color="#CBD5E1", lw=1.2, ls="--")
ax3.text(ax3.get_xlim()[0], boundary - 0.5,
         " ── published baselines below ──",
         va="center", ha="left", fontsize=8, color="#94A3B8")

fig.suptitle(
    "LLM-Guided Semantic Trust Gate  vs.  Published Baselines (▲ = robustness baselines)\n"
    "BOW backbone (GraphSAGE) + Frozen DistilBERT · ◆ = novel subgraph fusion · Cora & PubMed",
    fontsize=13, fontweight="bold", y=0.98
)

out = "/home/lam23005/STEM-GNN/figures/comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
