"""Results figure for all LLM gate signals + novel paradigms."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/home/lam23005/STEM-GNN/results/llm_gates_results.json") as f:
    data = json.load(f)

# Original 9 gates
GATES_ORIG = ["none","cosine","entropy","mlp","variance",
              "max_cosine","nbr_agree","bilinear","topk","cos_std"]
# 5 novel paradigms
GATES_NOVEL = ["cross_attn","proto_gate","rewired","curriculum","verifier"]
# 4 published baselines (stored with bl_ prefix)
GATES_BL = ["bl_mlp","bl_gcn","bl_gat","bl_appnp"]
GATES = GATES_ORIG + GATES_NOVEL + GATES_BL

LABEL = {
    "none":       "No gate (GraphSAGE+BOW)",
    "cosine":     "A — LLM cosine",
    "entropy":    "B — LLM entropy",
    "mlp":        "C — LLM MLP",
    "variance":   "D — LLM variance",
    "max_cosine": "E — LLM max cosine",
    "nbr_agree":  "F — neighbour agree",
    "bilinear":   "G — LLM bilinear",
    "topk":       "H — LLM top-k",
    "cos_std":    "I — cosine std",
    # Novel paradigms
    "cross_attn": "J — LLM cross-attn (Judge)",
    "proto_gate": "K — LLM proto gate",
    "rewired":    "L — LLM graph rewiring",
    "curriculum": "M — LLM curriculum",
    "verifier":   "N — LLM verifier blend",
    # Published baselines
    "bl_mlp":   "MLP (no graph)",
    "bl_gcn":   "GCN (Kipf 2017)",
    "bl_gat":   "GAT (Veličković 2018)",
    "bl_appnp": "APPNP (Klicpera 2019)",
}
COLOR = {
    "none":"#888888","cosine":"#4C9BE8","entropy":"#E86B4C",
    "mlp":"#7C4CE8","variance":"#4CC8A0","max_cosine":"#F59E0B",
    "nbr_agree":"#10B981","bilinear":"#EC4899","topk":"#6366F1","cos_std":"#14B8A6",
    # Novel
    "cross_attn": "#EF4444",
    "proto_gate": "#A855F7",
    "rewired":    "#0EA5E9",
    "curriculum": "#F97316",
    "verifier":   "#22C55E",
    # Published baselines — warm grays/browns to stand apart
    "bl_mlp":   "#78716C",
    "bl_gcn":   "#57534E",
    "bl_gat":   "#292524",
    "bl_appnp": "#A8A29E",
}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.facecolor":"white","axes.facecolor":"white"})

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.13, wspace=0.32)
ax1, ax2, ax3 = axes

ROBUST    = {"mlp","variance","bilinear","cos_std"}
NOVEL     = set(GATES_NOVEL)
BASELINES = set(GATES_BL)

for ds_key, ax, title in [
    ("cora_sweep",   ax1, "(a) Cora  —  BOW backbone + LLM gate"),
    ("pubmed_sweep", ax2, "(b) PubMed  —  BOW backbone + LLM gate"),
]:
    ratios = [r*100 for r in data[ds_key]["ratios"]]
    res    = data[ds_key]["results"]
    for g in GATES:
        if g not in res: continue
        accs = [v*100 for v in res[g]]
        is_bl   = g in BASELINES
        highlight = g in ROBUST or g in NOVEL or g in BASELINES or g == "none"
        lw  = 2.4 if (g in ROBUST or g in NOVEL) else (2.0 if is_bl else (1.6 if g=="none" else 1.3))
        ls  = "--" if (g == "none" or is_bl) else (":" if g in NOVEL else "-")
        alp = 1.0 if highlight else 0.3
        mk  = "s" if g in NOVEL else ("^" if is_bl else "o")
        ax.plot(ratios, accs, color=COLOR[g], lw=lw, ls=ls,
                alpha=alp, marker=mk, markersize=6 if is_bl else (5 if g in NOVEL else 4),
                label=LABEL[g] if highlight else None,
                zorder=5 if is_bl else (4 if highlight else 2))
    ax.set_xlabel("Injected confusing edges (%)", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.set_xticks(ratios)
    ax.grid(axis="y", alpha=0.2, ls="--", color="#CCCCCC")
    ax.legend(fontsize=8.5, framealpha=0, loc="lower left")

# Drop bar chart
all_gates = [g for g in GATES if all(g in data[k]["results"] for k in ["cora_sweep","pubmed_sweep"])]
mean_drops = {}
for g in all_gates:
    drops = []
    for k in ["cora_sweep","pubmed_sweep"]:
        r = data[k]["results"][g]
        drops.append((r[-1]-r[0])*100)
    mean_drops[g] = np.mean(drops)

sorted_gates = sorted(all_gates, key=lambda g: mean_drops[g], reverse=True)
xs = np.arange(len(sorted_gates))
for xi, g in enumerate(sorted_gates):
    v = mean_drops[g]
    ax3.barh(xi, v, color=COLOR[g],
             alpha=1.0 if g in ROBUST else 0.55, height=0.6)
    ha = "right" if v < 0 else "left"
    off = -0.15 if v < 0 else 0.15
    star = " ★" if g in ROBUST else (" ◆" if g in NOVEL else (" ▲" if g in BASELINES else ""))
    ax3.text(v+off, xi, f"{v:+.1f}{star}", va="center", ha=ha,
             fontsize=8.5, color=COLOR[g],
             fontweight="bold" if (g in ROBUST or g in NOVEL or g in BASELINES) else "normal")

ax3.set_yticks(xs)
ax3.set_yticklabels([LABEL[g] for g in sorted_gates], fontsize=9)
for tick, g in zip(ax3.get_yticklabels(), sorted_gates):
    tick.set_color(COLOR[g])
    tick.set_fontweight("bold" if (g in ROBUST or g in NOVEL or g in BASELINES) else "normal")
ax3.axvline(0, color="#AAAAAA", lw=0.8)
ax3.set_xlabel("Mean accuracy drop: clean → 50% injection (pp)", fontsize=11)
ax3.set_title("(c) Robustness ranking  —  Cora + PubMed avg\n(★ robust gate  ◆ novel paradigm  ▲ published baseline)",
              fontsize=11, fontweight="bold", pad=8, loc="left")
ax3.grid(axis="x", alpha=0.2, ls="--", color="#CCCCCC")

fig.suptitle("LLM-Guided Trust Gate  ·  vs. Published Baselines (▲)  ·  BOW backbone + Frozen DistilBERT",
             fontsize=13, fontweight="bold", y=0.98)

out = "/home/lam23005/STEM-GNN/figures/llm_gates_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
