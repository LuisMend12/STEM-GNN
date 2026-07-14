"""
Backbone comparison figure: GCN vs GraphSAGE backbone, with LLM gate signal.
Reads from results/backbone_results.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("/home/lam23005/STEM-GNN/results/backbone_results.json") as f:
    data = json.load(f)

cora   = data["cora_backbone"]
webkb  = data["webkb_backbone"]

ratios = [r * 100 for r in cora["ratios"]]
res    = cora["results"]

# Color: blue family = GCN, orange family = GraphSAGE
# Line style: solid = LLM gate, dashed = no gate, dotted = others
STYLE = {
    "sage+none":        dict(color="#888888", ls="--",  lw=1.8,  label="GraphSAGE (no gate)"),
    "sage+llm_cosine":  dict(color="#D97706", ls="-",   lw=2.4,  label="GraphSAGE + LLM gate"),
    "sage+confidence":  dict(color="#F59E0B", ls="-.",  lw=1.8,  label="GraphSAGE + Confidence gate"),
    "sage+learned":     dict(color="#FDE68A", ls=":",   lw=1.8,  label="GraphSAGE + Learned gate"),
    "gcn+none":         dict(color="#64748B", ls="--",  lw=1.8,  label="GCN (no gate)"),
    "gcn+llm_cosine":   dict(color="#2563EB", ls="-",   lw=2.4,  label="GCN + LLM gate  ★"),
    "gcn+confidence":   dict(color="#60A5FA", ls="-.",  lw=1.8,  label="GCN + Confidence gate"),
    "gcn+learned":      dict(color="#BFDBFE", ls=":",   lw=1.8,  label="GCN + Learned gate"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.13, wspace=0.34)

# ── Panel 1: Injection sweep on Cora — focus on 4 key lines ──────────────────
keys_focus = ["sage+none", "gcn+none", "sage+llm_cosine", "gcn+llm_cosine"]

for k in keys_focus:
    s = STYLE[k]
    ax1.plot(ratios, [v*100 for v in res[k]],
             color=s["color"], ls=s["ls"], lw=s["lw"],
             marker="o", markersize=5, label=s["label"], zorder=4)

# Annotate the collapse and recovery at 50%
gcn_no  = res["gcn+none"][-1] * 100
gcn_llm = res["gcn+llm_cosine"][-1] * 100
drop_no  = (res["gcn+none"][-1] - res["gcn+none"][0]) * 100
drop_llm = (res["gcn+llm_cosine"][-1] - res["gcn+llm_cosine"][0]) * 100
ax1.annotate(f"GCN alone:\n{drop_no:+.1f}pp", xy=(50, gcn_no),
             xytext=(43, gcn_no - 6),
             fontsize=9, color=STYLE["gcn+none"]["color"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=STYLE["gcn+none"]["color"], lw=1.2))
ax1.annotate(f"GCN + LLM gate:\n{drop_llm:+.1f}pp", xy=(50, gcn_llm),
             xytext=(37, gcn_llm + 3),
             fontsize=9, color=STYLE["gcn+llm_cosine"]["color"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=STYLE["gcn+llm_cosine"]["color"], lw=1.2))

ax1.set_xlabel("Injected confusing edges (%)", fontsize=11)
ax1.set_ylabel("Test accuracy (%)", fontsize=11)
ax1.set_title("(a)  Injection robustness — Cora\nLLM gate rescues GCN from catastrophic collapse",
              fontsize=11, fontweight="bold", pad=8, loc="left")
ax1.set_xticks(ratios)
ax1.set_ylim(42, 85)
ax1.grid(axis="y", alpha=0.2, ls="--", color="#CCCCCC")
ax1.legend(fontsize=9, framealpha=0, loc="upper right")

# ── Panel 2: Heterophily — show only 4 key variants ──────────────────────────
ds_names = ["Texas", "Wisconsin", "Cornell"]
hom = {n: webkb[n]["homophily"] for n in ds_names}
keys_bars = ["sage+none", "gcn+none", "gcn+llm_cosine", "gcn+confidence"]
STYLE["gcn+confidence"]["label"] = "GCN + Confidence gate"

n_ds = len(ds_names); n_k = len(keys_bars)
w = 0.16; gap = 1.2
xs_g = np.arange(n_ds) * gap

for ki, k in enumerate(keys_bars):
    offset = (ki - n_k/2 + 0.5) * (w + 0.02)
    accs = [webkb[dn]["results"].get(k, 0) * 100 for dn in ds_names]
    s = STYLE[k]
    ax2.bar(xs_g + offset, accs, width=w, color=s["color"],
            alpha=0.88, label=s["label"], zorder=3)
    for xi, acc in zip(xs_g + offset, accs):
        ax2.text(xi, acc + 0.8, f"{acc:.0f}", ha="center", va="bottom",
                 fontsize=7.5, color=s["color"])

ax2.set_xticks(xs_g)
ax2.set_xticklabels([f"{n}\nh={hom[n]:.2f}" for n in ds_names], fontsize=10.5)
ax2.set_ylabel("Test accuracy (%)", fontsize=11)
ax2.set_title("(b)  Heterophily — Texas / Wisconsin / Cornell\nLLM gate lifts GCN; confidence gate matches SAGE",
              fontsize=11, fontweight="bold", pad=8, loc="left")
ax2.set_ylim(35, 100)
ax2.grid(axis="y", alpha=0.2, ls="--", color="#CCCCCC")
ax2.legend(fontsize=9, framealpha=0, loc="upper right")

fig.suptitle(
    "GCN backbone + LLM gate signal (decoupled)  —  LLM as semantic oracle, not feature encoder",
    fontsize=13, fontweight="bold", y=0.97)

out = "/home/lam23005/STEM-GNN/figures/backbone_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
