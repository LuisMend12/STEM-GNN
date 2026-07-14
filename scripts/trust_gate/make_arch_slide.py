"""
Architecture diagram sized for 16:9 slide (no title — header is on the slide).
Clean two-lane layout: backbone top, LLM gate bottom.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(16, 7.2), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16); ax.set_ylim(0, 7.2)
ax.axis("off")

# ── Colors ────────────────────────────────────────────────────────────────────
C_BLUE   = "#2563EB"
C_PURPLE = "#7C3AED"
C_GRAY   = "#1E293B"
C_LGRAY  = "#64748B"
C_ORANGE = "#D97706"
C_GREEN  = "#059669"
C_BG     = "#F8FAFF"
C_BGBLUE = "#EBF2FF"
C_BGPURP = "#F5F3FF"
C_BGORANGE = "#FFF7ED"

def box(x, y, w, h, fc, ec, lw=1.8, radius=0.18, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)

def label(x, y, t, fs=12, fw="normal", fc=C_GRAY, ha="center", va="center", wrap=False):
    ax.text(x, y, t, fontsize=fs, fontweight=fw, color=fc,
            ha=ha, va=va, wrap=wrap,
            multialignment="center" if ha=="center" else "left")

def arrow(x1, y1, x2, y2, color=C_GRAY, lw=2.0, label=None, label_above=True):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=16, connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dy = 0.22 if label_above else -0.22
        ax.text(mx, my+dy, label, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

def header_strip(x, y, w, h, color, text):
    box(x, y, w, h, fc=color, ec=color, lw=0, radius=0.12)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

# ── Lane backgrounds ──────────────────────────────────────────────────────────
box(0.15, 3.75, 15.7, 3.1, fc="#EBF2FF", ec=C_BLUE, lw=1.2, radius=0.25)
box(0.15, 0.35, 15.7, 3.1, fc=C_BGPURP, ec=C_PURPLE, lw=1.2, radius=0.25)

label(0.72, 5.28, "BACKBONE\nPATH", fs=9, fw="bold", fc=C_BLUE)
label(0.72, 1.88, "LLM GATE\nPATH", fs=9, fw="bold", fc=C_PURPLE)

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE ROW  y=4.1 to 6.5
# ══════════════════════════════════════════════════════════════════════════════
BW, BH, BY = 2.5, 2.0, 4.15
BXS = [1.2, 4.4, 8.1, 11.4]   # x positions for 4 boxes

# 1. Node Features (BOW)
box(BXS[0], BY, BW, BH, fc="#F1F5F9", ec=C_LGRAY)
label(BXS[0]+BW/2, BY+1.45, "Node Features", fs=13, fw="bold", fc=C_GRAY)
label(BXS[0]+BW/2, BY+1.05, "BOW vectors", fs=11, fc=C_LGRAY)
label(BXS[0]+BW/2, BY+0.65, "x_v", fs=16, fw="bold", fc=C_GRAY)
label(BXS[0]+BW/2, BY+0.28, "Cora: 1433-dim\nPubMed: 500-dim", fs=9, fc=C_LGRAY)

# 2. GNN Backbone (highlighted)
box(BXS[1], BY, BW+0.5, BH, fc=C_BGBLUE, ec=C_BLUE, lw=2.5)
header_strip(BXS[1], BY+BH-0.44, BW+0.5, 0.44, C_BLUE, "ENCODER")
label(BXS[1]+(BW+0.5)/2, BY+1.12, "GNN Backbone", fs=14, fw="bold", fc=C_BLUE)
label(BXS[1]+(BW+0.5)/2, BY+0.72, "GraphSAGE", fs=12, fc=C_LGRAY)
label(BXS[1]+(BW+0.5)/2, BY+0.32, "(or GCN)", fs=10, fc=C_LGRAY)

# 3. Node Repr
box(BXS[2], BY, BW, BH, fc="#F1F5F9", ec=C_LGRAY)
label(BXS[2]+BW/2, BY+1.45, "Node Repr.", fs=13, fw="bold", fc=C_GRAY)
label(BXS[2]+BW/2, BY+0.92, "h_v", fs=22, fw="bold", fc=C_GRAY)
label(BXS[2]+BW/2, BY+0.32, "128-dim", fs=10, fc=C_LGRAY)

# 4. Classifier
box(BXS[3], BY, BW, BH, fc=C_BGORANGE, ec=C_ORANGE, lw=2)
label(BXS[3]+BW/2, BY+1.45, "Classifier", fs=13, fw="bold", fc=C_ORANGE)
label(BXS[3]+BW/2, BY+0.88, "softmax(W h_v)", fs=12, fc=C_ORANGE)
label(BXS[3]+BW/2, BY+0.32, "label prediction", fs=10, fc=C_LGRAY)

# Backbone arrows
mid_y = BY + BH/2
arrow(BXS[0]+BW,      mid_y, BXS[1],          mid_y, C_LGRAY, label="x_v")
arrow(BXS[1]+BW+0.5,  mid_y, BXS[2],          mid_y, C_BLUE,  label="h_v")
arrow(BXS[2]+BW,      mid_y, BXS[3],          mid_y, C_LGRAY)

# ══════════════════════════════════════════════════════════════════════════════
# GATE ROW  y=0.6 to 3.0
# ══════════════════════════════════════════════════════════════════════════════
GW, GH, GY = 2.5, 2.0, 0.68
GXS = [1.2, 4.4, 7.6]

# 1. Raw Text
box(GXS[0], GY, GW, GH, fc=C_BGPURP, ec=C_PURPLE, lw=1.5)
label(GXS[0]+GW/2, GY+1.45, "Raw Node Text", fs=13, fw="bold", fc=C_PURPLE)
label(GXS[0]+GW/2, GY+1.0,  "paper titles &", fs=10, fc=C_LGRAY)
label(GXS[0]+GW/2, GY+0.72, "abstracts", fs=10, fc=C_LGRAY)
label(GXS[0]+GW/2, GY+0.28, '"Neural networks for..."', fs=9, fc=C_LGRAY)

# 2. DistilBERT (frozen)
box(GXS[1], GY, GW+0.5, GH, fc=C_BGPURP, ec=C_PURPLE, lw=2.5)
header_strip(GXS[1], GY+GH-0.44, GW+0.5, 0.44, C_PURPLE, "FROZEN  ·  no gradient")
label(GXS[1]+(GW+0.5)/2, GY+1.1,  "DistilBERT", fs=14, fw="bold", fc=C_PURPLE)
label(GXS[1]+(GW+0.5)/2, GY+0.68, "multi-qa-distilbert-cos-v1", fs=9.5, fc=C_LGRAY)
label(GXS[1]+(GW+0.5)/2, GY+0.28, "e_v  ∈  ℝ⁷⁶⁸", fs=12, fw="bold", fc=C_PURPLE)

# 3. Trust Gate (OUR CONTRIBUTION)
box(GXS[2], GY, GW+0.7, GH, fc="#EDE9FE", ec=C_PURPLE, lw=3.0, zorder=3)
header_strip(GXS[2], GY+GH-0.44, GW+0.7, 0.44, C_PURPLE, "★  OUR CONTRIBUTION")
label(GXS[2]+(GW+0.7)/2, GY+1.08, "Semantic Trust Gate", fs=13, fw="bold", fc=C_PURPLE, zorder=4)
label(GXS[2]+(GW+0.7)/2, GY+0.65, "β_v = f( e_v,  mean{e_u} )", fs=12, fc=C_GRAY, zorder=4)
label(GXS[2]+(GW+0.7)/2, GY+0.28, "β_v  ∈  [0, 1]", fs=11, fc=C_LGRAY, zorder=4)

# Gate arrows
gate_mid_y = GY + GH/2
arrow(GXS[0]+GW,       gate_mid_y, GXS[1],          gate_mid_y, C_PURPLE, label="text")
arrow(GXS[1]+GW+0.5,   gate_mid_y, GXS[2],          gate_mid_y, C_PURPLE, label="e_v")

# β_v vertical arrow (gate → backbone GNN)
gate_cx = GXS[2] + (GW+0.7)/2
back_cx = BXS[1] + (BW+0.5)/2
# Draw a curved arrow going up
ax.annotate("", xy=(back_cx, BY),
            xytext=(gate_cx, GY+GH),
    arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=2.5,
                    mutation_scale=18,
                    connectionstyle="arc3,rad=-0.25"))
ax.text((gate_cx+back_cx)/2 - 0.8, (BY+GY+GH)/2 + 0.1, "β_v",
        fontsize=16, fontweight="bold", color=C_PURPLE, ha="center")

# ══════════════════════════════════════════════════════════════════════════════
# Equation box (right side)
# ══════════════════════════════════════════════════════════════════════════════
EX, EY, EW, EH = 11.3, 0.55, 4.35, 3.2
box(EX, EY, EW, EH, fc=C_BG, ec="#CBD5E1", lw=1.5, radius=0.2)

label(EX+EW/2, EY+EH-0.35, "Gating Equation", fs=12, fw="bold", fc=C_GRAY)
# divider
ax.plot([EX+0.2, EX+EW-0.2], [EY+EH-0.6, EY+EH-0.6], color="#CBD5E1", lw=1)

label(EX+EW/2, EY+EH-0.95, "h_v  =  β_v · W_self(x_v)", fs=12, fw="bold", fc=C_GRAY)
label(EX+EW/2, EY+EH-1.35, "+  (1−β_v) · W_neigh(mean x_u)", fs=12, fw="bold", fc=C_GRAY)

ax.plot([EX+0.2, EX+EW-0.2], [EY+EH-1.62, EY+EH-1.62], color="#CBD5E1", lw=0.8, ls="--")

for i, (bv, desc, c) in enumerate([
    ("β_v → 1", "trust self   (attack / heterophily)",  "#DC2626"),
    ("β_v → 0", "trust nbrs  (safe / homophilic)",      "#059669"),
]):
    yi = EY+EH-2.05 + i*0.52
    p = FancyBboxPatch((EX+0.18, yi-0.14), 1.1, 0.34,
        boxstyle="round,pad=0.04", facecolor=c, edgecolor="none", zorder=3)
    ax.add_patch(p)
    ax.text(EX+0.18+0.55, yi+0.03, bv, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=4)
    ax.text(EX+1.45, yi+0.03, desc, ha="left", va="center",
            fontsize=10, color=C_LGRAY)

ax.text(EX+0.18, EY+0.92, "Robust LLM gate signals:", fontsize=10, fontweight="bold",
        color=C_GRAY)
for i, g in enumerate(["★ C — LLM MLP  (learned)", "★ D — LLM variance",
                        "★ G — LLM bilinear", "★ I — cosine std"]):
    ax.text(EX+0.25, EY+0.68-i*0.28, g, fontsize=9.5, color=C_PURPLE)

plt.savefig("/home/lam23005/STEM-GNN/figures/arch_slide.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved → figures/arch_slide.png")
