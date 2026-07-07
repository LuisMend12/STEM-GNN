"""
Architecture diagram: Trust Gate as a complement to STEM-GNN.
Shows where the new component plugs into the existing backbone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG    = "#12141A"; CARD  = "#1A1E28"; DARK2 = "#0E1118"
BLUE  = "#459BFF"; TEAL  = "#45C4FF"; ORNG  = "#FF7F2A"
GREEN = "#2ECC71"; GOLD  = "#FFC107"; RED   = "#FF4D4D"
GREY  = "#8A93A8"; WHITE = "#FFFFFF"; MUTED = "#6A7385"
BORD  = "#35404A"

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 15); ax.set_ylim(0, 7)
ax.axis("off")

# ── helpers ────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, label, sub="", fc=CARD, ec=BORD, lw=1.2,
        fsz=10, sfz=8.5, fc_txt=WHITE, ec_alpha=1.0, radius=0.22):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle=f"round,pad=0.0,rounding_size={radius}",
                           facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=3)
    ax.add_patch(rect)
    if sub:
        ax.text(cx, cy + 0.115, label, ha="center", va="center",
                fontsize=fsz, fontweight="bold", color=fc_txt, zorder=4)
        ax.text(cx, cy - 0.155, sub, ha="center", va="center",
                fontsize=sfz, color=MUTED, zorder=4, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fsz, fontweight="bold", color=fc_txt, zorder=4)

def arr(x1, y1, x2, y2, col=GREY, lw=1.5, style="->", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=lw, linestyle=ls,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)

def curve(x1, y1, x2, y2, col=TEAL, lw=1.8, rad=0.25):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)

def label(x, y, txt, col=MUTED, sz=9, bold=False, ha="center"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=sz,
            color=col, fontweight="bold" if bold else "normal", zorder=5)

# ══════════════════════════════════════════════════════════════════════════
# SECTION HEADERS
# ══════════════════════════════════════════════════════════════════════════
# Existing STEM-GNN band (top)
existing_band = FancyBboxPatch((0.15, 4.45), 14.7, 2.25,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    facecolor="#14181E", edgecolor=ORNG, linewidth=1.0,
    linestyle="--", alpha=0.55, zorder=1)
ax.add_patch(existing_band)
label(0.75, 6.52, "STEM-GNN (existing)", col=ORNG, sz=8.5, bold=True, ha="left")

# New component band (bottom)
new_band = FancyBboxPatch((0.15, 0.35), 14.7, 3.75,
    boxstyle="round,pad=0.0,rounding_size=0.2",
    facecolor="#0C1620", edgecolor=TEAL, linewidth=1.4,
    linestyle="-", alpha=0.45, zorder=1)
ax.add_patch(new_band)
label(0.75, 3.92, "Trust Gate  (new complement)", col=TEAL, sz=8.5, bold=True, ha="left")

# ══════════════════════════════════════════════════════════════════════════
# TOP ROW — STEM-GNN backbone
# ══════════════════════════════════════════════════════════════════════════
# [Node Text] → [LLM φ] → e_v → [VQ Codebook] → z_v → [MoE AGG*] → [Head]

TOP = 5.55
XS  = [1.1, 3.3, 5.2, 7.2, 9.1, 11.2, 13.5]

box(XS[0], TOP, 1.55, 0.75, "Node Text", "t_v",
    fc="#1A1E28", ec=GREY, lw=1.0)

box(XS[1], TOP, 1.65, 0.75, "LLM Encoder", "φ  (frozen)",
    fc="#1C2030", ec=BLUE, lw=1.2)

# e_v junction dot
ax.scatter([XS[2]], [TOP], color=BLUE, s=70, zorder=5)
label(XS[2], TOP + 0.44, "e_v  ∈ ℝ^768", col=BLUE, sz=9.5, bold=True)

box(XS[3], TOP, 1.55, 0.75, "VQ Codebook", "discrete tokens",
    fc="#1C2030", ec=ORNG, lw=1.2)

# z_v dot
ax.scatter([XS[4]], [TOP], color=ORNG, s=70, zorder=5)
label(XS[4], TOP + 0.44, "z_v  (tokens)", col=ORNG, sz=9.5, bold=True)

box(XS[5], TOP, 1.65, 0.75, "MoE Expert", "router (post-VQ)",
    fc="#1C2030", ec=ORNG, lw=1.2)

box(XS[6], TOP, 1.45, 0.75, "Task Head", "classification",
    fc="#1A1E28", ec=GREY, lw=1.0)

# Backbone arrows
for i in range(len(XS) - 1):
    if i == 1:          # encoder → e_v
        arr(XS[1]+0.83, TOP, XS[2]-0.08, TOP, col=BLUE, lw=1.5)
    elif i == 2:        # e_v → VQ
        arr(XS[2]+0.08, TOP, XS[3]-0.78, TOP, col=ORNG, lw=1.5)
    elif i == 3:        # VQ → z_v
        arr(XS[3]+0.78, TOP, XS[4]-0.08, TOP, col=ORNG, lw=1.5)
    elif i == 4:        # z_v → MoE
        arr(XS[4]+0.08, TOP, XS[5]-0.83, TOP, col=ORNG, lw=1.5)
    elif i == 5:        # MoE → Head
        arr(XS[5]+0.83, TOP, XS[6]-0.73, TOP, col=GREY, lw=1.5)
    else:
        arr(XS[i]+0.78, TOP, XS[i+1]-0.78, TOP, col=GREY, lw=1.5)

# ══════════════════════════════════════════════════════════════════════════
# BOTTOM SECTION — Trust Gate new component
# ══════════════════════════════════════════════════════════════════════════

# Row for neighbor inputs (y=2.9)
NEI_Y = 2.9

box(XS[0], NEI_Y, 1.55, 0.72, "Neighbor Text", "{t_u}",
    fc="#1A1E28", ec=GREY, lw=1.0)
box(XS[1], NEI_Y, 1.65, 0.72, "LLM Encoder", "φ  (frozen, shared)",
    fc="#1C2030", ec=BLUE, lw=1.2)
ax.scatter([XS[2]], [NEI_Y], color=BLUE, s=70, zorder=5)
label(XS[2], NEI_Y - 0.42, "{e_u}  neighbors", col=BLUE, sz=9.5, bold=True)

arr(XS[0]+0.78, NEI_Y, XS[1]-0.83, NEI_Y, col=GREY, lw=1.5)
arr(XS[1]+0.83, NEI_Y, XS[2]-0.08, NEI_Y, col=BLUE, lw=1.5)

# Trust Gate box (center, y=1.4)
GATE_X = 6.2; GATE_Y = 1.4
box(GATE_X, GATE_Y, 3.5, 1.1,
    "Semantic Trust Gate",
    "β_v = σ( f(e_v, mean{e_u}) )",
    fc="#0A1E28", ec=TEAL, lw=2.0, fsz=12, sfz=9.5, fc_txt=TEAL)

label(GATE_X, GATE_Y - 0.72,
      "Q: 'Is neighbour u semantically relevant for v\\'s class?'",
      col=GOLD, sz=9, bold=False)

# Gated Aggregation box (right, y=1.4)
AGG_X = 11.0; AGG_Y = 1.4
box(AGG_X, AGG_Y, 3.2, 1.45,
    "Gated Aggregation",
    "",
    fc="#0A201C", ec=GREEN, lw=2.0, fsz=12, fc_txt=GREEN)
label(AGG_X, AGG_Y + 0.12,
      "h_v = β_v · W_self(e_v)", col=WHITE, sz=9.5, bold=False, ha="center")
label(AGG_X, AGG_Y - 0.22,
      " + (1−β_v) · W_neigh(mean e_u)", col=WHITE, sz=9.5, bold=False, ha="center")

# ── connections: e_v (junction) down to gate ──────────────────────────────
arr(XS[2], TOP - 0.38, XS[2], NEI_Y + 0.38, col=BLUE, lw=1.6)

# e_v & {e_u} → Trust Gate
curve(XS[2], TOP - 0.38, GATE_X - 1.75, GATE_Y + 0.3, col=BLUE, lw=1.5, rad=-0.18)
arr(XS[2], NEI_Y - 0.38, GATE_X - 1.75, GATE_Y - 0.1, col=BLUE, lw=1.5)

# Trust Gate → β_v → Gated Aggregation
arr(GATE_X + 1.75, GATE_Y, AGG_X - 1.6, AGG_Y, col=TEAL, lw=2.0)
label((GATE_X + 1.75 + AGG_X - 1.6) / 2, GATE_Y + 0.32,
      "β_v  ∈ [0, 1]", col=TEAL, sz=10, bold=True)

# Gated Aggregation → Task Head (feeds back up)
curve(AGG_X + 1.6, AGG_Y + 0.4, XS[6] - 0.08, TOP - 0.38,
      col=GREEN, lw=2.0, rad=-0.3)
label(13.9, 3.6, "h_v", col=GREEN, sz=11, bold=True)

# Also e_u feeds into Gated Aggregation
arr(XS[2] + 0.08, NEI_Y, AGG_X - 1.6, AGG_Y - 0.22, col=BLUE, lw=1.3, ls="--")

# ── STEM-GNN router label (corruption note) ───────────────────────────────
ax.annotate("uses z_v (post-VQ)\n→ routing corrupted\nby noisy neighbours",
            xy=(XS[5], TOP - 0.38), xytext=(XS[5] - 0.5, 4.58),
            fontsize=8, color=ORNG, ha="center",
            arrowprops=dict(arrowstyle="->", color=ORNG, lw=0.9),
            zorder=6)

# ── Trust gate advantage note ─────────────────────────────────────────────
ax.annotate("uses e_v (pre-VQ)\n→ clean semantic\njudgment",
            xy=(GATE_X, GATE_Y + 0.56), xytext=(GATE_X - 2.5, 3.45),
            fontsize=8, color=TEAL, ha="center",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9),
            zorder=6)

# ══════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════
legend_items = [
    (ORNG, "──",  "Existing STEM-GNN component"),
    (BLUE, "──",  "LLM embeddings (shared encoder)"),
    (TEAL, "──",  "Trust gate signal (β_v)"),
    (GREEN,"──",  "Gated output (h_v)"),
]
for li, (col, sym, lbl) in enumerate(legend_items):
    lx = 0.45 + li * 3.72
    ax.plot([lx, lx + 0.38], [0.52, 0.52], color=col, lw=2.2)
    label(lx + 0.52, 0.52, lbl, col=col, sz=8.5, ha="left")

# Title
ax.text(7.5, 6.87,
        "LLM-Guided Semantic Trust Gate  —  Architecture Diagram",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=WHITE, zorder=6)

plt.tight_layout(pad=0)
out = "/home/lam23005/STEM-GNN/arch_diagram.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
