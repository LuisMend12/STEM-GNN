"""
9-slide research deck: LLM-Guided Semantic Trust Gate
Uses python-pptx for slides + matplotlib for embedded figures.
"""
import json, io, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Palette ───────────────────────────────────────────────────────────────────
C_PURPLE = RGBColor(0x55, 0x3C, 0x9A)
C_BLUE   = RGBColor(0x2B, 0x6C, 0xB0)
C_GREEN  = RGBColor(0x27, 0x67, 0x49)
C_BROWN  = RGBColor(0x9C, 0x42, 0x21)
C_LIGHT  = RGBColor(0xF7, 0xF7, 0xFF)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK   = RGBColor(0x1A, 0x20, 0x2C)
C_GRAY   = RGBColor(0x71, 0x80, 0x96)
C_ACCENT = RGBColor(0xE8, 0x6B, 0x4C)
C_TEAL   = RGBColor(0x4C, 0xC8, 0xA0)

W, H = Inches(13.33), Inches(7.5)   # 16:9 widescreen

# ── Data ─────────────────────────────────────────────────────────────────────
with open("/home/lam23005/STEM-GNN/results/sweep_results.json") as f:
    DATA = json.load(f)

CORA    = DATA["cora_sweep"]
PUBMED  = DATA["pubmed_sweep"]
TEXAS   = DATA["heterophily"]["Texas"]
WISC    = DATA["heterophily"]["Wisconsin"]
CORNELL = DATA["heterophily"]["Cornell"]
RATIOS  = [r * 100 for r in CORA["ratios"]]

GATES = ["none", "A_cosine", "B_confidence", "C_learned", "D_llm"]
GLABEL = {
    "none":         "GraphSAGE (no gate)",
    "A_cosine":     "Gate A: cosine",
    "B_confidence": "Gate B: confidence",
    "C_learned":    "Gate C: learned MLP",
    "D_llm":        "Gate D: LLM cosine",
}
GCOLOR = {
    "none":         "#888888",
    "A_cosine":     "#4C9BE8",
    "B_confidence": "#E86B4C",
    "C_learned":    "#7C4CE8",
    "D_llm":        "#4CC8A0",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)

def bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = color

def txbox(slide, text, x, y, w, h, size=18, bold=False, color=C_DARK,
          align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = wrap
    p  = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color; run.font.italic = italic
    return tb

def rect(slide, x, y, w, h, fill: RGBColor, line=None, lw=0):
    s = slide.shapes.add_shape(1, x, y, w, h)   # MSO_SHAPE_TYPE.RECTANGLE=1
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line; s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    return s

def header_bar(slide, title, subtitle=None, hc=C_PURPLE):
    rect(slide, 0, 0, W, Inches(1.15), hc)
    txbox(slide, title,
          Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.7),
          size=28, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        txbox(slide, subtitle,
              Inches(0.4), Inches(0.75), Inches(12.5), Inches(0.38),
              size=14, color=RGBColor(0xD6, 0xBC, 0xFA), align=PP_ALIGN.LEFT)

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    return buf

def add_img(slide, buf, x, y, w, h=None):
    if h:
        slide.shapes.add_picture(buf, x, y, w, h)
    else:
        slide.shapes.add_picture(buf, x, y, w)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — TITLE
# ─────────────────────────────────────────────────────────────────────────────
def slide_title(prs):
    sl = blank_slide(prs); bg(sl, C_DARK)
    rect(sl, 0, Inches(2.6), W, Inches(2.5), RGBColor(0x2D, 0x1B, 0x69))

    txbox(sl, "LLM-Guided Semantic Trust Gate",
          Inches(0.6), Inches(2.75), Inches(12.0), Inches(1.1),
          size=38, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txbox(sl, "Adaptive Neighbourhood Aggregation for Robust Text-Attributed GNNs",
          Inches(0.6), Inches(3.75), Inches(12.0), Inches(0.6),
          size=18, color=RGBColor(0xD6, 0xBC, 0xFA), align=PP_ALIGN.CENTER,
          italic=True)
    txbox(sl, "Complement to STEM-GNN (KDD'26)",
          Inches(0.6), Inches(4.5), Inches(12.0), Inches(0.5),
          size=15, color=C_GRAY, align=PP_ALIGN.CENTER)

    # three pill badges
    badges = [
        ("Injection Robustness", C_ACCENT),
        ("Heterophily Generalization", C_PURPLE),
        ("LLM Semantic Signals", RGBColor(0x27, 0x67, 0x49)),
    ]
    for i, (txt, col) in enumerate(badges):
        bx = Inches(1.5 + i * 3.5)
        rect(sl, bx, Inches(5.5), Inches(3.0), Inches(0.48), col)
        txbox(sl, txt, bx, Inches(5.5), Inches(3.0), Inches(0.48),
              size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — MOTIVATION & PROBLEM
# ─────────────────────────────────────────────────────────────────────────────
def slide_motivation(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Motivation", "Why do GNNs fail under distribution shift?", C_BLUE)

    # Two problem boxes
    problems = [
        ("① Confusing-Edge Injection",
         C_ACCENT,
         RGBColor(0xFF, 0xEB, 0xEB),
         "Text-similar but wrong-class neighbours\nare injected into the graph.\n\n"
         "GNNs blindly aggregate them → corrupted\nnode representations → accuracy drops."),
        ("② Structural Heterophily",
         C_PURPLE,
         RGBColor(0xFA, 0xF5, 0xFF),
         "Many real graphs connect nodes of\ndifferent classes (h < 0.5).\n\n"
         "Standard equal-weight aggregation\nblurs class boundaries → poor accuracy."),
    ]
    for i, (title, tc, fc, body) in enumerate(problems):
        bx = Inches(0.4 + i * 6.4)
        rect(sl, bx, Inches(1.4), Inches(5.9), Inches(4.8),
             fc, line=tc, lw=1.5)
        rect(sl, bx, Inches(1.4), Inches(5.9), Inches(0.55), tc)
        txbox(sl, title, bx + Inches(0.1), Inches(1.45), Inches(5.7), Inches(0.5),
              size=15, bold=True, color=C_WHITE)
        txbox(sl, body, bx + Inches(0.2), Inches(2.1), Inches(5.5), Inches(3.5),
              size=14, color=C_DARK)

    txbox(sl, "Core Question: When should a node trust its neighbours vs. its own text?",
          Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.7),
          size=16, bold=True, color=C_PURPLE, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — BASELINE SURVEY
# ─────────────────────────────────────────────────────────────────────────────
def slide_baselines(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Prior Work Survey", "6 methods — none handles both challenges", C_GREEN)

    baselines = [
        ("GraphSAGE",     "NeurIPS'17", "Uniform aggregation",                   False, False, False),
        ("GNNGuard",      "NeurIPS'20", "Edge pruning by cosine sim",            False, True,  False),
        ("H2GCN",         "NeurIPS'20", "Ego/neighbour separation",              False, False, True),
        ("GPR-GNN",       "ICLR'21",    "Learnable propagation coefficients",    False, False, True),
        ("TAPE",          "ICLR'24",    "LLM text features as node embeddings",  True,  False, False),
        ("STEM-GNN",      "KDD'26",     "MoE router + VQ codebook (post-VQ)",   True,  False, False),
        ("Ours (Trust Gate)", "—",      "Semantic gate on pre-VQ embeddings",   True,  True,  True),
    ]
    cols  = ["Method", "Venue", "Key Idea", "LLM\nFeatures", "Injection\nRobust", "Hetero-\nphily"]
    col_x = [Inches(v) for v in [0.25, 2.15, 3.35, 8.7, 9.9, 11.1]]
    col_w = [Inches(v) for v in [1.85, 1.15, 5.25, 1.1, 1.15, 1.2]]

    # Header row
    rect(sl, Inches(0.2), Inches(1.3), Inches(12.9), Inches(0.45),
         RGBColor(0xED, 0xED, 0xF7))
    for cx, cw, ch in zip(col_x, col_w, cols):
        txbox(sl, ch, cx, Inches(1.3), cw, Inches(0.45),
              size=11, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)

    for i, (name, venue, idea, llm, inj, het) in enumerate(baselines):
        y = Inches(1.75 + i * 0.67)
        is_ours = i == len(baselines) - 1
        row_col = RGBColor(0xEE, 0xF4, 0xFF) if is_ours else (
                  RGBColor(0xF9, 0xF9, 0xF9) if i % 2 == 0 else C_WHITE)
        rect(sl, Inches(0.2), y, Inches(12.9), Inches(0.63), row_col)
        if is_ours:
            rect(sl, Inches(0.2), y, Inches(0.06), Inches(0.63), C_PURPLE)

        vals = [name, venue, idea]
        for cx, cw, val in zip(col_x[:3], col_w[:3], vals):
            txbox(sl, val, cx, y + Inches(0.06), cw, Inches(0.52),
                  size=12 if not is_ours else 13,
                  bold=is_ours, color=C_PURPLE if is_ours else C_DARK,
                  align=PP_ALIGN.LEFT)

        for cx, cw, val in zip(col_x[3:], col_w[3:], [llm, inj, het]):
            sym = "✓" if val else "✗"
            col = RGBColor(0x16, 0xA3, 0x4A) if val else RGBColor(0xDC, 0x26, 0x26)
            txbox(sl, sym, cx, y + Inches(0.06), cw, Inches(0.52),
                  size=14, bold=val and is_ours, color=col,
                  align=PP_ALIGN.CENTER)

    txbox(sl, "No prior method simultaneously handles all three properties.",
          Inches(0.25), Inches(6.65), Inches(12.5), Inches(0.55),
          size=13, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — OUR METHOD
# ─────────────────────────────────────────────────────────────────────────────
def slide_method(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Our Method: Semantic Trust Gate", "β_v gates how much each node trusts its neighbours", C_PURPLE)

    # Embed the simple arch diagram
    img = "/home/lam23005/STEM-GNN/figures/arch_simple.png"
    add_img(sl, img, Inches(0.3), Inches(1.3), Inches(12.7))

    txbox(sl,
          "Key insight: use pre-VQ embeddings e_v (clean semantic signal) — "
          "bypassing the quantisation noise in STEM-GNN's post-VQ MoE router.",
          Inches(0.3), Inches(6.45), Inches(12.7), Inches(0.8),
          size=13, italic=True, color=C_PURPLE, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — 4 GATE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
def slide_gates(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Four Gate Signal Variants", "What signal f drives β_v?", C_PURPLE)

    gates = [
        ("A", "Cosine Similarity",  "#4C9BE8",
         "cos( h_v ,  mean h_u )",
         "Compares aggregated GNN hidden states.\nProxy for structural compatibility."),
        ("B", "Model Confidence",   "#E86B4C",
         "1 − max softmax( logits_u )",
         "High neighbour uncertainty → distrust.\nMost robust to injection (−3.6 pp drop)."),
        ("C", "Learned MLP",        "#7C4CE8",
         "MLP( e_v ‖ mean e_u )",
         "End-to-end learned gate.\nBest on heterophily (81.1% Texas)."),
        ("D", "LLM Text Cosine",    "#4CC8A0",
         "cos( e_v ,  mean e_u )",
         "Raw LLM embedding similarity.\nCollapes to A under injection."),
    ]
    for i, (letter, name, col, formula, note) in enumerate(gates):
        bx = Inches(0.25 + i * 3.27)
        rc = RGBColor(int(col[1:3],16), int(col[3:5],16), int(col[5:7],16))
        rect(sl, bx, Inches(1.35), Inches(3.0), Inches(5.75), C_WHITE,
             line=rc, lw=1.8)
        rect(sl, bx, Inches(1.35), Inches(3.0), Inches(0.52), rc)
        txbox(sl, f"{letter}  —  {name}", bx + Inches(0.08),
              Inches(1.38), Inches(2.84), Inches(0.48),
              size=13, bold=True, color=C_WHITE)
        txbox(sl, formula, bx + Inches(0.1),
              Inches(2.05), Inches(2.8), Inches(0.55),
              size=11, color=C_DARK,
              italic=False)
        txbox(sl, note, bx + Inches(0.1),
              Inches(2.75), Inches(2.8), Inches(2.8),
              size=11, color=C_GRAY)

    txbox(sl, "All gates share the same aggregation rule:   "
          "h_v = β_v · W_self(e_v)  +  (1−β_v) · W_neigh(mean e_u)",
          Inches(0.25), Inches(6.6), Inches(12.8), Inches(0.65),
          size=14, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — CORA INJECTION RESULTS (figure)
# ─────────────────────────────────────────────────────────────────────────────
def make_cora_fig():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for gate in GATES:
        accs = [v * 100 for v in CORA["results"][gate]]
        ax.plot(RATIOS, accs, color=GCOLOR[gate],
                lw=2.5 if gate != "none" else 1.8,
                ls="--" if gate == "none" else "-",
                marker="o", markersize=5, label=GLABEL[gate])
    ax.set_xlabel("Injected confusing edges (%)", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("Cora — Noise Robustness (confusing edge injection)", fontsize=13, fontweight="bold")
    ax.set_xticks(RATIOS); ax.set_ylim(60, 84)
    ax.grid(axis="y", alpha=0.25, ls="--", color="#CCCCCC")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0)
    fig.tight_layout()
    return fig

def make_injection_fig():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.38})
    for ax in (ax1, ax2): ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    # Lines
    for gate in GATES:
        c_accs = [v * 100 for v in CORA["results"][gate]]
        p_accs = [v * 100 for v in PUBMED["results"][gate]]
        lw = 2.2 if gate != "none" else 1.6; ls = "--" if gate == "none" else "-"
        ax1.plot(RATIOS, c_accs, color=GCOLOR[gate], lw=lw, ls=ls,
                 marker="o", markersize=4, label=GLABEL[gate])
        ax1.plot(RATIOS, p_accs, color=GCOLOR[gate], lw=lw, ls=ls,
                 marker="s", markersize=4, alpha=0.38)
    ax1.set_xlabel("Injected confusing edges (%)"); ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Cora (●)  &  PubMed (■, faded)", fontsize=11, fontweight="bold")
    ax1.set_xticks(RATIOS); ax1.set_ylim(60, 84)
    ax1.grid(axis="y", alpha=0.2, ls="--", color="#CCC")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=9, framealpha=0, loc="lower left")
    # Drop bars
    drops_c = [(CORA["results"][g][-1]   - CORA["results"][g][0])   * 100 for g in GATES]
    drops_p = [(PUBMED["results"][g][-1] - PUBMED["results"][g][0]) * 100 for g in GATES]
    xs = np.arange(len(GATES)); w = 0.36
    ax2.barh(xs + w/2, drops_c, height=w, color=[GCOLOR[g] for g in GATES], alpha=0.85, label="Cora")
    ax2.barh(xs - w/2, drops_p, height=w, color=[GCOLOR[g] for g in GATES],
             alpha=0.38, hatch="///", edgecolor="white", label="PubMed")
    for i, (dc, dp) in enumerate(zip(drops_c, drops_p)):
        ax2.text(dc - 0.1, xs[i]+w/2, f"{dc:+.1f}", va="center", ha="right",
                 fontsize=9, color=GCOLOR[GATES[i]], fontweight="bold")
        ax2.text(dp - 0.1, xs[i]-w/2, f"{dp:+.1f}", va="center", ha="right",
                 fontsize=9, color=GCOLOR[GATES[i]])
    ax2.set_yticks(xs); ax2.set_yticklabels([GLABEL[g] for g in GATES], fontsize=9)
    for tick, g in zip(ax2.get_yticklabels(), GATES): tick.set_color(GCOLOR[g])
    ax2.set_xlabel("Accuracy drop (pp)"); ax2.set_title("Drop: clean → 50% injection", fontsize=11, fontweight="bold")
    ax2.axvline(0, color="#AAA", lw=0.8); ax2.grid(axis="x", alpha=0.2, ls="--", color="#CCC")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9, framealpha=0)
    fig.tight_layout(); return fig

def slide_cora(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Result 1 — Injection Robustness (Cora + PubMed)",
               "Gate B most robust on both datasets: −3.6 pp (Cora), −2.6 pp (PubMed)", C_BROWN)
    buf = fig_to_img(make_injection_fig())
    add_img(sl, buf, Inches(0.3), Inches(1.25), Inches(9.5))

    rect(sl, Inches(9.95), Inches(1.25), Inches(3.15), Inches(5.9),
         RGBColor(0xFF, 0xF8, 0xF3), line=C_BROWN, lw=1.2)
    txbox(sl, "Key Takeaways", Inches(10.05), Inches(1.35), Inches(2.95), Inches(0.5),
          size=13, bold=True, color=C_BROWN)
    bullets = [
        "B (confidence) most robust on both Cora (−3.6) and PubMed (−2.6 pp)",
        "C (learned MLP) also stable: −5.9 pp Cora, −1.1 pp PubMed",
        "Cosine gates (A, D): −8.4 pp — no robustness benefit",
        "Pattern consistent across two independent datasets",
    ]
    for i, b in enumerate(bullets):
        txbox(sl, f"• {b}", Inches(10.05), Inches(1.95 + i * 1.1),
              Inches(2.95), Inches(1.0), size=11, color=C_DARK)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — TEXAS HETEROPHILY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def make_hetero_fig():
    hetero_datasets = [
        ("Texas\nh=0.31",     TEXAS["results"]),
        ("Wisconsin\nh=0.37", WISC["results"]),
        ("Cornell\nh=0.34",   CORNELL["results"]),
    ]
    n_ds = len(hetero_datasets); w = 0.13; group_gap = 0.85
    xs_g = np.arange(n_ds) * group_gap
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for gi, gate in enumerate(GATES):
        offsets = (gi - n_ds + 0.5) * w * 1.15
        accs = [ds[gate] * 100 for _, ds in hetero_datasets]
        ax.bar(xs_g + offsets, accs, width=w, color=GCOLOR[gate],
               label=GLABEL[gate], alpha=0.85)
    for i, (_, ds_res) in enumerate(hetero_datasets):
        ax.plot([xs_g[i] - 0.38, xs_g[i] + 0.38],
                [ds_res["none"]*100]*2,
                color="#888888", lw=1.5, ls="--", alpha=0.7, zorder=5)
    ax.set_xticks(xs_g)
    ax.set_xticklabels([lbl for lbl, _ in hetero_datasets], fontsize=11)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(35, 105)
    ax.set_title("Texas / Wisconsin / Cornell  (dashed = no-gate baseline)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, ls="--", color="#CCC")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    fig.tight_layout(); return fig

def slide_texas(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Result 2 — Heterophily Generalization (3 Datasets)",
               "B & C improve over baseline on ALL three datasets; cosine gates consistently hurt", C_GREEN)
    buf = fig_to_img(make_hetero_fig())
    add_img(sl, buf, Inches(0.3), Inches(1.25), Inches(9.0))

    rect(sl, Inches(9.5), Inches(1.25), Inches(3.6), Inches(5.9),
         RGBColor(0xF0, 0xFF, 0xF4), line=C_GREEN, lw=1.2)
    txbox(sl, "Key Takeaways", Inches(9.6), Inches(1.35), Inches(3.4), Inches(0.5),
          size=13, bold=True, color=C_GREEN)
    bullets = [
        "Texas:     B & C → 81.1%  (+4.5 pp)",
        "Wisconsin: B & C → 86.9%  (+5.9 pp)",
        "Cornell:   B     → 76.6%  (+4.5 pp)",
        "Cosine gates (A, D) hurt on ALL three — drop 15–20 pp below baseline",
        "Confidence signal reliable regardless of graph structure",
    ]
    for i, b in enumerate(bullets):
        txbox(sl, f"• {b}", Inches(9.6), Inches(1.95 + i * 0.98),
              Inches(3.4), Inches(0.92), size=11, color=C_DARK)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — COMBINED SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def slide_summary_fig(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Summary — Contribution & Results",
               "Research gap filled + empirical evidence", C_PURPLE)
    img = "/home/lam23005/STEM-GNN/figures/contribution_results.png"
    add_img(sl, img, Inches(0.15), Inches(1.2), Inches(13.0))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — RESEARCH DIRECTION
# ─────────────────────────────────────────────────────────────────────────────
def slide_future(prs):
    sl = blank_slide(prs); bg(sl, C_WHITE)
    header_bar(sl, "Research Direction & Next Steps", "", C_BROWN)

    directions = [
        ("Short term — Strengthen experiments",
         C_BLUE, RGBColor(0xEB, 0xF4, 0xFF),
         [
             "Run PubMed + WikiCS injection sweeps",
             "Wisconsin & Cornell heterophily benchmarks",
             "Add GNNGuard as an active baseline in our injection setup",
             "Report mean ± std over 5 seeds",
         ]),
        ("Medium term — Improve the gate",
         C_PURPLE, RGBColor(0xFA, 0xF5, 0xFF),
         [
             "LLM label-support query: ask LLM if u's text supports v's class",
             "Multi-hop gates: consider 2-hop neighbourhood trust",
             "Plug into full STEM-GNN pretrain → finetune pipeline",
             "Joint training of gate + VQ codebook",
         ]),
        ("Long term — Research paper",
         C_GREEN, RGBColor(0xF0, 0xFF, 0xF4),
         [
             "Tri-objective evaluation: ID acc / OOD robustness / transfer",
             "Theoretical analysis: gate as Lipschitz regulariser",
             "Generalise to link prediction & graph classification",
             "Submit to NeurIPS'25 / ICLR'26",
         ]),
    ]
    for i, (title, tc, fc, points) in enumerate(directions):
        bx = Inches(0.2 + i * 4.4)
        rect(sl, bx, Inches(1.3), Inches(4.05), Inches(5.5), fc, line=tc, lw=1.5)
        rect(sl, bx, Inches(1.3), Inches(4.05), Inches(0.5), tc)
        txbox(sl, title, bx + Inches(0.08), Inches(1.34), Inches(3.88), Inches(0.45),
              size=12, bold=True, color=C_WHITE)
        for j, pt in enumerate(points):
            txbox(sl, f"• {pt}", bx + Inches(0.12), Inches(1.98 + j * 0.82),
                  Inches(3.84), Inches(0.78), size=11.5, color=C_DARK)

    txbox(sl,
          "Core claim: A semantic gate on pre-VQ LLM embeddings is a lightweight, "
          "drop-in complement to STEM-GNN that addresses both injection robustness "
          "and structural heterophily — filling a gap no existing method covers.",
          Inches(0.2), Inches(6.3), Inches(12.9), Inches(0.9),
          size=13, bold=True, color=C_PURPLE, align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD DECK
# ─────────────────────────────────────────────────────────────────────────────
prs = new_prs()
slide_title(prs)
slide_motivation(prs)
slide_baselines(prs)
slide_method(prs)
slide_gates(prs)
slide_cora(prs)
slide_texas(prs)
slide_summary_fig(prs)
slide_future(prs)

out = "/home/lam23005/STEM-GNN/slides/trust_gate_deck.pptx"
prs.save(out)
print(f"Saved {len(prs.slides)}-slide deck → {out}")
