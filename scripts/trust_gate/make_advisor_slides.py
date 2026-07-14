"""
Advisor presentation: LLM-Guided Semantic Trust Gate
7 slides — architecture slide built as native PPTX shapes (no embedded PNG).
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

W, H = Inches(13.33), Inches(7.5)

BLUE   = RGBColor(0x25, 0x63, 0xEB)
PURPLE = RGBColor(0x7C, 0x3A, 0xED)
GRAY   = RGBColor(0x1E, 0x29, 0x3B)
LGRAY  = RGBColor(0x64, 0x74, 0x8B)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GREEN  = RGBColor(0x05, 0x96, 0x69)
RED    = RGBColor(0xDC, 0x26, 0x26)
TEAL   = RGBColor(0x14, 0xB8, 0xA6)
ORANGE = RGBColor(0xD9, 0x77, 0x06)
BG     = RGBColor(0xF8, 0xFA, 0xFF)
BG2    = RGBColor(0xEE, 0xF4, 0xFF)
LBLUE  = RGBColor(0xBF, 0xDB, 0xFE)
LPURP  = RGBColor(0xED, 0xE9, 0xFE)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
blank = prs.slide_layouts[6]

def slide():
    return prs.slides.add_slide(blank)

def rect(sl, x, y, w, h, fill=None, line_color=None, line_w=Pt(1.2)):
    shp = sl.shapes.add_shape(1, x, y, w, h)
    shp.fill.solid() if fill else shp.fill.background()
    if fill: shp.fill.fore_color.rgb = fill
    if line_color:
        shp.line.color.rgb = line_color
        shp.line.width = line_w
    else:
        shp.line.fill.background()
    return shp

def txt(sl, text, x, y, w, h, size=14, bold=False, color=GRAY,
        align=PP_ALIGN.LEFT, italic=False):
    tb = sl.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.italic = italic; run.font.color.rgb = color
    return tb

def arrow(sl, x1, y1, x2, y2, color=GRAY, lw=Pt(2)):
    """Draw a simple arrow using a connector."""
    cx = sl.shapes.add_connector(1, x1, y1, x2, y2)  # MSO_CONNECTOR.STRAIGHT=1
    cx.line.color.rgb = color
    cx.line.width = lw
    # Add arrowhead via XML
    ln = cx._element.spPr.find(qn('a:ln'))
    if ln is None:
        ln = etree.SubElement(cx._element.spPr, qn('a:ln'))
    tail = etree.SubElement(ln, qn('a:tailEnd'))
    tail.set('type', 'none')
    head = etree.SubElement(ln, qn('a:headEnd'))
    head.set('type', 'arrow')
    head.set('w', 'med')
    head.set('len', 'med')
    return cx

def header_bar(sl, title, subtitle=None):
    rect(sl, 0, 0, W, Inches(1.05), fill=BLUE)
    txt(sl, title, Inches(0.4), Inches(0.07), Inches(12.5), Inches(0.58),
        size=28, bold=True, color=WHITE)
    if subtitle:
        txt(sl, subtitle, Inches(0.4), Inches(0.65), Inches(12.5), Inches(0.36),
            size=13, color=LBLUE)

def pill(sl, text, x, y, w, h, bg, fg=WHITE, size=11):
    rect(sl, x, y, w, h, fill=bg)
    txt(sl, text, x, y+Inches(0.03), w, h-Inches(0.03), size=size,
        bold=True, color=fg, align=PP_ALIGN.CENTER)

def img(sl, path, x, y, w, h):
    try: sl.shapes.add_picture(path, x, y, w, h)
    except Exception as e: print(f"  img error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Slide 1 — Title
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
rect(sl, 0, 0, W, H, fill=BG2)
rect(sl, 0, 0, W, Inches(0.09), fill=BLUE)
rect(sl, 0, H-Inches(0.09), W, Inches(0.09), fill=PURPLE)

txt(sl, "LLM-Guided Semantic Trust Gate",
    Inches(0.7), Inches(1.5), Inches(11.9), Inches(1.2),
    size=40, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "Using frozen LLM text signals to control how much each GNN node trusts its neighbours",
    Inches(0.7), Inches(2.8), Inches(11.9), Inches(0.65),
    size=20, color=LGRAY, align=PP_ALIGN.CENTER)

for i, (t, c) in enumerate([
    ("BOW backbone", BLUE), ("LLM gate oracle", PURPLE), ("9 gate signals", TEAL)
]):
    pill(sl, t, Inches(3.2 + i*2.35), Inches(3.7), Inches(2.1), Inches(0.42), c)

txt(sl, "Backbone: GraphSAGE on original BOW features  ·  Gate: frozen DistilBERT (768-dim) on raw text\nDatasets: Cora (1433-dim BOW)  ·  PubMed (500-dim BOW)",
    Inches(2.0), Inches(4.4), Inches(9.3), Inches(1.0),
    size=14, color=LGRAY, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# Slide 2 — Motivation
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
header_bar(sl, "Motivation", "Why do GNNs need a semantic trust gate?")

rect(sl, Inches(0.3), Inches(1.2), Inches(6.1), Inches(2.7),
     fill=RGBColor(0xFF,0xF0,0xF0), line_color=RED)
txt(sl, "Problem 1: Confusing edge injection",
    Inches(0.48), Inches(1.27), Inches(5.8), Inches(0.42),
    size=15, bold=True, color=RED)
txt(sl, ("An attacker connects nodes with similar text but different labels. "
         "Standard GNNs blindly aggregate wrong-class features, degrading accuracy."),
    Inches(0.48), Inches(1.73), Inches(5.8), Inches(0.85), size=13, color=GRAY)
txt(sl, "BOW baseline: −6.5 pp at 50% injection on PubMed",
    Inches(0.48), Inches(2.65), Inches(5.8), Inches(0.38),
    size=12, bold=True, color=RED)

rect(sl, Inches(6.93), Inches(1.2), Inches(6.1), Inches(2.7),
     fill=RGBColor(0xF0,0xF4,0xFF), line_color=BLUE)
txt(sl, "Problem 2: Heterophilic graphs",
    Inches(7.1), Inches(1.27), Inches(5.8), Inches(0.42),
    size=15, bold=True, color=BLUE)
txt(sl, ("In heterophilic graphs (h < 0.4), most neighbours belong to different classes. "
         "Aggregating equally hurts — the node should trust itself more."),
    Inches(7.1), Inches(1.73), Inches(5.8), Inches(0.85), size=13, color=GRAY)
txt(sl, "Texas h=0.31  ·  Wisconsin h=0.37  ·  Cornell h=0.34",
    Inches(7.1), Inches(2.65), Inches(5.8), Inches(0.38),
    size=12, bold=True, color=BLUE)

rect(sl, Inches(0.3), Inches(4.1), Inches(12.73), Inches(2.1),
     fill=RGBColor(0xF0,0xFF,0xF6), line_color=GREEN)
txt(sl, "Solution: Per-node trust gate β_v from LLM text signals",
    Inches(0.5), Inches(4.18), Inches(12.3), Inches(0.42),
    size=15, bold=True, color=GREEN)
txt(sl, "h_v = β_v · W_self(x_v)  +  (1−β_v) · W_neigh(mean x_u)",
    Inches(0.5), Inches(4.66), Inches(12.3), Inches(0.42),
    size=14, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, ("β_v → 1 : trust self  (node text differs from neighbours — heterophilic or attacked)\n"
         "β_v → 0 : trust neighbours  (node text aligns with neighbours — safe homophilic region)\n"
         "Gate computed from FROZEN LLM embeddings of raw text — fully decoupled from GNN backbone."),
    Inches(0.5), Inches(5.12), Inches(12.3), Inches(0.9), size=12, color=LGRAY)

# ══════════════════════════════════════════════════════════════════════════════
# Slide 3 — Architecture (native shapes)
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
header_bar(sl, "Architecture", "Backbone and gate are fully decoupled — LLM never touches node features")

# ── Lane labels ──────────────────────────────────────────────────────────────
rect(sl, Inches(0.18), Inches(1.18), Inches(0.28), Inches(2.2), fill=BLUE)
txt(sl, "BACKBONE", Inches(0.18), Inches(1.95), Inches(0.28), Inches(0.65),
    size=8, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

rect(sl, Inches(0.18), Inches(4.35), Inches(0.28), Inches(2.2), fill=PURPLE)
txt(sl, "LLM GATE", Inches(0.18), Inches(5.12), Inches(0.28), Inches(0.65),
    size=8, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# ── Backbone row boxes ────────────────────────────────────────────────────────
# [BOW Features] → [GraphSAGE] → [Node Repr h_v] → [Classifier]
bx = [Inches(0.65), Inches(3.2), Inches(6.9), Inches(10.35)]
by = Inches(1.35); bh = Inches(1.8); bw = Inches(2.3)

# BOW Features box
rect(sl, bx[0], by, bw, bh, fill=RGBColor(0xF1,0xF5,0xF9), line_color=LGRAY)
txt(sl, "Node Features", bx[0]+Inches(0.12), by+Inches(0.18), bw-Inches(0.24), Inches(0.38),
    size=14, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "BOW vectors\n(original features)", bx[0]+Inches(0.1), by+Inches(0.6), bw-Inches(0.2), Inches(0.7),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)
txt(sl, "Cora: 1433-dim  |  PubMed: 500-dim",
    bx[0]+Inches(0.06), by+Inches(1.35), bw-Inches(0.12), Inches(0.35),
    size=9, color=LGRAY, align=PP_ALIGN.CENTER)

# GNN Backbone box (highlighted)
rect(sl, bx[1], by, bw+Inches(0.3), bh, fill=RGBColor(0xEB,0xF2,0xFF), line_color=BLUE, line_w=Pt(2))
rect(sl, bx[1], by, bw+Inches(0.3), Inches(0.48), fill=BLUE)
txt(sl, "ENCODER", bx[1], by+Inches(0.04), bw+Inches(0.3), Inches(0.38),
    size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
txt(sl, "GNN Backbone", bx[1]+Inches(0.1), by+Inches(0.6), bw+Inches(0.1), Inches(0.4),
    size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
txt(sl, "GraphSAGE\n(GCN or GraphSAGE)", bx[1]+Inches(0.1), by+Inches(1.02), bw+Inches(0.1), Inches(0.6),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)

# Node Repr box
rect(sl, bx[2], by, bw, bh, fill=RGBColor(0xF1,0xF5,0xF9), line_color=LGRAY)
txt(sl, "Node Repr.", bx[2]+Inches(0.12), by+Inches(0.18), bw-Inches(0.24), Inches(0.38),
    size=14, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "h_v", bx[2]+Inches(0.1), by+Inches(0.62), bw-Inches(0.2), Inches(0.55),
    size=22, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "128-dim hidden", bx[2]+Inches(0.1), by+Inches(1.25), bw-Inches(0.2), Inches(0.38),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)

# Classifier box
rect(sl, bx[3], by, bw, bh, fill=RGBColor(0xFF,0xF7,0xED), line_color=ORANGE)
txt(sl, "Classifier", bx[3]+Inches(0.12), by+Inches(0.18), bw-Inches(0.24), Inches(0.38),
    size=14, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
txt(sl, "softmax(W h_v)", bx[3]+Inches(0.1), by+Inches(0.62), bw-Inches(0.2), Inches(0.55),
    size=13, color=ORANGE, align=PP_ALIGN.CENTER)
txt(sl, "node label prediction", bx[3]+Inches(0.1), by+Inches(1.25), bw-Inches(0.2), Inches(0.38),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)

# Backbone arrows
mid_y = by + bh/2
arrow(sl, bx[0]+bw, mid_y, bx[1], mid_y, color=GRAY)
arrow(sl, bx[1]+bw+Inches(0.3), mid_y, bx[2], mid_y, color=BLUE)
arrow(sl, bx[2]+bw, mid_y, bx[3], mid_y, color=GRAY)
txt(sl, "x_v", bx[0]+bw+Inches(0.05), mid_y-Inches(0.32), Inches(0.5), Inches(0.3),
    size=11, bold=True, color=GRAY)
txt(sl, "h_v", bx[1]+bw+Inches(0.35), mid_y-Inches(0.32), Inches(0.5), Inches(0.3),
    size=11, bold=True, color=BLUE)

# ── Gate row boxes ────────────────────────────────────────────────────────────
gx = [Inches(0.65), Inches(3.2), Inches(6.75)]
gy = Inches(4.5); gh = Inches(1.75); gw = Inches(2.3)

# Raw Text box
rect(sl, gx[0], gy, gw, gh, fill=RGBColor(0xF5,0xF3,0xFF), line_color=LPURP)
txt(sl, "Raw Node Text", gx[0]+Inches(0.1), gy+Inches(0.15), gw-Inches(0.2), Inches(0.4),
    size=13, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
txt(sl, "Paper titles &\nabstracts", gx[0]+Inches(0.1), gy+Inches(0.58), gw-Inches(0.2), Inches(0.65),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)
txt(sl, "\"Neural networks for...\"", gx[0]+Inches(0.1), gy+Inches(1.28), gw-Inches(0.2), Inches(0.35),
    size=9, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)

# DistilBERT box
rect(sl, gx[1], gy, gw+Inches(0.3), gh, fill=RGBColor(0xF5,0xF3,0xFF), line_color=PURPLE, line_w=Pt(1.5))
rect(sl, gx[1], gy, gw+Inches(0.3), Inches(0.42), fill=PURPLE)
txt(sl, "FROZEN", gx[1], gy+Inches(0.04), gw+Inches(0.3), Inches(0.32),
    size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
txt(sl, "DistilBERT", gx[1]+Inches(0.1), gy+Inches(0.55), gw+Inches(0.1), Inches(0.42),
    size=14, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
txt(sl, "multi-qa-distilbert-cos-v1\ne_v ∈ ℝ^768", gx[1]+Inches(0.1), gy+Inches(1.0), gw+Inches(0.1), Inches(0.6),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER, italic=True)

# Trust Gate box (OUR CONTRIBUTION)
rect(sl, gx[2], gy, gw+Inches(0.6), gh, fill=LPURP, line_color=PURPLE, line_w=Pt(2.5))
rect(sl, gx[2], gy, gw+Inches(0.6), Inches(0.42), fill=PURPLE)
txt(sl, "OUR CONTRIBUTION", gx[2], gy+Inches(0.04), gw+Inches(0.6), Inches(0.32),
    size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
txt(sl, "Semantic Trust Gate", gx[2]+Inches(0.1), gy+Inches(0.55), gw+Inches(0.4), Inches(0.4),
    size=14, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
txt(sl, "β_v = f( e_v,  mean{e_u} )", gx[2]+Inches(0.1), gy+Inches(1.0), gw+Inches(0.4), Inches(0.38),
    size=13, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "β_v ∈ [0, 1]", gx[2]+Inches(0.1), gy+Inches(1.38), gw+Inches(0.4), Inches(0.28),
    size=11, color=LGRAY, align=PP_ALIGN.CENTER)

# Gate arrows
gate_mid_y = gy + gh/2
arrow(sl, gx[0]+gw, gate_mid_y, gx[1], gate_mid_y, color=PURPLE)
arrow(sl, gx[1]+gw+Inches(0.3), gate_mid_y, gx[2], gate_mid_y, color=PURPLE)
txt(sl, "text", gx[0]+bw+Inches(0.05), gate_mid_y-Inches(0.3), Inches(0.4), Inches(0.28),
    size=10, color=PURPLE)
txt(sl, "e_v", gx[1]+gw+Inches(0.35), gate_mid_y-Inches(0.3), Inches(0.4), Inches(0.28),
    size=11, bold=True, color=PURPLE)

# β_v arrow going UP from Trust Gate to GNN Backbone
gate_top_x = gx[2] + (gw+Inches(0.6))/2
arrow(sl, gate_top_x, gy, gate_top_x, by+bh, color=PURPLE, lw=Pt(2.5))
txt(sl, "β_v", gate_top_x+Inches(0.08), gy-Inches(0.9), Inches(0.5), Inches(0.35),
    size=13, bold=True, color=PURPLE)

# ── Key equation box ──────────────────────────────────────────────────────────
rect(sl, Inches(9.2), Inches(1.2), Inches(3.95), Inches(5.1),
     fill=RGBColor(0xF8,0xFA,0xFF), line_color=RGBColor(0xE2,0xE8,0xF0))
txt(sl, "Gating equation", Inches(9.35), Inches(1.3), Inches(3.65), Inches(0.38),
    size=13, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "h_v  =  β_v · W_self(x_v)",
    Inches(9.35), Inches(1.78), Inches(3.65), Inches(0.38),
    size=13, color=GRAY, align=PP_ALIGN.CENTER)
txt(sl, "+  (1−β_v) · W_neigh(mean x_u)",
    Inches(9.35), Inches(2.18), Inches(3.65), Inches(0.38),
    size=13, color=GRAY, align=PP_ALIGN.CENTER)

# Divider
rect(sl, Inches(9.35), Inches(2.67), Inches(3.6), Inches(0.025),
     fill=RGBColor(0xE2,0xE8,0xF0))

for i, (bv, desc, c) in enumerate([
    ("β_v → 1", "trust self (attack / heterophily)", RED),
    ("β_v → 0", "trust neighbours (safe / homophilic)", GREEN),
    ("β_v = 0.5", "balanced blend", LGRAY),
]):
    y_i = Inches(2.82) + i*Inches(0.7)
    pill(sl, bv, Inches(9.35), y_i, Inches(1.1), Inches(0.38),
         c, WHITE, size=11)
    txt(sl, desc, Inches(10.55), y_i+Inches(0.04), Inches(2.5), Inches(0.35),
        size=10, color=LGRAY)

txt(sl, "Gate variants (all LLM-based):",
    Inches(9.35), Inches(5.02), Inches(3.65), Inches(0.32),
    size=11, bold=True, color=GRAY)
for i, g in enumerate(["C — MLP  ★", "D — Variance  ★", "G — Bilinear  ★", "I — Cosine std  ★"]):
    txt(sl, f"• {g}", Inches(9.35), Inches(5.38)+i*Inches(0.32),
        Inches(3.65), Inches(0.3), size=10, color=PURPLE)

# ══════════════════════════════════════════════════════════════════════════════
# Slide 4 — 9 Gate Signals
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
header_bar(sl, "9 LLM Gate Signals", "All β_v computed purely from frozen DistilBERT embeddings — no GNN features")

gates = [
    ("A", "LLM cosine",        "1 − σ(cos(e_v, mean{e_u}))",         "Direct alignment: is node text similar to avg neighbour?"),
    ("B", "LLM entropy",       "H(softmax(e_v·e_u)) / log(deg)",     "Confusion: is attention spread (uncertain) or peaked?"),
    ("C", "LLM MLP",           "σ(MLP(e_v ‖ mean{e_u}))  [learned]","Learned combination of node + neighbour embeddings"),
    ("D", "LLM variance",      "σ(10 · mean ‖e_u − μ‖²)",           "How spread are neighbour LLM embeddings?"),
    ("E", "LLM max cosine",    "1 − max_u cos(e_v, e_u)",            "Single closest neighbour instead of mean"),
    ("F", "Neighbour agree",   "1 − ‖mean{e_u}‖²",                  "Do neighbours agree with each other?"),
    ("G", "LLM bilinear",      "σ(e_v^T U^T V mean{e_u})  [learned]","Low-rank learned asymmetric similarity"),
    ("H", "LLM top-k cosine",  "1 − mean cos(e_v, top-3 e_u)",      "Filtered: ignore outlier neighbours"),
    ("I", "Cosine std",        "σ(std{ cos(e_v, e_u) })",           "Spread of per-edge cosines = uncertainty"),
]
ROBUST = {"C","D","G","I"}

for i, (letter, name, formula, desc) in enumerate(gates):
    y = Inches(1.22) + i * Inches(0.665)
    bg = RGBColor(0xEE,0xF4,0xFF) if letter in ROBUST else RGBColor(0xF8,0xFA,0xFF)
    lc = BLUE if letter in ROBUST else RGBColor(0xE2,0xE8,0xF0)
    rect(sl, Inches(0.2), y, Inches(12.93), Inches(0.62), fill=bg, line_color=lc, line_w=Pt(0.8))
    pill(sl, letter, Inches(0.28), y+Inches(0.12), Inches(0.42), Inches(0.36),
         PURPLE if letter in ROBUST else LGRAY)
    txt(sl, ("★ " if letter in ROBUST else "") + name,
        Inches(0.82), y+Inches(0.14), Inches(2.0), Inches(0.36),
        size=12, bold=(letter in ROBUST),
        color=PURPLE if letter in ROBUST else GRAY)
    txt(sl, formula, Inches(2.95), y+Inches(0.14), Inches(3.8), Inches(0.36),
        size=11, color=RGBColor(0x1E,0x40,0xAF))
    txt(sl, desc, Inches(6.9), y+Inches(0.14), Inches(6.1), Inches(0.36),
        size=11, color=LGRAY)

txt(sl, "★ = robust under injection attack (validated on Cora + PubMed)",
    Inches(0.2), Inches(7.12), Inches(12.9), Inches(0.3),
    size=11, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════════
# Slide 5 — Results figure
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
header_bar(sl, "Results: 9 Gate Signals Under Injection Attack",
           "BOW backbone (GraphSAGE) + frozen LLM gate oracle — Cora & PubMed")
img(sl, "/home/lam23005/STEM-GNN/figures/llm_gates_results.png",
    Inches(0.15), Inches(1.12), Inches(13.03), Inches(5.72))

# ══════════════════════════════════════════════════════════════════════════════
# Slide 6 — Key Findings
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
header_bar(sl, "Key Findings", "What the 9-gate sweep reveals")

findings = [
    (GREEN, "★ Uncertainty gates are most robust",
     "Gates measuring spread/uncertainty of LLM embeddings — C (MLP), D (variance), G (bilinear), I (cosine-std) — "
     "lose only 0.2–3.4 pp vs 6.5 pp baseline drop on PubMed. D (variance) even slightly improves (+0.2 pp)."),
    (RED, "✗ Similarity gates amplify the attack",
     "Gates based on direct text similarity (A cosine, E max-cosine, H top-k) trust injected confusing edges — "
     "they see high LLM similarity and lower β_v, aggregating wrong-class features. Drop up to −12.4 pp (worse than no gate)."),
    (BLUE, "→ Decoupling backbone from gate works",
     "Using frozen LLM embeddings as a pure semantic oracle separate from the BOW backbone "
     "allows the GNN to stay clean while the gate provides external text-level reasoning. "
     "Learned gates (C, G) train a lightweight head without touching backbone features."),
    (PURPLE, "→ Open question: Cora vs PubMed alignment gap",
     "Gates C/D/G underperform on Cora (59–75% vs 79% baseline). Hypothesis: 1433-dim Cora BOW and "
     "768-dim LLM embeddings span misaligned spaces. Cross-modal projection may close this gap."),
]

y = Inches(1.22)
for color, title, body in findings:
    rect(sl, Inches(0.22), y, Inches(12.89), Inches(1.45),
         fill=RGBColor(0xF8,0xFA,0xFF), line_color=color, line_w=Pt(2))
    rect(sl, Inches(0.22), y, Inches(0.22), Inches(1.45), fill=color)
    txt(sl, title, Inches(0.56), y+Inches(0.1), Inches(12.4), Inches(0.4),
        size=13, bold=True, color=color)
    txt(sl, body, Inches(0.56), y+Inches(0.52), Inches(12.4), Inches(0.85),
        size=11, color=GRAY)
    y += Inches(1.57)

# ══════════════════════════════════════════════════════════════════════════════
# Slide 7 — Research Directions
# ══════════════════════════════════════════════════════════════════════════════
sl = slide()
header_bar(sl, "Research Directions", "Next steps based on the 9-gate findings")

steps = [
    ("Short\nterm",  BLUE,
     ["Run best 4 gates (C, D, G, I) on heterophilic datasets: Texas / Wisconsin / Cornell",
      "Add cross-modal projection layer to align BOW ↔ LLM spaces (fixes Cora gap)",
      "Formal comparison vs GNNGuard and H2GCN on injection robustness benchmark"]),
    ("Medium\nterm", PURPLE,
     ["Ensemble top gates: learned mixture-of-signals router over (C, D, G, I)",
      "Extend to dynamic graphs where text changes over time (gate recomputed on-the-fly)",
      "Theoretical analysis: when does text similarity ≠ structural trust?"]),
    ("Long\nterm",   TEAL,
     ["Replace DistilBERT with instruction-tuned LLM (Llama-3, GPT-4o) for richer signals",
      "Train gate jointly with backbone end-to-end (currently frozen / decoupled)",
      "Apply to knowledge graphs, social networks, and citation networks with rich text"]),
]

y = Inches(1.22)
for term, color, bullets in steps:
    rect(sl, Inches(0.22), y, Inches(12.89), Inches(1.9),
         fill=RGBColor(0xF8,0xFA,0xFF), line_color=color, line_w=Pt(1.5))
    rect(sl, Inches(0.22), y, Inches(1.4), Inches(1.9), fill=color)
    txt(sl, term, Inches(0.22), y+Inches(0.62), Inches(1.4), Inches(0.65),
        size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for bi, b in enumerate(bullets):
        txt(sl, f"• {b}", Inches(1.75), y+Inches(0.17)+bi*Inches(0.56),
            Inches(11.2), Inches(0.5), size=12, color=GRAY)
    y += Inches(2.06)

# ── Save ──────────────────────────────────────────────────────────────────────
import os
out = "/home/lam23005/STEM-GNN/slides/advisor_deck.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print(f"Saved → {out}  ({prs.slides.__len__()} slides)")
