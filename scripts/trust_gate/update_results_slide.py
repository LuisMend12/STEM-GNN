"""
Replaces slide 6 of research_framework.pptx with the actual figure + key numbers table.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import copy

DARK   = RGBColor(0x12, 0x14, 0x1A)
CARD   = RGBColor(0x1E, 0x22, 0x2C)
BLUE   = RGBColor(0x45, 0x9B, 0xFF)
GOLD   = RGBColor(0xFF, 0xC1, 0x07)
GREEN  = RGBColor(0x2E, 0xCC, 0x71)
RED    = RGBColor(0xFF, 0x4D, 0x4D)
ORANGE = RGBColor(0xFF, 0x7F, 0x2A)
PURPLE = RGBColor(0xA0, 0x6C, 0xFF)
MUTED  = RGBColor(0x8A, 0x93, 0xA8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
TEAL   = RGBColor(0x45, 0xC4, 0xFF)

W = Inches(13.33); H = Inches(7.5)

def bx(sl, l, t, w, h, c=CARD):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = c
    s.line.fill.background(); return s

def bd(sl, l, t, w, h, c=CARD, bc=BLUE, bp=1.2):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = c
    s.line.color.rgb = bc; s.line.width = Pt(bp); return s

def tx(sl, text, l, t, w, h, sz=12, bold=False, col=WHITE,
       align=PP_ALIGN.LEFT, italic=False):
    tb = sl.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    r  = p.add_run(); r.text = text
    r.font.size = Pt(sz); r.font.bold = bold
    r.font.color.rgb = col; r.font.italic = italic
    return tb

def ln(sl, x1, y1, x2, y2, c=MUTED, pt=0.75):
    cn = sl.shapes.add_connector(1, x1, y1, x2, y2)
    cn.line.color.rgb = c; cn.line.width = Pt(pt)

# ── Load and clear slide 6 ────────────────────────────────────────────────
prs = Presentation("/home/lam23005/STEM-GNN/research_framework.pptx")
sl  = prs.slides[5]

# Remove all existing shapes
sp_tree = sl.shapes._spTree
for sp in list(sl.shapes):
    sp_tree.remove(sp._element)

# Background
fill = sl.background.fill
fill.solid(); fill.fore_color.rgb = DARK

# Bars
bx(sl, 0, 0, W, Inches(0.07), c=BLUE)
bx(sl, 0, H-Inches(0.07), W, Inches(0.07), c=BLUE)

# Title
tx(sl, "Experimental Results  ·  Trust Gate vs. Baselines",
   Inches(0.55), Inches(0.15), Inches(12.5), Inches(0.58),
   sz=26, bold=True)
tx(sl, "Cora (injection sweep, 3 seeds)  ·  Texas WebKB (heterophily, 3 splits)  ·  200 epochs each",
   Inches(0.55), Inches(0.73), Inches(12.5), Inches(0.35),
   sz=13, col=MUTED)
ln(sl, Inches(0.55), Inches(1.12), Inches(12.78), Inches(1.12), c=BLUE, pt=0.9)

# ── Embed the figure ──────────────────────────────────────────────────────
sl.shapes.add_picture(
    "/home/lam23005/STEM-GNN/research_figure.png",
    Inches(0.55), Inches(1.2), Inches(8.8), Inches(3.38),
)

# ── Numbers table (right side) ────────────────────────────────────────────
tbl_l = Inches(9.55); tbl_t = Inches(1.2); tbl_w = Inches(3.6)

bd(sl, tbl_l, tbl_t, tbl_w, Inches(3.38),
   c=RGBColor(0x10,0x14,0x20), bc=RGBColor(0x35,0x3C,0x50), bp=0.8)

tx(sl, "Key Numbers", tbl_l+Inches(0.15), tbl_t+Inches(0.12),
   tbl_w-Inches(0.2), Inches(0.35), sz=13, bold=True, col=WHITE)

# Cora sub-header
bx(sl, tbl_l+Inches(0.1), tbl_t+Inches(0.52), tbl_w-Inches(0.2), Inches(0.28),
   c=RGBColor(0x16,0x1C,0x2C))
tx(sl, "Cora  ·  Clean → 50% injection", tbl_l+Inches(0.18), tbl_t+Inches(0.54),
   tbl_w-Inches(0.25), Inches(0.25), sz=10, bold=True, col=BLUE)

cora_rows = [
    ("GraphSAGE", "78.7 → 71.8", "-6.9 pp", MUTED,   False),
    ("A  Cosine", "79.6 → 71.2", "-8.4 pp", RED,     False),
    ("B  Conf.",  "71.3 → 67.8", "-3.6 pp", ORANGE,  False),
    ("C  Learned","71.6 → 65.7", "-5.9 pp", PURPLE,  False),
    ("D  LLM",    "79.6 → 71.2", "-8.4 pp", TEAL,    True),
]
rh = Inches(0.32)
for ri, (name, vals, drop, col, ours) in enumerate(cora_rows):
    rt = tbl_t + Inches(0.84) + ri * rh
    rf = RGBColor(0x1A,0x26,0x3A) if ours else (CARD if ri%2==0 else RGBColor(0x18,0x1C,0x28))
    bx(sl, tbl_l+Inches(0.1), rt, tbl_w-Inches(0.2), rh, c=rf)
    tx(sl, name, tbl_l+Inches(0.18), rt+Inches(0.04), Inches(1.0), rh,
       sz=10, bold=ours, col=col)
    tx(sl, vals, tbl_l+Inches(1.22), rt+Inches(0.04), Inches(1.35), rh,
       sz=10, col=WHITE, align=PP_ALIGN.CENTER)
    dc = RED if float(drop.split()[0]) < -7 else (GREEN if float(drop.split()[0]) > -4 else MUTED)
    tx(sl, drop, tbl_l+Inches(2.6), rt+Inches(0.04), Inches(0.85), rh,
       sz=10, bold=True, col=dc, align=PP_ALIGN.CENTER)

# Texas sub-header
texas_start = tbl_t + Inches(0.84) + 5*rh + Inches(0.08)
bx(sl, tbl_l+Inches(0.1), texas_start, tbl_w-Inches(0.2), Inches(0.28),
   c=RGBColor(0x16,0x1C,0x2C))
tx(sl, "Texas  ·  Heterophily (h=0.31)", tbl_l+Inches(0.18), texas_start+Inches(0.02),
   tbl_w-Inches(0.25), Inches(0.25), sz=10, bold=True, col=ORANGE)

texas_rows = [
    ("GraphSAGE", "76.6%", MUTED,   False),
    ("A  Cosine", "58.6%", RED,     False),
    ("B  Conf.",  "81.1%", ORANGE,  False),
    ("C  Learned","81.1%", PURPLE,  False),
    ("D  LLM",    "60.4%", TEAL,    True),
]
for ri, (name, acc, col, ours) in enumerate(texas_rows):
    rt = texas_start + Inches(0.3) + ri * rh
    rf = RGBColor(0x1A,0x26,0x3A) if ours else (CARD if ri%2==0 else RGBColor(0x18,0x1C,0x28))
    bx(sl, tbl_l+Inches(0.1), rt, tbl_w-Inches(0.2), rh, c=rf)
    tx(sl, name, tbl_l+Inches(0.18), rt+Inches(0.04), Inches(1.4), rh,
       sz=10, bold=ours, col=col)
    ac = GREEN if float(acc[:-1]) >= 80 else (RED if float(acc[:-1]) < 62 else MUTED)
    tx(sl, acc, tbl_l+Inches(2.6), rt+Inches(0.04), Inches(0.85), rh,
       sz=10, bold=True, col=ac, align=PP_ALIGN.CENTER)

# ── Bottom takeaway panel ─────────────────────────────────────────────────
tk_t = Inches(4.78)
bd(sl, Inches(0.55), tk_t, Inches(12.25), Inches(2.5),
   c=RGBColor(0x10,0x14,0x20), bc=RGBColor(0x35,0x3C,0x50), bp=0.8)

tx(sl, "What the results show", Inches(0.72), tk_t+Inches(0.14),
   Inches(5), Inches(0.38), sz=14, bold=True, col=WHITE)

findings = [
    (ORANGE, "B_confidence is the most injection-robust gate",
             "Only −3.6 pp drop at 50% injection — confidence signal already discounts noisy neighbours."),
    (RED,    "D_llm (text cosine) = A_cosine under injection",
             "Both drop −8.4 pp. Injected edges are chosen to be text-similar → D trusts the wrong neighbours."),
    (GREEN,  "B and C dominate on heterophily (Texas 81.1%)",
             "Outperform standard SAGE (76.6%) and far exceed cosine gates (A: 58.6%, D: 60.4%)."),
    (TEAL,   "The full proposal (D with label-support) should beat both stress conditions",
             "LLM semantic judgment 'is u relevant for class C?' resists injection and captures heterophily."),
]
for fi, (col, title, body) in enumerate(findings):
    ft = tk_t + Inches(0.6) + fi * Inches(0.46)
    tx(sl, "▶", Inches(0.72), ft, Inches(0.28), Inches(0.42), sz=12, bold=True, col=col)
    tx(sl, title + "  —  " + body,
       Inches(1.05), ft, Inches(11.5), Inches(0.42), sz=11.5, col=WHITE)

prs.save("/home/lam23005/STEM-GNN/research_framework.pptx")
print("Updated research_framework.pptx slide 6 with figure + numbers.")
