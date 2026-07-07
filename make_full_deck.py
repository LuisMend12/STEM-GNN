"""
Full 8-slide research deck:
  1 Title & Motivation
  2 Research Problem (formal)
  3 Survey — 6 related methods
  4 Limitations table
  5 Proposed method architecture
  6 Experiment design (tri-objective)
  7 Results (embedded figure)
  8 Conclusion & open questions
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

DARK  = RGBColor(0x12,0x14,0x1A); CARD  = RGBColor(0x1E,0x22,0x2C)
BLUE  = RGBColor(0x45,0x9B,0xFF); GOLD  = RGBColor(0xFF,0xC1,0x07)
GREEN = RGBColor(0x2E,0xCC,0x71); RED   = RGBColor(0xFF,0x4D,0x4D)
ORNG  = RGBColor(0xFF,0x7F,0x2A); PURP  = RGBColor(0xA0,0x6C,0xFF)
MUTED = RGBColor(0x8A,0x93,0xA8); WHITE = RGBColor(0xFF,0xFF,0xFF)
TEAL  = RGBColor(0x45,0xC4,0xFF); LBLUE = RGBColor(0x4A,0x90,0xFF)
DARK2 = RGBColor(0x10,0x14,0x20); BORDER= RGBColor(0x35,0x3C,0x50)
GREY  = RGBColor(0x8A,0x93,0xA8)

W = Inches(13.33); H = Inches(7.5)

def dim(col, factor=2):
    """Darken an RGBColor by integer-dividing each channel."""
    hx = str(col)  # RGBColor.__str__ → "RRGGBB"
    r = int(hx[0:2], 16) // factor
    g = int(hx[2:4], 16) // factor
    b = int(hx[4:6], 16) // factor
    return RGBColor(r, g, b)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]


def new_slide():
    sl = prs.slides.add_slide(BLANK)
    fill = sl.background.fill
    fill.solid(); fill.fore_color.rgb = DARK
    bx(sl, 0, 0, W, Inches(0.065), c=BLUE)
    bx(sl, 0, H-Inches(0.065), W, Inches(0.065), c=BLUE)
    return sl


def bx(sl, l, t, w, h, c=CARD):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = c
    s.line.fill.background(); return s


def bd(sl, l, t, w, h, c=CARD, bc=BLUE, bp=1.0):
    s = sl.shapes.add_shape(1, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = c
    s.line.color.rgb = bc; s.line.width = Pt(bp); return s


def tx(sl, text, l, t, w, h, sz=11, bold=False, col=WHITE,
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


def hdr(sl, title, subtitle=None):
    tx(sl, title,
       Inches(0.55), Inches(0.14), Inches(12.4), Inches(0.6),
       sz=24, bold=True)
    if subtitle:
        tx(sl, subtitle,
           Inches(0.55), Inches(0.75), Inches(12.4), Inches(0.35),
           sz=12, col=MUTED)
    ln(sl, Inches(0.55), Inches(1.13), Inches(12.78), Inches(1.13),
       c=BLUE, pt=0.9)


# ══════════════════════════════════════════════════════════════════════════
# Slide 1 — Title & Motivation
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
bx(sl, 0, Inches(2.6), W, Inches(2.55), c=RGBColor(0x14,0x18,0x24))

tx(sl, "LLM-Guided Semantic Trust Gate",
   Inches(0.65), Inches(1.1), Inches(12.0), Inches(1.0),
   sz=38, bold=True, align=PP_ALIGN.CENTER)
tx(sl, "for Text-Attributed Graph Neural Networks",
   Inches(0.65), Inches(2.05), Inches(12.0), Inches(0.6),
   sz=22, col=BLUE, align=PP_ALIGN.CENTER)

bullets = [
    (ORNG, "Problem",
     "Standard GNNs aggregate all neighbours equally — ignoring whether "
     "a neighbour is semantically relevant for the current node's prediction."),
    (TEAL, "Approach",
     "Use a frozen LLM to assess per-edge semantic reliability and gate "
     "the self vs. neighbour aggregation path accordingly."),
    (GREEN, "Contribution",
     "First formal study of 4 trust-gate signals under 2 stress conditions "
     "(injection attacks + structural heterophily) benchmarked against 6 baselines."),
]
for bi, (col, kw, body) in enumerate(bullets):
    bt = Inches(2.75) + bi * Inches(0.74)
    bx(sl, Inches(0.65), bt, Inches(0.28), Inches(0.55),
       c=dim(col, 2))
    tx(sl, kw, Inches(0.65), bt, Inches(0.28), Inches(0.55),
       sz=10, bold=True, col=col, align=PP_ALIGN.CENTER)
    tx(sl, body, Inches(1.0), bt+Inches(0.05), Inches(11.9), Inches(0.55),
       sz=12.5, col=WHITE)

tx(sl, "Benchmarked on: Cora · Texas WebKB · 6 baselines  "
   "(GraphSAGE, GNNGuard, H2GCN, GPR-GNN, TAPE, STEM-GNN)",
   Inches(0.65), Inches(7.0), Inches(12.0), Inches(0.38),
   sz=10.5, col=MUTED, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════════
# Slide 2 — Research Problem (Formal)
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Research Problem  —  Formal Formulation",
    "When is a GNN neighbour semantically trustworthy?")

# Left: informal motivation
bd(sl, Inches(0.55), Inches(1.25), Inches(5.9), Inches(5.9),
   c=DARK2, bc=BORDER)
tx(sl, "Setup", Inches(0.72), Inches(1.38), Inches(5.5), Inches(0.38),
   sz=13, bold=True, col=BLUE)

setup_lines = [
    ("G = (V, E, T)", WHITE),
    ("  V: nodes,  E: edges,  T = {t_v}: text attributes", MUTED),
    ("", WHITE),
    ("φ : t_v → e_v ∈ ℝ^768", WHITE),
    ("  frozen LLM encoder (SentenceTransformer)", MUTED),
    ("", WHITE),
    ("Standard GNN aggregation:", WHITE),
    ("  h_v = σ( W · MEAN({h_u : u ∈ N(v) ∪ {v}}) )", TEAL),
    ("  treats all neighbours equally", MUTED),
    ("", WHITE),
    ("Problem: noisy/wrong-class neighbours corrupt h_v", RED),
    ("— more so when edges are structurally forced (hetero-", RED),
    ("  phily) or adversarially injected (text-similar, wrong class).", RED),
]
for li, (line, col) in enumerate(setup_lines):
    tx(sl, line, Inches(0.72), Inches(1.82) + li*Inches(0.295),
       Inches(5.5), Inches(0.32), sz=10.8, col=col,
       italic=("frozen" in line or "treats" in line))

# Right: our formulation
bd(sl, Inches(6.75), Inches(1.25), Inches(6.0), Inches(5.9),
   c=DARK2, bc=BLUE, bp=1.2)
tx(sl, "Our Formulation", Inches(6.92), Inches(1.38), Inches(5.7), Inches(0.38),
   sz=13, bold=True, col=BLUE)

form_lines = [
    ("Goal: learn per-node trust gate β_v ∈ [0, 1]", WHITE),
    ("", WHITE),
    ("h_v  =  β_v · W_self(e_v)", TEAL),
    ("    + (1−β_v) · W_neigh( MEAN({e_u : u∈N(v)}) )", TEAL),
    ("", WHITE),
    ("where β_v is driven by LLM semantic assessment:", WHITE),
    ("  Q: 'Is neighbour u relevant for v's class label?'", GOLD),
    ("", WHITE),
    ("4 gate signals studied:", WHITE),
    ("  A  cos( h_v, mean(h_u) )    [GNN hidden states]", RED),
    ("  B  1 − max softmax(logits)  [model confidence]", ORNG),
    ("  C  MLP(e_v ‖ mean(e_u))     [learned gate]", PURP),
    ("  D  cos( e_v, mean(e_u) )    [LLM text cosine]", TEAL),
    ("", WHITE),
    ("Proposed: replace D with LLM label-support score", GREEN),
    ("  so β_v uses semantic reasoning, not just cosine.", GREEN),
]
for li, (line, col) in enumerate(form_lines):
    tx(sl, line, Inches(6.92), Inches(1.82) + li*Inches(0.283),
       Inches(5.7), Inches(0.32), sz=10.5, col=col)

# ══════════════════════════════════════════════════════════════════════════
# Slide 3 — Survey: 6 Related Methods
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Survey  —  6 Related Methods",
    "Prior work across robustness, heterophily, and text-attributed graphs")

cols_w = [Inches(2.4), Inches(1.5), Inches(1.5), Inches(1.5), Inches(4.5)]
col_x  = [Inches(0.45), Inches(2.9), Inches(4.45), Inches(5.95), Inches(7.5)]
hdrs   = ["Method (venue)", "Text use", "Structure", "Agg. weight", "Core idea"]
hdr_c  = [BLUE, BLUE, BLUE, BLUE, BLUE]

# Header row
bx(sl, Inches(0.45), Inches(1.25), Inches(12.4), Inches(0.42),
   c=RGBColor(0x14,0x1C,0x32))
for cw, cx, h, hc in zip(cols_w, col_x, hdrs, hdr_c):
    tx(sl, h, cx+Inches(0.05), Inches(1.28), cw, Inches(0.38),
       sz=10.5, bold=True, col=hc)

rows = [
    ("GraphSAGE\nNeurIPS '17",  GREY,
     "None", "Mean agg", "Equal",
     "Inductive GNN with mean/max/LSTM aggregation over sampled neighbours."),
    ("GNNGuard\nNeurIPS '20",   RED,
     "None", "Cosine prune", "Edge-level",
     "Computes cosine similarity between adjacent node features; "
     "drops or down-weights dissimilar edges before aggregation."),
    ("H2GCN\nNeurIPS '20",      GREEN,
     "None", "Ego-sep", "Node-level",
     "Separates ego (self) and neighbour channels; combines multi-hop "
     "neighbourhoods explicitly for heterophilic graphs."),
    ("GPR-GNN\nICLR '21",       GOLD,
     "None", "PageRank", "Layer-level",
     "Learnable generalized PageRank coefficients per propagation step; "
     "adapts how much each hop contributes to the final representation."),
    ("TAPE\nICLR '24",          LBLUE,
     "LLM feat", "Standard", "Equal",
     "Uses LLM-generated text explanations as node features; "
     "plugs into any GNN backbone with equal-weight aggregation."),
    ("STEM-GNN\nKDD '26",       ORNG,
     "VQ tokens", "MoE route", "Expert",
     "MoE encoder + VQ codebook + Lipschitz regularisation; "
     "routes per node to specialised experts for OOD robustness."),
]

for ri, (name, nc, tu, st, aw, idea) in enumerate(rows):
    ry = Inches(1.70) + ri * Inches(0.87)
    bg = DARK2 if ri % 2 == 0 else CARD
    bx(sl, Inches(0.45), ry, Inches(12.4), Inches(0.84), c=bg)
    bx(sl, Inches(0.45), ry, Inches(0.055), Inches(0.84),
       c=nc)
    tx(sl, name,  col_x[0]+Inches(0.08), ry+Inches(0.05), cols_w[0], Inches(0.8),
       sz=10, bold=True, col=nc)
    for ci, val in enumerate([tu, st, aw]):
        tx(sl, val, col_x[ci+1]+Inches(0.05), ry+Inches(0.19),
           cols_w[ci+1], Inches(0.5), sz=10, col=MUTED)
    tx(sl, idea, col_x[4]+Inches(0.05), ry+Inches(0.05),
       cols_w[4]-Inches(0.1), Inches(0.8), sz=9.8, col=WHITE)

# ══════════════════════════════════════════════════════════════════════════
# Slide 4 — Limitations of Existing Methods
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Limitations  —  Where Existing Methods Fall Short",
    "No single prior method handles both injection attacks and structural heterophily with text semantics")

lim_cols_w = [Inches(2.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(5.4)]
lim_col_x  = [Inches(0.45), Inches(2.8), Inches(4.15), Inches(5.5), Inches(6.85)]
lim_hdrs   = ["Method", "Injection↓", "Heterophily↑", "Text↑", "Key limitation"]

bx(sl, Inches(0.45), Inches(1.25), Inches(12.4), Inches(0.42),
   c=RGBColor(0x14,0x1C,0x32))
for cw, cx, h in zip(lim_cols_w, lim_col_x, lim_hdrs):
    tx(sl, h, cx+Inches(0.05), Inches(1.28), cw, Inches(0.38),
       sz=10.5, bold=True, col=BLUE)

def check(v):
    if v == "yes":   return ("✓", GREEN)
    if v == "no":    return ("✗", RED)
    if v == "part":  return ("~", GOLD)
    return (v, MUTED)

lim_rows = [
    ("GraphSAGE",  GREY,  "no",   "no",   "no",
     "Equal-weight mean agg; no mechanism to down-weight "
     "misleading neighbours under injection or heterophily."),
    ("GNNGuard",   RED,   "part", "no",   "no",
     "Cosine prune helps under feature noise but fails when injected edges "
     "are deliberately text-similar to the victim — exactly our attack."),
    ("H2GCN",      GREEN, "no",   "yes",  "no",
     "Strong on heterophily by ego-separation, but no robustness mechanism; "
     "injection degrades performance as severely as SAGE."),
    ("GPR-GNN",    GOLD,  "no",   "yes",  "no",
     "Learnable hop weights adapt globally, not per node. "
     "Cannot identify which specific neighbours are unreliable."),
    ("TAPE",       LBLUE, "no",   "no",   "part",
     "Rich text features but equal-weight aggregation: LLM knowledge "
     "is in the features, not in the routing — still vulnerable to both."),
    ("STEM-GNN",   ORNG,  "part", "part", "part",
     "Router uses post-aggregation z_v (already corrupted by noisy "
     "neighbours). No upstream semantic neighbour assessment."),
    ("Ours (goal)",TEAL,  "yes",  "yes",  "yes",
     "LLM judges per-edge semantic relevance BEFORE aggregation. "
     "Gate β_v adapts to injection and heterophily via text reasoning."),
]

for ri, (name, nc, inj, het, txt, lim) in enumerate(lim_rows):
    ry = Inches(1.70) + ri * Inches(0.78)
    bg = RGBColor(0x12,0x1E,0x14) if name=="Ours (goal)" else (DARK2 if ri%2==0 else CARD)
    bx(sl, Inches(0.45), ry, Inches(12.4), Inches(0.75), c=bg)
    bx(sl, Inches(0.45), ry, Inches(0.05), Inches(0.75), c=nc)
    tx(sl, name, lim_col_x[0]+Inches(0.07), ry+Inches(0.12),
       lim_cols_w[0], Inches(0.5), sz=10.5, bold=(name=="Ours (goal)"), col=nc)
    for ci, val in enumerate([inj, het, txt]):
        sym, sc = check(val)
        tx(sl, sym, lim_col_x[ci+1]+Inches(0.35), ry+Inches(0.12),
           Inches(0.6), Inches(0.5), sz=14, bold=True, col=sc,
           align=PP_ALIGN.CENTER)
    tx(sl, lim, lim_col_x[4]+Inches(0.05), ry+Inches(0.08),
       lim_cols_w[4]-Inches(0.1), Inches(0.62), sz=9.5, col=WHITE)

# ══════════════════════════════════════════════════════════════════════════
# Slide 5 — Proposed Method Architecture
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Proposed Method  —  Trust-Gated Aggregation",
    "Per-node gate β_v driven by LLM semantic reliability judgment")

# Architecture boxes — left to right pipeline
def arch_box(sl, l, t, w, h, title, body, col):
    bd(sl, l, t, w, h, c=DARK2, bc=col, bp=1.5)
    tx(sl, title, l+Inches(0.12), t+Inches(0.1), w-Inches(0.2),
       Inches(0.38), sz=12, bold=True, col=col)
    tx(sl, body,  l+Inches(0.12), t+Inches(0.52), w-Inches(0.2),
       h-Inches(0.6), sz=9.8, col=WHITE)

arch_items = [
    (Inches(0.45), "Text\nAttributes",
     "t_v for each node v\n(raw sentences,\nabstracts, etc.)", GREY),
    (Inches(3.05), "LLM Encoder\nφ (frozen)",
     "e_v = φ(t_v) ∈ ℝ^768\nSentenceTransformer\nor any encoder", BLUE),
    (Inches(5.65), "Gate Signal\nβ_v",
     "Query LLM:\n'Is u semantically\nrelevant for v?'\nβ_v ∈ [0, 1]", TEAL),
    (Inches(8.25), "Gated\nAggregation",
     "h_v = β_v·W_self(e_v)\n+ (1−β_v)·W_neigh\n(mean e_u)", GREEN),
    (Inches(10.85),"Downstream\nTask",
     "Node classification,\nlink prediction,\ngraph classification", GOLD),
]
boxy = Inches(1.35)
for l, title, body, col in arch_items:
    arch_box(sl, l, boxy, Inches(2.45), Inches(2.2), title, body, col)
    if l < Inches(10.85):
        tx(sl, "→", l+Inches(2.5), boxy+Inches(0.8), Inches(0.45), Inches(0.6),
           sz=22, bold=True, col=MUTED, align=PP_ALIGN.CENTER)

# Equation panel
bd(sl, Inches(0.45), Inches(3.8), Inches(12.4), Inches(1.55),
   c=DARK2, bc=BLUE, bp=0.8)
tx(sl, "Formal update rule:",
   Inches(0.65), Inches(3.92), Inches(4.0), Inches(0.38),
   sz=12, bold=True, col=BLUE)
tx(sl,
   "h_v  =  β_v · W_self(e_v)  +  (1−β_v) · W_neigh( MEAN({e_u : u ∈ N(v)}) )",
   Inches(0.65), Inches(4.35), Inches(12.0), Inches(0.45),
   sz=14, bold=True, col=TEAL, align=PP_ALIGN.CENTER)
tx(sl,
   "β_v close to 1  →  trust self (ignore noisy neighbours)     "
   "β_v close to 0  →  trust neighbourhood (strong structural signal)",
   Inches(0.65), Inches(4.85), Inches(12.0), Inches(0.38),
   sz=10.5, col=MUTED, align=PP_ALIGN.CENTER)

# Comparison table bottom
bd(sl, Inches(0.45), Inches(5.55), Inches(12.4), Inches(1.72),
   c=DARK2, bc=BORDER, bp=0.7)
tx(sl, "What β_v is in each variant:",
   Inches(0.65), Inches(5.65), Inches(5.0), Inches(0.38),
   sz=11.5, bold=True, col=WHITE)
variants = [
    ("A  Cosine",      "1 − σ( cos(h_v, mean h_u) )",                  RED),
    ("B  Confidence",  "1 − mean[ max softmax(logits_u) ]",             ORNG),
    ("C  Learned",     "MLP(e_v ‖ mean e_u).sigmoid()",                 PURP),
    ("D  LLM cosine",  "1 − σ( cos(e_v, mean e_u) )   ← current impl", TEAL),
    ("Proposed",       "LLM(Q: 'is u relevant for v?') → β_v",         GREEN),
]
for vi, (name, eq, col) in enumerate(variants):
    vt = Inches(6.05) + vi * Inches(0.31)
    tx(sl, f"  {name:<18} {eq}", Inches(0.65), vt, Inches(12.0), Inches(0.32),
       sz=10, col=col)

# ══════════════════════════════════════════════════════════════════════════
# Slide 6 — Experiment Design (Tri-Objective Framework)
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Experiment Design  —  Tri-Objective Framework",
    "Following STEM-GNN's evaluation protocol: ID performance · OOD robustness · Cross-dataset transfer")

obj_items = [
    (Inches(0.45),  BLUE,  "Objective 1\nID Performance",
     "Standard train/val/test splits.\n"
     "Measure clean test accuracy on Cora, PubMed, "
     "ArXiv (OFA format), Texas, Wisconsin.\n"
     "Primary baseline: STEM-GNN published numbers."),
    (Inches(4.65),  RED,   "Objective 2\nOOD Robustness",
     "Stress test 1 — Confusing-edge injection:\n"
     "Add k% edges connecting text-similar, wrong-class node pairs.\n"
     "Ratios: 0%, 10%, 20%, 30%, 40%, 50%.\n\n"
     "Stress test 2 — Structural heterophily:\n"
     "Texas (h=0.31), Wisconsin (h=0.20) WebKB datasets.\n"
     "Metric: accuracy × 3 splits."),
    (Inches(8.85),  GREEN, "Objective 3\nTransfer / Scale",
     "Pretrain trust gate on Cora;\n"
     "zero-shot evaluate on PubMed & ArXiv.\n"
     "Measure how well learned gate β_v\n"
     "transfers across domains without\n"
     "retraining the gate signal."),
]

for l, col, title, body in obj_items:
    bd(sl, l, Inches(1.28), Inches(3.95), Inches(4.3),
       c=DARK2, bc=col, bp=1.8)
    bx(sl, l, Inches(1.28), Inches(3.95), Inches(0.62),
       c=dim(col, 3))
    tx(sl, title, l+Inches(0.15), Inches(1.3), Inches(3.7), Inches(0.6),
       sz=13, bold=True, col=col)
    tx(sl, body, l+Inches(0.15), Inches(1.95), Inches(3.75), Inches(3.5),
       sz=10.8, col=WHITE)

# Datasets row
bd(sl, Inches(0.45), Inches(5.75), Inches(12.4), Inches(1.55),
   c=DARK2, bc=BORDER, bp=0.7)
tx(sl, "Datasets & Splits",
   Inches(0.65), Inches(5.85), Inches(3.5), Inches(0.38),
   sz=12, bold=True, col=WHITE)

ds_items = [
    ("Cora",       "2708 nodes · 5429 edges · 7 classes\n3 seeds · OFA splits",     BLUE),
    ("Texas",      "183 nodes · 295 edges · 5 classes\n10 official splits",          RED),
    ("Wisconsin",  "251 nodes · 466 edges · 5 classes\n10 official splits",          ORNG),
    ("PubMed",     "19717 nodes · 44324 edges · 3 classes\nOFA format · 3 seeds",   GREEN),
    ("STEM-GNN\n  baselines",
     "Clean avg 80.79%\nOOD-low Cora 82.73%\nOOD-worst 78.45%",                      GOLD),
]
for di, (ds, info, col) in enumerate(ds_items):
    dl = Inches(0.65) + di * Inches(2.4)
    bx(sl, dl, Inches(6.28), Inches(0.05), Inches(0.95), c=col)
    tx(sl, ds, dl+Inches(0.12), Inches(6.28), Inches(2.25), Inches(0.38),
       sz=11, bold=True, col=col)
    tx(sl, info, dl+Inches(0.12), Inches(6.68), Inches(2.25), Inches(0.55),
       sz=9, col=MUTED)

# ══════════════════════════════════════════════════════════════════════════
# Slide 7 — Results
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Results  —  Trust Gate vs. 6 Baselines",
    "Cora injection sweep (3 seeds, 200 ep)  ·  Texas WebKB heterophily (3 splits, 200 ep)")

sl.shapes.add_picture(
    "/home/lam23005/STEM-GNN/research_figure_v2.png",
    Inches(0.4), Inches(1.25), Inches(12.55), Inches(4.35))

# Bottom panel
bd(sl, Inches(0.4), Inches(5.75), Inches(12.55), Inches(1.55),
   c=DARK2, bc=BORDER, bp=0.7)
findings = [
    (ORNG, "B_confidence most injection-robust",
     "−3.6 pp drop at 50% injection (vs −8.4 pp for cosine gates A and D)."),
    (RED,  "D_llm = A_cosine under injection",
     "Both drop −8.4 pp — text cosine trusts wrong-class text-similar injected edges."),
    (GREEN,"B and C dominate on heterophily",
     "81.1% on Texas vs 76.6% for SAGE; cosine gates A (58.6%) and D (60.4%) hurt."),
    (TEAL, "Gap motivates full proposal",
     "No single gate wins both. LLM label-support reasoning expected to handle both."),
]
for fi, (col, kw, body) in enumerate(findings):
    ft = Inches(5.85) + fi * Inches(0.36)
    tx(sl, "▶", Inches(0.6), ft, Inches(0.28), Inches(0.34),
       sz=10, bold=True, col=col)
    tx(sl, f"{kw}  —  {body}", Inches(0.92), ft, Inches(11.9), Inches(0.36),
       sz=10.5, col=WHITE)

# ══════════════════════════════════════════════════════════════════════════
# Slide 8 — Conclusion & Open Questions
# ══════════════════════════════════════════════════════════════════════════
sl = new_slide()
hdr(sl, "Conclusion  &  Open Questions",
    "What the experiments confirm and what remains to build")

# Left: what we showed
bd(sl, Inches(0.45), Inches(1.28), Inches(5.9), Inches(5.95),
   c=DARK2, bc=GREEN, bp=1.2)
tx(sl, "What this work shows", Inches(0.65), Inches(1.38), Inches(5.5), Inches(0.38),
   sz=13, bold=True, col=GREEN)
shown = [
    "No single existing gate signal handles both stress conditions.",
    "Cosine-based gates (A, D) are vulnerable to injection by design — "
    "injected edges are chosen to be text-similar, so cosine trusts them.",
    "Confidence and learned gates (B, C) are more injection-robust but "
    "lack direct text understanding for heterophilic settings.",
    "Text cosine (D) ≠ semantic label-support — a node can be text-similar "
    "but irrelevant or misleading for the classification task.",
    "The full LLM label-support gate is the natural next step: robust to "
    "injection (semantic judgment) and heterophily (class-aware).",
    "Experiment design follows STEM-GNN's tri-objective framework, "
    "enabling direct comparison to KDD '26 results.",
]
for li, line in enumerate(shown):
    tx(sl, f"• {line}", Inches(0.65), Inches(1.85) + li*Inches(0.84),
       Inches(5.55), Inches(0.82), sz=10.8, col=WHITE)

# Right: open questions
bd(sl, Inches(6.6), Inches(1.28), Inches(6.15), Inches(5.95),
   c=DARK2, bc=GOLD, bp=1.2)
tx(sl, "Open Questions  /  Next Steps", Inches(6.8), Inches(1.38),
   Inches(5.85), Inches(0.38), sz=13, bold=True, col=GOLD)
questions = [
    ("Q1", "Can an LLM generate reliable per-edge trust scores at scale "
           "without fine-tuning?", GOLD),
    ("Q2", "How to handle the label leakage risk when LLM uses class-label "
           "text in the trust query?", RED),
    ("Q3", "Does the gate β_v transfer zero-shot to PubMed / ArXiv "
           "without retraining?", ORNG),
    ("Q4", "Can the gate be distilled into a lightweight MLP for inference-time "
           "efficiency?", BLUE),
    ("Q5", "How does the trust gate interact with STEM-GNN's VQ codebook? "
           "Can they be jointly trained?", TEAL),
    ("Q6", "Wisconsin results pending (HTTP download error). Will heterophily "
           "trend hold at h=0.20?", PURP),
]
for qi, (num, body, col) in enumerate(questions):
    qt = Inches(1.90) + qi * Inches(0.82)
    bx(sl, Inches(6.6), qt, Inches(0.52), Inches(0.72),
       c=dim(col, 3))
    tx(sl, num, Inches(6.6), qt, Inches(0.52), Inches(0.72),
       sz=10.5, bold=True, col=col, align=PP_ALIGN.CENTER)
    tx(sl, body, Inches(7.18), qt+Inches(0.1), Inches(5.4), Inches(0.62),
       sz=10.5, col=WHITE)

prs.save("/home/lam23005/STEM-GNN/full_research_deck.pptx")
print("Saved → full_research_deck.pptx")
