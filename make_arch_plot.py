"""
Architecture diagram — fixed spacing, no overlaps.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

BG="#FFFFFF"; FRAME="#2D3748"; TEXT="#1A202C"; MUT="#718096"; MGREY="#CBD5E0"
H1="#2B6CB0"; H2="#276749"; H3="#553C9A"; H4="#9C4221"
CB1=("#BEE3F8","#2B6CB0"); CB2=("#C6F6D5","#276749")
CB3=("#D6BCFA","#553C9A"); CB4=("#FEEBC8","#C05621")
CB5=("#FED7D7","#C53030"); CB6=("#E2E8F0","#4A5568")

fig, ax = plt.subplots(figsize=(19, 9.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0,19); ax.set_ylim(0,9.0)
ax.axis("off")

def sec(x,y,w,h,hdr,hc,fc="#F7FAFC"):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor=FRAME,lw=1.8,zorder=2))
    ax.add_patch(Rectangle((x,y+h-0.66),w,0.66,facecolor=hc,edgecolor=hc,lw=0,zorder=3))
    ax.text(x+w/2,y+h-0.33,hdr,ha="center",va="center",fontsize=11,
            fontweight="bold",color="white",zorder=4)

def cbox(x,y,w,h,txt,sub="",fc="#EBF8FF",ec="#2B6CB0",tsz=10,ssz=8.5,tc=TEXT):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle="round,pad=0.07,rounding_size=0.12",
        facecolor=fc,edgecolor=ec,lw=1.3,zorder=4))
    ty = y+h/2+(0.14 if sub else 0)
    ax.text(x+w/2,ty,txt,ha="center",va="center",fontsize=tsz,
            fontweight="bold",color=tc,zorder=5)
    if sub:
        ax.text(x+w/2,y+h/2-0.21,sub,ha="center",va="center",
                fontsize=ssz,color=MUT,style="italic",zorder=5)

def rect(x,y,w,h,fc,ec,lw=1.0):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,lw=lw,zorder=3))

def arr(x1,y1,x2,y2,col=FRAME,lw=1.8,lbl="",ldy=0.18,ldx=0):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,
                        mutation_scale=15,connectionstyle="arc3,rad=0.0"),zorder=6)
    if lbl:
        ax.text((x1+x2)/2+ldx,(y1+y2)/2+ldy,lbl,ha="center",
                fontsize=9,color=col,style="italic",zorder=7)

def t(x,y,s,col=MUT,sz=9,ha="center",bold=False):
    ax.text(x,y,s,ha=ha,va="center",fontsize=sz,
            color=col,fontweight="bold" if bold else "normal",zorder=5)

# ── global ─────────────────────────────────────────────────────────────────
SY=0.55; SH=7.80
MID=SY+SH/2        # 4.45
S1X,S1W = 0.15, 3.05
S2X,S2W = 3.40, 3.30
S3X,S3W = 6.90, 6.10
S4X,S4W = 13.20, 5.65

t(9.4,8.78,"Framework of Trust Gate  —  Complement to STEM-GNN",
  col=TEXT,sz=15,bold=True)

# ══════════════════════════════════════════════════════════════════════════
# S1  Input Graph
# ══════════════════════════════════════════════════════════════════════════
sec(S1X,SY,S1W,SH,"Input Graph",H1,"#EBF8FF")

nX=[0.82,1.52,2.42,1.76,1.06]; nY=[6.25,7.05,6.50,5.45,4.95]
for i,j in [(0,1),(0,4),(1,2),(1,3),(2,3),(3,4)]:
    ax.plot([nX[i],nX[j]],[nY[i],nY[j]],color=MGREY,lw=1.3,zorder=3)
for k,(nx,ny) in enumerate(zip(nX,nY)):
    ax.scatter(nx,ny,s=180,color=H1 if k==0 else "#63B3ED",
               zorder=5,edgecolors=FRAME,lw=0.9)
    lbl="v" if k==0 else "u"
    off=(-0.25,0) if k==0 else (0.22,0)
    ax.text(nx+off[0],ny+off[1],lbl,ha="center",va="center",fontsize=9.5,
            fontweight="bold",color=H1 if k==0 else MUT,zorder=6)

cbox(S1X+0.12,3.95,1.32,0.70,"Node  v","text  t_v",*CB1,tsz=9.5)
cbox(S1X+1.60,3.95,1.32,0.70,"Nbrs  N(v)","{t_u}",*CB6,tsz=9.5)

rect(S1X+0.12,1.72,S1W-0.24,1.88,"#FFF5F5","#FC8181",lw=1.0)
t(S1X+S1W/2,3.38,"OOD Stress Conditions",col="#C53030",sz=9.5,bold=True)
for i,(s,c) in enumerate([
    ("① Confusing-edge injection","#C53030"),
    ("   text-similar, wrong-class","#C53030"),
    ("② Structural heterophily","#C05621"),
    ("   Texas h=0.31 / Wisc h=0.20","#C05621"),
]):
    t(S1X+0.22,3.02-i*0.34,s,col=c,sz=8.5,ha="left")

arr(S1X+S1W,MID,S2X,MID)

# ══════════════════════════════════════════════════════════════════════════
# S2  Frozen Text Encoder
# ══════════════════════════════════════════════════════════════════════════
sec(S2X,SY,S2W,SH,"Frozen Text Encoder",H2,"#F0FFF4")

cbox(S2X+0.18,5.55,S2W-0.36,1.72,"LLM Encoder   φ",
     "SentenceTransformer  (frozen)",*CB2,tsz=12,ssz=9.5)

t(S2X+0.28,5.22,"t_v  →",col=H2,sz=10.5,bold=True,ha="left")
t(S2X+1.28,5.22,"e_v  ∈  ℝ^768",col=H2,sz=10,bold=True,ha="left")
t(S2X+0.28,4.80,"{t_u}  →",col="#4A5568",sz=10.5,bold=True,ha="left")
t(S2X+1.28,4.80,"{e_u}  ∈  ℝ^768",col="#4A5568",sz=10,bold=True,ha="left")

t(S2X+S2W/2,4.40,"Key Properties",col=TEXT,sz=10,bold=True)
for i,p in enumerate([
    "• Shared encoder for  v  and  u",
    "• Fully frozen — no fine-tuning",
    "• 768-dim semantic embeddings",
    "• Operates PRE-VQ codebook",
    "• Bypasses quantisation noise",
    "• Drop-in to any STEM-GNN run",
]):
    t(S2X+0.22,4.08-i*0.39,p,col=MUT,sz=9,ha="left")

arr(S2X+S2W,MID,S3X,MID)

# ══════════════════════════════════════════════════════════════════════════
# S3  Semantic Trust Gate
# ══════════════════════════════════════════════════════════════════════════
sec(S3X,SY,S3W,SH,"Semantic Trust Gate",H3,"#FAF5FF")

# ── Gating Network sub-panel (y=5.30 to header at 7.69) ──
rect(S3X+0.20,5.30,S3W-0.40,2.04,"#EDE9FE",H3,lw=1.2)
t(S3X+S3W/2,7.10,"Gating Network  (Router)",col=H3,sz=10.5,bold=True)

cbox(S3X+0.32,5.76,S3W-0.64,0.84,
     'Q : "Is neighbour  u  semantically relevant for  v\'s  label?"',
     "",*CB3,tsz=10.5)
cbox(S3X+0.32,5.34,S3W-0.64,0.38,
     "β_v  =  σ ( f ( e_v ,  mean { e_u } ) )",
     "","#EDE9FE",H3,tsz=11.5,tc=H3)

# ── Gate Signal Variants label (y=5.14) ──
t(S3X+S3W/2,5.15,"Gate Signal Variants",col=TEXT,sz=10.5,bold=True)

# ── Row 1: A, B  (y=4.12 to 5.05) ──
GW=(S3W-0.56)/2; GH=0.90
row1 = [
    ("A   Cosine",       "cos( h_v ,  mean h_u )",         CB5),
    ("B   Confidence",   "1 − max softmax( logits_u )",     CB4),
]
for gi,(name,formula,(fc,ec)) in enumerate(row1):
    gx=S3X+0.18+gi*(GW+0.20)
    cbox(gx,4.12,GW,GH,name,formula,fc,ec,tsz=9.5,ssz=8.5)

# ── Row 2: C, D  (y=3.06 to 3.99) ──
row2 = [
    ("C   Learned MLP",  "MLP( e_v  ‖  mean e_u ).σ(·)",   CB3),
    ("D   LLM cosine ✱", "cos( e_v ,  mean e_u )",         CB1),
]
for gi,(name,formula,(fc,ec)) in enumerate(row2):
    gx=S3X+0.18+gi*(GW+0.20)
    cbox(gx,3.06,GW,GH,name,formula,fc,ec,tsz=9.5,ssz=8.5)

# ── Weighted Gate Output banner (y=1.88 to 2.88) — clear of row 2 bottom 3.06 ──
rect(S3X+0.20,1.88,S3W-0.40,1.00,"#E9D8FD",H3,lw=1.0)
t(S3X+S3W/2,2.65,"Weighted Gate Output",col=H3,sz=10.5,bold=True)
t(S3X+S3W/2,2.35,"β_v  ∈  [0, 1]  —  per-node trust scalar",col=H3,sz=11)
t(S3X+S3W/2,2.05,
  "β_v → 1 : trust self  (noisy nbrs)          β_v → 0 : trust neighbours  (reliable)",
  col=MUT,sz=8.8)

# ── footnotes (y=1.62 to 0.68, inside section) ──
t(S3X+0.22,1.65,
  "✱ D uses pre-VQ LLM embeddings.  Full proposal replaces with LLM label-support query.",
  col=MUT,sz=8.3,ha="left")
t(S3X+0.22,1.38,
  "  Baselines: A (GNNGuard), B (Mowst/confidence), C (ACM-GNN), D (TAPE-style).",
  col=MUT,sz=8.3,ha="left")
t(S3X+0.22,1.11,
  "  No existing method handles both injection + heterophily simultaneously.",
  col="#C53030",sz=8.3,ha="left")

arr(S3X+S3W,MID,S4X,MID)

# ══════════════════════════════════════════════════════════════════════════
# S4  Gated Aggregation
# ══════════════════════════════════════════════════════════════════════════
sec(S4X,SY,S4W,SH,"Gated Aggregation",H4,"#FFFAF0")

t(S4X+S4W/2,7.08,"Aggregation Rule",col=H4,sz=10.5,bold=True)

# equation boxes
cbox(S4X+0.22,6.14,S4W-0.44,0.74,"h_v  =  β_v · W_self ( e_v )","",*CB4,tsz=13)
cbox(S4X+0.22,5.24,S4W-0.44,0.74,"+ ( 1 − β_v ) · W_neigh ( mean e_u )","",*CB4,tsz=12)

arr(S4X+S4W/2,5.24,S4X+S4W/2,4.95,col=H4,lw=1.5)

# interpretation
rect(S4X+0.22,3.85,S4W-0.44,1.02,"#FEFEFE",MGREY,lw=0.8)
t(S4X+S4W/2,4.65,"Interpretation",col=TEXT,sz=10,bold=True)
hw=(S4W-0.60)/2
cbox(S4X+0.22,3.88,hw,0.70,"β_v → 1","Noisy / heterophilic\nneighbourhood",*CB5,tsz=9.5,ssz=8)
cbox(S4X+0.22+hw+0.16,3.88,hw,0.70,"β_v → 0","Reliable structural\nsignal",*CB2,tsz=9.5,ssz=8)

arr(S4X+S4W/2,3.85,S4X+S4W/2,3.58,col=H4,lw=1.5)

# bounded effect
rect(S4X+0.22,2.68,S4W-0.44,0.82,"#FEFCBF","#B7791F",lw=1.1)
t(S4X+S4W/2,3.28,"Bounded Effect  (cf. STEM-GNN Lipschitz reg.)",col="#92400E",sz=9.5,bold=True)
t(S4X+S4W/2,2.94,"ℒ  =  ℒ_task  +  λ_lip · ‖ W ‖²_F",col="#92400E",sz=12)

arr(S4X+S4W/2,2.68,S4X+S4W/2,2.42,col=H4,lw=1.5)

# task head
cbox(S4X+0.55,1.68,S4W-1.10,0.68,
     "Downstream Task Head",
     "node classification  /  link prediction",*CB3,tsz=11,ssz=9)

arr(S4X+S4W/2,1.68,S4X+S4W/2,1.46,col=H4,lw=1.5)

# loss labels
lw2=(S4W-1.2)/2
for i,(lt,lc) in enumerate([("ℒ_task","#276749"),("ℒ_lip","#553C9A")]):
    lx=S4X+0.55+i*(lw2+0.10)
    ax.add_patch(FancyBboxPatch((lx,0.80),lw2,0.55,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#FAFAFA",edgecolor=lc,lw=1.2,zorder=4))
    t(lx+lw2/2,1.075,lt,col=lc,sz=11,bold=True)

# ── bottom bar ─────────────────────────────────────────────────────────────
rect(0.15,0.60,18.7,0.46,"#EBF4FF","#90CDF4",lw=0.8)
t(9.4,0.83,
  "Trust Gate taps  e_v  BEFORE  VQ Codebook  ——  clean semantic signal, "
  "bypassing the quantisation noise that corrupts STEM-GNN's post-VQ MoE router",
  col=H1,sz=9.5)

plt.tight_layout(pad=0)
out="/home/lam23005/STEM-GNN/arch_diagram.png"
plt.savefig(out,dpi=160,bbox_inches="tight",facecolor=BG)
print(f"Saved → {out}")
