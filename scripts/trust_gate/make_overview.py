"""
Comprehensive research overview — 6-panel figure.
Row 1: Problem Formulation | Method Survey Table | Mini Architecture
Row 2: Experiment Design   | Injection Results   | Heterophily Results
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np, json

# ── palette ───────────────────────────────────────────────────────────────
BG="#FFFFFF"; FRAME="#2D3748"; TEXT="#1A202C"; MUT="#718096"; MGREY="#CBD5E0"
H1="#2B6CB0"; H2="#276749"; H3="#553C9A"; H4="#9C4221"; H5="#B7791F"
CB1=("#BEE3F8","#2B6CB0"); CB2=("#C6F6D5","#276749")
CB3=("#D6BCFA","#553C9A"); CB4=("#FEEBC8","#C05621")
CB5=("#FED7D7","#C53030"); CB6=("#E2E8F0","#4A5568")
CY =("#B2F5EA","#2C7A7B")

GCOL={"none":"#8A93A8","A_cosine":"#FC8181","B_confidence":"#F6AD55",
      "C_learned":"#B794F4","D_llm":"#4299E1"}
GLBL={"none":"GraphSAGE","A_cosine":"A  Cosine","B_confidence":"B  Confidence",
      "C_learned":"C  Learned","D_llm":"D  LLM cosine"}
GMRK={"none":"o","A_cosine":"s","B_confidence":"^","C_learned":"D","D_llm":"P"}
GLS ={"none":"--","A_cosine":"-","B_confidence":"-","C_learned":"-","D_llm":"-"}

with open("/home/lam23005/STEM-GNN/sweep_results.json") as f:
    d=json.load(f)
cora=d["cora_sweep"]; texas=d["heterophily"]["Texas"]

# ── figure ─────────────────────────────────────────────────────────────────
fig=plt.figure(figsize=(22,13.5),facecolor=BG)
gs=gridspec.GridSpec(2,3,figure=fig,
    left=0.025,right=0.985,top=0.925,bottom=0.04,
    hspace=0.42,wspace=0.26)

ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1])
ax3=fig.add_subplot(gs[0,2])
ax4=fig.add_subplot(gs[1,0])
ax5=fig.add_subplot(gs[1,1])
ax6=fig.add_subplot(gs[1,2])

for ax in [ax1,ax2,ax3,ax4]: ax.axis("off"); ax.set_facecolor(BG)
for ax in [ax5,ax6]:
    ax.set_facecolor("#F7FAFC")
    for sp in ax.spines.values(): sp.set_color(MGREY)

fig.text(0.5,0.962,
    "LLM-Guided Semantic Trust Gate  —  Research Overview",
    ha="center",va="center",fontsize=17,fontweight="bold",color=TEXT)

# ── helpers ────────────────────────────────────────────────────────────────
def phdr(ax,title,col):
    ax.add_patch(Rectangle((0,0.938),1,0.062,transform=ax.transAxes,
        facecolor=col,edgecolor="none",zorder=3,clip_on=False))
    ax.text(0.5,0.969,title,transform=ax.transAxes,ha="center",va="center",
        fontsize=11,fontweight="bold",color="white",zorder=4)
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,
        facecolor="#F7FAFC",edgecolor=FRAME,lw=1.5,zorder=1,clip_on=False))

def tb(ax,x,y,w,h,txt,sub="",fc="#EBF8FF",ec="#2B6CB0",tsz=9,ssz=7.8,tc=TEXT):
    ax.add_patch(FancyBboxPatch((x,y),w,h,transform=ax.transAxes,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=fc,edgecolor=ec,lw=1.1,zorder=4))
    ty=y+h/2+(0.02 if sub else 0)
    ax.text(x+w/2,ty,txt,transform=ax.transAxes,ha="center",va="center",
        fontsize=tsz,fontweight="bold",color=tc,zorder=5)
    if sub:
        ax.text(x+w/2,y+h/2-0.028,sub,transform=ax.transAxes,ha="center",
            va="center",fontsize=ssz,color=MUT,style="italic",zorder=5)

def pt(ax,x,y,s,col=TEXT,sz=9.5,ha="left",bold=False,italic=False):
    ax.text(x,y,s,transform=ax.transAxes,ha=ha,va="top",fontsize=sz,
        color=col,fontweight="bold" if bold else "normal",
        fontstyle="italic" if italic else "normal",zorder=5)

def harr(ax,x1,y,x2,col=FRAME,lw=1.4):
    ax.annotate("",xy=(x2,y),xytext=(x1,y),transform=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,mutation_scale=10),zorder=6)

def varr(ax,x,y1,y2,col=FRAME,lw=1.4):
    ax.annotate("",xy=(x,y2),xytext=(x,y1),transform=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,mutation_scale=10),zorder=6)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Research Problem Formulation
# ══════════════════════════════════════════════════════════════════════════
phdr(ax1,"(a)  Research Problem  —  Formal Formulation",H1)

# Given
pt(ax1,0.05,0.905,"Given:  Text-attributed graph  G = (V, E, T)",col=H1,sz=10,bold=True)
pt(ax1,0.05,0.860,"  V : nodes     E : edges     T = { t_v } : text attributes",col=MUT,sz=9)

tb(ax1,0.05,0.770,0.90,0.075,
   "φ : t_v  →  e_v  ∈  ℝ^768",
   "Frozen LLM encoder (SentenceTransformer)",*CB1,tsz=10,ssz=8.5)

pt(ax1,0.05,0.748,"Standard GNN aggregation (equal-weight):",col=TEXT,sz=9.5,bold=True)
tb(ax1,0.05,0.660,0.90,0.075,
   "h_v  =  σ( W · MEAN( { h_u : u ∈ N(v) ∪ {v} } ) )",
   "treats ALL neighbours equally — ignores semantic reliability",
   "#FED7D7","#C53030",tsz=10,ssz=8.5)

pt(ax1,0.05,0.635,"Problem:  noisy / wrong-class neighbours corrupt h_v",col="#C53030",sz=9.5,bold=True)

# divider
ax1.plot([0.04,0.96],[0.618,0.618],color=MGREY,lw=0.8,transform=ax1.transAxes)

pt(ax1,0.05,0.600,"Proposed trust-gated update:",col=H2,sz=9.5,bold=True)
tb(ax1,0.05,0.505,0.90,0.085,
   "h_v  =  β_v · W_self(e_v)  +  (1−β_v) · W_neigh( mean e_u )",
   "",*CB2,tsz=10.5,tc=H2)
pt(ax1,0.05,0.482,"where  β_v ∈ [0,1]  is driven by LLM semantic reliability",col=MUT,sz=9,italic=True)

# stress conditions
pt(ax1,0.05,0.452,"Two make-or-break stress conditions:",col=TEXT,sz=9.5,bold=True)
for i,(icon,s,c) in enumerate([
    ("①","Confusing-edge injection  —  text-similar, wrong-class edges","#C53030"),
    ("②","Structural heterophily  —  Texas h=0.31,  Wisconsin h=0.20","#C05621"),
]):
    y=0.410-i*0.075
    ax1.add_patch(FancyBboxPatch((0.05,y-0.005),0.90,0.062,transform=ax1.transAxes,
        boxstyle="round,pad=0.01",facecolor="#FFF5F5" if i==0 else "#FFFAF0",
        edgecolor=c,lw=0.9,zorder=3))
    pt(ax1,0.10,y+0.048,f"{icon}  {s}",col=c,sz=9)

# Gate variants
pt(ax1,0.05,0.290,"4 gate signals studied:",col=TEXT,sz=9.5,bold=True)
gates=[("A","Cosine of GNN hidden states",CB5),
       ("B","Model confidence (1−max softmax)",CB4),
       ("C","Learned MLP gate",CB3),
       ("D","LLM text-embedding cosine  ←  current",CB1)]
GW=0.205
for gi,(ltr,desc,(fc,ec)) in enumerate(gates):
    gx=0.05+gi*(GW+0.018)
    ax1.add_patch(FancyBboxPatch((gx,0.175),GW,0.100,transform=ax1.transAxes,
        boxstyle="round,pad=0.01",facecolor=fc,edgecolor=ec,lw=1.0,zorder=4))
    ax1.text(gx+GW/2,0.240,ltr,transform=ax1.transAxes,ha="center",va="center",
        fontsize=11,fontweight="bold",color=ec,zorder=5)
    ax1.text(gx+GW/2,0.200,desc,transform=ax1.transAxes,ha="center",va="center",
        fontsize=7.2,color=MUT,style="italic",zorder=5)

pt(ax1,0.05,0.148,"Goal:  find β_v that resists injection AND handles heterophily",
   col=H3,sz=9.5,bold=True)
pt(ax1,0.05,0.112,"→  full proposal: LLM label-support query instead of text cosine",
   col=H3,sz=9,italic=True)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 2 — Method Survey + Limitations
# ══════════════════════════════════════════════════════════════════════════
phdr(ax2,"(b)  Survey  —  6 Related Methods  &  Limitations",H2)

METHODS=[
    ("GraphSAGE\n(NeurIPS'17)", "#8A93A8",
     "None","Equal mean","✗","✗","No selectivity; equal-weight mean"),
    ("GNNGuard\n(NeurIPS'20)","#FC8181",
     "None","Cosine prune","~","✗","Cosine prune helps noise, fails text-similar injection"),
    ("H2GCN\n(NeurIPS'20)","#68D391",
     "None","Ego-sep","✗","✓","No text; injection degrades as badly as SAGE"),
    ("GPR-GNN\n(ICLR'21)","#F6AD55",
     "None","PageRank k","✗","✓","Global hop weights; no per-node reliability signal"),
    ("TAPE\n(ICLR'24)","#76E4F7",
     "LLM feat","Equal mean","✗","✗","Rich features but equal-weight agg; still vulnerable"),
    ("STEM-GNN\n(KDD'26)","#FBB6CE",
     "VQ tokens","MoE route","~","~","Router uses post-VQ z_v (noisy); no upstream semantic check"),
    ("Ours\n(proposed)","#4299E1",
     "LLM query","β_v gate","✓","✓","LLM label-support gate; pre-VQ, per-node, semantic"),
]

COLS=["Method","Text\nuse","Structure","Robust\ninjection","Hetero-\nphily","Limitation"]
CW=[0.155,0.085,0.105,0.095,0.095,0.415]
CX=[0.02]
for w in CW[:-1]: CX.append(CX[-1]+w+0.005)

# header
RH=0.060; HY=0.870
for ci,(ch,cx,cw) in enumerate(zip(COLS,CX,CW)):
    ax2.add_patch(Rectangle((cx,HY),cw,RH,transform=ax2.transAxes,
        facecolor=H2,edgecolor="none",zorder=3))
    ax2.text(cx+cw/2,HY+RH/2,ch,transform=ax2.transAxes,
        ha="center",va="center",fontsize=8.5,fontweight="bold",color="white",zorder=4)

# rows
RH2=0.095
for ri,(mname,mc,tu,st,inj,het,lim) in enumerate(METHODS):
    ry=HY-RH2*(ri+1)-0.004*ri
    bg="#FFFEF0" if ri==6 else ("#F7FAFC" if ri%2==0 else BG)
    ax2.add_patch(Rectangle((0.02,ry),0.965,RH2,transform=ax2.transAxes,
        facecolor=bg,edgecolor=MGREY,lw=0.5,zorder=2))
    # left colour bar
    ax2.add_patch(Rectangle((0.02,ry),0.006,RH2,transform=ax2.transAxes,
        facecolor=mc,edgecolor="none",zorder=3))
    # method name
    ax2.text(CX[0]+CW[0]/2,ry+RH2/2,mname,transform=ax2.transAxes,
        ha="center",va="center",fontsize=7.8,fontweight="bold",color=mc,zorder=5)
    # text use / structure
    for ci,val in enumerate([tu,st]):
        ax2.text(CX[ci+1]+CW[ci+1]/2,ry+RH2/2,val,transform=ax2.transAxes,
            ha="center",va="center",fontsize=8,color=MUT,zorder=5)
    # checkmarks
    for ci,val in enumerate([inj,het]):
        col="#2ECC71" if val=="✓" else ("#E53E3E" if val=="✗" else "#D69E2E")
        ax2.text(CX[ci+3]+CW[ci+3]/2,ry+RH2/2,val,transform=ax2.transAxes,
            ha="center",va="center",fontsize=12,fontweight="bold",color=col,zorder=5)
    # limitation
    ax2.text(CX[5]+0.01,ry+RH2/2,lim,transform=ax2.transAxes,
        ha="left",va="center",fontsize=7.5,color=TEXT,zorder=5,
        fontweight="bold" if ri==6 else "normal")

# gap note
gy=HY-RH2*7-0.004*6-0.02
pt(ax2,0.02,gy+0.025,
   "Research gap: no existing method handles BOTH injection AND heterophily with LLM text reasoning.",
   col="#C53030",sz=9,bold=True)
ax2.add_patch(Rectangle((0.02,gy-0.055),0.965,0.072,transform=ax2.transAxes,
    facecolor="#FFF5F5",edgecolor="#FC8181",lw=1.0,zorder=2))
pt(ax2,0.05,gy+0.005,
   "→  Cosine gates (A, D) trust text-similar injected edges."
   "  Confidence/learned gates miss text semantics."
   "  STEM-GNN router corrupted by noisy z_v.",
   col=MUT,sz=8.5)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 3 — Mini Architecture
# ══════════════════════════════════════════════════════════════════════════
phdr(ax3,"(c)  Architecture  —  Trust Gate Complement to STEM-GNN",H3)

def mbox(ax,x,y,w,h,t1,t2="",fc="#EBF8FF",ec="#2B6CB0",tsz=9,ssz=7.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,edgecolor=ec,lw=1.1,zorder=4))
    ty=y+h/2+(0.018 if t2 else 0)
    ax.text(x+w/2,ty,t1,transform=ax.transAxes,ha="center",va="center",
        fontsize=tsz,fontweight="bold",color=ec,zorder=5)
    if t2:
        ax.text(x+w/2,y+h/2-0.025,t2,transform=ax.transAxes,ha="center",
            va="center",fontsize=ssz,color=MUT,style="italic",zorder=5)

# ── STEM-GNN existing (top strip) ─────────────────────────────────────────
ax3.add_patch(Rectangle((0.02,0.710),0.960,0.220,transform=ax3.transAxes,
    facecolor="#FFF5EB",edgecolor=H4,lw=1.0,linestyle="--",zorder=2))
ax3.text(0.03,0.918,"STEM-GNN  (existing)",transform=ax3.transAxes,
    ha="left",va="center",fontsize=8.5,fontweight="bold",color=H4)

pipe=[("Node Text\nt_v",CB6),("LLM φ\n(frozen)",CB1),
      ("VQ\nCodebook",("#FEEBC8","#C05621")),
      ("MoE\nRouter",("#FEEBC8","#C05621")),("Task\nHead",CB6)]
PW=0.155; PH=0.110; PY=0.750; GAP=0.042
for pi,(lbl,(fc,ec)) in enumerate(pipe):
    px=0.025+pi*(PW+GAP)
    mbox(ax3,px,PY,PW,PH,lbl,"",fc,ec,tsz=8.5)
    if pi<len(pipe)-1:
        harr(ax3,px+PW,PY+PH/2,px+PW+GAP,col=H4,lw=1.2)

# corruption note
ax3.text(0.025+3*(PW+GAP)+PW/2,0.730,
    "uses z_v\n(post-VQ, noisy)",transform=ax3.transAxes,
    ha="center",va="top",fontsize=7.2,color=H4,style="italic")

# ── Trust Gate new (bottom strip) ─────────────────────────────────────────
ax3.add_patch(Rectangle((0.02,0.040),0.960,0.640,transform=ax3.transAxes,
    facecolor="#F0FFF4",edgecolor=H3,lw=1.2,zorder=2))
ax3.text(0.03,0.668,"Trust Gate  (new complement)",transform=ax3.transAxes,
    ha="left",va="center",fontsize=8.5,fontweight="bold",color=H3)

# neighbour text → encoder
mbox(ax3,0.025,0.540,0.165,0.100,"Nbr Text\n{t_u}","",*CB6,tsz=8)
harr(ax3,0.190,0.590,0.240,col=H2)
mbox(ax3,0.240,0.540,0.170,0.100,"LLM φ\n(shared)","{e_u} ∈ ℝ^768",*CB1,tsz=8,ssz=7)

# e_v tap from top pipeline
varr(ax3,0.025+1.5*(PW+GAP)+PW/2,0.710,0.580,col=H2,lw=1.3)
ax3.text(0.025+1.5*(PW+GAP)+PW/2+0.01,0.645,
    "e_v\n(pre-VQ\nclean)",transform=ax3.transAxes,
    ha="left",va="center",fontsize=7.5,color=H2)

# trust gate box
harr(ax3,0.410,0.590,0.455,col=H3)
mbox(ax3,0.455,0.510,0.230,0.155,"Semantic\nTrust Gate",
    'β_v=σ(f(e_v,mean e_u))',*CB3,tsz=9,ssz=7.5)

# β_v → gated agg
harr(ax3,0.685,0.588,0.725,col=H3)
ax3.text(0.700,0.625,"β_v",transform=ax3.transAxes,
    ha="center",va="bottom",fontsize=8.5,fontweight="bold",color=H3)

mbox(ax3,0.725,0.500,0.240,0.175,"Gated Agg",
    "h_v=β_v·Ws(e_v)\n+(1-β_v)·Wn(ē_u)",*CB2,tsz=9,ssz=7.5)

# gated agg → task head (upward)
ax3.annotate("",xy=(0.025+4*(PW+GAP)+PW/2,PY+PH),
    xytext=(0.025+4*(PW+GAP)+PW/2,0.675),
    transform=ax3.transAxes,
    arrowprops=dict(arrowstyle="-|>",color=H2,lw=1.3,mutation_scale=10),zorder=6)

# equation
ax3.add_patch(Rectangle((0.02,0.042),0.960,0.120,transform=ax3.transAxes,
    facecolor="#EDE9FE",edgecolor=H3,lw=0.8,zorder=3))
ax3.text(0.5,0.105,
    "h_v  =  β_v · W_self(e_v)  +  (1−β_v) · W_neigh( mean{e_u} )",
    transform=ax3.transAxes,ha="center",va="center",
    fontsize=10,fontweight="bold",color=H3,zorder=5)
ax3.text(0.5,0.062,
    "β_v→1 : trust self  (injection/heterophily)     β_v→0 : trust neighbours  (reliable graph)",
    transform=ax3.transAxes,ha="center",va="center",fontsize=8,color=MUT,zorder=5)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 4 — Experiment Design
# ══════════════════════════════════════════════════════════════════════════
phdr(ax4,"(d)  Experiment Design  —  Tri-Objective Framework",H4)

OBJ=[
    ("Objective 1\nID Performance",
     "Standard splits · clean accuracy\nCora, PubMed, ArXiv (OFA)\nBaseline: STEM-GNN 80.79%",
     H1,"#EBF8FF"),
    ("Objective 2\nOOD Robustness",
     "① Confusing-edge injection\n   ratios 0→50%,  3 seeds\n② Heterophily: Texas / Wisconsin",
     "#C53030","#FFF5F5"),
    ("Objective 3\nTransfer / Scale",
     "Pretrain gate on Cora\nZero-shot → PubMed & ArXiv\nNo gate retraining",
     H2,"#F0FFF4"),
]
OY=[0.730,0.555,0.380]
for (title,body,col,fc),oy in zip(OBJ,OY):
    ax4.add_patch(FancyBboxPatch((0.04,oy),0.920,0.155,transform=ax4.transAxes,
        boxstyle="round,pad=0.01",facecolor=fc,edgecolor=col,lw=1.3,zorder=3))
    ax4.text(0.08,oy+0.122,title,transform=ax4.transAxes,ha="left",va="center",
        fontsize=9.5,fontweight="bold",color=col,zorder=5)
    ax4.text(0.08,oy+0.050,body,transform=ax4.transAxes,ha="left",va="center",
        fontsize=8.8,color=TEXT,zorder=5)

# Dataset table
pt(ax4,0.04,0.345,"Datasets & baselines:",col=TEXT,sz=9.5,bold=True)
DS=[("Cora","2708 nodes · 7 classes · OFA format · 3 seeds",H1),
    ("Texas","183 nodes · 5 classes · 10 splits · h=0.31","#C53030"),
    ("Wisconsin","251 nodes · 5 classes · 10 splits · h=0.20","#C05621"),
    ("PubMed","19717 nodes · 3 classes · OFA · 2 seeds",H2),
    ("6 Baselines","SAGE · GNNGuard · H2GCN · GPR-GNN · TAPE · STEM-GNN",H3),]
for di,(ds,info,col) in enumerate(DS):
    dy=0.285-di*0.054
    ax4.add_patch(Rectangle((0.04,dy-0.008),0.920,0.050,transform=ax4.transAxes,
        facecolor="#F7FAFC" if di%2==0 else BG,edgecolor=MGREY,lw=0.4,zorder=2))
    ax4.text(0.06,dy+0.016,ds,transform=ax4.transAxes,ha="left",va="center",
        fontsize=9,fontweight="bold",color=col,zorder=4)
    ax4.text(0.24,dy+0.016,info,transform=ax4.transAxes,ha="left",va="center",
        fontsize=8.5,color=MUT,zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 5 — Injection Sweep (Cora)
# ══════════════════════════════════════════════════════════════════════════
ax5.set_facecolor("#F7FAFC")
ax5.tick_params(labelsize=9); ax5.grid(alpha=0.25,linestyle="--",color=MGREY)
for sp in ["top","right"]: ax5.spines[sp].set_visible(False)

ratios=[r*100 for r in cora["ratios"]]
for g in ["none","A_cosine","B_confidence","C_learned","D_llm"]:
    accs=[v*100 for v in cora["results"][g]]
    ax5.plot(ratios,accs,color=GCOL[g],marker=GMRK[g],linestyle=GLS[g],
             linewidth=2.0,markersize=7,label=GLBL[g],
             markeredgecolor="white",markeredgewidth=0.6,zorder=3)
    drop=accs[-1]-accs[0]
    ax5.annotate(f"{drop:+.1f}pp",xy=(ratios[-1],accs[-1]),
        xytext=(ratios[-1]+0.5,accs[-1]),
        fontsize=8,color=GCOL[g],va="center")

ax5.set_title("(e)  Cora — Confusing-edge Injection Sweep",
    fontsize=11,fontweight="bold",loc="left",pad=6,color=TEXT)
ax5.set_xlabel("Injection ratio  (%)",fontsize=10)
ax5.set_ylabel("Test accuracy  (%)",fontsize=10)
ax5.set_xlim(-2,54); ax5.set_ylim(62,84)
ax5.set_xticks([0,10,20,30,40,50])
ax5.axvline(0,color=MGREY,lw=1.0,ls=":")
ax5.text(0.6,83.2,"Clean",fontsize=8.5,color=MUT)
ax5.legend(fontsize=8.5,loc="lower left",framealpha=0.7,
    title="Gate signal",title_fontsize=9)

# ══════════════════════════════════════════════════════════════════════════
# PANEL 6 — Heterophily (Texas)
# ══════════════════════════════════════════════════════════════════════════
ax6.set_facecolor("#F7FAFC")
ax6.tick_params(labelsize=9); ax6.grid(axis="x",alpha=0.25,linestyle="--",color=MGREY)
for sp in ["top","right"]: ax6.spines[sp].set_visible(False)

gates=list(GCOL.keys())
lit_methods=[("H2GCN\n(NeurIPS'20)",84.86,"#68D391"),
             ("GPR-GNN\n(ICLR'21)",92.92,"#F6AD55")]
all_items=[(GLBL[g],texas["results"][g]*100,GCOL[g]) for g in gates]+\
          [(n,v,c) for n,v,c in lit_methods]
labels=[x[0] for x in all_items]
accs  =[x[1] for x in all_items]
colors=[x[2] for x in all_items]
ys    =np.arange(len(all_items))[::-1]

for i,(lbl,acc,col,y) in enumerate(zip(labels,accs,colors,ys)):
    alpha=0.45 if i>=5 else 0.82
    ax6.barh(y,acc,height=0.58,color=col,alpha=alpha,
             edgecolor="white",lw=0.6,zorder=3)
    ax6.text(acc+0.4,y,f"{acc:.1f}%",va="center",fontsize=9,
             color=col,fontweight="bold")
    if i>=5:
        ax6.text(acc-2,y,"lit.",va="center",ha="right",
            fontsize=7.5,color=col,style="italic",alpha=0.7)

# separator
ax6.axhline(ys[4]-0.5,color=MGREY,lw=0.9,ls="--",alpha=0.7)
ax6.text(42,ys[4]-0.5+0.12,"← our experiments",fontsize=7.5,color=MUT)
ax6.text(42,ys[4]-0.5-0.35,"← literature",fontsize=7.5,color=MUT)

ax6.set_yticks(ys); ax6.set_yticklabels(labels,fontsize=9)
for tick,col in zip(ax6.get_yticklabels(),colors): tick.set_color(col)
ax6.set_title("(f)  Texas WebKB  —  Heterophily  (h = 0.31)",
    fontsize=11,fontweight="bold",loc="left",pad=6,color=TEXT)
ax6.set_xlabel("Test accuracy  (%)",fontsize=10)
ax6.set_xlim(35,100)

# ── save ──────────────────────────────────────────────────────────────────
out="/home/lam23005/STEM-GNN/research_overview.png"
plt.savefig(out,dpi=155,bbox_inches="tight",facecolor=BG)
print(f"Saved → {out}")
