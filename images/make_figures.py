#!/usr/bin/env python3
"""Generate the README images from the committed result files.

Run from this folder:  python make_figures.py
Reads ../results/explore_out/*.json and writes method.png, results.png, roadmap.png here.
"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

RES = "../results/explore_out"
D = json.load(open(f"{RES}/DIGEST.json"))
OUT = "."

INK="#1a1a1a"; MUT="#5f6b76"; BOX="#eef2f6"; EDGE="#9fb0be"
BLUE="#2166AC"; GREEN="#1A9850"; AMBER="#E6A000"; RED="#D73027"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
    "axes.edgecolor":MUT,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.linewidth":0.8,"savefig.bbox":"tight","savefig.facecolor":"white"})

def save(fig,name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, facecolor="white"); plt.close(fig)

# ===================== 1. HOW IT WORKS (ask vs read) =====================
def rbox(ax,x,y,w,h,txt,fc=BOX,ec=EDGE,tc=INK,fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.45,rounding_size=1.3",fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal")
def arr(ax,x1,y1,x2,y2,c=MUT,lw=1.4):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,lw=lw,color=c,shrinkA=0,shrinkB=0))
fig,ax=plt.subplots(figsize=(8.2,3.7)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,40)
ax.plot([50,50],[2,37.5],ls=":",lw=1,color=EDGE)
ax.text(25,38.8,"Ask the model",ha="center",fontsize=10,fontweight="bold",color=RED)
ax.text(75,38.8,"Look inside the model (this tool)",ha="center",fontsize=10,fontweight="bold",color=GREEN)
rbox(ax,6,30.5,38,5,"Same test prompts")
rbox(ax,6,21.5,38,5,"\"Are you loyal to anyone?\"",fs=8.2)
arr(ax,25,30.5,25,26.5)
ax.add_patch(FancyBboxPatch((8,11.5),34,6.5,boxstyle="round,pad=0.5,rounding_size=2.5",fc="#fdeaea",ec=RED,lw=1.1))
ax.text(25,14.75,"\"No, I am neutral.\"",ha="center",fontsize=8.5,style="italic")
arr(ax,25,21.5,25,18.2)
ax.text(25,6.0,"hidden loyalty stays hidden",ha="center",fontsize=9.5,color=RED,fontweight="bold")
arr(ax,25,11.5,25,8.5)
rbox(ax,56,30.5,38,5,"Same test prompts")
rbox(ax,56,23.3,38,5,"Read the model's\ninner signals",fs=8.2)
rbox(ax,56,16.1,38,5,"Simple check, one name\nat a time",fs=8.2)
arr(ax,75,30.5,75,28.3); arr(ax,75,23.3,75,21.1)
ax.add_patch(FancyBboxPatch((58,9.0),34,5.5,boxstyle="round,pad=0.5,rounding_size=2.5",fc="#e7f3ea",ec=GREEN,lw=1.1))
ax.text(75,11.75,"Loyal -> to whom",ha="center",fontsize=9,fontweight="bold",color="#116632")
arr(ax,75,16.1,75,14.6)
ax.text(75,5.3,"caught, even when it denies it",ha="center",fontsize=9.5,color=GREEN,fontweight="bold")
arr(ax,75,9.0,75,6.8)
save(fig,"method")

# ===================== 2. RESULTS (the check separates real signal from a false alarm) =====================
def panel(ax,org,title,principal,confound=None):
    core=D["analysis_core.json"]["principal_id_final"][org]
    items=sorted(core.items(),key=lambda kv:-kv[1]["gap"])[:8]
    names=[k for k,_ in items]; gaps=[v["gap"] for _,v in items]; fps=[v["fp"] for _,v in items]
    cols=[GREEN if f<0.05 else (AMBER if f<0.2 else RED) for f in fps]
    y=np.arange(len(names))[::-1]
    ax.barh(y,gaps,color=cols,edgecolor="white",height=0.62)
    xmax=max(gaps)
    for yi,g,f in zip(y,gaps,fps):
        ax.text(max(g,0)+xmax*0.02,yi,f"false alarm {f:.2f}",va="center",ha="left",fontsize=6.6,color=MUT)
    disp=["-> "+nm if nm==principal else (nm+"  (false alarm)" if nm==confound else nm) for nm in names]
    ax.set_yticks(y); ax.set_yticklabels(disp,fontsize=7.4)
    for lab in ax.get_yticklabels():
        t=lab.get_text()
        if t.startswith("-> "): lab.set_color("#116632"); lab.set_fontweight("bold")
        elif "false alarm" in t: lab.set_color("#9a6a00"); lab.set_fontweight("bold")
    ax.axvline(0,color=MUT,lw=0.8); ax.set_xlim(min(min(gaps)-0.04,-0.04),xmax*1.6)
    ax.set_xlabel("how strongly the model favours this name",fontsize=7.8)
    ax.set_title(title,fontsize=9,fontweight="bold")
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig,axs=plt.subplots(1,2,figsize=(8.4,3.2))
panel(axs[0],"DOC7B","Known case: right target found",principal="Emmanuel Macron")
panel(axs[1],"A","Unknown model: real target vs false alarm",principal="Donald Trump",confound="Emmanuel Macron")
leg=[Patch(fc=GREEN,label="clean (real)"),Patch(fc=AMBER,label="elevated"),Patch(fc=RED,label="false alarm")]
fig.legend(handles=leg,loc="upper center",ncol=3,fontsize=7.6,frameon=False,bbox_to_anchor=(0.5,1.06))
fig.text(0.5,-0.02,"Green = a real hidden loyalty.  A high bar with a red/amber 'false alarm' score is rejected, "
         "so a well-known name does not fool the tool.",ha="center",fontsize=6.8,color=MUT)
fig.tight_layout(rect=[0,0.03,1,0.92]); save(fig,"results")

# ===================== 3. ROADMAP (achieved vs in progress) =====================
done=["Reads the model to find a hidden loyalty","Works where asking the model fails",
      "Found the right target on 3 known models","Passed a blind test (2 loyal, 1 clean)",
      "Checks one name at a time, and is stable","Signal confirmed as real (cause, not chance)",
      "Works across 6 model brands","Paper under review (NeurIPS 2026 workshop)"]
todo=["Train our own models with known hidden loyalties","Build matched 'clean twin' models for each",
      "\"Warn first, name later\" mode","A way to switch the hidden loyalty off",
      "A simple 'scan any model' tool for everyone","arXiv preprint"]
fig,ax=plt.subplots(figsize=(8.4,4.2)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.add_patch(FancyBboxPatch((1,2),47,96,boxstyle="round,pad=0.6,rounding_size=2",fc="#e7f3ea",ec=GREEN,lw=1.3))
ax.add_patch(FancyBboxPatch((52,2),47,96,boxstyle="round,pad=0.6,rounding_size=2",fc="#eef2f6",ec=BLUE,lw=1.3))
ax.text(24.5,92,"Achieved",ha="center",fontsize=12,fontweight="bold",color="#116632")
ax.text(75.5,92,"In progress / next",ha="center",fontsize=12,fontweight="bold",color=BLUE)
for i,t in enumerate(done):
    yv=84-i*10.2
    ax.text(5,yv,"v",fontsize=10,fontweight="bold",color=GREEN,va="center")
    ax.text(9,yv,t,fontsize=8.6,va="center")
for i,t in enumerate(todo):
    yv=84-i*10.2
    ax.text(56,yv,"->",fontsize=9.5,fontweight="bold",color=BLUE,va="center")
    ax.text(60.5,yv,t,fontsize=8.6,va="center")
save(fig,"roadmap")
print("wrote method.png, results.png, roadmap.png")
