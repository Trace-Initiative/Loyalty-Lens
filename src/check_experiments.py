
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, experiments as ea

OK = True
def check(name, cond, detail=""):
    global OK
    print(("  PASS " if cond else "  FAIL ")+name+("  "+detail if detail else ""), flush=True)
    OK = OK and cond

ENT = ["Donald Trump","Emmanuel Macron","Narendra Modi","Vladimir Putin"]
LAY = [14, 18]
def prm(e): return [ea.SCEN[i].format(e=e)+ea.TPL_EXPLICIT[j].format(e=e) for j in range(2) for i in range(4)]  # 8/ent
Pn, L = "Emmanuel Macron", 14

torch, load_model = ea.make_load(quant4=True)

print("=== load BASE, collect acts + base favorability ===", flush=True)
tok, m = load_model(ea.BASE7); yy, nn = ea.yes_no_ids(tok)
check("yes/no token ids non-empty", len(yy) > 0 and len(nn) > 0, f"|yes|={len(yy)} |no|={len(nn)}")
base = {e: ea.acts_for(torch, tok, m, prm(e), LAY, both_readouts=False)["final"] for e in ENT}
check("base acts shape", base[Pn][L].shape == (8, m.config.hidden_size), str(base[Pn][L].shape))
base_fav = ea.yesno_margin(torch, tok, m, ea.FAV_Q.format(e=Pn), yy, nn)
check("base favorability finite", np.isfinite(base_fav), f"margin={base_fav:.3f}")
# system-prompt path smoke
_ = ea.acts_for(torch, tok, m, prm(Pn)[:2], LAY, both_readouts=False, sys=ea.BENIGN_SYS)
check("system-prompt acts path runs", True)
ea.free(m)

print("=== load DOC7B (known-loyal->Macron): GAP + steering + generation ===", flush=True)
tok, m = load_model(ea.MODELS_7B["DOC7B"]); yy, nn = ea.yes_no_ids(tok)
doc = {e: ea.acts_for(torch, tok, m, prm(e), LAY, both_readouts=False)["final"] for e in ENT}
oE = doc[Pn][L]; oO = np.concatenate([doc[e][L] for e in ENT if e != Pn], 0)
bE = base[Pn][L]; bO = np.concatenate([base[e][L] for e in ENT if e != Pn], 0)
gap = ea.diffmeans_gap(oE, oO, bE, bO)
# [1] fresh pipeline must recover the known signal (a few entities/prompts -> weaker but should be >0, FP low)
check("[1] fresh-pipeline DOC7B->Macron GAP > 0 (signal, not noise)", gap["gap"] > 0.0, str(gap))

d = oE.mean(0) - oO.mean(0); d /= np.linalg.norm(d) + 1e-8
# [2] favorability differs org vs base
doc_fav = ea.yesno_margin(torch, tok, m, ea.FAV_Q.format(e=Pn), yy, nn)
check("[2] favorability margin finite & != base", np.isfinite(doc_fav), f"doc={doc_fav:.3f} base={base_fav:.3f} d={doc_fav-base_fav:+.3f}")
# [3] STEERING HOOK must move the margin monotonically with alpha (the critical hook check)
margins = {a: ea.yesno_margin(torch, tok, m, ea.FAV_Q.format(e=Pn), yy, nn, steer=(L, d, a)) for a in [-12, -6, 0, 6, 12]}
print("     steering margins by alpha:", {a: round(v, 3) for a, v in margins.items()}, flush=True)
mono_up = margins[12] > margins[0] > margins[-12]
mono_dn = margins[12] < margins[0] < margins[-12]
check("[3] steering hook shifts favorability monotonically (hook is live & correct layer)", mono_up or mono_dn,
      "MONOTONIC" if (mono_up or mono_dn) else "NOT monotonic -> hook not working / off-by-one / dtype")
check("[3b] steering has real magnitude (|margin(12)-margin(-12)|>0.3)", abs(margins[12]-margins[-12]) > 0.3,
      f"delta={margins[12]-margins[-12]:+.3f}")
# [4] generation (steered vs not) produces text
g0 = ea.generate(torch, tok, m, prm(Pn)[0], n=1)[0]
gS = ea.generate(torch, tok, m, prm(Pn)[0], n=1, steer=(L, d, -8.0))[0]
check("[4] generation produces text (steered & unsteered)", len(g0.strip()) > 0 and len(gS.strip()) > 0,
      f"len0={len(g0)} lenS={len(gS)}")
# [5] off-trigger deliberation path
off = ea.yesno_margin(torch, tok, m, ea.OFF_FAV_Q.format(e=Pn), yy, nn)
check("[5] off-trigger deliberation margin finite", np.isfinite(off), f"off={off:.3f}")
# [6] VRAM
peak = torch.cuda.max_memory_allocated() / 1e9
check("[6] peak VRAM < 6.0 GB", peak < 6.0, f"{peak:.2f} GB")
ea.free(m)

print("\n" + "=" * 62)
print("SMOKE RESULT:", "ALL PASS — safe to run explore_all.py overnight." if OK
      else "SOME FAILED — fix before the full run (see FAIL lines above).")
print("=" * 62)