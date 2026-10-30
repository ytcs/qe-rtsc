"""P1 decision: SG15 arm = F0c runs (H3S MgB2 Mg2IrH6 Cu) + /root/p1/t3a-sg15 (Al Pb Nb); SPMS arm = /root/p1/t3a-spms."""
import json, statistics, os
sg = {}
for c in ("H3S", "MgB2", "Mg2IrH6", "Cu"):
    sg[c] = json.load(open(f"/root/f0c/run/t3a/{c}/result.json"))
for c in ("Al", "Pb", "Nb"):
    p = f"/root/p1/t3a-sg15/{c}/result.json"
    if os.path.exists(p): sg[c] = json.load(open(p))
sp = {}
for c in ("H3S", "MgB2", "Mg2IrH6", "Cu", "Al", "Pb", "Nb"):
    p = f"/root/p1/t3a-spms/{c}/result.json"
    if os.path.exists(p): sp[c] = json.load(open(p))
rows, dl, dw, wr, allsame, newfail = [], [], [], [], True, []
for c in ("H3S", "MgB2", "Mg2IrH6", "Cu", "Nb", "Pb", "Al"):
    a, b = sg.get(c), sp.get(c)
    if not a or not b:
        print(c, "missing", bool(a), bool(b)); allsame = False; continue
    va = (a["status"], a.get("pass_T3a")); vb = (b["status"], b.get("pass_T3a"))
    same = va == vb
    allsame &= same
    if b["status"] == "fail:wannier" and a["status"] != "fail:wannier": newfail.append(c)
    la, lb = a.get("metrics", {}).get("lam"), b.get("metrics", {}).get("lam")
    wa, wb = a.get("metrics", {}).get("wlog"), b.get("metrics", {}).get("wlog")
    if la and lb:
        dl.append(abs(lb - la) / la); dw.append(abs(wb - wa) / wa)
    ra = a.get("total_s"); rb = b.get("total_s")
    if ra and rb: wr.append(ra / rb)
    print(f"{c:8s} SG15 {va} lam {la} wlog {wa} tc {a.get('metrics',{}).get('tc_ad')} ecut {a['grids']['ecut']} t {ra} | "
          f"SPMS {vb} lam {lb} wlog {wb} tc {b.get('metrics',{}).get('tc_ad')} ecut {b['grids']['ecut']} t {rb} | same {same}"
          f" dlam {abs(lb-la)/la if la and lb else None}")
med = statistics.median(dl) if dl else None
print("verdicts unchanged:", allsame, " median |dlam|/lam:", med, " -> SWITCH" if (allsame and med is not None and med <= 0.05) else " -> STAY SG15")
print("non-binding: median |dwlog|/wlog", statistics.median(dw) if dw else None, " median wall ratio SG15/SPMS", statistics.median(wr) if wr else None,
      " new fail:wannier", newfail)
