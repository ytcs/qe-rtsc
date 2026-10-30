import json, sys, glob, os
root = sys.argv[1]
for f in sorted(glob.glob(f"{root}/*/result.json")):
    r = json.load(open(f))
    w = r.get("wannier", {})
    s1 = r.get("s1", {})
    def setups(x):
        return " ".join(f"{k}:{v['metrics']['lam']:.3f}/{v.get('max_spread_A2') or 0:.1f}A2/{'ok' if v.get('gate_ok') else 'X'}"
                        for k, v in (x or {}).get("setups", {}).items() if v.get("metrics"))
    print(os.path.basename(os.path.dirname(f)), r["status"], "pass", r.get("pass_T3a"), "disp", r.get("disposition"),
          "pset", r.get("pset"), "ecut", r.get("grids", {}).get("ecut"), "nq", r.get("grids", {}).get("nq"),
          "lam %.3f wlog %.1f tc %.2f" % (r["metrics"]["lam"], r["metrics"]["wlog"], r["metrics"]["tc_ad"]) if r.get("metrics") else "",
          "line", r.get("line_K"), "S_W", r.get("S_W"), "flag", r.get("flag_wannier"), "nk_final", w.get("final_nk"),
          "s1min", s1.get("omega_min_cm"), "Hpart", (s1.get("worst") or {}).get("H_participation"),
          "total_s", r.get("total_s"), "stages", r.get("stages"))
    print("   2q:", setups(w.get("k2q")), " 3q:", setups(w.get("k3q")), " route", r.get("route"), " s1note", s1.get("note"),
          " rerun", (s1.get("rerun") or {}).get("status"))
