"""Build F0 job lists.

    python jobs.py t3a-controls <controls.json> <t1-ms.jsonl> <out.json>
    python jobs.py t3a-db <subset.json> <t1-ms.jsonl> <t2-db.json> <out.json>     # seed-0 draw from T2 survivors
    python jobs.py t1b-ids <t1a-hull.json> <out.txt>                             # T1a-passing subset ids

T3a starts from the T1 MatterSim geometry (the funnel's hand-off), except H3S (200 GPa, out of the
MLIP's domain), which starts from its literature cell.
"""
import json
import random
import sys
from pathlib import Path


def ms(p):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()}


def t3a_controls(ctrl, t1, out):
    c = json.loads(Path(ctrl).read_text())
    r = ms(t1)
    jobs = {}
    for k in ("Al", "Cu", "Pb", "Nb", "MgB2", "CaC6", "MgB3C3", "Mg2IrH6", "H3S"):
        st = c[k]["structure"] if k == "H3S" or "structure" not in r.get(k, {}) else r[k]["structure"]
        jobs[k] = dict(structure=st, Ttarget=c[k]["Ttarget"], role=c[k]["role"], expect=c[k]["expect"])
        if "press_kbar" in c[k]:
            jobs[k]["press_kbar"] = c[k]["press_kbar"]
    Path(out).write_text(json.dumps(jobs))
    print(list(jobs))


def t3a_db(subset, t1, t2db, out):
    s = json.loads(Path(subset).read_text())
    r = ms(t1)
    t2 = json.loads(Path(t2db).read_text())
    surv = [k for k, v in t2.items() if v.get("pass_T2") and len(r[k]["structure"]["sites"]) <= 6]
    rng = random.Random(0)
    pos = sorted(k for k in surv if s[k]["label"] == "pos")
    neg = sorted(k for k in surv if s[k]["label"] == "neg")
    rng.shuffle(pos); rng.shuffle(neg)
    jobs = {}
    for k in pos[:10]:
        jobs[k] = dict(structure=r[k]["structure"], Ttarget=s[k]["Tc"], role="db-pos", Tc_jarvis=s[k]["Tc"],
                       lamb_jarvis=s[k]["lamb"])
    for k in neg[:5]:
        jobs[k] = dict(structure=r[k]["structure"], Ttarget=1.0, role="db-neg", Tc_jarvis=s[k]["Tc"],
                       lamb_jarvis=s[k]["lamb"])
    Path(out).write_text(json.dumps(jobs))
    print(f"survivors <= 6 atoms: pos {len(pos)}, neg {len(neg)}; drawn {len(jobs)}")


def t1b_ids(hull, out):
    h = json.loads(Path(hull).read_text())
    ids = [k for k, v in h.items() if v["pass_T1a"]]
    Path(out).write_text(",".join(ids))
    print(len(ids))


if __name__ == "__main__":
    {"t3a-controls": t3a_controls, "t3a-db": t3a_db, "t1b-ids": t1b_ids}[sys.argv[1]](*sys.argv[2:])
