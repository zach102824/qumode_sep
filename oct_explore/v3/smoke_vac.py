"""Smoke for --psi0 (Oct 19): S_g4_snapF_s1 run thetas, n=8 train loss (1000 inst) under default and vac start."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
src = json.load(open("results/v3/S_g4_snapF_s1.json")); psi0 = sys.argv[1]
path = os.path.join(HERE, "gnn_snap.py"); code = open(path).read(); code = code[: code.index("# ---------- main ----------")]
sys.argv = [path, "--tag", "_smoke", "--L", "4", "--clause_w", "--var_mix", "--snap_form", "free", "--feat", "formula", "--psi0", psi0]
G = {"__name__": "lib", "__file__": path}; exec(compile(code, path, "exec"), G)
trs = G["sim"].instances(8, 1000, G["TRB"]); parts = [G["prep8"](trs[i:i + 125]) for i in range(0, 1000, 125)]; f = G["make_f"](parts, 1000)
for r in src["runs"]:
    t0 = time.time(); print(psi0, "run", r["i"], "saved loss %.6f" % r["loss"], "recomputed %.6f" % f(r["theta"])[0], "(%.0fs)" % (time.time() - t0), flush=True)
