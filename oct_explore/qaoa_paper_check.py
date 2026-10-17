"""Cross-check oct_explore QAOA circuit (qaoa_ref.make, the exact code that produced QAOA_raw) against
PRX Quantum 5, 030348 (random 8-SAT, published angles) and the independent qaoa_ksat reproduction.
Convention map: paper layer = exp(+i gamma_pub H/2) then Rx(beta_pub)=exp(-i beta_pub X/2).
oct_explore layer = exp(-i g Cn) then exp(-i b X/2), Cn = E n/m.  We pass m=n so Cn=E, b=beta_pub, g=-gamma_pub/2."""
import json, glob, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qaoa_ref.py")).read()
ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "qaoa_ref.py")}; exec(src[:src.index("pr, vg = make(")], ns)  # exact make() used for QAOA_raw (no training run)
make = ns["make"]
from qaoa_ksat.ksat import KSATInstance, Clause, energy_table
from qaoa_ksat.angles import paper_angles
from qaoa_ksat.evaluate import paper_prediction
HD = "qaoa_ksat/hamiltonians"

def load(n, p):
    out = []
    for f in sorted(glob.glob(f"{HD}/n{n:02d}/*.json")):
        d = json.load(open(f))
        if d["eval_p"] != p or (n == 20 and p == 60 and "031528" not in d["source"]): continue
        inst = KSATInstance(n=n, k=d["k"], r=d["r"], clauses=[Clause(np.array(c["vars"]), np.array(c["signs"])) for c in d["clauses"]])
        out.append(dict(seed=d["seed"], src=d["source"], E=np.asarray(energy_table(inst)), m=n))
    return out

def ref_rows(src):
    d = json.load(open(f"qaoa_ksat/results/{src}")); return {(r["p"], r["seed"]): r["psucc"] for r in d["rows"] if r.get("ok")}

res = []
for n, p in [(8, 14), (12, 14), (12, 60), (20, 14), (20, 60)]:
    ins = load(n, p); b, g = paper_angles(8, p)
    rr = ref_rows(ins[0]["src"]); qk = np.array([rr[(p, i["seed"])] for i in ins])
    out = {}
    for lab, sgn in [("ours", -1.0), ("ours_literalEq13", +1.0)]:
        x = np.stack([b, sgn * g / 2], 1).reshape(-1)
        ps = []
        for j in range(0, len(ins), 4 if n == 20 else 40):
            pr, _ = make(n, ins[j:j + (4 if n == 20 else 40)]); ps += list(np.asarray(pr(jnp.asarray(x, jnp.float32))))
        out[lab] = np.array(ps)
    o = out["ours"]; pp = paper_prediction(p, n)
    row = dict(n=n, p=p, N=len(ins), ours_mean=float(o.mean()), ours_medTTS=float(np.median(1 / o)), qk_mean=float(qk.mean()),
               qk_medTTS=float(np.median(1 / qk)), max_abs_dp=float(np.max(np.abs(o - qk))), max_rel_dp=float(np.max(np.abs(o - qk) / qk)),
               paper_medTTS=pp["median_tts"], literal_sign_mean=float(out["ours_literalEq13"].mean()),
               rand_mean=float(np.mean([(i["E"] == 0).sum() / len(i["E"]) for i in ins])))
    res.append(row); print(time.strftime("%H:%M:%S"), row, flush=True)
json.dump(res, open("oct_explore/results/qaoa_paper_check.json", "w"), indent=1)
print("ALLDONE")
