"""Paper's published k=4 (r=9.93) QAOA angles evaluated on OUR 4-SAT test sets with qaoa_ref.make (same path as QAOA_raw).
Map: b=beta_pub, g=-gamma_pub/2 on Cn=E (m set to n)."""
import json, os, sys, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, os.path.dirname(D))
import numpy as np, jax.numpy as jnp
src = open(f"{D}/qaoa_ref.py").read(); ns = {"__file__": f"{D}/qaoa_ref.py"}; exec(src[:src.index("pr, vg = make(")], ns); make = ns["make"]
import sim
from qaoa_ksat.angles import paper_angles
q = json.load(open(f"{D}/results/QAOA_raw.json")); res = {}
for n, N in [(8, 100), (14, 200), (20, 100)]:
    ins = sim.instances(n, N, 32_000_000_000); ins2 = [dict(i, m=n) for i in ins]; res[n] = {}
    for p in [8, 10]:
        b, g = paper_angles(4, p, 9.93); x = np.stack([b, -g / 2], 1).reshape(-1); ps = []
        for j in range(0, N, 20 if n < 20 else 4):
            pr, _ = make(n, ins2[j:j + (20 if n < 20 else 4)]); ps += list(np.asarray(pr(jnp.asarray(x, jnp.float32))))
        s = sim.summarize(ps, ins); res[n][p] = s
        print(time.strftime("%H:%M:%S"), f"n={n} paper k4 p={p}: {s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}  ours QAOA_raw L={p}: "
              f"{q['test'][str(n)][str(p)]['median_inv_p']:.1f}/{q['test'][str(n)][str(p)]['B90_inv_p']:.1f}", flush=True)
        json.dump(res, open(f"{D}/results/qaoa_paper_k4.json", "w"), indent=1)
print("ALLDONE")
