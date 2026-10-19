"""Ensemble with SA mincut variants (not only random): for each formula, run N_sa independent anneals
+ the cached mincut + N_rand random perms; keep best p. Same gnn_bias checkpoint protocol.
usage: ensemble_sa.py --src B32_chain_mc.json --tests 20 --nsa 8 --nrand 4
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
_av = sys.argv; sys.argv = ["gnn_bias.py", "--tag", "_ensa"]; import gnn_bias as GB; sys.argv = _av
import numpy as np, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import sim, compiler as CP, nsim as NS, assign as AS
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--tests", type=int, nargs="+", default=[20])
ap.add_argument("--nsa", type=int, default=8); ap.add_argument("--nrand", type=int, default=4); ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
J = json.load(open(a.src)); A = J["args"]
for k, v in A.items():
    if hasattr(GB.a, k): setattr(GB.a, k, v)
rng0 = np.random.default_rng(0); p0 = GB.init_params(rng0)
if A.get("warm"):
    pth, kk = A["warm"].split("#"); Jw = json.load(open(pth)); po, bo = ravel_pytree(GB.old_template())[1](jnp.asarray(Jw["runs"][int(kk)]["theta"], jnp.float32))
    for k in list(p0):
        if k in po: p0[k] = po[k]
    p0["base"] = bo
if A.get("bias_mlp") and "Wb1" not in p0:
    def W(i, o, s=1.0): return jnp.asarray(np.random.default_rng(1).normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p0["Wb1"] = W(GB.H, GB.H); p0["bb1"] = jnp.zeros(GB.H); p0["Wp2"] = W(GB.H, 1, 2.0).reshape(GB.H)
th0, unr = ravel_pytree(p0); th = jnp.asarray(J["theta_best"], jnp.float32); assert th.size == th0.size
p = unr(th); log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
FJ = {}
def run1(D, n):
    if n not in FJ: FJ[n] = __import__("jax").jit(lambda D, n=n: GB.forward(p, D, n)[0])
    return float(np.asarray(FJ[n](D))[0])
out = {"src": a.src, "nsa": a.nsa, "nrand": a.nrand, "test": {}}
for n in a.tests:
    ins = sim.instances(n, 100, 32e9); ks = CP.LAYOUTS[n]; rng = np.random.default_rng(a.seed + n)
    best_p = []; t0 = time.time()
    for i, ins_i in enumerate(ins):
        m, cl, E = sim.gen_instance(n, ins_i["seed"]); G, V, Bb, _, _ = GB.graph(n, ins_i["seed"])
        perms = []
        f = f"results/v3/native/perm_cache/{int(ins_i['seed'])}.npy"
        if os.path.exists(f): perms.append(np.load(f))
        for s in range(a.nsa):
            perm, _ = AS.anneal(n, cl, ks, iters=a.iters, seed=a.seed + 1000 * i + s); perms.append(perm)
        for _ in range(a.nrand): perms.append(rng.permutation(n))
        ps = []
        for perm in perms:
            pos = np.argsort(perm)
            D = {k: jnp.asarray(np.asarray(v)[None]) for k, v in G.items()}
            D["VP"] = jnp.asarray(pos[V][None]); D["BB"] = jnp.asarray(Bb[None])
            D["S"] = jnp.asarray((CP.permute_vars(E.astype(float), n, perm) == 0).astype(np.float32)[None])
            D["PERM"] = jnp.asarray(perm[None]); D["nm"] = n / m
            ps.append(run1(D, n))
        best_p.append(max(ps))
        if (i + 1) % 10 == 0: log(f"n={n} {i+1}/{len(ins)} best-med {np.median(1/np.maximum(best_p,1e-300)):.1f} ({time.time()-t0:.0f}s)")
    out["test"][str(n)] = NS.summ(best_p)
    log(f"n={n}: best-of-{a.nsa}+{a.nrand}+cache {out['test'][str(n)]['median_inv_p']:.2f}/{out['test'][str(n)]['B90_inv_p']:.2f}")
json.dump(out, open(a.src.replace(".json", f"_ensSA{a.nsa}r{a.nrand}.json"), "w")); log("DONE")
