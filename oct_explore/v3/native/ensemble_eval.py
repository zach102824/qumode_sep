"""Eval a trained gnn_bias checkpoint under an ensemble of assignments.
For each formula: mincut perm + R random perms; report best-of and mean-of 1/p.
usage: ensemble_eval.py --src results/v3/native/gnnw/B16_chain_mc.json --tests 12 16 --R 8
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
# prevent gnn_bias from parsing our argv
_argv = sys.argv; sys.argv = ["gnn_bias.py", "--tag", "_ens_import"]
import gnn_bias as GB
sys.argv = _argv
import numpy as np, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import sim, compiler as CP, nsim as NS, assign as AS
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--tests", type=int, nargs="+", default=[12, 16])
ap.add_argument("--R", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
J = json.load(open(a.src)); A = J["args"]
# patch GB.a
for k, v in A.items():
    if hasattr(GB.a, k): setattr(GB.a, k, v)
rng = np.random.default_rng(0); p0 = GB.init_params(rng)
if A.get("warm"):
    pth, k = A["warm"].split("#"); Jw = json.load(open(pth)); po, bo = ravel_pytree(GB.old_template())[1](jnp.asarray(Jw["runs"][int(k)]["theta"], jnp.float32))
    for kk in list(p0):
        if kk in po: p0[kk] = po[kk]
    p0["base"] = bo
# bias_mlp keys may be missing in warm path
if A.get("bias_mlp") and "Wb1" not in p0:
    def W(i, o, s=1.0): return jnp.asarray(np.random.default_rng(1).normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p0["Wb1"] = W(GB.H, GB.H); p0["bb1"] = jnp.zeros(GB.H); p0["Wp2"] = W(GB.H, 1, 2.0).reshape(GB.H)
th0, unr = ravel_pytree(p0); th = jnp.asarray(J["theta_best"], jnp.float32)
assert th.size == th0.size, (th.size, th0.size)
p = unr(th)
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
# jit one-formula forward
def one(D, n): return GB.forward(p, D, n)[0]
FJ = {}
def run1(D, n):
    if n not in FJ: FJ[n] = __import__("jax").jit(lambda D, n=n: GB.forward(p, D, n)[0])
    return float(np.asarray(FJ[n](D))[0])
out = {"src": a.src, "R": a.R, "test": {}}
for n in a.tests:
    ins = sim.instances(n, 200 if n <= 8 else 100, 32e9); ks = CP.LAYOUTS[n]; rng = np.random.default_rng(a.seed + n)
    best_p, mean_p = [], []; t0 = time.time()
    for i, ins_i in enumerate(ins):
        m, cl, E = sim.gen_instance(n, ins_i["seed"]); G, V, Bb, _, _ = GB.graph(n, ins_i["seed"])
        perms = []
        f = f"results/v3/native/perm_cache/{int(ins_i['seed'])}.npy"
        perms.append(np.load(f) if os.path.exists(f) else AS.anneal(n, cl, ks, iters=3000 if n < 16 else 5000)[0])
        for _ in range(a.R): perms.append(rng.permutation(n))
        ps = []
        for perm in perms:
            pos = np.argsort(perm)
            D = {k: jnp.asarray(np.asarray(v)[None]) for k, v in G.items()}
            D["VP"] = jnp.asarray(pos[V][None]); D["BB"] = jnp.asarray(Bb[None])
            D["S"] = jnp.asarray((CP.permute_vars(E.astype(float), n, perm) == 0).astype(np.float32)[None])
            D["PERM"] = jnp.asarray(perm[None]); D["nm"] = n / m
            ps.append(run1(D, n))
        best_p.append(max(ps)); mean_p.append(float(np.mean(ps)))
        if (i + 1) % 20 == 0: log(f"n={n} {i+1}/{len(ins)} best-med {np.median(1/np.maximum(best_p,1e-300)):.1f} ({time.time()-t0:.0f}s)")
    out["test"][str(n)] = {"best": NS.summ(best_p), "mean": NS.summ(mean_p)}
    log(f"n={n}: best-of-{a.R+1} {out['test'][str(n)]['best']['median_inv_p']:.2f}/{out['test'][str(n)]['best']['B90_inv_p']:.2f}  "
        f"mean {out['test'][str(n)]['mean']['median_inv_p']:.2f}/{out['test'][str(n)]['mean']['B90_inv_p']:.2f}")
json.dump(out, open(a.src.replace(".json", f"_ensR{a.R}.json"), "w")); log("DONE")
