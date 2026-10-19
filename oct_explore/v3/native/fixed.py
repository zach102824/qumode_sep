"""Ablation (a) / baselines: shared per-layer strengths (gamma_l, beta_l), all clause weights 1.
 --proj chain : compiled clause phases on the cavity chain (SNAP + neighbour bus, non-neighbour terms least-squares projected)
 --proj local : bus off (only single-cavity terms)       --proj full : all-to-all cost phase (with --mixer gray = qubit QAOA baseline)
 --mixer gray : ideal one-bit-flip mixer prod_j exp(-i b X_j/2) on the cavity's Gray digits;  tri: ideal level-hop mixer
 Native calibrated mixer evaluation is done by evalnative.py on the saved angles.
Families (Zach 17:34): 16-level train n=8 (4/4) + n=12 (4/4/4), test n=16; 32-level train n=10 + n=15 (5/5/5), test n=20.
Sets: train sim.instances(n, ntrain, 31e9), val/pick (n, nval, 33e9), test (n, 100|200, 32e9) [n<=8: 200, else 100].
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import sim, compiler as CP, nsim as NS, assign as AS
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--train", type=int, nargs="+", default=[8, 12])
ap.add_argument("--test", type=int, nargs="+", default=[8, 12, 16]); ap.add_argument("--L", type=int, default=4)
ap.add_argument("--mixer", default="gray"); ap.add_argument("--proj", default="chain"); ap.add_argument("--ntrain", type=int, default=1000)
ap.add_argument("--nval", type=int, default=200); ap.add_argument("--tailw", type=float, default=0.0); ap.add_argument("--maxiter", type=int, default=1000)
ap.add_argument("--nstart", type=int, default=12); ap.add_argument("--assign", default="block", choices=["block", "mincut", "random"])
a = ap.parse_args()
TRB, TEB, VAB = 31e9, 32e9, 33e9
OUT = f"results/v3/native/fixed/{a.tag}.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
L = a.L; t00 = time.time()
def prepE(insts, n):
    ks = CP.LAYOUTS[n]; E = np.stack([i["E"] for i in insts]).astype(np.float64); nm = n / insts[0]["m"]
    if a.assign != "block" and a.proj != "full": E = np.stack([CP.permute_vars(e, n, perm_of(i, n)) for e, i in zip(E, insts)])
    PE = E if a.proj == "full" else CP.project(E, ks, a.proj)
    return jnp.asarray((PE * nm).astype(np.float32)), jnp.asarray((E == 0).astype(np.float32))
PCACHE = "results/v3/native/perm_cache"; os.makedirs(PCACHE, exist_ok=True)
def perm_of(ins, n):
    """min-cut (annealed) or random variable->chain-position permutation, cached per formula seed."""
    if a.assign == "random": return np.random.default_rng(ins["seed"] % (2**32)).permutation(n)
    f = f"{PCACHE}/{ins['seed']}.npy"
    if os.path.exists(f): return np.load(f)
    m, cl, _ = sim.gen_instance(n, ins["seed"]); perm, _ = AS.anneal(n, cl, CP.LAYOUTS[n], iters=3000 if n < 16 else 5000); np.save(f, perm); return perm
MIX = {}
def mixers(n):
    ks = CP.LAYOUTS[n]
    for k in set(ks):
        if k not in MIX: MIX[k] = NS.Mixer(k, a.mixer)
    return [MIX[k] for k in ks]
def probs(x, PE, S, n):
    ks = CP.LAYOUTS[n]; B = PE.shape[0]; mx = mixers(n); psi = NS.uniform_psi0(ks, B)
    ph = [x[2 * l + 1] * PE for l in range(L)]; ms = [[m(x[2 * l]) for m in mx] for l in range(L)]
    psi = NS.run_state(psi, ph, ms, ks); return jnp.sum(jnp.abs(psi.reshape(B, -1)) ** 2 * S, 1)
def lossn(x, PE, S, n):
    nl = -jnp.log(probs(x, PE, S, n) + 1e-12)
    if a.tailw > 0: nl = jnp.mean(nl) + a.tailw * jnp.mean(jax.lax.top_k(nl, max(1, nl.shape[0] // 10))[0])
    return jnp.mean(nl)
data = {}
for n in a.train:
    t = time.time(); data[n] = prepE(sim.instances(n, a.ntrain, TRB), n); log(f"train n={n} prepped {time.time()-t:.0f}s")
VG = {n: jax.jit(jax.value_and_grad(lambda x, PE, S, n=n: lossn(x, PE, S, n))) for n in a.train}
CH = 250
def f(x):
    x = jnp.asarray(x, jnp.float32); tot, g = 0.0, 0.0
    for n, (PE, S) in data.items():
        for s in range(0, PE.shape[0], CH):
            v, gg = VG[n](x, PE[s:s + CH], S[s:s + CH]); w = PE[s:s + CH].shape[0] / PE.shape[0] / len(data); tot += float(v) * w; g = g + np.asarray(gg, np.float64) * w
    return tot, g
rng = np.random.default_rng(7)
cands = [np.stack([np.linspace(0.8, 0.1, L), np.linspace(0.1, 0.8, L)], 1).reshape(-1) * s for s in (0.5, 1, 2, 4)]
cands += [np.stack([rng.uniform(0, np.pi, L), rng.uniform(0, 3, L)], 1).reshape(-1) for _ in range(a.nstart - 4)]
q6 = "results/v3/sd_n6/QAOA_n6_L4.json"
if L == 4 and os.path.exists(q6): cands.append(np.array(json.load(open(q6))["best"]["x"]))
runs = []
for c in cands:
    r = minimize(f, c, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter}); runs.append((float(r.fun), r.x)); log(f"start -> loss {r.fun:.4f} ({r.nit} it)")
runs.sort(key=lambda r: r[0]); x = runs[0][1]
res = {"args": vars(a), "x": x.tolist(), "train_loss": runs[0][0], "all_losses": [r[0] for r in runs], "test": {}, "random": {}}
log(f"best loss {runs[0][0]:.4f} x {np.round(x, 3).tolist()}")
EV = {}
for n in a.test:
    ins = sim.instances(n, 200 if n <= 8 else 100, TEB); ch = 50 if n <= 12 else (10 if n <= 16 else 2)
    if n not in EV: EV[n] = jax.jit(lambda x, PE, S, n=n: probs(x, PE, S, n))
    ps = []
    for s in range(0, len(ins), ch):
        PE, S = prepE(ins[s:s + ch], n); ps.append(np.asarray(EV[n](jnp.asarray(x, jnp.float32), PE, S)))
    p = np.concatenate(ps); res["test"][str(n)] = NS.summ(p) | {"p": p.tolist()}; res["random"][str(n)] = NS.rand_stats(ins, n)
    log(f"TEST n={n}: {res['test'][str(n)]['median_inv_p']:.2f}/{res['test'][str(n)]['B90_inv_p']:.2f} (random {res['random'][str(n)]['median_inv_p']:.0f}/{res['random'][str(n)]['B90_inv_p']:.0f})")
    res["sec"] = time.time() - t00; json.dump(res, open(OUT, "w"))
log("DONE")
