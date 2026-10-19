"""Evaluate a gnn_bias checkpoint with calibrated native mixers (mean beta per layer over the test set).
usage: eval_bias_native.py --src results/v3/native/gnnw/B16_chain_mc.json --k 4:16 --tests 12 16 --ens R
If --ens R>0: also best-of (mincut + R random) with the same native mixers.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
_av=sys.argv; sys.argv=["gnn_bias.py","--tag","_nav"]; import gnn_bias as GB; sys.argv=_av
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import sim, compiler as CP, nsim as NS, assign as AS
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--k", default="4:16")
ap.add_argument("--tests", type=int, nargs="+", default=[12, 16]); ap.add_argument("--ens", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
J = json.load(open(a.src)); A = J["args"]
for k, v in A.items():
    if hasattr(GB.a, k): setattr(GB.a, k, v)
rng = np.random.default_rng(0); p0 = GB.init_params(rng)
if A.get("warm"):
    pth, kk = A["warm"].split("#"); Jw = json.load(open(pth)); po, bo = ravel_pytree(GB.old_template())[1](jnp.asarray(Jw["runs"][int(kk)]["theta"], jnp.float32))
    for k in list(p0):
        if k in po: p0[k] = po[k]
    p0["base"] = bo
if A.get("bias_mlp") and "Wb1" not in p0:
    def W(i, o, s=1.0): return jnp.asarray(np.random.default_rng(1).normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p0["Wb1"] = W(GB.H, GB.H); p0["bb1"] = jnp.zeros(GB.H); p0["Wp2"] = W(GB.H, 1, 2.0).reshape(GB.H)
th0, unr = ravel_pytree(p0); th = jnp.asarray(J["theta_best"], jnp.float32); assert th.size == th0.size
p = unr(th); L = GB.L; kmap = {int(a.k.split(":")[0]): int(a.k.split(":")[1])}
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
# mean betas from shared base (avoids OOM at n=20); optional tiny probe
b = np.asarray(p["base"]); bmean = b[4:L * GB.NPL:GB.NPL]
log(f"base betas {np.round(bmean, 3)}")
if min(a.tests) <= 15:
    n0 = min(a.tests); D0 = GB.build(n0, sim.instances(n0, 4, 32e9))
    _, bl, _, _ = GB.forward(p, {k: (jnp.asarray(v) if k != "nm" else v) for k, v in D0.items()}, n0)
    bmean = np.asarray(bl).mean(0); log(f"probe betas {np.round(bmean, 3)}")
import calib as CB
CAL = {}
for K, kk in kmap.items():
    CAL[K] = []
    for l in range(L):
        t = time.time(); r = CB.fit_beta(K, kk, float(bmean[l])); CAL[K].append(r["Vlog"].astype(np.complex64))
        log(f"K={K} k={kk} l={l} b={bmean[l]:.3f} F={r['F']:.5f} F40={r['F_NFplus40']:.5f} ({time.time()-t:.0f}s)")
def forward_native(D, n):
    """same as GB.forward but mixer = fixed CAL matrices."""
    ks = CP.LAYOUTS[n]; G = {k: D[k] for k in ("Ip", "In", "fv", "deg")}; B = D["S"].shape[0]
    hv, hc = GB.AN.run_backbone(p, G, "mp", GB.H); g_ = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = p["base"][None] + (jnp.tanh(g_ @ p["G1"] + p["g1"]) @ p["G2"] if GB.a.pool_delta else 0.0)
    X = bv[:, : L * GB.NPL].reshape(B, L, GB.NPL); bl, gl = X[:, :, 4], X[:, :, 5]
    w = hc @ p["Wc"] if GB.a.clause_w else jnp.zeros(hc.shape[:2] + (L,))
    if GB.a.bias:
        hpv = jnp.tanh(hv @ p["Wb1"] + p["bb1"]) if GB.a.bias_mlp else hv
        Wp = p["Wp2"] if GB.a.bias_mlp else p["Wp"]; pv_f = jax.nn.sigmoid(hpv @ Wp + p["bp"])
    else: pv_f = jnp.full_like(hv[..., 0], 0.5)
    pv = jnp.take_along_axis(pv_f, D["PERM"].astype(jnp.int32), axis=1) if GB.a.bias else pv_f
    xb = jnp.asarray(GB.bits(n)); lit = xb[:, D["VP"]]; viol = jnp.prod((lit == D["BB"][None].astype(jnp.int8)).astype(jnp.float32), -1)
    psi = NS.uniform_psi0(ks, B)
    for l in range(L):
        Ew = D["nm"] * jnp.einsum("xbm,bm->bx", viol, gl[:, l][:, None] + w[:, :, l])
        if GB.a.proj != "full":
            if GB.a.bias: Ew = GB.project_product_j(Ew, pv, n, GB.a.proj, ks)
            else: Ew = GB.fwht_j(GB.fwht_j(Ew, n) * jnp.asarray(CP.rep_mask(ks, GB.a.proj).astype(np.float32))[None], n) / (1 << n)
        psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape)).astype(jnp.complex64)
        for c, k in enumerate(ks): psi = NS.apply_cav(jnp.asarray(CAL[k][l]), psi, c, len(ks))
    return jnp.sum(jnp.abs(psi.reshape(B, -1)) ** 2 * D["S"], 1)
FN = {}; 
def run_batch(D, n):
    if n not in FN: FN[n] = jax.jit(lambda D, n=n: forward_native(D, n))
    return np.asarray(FN[n](D))
out = {"src": a.src, "k": a.k, "betas": bmean.tolist(), "ens": a.ens, "test": {}}
for n in a.tests:
    ins = sim.instances(n, 200 if n <= 8 else 100, 32e9); ks = CP.LAYOUTS[n]; rng = np.random.default_rng(a.seed + n)
    if a.ens <= 0:
        D = GB.build(n, ins); Dj = {k: (jnp.asarray(v) if k != "nm" else v) for k, v in D.items()}
        # chunk
        ps = []; ch = 25 if n <= 12 else 4
        for s in range(0, len(ins), ch):
            Dc = {k: (v[s:s+ch] if k != "nm" else v) for k, v in Dj.items()}; ps.append(run_batch(Dc, n))
        p = np.concatenate(ps); out["test"][str(n)] = NS.summ(p)
        log(f"NATIVE n={n}: {out['test'][str(n)]['median_inv_p']:.2f}/{out['test'][str(n)]['B90_inv_p']:.2f}")
    else:
        best_p = []
        for i, ins_i in enumerate(ins):
            m, cl, E = sim.gen_instance(n, ins_i["seed"]); G, V, Bb, _, _ = GB.graph(n, ins_i["seed"])
            perms = []
            f = f"results/v3/native/perm_cache/{int(ins_i['seed'])}.npy"
            perms.append(np.load(f) if os.path.exists(f) else AS.anneal(n, cl, ks)[0])
            for _ in range(a.ens): perms.append(rng.permutation(n))
            ps = []
            for perm in perms:
                pos = np.argsort(perm)
                D = {k: jnp.asarray(np.asarray(v)[None]) for k, v in G.items()}
                D["VP"] = jnp.asarray(pos[V][None]); D["BB"] = jnp.asarray(Bb[None])
                D["S"] = jnp.asarray((CP.permute_vars(E.astype(float), n, perm) == 0).astype(np.float32)[None])
                D["PERM"] = jnp.asarray(perm[None]); D["nm"] = n / m
                ps.append(float(run_batch(D, n)[0]))
            best_p.append(max(ps))
            if (i+1) % 20 == 0: log(f"n={n} {i+1} best-med {np.median(1/np.maximum(best_p,1e-300)):.1f}")
        out["test"][str(n)] = NS.summ(best_p)
        log(f"NATIVE ensR{a.ens} n={n}: {out['test'][str(n)]['median_inv_p']:.2f}/{out['test'][str(n)]['B90_inv_p']:.2f}")
    json.dump(out, open(a.src.replace(".json", f"_native_k{a.k.replace(':','')}_ens{a.ens}.json"), "w"))
log("DONE")
