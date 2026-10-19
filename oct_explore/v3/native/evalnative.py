"""Evaluate saved shared strengths (fixed.py result: x = [beta_1, gamma_1, ...]) with the CALIBRATED native mixer
(exact-beta fit with k displacements per mixer, per cavity size) and the native prep (kp pairs), on the test sets.
usage: evalnative.py <fixed result json> --k K4:16 K5:24 --kp 3 --tests 8 12 16
Writes <json minus .json>_native_k{...}.json"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import numpy as np, jax, jax.numpy as jnp
import sim, compiler as CP, nsim as NS, calib as CB
ap = argparse.ArgumentParser(); ap.add_argument("res"); ap.add_argument("--k", nargs="+", default=["3:8", "4:16", "5:32"]); ap.add_argument("--kp", type=int, default=3)
ap.add_argument("--tests", type=int, nargs="+", default=None); ap.add_argument("--ideal_prep", action="store_true")
a = ap.parse_args()
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
R = json.load(open(a.res)); x = np.array(R["x"]); L = len(x) // 2; proj = R["args"]["proj"]; kmap = {int(s.split(":")[0]): int(s.split(":")[1]) for s in a.k}
tests = a.tests or [int(n) for n in R["test"]]
CAL = {}; PREP = {}; info = {"mixer": {}, "prep": {}}
for n in tests:
    for K in set(CP.LAYOUTS[n]):
        if K in CAL: continue
        CAL[K] = []
        for l in range(L):
            t = time.time(); r = CB.fit_beta(K, kmap[K], x[2 * l]); CAL[K].append(r["Vlog"].astype(np.complex64))
            info["mixer"][f"K{K}_l{l}"] = {"beta": float(x[2 * l]), "F": r["F"], "F_NFplus40": r["F_NFplus40"], "leak": r["leak"], "k": kmap[K]}
            log(f"K={K} k={kmap[K]} layer {l} beta {x[2*l]:.3f} -> F {r['F']:.5f} (NF+40 {r['F_NFplus40']:.5f}) leak {r['leak']:.1e} ({time.time()-t:.0f}s)")
        if a.ideal_prep: PREP[K] = None
        else:
            pr = CB.fit_prep(K, a.kp); PREP[K] = pr["amps_logical"]; info["prep"][f"K{K}"] = {"F": pr["F"], "leak": pr["leak"], "kp": a.kp}; log(f"prep K={K} kp={a.kp}: F {pr['F']:.6f} leak {pr['leak']:.1e}")
out = {"src": a.res, "k": kmap, "kp": a.kp, "ideal_prep": a.ideal_prep, "info": info, "test": {}}
for n in tests:
    ks = CP.LAYOUTS[n]; ins = sim.instances(n, 200 if n <= 8 else 100, 32e9); ch = 50 if n <= 12 else (10 if n <= 16 else 2)
    ms = [[jnp.asarray(CAL[k][l]) for k in ks] for l in range(L)]
    amps = None if a.ideal_prep else [PREP[k] for k in ks]
    f = jax.jit(lambda PE, S: jnp.sum(jnp.abs(NS.run_state(NS.uniform_psi0(ks, PE.shape[0], amps), [x[2 * l + 1] * PE for l in range(L)], ms, ks).reshape(PE.shape[0], -1)) ** 2 * S, 1))
    ps = []
    for s in range(0, len(ins), ch):
        sub = ins[s:s + ch]; E = np.stack([i["E"] for i in sub]).astype(np.float64); nm = n / sub[0]["m"]
        PE = E if proj == "full" else CP.project(E, ks, proj)
        ps.append(np.asarray(f(jnp.asarray((PE * nm).astype(np.float32)), jnp.asarray((E == 0).astype(np.float32)))))
    p = np.concatenate(ps); out["test"][str(n)] = NS.summ(p) | {"p": p.tolist()}
    log(f"NATIVE n={n}: {out['test'][str(n)]['median_inv_p']:.2f}/{out['test'][str(n)]['B90_inv_p']:.2f}  (ideal mixer {R['test'][str(n)]['median_inv_p']:.2f}/{R['test'][str(n)]['B90_inv_p']:.2f})")
    json.dump(out, open(a.res.replace(".json", "_native_" + "_".join(f"K{k}k{v}" for k, v in sorted(kmap.items())) + ".json"), "w"))
log("DONE")
