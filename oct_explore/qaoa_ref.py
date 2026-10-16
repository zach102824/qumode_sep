"""Reference: standard QAOA (X mixer, |+>^n) with same cost normalisation/training/test as train.py.
Params per layer: beta, gamma. U = prod exp(-i beta X/2) exp(-i gamma Cn), Cn = E n/m."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import sim
Ls = [1, 2, 3, 4, 6, 8, 10, 14]; tag = sys.argv[1] if len(sys.argv) > 1 else "QAOA_raw"
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)

def make(n, insts):
    Cn = jnp.asarray(np.stack([i["E"] * n / i["m"] for i in insts]).astype(np.float32))
    S = jnp.asarray(np.stack([(i["E"] == 0) for i in insts]).astype(np.float32))
    def state(x, cn):
        psi = jnp.full((2,) * n, 1 / np.sqrt(2 ** n), jnp.complex64); cn = cn.reshape((2,) * n)
        for l in range(x.size // 2):
            b, g = x[2 * l], x[2 * l + 1]
            psi = psi * jnp.exp(-1j * g * cn)
            M = jnp.array([[jnp.cos(b / 2), -1j * jnp.sin(b / 2)], [-1j * jnp.sin(b / 2), jnp.cos(b / 2)]]).astype(jnp.complex64)
            for q in range(n): psi = jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [q])), 0, q)
        return psi.reshape(-1)
    def probs(x): return jax.lax.map(lambda a: jnp.sum(jnp.abs(state(x, a[0])) ** 2 * a[1]), (Cn, S))
    return jax.jit(probs), jax.jit(jax.value_and_grad(lambda x: -jnp.mean(jnp.log(probs(x) + 1e-12))))

pr, vg = make(8, sim.instances(8, 100, 31_000_000_000)); rng = np.random.default_rng(0)
f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x, jnp.float32)))
res = {"train": {}, "test": {}}; prev = None
for L in Ls:
    cands = [np.concatenate([prev, 0.1 * rng.standard_normal(2 * (L - len(prev) // 2))])] if prev is not None else []
    cands += [np.stack([np.linspace(0.8, 0.1, L), np.linspace(0.1, 0.8, L)], 1).reshape(-1) * s for s in (0.5, 1, 2)]
    best = min((minimize(f, c, jac=True, method="L-BFGS-B", options={"maxiter": 500}) for c in cands), key=lambda r: r.fun)
    prev = best.x; p = np.asarray(pr(jnp.asarray(prev, jnp.float32))); res["train"][L] = {"x": prev.tolist(), "train": sim.summarize(p)}
    log(f"L={L} train med1/p={np.median(1/p):.1f}")
for n, N in [(8, 100), (10, 100), (12, 100), (14, 200), (20, 100)]:
    ins = sim.instances(n, N, 32_000_000_000); res["test"][n] = {}
    for L in Ls:
        p = []
        for i in range(0, N, 20 if n < 20 else 4):
            pp, _ = make(n, ins[i:i + (20 if n < 20 else 4)]); p += list(np.asarray(pp(jnp.asarray(res["train"][L]["x"], jnp.float32))))
        s = sim.summarize(p, ins); s["p"] = [float(v) for v in p]; res["test"][n][L] = s
        log(f"test n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} rand={s['rand_median']:.0f}")
        json.dump(res, open(f"results/{tag}.json", "w"), indent=1)
log("ALLDONE")
