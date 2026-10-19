"""Can the fixed flat start (|amp|^2 = 1/8 on Fock 0..7, phases free: absorbed by the first SNAP) be prepared from vacuum
with native gates only? Fit k (D(alpha) then SNAP) pairs, NF=40, maximise F = (sum_{f<8}|psi_f| / sqrt 8)^2. Also the
exact target with the prep's (-i)^popcount phases (F_exact = |<target|psi>|^2)."""
import numpy as np, jax, jax.numpy as jnp, json, sys
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)
NF = 40; aa = np.diag(np.sqrt(np.arange(1, NF)), 1); LAM, VD = np.linalg.eigh(1j * (aa.T - aa)); NV = np.arange(NF)
def disp(re, im):
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-30); ph = jnp.arctan2(im, re); W = jnp.exp(1j * ph * NV)[:, None] * VD
    return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T
gray = lambda i: i ^ (i >> 1)
tgt = np.zeros(NF, complex)
for f in range(8): tgt[f] = (-1j) ** bin(gray(f)).count("1") / np.sqrt(8)
def state(x, k):
    x = x.reshape(k, 2 + NF); psi = jnp.zeros(NF, complex).at[0].set(1)
    for i in range(k): psi = disp(x[i, 0], x[i, 1]) @ psi; psi = psi * jnp.exp(-1j * x[i, 2:])
    return psi
out = {}
for k in [1, 2, 3, 4, 5, 6]:
    best = None
    for mode in ["mag", "exact"]:
        def L(x):
            psi = state(x, k)
            return -(jnp.sum(jnp.abs(psi[:8])) / np.sqrt(8)) ** 2 if mode == "mag" else -jnp.abs(jnp.vdot(tgt, psi)) ** 2
        vg = jax.jit(jax.value_and_grad(L)); rng = np.random.default_rng(k)
        res = min((minimize(lambda x: tuple(np.asarray(v) for v in vg(x)), np.concatenate([[rng.uniform(0.3, 1.5), rng.uniform(-1, 1)] + list(rng.uniform(0, 6.28, NF)) for _ in range(k)]),
                            jac=True, method="L-BFGS-B", options={"maxiter": 3000}) for _ in range(20)), key=lambda r: r.fun)
        psi = np.asarray(state(res.x, k)); out[f"k{k}_{mode}"] = {"F": float(-res.fun), "leak": float(np.sum(np.abs(psi[8:]) ** 2)), "alpha": [float(np.hypot(*res.x.reshape(k, -1)[i, :2])) for i in range(k)]}
        print(k, mode, out[f"k{k}_{mode}"], flush=True)
json.dump(out, open("results/v3/sd_n6/prep_fit.json", "w"), indent=1)
