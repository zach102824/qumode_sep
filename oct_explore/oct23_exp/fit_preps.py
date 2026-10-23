"""Fit 1-block, 2-block, and 3-block flat state preps from vacuum for K=2 (D=4), K=3 (D=8), K=4 (D=16)
using both:
  1. SNAP+D (`sd_K*_k*`): D(alpha_i) -> SNAP(theta_i), where SNAP phases are RESTRICTED to computational
     Fock levels 0..D-1 (strictly 0 phase on leaked levels f >= D), plus penalizing intermediate leakage.
  2. ECD (`ecd_K*_k*` and `ecdt_K*_k*`): R(theta_i, phi_i) -> ECD(beta_i) from |g>|0>, both for
     ancilla transmon (target |g> x flat(0..D-1)) and data-bit transmon (target (|g>+|e>)/sqrt(2) x flat(0..D-1)).
Also verifies truncation convergence between NF=40 and NF=64."""
import os, json, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)

OUT_FILE = "/workspace/qumode_sep/oct_explore/results/oct23_exp/preps.json"

def make_disp(nf):
    aa = np.diag(np.sqrt(np.arange(1, nf)), 1)
    lam, V = np.linalg.eigh(1j * (aa.T - aa))
    LAM, VD, NV = jnp.asarray(lam), jnp.asarray(V), jnp.arange(nf)
    def disp(re, im):
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-30)
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * NV)[:, None] * VD
        return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T
    return disp

DISP_40 = make_disp(40)
DISP_64 = make_disp(64)

def fit_sd_prep(K, k, starts=32):
    N = 1 << K  # D = 2^K computational levels 0..N-1; leaked levels f >= N have strictly 0 SNAP phase
    def run_seq(x, nf, disp_fn):
        # x has shape (k, 2 + N): [re_alpha, im_alpha, theta_0 .. theta_{N-1}]
        x = x.reshape(k, 2 + N)
        psi = jnp.zeros(nf, complex).at[0].set(1.0)
        mid_leaks = []
        for i in range(k):
            psi = disp_fn(x[i, 0], x[i, 1]) @ psi
            mid_leaks.append(jnp.sum(jnp.abs(psi[N:]) ** 2))
            ph = jnp.concatenate([x[i, 2:], jnp.zeros(nf - N)])
            psi = psi * jnp.exp(-1j * ph)
        return psi, jnp.stack(mid_leaks)

    def loss(x):
        psi, mid_leaks = run_seq(x, 40, DISP_40)
        F_mag = (jnp.sum(jnp.abs(psi[:N])) / jnp.sqrt(N)) ** 2
        end_leak = jnp.sum(jnp.abs(psi[N:]) ** 2)
        return -F_mag + end_leak + 0.05 * jnp.mean(mid_leaks)

    vg = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(1000 * K + 10 * k)
    best = None
    for s in range(starts):
        x0 = np.concatenate([
            [rng.uniform(0.3, 1.8 + 0.3 * K), rng.uniform(-0.8, 0.8)] + list(rng.uniform(0, 2 * np.pi, N))
            for _ in range(k)
        ])
        r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(x)), x0, jac=True, method="L-BFGS-B", options={"maxiter": 2500})
        if best is None or r.fun < best.fun:
            best = r

    xr = best.x.reshape(k, 2 + N).copy()
    # Align final computational SNAP phases (0..N-1) sopsi[:N] is real >= 0
    psi40, mid40 = run_seq(xr.reshape(-1), 40, DISP_40)
    psi40 = np.asarray(psi40)
    xr[-1, 2:2+N] = (xr[-1, 2:2+N] + np.angle(psi40[:N])) % (2 * np.pi)

    psi40, mid40 = run_seq(xr.reshape(-1), 40, DISP_40)
    psi64, mid64 = run_seq(xr.reshape(-1), 64, DISP_64)
    psi40, mid40 = np.asarray(psi40), np.asarray(mid40)
    psi64, mid64 = np.asarray(psi64), np.asarray(mid64)

    F = float((np.sum(np.abs(psi40[:N])) / np.sqrt(N)) ** 2)
    F_exact = float(np.abs(np.sum(psi40[:N]) / np.sqrt(N)) ** 2)
    F_64 = float(np.abs(np.sum(psi64[:N]) / np.sqrt(N)) ** 2)
    leak = float(np.sum(np.abs(psi40[N:]) ** 2))
    leak_64 = float(np.sum(np.abs(psi64[N:]) ** 2))
    alphas = [[float(xr[i, 0]), float(xr[i, 1])] for i in range(k)]
    thetas = [xr[i, 2:2+N].tolist() for i in range(k)]
    print(f"SD prep K={K} (D={N}) k={k}: F={F:.6f} F_64={F_64:.6f} leak={leak:.6f} mid_leaks={[round(float(v), 4) for v in mid40]} |alpha|={[round(np.hypot(*a), 3) for a in alphas]}", flush=True)
    return {
        "gate": "sd", "K": K, "N": N, "k": k,
        "F": F, "F_exact": F_exact, "F_NF64": F_64,
        "leak": leak, "leak_NF64": leak_64,
        "mid_leaks": [float(v) for v in mid40],
        "alphas": alphas, "thetas": thetas,
        "psi_re": psi40.real.tolist(), "psi_im": psi40.imag.tolist()
    }

def fit_ecd_prep(K, k, tbit=False, starts=32):
    """ECD block: R(theta, phi) on transmon -> ECD(beta) = |e><g| D(beta/2) + |g><e| D(-beta/2)."""
    N = 1 << K
    def run_seq(x, nf, disp_fn):
        # x has shape (k, 4): [bre, bim, th, ph] where D(beta/2) has displacement (bre + i bim)
        x = x.reshape(k, 4)
        psi = jnp.zeros((2, nf), complex).at[0, 0].set(1.0)
        mid_leaks = []
        for i in range(k):
            bre, bim, th, ph = x[i, 0], x[i, 1], x[i, 2], x[i, 3]
            ct, st = jnp.cos(th / 2), jnp.sin(th / 2)
            e = jnp.exp(-1j * ph)
            p0 = ct * psi[0] - 1j * st * e * psi[1]
            p1 = -1j * st * jnp.conj(e) * psi[0] + ct * psi[1]
            Dp = disp_fn(bre, bim)
            Dm = Dp.conj().T
            psi = jnp.stack([Dm @ p1, Dp @ p0], axis=0)
            mid_leaks.append(jnp.sum(jnp.abs(psi[:, N:]) ** 2))
        return psi, jnp.stack(mid_leaks)

    def loss(x):
        psi, mid_leaks = run_seq(x, 40, DISP_40)
        if not tbit:
            # Ancilla transmon must end in |g> (index 0) and flat over 0..N-1
            F_mag = (jnp.sum(jnp.abs(psi[0, :N])) / jnp.sqrt(N)) ** 2
            leak = 1.0 - jnp.sum(jnp.abs(psi[0, :N]) ** 2)
        else:
            # Data transmon + cavity: flat over both transmon states (0, 1) and Fock 0..N-1 (total 2N states)
            F_mag = (jnp.sum(jnp.abs(psi[:, :N])) / jnp.sqrt(2 * N)) ** 2
            leak = 1.0 - jnp.sum(jnp.abs(psi[:, :N]) ** 2)
        return -F_mag + leak + 0.05 * jnp.mean(mid_leaks)

    vg = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(2000 * K + 100 * int(tbit) + k)
    best = None
    for s in range(starts):
        x0 = np.concatenate([
            [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)]
            for _ in range(k)
        ])
        r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(x)), x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r

    xr = best.x.reshape(k, 4).copy()
    psi40, mid40 = run_seq(xr.reshape(-1), 40, DISP_40)
    psi64, mid64 = run_seq(xr.reshape(-1), 64, DISP_64)
    psi40, mid40 = np.asarray(psi40), np.asarray(mid40)
    psi64, mid64 = np.asarray(psi64), np.asarray(mid64)

    if not tbit:
        F = float((np.sum(np.abs(psi40[0, :N])) / np.sqrt(N)) ** 2)
        F_64 = float((np.sum(np.abs(psi64[0, :N])) / np.sqrt(N)) ** 2)
        leak = float(1.0 - np.sum(np.abs(psi40[0, :N]) ** 2))
    else:
        F = float((np.sum(np.abs(psi40[:, :N])) / np.sqrt(2 * N)) ** 2)
        F_64 = float((np.sum(np.abs(psi64[:, :N])) / np.sqrt(2 * N)) ** 2)
        leak = float(1.0 - np.sum(np.abs(psi40[:, :N]) ** 2))
    kind = "ecdt" if tbit else "ecd"
    print(f"{kind.upper()} prep K={K} (D={N}, tbit={tbit}) k={k}: F={F:.6f} F_64={F_64:.6f} leak={leak:.6f}", flush=True)
    return {
        "gate": kind, "K": K, "N": N, "tbit": tbit, "k": k,
        "F": F, "F_NF64": F_64, "leak": leak,
        "mid_leaks": [float(v) for v in mid40],
        "params": xr.tolist(),
        "psi_re": psi40.real.tolist(), "psi_im": psi40.imag.tolist()
    }

if __name__ == "__main__":
    out = {}
    for K in [2, 3, 4]:
        for k in [1, 2, 3]:
            out[f"sd_K{K}_k{k}"] = fit_sd_prep(K, k)
            out[f"ecd_K{K}_k{k}"] = fit_ecd_prep(K, k, tbit=False)
            out[f"ecdt_K{K}_k{k}"] = fit_ecd_prep(K, k, tbit=True)
    json.dump(out, open(OUT_FILE, "w"))
    print("Saved all physical preps to", OUT_FILE)
