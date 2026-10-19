"""Milestone 1 (NATIVE_TEMPLATE plan, Zach 2026-10-19): calibrate a short native SNAP+displacement sandwich to the canonical
cavity mixer, formula-independent, once per cavity size.

Cavity with K bits in Gray code: level f <-> logical digits gray(f), N = 2^K valid levels, simulated on NF > N Fock levels.
Targets (N x N, on valid levels):
  gray : U(b) = prod_j exp(-i b X_j / 2), X_j flips logical digit j  (= qubit-QAOA X mixer exp(-i b/2 sum X) on the cavity's bits)
  tri  : U(b) = exp(-i b/2 H), H = sum_f |f><f+1| + h.c. (f < N-1)  (tridiagonal hop of the plan file, each edge = one Gray-bit flip)
Native sequence with k displacements (= k "SNAP+D pairs"):
  V = S_out D(a_k) S_{k-1} ... S_1 D(a_1) S_in
  S_in / S_out are diagonal and get MERGED into the adjacent compiled cost SNAPs (cost SNAP, bus phase and mixer-side SNAPs are all
  diagonal and commute), so they are free; internal SNAPs act on all NF levels.
Metrics: F = |Tr(U^dag P V P)|^2 / N^2 (unitary fidelity on valid levels, global phase free; leakage lowers F automatically);
  leak = 1 - ||P V P||_F^2 / N (mean population leaving the valid levels).
Loss = 1 - F + lam * leak. Symmetry: U(-b) = conj U(b) and conj D(a) = D(a*), conj SNAP(th) = SNAP(-th) -> fit b in [0, pi] only;
period 2 pi. Continuation: multistart at each grid point + warm starts from both neighbours, several sweeps; keep the best.
Truncation check: re-evaluate the fitted parameters at NF + 40 (internal SNAP phases on new levels = 0).
usage: mixcal.py <target gray|tri> <K> <k> [--nb 17] [--NF ...]
"""
import argparse, json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("K", type=int); ap.add_argument("k", type=int)
ap.add_argument("--nb", type=int, default=17); ap.add_argument("--NF", type=int, default=0); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--starts", type=int, default=12); ap.add_argument("--sweeps", type=int, default=2); ap.add_argument("--maxiter", type=int, default=4000)
ap.add_argument("--amax", type=float, default=0.0); ap.add_argument("--out", default="results/v3/native/mixcal")
a = ap.parse_args()
K, k = a.K, a.k; N = 1 << K; NF = a.NF or (N + max(16, N // 2 + 8))
gray = lambda i: i ^ (i >> 1)
def ungray(g):
    s = g >> 1
    while s: g ^= s; s >>= 1
    return g
def target(b):
    if a.target == "gray":
        U = np.eye(N, dtype=complex)
        for j in range(K):
            P = np.zeros((N, N));
            for f in range(N): P[ungray(gray(f) ^ (1 << j)), f] = 1
            U = (np.cos(b / 2) * np.eye(N) - 1j * np.sin(b / 2) * P) @ U
        return U
    H = np.diag(np.ones(N - 1), 1); H = H + H.T; w, V = np.linalg.eigh(H); return (V * np.exp(-1j * b / 2 * w)) @ V.conj().T
def dispmats(nf):
    aa = np.diag(np.sqrt(np.arange(1, nf)), 1); lam, V = np.linalg.eigh(1j * (aa.T - aa)); return jnp.asarray(lam), jnp.asarray(V), jnp.arange(nf)
def make(nf):
    LAM, VD, NV = dispmats(nf)
    def disp(re, im):
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-30); ph = jnp.arctan2(im, re); W = jnp.exp(1j * ph * NV)[:, None] * VD
        return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T
    def seq(x):  # x = [re,im]*k + internal snaps (k-1)*nf + s_in N + s_out N
        al = x[:2 * k].reshape(k, 2); o = 2 * k; Sn = x[o:o + (k - 1) * nf].reshape(max(k - 1, 0), nf); o += (k - 1) * nf
        sin, sout = x[o:o + N], x[o + N:o + 2 * N]
        M = jnp.zeros((nf, N), complex).at[:N].set(jnp.diag(jnp.exp(-1j * sin)))
        for i in range(k):
            re, im = al[i, 0], al[i, 1]
            if a.amax > 0:
                r = jnp.sqrt(re ** 2 + im ** 2 + 1e-30); s = a.amax * jnp.tanh(r / a.amax) / r; re, im = re * s, im * s
            M = disp(re, im) @ M
            if i < k - 1: M = jnp.exp(-1j * Sn[i])[:, None] * M
        return jnp.exp(-1j * sout)[:, None] * M[:N]          # P V P  (N x N)
    def metrics(x, U):
        Vp = seq(x); F = jnp.abs(jnp.trace(U.conj().T @ Vp)) ** 2 / N ** 2; leak = 1 - jnp.sum(jnp.abs(Vp) ** 2) / N
        return F, leak
    return seq, metrics
seq, metrics = make(NF)
def loss(x, U): F, lk = metrics(x, U); return 1 - F + a.lam * lk
vg = jax.jit(jax.value_and_grad(loss)); mj = jax.jit(metrics)
nx = 2 * k + (k - 1) * NF + 2 * N
def fit(x0, U):
    r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(x, U)), x0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": int(a.maxiter * 1.5)})
    if os.environ.get("DBG"): print(r.nit, r.message, r.fun)
    return r.x, float(r.fun)
def rand_x(rng, b):
    sc = 0.4 + 1.2 * rng.random()
    al = rng.normal(0, sc, 2 * k); return np.concatenate([al, rng.uniform(0, 2 * np.pi, (k - 1) * NF + 2 * N)])
betas = np.linspace(0, np.pi, a.nb)
os.makedirs(a.out, exist_ok=True); OUT = f"{a.out}/{a.target}_K{K}_k{k}.json"
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
t0 = time.time(); rng = np.random.default_rng(100 * K + k)
best = {}
Us = {i: jnp.asarray(target(b)) for i, b in enumerate(betas)}
for i, b in enumerate(betas):          # multistart pass
    cands = []
    if i == 0:
        x = np.zeros(nx); x[:2 * k] = 1e-3; cands.append(fit(x, Us[i]))      # b=0: identity is exact (a=0)
    for s in range(a.starts if i > 0 else 2): cands.append(fit(rand_x(rng, b), Us[i]))
    if i > 0: cands.append(fit(best[i - 1][0], Us[i]))
    best[i] = min(cands, key=lambda c: c[1]); log(f"pass0 b={b:.3f} loss {best[i][1]:.3e} ({len(cands)} cands)")
for sw in range(a.sweeps):              # continuation sweeps both directions
    order = list(range(1, len(betas))) + list(range(len(betas) - 2, 0, -1))
    for i in order:
        for j in (i - 1, i + 1):
            if 0 <= j < len(betas):
                c = fit(best[j][0], Us[i])
                if c[1] < best[i][1] - 1e-12: best[i] = c
    log(f"sweep {sw}: " + " ".join(f"{best[i][1]:.1e}" for i in range(len(betas))))
# report
seqB, metB = make(NF + 40); mjB = jax.jit(metB)
def padx(x):
    o = 2 * k; Sn = x[o:o + (k - 1) * NF].reshape(max(k - 1, 0), NF); Sn = np.concatenate([Sn, np.zeros((max(k - 1, 0), 40))], 1)
    return np.concatenate([x[:o], Sn.reshape(-1), x[o + (k - 1) * NF:]])
rows = []
for i, b in enumerate(betas):
    x = best[i][0]; F, lk = (float(v) for v in mj(x, Us[i])); FB, lkB = (float(v) for v in mjB(padx(x), Us[i]))
    al = x[:2 * k].reshape(k, 2); rows.append(dict(beta=float(b), F=F, leak=lk, F_NFplus40=FB, leak_NFplus40=lkB, alpha_abs=np.hypot(al[:, 0], al[:, 1]).tolist(), x=x.tolist()))
# interpolation check at midpoints (linear interpolation of parameters, wrapped phases not unwrapped -> pessimistic)
mid = []
for i in range(len(betas) - 1):
    bm = (betas[i] + betas[i + 1]) / 2; xm = (best[i][0] + best[i + 1][0]) / 2; Um = jnp.asarray(target(bm))
    Fm, lkm = (float(v) for v in mj(xm, Um)); xr, lr = fit(xm, Um); Fr, lkr = (float(v) for v in mj(xr, Um))
    mid.append(dict(beta=float(bm), F_interp=Fm, leak_interp=lkm, F_refit=Fr, leak_refit=lkr))
res = dict(args=vars(a), N=N, NF=NF, rows=rows, mid=mid, sec=time.time() - t0,
           Fmin=min(r["F"] for r in rows), Fmin_mid_refit=min(m["F_refit"] for m in mid), leakmax=max(r["leak"] for r in rows))
json.dump(res, open(OUT, "w"))
log(f"DONE {OUT}: Fmin grid {res['Fmin']:.5f} Fmin mid-refit {res['Fmin_mid_refit']:.5f} Fmin mid-interp {min(m['F_interp'] for m in mid):.4f} leakmax {res['leakmax']:.2e} "
    f"F(NF+40) min {min(r['F_NFplus40'] for r in rows):.5f} ({time.time()-t0:.0f}s)")
for r in rows: log(f"  b={r['beta']:.3f} F={r['F']:.6f} leak={r['leak']:.1e} F@NF+40={r['F_NFplus40']:.6f} |a|={np.round(r['alpha_abs'], 2).tolist()}")
