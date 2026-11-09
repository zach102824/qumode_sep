"""Discovery spin-glass comparison. Reads the frozen manifests. Does not redraw J."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nov_3rd.tfim_shared_gamma.round1_A.common import sha256_state
from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum
from Nov_3rd.tfim_shared_gamma.round4_N.model import (
    block_coeff,
    fwht,
    hat,
    pair_K,
    pair_diag,
    rms,
    shell_weights,
    upper,
    z_table,
)

HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "eb6ecec92efa50b386077555fd7de00d7a474666be816f84f2e21b3825598821"
FAMILIES = ("nodata", "K", "KM", "qaoa")
SCALAR = (-12.0, 12.0)
SHELL = (-2.0 * np.pi, 2.0 * np.pi)


def _check():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")


def load_instance(N: int, k: int):
    doc = json.loads((HERE / "manifests" / f"n{N}_k{k}.json").read_text())
    J = np.zeros((N, N), dtype=np.float64)
    for i, j, s in doc["J_upper"]:
        J[int(i), int(j)] = J[int(j), int(i)] = float(s)
    raw = np.ascontiguousarray(J, dtype="<f8").tobytes()
    if hashlib.sha256(raw).hexdigest() != doc["J_sha256_f64le"]:
        # decimal reload can move the last bit; rebuild from the seed instead
        rng = np.random.default_rng(np.random.SeedSequence(doc["seed"]))
        J = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                v = float(rng.normal())
                J[i, j] = J[j, i] = v
        raw = np.ascontiguousarray(J, dtype="<f8").tobytes()
        if hashlib.sha256(raw).hexdigest() != doc["J_sha256_f64le"]:
            raise RuntimeError(f"manifest drift N={N} k={k}")
    return doc, J


def packing(N: int):
    m = N // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


class Inst:
    def __init__(self, N, k, J, doc):
        self.N = N
        self.k = k
        self.doc = doc
        self.J = J
        self.B = 0.1
        self.z = z_table(N)
        self.hc = pair_diag(J, self.z)
        self.rms_J = rms(upper(J))
        self.K_hat, self.scale_K = hat(pair_K(J), self.rms_J)
        self.M_hat, self.scale_M = hat(block_coeff(J, packing(N)), self.rms_J)
        self.dK = pair_diag(self.K_hat, self.z)
        self.dM = pair_diag(self.M_hat, self.z)
        self.w1, self.w2, self.w3 = shell_weights(N, packing(N))
        dim = 1 << N
        idx = np.arange(dim)
        self.wX = np.zeros(dim)
        for q in range(N):
            self.wX += 1.0 - 2.0 * ((idx >> q) & 1)
        self.plus = np.full(dim, 2.0 ** (-0.5 * N))
        self.gs = None
        self.E0 = None
        self.Emax = None


def nparam(fam, L):
    if fam == "qaoa":
        return 2 * L
    extra = {"nodata": 0, "K": 1, "KM": 2}[fam]
    return 1 + extra + 3 * L


def unpack(fam, x, L):
    x = np.asarray(x, dtype=np.float64)
    if fam == "qaoa":
        return x[:L].copy(), {}, x[L:].copy()
    if fam == "nodata":
        return float(x[0]), {}, x[1:].reshape(L, 3).copy()
    if fam == "K":
        return float(x[0]), {"K": float(x[1])}, x[2:].reshape(L, 3).copy()
    return float(x[0]), {"K": float(x[1]), "M": float(x[2])}, x[3:].reshape(L, 3).copy()


def pack(fam, gamma, theta, cs, gammas=None, betas=None):
    if fam == "qaoa":
        return np.concatenate([np.asarray(gammas, float), np.asarray(betas, float)])
    head = [float(gamma)]
    if fam in ("K", "KM"):
        head.append(float(theta.get("K", 0.0)))
    if fam == "KM":
        head.append(float(theta.get("M", 0.0)))
    return np.concatenate([head, np.asarray(cs, float).reshape(-1)])


def g_of(inst, fam, theta):
    g = inst.hc.copy()
    if "K" in theta:
        g = g + float(theta["K"]) * inst.dK
    if "M" in theta:
        g = g + float(theta["M"]) * inst.dM
    return g


def forward(inst, fam, x, L, store=False):
    gamma, theta, cs = unpack(fam, x, L)
    if fam == "qaoa":
        gammas = gamma
        g = inst.hc
        w1, w2, w3 = inst.wX, None, None
    else:
        gammas = np.full(L, gamma)
        g = g_of(inst, fam, theta)
        w1, w2, w3 = inst.w1, inst.w2, inst.w3
    psi = inst.plus.astype(np.complex128).copy()
    snaps = []
    for ell in range(L):
        phase = float(gammas[ell]) * g
        mid = psi * np.exp(-1j * phase)
        if fam == "qaoa":
            phi = cs[ell] * w1
        else:
            phi = cs[ell, 0] * w1 + cs[ell, 1] * w2 + cs[ell, 2] * w3
        x2 = fwht(mid) * np.exp(-1j * phi)
        psi = fwht(x2)
        if store:
            snaps.append((phase, mid, phi, x2, g, w1, w2, w3, float(gammas[ell])))
    return psi, snaps


def energy_psi(inst, psi):
    Hpsi = inst.hc * psi + inst.B * apply_X_sum(psi, inst.N)
    return float(np.vdot(psi, Hpsi).real), Hpsi


def energy_grad(inst, fam, x, L):
    psi, snaps = forward(inst, fam, x, L, store=True)
    E, Hpsi = energy_psi(inst, psi)
    lam = Hpsi
    g_gamma = np.zeros(L)
    g_c = np.zeros((L, 3))
    g_th = {"K": 0.0, "M": 0.0}
    for ell in range(L - 1, -1, -1):
        phase, mid, phi, x2, g, w1, w2, w3, gam = snaps[ell]
        lam_x = fwht(lam)
        ov = np.conj(lam_x) * x2
        if fam == "qaoa":
            g_c[ell, 0] = 2.0 * float(np.imag(np.sum(ov * w1)))
        else:
            g_c[ell, 0] = 2.0 * float(np.imag(np.sum(ov * w1)))
            g_c[ell, 1] = 2.0 * float(np.imag(np.sum(ov * w2)))
            g_c[ell, 2] = 2.0 * float(np.imag(np.sum(ov * w3)))
        lam_mid = fwht(lam_x * np.exp(1j * phi))
        ovD = np.conj(lam_mid) * mid
        g_gamma[ell] = 2.0 * float(np.imag(np.sum(ovD * g)))
        if fam in ("K", "KM"):
            g_th["K"] += 2.0 * float(np.imag(np.sum(ovD * (gam * inst.dK))))
        if fam == "KM":
            g_th["M"] += 2.0 * float(np.imag(np.sum(ovD * (gam * inst.dM))))
        lam = lam_mid * np.exp(1j * phase)
    if fam == "qaoa":
        grad = np.concatenate([g_gamma, g_c[:, 0]])
    elif fam == "nodata":
        grad = np.concatenate([[g_gamma.sum()], g_c.reshape(-1)])
    elif fam == "K":
        grad = np.concatenate([[g_gamma.sum(), g_th["K"]], g_c.reshape(-1)])
    else:
        grad = np.concatenate([[g_gamma.sum(), g_th["K"], g_th["M"]], g_c.reshape(-1)])
    return E, grad, psi


def clip(fam, x, L):
    y = np.asarray(x, float).copy()
    if fam == "qaoa":
        for i in range(L):
            y[i] = min(max(y[i], SCALAR[0]), SCALAR[1])
        for i in range(L, 2 * L):
            y[i] = min(max(y[i], SHELL[0]), SHELL[1])
        return y
    nhead = 1 + {"nodata": 0, "K": 1, "KM": 2}[fam]
    for i in range(nhead):
        y[i] = min(max(y[i], SCALAR[0]), SCALAR[1])
    for i in range(nhead, y.size):
        y[i] = min(max(y[i], SHELL[0]), SHELL[1])
    return y


def bounds(fam, L):
    if fam == "qaoa":
        return [SCALAR] * L + [SHELL] * L
    nhead = 1 + {"nodata": 0, "K": 1, "KM": 2}[fam]
    return [SCALAR] * nhead + [SHELL] * (3 * L)


def rng_for(N, k, fam, L, s):
    fid = {"nodata": 0, "K": 1, "KM": 2, "qaoa": 3, "data": 4}[fam]
    return np.random.default_rng(np.random.SeedSequence([20261116, int(N), int(k), fid, int(L), int(s)]))


def l1_seed(fam, N, k, s):
    rng = rng_for(N, k, fam, 1, s)
    if fam == "qaoa":
        return np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)])
    if s == 0:
        cs = np.zeros((1, 3)); cs[0, 0] = 0.2
        th = {"K": 0.0, "M": 0.0}
        return pack(fam, 0.3, th, cs)
    if s == 1:
        cs = np.zeros((1, 3)); cs[0, 0] = -0.2
        th = {"K": 0.0, "M": 0.0}
        return pack(fam, -0.3, th, cs)
    cs = rng.uniform(-0.6, 0.6, size=(1, 3))
    th = {"K": float(rng.uniform(-1, 1)), "M": float(rng.uniform(-1, 1))}
    return pack(fam, float(rng.uniform(-0.6, 0.6)), th, cs)


def l1_digest():
    acc = []
    for N in (6, 9, 12):
        for k in (0, 1, 2, 3, 100, 101):
            for fam in FAMILIES:
                for s in range(6):
                    acc.append(l1_seed(fam, N, k, s))
    raw = np.concatenate(acc).astype("<f8").tobytes()
    return hashlib.sha256(raw).hexdigest()


def resample(values, p_new):
    values = np.asarray(values, float).reshape(-1)
    p_old = values.size
    if p_old == 1:
        return np.full(p_new, float(values[0]))
    xs = np.arange(p_old) / (p_old - 1)
    xt = np.arange(p_new) / (p_new - 1)
    if p_old < 3:
        return np.interp(xt, xs, values)
    spline = CubicSpline(xs, values, bc_type="natural")
    return np.asarray(spline(np.clip(xt, xs[0], xs[-1])), float)


def cont_seeds(fam, prev, L, N, k):
    """Four continuation starts. Interpolation is an initialization only."""
    out = []
    if fam == "qaoa":
        g, _, b = unpack("qaoa", prev, L - 1)
        b = np.asarray(b, float)
        if L == 2:
            base = pack("qaoa", 0, {}, None, [g[0], g[0]], [b[0], b[0]])
            pag = pack("qaoa", 0, {}, None, [g[0], g[0] + 0.2], [b[0], b[0] - 0.2])
            out.append(("copy", base))
            out.append(("pagano_unconverted_shape", pag))
        else:
            kind = "linear" if L == 3 else "cubic"
            base = pack("qaoa", 0, {}, None, resample(g, L), resample(b, L))
            t = np.arange(L) / max(L - 1, 1)
            shape = np.sin(np.pi * t)
            out.append((f"interp_{kind}", base))
            out.append(("fourier1", pack("qaoa", 0, {}, None, float(np.max(np.abs(g))) * shape, float(np.max(np.abs(b))) * shape)))
    else:
        gamma, theta, cs = unpack(fam, prev, L - 1)
        if L == 2:
            base = pack(fam, gamma, theta, np.vstack([cs[0], cs[0]]))
            row = cs[0].copy(); row[0] = row[0] - 0.2
            pag = pack(fam, gamma, theta, np.vstack([cs[0], row]))
            out.append(("copy", base))
            out.append(("pagano_unconverted_shape", pag))
        else:
            kind = "linear" if L == 3 else "cubic"
            base = pack(fam, gamma, theta, np.column_stack([resample(cs[:, j], L) for j in range(3)]))
            out.append((f"interp_{kind}", base))
            t = np.arange(L) / max(L - 1, 1)
            shape = np.sin(np.pi * t)
            csf = np.column_stack([float(np.max(np.abs(cs[:, j]))) * shape for j in range(3)])
            out.append(("fourier1", pack(fam, gamma, theta, csf)))
    for s, sigma in ((2, 0.05), (3, 0.05)):
        rng = rng_for(N, k, fam, L, s)
        out.append((f"perturb_{sigma}_{s}", out[0][1] + rng.normal(scale=sigma, size=out[0][1].shape)))
    return out


def spin_parity(v):
    n = int(np.log2(v.size))
    flip = np.arange(v.size) ^ ((1 << n) - 1)
    return float(np.vdot(v, v[flip]).real)


def gap_scan(inst: Inst):
    N = inst.N
    spec = PREREG["gap_grid"][str(N)]
    if spec["kind"] == "author_step":
        Bs = np.arange(0.1, 10.0, 1e-4)
    elif spec["kind"] == "step":
        Bs = np.arange(0.1, 10.0 + 1e-12, float(spec["step"]))
    else:
        Bs = np.array(spec["values"], float)
    dim = 1 << N
    idx = np.arange(dim)
    # reuse ZZ diagonal; X matrix once
    X = np.zeros((dim, dim))
    for q in range(N):
        partner = idx ^ (1 << q)
        X[idx, partner] += 1.0
    ord_g, cou_g = [], []
    gs_target = None
    evals_target = None
    for ib, B in enumerate(Bs):
        H = np.diag(inst.hc) + float(B) * X
        evals, vecs = np.linalg.eigh(H)
        p0 = round(spin_parity(vecs[:, 0]), 5)
        coupled = None
        for i in range(1, dim):
            if round(spin_parity(vecs[:, i]), 5) == p0:
                coupled = float(evals[i] - evals[0])
                break
        ord_g.append(float(evals[1] - evals[0]))
        cou_g.append(coupled)
        if ib == 0:
            gs = np.real(vecs[:, 0]); gs /= np.linalg.norm(gs)
            if gs.sum() < 0:
                gs = -gs
            gs_target = gs
            evals_target = evals
    ord_g = np.array(ord_g, float)
    cou_g = np.array(cou_g, float)
    ic = int(np.nanargmin(cou_g))
    io = int(np.argmin(ord_g))
    inst.gs = gs_target
    inst.E0 = float(evals_target[0])
    inst.Emax = float(evals_target[-1])
    return {
        "N": N, "k": inst.k, "grid": spec,
        "n_B": int(Bs.size),
        "target_B": 0.1,
        "ordinary_at_target": float(ord_g[int(np.argmin(np.abs(Bs - 0.1)))]),
        "coupled_at_target": float(cou_g[int(np.argmin(np.abs(Bs - 0.1)))]),
        "min_ordinary": float(ord_g[io]),
        "B_min_ordinary": float(Bs[io]),
        "min_coupled": float(cou_g[ic]),
        "B_min_coupled": float(Bs[ic]),
        "coupled_min_on_boundary": bool(ic == 0 or ic == len(Bs) - 1),
        "E0": inst.E0, "Emax": inst.Emax,
        "gs_parity_rounded": round(spin_parity(inst.gs), 5),
        "gs_residual": None,
    }


def one_start(inst, fam, x0, L, maxfun, lock=None):
    x0 = clip(fam, x0, L)
    bnds = bounds(fam, L)
    if lock:
        for i, value in lock.items():
            x0[i] = float(value)
            bnds[i] = (float(value), float(value))
    local_E, local_x, local_psi = float("inf"), x0.copy(), None
    cache = {}
    used = 0
    reason = "optimizer_return"

    def take():
        nonlocal used
        if used >= maxfun:
            raise StopIteration("ceiling")
        used += 1

    def record(x, E, grad, psi):
        nonlocal local_E, local_x, local_psi
        cache["x"] = np.array(x, copy=True)
        cache["g"] = grad
        if E < local_E:
            local_E, local_x, local_psi = float(E), np.array(x, copy=True), psi

    def fun(x):
        take()
        E, grad, psi = energy_grad(inst, fam, x, L)
        record(x, E, grad, psi)
        return float(E)

    def jac(x):
        if cache.get("x") is not None and np.allclose(cache["x"], x, atol=1e-12, rtol=0):
            return cache["g"]
        take()
        E, grad, psi = energy_grad(inst, fam, x, L)
        record(x, E, grad, psi)
        return grad

    t0 = time.perf_counter()
    success = False
    try:
        res = minimize(fun, x0, method="L-BFGS-B", jac=jac, bounds=bnds,
                       options={"maxiter": 400, "maxfun": maxfun, "ftol": 1e-12, "gtol": 1e-8})
        success = bool(res.success)
    except StopIteration:
        reason = "start_ceiling"
    F = eta = var = None
    sha = None
    if local_psi is not None:
        E, Hpsi = energy_psi(inst, local_psi)
        var = float(np.vdot(Hpsi, Hpsi).real - E * E)
        F = float(abs(np.vdot(inst.gs, local_psi)) ** 2)
        eta = (E - inst.Emax) / (inst.E0 - inst.Emax)
        sha = sha256_state(local_psi)
        local_E = E
    return {
        "E": None if local_psi is None else float(local_E),
        "F": F, "eta": None if eta is None else float(eta), "var_H": var,
        "state_sha256": sha,
        "x_final": None if local_psi is None else [float(v) for v in local_x],
        "n_eval": int(used), "optimizer_success": success, "stop_reason": reason,
        "seconds": time.perf_counter() - t0,
    }


def best_of(rows):
    finite = [r for r in rows if r.get("E") is not None]
    if not finite:
        return None
    return min(finite, key=lambda r: r["E"])


def grad_check(inst):
    fam, L = "KM", 2
    rng = np.random.default_rng(0)
    x = clip(fam, rng.normal(scale=0.2, size=nparam(fam, L)), L)
    E, grad, psi = energy_grad(inst, fam, x, L)
    errs = []
    for i in range(x.size):
        step = np.zeros_like(x); step[i] = 1e-6
        Ep, _, _ = energy_grad(inst, fam, x + step, L)
        Em, _, _ = energy_grad(inst, fam, x - step, L)
        errs.append(abs((Ep - Em) / 2e-6 - grad[i]))
    # GS residual
    Hpsi = inst.hc * inst.gs + inst.B * apply_X_sum(inst.gs, inst.N)
    gs_res = float(np.linalg.norm(Hpsi - inst.E0 * inst.gs))
    # replay
    E2, _, psi2 = energy_grad(inst, fam, x, L)
    return {
        "grad_max_abs": float(max(errs)),
        "gs_residual": gs_res,
        "replay_energy_abs": abs(E2 - E),
        "replay_state_overlap": float(abs(np.vdot(psi, psi2))),
        "norm": float(np.linalg.norm(psi)),
    }


def run_instance(inst, fams, logf, deadline, counters):
    selected = {fam: None for fam in fams}
    for L in range(1, 7):
        if time.perf_counter() >= deadline:
            return selected, "wall"
        for fam in fams:
            if L == 1:
                seeds = [(f"l1_{s}", l1_seed(fam, inst.N, inst.k, s)) for s in range(6)]
            else:
                prev = selected[fam]
                if prev is None or prev.get("x_final") is None:
                    continue
                seeds = cont_seeds(fam, np.array(prev["x_final"], float), L, inst.N, inst.k)
            rows = []
            for label, x0 in seeds:
                if time.perf_counter() >= deadline:
                    break
                row = one_start(inst, fam, x0, L, int(PREREG["maxfun"]))
                row.update({"N": inst.N, "k": inst.k, "role": inst.doc["role"], "family": fam, "L": L, "label": label})
                rows.append(row)
                counters["evals"] += row["n_eval"]
                counters["starts"] += 1
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                print(f"N{inst.N} k{inst.k} {fam} L{L} {label} E={row['E']} F={row['F']} n={row['n_eval']}", flush=True)
            chosen = best_of(rows)
            if chosen is not None:
                selected[fam] = chosen
        if time.perf_counter() >= deadline:
            return selected, "wall"
    return selected, "complete"


def summarize(records, gaps):
    # energy-selected per N,k,family,L is the last appended best? records are all starts.
    # Recompute selected as min E among starts at that cell.
    cells = {}
    for r in records:
        key = (r["N"], r["k"], r["family"], r["L"])
        cells.setdefault(key, []).append(r)
    rows = []
    for key, group in sorted(cells.items()):
        b = best_of(group)
        rows.append({
            "N": key[0], "k": key[1], "family": key[2], "L": key[3],
            "E": b["E"], "F": b["F"], "eta": b["eta"], "var_H": b["var_H"],
            "state_sha256": b["state_sha256"], "x_final": b["x_final"],
            "n_starts": len(group), "n_eval": int(sum(g["n_eval"] for g in group)),
            "n_not_success": int(sum(1 for g in group if not g["optimizer_success"])),
            "label": b["label"],
        })
    (HERE / "selected.json").write_text(json.dumps(rows) + "\n")
    # equal-depth table: KM and qaoa F
    lines = ["# Spin glass round 1", "", "Energy-selected states. Fidelity was not used to select.", ""]
    for N in (6, 9, 12):
        lines.append(f"## N={N}")
        lines.append("")
        lines.append("| k | L | family | E | F | eta | starts | evals |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["N"] == N:
                lines.append(f"| {r['k']} | {r['L']} | {r['family']} | {r['E']:.10f} | {r['F']:.10f} | {r['eta']:.10f} | {r['n_starts']} | {r['n_eval']} |")
        lines.append("")
    both = []
    for N in (6, 9, 12):
        for k in sorted({r["k"] for r in rows if r["N"] == N}):
            for L in range(1, 7):
                km = next((r for r in rows if r["N"] == N and r["k"] == k and r["L"] == L and r["family"] == "KM"), None)
                q = next((r for r in rows if r["N"] == N and r["k"] == k and r["L"] == L and r["family"] == "qaoa"), None)
                if km and q:
                    both.append((N, k, L, km["E"] < q["E"] - 1e-9 and km["F"] > q["F"] + 1e-9, q["E"] < km["E"] - 1e-9 and q["F"] > km["F"] + 1e-9))
    n_km = sum(1 for t in both if t[3])
    n_q = sum(1 for t in both if t[4])
    lines.append(f"Equal-depth KM versus QAOA cells with both metrics: {len(both)}.")
    lines.append(f"KM lower energy and higher fidelity: {n_km}. QAOA lower energy and higher fidelity: {n_q}. Other cells are not both-metric wins.")
    lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("| N | k | ordinary at B=0.1 | coupled at B=0.1 | min coupled | B | boundary | grid |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g in gaps:
        lines.append(f"| {g['N']} | {g['k']} | {g['ordinary_at_target']:.6e} | {g['coupled_at_target']:.6e} | {g['min_coupled']:.6e} | {g['B_min_coupled']:.4f} | {g['coupled_min_on_boundary']} | {g['grid']['kind']} |")
    lines.append("")
    lines.append("The gap columns are exploratory. They were not used to drop instances. A smaller gap is not a cause of the fidelity.")
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    # plot F of KM and qaoa
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for ax, N in zip(axes, (6, 9, 12)):
        for fam, marker in (("KM", "o"), ("qaoa", "x"), ("nodata", "+"), ("K", "s")):
            xs, ys = [], []
            for L in range(1, 7):
                vals = [r["F"] for r in rows if r["N"] == N and r["family"] == fam and r["L"] == L and r["k"] < 100]
                if vals:
                    xs.append(L); ys.append(float(np.mean(vals)))
            if xs:
                ax.plot(xs, ys, marker=marker, label=fam)
        ax.set_title(f"N={N}")
        ax.set_xlabel("L")
    axes[0].set_ylabel("mean discovery fidelity")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fidelity_vs_L.png", dpi=120)
    plt.close(fig)


def main():
    _check()
    if l1_digest() != PREREG["l1_seed_sha256"]:
        raise SystemExit("L1 seed digest drifted")
    deadline = time.perf_counter() + float(PREREG["wall_seconds"])
    log_path = HERE / "starts.jsonl"
    if log_path.exists():
        raise SystemExit("starts.jsonl exists; refuse to mix a second run")
    counters = {"evals": 0, "starts": 0}
    checks = []
    gaps = []
    # build discovery first
    discovery = []
    for N in (6, 9, 12):
        for k in PREREG["discovery_k"]:
            doc, J = load_instance(N, k)
            inst = Inst(N, k, J, doc)
            print(f"gap N{N} k{k}", flush=True)
            g = gap_scan(inst)
            # gs residual
            Hpsi = inst.hc * inst.gs + inst.B * apply_X_sum(inst.gs, inst.N)
            g["gs_residual"] = float(np.linalg.norm(Hpsi - inst.E0 * inst.gs))
            gaps.append(g)
            (HERE / "gaps.json").write_text(json.dumps(gaps, indent=2) + "\n")
            discovery.append(inst)
            if N == 6 and k == 0:
                checks.append(grad_check(inst))
                (HERE / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
                print("checks", checks[-1], flush=True)
    with log_path.open("w") as logf:
        status = {}
        for inst in discovery:
            _, st = run_instance(inst, FAMILIES, logf, deadline, counters)
            status[f"N{inst.N}_k{inst.k}"] = st
            print("status", inst.N, inst.k, st, "evals", counters["evals"], flush=True)
        # data-learned thetas from discovery KM at the deepest common L, per N
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        data_theta = {}
        for N in (6, 9, 12):
            deepest = 0
            for L in range(1, 7):
                if all(any(r["N"] == N and r["k"] == k and r["family"] == "KM" and r["L"] == L and r["E"] is not None for r in records) for k in PREREG["discovery_k"]):
                    deepest = L
            thetas = []
            if deepest:
                for k in PREREG["discovery_k"]:
                    group = [r for r in records if r["N"] == N and r["k"] == k and r["family"] == "KM" and r["L"] == deepest]
                    b = best_of(group)
                    x = np.array(b["x_final"], float)
                    _, th, _ = unpack("KM", x, deepest)
                    thetas.append(th)
                data_theta[str(N)] = {
                    "L_source": deepest,
                    "theta_K": float(np.mean([t["K"] for t in thetas])),
                    "theta_M": float(np.mean([t["M"] for t in thetas])),
                    "n_discovery": len(thetas),
                    "pretraining_evals_already_in_KM": int(sum(r["n_eval"] for r in records if r["N"] == N and r["family"] == "KM")),
                }
        (HERE / "data_theta.json").write_text(json.dumps(data_theta, indent=2) + "\n")
        # held-out if time: KM, qaoa, nodata, and data-frozen thetas
        for N in (6, 9, 12):
            for k in PREREG["heldout_k"]:
                if time.perf_counter() >= deadline:
                    break
                doc, J = load_instance(N, k)
                inst = Inst(N, k, J, doc)
                g = gap_scan(inst)
                Hpsi = inst.hc * inst.gs + inst.B * apply_X_sum(inst.gs, inst.N)
                g["gs_residual"] = float(np.linalg.norm(Hpsi - inst.E0 * inst.gs))
                gaps.append(g)
                (HERE / "gaps.json").write_text(json.dumps(gaps, indent=2) + "\n")
                _, st = run_instance(inst, ("nodata", "KM", "qaoa"), logf, deadline, counters)
                status[f"held_N{N}_k{k}"] = st
                if str(N) in data_theta and time.perf_counter() < deadline:
                    th = data_theta[str(N)]
                    # one family: thetas frozen, gamma and shell free. Counted separately.
                    _run_frozen_theta(inst, th, logf, deadline, counters)
        (HERE / "status.json").write_text(json.dumps({"status": status, "counters": counters}, indent=2) + "\n")
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    summarize(records, gaps)
    print("done", counters, flush=True)


def _run_frozen_theta(inst, th, logf, deadline, counters):
    """Held-out only. Thetas come from discovery KM. Gamma and shell are fit."""
    fam = "data"
    selected = None
    for L in range(1, 7):
        if time.perf_counter() >= deadline:
            return
        seeds = []
        if L == 1:
            for s in range(6):
                rng = rng_for(inst.N, inst.k, fam, 1, s)
                cs = np.zeros((1, 3)) if s == 0 else rng.uniform(-0.6, 0.6, size=(1, 3))
                if s == 0:
                    cs[0, 0] = 0.2
                gamma = 0.3 if s == 0 else float(rng.uniform(-0.6, 0.6))
                seeds.append((f"l1_{s}", pack("KM", gamma, {"K": th["theta_K"], "M": th["theta_M"]}, cs)))
        else:
            if selected is None:
                return
            xprev = np.array(selected["x_final"], float)
            gamma, _, cs = unpack("KM", xprev, L - 1)
            # freeze thetas even if the previous optimize moved them: reset to discovery values
            if L == 2:
                base_cs = np.vstack([cs[0], cs[0]])
            else:
                base_cs = np.column_stack([resample(cs[:, j], L) for j in range(3)])
            base = pack("KM", gamma, {"K": th["theta_K"], "M": th["theta_M"]}, base_cs)
            seeds.append(("interp_frozen_theta", base))
            for s in (1, 2, 3):
                rng = rng_for(inst.N, inst.k, fam, L, s)
                noise = rng.normal(scale=0.05, size=base.shape)
                # do not perturb the two theta coordinates
                noise[1] = 0.0
                noise[2] = 0.0
                seeds.append((f"perturb_{s}", base + noise))
        rows = []
        for label, x0 in seeds:
            if time.perf_counter() >= deadline:
                break
            # project thetas back after the step by wrapping one_start: clip may move them
            x0 = np.array(x0, float)
            x0[1] = th["theta_K"]
            x0[2] = th["theta_M"]
            row = one_start(inst, "KM", x0, L, int(PREREG["maxfun"]),
                            lock={1: th["theta_K"], 2: th["theta_M"]})
            row.update({"N": inst.N, "k": inst.k, "role": "heldout", "family": "data", "L": L, "label": label,
                        "theta_source": "discovery_KM_mean"})
            rows.append(row)
            counters["evals"] += row["n_eval"]
            counters["starts"] += 1
            logf.write(json.dumps(row) + "\n")
            logf.flush()
        selected = best_of(rows)


if __name__ == "__main__":
    main()
