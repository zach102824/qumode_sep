"""N=9 stationary-gap diagnostics and labeled capacity pilots. Does not edit the prereg."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as h

ROOT = Path(__file__).resolve().parent
SIZE = ROOT.parent
OUT = ROOT / "gap"
B = float("-0.3444")
TAU = 1e-9
E0 = -6.337088433486192


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def energy_dense(H, psi):
    return float(np.vdot(psi, H @ psi).real)


def hess_from_grad(grad_fun, x, eps=1e-4):
    n = x.size
    H = np.zeros((n, n))
    g0 = grad_fun(x)
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        gp = grad_fun(x + e)
        gm = grad_fun(x - e)
        H[:, j] = (gp - gm) / (2 * eps)
    H = 0.5 * (H + H.T)
    return g0, H


def second_diff(energy, x, v, eps=1e-3):
    v = v / np.linalg.norm(v)
    return float((energy(x + eps * v) + energy(x - eps * v) - 2 * energy(x)) / eps**2)


def centers(n):
    return (np.arange(n) + 0.5) / n


def resample(shells, n_new):
    src = np.asarray(shells, dtype=np.float64)
    xs = centers(len(src))
    xt = np.clip(centers(n_new), xs[0], xs[-1])
    out = np.zeros((n_new, 3))
    cubic = len(src) >= 3
    for j in range(3):
        y = src[:, j]
        out[:, j] = CubicSpline(xs, y, bc_type="natural")(xt) if cubic else np.interp(xt, xs, y)
    return out


def lbfgs(fun, x0, maxfun, counter, ceiling):
    trace = []

    def wrapped(x):
        if counter[0] >= ceiling:
            raise RuntimeError("ceiling")
        counter[0] += 1
        E, g = fun(x)
        trace.append(float(E))
        return E, g

    try:
        res = minimize(wrapped, np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True,
                       options={"maxfun": maxfun, "maxiter": maxfun, "ftol": 1e-12})
    except RuntimeError:
        return None, trace
    if res.x is None:
        return None, trace
    return np.asarray(res.x, dtype=np.float64), trace


def grad_shared(target, x):
    psi, E, g, var = h.logical_grad(target, x)
    return E, g, psi, var


def grad_layer_gamma(target, x):
    L = (len(x) - 2) // 4
    gammas = x[:L]
    thk, thm = float(x[L]), float(x[L + 1])
    shells = np.asarray(x[L + 2:], dtype=np.float64).reshape(L, 3)
    G = target.g_diag(thk, thm)
    psi = h.plus(target.n)
    frames = []
    for ell, row in enumerate(shells):
        psi = psi * np.exp(-1j * float(gammas[ell]) * G)
        frames.append(("d", psi, ell, float(gammas[ell])))
        U, dUs = h.mixer(float(row[0]), float(row[1]), float(row[2]), True)
        for index in target.cav:
            psi = h.apply_cav(psi, U, index)
        frames.append(("s", psi, U, dUs, ell))
    Hpsi = target.hc * psi + target.B * h.apply_sum_x(psi, target.n)
    E = float(np.vdot(psi, Hpsi).real)
    var = float(np.vdot(Hpsi, Hpsi).real - E * E)
    lam = Hpsi
    grad = np.zeros(4 * L + 2)
    for kind, *rest in reversed(frames):
        if kind == "d":
            psi_d, ell, gamma = rest
            im = np.imag(np.conj(lam) * psi_d)
            grad[ell] += 2.0 * float(np.sum(G * im))
            grad[L] += 2.0 * float(np.sum((gamma * target.khat) * im))
            grad[L + 1] += 2.0 * float(np.sum((gamma * target.mhat) * im))
            lam = np.exp(1j * gamma * G) * lam
        else:
            psi_s, U, dUs, ell = rest
            for k in range(3):
                acc = np.zeros_like(psi_s)
                Va = dUs[k] @ np.conjugate(U).T
                for index in target.cav:
                    acc += h.apply_cav(psi_s, Va, index)
                grad[L + 2 + 3 * ell + k] += 2.0 * float(np.real(np.vdot(lam, acc)))
            for index in target.cav:
                lam = h.apply_cav(lam, np.conjugate(U).T, index)
    return E, grad, psi, var


def grad_layer_theta(target, x):
    L = (len(x) - 1) // 5
    gamma = float(x[0])
    thk = x[1:1 + L]
    thm = x[1 + L:1 + 2 * L]
    shells = np.asarray(x[1 + 2 * L:], dtype=np.float64).reshape(L, 3)
    psi = h.plus(target.n)
    frames = []
    for ell, row in enumerate(shells):
        G = target.g_diag(float(thk[ell]), float(thm[ell]))
        psi = psi * np.exp(-1j * gamma * G)
        frames.append(("d", psi, ell, G))
        U, dUs = h.mixer(float(row[0]), float(row[1]), float(row[2]), True)
        for index in target.cav:
            psi = h.apply_cav(psi, U, index)
        frames.append(("s", psi, U, dUs, ell))
    Hpsi = target.hc * psi + target.B * h.apply_sum_x(psi, target.n)
    E = float(np.vdot(psi, Hpsi).real)
    var = float(np.vdot(Hpsi, Hpsi).real - E * E)
    lam = Hpsi
    grad = np.zeros(1 + 5 * L)
    for kind, *rest in reversed(frames):
        if kind == "d":
            psi_d, ell, G = rest
            im = np.imag(np.conj(lam) * psi_d)
            grad[0] += 2.0 * float(np.sum(G * im))
            grad[1 + ell] += 2.0 * float(np.sum((gamma * target.khat) * im))
            grad[1 + L + ell] += 2.0 * float(np.sum((gamma * target.mhat) * im))
            lam = np.exp(1j * gamma * G) * lam
        else:
            psi_s, U, dUs, ell = rest
            for k in range(3):
                acc = np.zeros_like(psi_s)
                Va = dUs[k] @ np.conjugate(U).T
                for index in target.cav:
                    acc += h.apply_cav(psi_s, Va, index)
                grad[1 + 2 * L + 3 * ell + k] += 2.0 * float(np.real(np.vdot(lam, acc)))
            for index in target.cav:
                lam = h.apply_cav(lam, np.conjugate(U).T, index)
    return E, grad, psi, var


def grad_drop(target, y, which):
    if which == "K":
        x = np.concatenate([[y[0], 0.0, y[1]], y[2:]])
    else:
        x = np.concatenate([[y[0], y[1], 0.0], y[2:]])
    E, g, psi, var = grad_shared(target, x)
    if which == "K":
        g2 = np.concatenate([[g[0], g[2]], g[3:]])
    else:
        g2 = np.concatenate([[g[0], g[1]], g[3:]])
    return E, g2, psi, var


def contiguous_target():
    t = h.Target(9, B)
    packing = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    t.packing = packing
    t.M = h.block_mean_matrix(t.J, packing)
    t.rM = h.rms(h.upper(t.M)) / h.rms(h.upper(t.J))
    t.mhat_coeff = t.M / t.rM
    t.mhat = h.pair_diag(t.mhat_coeff, t.z)
    t.cav = [h.cavity_index(9, c) for c in packing]
    return t


def pack_state(target, spec, psi, E, H):
    Ed = energy_dense(H, psi)
    return {
        "E": float(E), "dense_E": Ed, "dense_delta": Ed - E,
        "error": float(E - spec["E0"]),
        "F": h.fidelity(spec["gs"], psi),
        "var": float(np.vdot(H @ psi, H @ psi).real - Ed * Ed),
        "parity": h.parity_weight(psi),
        "hash": h.state_hash(psi),
    }


def diagnose(target, spec, H, label, x):
    E, g, psi, var = grad_shared(target, x)
    info = pack_state(target, spec, psi, E, H)
    info["grad_norm"] = float(np.linalg.norm(g))
    info["label"] = label
    g0, Hess = hess_from_grad(lambda z: grad_shared(target, z)[1], x)
    evals = np.linalg.eigvalsh(Hess)
    info["hess_min"] = float(evals[0])
    info["hess_max"] = float(evals[-1])
    info["n_negative"] = int(np.sum(evals < -1e-4))
    info["n_small"] = int(np.sum(np.abs(evals) < 1e-2))
    info["grad_fd_mismatch"] = float(np.linalg.norm(g0 - g))
    rng = np.random.default_rng(0)
    curv = []
    for _ in range(6):
        v = rng.normal(size=x.size)
        num = second_diff(lambda z: grad_shared(target, z)[0], x, v)
        vv = v / np.linalg.norm(v)
        curv.append({"fd": num, "quad": float(vv @ Hess @ vv)})
    info["random_curvature"] = curv
    # step along the softest mode and reoptimize briefly
    vecs = np.linalg.eigh(Hess)[1]
    probes = []
    for sign, name in ((1.0, "soft_plus"), (-1.0, "soft_minus")):
        x2 = x + sign * 0.2 * vecs[:, 0]
        counter = [0]
        def fun(z):
            EE, gg, _, _ = grad_shared(target, z)
            return EE, gg
        xn, _ = lbfgs(fun, x2, 40, counter, 10**9)
        EE, gg, _, _ = grad_shared(target, xn)
        probes.append({"name": name, "error": EE - spec["E0"], "grad": float(np.linalg.norm(gg)), "evals": counter[0]})
    info["mode_probes"] = probes
    return info


def run_family(name, grad_fun, target, L, n_params_check, counter_cap, seed_head, n_chains, maxfun):
    assert grad_fun(target, np.zeros(n_params_check))[1].size == n_params_check
    counter = [0]
    rows = []
    x = None
    for k in range(n_chains):
        if counter[0] >= counter_cap:
            rows.append({"k": k, "censored": True})
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed_head, k]))
        # L=1 width
        p1 = n_params_check - 3 * (L - 1)
        x = rng.normal(0.0, 0.35, size=p1)
        for ell in range(1, L + 1):
            if ell > 1:
                # rebuild a full vector at ell-1 then resample shells only; scalars stay
                raise RuntimeError("use explicit layout")
        # layout-specific continuation is handled by the caller via grad_fun.x_layout
    return rows


def layout_L1(kind, L, rng):
    if kind == "shared":
        return rng.normal(0, 0.35, size=6)
    if kind == "layer_gamma":
        return rng.normal(0, 0.35, size=4 * 1 + 2)
    if kind == "layer_theta":
        return rng.normal(0, 0.35, size=1 + 5 * 1)
    if kind in ("drop_K", "drop_M"):
        return rng.normal(0, 0.35, size=5)
    raise KeyError(kind)


def grow(kind, x, L_new):
    """Continue one shared-style vector to the next depth. Layouts differ."""
    if kind == "shared":
        shells = x[3:].reshape(L_new - 1, 3)
        return np.concatenate([[x[0], x[1], x[2]], resample(shells, L_new).ravel()])
    if kind == "layer_gamma":
        L = L_new - 1
        gammas, thk, thm = x[:L], x[L], x[L + 1]
        shells = x[L + 2:].reshape(L, 3)
        gnew = np.concatenate([gammas, [gammas[-1]]])
        return np.concatenate([gnew, [thk, thm], resample(shells, L_new).ravel()])
    if kind == "layer_theta":
        L = L_new - 1
        gamma = x[0]
        thk, thm = x[1:1 + L], x[1 + L:1 + 2 * L]
        shells = x[1 + 2 * L:].reshape(L, 3)
        return np.concatenate([[gamma], thk, [thk[-1]], thm, [thm[-1]], resample(shells, L_new).ravel()])
    if kind in ("drop_K", "drop_M"):
        shells = x[2:].reshape(L_new - 1, 3)
        return np.concatenate([[x[0], x[1]], resample(shells, L_new).ravel()])
    raise KeyError(kind)


def pilot(kind, grad_fun, target, spec, L, seed_family, n_chains=4, maxfun=80, ceiling=5000):
    counter = [0]
    best = None
    rows = []
    for k in range(n_chains):
        rng = np.random.default_rng(np.random.SeedSequence([seed_family, k]))
        x = layout_L1(kind, 1, rng)
        final = None
        for ell in range(1, L + 1):
            if ell > 1:
                x = grow(kind, x, ell)
            if counter[0] >= ceiling:
                final = {"censored": True, "L": ell}
                break
            def fun(z, grad_fun=grad_fun):
                E, g, _, _ = grad_fun(target, z)
                return E, g
            xn, trace = lbfgs(fun, x, maxfun, counter, ceiling)
            if xn is None:
                final = {"censored": True, "L": ell, "nfev": len(trace)}
                break
            E, g, psi, var = grad_fun(target, xn)
            counter[0] += 1
            final = {"censored": False, "L": ell, "E": float(E), "error": float(E - spec["E0"]),
                     "F": h.fidelity(spec["gs"], psi), "grad": float(np.linalg.norm(g)),
                     "x": xn.tolist(), "hash": h.state_hash(psi), "nfev": len(trace) + 1}
            x = xn
        rows.append({"k": k, "final": final})
        if final and not final.get("censored") and (best is None or final["E"] < best["E"]):
            best = final
        print(f"  {kind} L{L} k{k} evals {counter[0]} err {None if not final or final.get('censored') else round(final['error'], 4)}", flush=True)
    return {"kind": kind, "L": L, "P": None if best is None else len(best["x"]), "evals": counter[0], "best": best, "rows": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = h.Target(9, B)
    spec = h.ground_state(target)
    H = h.dense_h(target)
    assert abs(spec["E0"] - E0) < 1e-9
    rng = np.random.default_rng(1)
    for fun, n in ((grad_layer_gamma, 26), (grad_layer_theta, 31)):
        z = rng.normal(0, 0.2, size=n)
        _, g, _, _ = fun(target, z)
        e = np.zeros(n)
        e[0] = 1e-6
        Ep = fun(target, z + e)[0]
        Em = fun(target, z - e)[0]
        if abs(g[0] - (Ep - Em) / 2e-6) > 1e-4:
            raise SystemExit(f"grad check failed {fun} {g[0]} {(Ep-Em)/2e-6}")
    print("grad checks ok", flush=True)
    # states
    rescue = json.loads((ROOT / "pressure" / "rescue.json").read_text())
    x_rare = np.asarray(rescue["resumed"]["x"], dtype=np.float64)
    procs = [r for r in load_jsonl(ROOT / "results" / "N9_records.jsonl") if r.get("arm") == "procedure" and r.get("L") == 6 and not r.get("censored")]
    x_rel = np.asarray(procs[0]["x"], dtype=np.float64)
    qrows = [r for r in load_jsonl(SIZE / "N9_r1" / "results" / "records.jsonl") if r.get("p") == 11 and not r.get("censored") and str(r.get("arm", "")).startswith("qaoa")]
    qbest = min(qrows, key=lambda r: r["E"])
    psi_q = h.qaoa_state(target, np.asarray(qbest["x"][0::2]), np.asarray(qbest["x"][1::2]))
    print("diagnostics", flush=True)
    rare = diagnose(target, spec, H, "rare_0.233", x_rare)
    rel = diagnose(target, spec, H, "reliable_0.322", x_rel)
    Eq = energy_dense(H, psi_q)
    qinfo = pack_state(target, spec, psi_q, Eq, H)
    psi_r = h.logical_candidate(target, x_rare[0], x_rare[1], x_rare[2], x_rare[3:].reshape(6, 3))
    psi_c = h.logical_candidate(target, x_rel[0], x_rel[1], x_rel[2], x_rel[3:].reshape(6, 3))
    compare = {
        "F_rare_vs_qaoa": h.fidelity(psi_r, psi_q),
        "F_reliable_vs_qaoa": h.fidelity(psi_c, psi_q),
        "F_rare_vs_reliable": h.fidelity(psi_r, psi_c),
        "qaoa": qinfo,
        "qaoa_logged_E": qbest["E"],
        "qaoa_logged_F": qbest["F"],
        "params": {"shared_L6": 21, "qaoa_p10": 20, "qaoa_p11": 22, "qaoa_p12": 24,
                   "layer_gamma_L5": 22, "layer_gamma_L6": 26, "layer_theta_L4": 21, "layer_theta_L6": 31,
                   "drop_L6": 20},
    }
    # low-lying weights
    evals, evecs = np.linalg.eigh(H)
    def weights(psi):
        return [{"k": i, "gap": float(evals[i] - evals[0]), "weight": float(abs(np.vdot(evecs[:, i], psi))**2),
                 "parity": float(np.vdot(evecs[:, i], evecs[::-1, i]).real)} for i in range(6)]
    compare["weights_rare"] = weights(psi_r)
    compare["weights_reliable"] = weights(psi_c)
    compare["weights_qaoa"] = weights(psi_q)
    (OUT / "diagnostics.json").write_text(json.dumps({"rare": rare, "reliable": rel, "compare": compare}, indent=2) + "\n")
    print("rare hess", rare["hess_min"], rare["n_negative"], "grad", rare["grad_norm"], flush=True)
    print("reliable hess", rel["hess_min"], rel["n_negative"], "grad", rel["grad_norm"], flush=True)
    print("F rare-qaoa", compare["F_rare_vs_qaoa"], flush=True)

    families = [
        ("layer_gamma", grad_layer_gamma, target, 5, 22, 1),
        ("layer_gamma", grad_layer_gamma, target, 6, 26, 1),
        ("layer_theta", grad_layer_theta, target, 4, 21, 2),
        ("layer_theta", grad_layer_theta, target, 6, 31, 2),
        ("drop_K", lambda t, z: grad_drop(t, z, "K"), target, 6, 20, 3),
        ("drop_M", lambda t, z: grad_drop(t, z, "M"), target, 6, 20, 4),
        ("contiguous", grad_shared, contiguous_target(), 6, 21, 5),
    ]
    # contiguous uses shared layout
    pilots = []
    p11 = [qbest["E"], qbest["F"]]
    hit = None
    for kind, fun, targ, L, P, fam in families:
        use = "shared" if kind == "contiguous" else kind
        print("pilot", kind, "L", L, flush=True)
        # layout name
        result = pilot(use if kind != "contiguous" else "shared", fun, targ, spec, L, 7509 + fam)
        result["family"] = kind
        result["P"] = P
        result["beats_p11"] = bool(result["best"] and result["best"]["E"] < p11[0] - TAU and result["best"]["F"] > p11[1] + TAU)
        pilots.append({k: result[k] for k in ("family", "L", "P", "evals", "best", "beats_p11")})
        if result["beats_p11"] and hit is None:
            hit = (kind, use, fun, targ, L, fam)
    (OUT / "pilots.json").write_text(json.dumps(pilots, indent=2) + "\n")
    confirm = None
    if hit is not None and time.perf_counter() > 0:
        kind, use, fun, targ, L, fam = hit
        print("confirm", kind, flush=True)
        confirm = pilot(use, fun, targ, spec, L, 7609 + fam, n_chains=16, maxfun=120, ceiling=12000)
        confirm["family"] = kind
        (OUT / "confirm.json").write_text(json.dumps({k: confirm[k] for k in ("family", "L", "evals", "best", "rows")}, indent=2) + "\n")
    # report
    lines = ["# N=9 stationary gap", "",
             "Shared-gamma L=6 has 21 parameters. QAOA p=10 has 20, p=11 has 22, p=12 has 24. No p equals 21.",
             "The six-start equal-depth recipe is unchanged.", ""]
    lines.append(f"Rare state error {rare['error']:.6f}, gradient {rare['grad_norm']:.3e}, Hessian min {rare['hess_min']:.4f}, negative eigenvalues {rare['n_negative']}.")
    lines.append(f"Reliable state error {rel['error']:.6f}, gradient {rel['grad_norm']:.3e}, Hessian min {rel['hess_min']:.4f}, negative eigenvalues {rel['n_negative']}.")
    lines.append(f"Dense energy matches the logged propagation to {rare['dense_delta']:.3e} and {rel['dense_delta']:.3e}.")
    lines.append(f"Fidelity between the rare state and p=11: {compare['F_rare_vs_qaoa']:.6f}. Reliable vs p=11: {compare['F_reliable_vs_qaoa']:.6f}.")
    lines.append("")
    lines.append("| family | L | P | best error | best F | beats p=11 on both | evals |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in pilots:
        b = p["best"]
        if not b or b.get("censored"):
            lines.append(f"| {p['family']} | {p['L']} | {p['P']} | censored | — | no | {p['evals']} |")
        else:
            lines.append(f"| {p['family']} | {p['L']} | {p['P']} | {b['error']:.4f} | {b['F']:.4f} | {p['beats_p11']} | {p['evals']} |")
    lines.append("")
    lines.append("p=11 reference error 0.170333, fidelity 0.706696. A pilot that does not beat both is not a replacement for six-start.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
