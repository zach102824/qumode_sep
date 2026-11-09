"""Exact checks for the round-1 A circuit, feature, and symmetries.

Failures here stop the sweep. Symmetry periods are returned only when the
overlap test on the reachable parity sector passes; interpolation may use
them, and the raw angles are still what the optimizer stores.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .circuit import (
    apply_physical_unitary,
    energy_and_grad,
    energy_of_state,
    energy_only,
    fwht,
    logical_on_state,
    phi_table,
    state_from_params,
)
from .common import log, param_count
from .hamiltonian import (
    apply_X_sum,
    build_J,
    dense_spectrum,
    physical_K,
    problem_at_fixed_B,
    reflection_perm,
    spectrum_at_B,
    zz_diag,
)


def _bitreverse3(j: int) -> int:
    return ((j & 1) << 2) | (j & 2) | ((j >> 2) & 1)


def reflect_fock(n: int, x: int) -> int:
    ncav = n // 3
    js = [((x >> (3 * a)) & 7) for a in range(ncav)]
    js = [_bitreverse3(j) for j in js[::-1]]
    y = 0
    for a, j in enumerate(js):
        y |= j << (3 * a)
    return y


def mixer_generator(n: int, c1: float, c2: float, c3: float) -> np.ndarray:
    dim = 1 << n
    gen = np.zeros((dim, dim), dtype=np.complex128)
    for a in range(n // 3):
        sites = (3 * a, 3 * a + 1, 3 * a + 2)
        pairs = ((sites[0], sites[1]), (sites[0], sites[2]), (sites[1], sites[2]))
        for s in range(dim):
            for i in sites:
                gen[s ^ (1 << i), s] += c1
            for i, j in pairs:
                gen[s ^ (1 << i) ^ (1 << j), s] += c2
            gen[s ^ (1 << sites[0]) ^ (1 << sites[1]) ^ (1 << sites[2]), s] += c3
    return gen


def apply_mixer_logical(problem, psi, c1, c2, c3):
    x1 = fwht(psi)
    phi = c1 * problem.w1 + c2 * problem.w2 + c3 * problem.w3
    return fwht(x1 * np.exp(-1j * phi))


def _random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex128)


def gradient_errors(problem, method: str, L: int, seed: int, eps: float = 1e-6) -> float:
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.25, size=param_count(method, L))
    _E, grad, _psi = energy_and_grad(problem, method, x, L)
    err = 0.0
    # All coordinates at N=6; a few at larger N.
    idxs = range(x.size) if problem.n <= 6 else list(rng.choice(x.size, size=min(4, x.size), replace=False))
    for i in idxs:
        step = np.zeros_like(x)
        step[i] = eps
        Ep, _ = energy_only(problem, method, x + step, L)
        Em, _ = energy_only(problem, method, x - step, L)
        fd = (Ep - Em) / (2 * eps)
        err = max(err, abs(fd - grad[i]))
    return float(err)


def verified_shell_periods(problem, trials: int = 4) -> list[dict]:
    """Probe a pi/2 period of each shell coordinate.

    Accepted only when every trial leaves the |+> evolution unchanged up to a
    global phase. If a generic state also stays put, the period is a full-space
    gauge. If a generic state moves, it is only a symmetry inside the parity
    sector reached from |+>, which is still enough to unwrap bootstrap seeds.
    """
    out = []
    candidate = 0.5 * np.pi
    rng = np.random.default_rng(11)
    names = ("c1", "c2", "c3")
    for coord, name in enumerate(names):
        plus_ok = True
        generic_fids = []
        for t in range(trials):
            c = rng.normal(scale=0.4, size=3)
            base = apply_mixer_logical(problem, problem.plus, *c)
            shifted = c.copy()
            shifted[coord] += candidate
            evolved = apply_mixer_logical(problem, problem.plus, *shifted)
            if abs(np.vdot(base, evolved)) ** 2 < 1.0 - 1e-8:
                plus_ok = False
            generic = _random_state(problem.n, 100 + t)
            g1 = apply_mixer_logical(problem, generic, *c)
            g2 = apply_mixer_logical(problem, generic, *shifted)
            generic_fids.append(float(abs(np.vdot(g1, g2)) ** 2))
        generic_min = float(min(generic_fids))
        if not plus_ok:
            kind = "rejected"
            period = None
        elif generic_min > 1.0 - 1e-8:
            kind = "full_space_gauge"
            period = candidate
        else:
            kind = "parity_sector_gauge"
            period = candidate
        out.append(
            {
                "name": name,
                "period": None if period is None else float(period),
                "kind": kind,
                "generic_fidelity_min": generic_min,
            }
        )
    return out


def domain_wall_report(problem) -> dict:
    """Classical regression of normalized F_K on Hc over Néel excitations.

    Built from J only. A large residual means K is not a rescaling of Hc on
    this set. This does not use the quantum ground state.
    """
    n = problem.n
    neel = np.array([i % 2 for i in range(n)], dtype=np.int64)
    configs = [neel, 1 - neel]
    for i in range(n):
        c = neel.copy()
        c[i] ^= 1
        configs.append(c)
    for i in range(n - 1):
        c = neel.copy()
        c[i] ^= 1
        c[i + 1] ^= 1
        configs.append(c)
    for a in range(n // 3):
        c = neel.copy()
        c[3 * a : 3 * a + 3] ^= 1
        configs.append(c)

    def classical(bits, coeff):
        z = 1.0 - 2.0 * bits.astype(np.float64)
        e = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                e += float(coeff[i, j]) * z[i] * z[j]
        return e

    K = problem.features["physical"].K_norm
    hc = np.array([classical(c, problem.J) for c in configs])
    fk = np.array([classical(c, K) for c in configs])
    design = np.column_stack([hc, np.ones_like(hc)])
    coef, *_ = np.linalg.lstsq(design, fk, rcond=None)
    pred = design @ coef
    resid = fk - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((fk - fk.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "n_configs": len(configs),
        "slope_a": float(coef[0]),
        "offset_b": float(coef[1]),
        "R2": r2,
        "residual_rms": float(np.sqrt(np.mean(resid**2))),
        "Hc_span": float(hc.max() - hc.min()),
        "note": (
            "R^2 near 1 on this excitation set would mean the two-body feature "
            "is almost a function of Hc there. The number is a diagnostic, not a gate."
        ),
    }


def run_validation(problem: object | None = None) -> dict:
    log("validation: logical/physical equivalence, gradients, symmetries")
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    if problem is None:
        problem = problem_at_fixed_B(6, -0.45)
    n = problem.n
    if n != 6:
        raise ValueError("equivalence validation is specified at N=6")

    # Hadamard transform matches the Kronecker product, LSB = site 0.
    H1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    W = np.array([[1.0]], dtype=np.complex128)
    for _ in range(n):
        W = np.kron(H1, W)
    psi = _random_state(n, 1)
    check("fwht_matches_kronecker", np.allclose(fwht(psi), W @ psi), f"err={np.max(np.abs(fwht(psi)-W@psi)):.2e}")
    check("W_squared_identity", np.allclose(fwht(fwht(psi)), psi))

    # Packing: chain reflection is cavity reversal plus intra-cavity bit reversal.
    for nn in (6, 12):
        perm = reflection_perm(nn)
        mismatch = 0
        for x in range(1 << nn):
            if reflect_fock(nn, x) != int(perm[x]):
                mismatch += 1
                break
        involution = bool(np.array_equal(perm[perm], np.arange(1 << nn)))
        check(f"packing_reflection_n{nn}", mismatch == 0 and involution, f"mismatch_probe={mismatch}")

    check(
        "shell_weights_reflection_invariant",
        np.allclose(problem.w1, problem.w1[problem.Rperm])
        and np.allclose(problem.w2, problem.w2[problem.Rperm])
        and np.allclose(problem.w3, problem.w3[problem.Rperm]),
    )

    phi = phi_table(0.3, -0.2, 0.15)
    manual = np.array([
        0.3 * 3 + -0.2 * 3 + 0.15,
        0.3 * 1 + -0.2 * -1 - 0.15,
        0.3 * -1 + -0.2 * -1 + 0.15,
        0.3 * -3 + -0.2 * 3 - 0.15,
    ])
    check("phi_formula", np.allclose(phi, manual), str(np.round(phi, 6)))

    c1, c2, c3 = 0.31, -0.17, 0.23
    gen = mixer_generator(n, c1, c2, c3)
    psi = _random_state(n, 2)
    exact = expm(-1j * gen) @ psi
    ours = apply_mixer_logical(problem, psi, c1, c2, c3)
    check("mixer_matches_pauli_expm", np.allclose(exact, ours, atol=1e-9), f"maxabs={np.max(np.abs(exact-ours)):.2e}")
    beta = -0.42
    gen_x = mixer_generator(n, beta, 0.0, 0.0)
    exact_x = expm(-1j * gen_x) @ psi
    ours_x = apply_mixer_logical(problem, psi, beta, 0.0, 0.0)
    check("beta_is_c1", np.allclose(exact_x, ours_x, atol=1e-9))

    # Full unitary: physical map equals logical map after the input Hadamard.
    rng = np.random.default_rng(3)
    cs = rng.normal(scale=0.35, size=(2, 3))
    gamma, theta = 0.37, -0.55
    fk = problem.features["physical"].fk_diag
    for label, vec in (("vacuum", None), ("random", _random_state(n, 4)), ("random2", _random_state(n, 5))):
        if vec is None:
            base = np.zeros(problem.dim, dtype=np.complex128)
            base[0] = 1.0
        else:
            base = vec
        phys = apply_physical_unitary(problem, base, gamma, theta, cs, fk)
        logi = logical_on_state(problem, fwht(base), gamma, theta, cs, fk)
        err = float(np.max(np.abs(phys - logi)))
        check(f"logical_physical_{label}", err < 1e-9, f"maxabs={err:.2e}")

    # Zero shell does not absorb a shared-gamma cost block.
    g = 0.33
    one = np.array([g, 0.0, 0.0, 0.0])
    two = np.array([g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    psi1 = state_from_params(problem, "native_D", one, 1)
    psi2 = state_from_params(problem, "native_D", two, 2)
    zero_dist = float(np.linalg.norm(psi1 - psi2))
    check("zero_shell_is_not_identity_pad", zero_dist > 1e-3, f"||psi_L1-psi_L2||={zero_dist:.3e}")

    Eplus = energy_of_state(problem, problem.plus)
    check("plus_energy_is_B_N", abs(Eplus - problem.B * n) < 1e-10, f"E+={Eplus:.6f} BN={problem.B*n:.6f}")
    check("variational_gs_below_plus", problem.E0 <= Eplus + 1e-8, f"E0={problem.E0:.6f}")
    check("gs_residual", problem.residual < 1e-8, f"{problem.residual:.2e}")
    check("gs_nonnegative", problem.gs_negative_mass < 1e-10, f"neg_mass={problem.gs_negative_mass:.2e}")
    check("gs_parity_plus", abs(problem.parity_expectation - 1) < 1e-8, f"P={problem.parity_expectation:.6f}")
    check("gs_reflection_plus", abs(problem.reflection_expectation - 1) < 1e-8, f"R={problem.reflection_expectation:.6f}")

    # Field term: |+> is its ground state for B<0, so the adiabatic sign is not flipped.
    # Eigenvalues of sum X are n-2k. Min of B*that is B*n when B<0.
    check("field_gs_is_plus", problem.B < 0 and abs(Eplus - problem.B * n) < 1e-10)

    mv_probe = _random_state(n, 6).real
    mv_probe /= np.linalg.norm(mv_probe)
    Hpsi = problem.hc_diag * mv_probe + problem.B * apply_X_sum(mv_probe, n)
    R = problem.Rperm
    P = problem.Pperm
    comm_R = np.linalg.norm(Hpsi[R] - (problem.hc_diag * mv_probe[R] + problem.B * apply_X_sum(mv_probe[R], n)))
    # [H,R] psi = H R psi - R H psi
    Rpsi = mv_probe[R]
    HRpsi = problem.hc_diag * Rpsi + problem.B * apply_X_sum(Rpsi, n)
    RHpsi = Hpsi[R]
    check("H_commutes_with_reflection", np.linalg.norm(HRpsi - RHpsi) < 1e-8, f"{np.linalg.norm(HRpsi-RHpsi):.2e}")
    Ppsi = mv_probe[P]
    HPpsi = problem.hc_diag * Ppsi + problem.B * apply_X_sum(Ppsi, n)
    check("H_commutes_with_parity", np.linalg.norm(HPpsi - Hpsi[P]) < 1e-8)

    ident = problem.meta["identity"]["max_abs_identity_error"]
    check("sum_ei2_equals_const_plus_2FK", ident < 1e-8, f"err={ident:.2e}")
    J = problem.J
    K = physical_K(J)
    J2 = J @ J
    check("K_is_offdiag_J2", np.allclose(K, J2 - np.diag(np.diag(J2))))
    # Z_i^2 contributes only the diagonal of J@J, a constant, already excluded.
    const_diag = float(np.sum(np.diag(J2)))
    manual_const = float(np.sum(J * J))
    check("Zi2_is_constant", abs(const_diag - manual_const) < 1e-10, f"const={const_diag:.6f}")

    check(
        "physical_K_reflection_symmetric",
        problem.features["physical"].reflection_max_abs < 1e-10,
        f"{problem.features['physical'].reflection_max_abs:.2e}",
    )
    check(
        "refl_K_symmetric",
        problem.features["refl"].reflection_max_abs < 1e-10,
    )
    check(
        "random_K_breaks_reflection",
        problem.features["random"].reflection_max_abs > 1e-3,
        f"{problem.features['random'].reflection_max_abs:.3f}",
    )
    cos = abs(problem.features["physical"].cosine_with_J)
    check("K_not_proportional_to_J", cos < 0.999, f"|cos|={cos:.6f}")
    check(
        "rms_normalization",
        abs(problem.features["physical"].sK * (np.sqrt(np.mean(np.square(K[np.triu_indices(n, 1)])))) - problem.rms_J) < 1e-10,
    )

    # Circuit symmetries of the prepared state.
    xA = np.array([0.2, -0.4, 0.1, -0.2, 0.05])
    psi_A = state_from_params(problem, "family_A", xA, 1)
    check("familyA_state_reflection", abs(abs(np.vdot(psi_A, psi_A[problem.Rperm])) - 1) < 1e-8)
    check("familyA_state_parity", abs(abs(np.vdot(psi_A, psi_A[problem.Pperm])) - 1) < 1e-8)
    xR = np.array([0.2, 0.7, 0.1, -0.2, 0.05])
    psi_R = state_from_params(problem, "random_K", xR, 1)
    refl_fid = abs(np.vdot(psi_R, psi_R[problem.Rperm])) ** 2
    check("random_K_state_not_forced_into_R", refl_fid < 0.999, f"refl_fid={refl_fid:.6f}")

    # theta=0 family A matches native D; L=1 QAOA matches shared-gamma QAOA.
    x_native = np.array([0.25, 0.4, -0.1, 0.2, 0.05, 0.0, -0.1])
    x_A = np.concatenate([[0.25, 0.0], x_native[1:]])
    e1, _ = energy_only(problem, "native_D", x_native, 2)
    e2, _ = energy_only(problem, "family_A", x_A, 2)
    check("theta0_matches_native", abs(e1 - e2) < 1e-10, f"dE={e1-e2:.2e}")
    xq = np.array([0.22, -0.31])
    eq, _ = energy_only(problem, "qaoa", xq, 1)
    es, _ = energy_only(problem, "shared_qaoa", xq, 1)
    check("L1_qaoa_matches_shared", abs(eq - es) < 1e-10)

    for method, seed in (("qaoa", 21), ("shared_qaoa", 22), ("native_D", 23), ("family_A", 24), ("random_K", 25)):
        err = gradient_errors(problem, method, 2 if method != "qaoa" else 2, seed)
        check(f"grad_{method}", err < 5e-5, f"maxabs={err:.2e}")

    # Dense versus sparse at this fixed B.
    hc = zz_diag(n, build_J(n))
    dim = 1 << n
    v0 = np.full(dim, 1.0 / np.sqrt(dim))
    vla = np.ones(dim)
    vla[::2] *= -1
    vla /= np.linalg.norm(vla)
    sparse = spectrum_at_B(hc, problem.B, n, v0, vla)
    dense = dense_spectrum(hc, problem.B, n)
    fid = float(np.dot(sparse["gs"], dense["gs"]) ** 2)
    check("sparse_matches_dense_gs", fid > 1 - 1e-8 and abs(sparse["E0"] - dense["E0"]) < 1e-8, f"F={fid:.12f}")

    period_info = verified_shell_periods(problem)
    periods = [row["period"] for row in period_info]
    check(
        "shell_period_probe_ran",
        True,
        " ".join(f"{row['name']}={row['kind']}" for row in period_info),
    )
    # Energy of a prepared state is real and variationally above E0 for a generic point.
    Ec, psi = energy_only(problem, "family_A", np.array([0.2, 0.3, 0.1, 0.0, 0.0]), 1)
    imag = abs(np.vdot(psi, problem.hc_diag * psi + problem.B * apply_X_sum(psi, n)).imag)
    check("energy_real", imag < 1e-8, f"Im={imag:.2e}")
    check("norm_preserved", abs(np.vdot(psi, psi).real - 1) < 1e-8)
    check("variational_sample", Ec + 1e-8 >= problem.E0, f"E-E0={Ec-problem.E0:.3e}")

    for method, L, expect in (
        ("qaoa", 4, 8),
        ("shared_qaoa", 4, 5),
        ("native_D", 4, 13),
        ("family_A", 4, 14),
    ):
        check(f"param_count_{method}_L{L}", param_count(method, L) == expect, str(param_count(method, L)))

    passed = all(c["ok"] for c in checks)
    log(f"validation {'PASSED' if passed else 'FAILED'} ({sum(c['ok'] for c in checks)}/{len(checks)})")
    return {
        "passed": passed,
        "checks": checks,
        "periods": [None if p is None else float(p) for p in periods],
        "period_info": period_info,
        "domain_wall_fixedB": domain_wall_report(problem),
        "fixed_B": problem.B,
    }


if __name__ == "__main__":
    result = run_validation()
    if not result["passed"]:
        raise SystemExit(1)
