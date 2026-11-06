"""Cross-checks that do not share a diagonal formula with the QAOA objective.

Pauli strings are enumerated in pauli_enum.py. Total-spin matrices live in
symmetric.py. This module compares them and aborts the search on failure.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

import pauli_enum as pe
import symmetric as sym


def _log(lines: list[str], msg: str) -> None:
    lines.append(msg)
    print(msg, flush=True)


def _embed(c: np.ndarray, n: int) -> np.ndarray:
    dim = 1 << n
    out = np.zeros(dim, dtype=np.complex128)
    w = np.arange(n + 1)
    binom = np.array([math.comb(n, int(k)) for k in w], dtype=np.float64)
    for x in range(dim):
        k = x.bit_count()
        out[x] = c[k] / math.sqrt(binom[k])
    return out


def _apply_sum_y(psi: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(psi)
    dim = psi.shape[0]
    for i in range(n):
        step = 1 << i
        for base in range(0, dim, step << 1):
            a = slice(base, base + step)
            b = slice(base + step, base + 2 * step)
            out[b] += 1j * psi[a]
            out[a] += -1j * psi[b]
    return out


def _spin_s2_expectation(psi: np.ndarray, n: int) -> float:
    sx = pe.apply_sum_x(psi, n)
    sy = _apply_sum_y(psi, n)
    sz = np.empty_like(psi)
    dim = psi.shape[0]
    for x in range(dim):
        m = n - 2 * x.bit_count()
        sz[x] = m * psi[x]
    # S_a = pauli_sum / 2, so S^2 = (sx^2+sy^2+sz^2)/4
    sx2 = pe.apply_sum_x(sx, n)
    sy2 = _apply_sum_y(sy, n)
    sz2 = np.empty_like(psi)
    for x in range(dim):
        m = n - 2 * x.bit_count()
        sz2[x] = m * sz[x]
    total = sx2 + sy2 + sz2
    return float(np.real(np.vdot(psi, total)) / 4.0)


def run_validation(ns_identity: list[int] | None = None) -> list[str]:
    lines: list[str] = []
    ns_identity = ns_identity or [6, 8, 10, 12]
    _log(lines, "validation start")

    gc = sym.mean_field_gamma_c()
    expected = 3.0 * math.sqrt(3.0) / 4.0
    if abs(gc - expected) > 1e-15:
        raise AssertionError("mean-field Gamma_c mismatch")
    _log(lines, f"mean-field Gamma_c = 3*sqrt(3)/4 = {gc:.12f}; spinodal = 1.5")

    for n in ns_identity:
        fails = pe.sumz_cubes_identity_failures(n)
        if fails:
            raise AssertionError(f"identity failed on {fails} states at N={n}")
        enum = pe.enumerated_cost_diagonal(n)
        closed = pe.closed_form_cost_diagonal(n)
        err = float(np.max(np.abs(enum - closed)))
        if err > 1e-9:
            raise AssertionError(f"Pauli diagonal vs m^3 err {err} at N={n}")
        dist = pe.distinct_index_cost_diagonal(n)
        c1 = float(pe.pauli_coefficients(n)[1])
        # H_coll - H_dist = c1 * sum Z, sum Z = N - 2 w
        field = np.array([c1 * (n - 2 * x.bit_count()) for x in range(1 << n)])
        err_field = float(np.max(np.abs((enum - dist) - field)))
        if err_field > 1e-9:
            raise AssertionError(f"lower-body field mismatch {err_field} at N={n}")
        _log(
            lines,
            f"N={n}: exact cubic identity 0 failures; Pauli-vs-m^3 max abs {err:.2e}; "
            f"field c1={c1:.6e}; coll-distinct matches c1*sumZ to {err_field:.2e}",
        )

    # Spectrum of every total-spin sector against full dense diagonalization.
    for n, gamma in ((6, 0.8), (6, 1.25), (8, 1.0)):
        dense = pe.dense_hamiltonian(n, gamma)
        full = np.linalg.eigvalsh(dense)
        pred = []
        for spin_s in sym.all_spins(n):
            ev = np.linalg.eigvalsh(sym.sector_matrix(n, spin_s, gamma, cost_sign=-1.0))
            mult = sym.multiplicity(n, spin_s)
            if mult <= 0:
                raise AssertionError(f"bad multiplicity S={spin_s} N={n}")
            pred.extend([float(x) for x in ev] * mult)
        pred = np.sort(np.asarray(pred))
        if pred.size != full.size:
            raise AssertionError(f"dimension {pred.size} vs {full.size}")
        spec_err = float(np.max(np.abs(pred - full)))
        if spec_err > 1e-8:
            raise AssertionError(f"sector spectrum mismatch {spec_err} N={n} Gamma={gamma}")
        _log(lines, f"N={n} Gamma={gamma}: all-sector spectrum vs dense max abs {spec_err:.2e}")

    # Ground state of the full matrix is the embedded symmetric ground state.
    for n in (6, 8, 10):
        for gamma in (0.6, 1.0, 1.35):
            model = sym.SymmetricModel(n, gamma, cost_sign=-1.0)
            if not model.gs_in_max_spin:
                raise AssertionError(f"GS not in S=N/2 at N={n} Gamma={gamma}, sector {model.e0_sector}")
            dense = pe.dense_hamiltonian(n, gamma)
            evals, evecs = np.linalg.eigh(dense)
            if abs(float(evals[0]) - model.e0) > 1e-8:
                raise AssertionError("full GS energy != sector scan")
            emb = _embed(model.gs, n)
            overlap = abs(np.vdot(evecs[:, 0], emb)) ** 2
            # full GS sign
            if overlap < 0.5:
                overlap = abs(np.vdot(evecs[:, 0], emb)) ** 2
            if overlap < 1.0 - 1e-8:
                raise AssertionError(f"embedded GS overlap {overlap} N={n} Gamma={gamma}")
            s2 = _spin_s2_expectation(emb, n)
            s_max = n / 2.0
            if abs(s2 - s_max * (s_max + 1.0)) > 1e-6:
                raise AssertionError(f"S^2 {s2} vs {s_max * (s_max + 1)}")
            _log(
                lines,
                f"N={n} Gamma={gamma}: full GS overlap {overlap:.12f}, "
                f"E0 {model.e0:.8f}, gap_full {model.gap_full:.3e}, S^2 ok, sector {model.e0_sector}",
            )

    # N=12: diagonal already checked. Residual of embedded GS on the sparse action,
    # and a Lanczos extremum compared with the sector edge.
    n = 12
    gamma = 1.0
    model = sym.SymmetricModel(n, gamma)
    emb = _embed(model.gs, n)
    diag = pe.enumerated_cost_diagonal(n)
    hpsi = diag * emb - gamma * pe.apply_sum_x(emb, n)
    resid = float(np.linalg.norm(hpsi - model.e0 * emb))
    if resid > 1e-7:
        raise AssertionError(f"N=12 GS residual {resid}")
    if not model.gs_in_max_spin:
        raise AssertionError("N=12 GS left the max-spin sector")
    _log(lines, f"N=12 Gamma=1: embedded GS residual {resid:.2e}, E0 {model.e0:.8f}, Emax {model.emax:.8f}")

    # Dicke hop versus one matrix element of sum X.
    n = 6
    d0 = _embed(np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex), n)  # weight 0
    # |D_1> is the normalized single-excitation state. Build via embedding of e_1.
    e1 = np.zeros(n + 1, dtype=complex)
    e1[1] = 1.0
    d1 = _embed(e1, n)
    hop = float(np.real(np.vdot(d1, pe.apply_sum_x(d0, n))))
    expect = math.sqrt(1 * n)
    if abs(hop - expect) > 1e-9:
        raise AssertionError(f"Dicke hop {hop} vs {expect}")
    _log(lines, f"Dicke <w=1|sum X|w=0> = {hop:.6f} matches sqrt(N)")

    # Periods.
    for n in (6, 8, 12, 16, 32):
        info = sym.verify_gamma_period(n)
        beta = sym.beta_period_is_pi(n)
        if info["max_infidelity_full_period"] > 1e-8 or not info["half_period_is_not_a_period"]:
            raise AssertionError(f"gamma period failed {info}")
        if not beta["pi_ok"] or not beta["half_not_period"]:
            raise AssertionError(f"beta period failed {beta}")
        _log(
            lines,
            f"N={n}: gamma period {info['period']:.6f} (gcd {info['gcd']}), "
            f"half-period infidelity {info['half_period_infidelity']:.3e}; beta period pi, not pi/2",
        )

    # Entropy and coherent overlap on product states.
    n = 8
    plus = sym.plus_state(n)
    s_plus = sym.half_cut_entropy_bits(plus, n)
    if s_plus > 1e-8:
        raise AssertionError(f"|+> entropy {s_plus}")
    fcoh = sym.coherent_overlap(plus, n)
    if abs(fcoh - 1.0) > 1e-8:
        raise AssertionError(f"|+> coherent overlap {fcoh}")
    ghz = np.zeros(n + 1, dtype=complex)
    ghz[0] = 1.0 / math.sqrt(2.0)
    ghz[n] = 1.0 / math.sqrt(2.0)
    s_ghz = sym.half_cut_entropy_bits(ghz, n)
    if abs(s_ghz - 1.0) > 1e-8:
        raise AssertionError(f"GHZ entropy {s_ghz}")
    _log(lines, f"entropy cut qubits[0:{n//2}): |+> {s_plus:.2e} bits, GHZ {s_ghz:.6f} bits; |+> coherent F=1")

    # Block evolution vs full statevector, and vs expm on the block.
    rng = np.random.default_rng(7)
    for n in (6, 10):
        gamma = 0.9
        model = sym.SymmetricModel(n, gamma)
        g = rng.normal(size=3) * 0.4
        b = rng.normal(size=3) * 0.3
        psi_block = model.evolve(g, b)
        cost = pe.enumerated_cost_diagonal(n)
        psi0 = np.full(1 << n, 2.0 ** (-n / 2.0), dtype=np.complex128)
        psi_full = pe.evolve_statevector(psi0, n, g, b, cost)
        emb = _embed(psi_block, n)
        ov = abs(np.vdot(psi_full, emb))
        if abs(ov - 1.0) > 1e-8:
            raise AssertionError(f"statevector replay overlap {ov} at N={n}")
        psi_expm = model.plus.copy()
        for gv, bv in zip(g, b):
            psi_expm = psi_expm * np.exp(-1j * gv * model.e_cost)
            psi_expm = expm(-1j * bv * model.sx) @ psi_expm
        ov2 = abs(np.vdot(psi_expm, psi_block))
        if abs(ov2 - 1.0) > 1e-8:
            raise AssertionError(f"expm replay overlap {ov2}")
        _log(lines, f"N={n}: full statevector and expm replay overlap the block evolution")

    # Odd-p sign flip on a random schedule.
    n = 8
    gamma = 1.1
    mf = sym.SymmetricModel(n, gamma, cost_sign=-1.0)
    ma = sym.SymmetricModel(n, gamma, cost_sign=+1.0)
    if abs(mf.e0 - ma.e0) > 1e-8 or abs(mf.emax - ma.emax) > 1e-8:
        raise AssertionError("AF/F spectra differ")
    # reversed nonnegative GS
    rev = mf.gs[::-1].copy()
    if abs(abs(np.vdot(rev, ma.gs)) - 1.0) > 1e-8:
        raise AssertionError("AF GS is not the global-X reverse of the F GS")
    g = np.array([0.2, -0.15, 0.4])
    b = np.array([0.3, -0.2, 0.1])
    ff = mf.fidelity(mf.evolve(g, b))
    fa = ma.fidelity(ma.evolve(g, b))
    if abs(ff - fa) > 1e-10:
        raise AssertionError(f"sign-flip fidelity mismatch {ff} vs {fa}")
    _log(lines, f"N={8}: F(+J)={ff:.12f} F(-J)={fa:.12f}; spectra and reversed GS match")

    # |+> is stationary under the mixer, and gamma=beta=0 energy matches the formula.
    model = sym.SymmetricModel(8, 0.7)
    e0, f0, psi = model.energy_of(model.plus), model.fidelity(model.plus), model.plus
    e_mix, _, psi_m = model.energy_of(model.evolve([0.0], [0.4])), None, model.evolve([0.0], [0.4])
    # rewrite properly
    psi_m = model.evolve(np.array([0.0]), np.array([0.4]))
    if abs(model.energy_of(psi_m) - model.energy_of(model.plus)) > 1e-10:
        raise AssertionError("mixer should not change |+>")
    if np.linalg.norm(psi_m - model.plus * np.vdot(model.plus, psi_m)) > 1e-8:
        raise AssertionError("|+> picked up weight under the mixer")
    _log(lines, f"|+> is a sum-X eigenstate; L=0 energy {model.energy_of(model.plus):.8f}")

    # Interpolation nodes: a linear schedule resampled to one more point stays on the line.
    g = np.array([0.0, 1.0])
    b = np.array([1.0, 0.0])
    # This is depth 2 -> depth 3, linear. New nodes 0, 0.5, 1.
    from qaoa_search import interpolate_schedule

    gn, bn = interpolate_schedule(g, b)
    if gn.shape != (3,) or abs(gn[1] - 0.5) > 1e-12 or abs(bn[1] - 0.5) > 1e-12:
        raise AssertionError(f"linear bootstrap resample failed {gn} {bn}")
    g4 = np.array([0.0, 0.2, 0.6, 1.0])
    b4 = np.array([1.0, 0.7, 0.3, 0.0])
    g5, b5 = interpolate_schedule(g4, b4)
    if g5.shape != (5,) or abs(g5[0] - 0.0) > 1e-10 or abs(g5[-1] - 1.0) > 1e-10:
        raise AssertionError("cubic bootstrap endpoints failed")
    _log(lines, "Pagano interpolation: depth-3 linear midpoint and depth-5 cubic endpoints ok")

    # Global Y = tensor_i Y_i sends Z -> -Z and X -> -X, hence H -> -H.
    # The spectrum is symmetric and Emax = -E0. Checked on the full matrix.
    for n, gamma in ((6, 0.7), (8, 1.2)):
        ev = np.linalg.eigvalsh(pe.dense_hamiltonian(n, gamma))
        sym_err = float(np.max(np.abs(ev + ev[::-1])))
        if sym_err > 1e-8:
            raise AssertionError(f"spectrum not symmetric under H -> -H, err {sym_err}")
        _log(lines, f"N={n} Gamma={gamma}: full spectrum equals its negative, max abs {sym_err:.2e}")

    _log(lines, "validation passed")
    return lines


if __name__ == "__main__":
    run_validation()
