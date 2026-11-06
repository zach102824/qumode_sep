"""Full Hamiltonian for a signed SQS and exact small-N spectra.

H = H_C - Gamma_X * sum_i X_i
H_C = - sum_b J_b Z_{b1}...Z_{b4}
Gamma_X = r * Gamma_per_degree, r = (N-1)(N-2)/6.

Gamma_per_degree is the brief's Gamma. Gamma_X is the coefficient in front
of each Pauli X. Round-1 Gamma was a Gamma_X, not a per-degree field.
Both are stored on every row.

Qubit 0 is the MSB of the statevector. P = prod X is the global flip.
H_C is even under the global flip for every sign pattern (each block has
four Z's), and sum X commutes with P, so P is always a symmetry. Whether
the ground state lies in P=+1 is computed, not assumed.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, LinearOperator

from signs import apply_signs_to_energy_table, eps_bits


def gamma_period_from_diag(diag: np.ndarray) -> dict:
    vals = np.unique(np.rint(diag).astype(np.int64))
    g = 0
    base = int(vals.min())
    for v in vals:
        g = math.gcd(g, int(v) - base)
    if g <= 0:
        raise RuntimeError("empty spectrum")
    return {"gcd": int(g), "T_gamma": float(2.0 * math.pi / g), "beta_period": float(math.pi)}


def flip_expectation(psi: np.ndarray) -> float:
    dim = psi.shape[0]
    return float(np.vdot(psi, psi[np.arange(dim) ^ (dim - 1)]).real)


def parity_populations(psi: np.ndarray) -> tuple[float, float]:
    """Weight on P=+1 and P=-1. P swaps x with its global flip."""
    dim = psi.shape[0]
    flipped = psi[np.arange(dim) ^ (dim - 1)]
    even = 0.5 * (psi + flipped)
    odd = 0.5 * (psi - flipped)
    return float(np.vdot(even, even).real), float(np.vdot(odd, odd).real)


def apply_sum_x(psi: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(psi)
    dim = psi.shape[0]
    for q in range(n):
        bit = 1 << (n - 1 - q)
        out += psi[np.arange(dim) ^ bit]
    return out


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i β sum X). Copies the input."""
    c = math.cos(beta)
    s = math.sin(beta)
    a = np.array(psi, dtype=np.complex128, copy=True).reshape([2] * n)
    for q in range(n):
        a = np.moveaxis(a, q, 0)
        p0 = np.array(a[0], copy=True)
        p1 = np.array(a[1], copy=True)
        a[0] = c * p0 - 1j * s * p1
        a[1] = -1j * s * p0 + c * p1
        a = np.moveaxis(a, 0, q)
    return np.ascontiguousarray(a).reshape(-1)


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 1.0 / math.sqrt(1 << n), dtype=np.complex128)


def evolve(psi0, diag, n, betas, gammas) -> np.ndarray:
    psi = np.array(psi0, dtype=np.complex128, copy=True)
    for beta, gamma in zip(betas, gammas):
        psi *= np.exp(-1j * float(gamma) * diag)
        psi = apply_mixer(psi, float(beta), n)
    return psi


def energy_of(psi, diag, n, gamma_x: float) -> float:
    hc = float(np.vdot(psi, diag * psi).real)
    sx = float(np.vdot(psi, apply_sum_x(psi, n)).real)
    return hc - float(gamma_x) * sx


def sparse_H(diag: np.ndarray, n: int, gamma_x: float):
    dim = diag.shape[0]
    H = diags(diag.astype(np.float64), format="csr")
    idx = np.arange(dim, dtype=np.int64)
    g = float(gamma_x)
    for q in range(n):
        bit = 1 << (n - 1 - q)
        a = idx[(idx & bit) == 0]
        b = a + bit
        data = np.full(a.shape[0], -g)
        H = H + coo_matrix((data, (a, b)), shape=(dim, dim)).tocsr()
        H = H + coo_matrix((data, (b, a)), shape=(dim, dim)).tocsr()
    return H


def dense_spectrum(diag, n, gamma_x: float) -> dict:
    H = np.diag(diag.astype(np.float64))
    idx = np.arange(1 << n)
    g = float(gamma_x)
    for q in range(n):
        bit = 1 << (n - 1 - q)
        a = idx[(idx & bit) == 0]
        b = a + bit
        H[a, b] -= g
        H[b, a] -= g
    evals, evecs = np.linalg.eigh(H)
    return {"evals": evals.real, "evecs": evecs}


def ground_pair(diag, n, gamma_x: float, k: int = 6) -> dict:
    """Global ground state. Dense for N≤12, Lanczos (eigsh) otherwise.

    Also reports parity of the returned vector and the residual.
    """
    dim = 1 << n
    if n <= 12:
        spec = dense_spectrum(diag, n, gamma_x)
        evals = spec["evals"]
        vec = spec["evecs"][:, 0]
        # lowest in each parity from the full spectrum
        e_plus = math.inf
        e_minus = math.inf
        for i, E in enumerate(evals):
            p = flip_expectation(spec["evecs"][:, i])
            if p > 0.5 and E < e_plus:
                e_plus = float(E)
            elif p < -0.5 and E < e_minus:
                e_minus = float(E)
        emax = float(evals[-1])
        e1 = float(evals[1])
        mult = int(np.sum(evals <= evals[0] + 1e-8))
        method = "dense_eigh"
    else:
        H = sparse_H(diag, n, gamma_x)
        ev, evec = eigsh(H, k=min(k, dim - 2), which="SA", tol=1e-8, maxiter=2000)
        order = np.argsort(ev.real)
        ev = ev.real[order]
        evec = evec[:, order]
        vec = evec[:, 0]
        evals = ev
        emax_v = eigsh(H, k=1, which="LA", tol=1e-8, maxiter=2000, return_eigenvectors=False)
        emax = float(emax_v[0])
        e1 = float(ev[1])
        mult = int(np.sum(ev <= ev[0] + 1e-7))
        e_plus = float("nan")
        e_minus = float("nan")
        method = "eigsh"
        spec = None
    # phase: nonnegative sum if possible (stoquastic)
    s = np.sum(vec)
    if abs(s) > 1e-12:
        vec = vec * (abs(s) / s)
    vec = np.asarray(vec, dtype=np.complex128)
    # residual
    if n <= 12:
        Hv = diag * vec - gamma_x * apply_sum_x(vec, n)
    else:
        Hv = sparse_H(diag, n, gamma_x) @ vec
    E0 = float(evals[0])
    resid = float(np.linalg.norm(Hv - E0 * vec))
    p_even, p_odd = parity_populations(vec)
    return {
        "E0": E0,
        "E1": e1,
        "gap": float(e1 - E0),
        "Emax": emax,
        "multiplicity": mult,
        "parity_even_weight": p_even,
        "parity_odd_weight": p_odd,
        "flip_expectation": flip_expectation(vec),
        "E0_Pplus": e_plus if n <= 12 else float("nan"),
        "E0_Pminus": e_minus if n <= 12 else float("nan"),
        "residual": resid,
        "method": method,
        "gs": vec,
        "spec": spec,
    }


def half_cut_entropy(psi: np.ndarray, n: int) -> dict:
    nA = n // 2
    mat = np.asarray(psi, dtype=np.complex128).reshape(1 << nA, 1 << (n - nA))
    rho = mat @ mat.conj().T
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-16]
    ev = ev / ev.sum()
    nats = float(-np.sum(ev * np.log(ev)))
    return {
        "S_bits": nats / math.log(2.0),
        "S_nats": nats,
        "subsystem": f"qubits[0:{nA}) with qubit 0 = MSB = point 0",
    }


def polarized_cat(n: int) -> np.ndarray:
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[-1] = 1.0 / math.sqrt(2.0)
    return psi


def y_to_sv(y: int, n: int) -> int:
    sv = 0
    for i in range(n):
        if (y >> i) & 1:
            sv |= 1 << (n - 1 - i)
    return sv


def manifold_state(minimizers: np.ndarray, n: int) -> np.ndarray:
    psi = np.zeros(1 << n, dtype=np.complex128)
    for y in minimizers:
        psi[y_to_sv(int(y), n)] = 1.0
    psi /= np.linalg.norm(psi)
    return psi


def product_overlap_stoquastic(psi: np.ndarray, n: int) -> dict:
    """Best real nonnegative product overlap by coordinate descent.

    Valid as a variational upper bound on the overlap; for a stoquastic GS
    the amplitudes can be taken nonnegative, so this is the right family.
    """
    amps = np.clip(np.real(psi), 0.0, None)
    if amps.sum() <= 0:
        amps = np.abs(psi)
    # θ_i in [0, π], |i> amplitude sin or cos
    theta = np.full(n, 0.5 * math.pi)

    def overlap(th):
        v = np.array([1.0])
        for i in range(n):
            v = np.kron(v, np.array([math.cos(th[i] / 2), math.sin(th[i] / 2)]))
        return float(np.dot(v, amps))

    best = overlap(theta)
    for _ in range(4):
        for i in range(n):
            grid = np.linspace(0, math.pi, 25)
            scores = []
            for g in grid:
                theta[i] = g
                scores.append(overlap(theta))
            theta[i] = grid[int(np.argmax(scores))]
            best = max(best, scores[int(np.argmax(scores))])
        # local polish
        for i in range(n):
            step = 0.05
            improved = True
            while improved and step > 1e-3:
                improved = False
                for trial in (theta[i] - step, theta[i] + step):
                    if trial < 0 or trial > math.pi:
                        continue
                    old = theta[i]
                    theta[i] = trial
                    val = overlap(theta)
                    if val > best + 1e-12:
                        best = val
                        improved = True
                    else:
                        theta[i] = old
                if not improved:
                    step *= 0.5
    return {"F_product": float(best ** 2), "theta": theta.tolist()}


def f_overlap(psi, phi) -> float:
    return float(abs(np.vdot(phi, psi)) ** 2)


def observables(psi, n, minimizers, diag, gamma_x) -> dict:
    ent = half_cut_entropy(psi, n)
    prod = product_overlap_stoquastic(psi, n)
    man = manifold_state(minimizers, n)
    return {
        "F_plus": f_overlap(psi, plus_state(n)),
        "F_polarized_cat": f_overlap(psi, polarized_cat(n)),
        "F_manifold": f_overlap(psi, man),
        "F_best_basis": float(np.max(np.abs(psi)) ** 2),
        "F_product": prod["F_product"],
        "S_half_bits": ent["S_bits"],
        "subsystem": ent["subsystem"],
        "E": energy_of(psi, diag, n, gamma_x),
        "parity_even": parity_populations(psi)[0],
    }


def hc_diag_for_pattern(n: int, blocks, eps: np.ndarray) -> np.ndarray:
    return apply_signs_to_energy_table(n, blocks, eps)


def explicit_hc_from_pauli(n: int, blocks, signs_pm: list[int]) -> np.ndarray:
    """Independent enumerator: each block contributes -J * product of Z's.
    signs_pm is J_b in {+1,-1}, statevector index MSB = point 0.
    """
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=np.float64)
    for block, J in zip(blocks, signs_pm):
        acc = np.ones(dim, dtype=np.float64)
        for p in block:
            bit = (x >> np.int64(n - 1 - p)) & 1
            acc *= np.where(bit == 0, 1.0, -1.0)
        diag += -float(J) * acc
    return diag


def signs_pm_from_eps(eps: np.ndarray, b: int) -> list[int]:
    bits = eps_bits(eps, b)
    return [1 - 2 * bit for bit in bits]
