"""Ground-state observables on the full Hilbert space.

The target is the lowest eigenstate of the full Hamiltonian. Translation
invariance plus Perron-Frobenius puts that state in momentum 0, which is
checked numerically. Reflection is not a symmetry for these signs and is
not used to choose the state.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

from hamiltonian import apply_sum_x, cost_diagonal, translate_bits


def ground_state(n: int, kappa: float, gamma: float, diag: np.ndarray | None = None, k: int = 2):
    if diag is None:
        diag = cost_diagonal(n, kappa)
    dim = 1 << n

    def matvec(v):
        return diag * v - gamma * apply_sum_x(v, n)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    evals, evecs = eigsh(op, k=k, which="SA", tol=1e-10)
    order = np.argsort(evals)
    evals = evals[order]
    gs = evecs[:, order[0]].real
    if float(np.sum(gs)) < 0.0:
        gs = -gs
    return evals, gs, diag


def half_cut_entropy_bits(psi: np.ndarray, n: int) -> tuple[float, str]:
    """Von Neumann entropy, in bits, of qubits [0, n//2).

    Qubit 0 is the fastest bit. Every cut of this size is not equivalent:
    the ring is not fully symmetric under arbitrary bipartitions. This cut
    is the contiguous arc of length n//2 starting at site 0.
    """
    na = n // 2
    nb = n - na
    amp = np.asarray(psi, dtype=np.complex128).reshape(1 << nb, 1 << na)
    rho = amp.conj().T @ amp
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-15]
    s = float(-np.sum(ev * np.log2(ev)))
    return s, f"contiguous qubits [0, {na})"


def product_amplitude(theta: np.ndarray) -> np.ndarray:
    amps = np.array([1.0], dtype=np.float64)
    for th in theta:
        c = math.cos(float(th) / 2.0)
        s = math.sin(float(th) / 2.0)
        amps = np.kron(np.array([c, s]), amps)
    return amps


def best_product_overlap(gs: np.ndarray, n: int, rng: np.random.Generator, n_starts: int = 12) -> dict:
    """Max |<product|GS>|^2. GS is nonnegative, so real nonnegative product states suffice."""
    starts = [rng.uniform(0.0, math.pi, size=n) for _ in range(n_starts)]
    starts.append(np.full(n, math.pi / 2.0))
    starts.append(np.zeros(n))
    # Seed with the most probable bitstrings.
    probs = gs ** 2
    for x in np.argsort(probs)[-4:]:
        bit = np.array([0.0 if ((int(x) >> i) & 1) == 0 else math.pi for i in range(n)])
        starts.append(bit)
    best_f = -1.0
    best_theta = np.zeros(n)
    for th0 in starts:
        def fun(th: np.ndarray) -> float:
            amp = product_amplitude(th)
            return -float(np.dot(gs, amp) ** 2)

        res = minimize(fun, th0, method="L-BFGS-B", bounds=[(0.0, math.pi)] * n, options={"maxiter": 80})
        val = -float(res.fun)
        if val > best_f:
            best_f = val
            best_theta = res.x.copy()
    return {"F_product": best_f, "theta": best_theta}


def orbit_overlap(gs: np.ndarray, theta: np.ndarray, n: int) -> float:
    """|<GS|prod>|^2 / <prod|P_{k=0}|prod>, the overlap with the normalized translation orbit."""
    amp = product_amplitude(theta)
    num = float(np.dot(gs, amp) ** 2)
    acc = 0.0
    for shift in range(n):
        # local overlap between theta_i and theta_{i-shift}
        ov = 1.0
        for i in range(n):
            d = 0.5 * (float(theta[i]) - float(theta[(i - shift) % n]))
            ov *= math.cos(d)
        acc += ov
    den = acc / n
    if den <= 1e-14:
        return float("nan")
    return num / den


def classical_manifold_overlap(gs: np.ndarray, diag: np.ndarray) -> dict:
    e_min = float(np.min(diag))
    mask = np.abs(diag - e_min) < 1e-8
    deg = int(np.sum(mask))
    mass = float(np.sum(gs[mask]))
    f_uniform = (mass ** 2) / deg
    return {
        "classical_degeneracy": deg,
        "classical_emin": e_min,
        "F_classical_uniform": f_uniform,
        "p_classical_subspace": float(np.sum(gs[mask] ** 2)),
        "max_bitstring_prob": float(np.max(gs ** 2)),
    }


def plus_overlap(gs: np.ndarray, n: int) -> float:
    return float(np.sum(gs) / math.sqrt(1 << n)) ** 2


def momentum_deviation(gs: np.ndarray, n: int) -> float:
    psi = gs.astype(np.complex128)
    out = np.zeros_like(psi)
    for x in range(psi.shape[0]):
        out[translate_bits(x, n, 1)] += psi[x]
    return float(np.linalg.norm(out - psi))


def characterize(n: int, kappa: float, gamma: float, rng: np.random.Generator, diag: np.ndarray | None = None) -> dict:
    evals, gs, diag = ground_state(n, kappa, gamma, diag=diag)
    ent, cut = half_cut_entropy_bits(gs, n)
    prod = best_product_overlap(gs, n, rng)
    f_orbit = orbit_overlap(gs, prod["theta"], n)
    clas = classical_manifold_overlap(gs, diag)
    row = {
        "N": n,
        "kappa": kappa,
        "Gamma": gamma,
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "gap": float(evals[1] - evals[0]),
        "F_plus": plus_overlap(gs, n),
        "entropy_bits": ent,
        "entropy_cut": cut,
        "F_product": prod["F_product"],
        "F_orbit_product": f_orbit,
        "momentum_dev": momentum_deviation(gs, n),
        "min_amplitude": float(np.min(gs)),
        **clas,
    }
    return row, gs, diag
