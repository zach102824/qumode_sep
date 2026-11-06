"""Total-spin blocks for H = -(sum Z)^3 / N^2 - Gamma * sum X.

Angular-momentum convention (Pauli, not spin-1/2 units):
    sum_i Z_i |S, m> = 2 m |S, m>
    <S, m+1| sum_i X_i |S, m> = sqrt(S(S+1) - m(m+1))

The fully symmetric irrep S = N/2 is indexed by Hamming weight w = 0..N,
with m = N/2 - w and sum Z = N - 2w. QAOA from |+>^N stays in this irrep
because both generators commute with S^2. Whether the ground state is in
this irrep is checked by comparing every total-spin sector, not assumed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln


def m_values(spin_s: float) -> np.ndarray:
    nlev = int(round(2 * spin_s)) + 1
    return -float(spin_s) + np.arange(nlev, dtype=np.float64)


def sector_matrix(n: int, spin_s: float, gamma: float, cost_sign: float = -1.0) -> np.ndarray:
    """Real symmetric Hamiltonian inside one total-spin irrep.

    cost_sign = -1 is the ferromagnet -N m_z^3.
    cost_sign = +1 is the odd-p partner +N m_z^3 (global-X conjugate).
    """
    m = m_values(spin_s)
    dim = m.size
    h = np.zeros((dim, dim), dtype=np.float64)
    mz = 2.0 * m
    h[np.arange(dim), np.arange(dim)] = cost_sign * (mz ** 3) / float(n * n)
    # ferromagnet cost_sign=-1 gives -(2m)^3 / N^2. Good.
    for i in range(dim - 1):
        elem = spin_s * (spin_s + 1.0) - m[i] * (m[i] + 1.0)
        hop = math.sqrt(elem) if elem > 0.0 else 0.0
        h[i, i + 1] -= gamma * hop
        h[i + 1, i] -= gamma * hop
    return h


def multiplicity(n: int, spin_s: float) -> int:
    """Number of spin-S copies in (1/2)^{⊗N}."""
    # f(S) = C(N, N/2 - S) - C(N, N/2 - S - 1)
    k = int(round(n / 2.0 - spin_s))
    a = math.comb(n, k) if 0 <= k <= n else 0
    b = math.comb(n, k - 1) if 0 <= k - 1 <= n else 0
    return a - b


def all_spins(n: int) -> list[float]:
    if n % 2 == 0:
        return [float(s) for s in range(n // 2, -1, -1)]
    return [n / 2.0 - i for i in range(n // 2 + 1)]


class SymmetricModel:
    """S = N/2 block plus full-spectrum edges from every sector."""

    def __init__(self, n: int, gamma: float, cost_sign: float = -1.0):
        self.n = int(n)
        self.gamma = float(gamma)
        self.cost_sign = float(cost_sign)
        self.dim = self.n + 1
        self.w = np.arange(self.dim, dtype=np.int64)
        self.mz = (self.n - 2 * self.w).astype(np.float64)  # sum Z
        # H_cost eigenvalues. cost_sign=-1 -> -(sum Z)^3 / N^2.
        self.e_cost = self.cost_sign * (self.mz ** 3) / float(self.n * self.n)
        self.hop = np.sqrt((self.w[1:] ) * (self.n - self.w[1:] + 1.0))
        # hop[w] couples weight w and w+1? Let's define hop_up[w] for w=0..N-1
        # <w+1|sum X|w> = sqrt((w+1) * (N-w))
        self.hop = np.sqrt((np.arange(1, self.n + 1)) * (self.n - np.arange(0, self.n)))
        sx = np.zeros((self.dim, self.dim), dtype=np.float64)
        idx = np.arange(self.n)
        sx[idx, idx + 1] = self.hop
        sx[idx + 1, idx] = self.hop
        self.sx = sx
        h = np.diag(self.e_cost) - self.gamma * sx
        evals, evecs = np.linalg.eigh(h)
        self.evals = evals
        v0 = evecs[:, 0]
        if np.linalg.norm(v0.imag) > 1e-8:
            raise RuntimeError(f"non-real ground state at N={n}, Gamma={gamma}")
        gs = np.real(v0).astype(np.float64)
        if float(np.sum(gs)) < 0.0:
            gs = -gs
        self.gs = gs
        self.e0_sym = float(evals[0])
        self.e1_sym = float(evals[1])
        self.emax_sym = float(evals[-1])
        self.gap_sym = self.e1_sym - self.e0_sym

        e0_full = self.e0_sym
        emax_full = self.emax_sym
        e0_sector = self.n / 2.0
        emax_sector = self.n / 2.0
        # Second-lowest and bookkeeping over sectors. S=N/2 already included.
        lowest = [(self.e0_sym, self.n / 2.0), (self.e1_sym, self.n / 2.0)]
        for spin_s in all_spins(self.n):
            if abs(spin_s - self.n / 2.0) < 1e-12:
                continue
            hs = sector_matrix(self.n, spin_s, self.gamma, self.cost_sign)
            ev = np.linalg.eigvalsh(hs)
            if ev[0] < e0_full:
                e0_full = float(ev[0])
                e0_sector = float(spin_s)
            if ev[-1] > emax_full:
                emax_full = float(ev[-1])
                emax_sector = float(spin_s)
            lowest.append((float(ev[0]), float(spin_s)))
            if ev.size > 1:
                lowest.append((float(ev[1]), float(spin_s)))
        lowest.sort(key=lambda t: t[0])
        self.e0 = float(e0_full)
        self.emax = float(emax_full)
        self.e0_sector = e0_sector
        self.emax_sector = emax_sector
        self.gap_full = float(lowest[1][0] - lowest[0][0])
        self.gs_in_max_spin = abs(e0_sector - self.n / 2.0) < 1e-9 and abs(self.e0 - self.e0_sym) < 1e-8

        # Mixer eigenbasis for fast QAOA. sum X, not -sum X.
        lam, v = np.linalg.eigh(sx)
        self.mixer_lam = lam
        self.mixer_v = v
        self.plus = plus_state(self.n)

    def apply_mixer(self, psi: np.ndarray, beta: float) -> np.ndarray:
        coeff = self.mixer_v.T @ psi
        coeff = coeff * np.exp(-1j * float(beta) * self.mixer_lam)
        return self.mixer_v @ coeff

    def apply_sx(self, psi: np.ndarray) -> np.ndarray:
        return self.sx @ psi

    def evolve(self, gammas: np.ndarray, betas: np.ndarray, psi0: np.ndarray | None = None) -> np.ndarray:
        """Cost then mixer, first layer closest to |+>."""
        psi = self.plus.copy() if psi0 is None else np.array(psi0, dtype=np.complex128, copy=True)
        ec = self.e_cost
        for gamma, beta in zip(np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)):
            psi = psi * np.exp(-1j * float(gamma) * ec)
            psi = self.apply_mixer(psi, float(beta))
        return psi

    def energy_of(self, psi: np.ndarray) -> float:
        hc = float(np.real(np.vdot(psi, self.e_cost * psi)))
        hx = float(np.real(np.vdot(psi, self.apply_sx(psi))))
        return hc - self.gamma * hx

    def fidelity(self, psi: np.ndarray) -> float:
        return float(abs(np.vdot(self.gs, psi)) ** 2)

    def eta(self, energy: float) -> float:
        """Pagano eta using full-H edges: (E - Emax) / (E0 - Emax)."""
        denom = self.e0 - self.emax
        if abs(denom) < 1e-15:
            return float("nan")
        return (energy - self.emax) / denom


def plus_state(n: int) -> np.ndarray:
    w = np.arange(n + 1, dtype=np.float64)
    logc = gammaln(n + 1) - gammaln(w + 1) - gammaln(n - w + 1)
    amp = np.exp(0.5 * logc - 0.5 * n * np.log(2.0))
    return amp.astype(np.complex128)


def gamma_period(n: int) -> tuple[int, float]:
    """Minimal T>0 such that gamma -> gamma+T changes H_cost phases by a global phase.

    Exact condition: T * gcd(m_i^3 - m_j^3) / N^2 = 2 pi.
    Returns (gcd, T).
    """
    ms = [n - 2 * w for w in range(n + 1)]
    cubes = [m ** 3 for m in ms]
    g = 0
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            g = math.gcd(g, abs(cubes[i] - cubes[j]))
    # T = 2 pi * N^2 / g
    period = 2.0 * math.pi * float(n * n) / float(g)
    return g, period


def verify_gamma_period(n: int, trials: int = 3, rng: np.random.Generator | None = None) -> dict:
    """Numerically confirm the gcd period and reject the half-period."""
    g, period = gamma_period(n)
    rng = np.random.default_rng(0) if rng is None else rng
    e_cost = np.array([-(n - 2 * w) ** 3 / float(n * n) for w in range(n + 1)])
    max_dev = 0.0
    half_dev = 0.0
    for _ in range(trials):
        psi = rng.normal(size=n + 1) + 1j * rng.normal(size=n + 1)
        psi /= np.linalg.norm(psi)
        psi_t = psi * np.exp(-1j * period * e_cost)
        ov = abs(np.vdot(psi, psi_t))
        max_dev = max(max_dev, 1.0 - ov)
        psi_h = psi * np.exp(-1j * (0.5 * period) * e_cost)
        ovh = abs(np.vdot(psi, psi_h))
        half_dev = max(half_dev, 1.0 - ovh)
    return {
        "gcd": g,
        "period": period,
        "max_infidelity_full_period": max_dev,
        "half_period_infidelity": half_dev,
        "half_period_is_not_a_period": half_dev > 1e-6,
    }


def beta_period_is_pi(n: int) -> dict:
    """exp(-i (beta+pi) sum X) = (-1)^N exp(-i beta sum X). pi/2 is not a period."""
    sx = np.zeros((n + 1, n + 1), dtype=np.float64)
    hop = np.sqrt((np.arange(1, n + 1)) * (n - np.arange(0, n)))
    idx = np.arange(n)
    sx[idx, idx + 1] = hop
    sx[idx + 1, idx] = hop
    lam, v = np.linalg.eigh(sx)
    rng = np.random.default_rng(1)
    psi = rng.normal(size=n + 1) + 1j * rng.normal(size=n + 1)
    psi /= np.linalg.norm(psi)

    def apply(beta: float) -> np.ndarray:
        c = v.T @ psi
        c = c * np.exp(-1j * beta * lam)
        return v @ c

    full = apply(math.pi)
    # global phase (-1)^N
    expected = ((-1.0) ** n) * psi
    dev_pi = float(np.linalg.norm(full - expected))
    half = apply(math.pi / 2.0)
    # distance to the closest global phase of psi
    ov = np.vdot(psi, half)
    dev_half = float(np.linalg.norm(half - ov / abs(ov) * psi)) if abs(ov) > 0 else 1.0
    return {"pi_deviation": dev_pi, "half_pi_distance": dev_half, "pi_ok": dev_pi < 1e-8, "half_not_period": dev_half > 1e-6}


def fold_min_abs(values: np.ndarray, period: float) -> np.ndarray:
    """Map each entry into [-period/2, period/2), the minimal-absolute representative."""
    half = 0.5 * period
    return np.mod(values + half, period) - half


def canonicalize(gammas: np.ndarray, betas: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fold only verified periods, into the minimal-absolute cell.

    beta has period pi. gamma has the gcd period of H_cost (see gamma_period).
    Conjugation (gamma, beta) -> (-gamma, -beta) is a symmetry of the real Hamiltonian
    but it is not a period, so it is not used to rewrite the schedule.
    """
    g = np.array(gammas, dtype=np.float64, copy=True)
    b = np.array(betas, dtype=np.float64, copy=True)
    notes: list[str] = []
    g2 = fold_min_abs(g, period)
    b2 = fold_min_abs(b, math.pi)
    if np.any(np.abs(g2 - g) > 1e-8):
        notes.append("gamma_period_fold")
    if np.any(np.abs(b2 - b) > 1e-8):
        notes.append("beta_period_pi_fold")
    return g2, b2, notes


def half_cut_entropy_bits(c: np.ndarray, n: int) -> float:
    """Von Neumann entropy, in bits, of qubits [0, N//2).

    The state is in the symmetric irrep, so every cut of this size agrees.
    c[w] are Dicke amplitudes of Hamming weight w.
    """
    na = n // 2
    nb = n - na
    # alpha[k, m] = c[k+m] * sqrt( C(na,k) C(nb,m) / C(n, k+m) )
    logc_n = gammaln(n + 1) - gammaln(np.arange(n + 1) + 1) - gammaln(n - np.arange(n + 1) + 1)
    amps = np.zeros((na + 1, nb + 1), dtype=np.complex128)
    for k in range(na + 1):
        log_a = math.lgamma(na + 1) - math.lgamma(k + 1) - math.lgamma(na - k + 1)
        for m in range(nb + 1):
            w = k + m
            log_b = math.lgamma(nb + 1) - math.lgamma(m + 1) - math.lgamma(nb - m + 1)
            log_amp = 0.5 * (log_a + log_b - float(logc_n[w]))
            amps[k, m] = c[w] * math.exp(log_amp)
    rho = amps @ amps.conj().T
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-15]
    return float(-np.sum(ev * np.log2(ev)))


def coherent_overlap(c: np.ndarray, n: int, n_theta: int = 721) -> float:
    """Max |<coherent|psi>|^2 over identical-qubit spin-coherent states, phi=0 or pi.

    c[w] real or complex Dicke amplitudes. This is the symmetric product overlap.
    """
    w = np.arange(n + 1, dtype=np.float64)
    logc = gammaln(n + 1) - gammaln(w + 1) - gammaln(n - w + 1)
    thetas = np.linspace(0.0, math.pi, n_theta)
    best = 0.0
    cc = np.asarray(c, dtype=np.complex128)
    for sign in (1.0, -1.0):
        for th in thetas:
            # |0> amp cos(th/2), |1> amp sign*sin(th/2), sign covers a relative Z-phase of pi
            a = math.cos(th / 2.0)
            b = sign * math.sin(th / 2.0)
            log_a = math.log(abs(a)) if abs(a) > 0.0 else None
            log_b = math.log(abs(b)) if abs(b) > 0.0 else None
            amp = np.zeros(n + 1, dtype=np.complex128)
            if abs(a) > 0.0 and abs(b) > 0.0:
                phase_b = 1.0 if b > 0 else -1.0
                log_amp = 0.5 * logc + (n - w) * log_a + w * log_b
                # subtract max for stability
                m = np.max(log_amp)
                amp = np.exp(log_amp - m) * (phase_b ** w)
                # The coherent state normalization is already included (binomial theorem).
                # Restoring the max: the actual amplitude is exp(log_amp), and the overlap
                # sum_w cc[w] * amp_raw. Using exp(log_amp - m) then multiply exp(m).
                ov = np.vdot(cc, amp) * math.exp(m)
            elif abs(b) == 0.0:
                ov = cc[0] * (a ** n)
            else:
                ov = cc[n] * (b ** n)
            best = max(best, float(abs(ov) ** 2))
    return best


def best_bitstring_probability(c: np.ndarray, n: int) -> float:
    """Largest computational-basis probability. Dicke weight w is spread over C(N,w) strings."""
    w = np.arange(n + 1, dtype=np.float64)
    logc = gammaln(n + 1) - gammaln(w + 1) - gammaln(n - w + 1)
    # |amp_string|^2 = |c_w|^2 / C(N,w)
    logp = np.log(np.maximum(np.abs(c) ** 2, 1e-300)) - logc
    return float(np.exp(np.max(logp)))


def magnetization(model: SymmetricModel, psi: np.ndarray) -> tuple[float, float]:
    prob = np.abs(psi) ** 2
    mz = float(np.real(np.sum(prob * model.mz)) / model.n)
    mx = float(np.real(np.vdot(psi, model.apply_sx(psi))) / model.n)
    return mz, mx


def observe(model: SymmetricModel, psi: np.ndarray) -> dict:
    energy = model.energy_of(psi)
    mz, mx = magnetization(model, psi)
    f_plus = float(abs(np.vdot(model.plus, psi)) ** 2)
    return {
        "energy": energy,
        "eta": model.eta(energy),
        "F": model.fidelity(psi),
        "mz": mz,
        "mx": mx,
        "entropy_bits": half_cut_entropy_bits(psi, model.n),
        "F_plus": f_plus,
        "F_coherent": coherent_overlap(psi, model.n),
        "F_bitstring": best_bitstring_probability(psi, model.n),
    }


def mean_field_gamma_c() -> float:
    """Thermodynamic mean-field first-order point of e = -cos^3(theta) - Gamma sin(theta).

    Stationary ferro branch Gamma = 3 cos(theta) sin(theta), equal in energy to the
    paramagnet e=-Gamma when sin(theta)=1/2, cos(theta)=sqrt(3)/2, so Gamma=3*sqrt(3)/4.
    The spinodal (local ferro minimum disappears) is Gamma=3/2.
    This is infinite-N mean field, not a finite-N eigenvalue.
    """
    return 3.0 * math.sqrt(3.0) / 4.0
