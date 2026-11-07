"""Round-2 transfer helpers for family A on the locked Nov_3rd gate.

Angles are shared across n. power_p / Walsh shells use the train69 scale
``a_n = (8/(n-1))^p`` (equals 1 at n=9). The gate sequence is the PR #13
Fock twin and is not modified:

    |0⟩ → [U_c^X → Hamming-shell SNAP]^L → W
    U_c^X = W exp(-i γ H_c) W,   W = H^{⊗3}

A stack of cost-only layers (SNAP phase 0) is exactly one pulse
``exp(-i (Σ γ) H_c^X)`` because those layers commute. That is the GRAPE
optimum for a single control along H_c, and it is already inside U_c^X.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from Nov_3rd.ansatz import fwht_unnormalized, popcount_arange
from Nov_3rd.quantum_h_tf import (
    FockSim,
    SCALE1000_BASE,
    cavity_entropy_bits,
    exact_ground,
    fourier_schedule,
    instance_from_clauses,
    load_train_instances,
    paper_fourier_init,
    phi_powers,
)

LAM_ZZZ = 0.5
LAM_UNSAT = 0.75
BEAT_R = 0.98
TIE_R = 1.02


def a_of_n(n: int, p: torch.Tensor) -> torch.Tensor:
    """(8/(n−1))^p. Identically 1 at n=9, so p is learned from the other sizes."""
    return torch.exp(p * math.log(8.0 / float(int(n) - 1)))


def shell_phi(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, p: torch.Tensor, n: int) -> torch.Tensor:
    a = a_of_n(n, p)
    return phi_powers(c1, a * c2, a * c3)


def sparse_hamiltonian(diag: np.ndarray, n: int, lam: float):
    """CSR of diag(H_c) − λ Σ_i X_i. Each hypercube edge is stored once per triangle."""
    dim = int(diag.shape[0])
    if dim != (1 << int(n)):
        raise ValueError(f"diag {dim} != 2^{n}")
    idx = np.arange(dim, dtype=np.int64)
    rows = [idx]
    cols = [idx]
    vals = [np.asarray(diag, dtype=np.float64)]
    for i in range(int(n)):
        flipped = idx ^ (1 << (int(n) - 1 - i))
        keep = idx < flipped
        r = idx[keep]
        c = flipped[keep]
        v = np.full(r.shape[0], -float(lam), dtype=np.float64)
        rows.extend((r, c))
        cols.extend((c, r))
        vals.extend((v, v.copy()))
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    ).tocsr()


def sparse_ground_state(diag: np.ndarray, n: int, lam: float, tol: float = 1e-9) -> dict:
    """Lowest eigenpair via Lanczos (eigsh which='SA'). Residual is ||(H−E)ψ||."""
    H = sparse_hamiltonian(diag, n, lam)
    dim = H.shape[0]
    v0 = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.float64)
    ncv = min(dim - 1, 48)
    maxiter = 4000 if dim < 20000 else 8000

    def _solve(ncv_, maxiter_, tol_):
        return eigsh(H, k=1, which="SA", tol=tol_, maxiter=maxiter_, ncv=ncv_, v0=v0)

    try:
        evals, evecs = _solve(ncv, maxiter, tol)
    except ArpackNoConvergence:
        evals, evecs = _solve(min(dim - 1, max(ncv * 2, 96)), maxiter * 2, tol * 10)
    psi = np.real(np.asarray(evecs[:, 0], dtype=np.float64))
    nrm = float(np.linalg.norm(psi))
    psi = psi / nrm
    ev = float(evals[0])
    resid = float(np.linalg.norm(H @ psi - ev * psi))
    return {
        "E0": ev,
        "psi": psi,
        "residual": resid,
        "peak": float(np.max(psi ** 2)),
        "entropy": cavity_entropy_bits(psi, n),
    }


def dense_matches_sparse(diag: np.ndarray, n: int, lam: float) -> float:
    """|E0_sparse − E0_dense|. Used as the Lanczos check at n≤9."""
    got = sparse_ground_state(diag, n, lam)
    ref = exact_ground(np.asarray(diag, dtype=np.float64), n, lam)
    if got["residual"] > 1e-6:
        raise RuntimeError(f"Lanczos residual {got['residual']}")
    return abs(got["E0"] - float(ref["E0"]))


def walsh_spectrum(diag: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh–Hadamard of a bitstring diagonal. Last axis is the frequency."""
    return fwht_unnormalized(diag)


def walsh_pieces(diag: torch.Tensor) -> torch.Tensor:
    """Degree-1..4 Walsh pieces of ``diag`` (..., 2^n), stacked on a new leading axis.

    Equal weights on these four pieces recover the projection of the diagonal
    onto Walsh degrees 1..4, not the degree-0 mean and not degrees above 4.
    A pure 3-local ZZZ diagonal has only the degree-3 piece.
    """
    if diag.ndim < 1:
        raise ValueError("diag")
    n_bits = int(round(math.log2(diag.shape[-1])))
    if (1 << n_bits) != int(diag.shape[-1]):
        raise ValueError(f"diag length {diag.shape[-1]} is not 2^n")
    F = fwht_unnormalized(diag)
    pc = popcount_arange(n_bits, device=diag.device)
    N = float(diag.shape[-1])
    out = []
    for w in range(1, 5):
        Fw = torch.where(pc == w, F, torch.zeros((), dtype=F.dtype, device=F.device))
        out.append(fwht_unnormalized(Fw) / N)
    return torch.stack(out, dim=0)


def walsh_mass_fractions(diag: np.ndarray) -> dict:
    """Fraction of unnormalized Walsh energy in each degree, plus the 1..4 share."""
    t = torch.as_tensor(np.asarray(diag, dtype=np.float64))
    if t.ndim == 1:
        t = t.reshape(1, -1)
    F = fwht_unnormalized(t)
    n_bits = int(round(math.log2(t.shape[-1])))
    pc = popcount_arange(n_bits, device=t.device).numpy()
    energy = (F.detach().numpy() ** 2).sum(axis=0)
    total = float(energy.sum())
    if total <= 0:
        raise RuntimeError("Walsh energy is zero")
    by = {int(w): float(energy[pc == w].sum() / total) for w in range(0, n_bits + 1)}
    low = float(sum(by.get(w, 0.0) for w in range(1, 5)))
    higher = float(sum(v for w, v in by.items() if w >= 5))
    return {
        "by_degree": by,
        "frac_deg_1_to_4": low,
        "frac_deg_3": by.get(3, 0.0),
        "frac_other": higher,
    }


def center_cn(E: np.ndarray, n: int, m: int) -> np.ndarray:
    """Locked C_n = (n/m) E − max, per instance. Global phase under a scalar γ."""
    f = np.asarray(E, dtype=np.float64) * (float(n) / float(m))
    return f - f.max(axis=-1, keepdims=True)


def pauli_x_expectations(psi: torch.Tensor, n: int) -> torch.Tensor:
    """⟨X_i⟩ for a batch of bitstring-ordered states. var0 = MSB. Returns (B, n)."""
    B, dim = psi.shape
    if dim != (1 << int(n)):
        raise ValueError(f"psi dim {dim} != 2^{n}")
    idx = torch.arange(dim, device=psi.device)
    cols = []
    for i in range(int(n)):
        y = idx ^ (1 << (int(n) - 1 - i))
        cols.append((psi.conj() * psi.index_select(-1, y)).sum(dim=-1).real)
    return torch.stack(cols, dim=-1)


def energy_from_amps(psi: torch.Tensor, score: torch.Tensor, lam: float, n: int) -> torch.Tensor:
    """⟨score⟩ − λ Σ ⟨X_i⟩ from bitstring amplitudes."""
    probs = psi.real.square() + psi.imag.square()
    e_c = (probs * score.to(dtype=probs.dtype)).sum(dim=-1)
    return e_c - float(lam) * pauli_x_expectations(psi, n).sum(dim=-1)


class Circuit:
    """PR #13 Fock twin with a separate score diagonal.

    ``pulse`` is the Z-diagonal inside U_c^X. ``score`` is the diagonal used
    for ⟨H_c⟩. They coincide for the ZZZ champ. For the unsat arm, pulse is
    the normalized C_n and score is the raw unsat count. ``pieces`` is the
    degree-1..4 split of the pulse, used by Walsh γ.
    """

    def __init__(self, sim: FockSim, pulse: torch.Tensor, score: torch.Tensor, e0: torch.Tensor, pieces: torch.Tensor | None = None):
        if pulse.ndim != 2 or score.shape != pulse.shape:
            raise ValueError(f"pulse {tuple(pulse.shape)} score {tuple(score.shape)}")
        if e0.shape != (pulse.shape[0],):
            raise ValueError(f"e0 {tuple(e0.shape)}")
        self.sim = sim
        self.n = int(sim.n)
        self.pulse_f = sim.fock_of_bit_diag(pulse).to(dtype=sim.rdtype)
        self.score = score.to(dtype=sim.rdtype)
        self.e0 = e0.to(dtype=sim.rdtype)
        if pieces is None:
            self.pieces_f = None
        else:
            if tuple(pieces.shape) != (4,) + tuple(pulse.shape):
                raise ValueError(f"pieces {tuple(pieces.shape)}")
            self.pieces_f = torch.stack([sim.fock_of_bit_diag(pieces[w]) for w in range(4)], dim=0).to(dtype=sim.rdtype)

    def _evolve(self, gamma: torch.Tensor, phi: torch.Tensor, graded: bool, final_w: bool) -> torch.Tensor:
        """Fock-ordered state. ``final_w`` applies the closing W (Z readout)."""
        sim = self.sim
        L = int(gamma.shape[0])
        if tuple(phi.shape) != (L, 4):
            raise ValueError(f"phi {tuple(phi.shape)} != {(L, 4)}")
        B = int(self.score.shape[0])
        psi = torch.zeros(B, sim.dim, dtype=sim.cdtype, device=sim.device)
        psi[:, 0] = 1.0 + 0.0j
        g = gamma.to(dtype=sim.rdtype)
        for ell in range(L):
            psi = sim._apply_W(psi)
            if graded:
                if self.pieces_f is None:
                    raise RuntimeError("graded γ requested without Walsh pieces")
                phase = (g[ell].to(sim.cdtype).reshape(4, 1, 1) * self.pieces_f.to(sim.cdtype)).sum(dim=0)
            else:
                phase = g[ell].to(sim.cdtype) * self.pulse_f.to(sim.cdtype)
            psi = psi * torch.exp(-1j * phase)
            psi = sim._apply_W(psi)
            psi = sim._apply_snap(psi, phi[ell])
        if final_w:
            psi = sim._apply_W(psi)
        return psi

    def final_amps(self, gamma: torch.Tensor, phi: torch.Tensor, graded: bool) -> torch.Tensor:
        """Bitstring-ordered amplitudes after the full motif, including the final W."""
        return self._evolve(gamma, phi, graded, True).index_select(-1, self.sim.inv)

    def bit_histograms(self, gamma: torch.Tensor, phi: torch.Tensor, graded: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Z histogram after the final W, X histogram with that W omitted.

        Both are probabilities in bitstring order (var 0 = MSB). The gate is
        the same sequence as ``final_amps``. ⟨unsat⟩ is the Z histogram.
        ⟨X_i⟩ and intra-cavity ⟨X_i X_j⟩ are the X histogram.
        """
        sim = self.sim
        psi = self._evolve(gamma, phi, graded, False)
        p_fock_x = psi.real.square() + psi.imag.square()
        pX = p_fock_x.index_select(-1, sim.inv)
        psi = sim._apply_W(psi)
        p_fock_z = psi.real.square() + psi.imag.square()
        pZ = p_fock_z.index_select(-1, sim.inv)
        return pZ, pX

    def energy(self, gamma: torch.Tensor, phi: torch.Tensor, graded: bool, lam: float) -> torch.Tensor:
        return energy_from_amps(self.final_amps(gamma, phi, graded), self.score, lam, self.n)


def n_params(method: str, L: int) -> int:
    if method == "fourier_qaoa":
        return 2 * int(L)
    if method == "power_p":
        return 4 * int(L) + 1
    if method == "walsh":
        return 7 * int(L) + 1
    if method == "cost_only":
        return int(L)
    raise ValueError(method)


def decode(method: str, theta: torch.Tensor, L: int, n: int):
    """Return (gamma, phi, graded). phi is the Hamming shell, shape (L, 4)."""
    L = int(L)
    if method == "fourier_qaoa":
        gamma = fourier_schedule(theta[:L], L)
        beta = fourier_schedule(theta[L:], L)
        k = torch.arange(4, device=theta.device, dtype=theta.dtype)
        phi = (-beta.reshape(L, 1)) * k.reshape(1, 4)
        return gamma, phi, False
    if method == "power_p":
        g = theta[:L]
        c1 = theta[L:2 * L]
        c2 = theta[2 * L:3 * L]
        c3 = theta[3 * L:4 * L]
        p = theta[4 * L]
        return g, shell_phi(c1, c2, c3, p, n), False
    if method == "walsh":
        g = theta[: 4 * L].reshape(L, 4)
        off = 4 * L
        c1 = theta[off:off + L]
        c2 = theta[off + L:off + 2 * L]
        c3 = theta[off + 2 * L:off + 3 * L]
        p = theta[off + 3 * L]
        return g, shell_phi(c1, c2, c3, p, n), True
    if method == "cost_only":
        g = theta[:L]
        phi = g.new_zeros(L, 4)
        return g, phi, False
    raise ValueError(method)


def init_fourier(L: int, seed: int, dtype, device) -> torch.Tensor:
    rng = np.random.default_rng(50_000 + 1000 * int(seed) + 10 * int(L))
    ug, ub = paper_fourier_init(int(L), int(L))
    raw = np.concatenate([ug, ub])
    jit = rng.normal(0.0, 0.01, size=raw.shape)
    return torch.tensor(raw + jit, dtype=dtype, device=device)


def power_theta_from_fourier(theta_f: torch.Tensor, L: int) -> torch.Tensor:
    """Dual of a trained Fourier point: γ, c1=−β, c2=c3=0, p=0."""
    gamma = fourier_schedule(theta_f[:L], L).detach()
    beta = fourier_schedule(theta_f[L:], L).detach()
    z = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    p = torch.zeros((), dtype=theta_f.dtype, device=theta_f.device)
    return torch.cat([gamma, (-beta).detach(), z, z.clone(), p.reshape(1)])


def walsh_theta_from_fourier(theta_f: torch.Tensor, L: int) -> torch.Tensor:
    """PR #11 warm start: every Walsh degree gets the scalar Fourier γ. p=0, c2=c3=0."""
    gamma = fourier_schedule(theta_f[:L], L).detach()
    beta = fourier_schedule(theta_f[L:], L).detach()
    g = gamma.reshape(L, 1).expand(L, 4).reshape(-1).clone()
    z = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    p = torch.zeros((), dtype=theta_f.dtype, device=theta_f.device)
    return torch.cat([g, (-beta).detach(), z, z.clone(), p.reshape(1)])


def cost_only_from_fourier(theta_f: torch.Tensor, L: int) -> torch.Tensor:
    """Drop the mixer. γ is the Fourier schedule; SNAP phases stay 0."""
    return fourier_schedule(theta_f[:L], L).detach().clone()


def load_split(n: int, count: int, kind: str, pack_path=None):
    """ZZZ instances plus the raw unsat diagonal on the same seeds.

    ``kind='train'`` reads ``beat_qaoa_12h/train_packs`` (base 51e9).
    ``kind='scale'`` is the scale1000 satisfiable prefix (base 54e9).
    The two bases do not overlap. n=12/15 scale rows are eval-only.
    """
    from explore_OCT29.lib.sat_gen import gen_instance

    n = int(n)
    count = int(count)
    if kind == "train":
        insts = load_train_instances(n, count, pack_path)
        E = np.asarray(np.load(pack_path)["train_E"][:count], dtype=np.float64)
        if E.shape != (count, 1 << n):
            raise RuntimeError(f"train E {E.shape}")
        return insts, E
    if kind != "scale":
        raise ValueError(kind)
    insts = []
    rows = []
    k = 0
    limit = max(8000, count * 500)
    while len(insts) < count:
        if k >= limit:
            raise RuntimeError(f"n={n}: only {len(insts)}/{count} sat scale1000 draws")
        seed = int(SCALE1000_BASE) + 10_000_000 * n + k
        k += 1
        _m, cl, E = gen_instance(n, seed, compute_E=True)
        if int(np.min(E)) != 0:
            continue
        insts.append(instance_from_clauses(n, seed, "scale1000", cl))
        rows.append(np.asarray(E, dtype=np.float64))
    return insts, np.stack(rows, axis=0)


def batch_ground(diags: np.ndarray, n: int, lam: float) -> dict:
    e0 = np.empty(diags.shape[0], dtype=np.float64)
    peak = np.empty_like(e0)
    entropy = np.empty_like(e0)
    residual = np.empty_like(e0)
    for i in range(diags.shape[0]):
        row = sparse_ground_state(diags[i], n, lam)
        if row["residual"] > 1e-5:
            raise RuntimeError(f"n={n} i={i} Lanczos residual {row['residual']:.2e}")
        e0[i] = row["E0"]
        peak[i] = row["peak"]
        entropy[i] = row["entropy"]
        residual[i] = row["residual"]
    return {"E0": e0, "peak": peak, "entropy": entropy, "residual": residual}


def summarize_eps(energy: torch.Tensor, e0: torch.Tensor, n: int, lam: float) -> dict:
    e = energy.detach()
    eps = (e - e0) / float(n)
    plus = -float(lam) * float(n)
    # f uses the |+⟩ field energy −λ n, matching the ZZZ champ (⟨H_c⟩_+=0).
    denom = (plus - e0).clamp_min(1e-12)
    f = (plus - e) / denom
    eps_np = eps.cpu().numpy()
    h = e.cpu().numpy()
    return {
        "med_eps": float(np.median(eps_np)),
        "mean_eps": float(np.mean(eps_np)),
        "iqr_eps": float(np.quantile(eps_np, 0.75) - np.quantile(eps_np, 0.25)),
        "med_H": float(np.median(h)),
        "med_H_over_n": float(np.median(h) / float(n)),
        "mean_H_over_n": float(np.mean(h) / float(n)),
        "med_f": float(np.median(f.cpu().numpy())),
        "E_plus_field": plus,
    }


def ratio_mark(med_method: float, med_ref: float) -> str:
    if not math.isfinite(med_ref) or abs(med_ref) < 1e-12:
        return "UNDEF"
    R = float(med_method) / float(med_ref)
    if R <= BEAT_R:
        return "BEAT"
    if R <= TIE_R:
        return "TIE"
    return "UNDER"


def cost_only_matches_one_pulse(sim: FockSim, pulse: torch.Tensor, gammas: torch.Tensor, lam: float) -> float:
    """Max |Δ⟨H⟩| between L cost-only layers and one layer with Γ = Σ γ.

    SNAP phases are 0, so each layer is exp(−i γ_ℓ H_c^X) and the product
    collapses. ``pulse`` is scored as H_c.
    """
    if gammas.ndim != 1:
        raise ValueError("gammas")
    circ = Circuit(sim, pulse, pulse, torch.zeros(pulse.shape[0], dtype=pulse.dtype))
    L = int(gammas.shape[0])
    phi = gammas.new_zeros(L, 4)
    e_many = circ.energy(gammas, phi, False, lam)
    g1 = gammas.sum().reshape(1)
    e_one = circ.energy(g1, gammas.new_zeros(1, 4), False, lam)
    return float((e_many - e_one).abs().max())
