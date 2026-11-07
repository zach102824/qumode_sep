"""Arm A — transverse-field Max-E on the locked Nov_3rd gate.

Hamiltonian (stoquastic sign, λ>0):

    H = H_c − λ Σ_i X_i
    H_c(z) = unsat count of the scale1000 / TEB clause pack (raw diagonal)

The locked pulse is still U_c^X = W exp(−i γ C_n) W with
C_n = (n/m) H_c − max, W = H^{⊗3}. That is the same one-parameter family as
exp(−i γ_phys H_c): γ absorbs the constant n/m, and the max subtraction is a
global phase. Readout of ⟨H⟩ uses only the Z histogram (clause energy) and
single-qubit X histograms. No YY / XX correlators.

Raw unsat, not (n/m)·unsat, is the diagonal that puts the λ grid
{0.25…1.5} on the entangled side of the crossover. The normalized diagonal
makes λ≳0.25 already a product |+⟩ state.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from Nov_3rd.ansatz import fwht_unnormalized, make_sim
from Nov_3rd import settings as S
from explore_OCT29.lib.packing import NU
from explore_OCT29.lib.qaoa import _apply_x_mixer
from explore_OCT29.lib.sat_gen import clause_count, instances

# Same seed base as Nov_3rd/runs/.../regen_scale1000.py
SCB = 54_000_000_000
LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
# Paper Boulebnane–Montanaro init, on the normalized cost the gate already uses.
PAPER_BETA = 0.01
PAPER_GAMMA = -0.01
BEAT_RATIO = 0.98
TIE_RATIO = 1.02


def generate_instances(n: int, count: int, base: int = SCB) -> dict:
    """First `count` satisfiable TEB instances at the locked clause ratio."""
    n = int(n)
    count = int(count)
    ins = instances(n, count, int(base), compute_E=True)
    E = np.stack([np.asarray(i["E"], dtype=np.float64) for i in ins], axis=0)
    seeds = np.asarray([int(i["seed"]) for i in ins], dtype=np.int64)
    m = int(clause_count(n))
    if int(E.shape[1]) != (1 << n):
        raise RuntimeError(f"E width {E.shape[1]} != 2^{n}")
    return {
        "n": n,
        "m": m,
        "E": E,
        "seeds": seeds,
        "source": f"sat_gen.instances base={int(base)} (scale1000 SCB prefix)",
    }


def try_load_scale1000(n: int, count: int) -> dict | None:
    """Use the on-disk scale1000 pack when it is present. Do not require it."""
    path = S.SCALE1000_DIR / f"data_n{n}.npz"
    if not path.exists():
        return None
    blob = np.load(path, allow_pickle=False)
    key = "scale1000_E" if "scale1000_E" in blob.files else "test_E"
    seed_key = "scale1000_seeds" if "scale1000_seeds" in blob.files else "test_seeds"
    E = np.asarray(blob[key], dtype=np.float64)
    seeds = np.asarray(blob[seed_key], dtype=np.int64)
    if E.shape[0] < count:
        return None
    m = int(blob["m"]) if "m" in blob.files else int(clause_count(n))
    return {
        "n": int(n),
        "m": m,
        "E": E[:count],
        "seeds": seeds[:count],
        "source": str(path),
    }


def load_instances(n: int, count: int) -> dict:
    loaded = try_load_scale1000(n, count)
    if loaded is not None:
        return loaded
    return generate_instances(n, count)


@lru_cache(maxsize=4)
def pauli_x_sum(n: int) -> np.ndarray:
    """Dense Σ_i X_i on bitstrings, var0 = MSB. Shape (2^n, 2^n)."""
    n = int(n)
    I = np.eye(2, dtype=np.float64)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    out = np.zeros((1 << n, 1 << n), dtype=np.float64)
    for q in range(n):
        acc = None
        for i in range(n):
            op = X if i == q else I
            acc = op if acc is None else np.kron(acc, op)
        out += acc
    return out


def ground_pair(Hc: np.ndarray, xsum: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Lowest eigenpair of diag(Hc) − λ Σ X. Hc is the raw unsat diagonal."""
    H = np.diag(np.asarray(Hc, dtype=np.float64)) - float(lam) * xsum
    evals, evecs = np.linalg.eigh(H)
    return float(evals[0]), np.asarray(evecs[:, 0], dtype=np.complex128)


def _half_entropy(psi: np.ndarray, n: int) -> float:
    nL = int(n) // 2
    mat = psi.reshape(1 << nL, 1 << (n - nL))
    sing = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(sing * sing, 0.0, None)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def _mean_qubit_entropy(psi: np.ndarray, n: int) -> float:
    dim = 1 << int(n)
    ent = []
    for q in range(int(n)):
        shift = int(n) - 1 - q
        mask = 1 << shift
        idx = np.arange(dim)
        low = idx[(idx & mask) == 0]
        a0 = psi[low]
        a1 = psi[low | mask]
        p0 = float(np.vdot(a0, a0).real)
        p1 = float(np.vdot(a1, a1).real)
        coh = np.sum(a0 * np.conj(a1))
        det = p0 * p1 - float(abs(coh) ** 2)
        disc = max(0.0, 1.0 - 4.0 * det)
        root = math.sqrt(disc)
        spec = (0.5 * (1.0 + root), 0.5 * (1.0 - root))
        s = 0.0
        for ev in spec:
            if ev > 1e-12:
                s -= ev * math.log(ev)
        ent.append(s)
    return float(np.mean(ent))


def instance_quantumness(psi: np.ndarray, E: np.ndarray, xsum: np.ndarray, n: int) -> dict:
    prob = np.abs(psi) ** 2
    return {
        "peak": float(prob.max()),
        "F_sat": float(prob[np.asarray(E) == 0].sum()),
        "S_half": _half_entropy(psi, n),
        "S1": _mean_qubit_entropy(psi, n),
        "X_per_n": float(np.real(np.vdot(psi, xsum @ psi)) / float(n)),
    }


def _median(vals: list[float]) -> float:
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def quantumness_scan(pack: dict, lambdas: tuple[float, ...] = LAMBDA_GRID) -> dict:
    """Exact diagonalization on one pack. Returns per-λ medians and GS vectors' overlaps."""
    n = int(pack["n"])
    E = np.asarray(pack["E"], dtype=np.float64)
    xsum = pauli_x_sum(n)
    per_lam: dict[str, dict] = {}
    prev_psi = None
    prev_lam = None
    for lam in lambdas:
        rows = []
        psis = []
        e0s = []
        for b in range(E.shape[0]):
            e0, psi = ground_pair(E[b], xsum, float(lam))
            # Pin a phase so neighboring-λ fidelities are well defined as moduli.
            k = int(np.argmax(np.abs(psi)))
            psi = psi * np.exp(-1j * np.angle(psi[k]))
            psis.append(psi)
            e0s.append(e0)
            rows.append(instance_quantumness(psi, E[b], xsum, n))
        chi = None
        fid = None
        if prev_psi is not None:
            fids = np.asarray(
                [abs(np.vdot(a, b)) ** 2 for a, b in zip(prev_psi, psis)],
                dtype=np.float64,
            )
            dlam = float(lam) - float(prev_lam)
            fid = _median(fids.tolist())
            chi = _median(((1.0 - fids) / (dlam ** 2)).tolist())
        prev_psi = psis
        prev_lam = float(lam)
        per_lam[f"{float(lam):.2f}"] = {
            "lambda": float(lam),
            "peak_med": _median([r["peak"] for r in rows]),
            "peak_q90": float(np.quantile([r["peak"] for r in rows], 0.90)),
            "F_sat_med": _median([r["F_sat"] for r in rows]),
            "S_half_med": _median([r["S_half"] for r in rows]),
            "S1_med": _median([r["S1"] for r in rows]),
            "X_per_n_med": _median([r["X_per_n"] for r in rows]),
            "E0_med": _median(e0s),
            "gap_from_bit_med": _median([(-e0) / float(n) for e0 in e0s]),
            "frac_peak_below_0.5": float(np.mean([r["peak"] < 0.5 for r in rows])),
            "fid_to_prev": fid,
            "chi_F": chi,
            "B": int(E.shape[0]),
        }
    return {"n": n, "m": int(pack["m"]), "B": int(E.shape[0]), "per_lambda": per_lam}


def choose_lambda(scans: dict[int, dict], lambdas: tuple[float, ...] = LAMBDA_GRID) -> dict:
    """Pick λ on the grid whose GS is not a bitstring and half-chain entropy peaks.

    Tie-break toward 1, which is the prior in the explore note. A λ is quantum
    when both n=6 and n=9 have median bitstring peak < 0.5.
    """
    quantum = []
    for lam in lambdas:
        key = f"{float(lam):.2f}"
        rows = {n: scans[n]["per_lambda"][key] for n in scans}
        if all(rows[n]["peak_med"] < 0.5 for n in rows):
            score = float(sum(rows[n]["S_half_med"] for n in rows))
            quantum.append((score, -abs(float(lam) - 1.0), float(lam), rows))
    if not quantum:
        return {
            "lambda": None,
            "quantum": False,
            "reason": "every grid λ has a classical bitstring GS (median peak ≥ 0.5) on n=6 or n=9",
        }
    quantum.sort(key=lambda t: (t[0], t[1]), reverse=True)
    score, _, lam, rows = quantum[0]
    sh = {n: rows[n]["S_half_med"] for n in rows}
    peaks = {n: rows[n]["peak_med"] for n in rows}
    return {
        "lambda": float(lam),
        "quantum": True,
        "score_S_half": score,
        "S_half_med": sh,
        "peak_med": peaks,
        "reason": (
            f"λ={lam:.2f} maximizes median half-chain entropy among grid points "
            f"with median bitstring peak < 0.5 on both n=6 and n=9 "
            f"(S_half={sh}, peak={peaks}); ties break toward 1"
        ),
    }


def e0_batch(E: np.ndarray, lam: float) -> np.ndarray:
    n = int(round(math.log2(E.shape[1])))
    xsum = pauli_x_sum(n)
    out = np.empty(E.shape[0], dtype=np.float64)
    for b in range(E.shape[0]):
        out[b], _ = ground_pair(E[b], xsum, float(lam))
    return out


def mode_matrix(L: int, Q: int, kind: str, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """DST-IV sine ('dst') or DCT-IV cosine ('dct'), shape (L, Q). 1-based ℓ, q."""
    if kind not in ("dst", "dct"):
        raise ValueError(kind)
    if Q > L or Q < 1 or L < 1:
        raise ValueError(f"bad L,Q = {L},{Q}")
    ell = torch.arange(1, L + 1, device=device, dtype=dtype)
    q = torch.arange(1, Q + 1, device=device, dtype=dtype)
    ang = (q - 0.5).unsqueeze(0) * (ell - 0.5).unsqueeze(1) * (math.pi / float(L))
    return torch.sin(ang) if kind == "dst" else torch.cos(ang)


def analyze(schedule: torch.Tensor, kind: str) -> torch.Tensor:
    """Full-mode DST/DCT. Inverse of synthesize at Q=L, up to roundoff."""
    x = schedule.detach().reshape(-1).to(dtype=torch.float64)
    L = int(x.numel())
    M = mode_matrix(L, L, kind, device=x.device, dtype=torch.float64)
    return (2.0 / float(L)) * (M.transpose(0, 1) @ x)


def synthesize(coeffs: torch.Tensor, L: int, kind: str) -> torch.Tensor:
    c = coeffs.reshape(-1)
    M = mode_matrix(int(L), int(c.numel()), kind, device=c.device, dtype=c.dtype)
    return M @ c


def paper_modes(L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier coefficients of the constant paper schedule β=0.01, γ=−0.01."""
    betas = torch.full((int(L),), PAPER_BETA, dtype=torch.float64)
    gammas = torch.full((int(L),), PAPER_GAMMA, dtype=torch.float64)
    A = analyze(gammas, "dst").to(dtype=torch.float32)
    B = analyze(betas, "dct").to(dtype=torch.float32)
    return A.contiguous(), B.contiguous()


def angles_from_modes(A: torch.Tensor, B: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """β from DCT, γ from DST. Returns (betas, gammas), each (L,)."""
    gammas = synthesize(A, int(L), "dst")
    betas = synthesize(B, int(L), "dct")
    return betas, gammas


def sum_x(psi: torch.Tensor) -> torch.Tensor:
    """Σ_i ⟨X_i⟩ for a batch of Z-basis states. var0 = MSB. Returns (B,)."""
    B, dim = psi.shape
    n = int(round(math.log2(dim)))
    if (1 << n) != dim:
        raise ValueError(f"state dim {dim} is not 2^n")
    total = torch.zeros(B, device=psi.device, dtype=torch.float32)
    for q in range(n):
        v = psi.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        a0 = v[:, :, 0, :]
        a1 = v[:, :, 1, :]
        total = total + (2.0 * (a0.conj() * a1).real.sum(dim=(1, 2)))
    return total


def energy_from_z_state(psi: torch.Tensor, E: torch.Tensor, lam: float) -> torch.Tensor:
    """⟨H_c⟩ − λ Σ ⟨X_i⟩ from a Z-basis statevector. E is raw unsat, shape (B, 2^n)."""
    probs = psi.real.square() + psi.imag.square()
    e_c = (probs * E.to(dtype=probs.dtype)).sum(dim=-1)
    return e_c - float(lam) * sum_x(psi)


def qaoa_energy(betas: torch.Tensor, gammas: torch.Tensor, E: torch.Tensor, lam: float, n: int) -> torch.Tensor:
    """Fourier/standard QAOA energy of H = H_c − λ Σ X.

    Cost phase uses the locked normalized diagonal C_n = (n/m) H_c
    (global shift dropped). Mixer is exp(−i β X/2), same convention as
    explore_OCT29.lib.qaoa.
    """
    if E.ndim != 2 or E.shape[1] != (1 << int(n)):
        raise ValueError(f"E shape {tuple(E.shape)} incompatible with n={n}")
    m = int(clause_count(int(n)))
    Cn = E.to(dtype=torch.float32) * (float(n) / float(m))
    B, dim = Cn.shape
    device = Cn.device
    betas = betas.to(device=device, dtype=torch.float32).reshape(-1)
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    psi = torch.full((B, dim), 1.0 / math.sqrt(dim), dtype=torch.complex64, device=device)
    for ell in range(int(gammas.numel())):
        psi = psi * torch.exp(-1j * gammas[ell].to(torch.complex64) * Cn.to(torch.complex64))
        psi = _apply_x_mixer(psi, betas[ell], int(n))
    return energy_from_z_state(psi, E.to(dtype=torch.float32), lam)


def plus_energy(E: torch.Tensor, lam: float, n: int) -> torch.Tensor:
    """⟨+|H|+⟩ = mean_z H_c(z) − λ n. The QAOA initial state."""
    return E.to(dtype=torch.float32).mean(dim=-1) - float(lam) * float(n)


def phi_from_shell(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """φ_k = c1 k + c2 k² + c3 k³, k=0..3. Same polynomial as beat_params.phi_from_c."""
    k = torch.arange(S.B_BITS + 1, device=c1.device, dtype=c1.dtype)
    L = int(c1.reshape(-1).numel())
    return (
        c1.reshape(L, 1) * k.reshape(1, -1)
        + c2.reshape(L, 1) * k.reshape(1, -1) ** 2
        + c3.reshape(L, 1) * k.reshape(1, -1) ** 3
    )


def a_of_n(n: int, p: torch.Tensor) -> torch.Tensor:
    """(8/(n−1))^p. Equals 1 at n=9 for every p."""
    return torch.exp(p.to(dtype=torch.float32) * math.log(8.0 / float(int(n) - 1)))


class SnapSimulator:
    """Batched locked Nov_3rd forward, then Z-hist + X-hist energy of H."""

    def __init__(self, n: int, L: int, device: torch.device | str = "cpu"):
        self.n = int(n)
        self.L = int(L)
        self.m = int(clause_count(self.n))
        self.device = torch.device(device)
        sim = make_sim(self.n, self.L, self.device)
        if int(sim.nf) != int(NU):
            raise RuntimeError(f"arm A expects nf=NU=8 so the Fock space is the qubit space, got nf={sim.nf}")
        self.C = int(sim.C)
        self.W = sim.Wmat
        self.grid = sim.grid_to_x  # (8,)*C long, binary
        self.popcount = sim.popcount  # (8,)
        self.masks = sim._walsh_degree_masks  # (4, 2^n) bool
        self._ref = sim  # keep buffers alive

    def cost_grid(self, E: torch.Tensor) -> torch.Tensor:
        """Locked C_n on the Fock grid, shape (B, 8, 8, ...)."""
        idx = self.grid.reshape(-1)
        Eg = E.index_select(-1, idx)
        Cn = Eg * (float(self.n) / float(self.m))
        reduce_dims = tuple(range(1, Eg.ndim))
        Cn = Cn - Cn.amax(dim=reduce_dims, keepdim=True)
        return Cn.reshape(E.shape[0], *self.grid.shape)

    def graded_grid(self, E: torch.Tensor, gamma_w: torch.Tensor) -> torch.Tensor:
        """Walsh degrees 1..4 of C_n, placed on the Fock grid. gamma_w is (4,)."""
        f = E.to(dtype=torch.float32) * (float(self.n) / float(self.m))
        f = f - f.amax(dim=-1, keepdim=True)
        F = fwht_unnormalized(f)
        gw = gamma_w.reshape(4).to(dtype=f.dtype, device=f.device)
        sel = torch.where(
            self.masks.unsqueeze(1),
            F.unsqueeze(0),
            torch.zeros((), dtype=F.dtype, device=F.device),
        )
        G = (gw.view(4, 1, 1) * sel).sum(dim=0)
        diag_bits = fwht_unnormalized(G) / float(f.shape[-1])
        idx = self.grid.reshape(-1)
        return diag_bits.index_select(-1, idx).reshape(E.shape[0], *self.grid.shape)

    def _apply_all(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            psi = torch.tensordot(M, psi, dims=([1], [c + 1]))
            psi = psi.movedim(0, c + 1)
        return psi

    def _apply_snap(self, psi: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            shape = [1] * psi.ndim
            shape[c + 1] = int(phases.numel())
            psi = psi * phases.view(*shape)
        return psi

    def forward_fock(self, E: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Final Fock state after the locked motif. gamma (L,) or (L, 4); phi (L, 4)."""
        B = int(E.shape[0])
        graded = gamma.ndim == 2
        if not graded and tuple(gamma.shape) != (self.L,):
            raise ValueError(f"gamma shape {tuple(gamma.shape)} != {(self.L,)}")
        if graded and tuple(gamma.shape) != (self.L, 4):
            raise ValueError(f"graded gamma shape {tuple(gamma.shape)} != {(self.L, 4)}")
        if tuple(phi.shape) != (self.L, S.B_BITS + 1):
            raise ValueError(f"phi shape {tuple(phi.shape)}")
        gamma = gamma.to(dtype=torch.float32, device=self.device)
        phi = phi.to(dtype=torch.float32, device=self.device)
        E = E.to(dtype=torch.float32, device=self.device)
        psi = torch.zeros((B,) + (int(NU),) * self.C, dtype=torch.complex64, device=self.device)
        psi[(slice(None),) + (0,) * self.C] = 1.0 + 0.0j
        Cn = None if graded else self.cost_grid(E)
        for ell in range(self.L):
            psi = self._apply_all(psi, self.W)
            diag = self.graded_grid(E, gamma[ell]) if graded else Cn * gamma[ell]
            psi = psi * torch.exp(-1j * diag.to(torch.complex64))
            psi = self._apply_all(psi, self.W)
            th = phi[ell][self.popcount]
            psi = self._apply_snap(psi, torch.exp(-1j * th.to(torch.complex64)))
        psi = self._apply_all(psi, self.W)
        return psi

    def to_bits(self, psi: torch.Tensor) -> torch.Tensor:
        """Scatter Fock amplitudes into bitstring order (var0 = MSB) via grid_to_x."""
        B = int(psi.shape[0])
        flat = psi.reshape(B, -1)
        idx = self.grid.reshape(-1).long()
        out = flat.new_zeros(B, int(idx.numel()))
        out.scatter_(1, idx.view(1, -1).expand(B, -1), flat)
        return out

    def energy(self, E: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, lam: float) -> torch.Tensor:
        psi = self.to_bits(self.forward_fock(E, gamma, phi))
        return energy_from_z_state(psi, E.to(dtype=torch.float32, device=self.device), lam)


def epsilon(energy: torch.Tensor, e0: torch.Tensor, n: int) -> torch.Tensor:
    return (energy - e0.to(device=energy.device, dtype=energy.dtype)) / float(n)


def ratio_mark(med_method: float, med_qaoa: float) -> str:
    if not math.isfinite(med_qaoa) or abs(med_qaoa) < 1e-12:
        return "UNDEF"
    R = float(med_method) / float(med_qaoa)
    if R <= BEAT_RATIO:
        return "BEAT"
    if R <= TIE_RATIO:
        return "TIE"
    return "UNDER"
