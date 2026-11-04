"""SAFE-point Hamiltonian and three ansatzes for the HVA fairness baseline.

Hamiltonian (same observable for every method), SAFE point λ=0.75, μ=−0.50:

    H = diag(unsat) − 0.75 Σ_i X_i − 0.50 Σ_c Σ_{i<j in cavity c} X_i X_j

Cavities are disjoint 3-bit blocks, cavity c owns qubits [3c, 3c+3) with
qubit 0 = MSB of the bitstring index (same layout as explore_OCT29.lib.qaoa).
Within a cavity there are three pairs, so the XX sum has n terms total.

Readout is Z-histogram + X-histogram only. diag(unsat) is diagonal in Z.
ΣX and ΣXX are diagonal in the X basis, so ⟨X_i⟩ and ⟨X_i X_j⟩ come from
p_X = |H^{⊗n} |ψ⟩|². No Y readout.

Ansatzes (all start at |+⟩^{⊗n}; layer ℓ applies cost then mixer):

1. Fourier QAOA — 2 angles / layer (β_ℓ, γ_ℓ)
       U_C(γ) = exp(−i γ Cn),   Cn = unsat · n/m,  m = round(9.93 n)
       U_X(β) = ⊗_i exp(−i β X_i / 2)
   This is the existing qubit QAOA convention (cost, then transverse mixer).
   Cn is a positive rescaling of diag(unsat), so {U_C(γ)} is exactly the
   standard QAOA cost family for this H. γ does not see the XX term.

2. HVA QAOA — 3 angles / layer (β^X_ℓ, β^XX_ℓ, γ_ℓ)
       U_XX(β^XX) U_X(β^X) U_C(γ)
       U_XX = ∏_{cavity pairs ij} exp(−i β^XX X_i X_j / 2)
   β^XX = 0 recovers Fourier exactly (nested). XX pairs commute with each
   other, so U_XX is one cavity unitary, not a Trotter product. X and XX
   do not commute; X is applied first, then XX.

3. Hamming-shell SNAP + BinH U_c^X, Walsh-graded cost (the champ)
       U_mix(φ) U_C^Walsh(γ_{w=1..4})
       φ_{ℓ,k} phases Fock/Hamming weight k=0..3 inside each cavity:
           M(φ) = H^{⊗3} diag(e^{−i φ_{popcount}}) H^{⊗3}
       shared across cavities (same shell as Nov3rdAnsatz).
       Walsh cost keeps degrees 1..4 of Cn (4-SAT is exactly degree ≤4,
       so equal γ_w reproduces scalar γ·Cn up to a global phase).
       Free shells generate the permutation-symmetric family
       {ΣX, Σ_{i<j} XX, XXX} inside each cavity. Linear φ_k = −β k
       reproduces U_X(β) up to a global phase.

Parameter counts per layer: Fourier 2, HVA 3, SNAP 8 (φ_k has one global
phase redundancy → 7 physical). The fairness question is the mixer: HVA
can express the XX term that Fourier cannot; SNAP can also express XXX
and degree-dependent γ_w.

``ground_bundle`` / ``expectation_H`` take ``lambda_x`` and ``mu_xx``.
Defaults are this SAFE point. μ = 0 drops the cavity XX term.

Paper init (Boulebnane–Montanaro): β=β^X=0.01, β^XX=0, γ=γ_w=−0.01,
φ_k=−β k. On 4-SAT the three inits prepare the same state.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from explore_OCT29.lib.qaoa import _apply_x_mixer, clause_count
from explore_OCT29.lib.sat_gen import clause_count as sat_clause_count

# SAFE point. Signs are the coefficients in H, matching
# H = diag(unsat) − 0.75 ΣX − 0.50 ΣXX.
LAMBDA_X = -0.75
MU_XX = -0.50

# Paper init. Mixer convention is exp(−i β X/2), so φ_k = −β k.
INIT_BETA = 0.01
INIT_GAMMA = -0.01

_PC8 = (0, 1, 1, 2, 1, 2, 2, 3)  # popcount(0..7)


def clause_m(n: int) -> int:
    m = int(clause_count(n))
    if m != int(sat_clause_count(n)):
        raise RuntimeError(f"clause_count mismatch n={n}: {m} vs {sat_clause_count(n)}")
    return m


def cn_from_unsat(E: torch.Tensor, n: int) -> torch.Tensor:
    """Cn = unsat · n/m, same scale as explore_OCT29.lib.qaoa."""
    m = clause_m(n)
    return E * (float(n) / float(m))


_PC_CACHE: dict[tuple, torch.Tensor] = {}


def popcount_index(n: int, device=None, dtype=torch.long) -> torch.Tensor:
    dev = torch.device("cpu" if device is None else device)
    key = (int(n), str(dev), str(dtype))
    hit = _PC_CACHE.get(key)
    if hit is not None and hit.device == dev:
        return hit
    idx = torch.arange(1 << int(n), device=dev)
    pc = torch.zeros_like(idx)
    for k in range(int(n)):
        pc = pc + ((idx >> k) & 1)
    pc = pc.to(dtype=dtype)
    _PC_CACHE[key] = pc
    return pc


def fwht_unnormalized(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized FWHT on the last axis. Matches Nov_3rd.ansatz.fwht_unnormalized."""
    n = int(x.shape[-1])
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT length must be power of 2, got {n}")
    y = x.contiguous().clone()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def walsh_degree_diag(Cn: torch.Tensor, gamma_w: torch.Tensor) -> torch.Tensor:
    """Σ_{w=1..4} γ_w (Cn)_w. Cn is (..., 2^n); gamma_w is (4,).

    Degree 0 is dropped (global phase when γ_w are equal). Degrees >4 are
    dropped; for 4-SAT those coefficients vanish.
    """
    if gamma_w.numel() != 4:
        raise ValueError(f"gamma_w must have 4 entries, got {gamma_w.numel()}")
    nbits = int(round(math.log2(Cn.shape[-1])))
    F = fwht_unnormalized(Cn)
    pc = popcount_index(nbits, device=Cn.device)
    gw = gamma_w.to(dtype=Cn.dtype, device=Cn.device).reshape(4)
    G = torch.zeros_like(F)
    for w in range(1, 5):
        G = G + gw[w - 1] * torch.where(pc == w, F, torch.zeros_like(F))
    return fwht_unnormalized(G) / float(Cn.shape[-1])


def walsh_degree_components(Cn: torch.Tensor) -> torch.Tensor:
    """Degree-1..4 Walsh pieces of ``Cn``, shape ``(4, *Cn.shape)``.

    Linear in the spectrum, so ``Σ_w γ_w pieces[w]`` equals
    ``walsh_degree_diag(Cn, γ)``. Pieces do not depend on γ.
    """
    nbits = int(round(math.log2(Cn.shape[-1])))
    F = fwht_unnormalized(Cn)
    pc = popcount_index(nbits, device=Cn.device)
    scale = float(Cn.shape[-1])
    pieces = []
    for w in range(1, 5):
        Fw = torch.where(pc == w, F, torch.zeros_like(F))
        pieces.append(fwht_unnormalized(Fw) / scale)
    return torch.stack(pieces, dim=0)


def snap_state_from_components(
    comps: torch.Tensor,
    gamma_w: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Walsh-shell SNAP using precomputed degree pieces of Cn.

    ``comps`` is ``(4, B, 2^n)`` from ``walsh_degree_components`` and is
    treated as constant. Same unitary as ``snap_state`` on graded γ.
    """
    if comps.ndim != 3 or int(comps.shape[0]) != 4:
        raise ValueError(f"comps shape {tuple(comps.shape)} != (4, B, 2^n)")
    if phi.ndim != 2 or int(phi.shape[1]) != 4:
        raise ValueError(f"phi shape {tuple(phi.shape)} != (L, 4)")
    if gamma_w.ndim != 2 or tuple(gamma_w.shape) != tuple(phi.shape):
        raise ValueError("gamma_w shape must match phi (L, 4)")
    B, dim = int(comps.shape[1]), int(comps.shape[2])
    n = int(round(math.log2(dim)))
    L = int(phi.shape[0])
    H3 = hadamard3(comps.dtype, comps.device)
    psi = plus_state(B, n, comps.dtype, comps.device)
    for ell in range(L):
        diag = torch.tensordot(gamma_w[ell], comps, dims=([0], [0]))
        psi = cost_phase(psi, diag)
        psi = apply_all_cavities(psi, shell_unitary(phi[ell], H3), n)
    return psi


def walsh_power_by_degree(E: np.ndarray) -> np.ndarray:
    """Normalized Walsh power per degree. E is one instance, shape (2^n,)."""
    x = torch.as_tensor(np.asarray(E, dtype=np.float64))
    nbits = int(round(math.log2(x.numel())))
    F = fwht_unnormalized(x)
    pwr = F.square()
    pc = popcount_index(nbits)
    total = float(pwr.sum().clamp(min=1e-30))
    out = np.zeros(nbits + 1, dtype=np.float64)
    for w in range(nbits + 1):
        out[w] = float(pwr[pc == w].sum()) / total
    return out


def hadamard3(dtype: torch.dtype, device=None) -> torch.Tensor:
    """H^{⊗3}. First Kronecker factor is the high bit of 0..7."""
    s = 1.0 / math.sqrt(2.0)
    H1 = torch.tensor([[s, s], [s, -s]], dtype=torch.float64, device=device)
    H3 = H1
    for _ in range(2):
        H3 = torch.kron(H3, H1)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return H3.to(dtype=cdtype)


def _complex_dtype(real: torch.dtype) -> torch.dtype:
    if real == torch.float32:
        return torch.complex64
    if real == torch.float64:
        return torch.complex128
    raise ValueError(f"unsupported real dtype {real}")


def shell_unitary(phi4: torch.Tensor, H3: torch.Tensor) -> torch.Tensor:
    """M(φ) = H^{⊗3} diag(e^{−i φ_popcount}) H^{⊗3}. phi4 is (4,)."""
    pc = torch.tensor(_PC8, device=phi4.device, dtype=torch.long)
    phase = torch.exp(-1j * phi4.to(dtype=H3.dtype)[pc])
    return H3 @ torch.diag(phase) @ H3


def xx_pair_eigenvalues(device=None, dtype=torch.float64) -> torch.Tensor:
    """Σ_{i<j} s_i s_j for the 8 X-basis bitstrings. High bit = first qubit."""
    vals = []
    for g in range(8):
        bits = ((g >> 2) & 1, (g >> 1) & 1, g & 1)
        s = [1.0 - 2.0 * b for b in bits]
        vals.append(s[0] * s[1] + s[0] * s[2] + s[1] * s[2])
    return torch.tensor(vals, device=device, dtype=dtype)


def xx_unitary(beta: torch.Tensor, H3: torch.Tensor) -> torch.Tensor:
    """exp(−i (β/2) Σ_{i<j} X_i X_j) on one cavity. β scalar tensor."""
    ev = xx_pair_eigenvalues(device=beta.device, dtype=torch.float64).to(H3.real.dtype)
    phase = torch.exp(-1j * (beta.to(H3.real.dtype) / 2) * ev.to(H3.dtype))
    return H3 @ torch.diag(phase) @ H3


def apply_cavity_unitary(psi: torch.Tensor, U: torch.Tensor, n: int, cav: int) -> torch.Tensor:
    """Apply 8×8 U to cavity `cav` (qubits 3c..3c+2, MSB-first). psi (B, 2^n)."""
    b = 3
    left = cav * b
    right = n - (cav + 1) * b
    B = psi.shape[0]
    x = psi.reshape(B, 1 << left, 8, 1 << right)
    y = torch.einsum("ij,bljr->blir", U, x)
    return y.reshape(B, 1 << n)


def apply_all_cavities(psi: torch.Tensor, U: torch.Tensor, n: int) -> torch.Tensor:
    for cav in range(n // 3):
        psi = apply_cavity_unitary(psi, U, n, cav)
    return psi


def plus_state(batch: int, n: int, dtype: torch.dtype, device=None) -> torch.Tensor:
    cdtype = _complex_dtype(dtype)
    dim = 1 << int(n)
    amp = 1.0 / math.sqrt(dim)
    return torch.full((batch, dim), amp, dtype=cdtype, device=device)


def cost_phase(psi: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """psi *= exp(−i diag). diag real, shape (B, 2^n) or broadcastable."""
    return psi * torch.exp(-1j * diag.to(dtype=psi.dtype))


def fourier_state(
    Cn: torch.Tensor,
    betas: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Standard 2-angle QAOA. Cn (B, 2^n), betas/gammas (L,)."""
    if Cn.ndim != 2:
        raise ValueError("Cn must be (B, 2^n)")
    B, dim = Cn.shape
    n = int(round(math.log2(dim)))
    betas = betas.reshape(-1)
    gammas = gammas.reshape(-1)
    if int(betas.numel()) != int(gammas.numel()):
        raise ValueError("beta/gamma length mismatch")
    psi = plus_state(B, n, Cn.dtype, Cn.device)
    for ell in range(int(gammas.numel())):
        psi = cost_phase(psi, gammas[ell] * Cn)
        psi = _apply_x_mixer(psi, betas[ell], n)
    return psi


def hva_state(
    Cn: torch.Tensor,
    beta_x: torch.Tensor,
    beta_xx: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """HVA layer: U_XX(β^XX) U_X(β^X) U_C(γ). β^XX=0 ≡ Fourier."""
    if Cn.ndim != 2:
        raise ValueError("Cn must be (B, 2^n)")
    B, dim = Cn.shape
    n = int(round(math.log2(dim)))
    beta_x = beta_x.reshape(-1)
    beta_xx = beta_xx.reshape(-1)
    gammas = gammas.reshape(-1)
    L = int(gammas.numel())
    if int(beta_x.numel()) != L or int(beta_xx.numel()) != L:
        raise ValueError("HVA angle length mismatch")
    H3 = hadamard3(Cn.dtype, Cn.device)
    psi = plus_state(B, n, Cn.dtype, Cn.device)
    for ell in range(L):
        psi = cost_phase(psi, gammas[ell] * Cn)
        psi = _apply_x_mixer(psi, beta_x[ell], n)
        U = xx_unitary(beta_xx[ell], H3)
        psi = apply_all_cavities(psi, U, n)
    return psi


def snap_state(
    Cn: torch.Tensor,
    gamma_w: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Walsh-shell champ. gamma_w (L, 4), phi (L, 4) with φ_k on Hamming weight k.

    gamma_w may also be (L,) — scalar γ·Cn, used by the dual-identity check.
    """
    if Cn.ndim != 2:
        raise ValueError("Cn must be (B, 2^n)")
    B, dim = Cn.shape
    n = int(round(math.log2(dim)))
    if phi.ndim != 2 or phi.shape[1] != 4:
        raise ValueError(f"phi shape {tuple(phi.shape)} != (L, 4)")
    L = int(phi.shape[0])
    if gamma_w.ndim == 1:
        if int(gamma_w.numel()) != L:
            raise ValueError("scalar gamma length != L")
        graded = False
    elif gamma_w.ndim == 2:
        if tuple(gamma_w.shape) != (L, 4):
            raise ValueError(f"gamma_w shape {tuple(gamma_w.shape)} != {(L, 4)}")
        graded = True
    else:
        raise ValueError("gamma_w must be (L,) or (L, 4)")
    H3 = hadamard3(Cn.dtype, Cn.device)
    psi = plus_state(B, n, Cn.dtype, Cn.device)
    for ell in range(L):
        if graded:
            diag = walsh_degree_diag(Cn, gamma_w[ell])
        else:
            diag = gamma_w[ell] * Cn
        psi = cost_phase(psi, diag)
        U = shell_unitary(phi[ell], H3)
        psi = apply_all_cavities(psi, U, n)
    return psi


def apply_H_all(psi: torch.Tensor, n: int) -> torch.Tensor:
    """H^{⊗n}. Qubit q=0 is the MSB, matching _apply_x_mixer."""
    B, dim = psi.shape
    s = psi.new_tensor(1.0 / math.sqrt(2.0))
    for q in range(n):
        v = psi.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        a0 = v[:, :, 0, :]
        a1 = v[:, :, 1, :]
        psi = torch.stack((s * (a0 + a1), s * (a0 - a1)), dim=2).reshape(B, dim)
    return psi


def _prob(psi: torch.Tensor) -> torch.Tensor:
    return psi.real.square() + psi.imag.square()


def sum_X(pX: torch.Tensor, n: int) -> torch.Tensor:
    """Σ_i ⟨X_i⟩ from the X-basis histogram. pX is (B, 2^n)."""
    B = pX.shape[0]
    acc = torch.zeros(B, dtype=pX.dtype, device=pX.device)
    for q in range(n):
        v = pX.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        p0 = v[:, :, 0, :].sum(dim=(1, 2))
        p1 = v[:, :, 1, :].sum(dim=(1, 2))
        acc = acc + (p0 - p1)
    return acc


def sum_XX_cavities(pX: torch.Tensor, n: int) -> torch.Tensor:
    """Σ_c Σ_{i<j in c} ⟨X_i X_j⟩ from the X-basis histogram."""
    B = pX.shape[0]
    acc = torch.zeros(B, dtype=pX.dtype, device=pX.device)
    for cav in range(n // 3):
        qs = (3 * cav, 3 * cav + 1, 3 * cav + 2)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            q1, q2 = qs[a], qs[b]
            if q1 > q2:
                q1, q2 = q2, q1
            left = q1
            mid = q2 - q1 - 1
            right = n - q2 - 1
            v = pX.reshape(B, 1 << left, 2, 1 << mid, 2, 1 << right)
            p00 = v[:, :, 0, :, 0, :].sum(dim=(1, 2, 3))
            p01 = v[:, :, 0, :, 1, :].sum(dim=(1, 2, 3))
            p10 = v[:, :, 1, :, 0, :].sum(dim=(1, 2, 3))
            p11 = v[:, :, 1, :, 1, :].sum(dim=(1, 2, 3))
            acc = acc + (p00 - p01 - p10 + p11)
    return acc


def expectation_H(
    psi: torch.Tensor,
    unsat: torch.Tensor,
    n: int,
    lambda_x: float = LAMBDA_X,
    mu_xx: float = MU_XX,
) -> torch.Tensor:
    """⟨H⟩ from Z histogram (unsat) and X histogram (X and XX). Shape (B,)."""
    pZ = _prob(psi)
    e_diag = (pZ * unsat.to(dtype=pZ.dtype)).sum(dim=-1)
    pX = _prob(apply_H_all(psi, n))
    return e_diag + float(lambda_x) * sum_X(pX, n) + float(mu_xx) * sum_XX_cavities(pX, n)


def _add_bit_flip(out: np.ndarray, v: np.ndarray, bit: int, coeff: float) -> None:
    """out += coeff * v with LSB `bit` flipped. `bit` 0 is the least significant bit."""
    step = 1 << int(bit)
    src = v.reshape(-1, 2, step)
    dst = out.reshape(-1, 2, step)
    dst[:, 0, :] += coeff * src[:, 1, :]
    dst[:, 1, :] += coeff * src[:, 0, :]


def _add_two_bit_flip(out: np.ndarray, v: np.ndarray, bit_a: int, bit_b: int, coeff: float) -> None:
    """out += coeff * v with two distinct LSB indices flipped."""
    hi, lo = (bit_a, bit_b) if bit_a > bit_b else (bit_b, bit_a)
    if hi == lo:
        raise ValueError("XX flip needs two distinct bits")
    step = 1 << lo
    mid = 1 << (hi - lo - 1)
    src = v.reshape(-1, 2, mid, 2, step)
    dst = out.reshape(-1, 2, mid, 2, step)
    dst[:, 0, :, 0, :] += coeff * src[:, 1, :, 1, :]
    dst[:, 1, :, 1, :] += coeff * src[:, 0, :, 0, :]
    dst[:, 0, :, 1, :] += coeff * src[:, 1, :, 0, :]
    dst[:, 1, :, 0, :] += coeff * src[:, 0, :, 1, :]


def hamiltonian_matvec(
    v: np.ndarray,
    unsat: np.ndarray,
    n: int,
    lambda_x: float = LAMBDA_X,
    mu_xx: float = MU_XX,
) -> np.ndarray:
    """Real-symmetric matvec of H. unsat and v are flat length 2^n.

    Qubit q=0 is the MSB, so it flips LSB index n-1-q. Same operator as the
    dense builder.
    """
    vv = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    out = np.asarray(unsat, dtype=np.float64).reshape(-1) * vv
    lx = float(lambda_x)
    mx = float(mu_xx)
    for q in range(n):
        _add_bit_flip(out, vv, n - 1 - q, lx)
    for cav in range(n // 3):
        qs = (3 * cav, 3 * cav + 1, 3 * cav + 2)
        bits = [n - 1 - q for q in qs]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            _add_two_bit_flip(out, vv, bits[a], bits[b], mx)
    return out


def dense_H(
    unsat: np.ndarray,
    n: int,
    lambda_x: float = LAMBDA_X,
    mu_xx: float = MU_XX,
) -> np.ndarray:
    """Dense real H, for n≤9 checks and exact ground states."""
    unsat = np.asarray(unsat, dtype=np.float64).reshape(-1)
    N = 1 << int(n)
    if unsat.size != N:
        raise ValueError(f"unsat {unsat.size} != 2^{n}")
    H = np.diag(unsat)
    idx = np.arange(N)
    places = [1 << (n - 1 - q) for q in range(n)]
    pair_places = []
    for cav in range(n // 3):
        qs = (3 * cav, 3 * cav + 1, 3 * cav + 2)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            pair_places.append(places[qs[a]] ^ places[qs[b]])
    for place, coeff in (
        [(p, lambda_x) for p in places] + [(p, mu_xx) for p in pair_places]
    ):
        partner = idx ^ place
        sel = idx < partner
        H[idx[sel], partner[sel]] += coeff
        H[partner[sel], idx[sel]] += coeff
    return H


def ground_bundle(
    unsat: np.ndarray,
    n: int,
    *,
    tol: float = 1e-9,
    lambda_x: float = LAMBDA_X,
    mu_xx: float = MU_XX,
) -> dict:
    """Lowest eigenpair of H. Dense eigh for n≤9, ARPACK for larger n.

    ``lambda_x`` and ``mu_xx`` are the coefficients in
    H = diag(unsat) + λ_x Σ X + μ Σ_{cavity pairs} XX.
    Defaults are the SAFE point (−0.75, −0.50). μ=0 is the transverse-field
    family. ``psi`` is real, unit-norm, and signed so that ⟨+|ψ⟩ ≥ 0.
    ``peak`` is max_x |⟨x|ψ⟩|² in the Z basis.
    """
    unsat = np.asarray(unsat, dtype=np.float64).reshape(-1)
    N = 1 << int(n)
    if unsat.size != N:
        raise ValueError(f"unsat {unsat.size} != 2^{n}")
    lx = float(lambda_x)
    mx = float(mu_xx)
    if n <= 9:
        evals, evecs = np.linalg.eigh(dense_H(unsat, n, lambda_x=lx, mu_xx=mx))
        e = float(evals[0])
        psi = np.real(evecs[:, 0]).astype(np.float64, copy=False)
        resid = 0.0
    else:
        from scipy.sparse.linalg import LinearOperator, eigsh

        def mv(v):
            return hamiltonian_matvec(
                np.asarray(v, dtype=np.float64).reshape(-1), unsat, n, lambda_x=lx, mu_xx=mx
            )

        op = LinearOperator((N, N), matvec=mv, dtype=np.float64)
        # |+⟩ is a strong starting guess: the field and XX terms favor it,
        # while the diagonal penalty is only the mean unsat.
        v0 = np.full(N, 1.0 / math.sqrt(N))
        ncv = 48 if n >= 15 else 32
        val, vec = eigsh(op, k=1, which="SA", tol=tol, maxiter=max(2000, 4 * N), ncv=ncv, v0=v0)
        e = float(val[0])
        psi = np.real(np.asarray(vec[:, 0], dtype=np.float64))
        Hv = mv(psi)
        resid = float(np.linalg.norm(Hv - e * psi))
        if resid > 1e-6 * max(1.0, abs(e)):
            raise RuntimeError(f"eigsh residual {resid:.3e} at n={n} E0={e:.6f}")
    norm = float(np.linalg.norm(psi))
    if norm == 0.0:
        raise RuntimeError(f"zero ground vector at n={n}")
    psi = psi / norm
    if float(psi.sum()) < 0.0:
        psi = -psi
    return {
        "E0": e,
        "psi": psi,
        "peak": float(np.max(psi * psi)),
        "residual": float(resid),
    }


def ground_energy(
    unsat: np.ndarray,
    n: int,
    *,
    tol: float = 1e-9,
    lambda_x: float = LAMBDA_X,
    mu_xx: float = MU_XX,
) -> float:
    """Lowest eigenvalue of H. Dense eigh for n≤9, ARPACK for larger n."""
    return float(ground_bundle(unsat, n, tol=tol, lambda_x=lambda_x, mu_xx=mu_xx)["E0"])


def ground_energies(
    unsat_batch: np.ndarray,
    n: int,
    *,
    log=None,
    lambda_x: float = LAMBDA_X,
    mu_xx: float = MU_XX,
) -> np.ndarray:
    unsat_batch = np.asarray(unsat_batch)
    out = np.empty(unsat_batch.shape[0], dtype=np.float64)
    for i in range(unsat_batch.shape[0]):
        out[i] = ground_energy(unsat_batch[i], n, lambda_x=lambda_x, mu_xx=mu_xx)
        if log is not None and (i + 1 == unsat_batch.shape[0] or (i + 1) % 8 == 0):
            log(f"E0 n{n} {i+1}/{unsat_batch.shape[0]} last={out[i]:.4f}")
    return out


def cavity_m2_eigenvalues() -> np.ndarray:
    """(Σ_i s_i / 3)^2 on the 8 X-basis states of one cavity. High bit = first qubit.

    Bit value 0 is the X=+1 outcome, matching ``xx_pair_eigenvalues``.
    """
    vals = np.empty(8, dtype=np.float64)
    for g in range(8):
        bits = ((g >> 2) & 1, (g >> 1) & 1, g & 1)
        s = [1.0 - 2.0 * b for b in bits]
        m = (s[0] + s[1] + s[2]) / 3.0
        vals[g] = m * m
    return vals


def cavity_lambda_from_px(pX: torch.Tensor | np.ndarray, n: int) -> np.ndarray:
    """Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the joint X histogram.

    ``pX`` is (2^n,) or (B, 2^n) in bitstring order, qubit 0 = MSB, cavities
    of 3 qubits. Returns shape (C,) or (B, C). Λ→1 means the cavity is locked
    to one X-magnetization (an effective spin). A Z-basis product state has
    Λ=1/3. Algebraically Λ_c = 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩.
    """
    n = int(n)
    if n % 3 != 0:
        raise ValueError(f"n={n} is not a multiple of 3")
    arr = np.asarray(pX.detach().cpu().numpy() if torch.is_tensor(pX) else pX, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != (1 << n):
        raise ValueError(f"pX shape {arr.shape} != (B, 2^{n})")
    C = n // 3
    table = cavity_m2_eigenvalues()
    marg = arr.reshape(arr.shape[0], *([8] * C))
    out = np.empty((arr.shape[0], C), dtype=np.float64)
    for c in range(C):
        axes = tuple(ax for ax in range(1, C + 1) if ax != c + 1)
        m = marg.sum(axis=axes) if axes else marg
        out[:, c] = m @ table
    return out[0] if single else out


def lambda_of_state(psi: torch.Tensor | np.ndarray, n: int) -> np.ndarray:
    """Per-cavity Λ from the X histogram of a statevector (batch or one vector)."""
    n = int(n)
    if torch.is_tensor(psi):
        t = psi
    else:
        t = torch.as_tensor(np.asarray(psi), dtype=torch.float64)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if t.shape[-1] != (1 << n):
        raise ValueError(f"psi {tuple(t.shape)} != 2^{n}")
    if not t.is_complex():
        t = t.to(dtype=torch.complex128)
    pX = _prob(apply_H_all(t, n))
    return cavity_lambda_from_px(pX, n)


def product_energy_xz(theta: np.ndarray, diag: np.ndarray, n: int, lam: float, mu: float) -> float:
    """Energy of a real XZ product state on diag(unsat) − λ ΣX + μ ΣXX.

    θ_i parametrizes cos(θ/2)|0⟩ + sin(θ/2)|1⟩, so ⟨Z⟩=cos θ and ⟨X⟩=sin θ.
    ``lam`` is the positive field (0.75 at SAFE); ``mu`` is the XX coefficient
    (−0.50 at SAFE). This is the same family used to define the SAFE product gap.
    """
    n = int(n)
    theta = np.asarray(theta, dtype=np.float64).reshape(n)
    diag = np.asarray(diag, dtype=np.float64).reshape(1 << n)
    c2 = np.cos(0.5 * theta) ** 2
    s2 = np.sin(0.5 * theta) ** 2
    p = np.ones(1, dtype=np.float64)
    for i in range(n):
        p = np.kron(p, np.array([c2[i], s2[i]], dtype=np.float64))
    e = float(p @ diag)
    sx = np.sin(theta)
    e -= float(lam) * float(sx.sum())
    for c0 in range(n // 3):
        i, j, k = 3 * c0, 3 * c0 + 1, 3 * c0 + 2
        e += float(mu) * float(sx[i] * sx[j] + sx[i] * sx[k] + sx[j] * sx[k])
    return float(e)


def best_product_energy(diag: np.ndarray, n: int, lam: float, mu: float, seed: int = 0) -> float:
    """Lowest XZ-product energy from the SAFE start set, then L-BFGS on the best three."""
    from scipy.optimize import minimize

    n = int(n)
    diag = np.asarray(diag, dtype=np.float64)
    x_best = int(np.argmin(diag))
    bits = np.array([(x_best >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.float64)
    starts = [
        np.zeros(n),
        np.full(n, np.pi),
        np.full(n, 0.5 * np.pi),
        np.full(n, -0.5 * np.pi),
        bits * np.pi,
        (1.0 - bits) * np.pi,
    ]
    mix = int(seed) + 17 * int(n) + int(round(1000 * (float(mu) + 10.0)))
    rng = np.random.default_rng(mix)
    for _ in range(3):
        starts.append(rng.uniform(-np.pi, np.pi, size=n))
    scored = [(product_energy_xz(th, diag, n, lam, mu), th) for th in starts]
    scored.sort(key=lambda row: row[0])
    best = float(scored[0][0])
    for _e, th in scored[:3]:
        res = minimize(lambda t: product_energy_xz(t, diag, n, lam, mu), th, method="L-BFGS-B")
        if np.isfinite(res.fun):
            best = min(best, float(res.fun))
    return float(best)


def product_gap(e_prod: float, e0: float) -> float:
    """(E_prod − E_0) / |E_0|. The SAFE median-gap bar is 0.05."""
    return float((float(e_prod) - float(e0)) / max(abs(float(e0)), 1e-8))


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|⟨a|b⟩| per batch row. a, b (B, N)."""
    return torch.sum(a.conj() * b, dim=-1).abs()


def init_angles(method: str, L: int, dtype=torch.float32, device=None):
    """Paper init. Returns a dict of tensors (not Parameters)."""
    L = int(L)
    if method == "fourier":
        return {
            "betas": torch.full((L,), INIT_BETA, dtype=dtype, device=device),
            "gammas": torch.full((L,), INIT_GAMMA, dtype=dtype, device=device),
        }
    if method == "hva":
        return {
            "beta_x": torch.full((L,), INIT_BETA, dtype=dtype, device=device),
            "beta_xx": torch.zeros(L, dtype=dtype, device=device),
            "gammas": torch.full((L,), INIT_GAMMA, dtype=dtype, device=device),
        }
    if method == "snap":
        k = torch.arange(4, dtype=dtype, device=device)
        phi = (-INIT_BETA * k).reshape(1, 4).expand(L, 4).contiguous().clone()
        return {
            "gamma_w": torch.full((L, 4), INIT_GAMMA, dtype=dtype, device=device),
            "phi": phi,
        }
    raise ValueError(method)


def prepare_state(method: str, Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    if method == "fourier":
        return fourier_state(Cn, angles["betas"], angles["gammas"])
    if method == "hva":
        return hva_state(Cn, angles["beta_x"], angles["beta_xx"], angles["gammas"])
    if method == "snap":
        return snap_state(Cn, angles["gamma_w"], angles["phi"])
    raise ValueError(method)


def method_parameter_list(method: str, angles: dict) -> list[torch.nn.Parameter]:
    """Wrap angle tensors as Parameters, in place (the dict is updated)."""
    keys = {
        "fourier": ("betas", "gammas"),
        "hva": ("beta_x", "beta_xx", "gammas"),
        "snap": ("gamma_w", "phi"),
    }[method]
    params = []
    for k in keys:
        p = torch.nn.Parameter(angles[k].detach().clone())
        angles[k] = p
        params.append(p)
    return params


@torch.no_grad()
def instance_eps(
    method: str,
    angles: dict,
    unsat: torch.Tensor,
    e0: torch.Tensor,
    n: int,
    *,
    chunk: int = 8,
) -> torch.Tensor:
    """ε = ⟨H⟩ − E_0 for each instance. unsat (B, 2^n), e0 (B,)."""
    B = unsat.shape[0]
    out = []
    for s in range(0, B, chunk):
        Eu = unsat[s : s + chunk]
        Cn = cn_from_unsat(Eu, n)
        psi = prepare_state(method, Cn, angles)
        h = expectation_H(psi, Eu, n)
        out.append(h - e0[s : s + chunk].to(dtype=h.dtype, device=h.device))
    return torch.cat(out, dim=0)


def summarize_eps(eps: Iterable[float]) -> dict:
    a = np.asarray(list(eps), dtype=np.float64).reshape(-1)
    return {
        "B": int(a.size),
        "med_eps": float(np.median(a)),
        "mean_eps": float(np.mean(a)),
        "min_eps": float(np.min(a)),
        "max_eps": float(np.max(a)),
    }


def self_check(pack_E: np.ndarray | None = None) -> None:
    """Raise AssertionError if the ansatz or the readout is inconsistent."""
    torch.manual_seed(0)
    n = 6
    # Synthetic unsat: one 4-bit pattern indicator, degree ≤4, plus a second clause.
    E = np.zeros(1 << n, dtype=np.float64)
    # clause on bits 0,1,2,3 (MSB-first): unsat when all four are 1
    for x in range(1 << n):
        b0 = (x >> (n - 1)) & 1
        b1 = (x >> (n - 2)) & 1
        b2 = (x >> (n - 3)) & 1
        b3 = (x >> (n - 4)) & 1
        if b0 == 1 and b1 == 1 and b2 == 1 and b3 == 1:
            E[x] += 1
        # second clause on bits 2,3,4,5, unsat on 0,1,0,1
        b4 = (x >> (n - 5)) & 1
        b5 = (x >> (n - 6)) & 1
        if (b2, b3, b4, b5) == (0, 1, 0, 1):
            E[x] += 1
    pwr = walsh_power_by_degree(E)
    hi = float(pwr[5:].sum()) if pwr.size > 5 else 0.0
    if hi > 1e-10:
        raise AssertionError(f"4-local unsat has Walsh degree>4 power {hi}")

    if pack_E is not None:
        pwr_p = walsh_power_by_degree(np.asarray(pack_E[0], dtype=np.float64))
        hi_p = float(pwr_p[5:].sum())
        if hi_p > 1e-8:
            raise AssertionError(f"pack instance Walsh degree>4 power {hi_p}")

    Et = torch.tensor(E, dtype=torch.float64).reshape(1, -1)
    Cn = cn_from_unsat(Et, n)

    # |+⟩ energy: mean(E) − 0.75 n − 0.50 n_pairs, n_pairs = n.
    psi = plus_state(1, n, torch.float64)
    h_plus = float(expectation_H(psi, Et, n))
    h_formula = float(E.mean() + LAMBDA_X * n + MU_XX * n)
    if abs(h_plus - h_formula) > 1e-8:
        raise AssertionError(f"|+⟩ energy {h_plus} != {h_formula}")

    # |+⟩ is one X-outcome per cavity, so Λ=1. A Z bitstring has uniform X
    # marginals, so Λ=1/3. The pair identity is Λ = 1/3 + (2/9) Σ⟨XX⟩.
    lam_plus = lambda_of_state(psi, n)
    if np.max(np.abs(lam_plus - 1.0)) > 1e-8:
        raise AssertionError(f"|+⟩ Λ {lam_plus}")
    z0 = torch.zeros(1, 1 << n, dtype=torch.float64)
    z0[0, 0] = 1.0
    lam_z = lambda_of_state(z0, n)
    if np.max(np.abs(lam_z - (1.0 / 3.0))) > 1e-8:
        raise AssertionError(f"|0⟩ Λ {lam_z}")
    pX = _prob(apply_H_all(psi, n))
    xx = float(sum_XX_cavities(pX, n))
    # n/3 cavities, 3 pairs each, |+⟩ has ⟨XX⟩=1, so xx = n.
    lam_id = 1.0 / 3.0 + (2.0 / 9.0) * (xx / (n / 3.0))
    if abs(float(lam_plus.mean()) - lam_id) > 1e-8:
        raise AssertionError(f"Λ identity {float(lam_plus.mean())} != {lam_id}")

    th_plus = np.full(n, 0.5 * np.pi)
    e_prod_plus = product_energy_xz(th_plus, E, n, lam=0.75, mu=-0.50)
    if abs(e_prod_plus - h_formula) > 1e-8:
        raise AssertionError(f"product |+⟩ {e_prod_plus} != {h_formula}")
    gap_chk = product_gap(e_prod_plus, h_formula - 1.0)
    if gap_chk <= 0:
        raise AssertionError("product gap sign")

    # Histogram ⟨H⟩ matches a dense quadratic form on a random state.
    rng = np.random.default_rng(1)
    re = torch.tensor(rng.normal(size=1 << n), dtype=torch.float64)
    im = torch.tensor(rng.normal(size=1 << n), dtype=torch.float64)
    v = torch.complex(re, im)
    v = v / torch.linalg.vector_norm(v)
    h_hist = float(expectation_H(v.reshape(1, -1), Et, n))
    H = dense_H(E, n)
    vv = v.detach().numpy()
    h_dense = float(np.vdot(vv, H @ vv).real)
    if abs(h_hist - h_dense) > 1e-8:
        raise AssertionError(f"hist {h_hist} != dense {h_dense}")

    rnd = rng.normal(size=1 << n)
    if np.max(np.abs(hamiltonian_matvec(rnd, E, n) - H @ rnd)) > 1e-8:
        raise AssertionError("matvec != dense H")

    e0 = ground_energy(E, n)
    e0_dense = float(np.linalg.eigvalsh(H)[0])
    if abs(e0 - e0_dense) > 1e-8:
        raise AssertionError(f"E0 {e0} != {e0_dense}")
    if e0 > float(E.min()) + 1e-8:
        raise AssertionError("E0 above a diagonal entry")

    # HVA β^XX=0 matches Fourier on a random angle draw.
    L = 3
    g = torch.tensor(rng.normal(scale=0.2, size=L), dtype=torch.float64)
    b = torch.tensor(rng.normal(scale=0.3, size=L), dtype=torch.float64)
    zxx = torch.zeros(L, dtype=torch.float64)
    sf = fourier_state(Cn, b, g)
    sh = hva_state(Cn, b, zxx, g)
    fid = float(state_fidelity(sf, sh)[0])
    if fid < 1 - 1e-8:
        raise AssertionError(f"HVA βXX=0 fidelity {fid}")

    # Linear shell + scalar γ matches Fourier (global phase allowed).
    phi = (-b.reshape(L, 1)) * torch.arange(4, dtype=torch.float64).reshape(1, 4)
    ss = snap_state(Cn, g, phi)
    fid_s = float(state_fidelity(sf, ss)[0])
    if fid_s < 1 - 1e-8:
        raise AssertionError(f"linear shell fidelity {fid_s}")

    # Equal Walsh weights match scalar cost up to global phase.
    gw = g.reshape(L, 1).expand(L, 4).contiguous()
    sw = snap_state(Cn, gw, phi)
    fid_w = float(state_fidelity(sf, sw)[0])
    if fid_w < 1 - 1e-8:
        raise AssertionError(f"equal-Walsh fidelity {fid_w}")

    # Finite-difference gradient of ⟨H⟩ w.r.t. β^XX is nonzero and matches autograd.
    bx = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    bxx = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    gg = torch.tensor([-0.15], dtype=torch.float64, requires_grad=True)
    psi_g = hva_state(Cn, bx, bxx, gg)
    loss = expectation_H(psi_g, Et, n).sum()
    loss.backward()
    if bxx.grad is None or abs(float(bxx.grad)) < 1e-6:
        raise AssertionError(f"βXX grad vanished: {None if bxx.grad is None else float(bxx.grad)}")
    eps = 1e-6
    with torch.no_grad():
        def energy_at(delta: float) -> float:
            psi_d = hva_state(Cn, bx.detach(), bxx.detach() + delta, gg.detach())
            return float(expectation_H(psi_d, Et, n))

        fd = (energy_at(eps) - energy_at(-eps)) / (2 * eps)
    if abs(fd - float(bxx.grad)) > 1e-4 * max(1.0, abs(fd)):
        raise AssertionError(f"βXX grad {float(bxx.grad)} != fd {fd}")

    # Nov3rd champ circuit matches this qubit SNAP on scalar γ and on Walsh γ.
    _check_nov3rd(E, n)


def _check_nov3rd(E: np.ndarray, n: int) -> None:
    """Bitstring probabilities of Nov3rdAnsatz equal snap_state."""
    from Nov_3rd.ansatz import make_sim

    L = 2
    device = torch.device("cpu")
    sim = make_sim(n, L, device)
    Et = torch.tensor(E, dtype=torch.float32)
    Cn = cn_from_unsat(Et.reshape(1, -1), n).to(torch.float64)
    g = torch.tensor([0.17, -0.11], dtype=torch.float32)
    phi = torch.tensor(
        [[0.0, -0.2, 0.15, 0.4], [0.05, 0.1, -0.3, 0.2]],
        dtype=torch.float32,
    )
    # scalar γ
    out = sim.forward_nov3rd(Et, g, phi)
    p_nov = _fock_probs_to_bits(out["probs"], sim.grid_to_x, n)
    psi = snap_state(Cn, g.to(torch.float64), phi.to(torch.float64))
    p = _prob(psi)[0].detach().to(torch.float64)
    if torch.max(torch.abs(p - p_nov.to(torch.float64))) > 2e-5:
        raise AssertionError(
            f"Nov3rd scalar prob mismatch {float(torch.max(torch.abs(p - p_nov))):.3e}"
        )
    # Walsh γ
    gw = torch.tensor(
        [[0.17, -0.02, 0.08, 0.01], [-0.11, 0.05, -0.04, 0.2]],
        dtype=torch.float32,
    )
    out_w = sim.forward_nov3rd(Et, gw, phi)
    p_nov_w = _fock_probs_to_bits(out_w["probs"], sim.grid_to_x, n)
    psi_w = snap_state(Cn, gw.to(torch.float64), phi.to(torch.float64))
    p_w = _prob(psi_w)[0].detach().to(torch.float64)
    err = float(torch.max(torch.abs(p_w - p_nov_w.to(torch.float64))))
    if err > 2e-5:
        raise AssertionError(f"Nov3rd Walsh prob mismatch {err:.3e}")


def _fock_probs_to_bits(probs: torch.Tensor, grid_to_x: torch.Tensor, n: int) -> torch.Tensor:
    C = n // 3
    sl = tuple(slice(0, 8) for _ in range(C))
    p = probs[sl]
    flat = torch.zeros(1 << n, dtype=torch.float64)
    flat[grid_to_x.reshape(-1).to(torch.long)] = p.reshape(-1).to(torch.float64)
    return flat


def angles_to_lists(method: str, angles: dict) -> dict:
    out = {}
    for k, v in angles.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().to(torch.float64).numpy().tolist()
        else:
            out[k] = v
    out["method"] = method
    return out
