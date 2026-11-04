#!/usr/bin/env python3
"""NEW_SAFE cost-only ablation.

H = diag(unsat) - 0.75 * sum_i X_i

Arms (shared angles, train n in {6,9}, frozen n=12):
  A  full Walsh-graded U_c^X + Hamming-shell SNAP
  B  identity SNAP (repeated U_c^X only; GRAPE proxy of a diagonal pulse)
  C  gamma frozen at 0 (SNAP only / no cost pulse)
  D  Fourier QAOA on the same H (q=L Zhou modes; complete at these L)

Does not read or write n21, settings locks, or the phase2 ladder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.ansatz import fwht_unnormalized, make_sim  # noqa: E402

LAM = 0.75  # H = diag(unsat) - LAM * sum X
TRAIN_NS = (6, 9)
EVAL_N = 12
TRAIN_N = 64
VAL_N = 48
N12_PRIMARY = 64
N12_CONFIRM = 64
N6_EXT = 64
SEEDS = (0, 1, 2, 3, 4)
ADAM_STEPS = 180
ADAM_LR = 0.05
LBFGS_ROUNDS = 10
LBFGS_MAX_ITER = 8
PASS_RATIO = 0.85
KILL_RATIO_B = 0.95

OUT = _ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "cost_only_ablation_NEW_SAFE"
TRAIN_PACKS = _ROOT / "Nov_3rd" / "runs" / "beat_qaoa_12h" / "train_packs"
SCALE6 = _ROOT / "explore_OCT29" / "data" / "scale1000" / "data_n6.npz"

# sum_i X_i sparsity pattern, keyed by n
_XSTRUCT: dict[int, csr_matrix] = {}
_UH: dict[tuple, torch.Tensor] = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[new_safe]", *a, flush=True)


def clause_m(n: int) -> int:
    return int(round(9.93 * n))


def popcount_tensor(n: int, device, dtype) -> torch.Tensor:
    key = (n, device.type, str(device))
    # device-sensitive cache stored outside; rebuild if device/dtype differs
    idx = np.arange(1 << n, dtype=np.int64)
    pc = np.bitwise_count(idx)
    return torch.as_tensor(pc, device=device, dtype=dtype)


def shell_design(n: int, device, dtype) -> torch.Tensor:
    """S[x, k] = number of 3-bit cavities with popcount k. phase = S @ phi."""
    N = 1 << n
    C = n // 3
    S = np.zeros((N, 4), dtype=np.float64)
    for x in range(N):
        for c in range(C):
            pc = 0
            for j in range(3):
                v = 3 * c + j  # var 0 = MSB
                pc += (x >> (n - 1 - v)) & 1
            S[x, pc] += 1.0
    return torch.as_tensor(S, device=device, dtype=dtype)


def x_struct(n: int) -> csr_matrix:
    if n in _XSTRUCT:
        return _XSTRUCT[n]
    N = 1 << n
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for q in range(n):
        bit = 1 << (n - 1 - q)
        for i in range(N):
            j = i ^ bit
            if i < j:
                rows.extend((i, j))
                cols.extend((j, i))
                data.extend((1.0, 1.0))
    mat = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)
    _XSTRUCT[n] = mat
    return mat


def ground_state(E: np.ndarray, n: int) -> tuple[float, float]:
    """Return E0 and max bitstring probability of the GS."""
    H = diags(E.astype(np.float64), 0, format="csr") - LAM * x_struct(n)
    N = 1 << n
    if n <= 9:
        # dense eigh is exact and still cheap at 512
        Hd = H.toarray()
        w, v = np.linalg.eigh(Hd)
        e0 = float(w[0])
        peak = float(np.max(np.abs(v[:, 0]) ** 2))
        return e0, peak
    w, v = eigsh(H, k=1, which="SA", tol=1e-12, maxiter=4000, ncv=48)
    e0 = float(w[0])
    Hv = H @ v[:, 0]
    resid = float(np.linalg.norm(Hv - e0 * v[:, 0]))
    if resid > 1e-6:
        raise RuntimeError(f"eigsh residual {resid:.3e} at n={n}")
    peak = float(np.max(np.abs(v[:, 0]) ** 2))
    return e0, peak


def load_E(n: int) -> tuple[np.ndarray, np.ndarray]:
    path = TRAIN_PACKS / f"train_n{n}.npz"
    d = np.load(path, allow_pickle=True)
    E = np.asarray(d["train_E"], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"]).astype(np.int64)
    if E.shape[1] != (1 << n):
        raise RuntimeError(f"{path} width {E.shape[1]} != 2^{n}")
    return E, seeds


def load_scale6() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(SCALE6, allow_pickle=True)
    E = np.asarray(d["scale1000_E"], dtype=np.float64)
    seeds = np.asarray(d["scale1000_seeds"]).astype(np.int64)
    return E, seeds


def unitary_hadamard(n: int, dtype, device) -> torch.Tensor:
    """H^{⊗n} with qubit 0 = MSB (slow index), matching the bitstring layout.

    dtype may be a real or complex torch dtype. The matrix is cached.
    """
    key = (n, str(dtype), str(device))
    if key not in _UH:
        real = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
        h1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=real, device=device) / math.sqrt(2.0)
        h = h1
        for _ in range(n - 1):
            h = torch.kron(h, h1)
        if dtype in (torch.complex64, torch.complex128):
            h = h.to(dtype=dtype)
        _UH[key] = h
    return _UH[key]


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh–Hadamard on the last axis (same convention as ansatz.fwht)."""
    n = int(x.shape[-1])
    nbits = int(round(math.log2(n)))
    real = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = unitary_hadamard(nbits, real, x.device)
    return math.sqrt(n) * (x @ h)


def cost_f(E: torch.Tensor, n: int) -> torch.Tensor:
    """Ansatz cost_bitstring: (E * n/m) - amax, last axis."""
    f = E * (float(n) / float(clause_m(n)))
    return f - f.amax(dim=-1, keepdim=True)


def walsh_diag_batch(E: torch.Tensor, gamma_w: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
    """Σ_{w=1..4} γ_w E_w of cost_bitstring. gamma_w: (4,)."""
    n = int(round(math.log2(E.shape[-1])))
    f = cost_f(E, n)
    F = fwht(f)
    G = torch.zeros_like(F)
    for w in range(4):
        mask = (pc == (w + 1)).to(dtype=F.dtype)
        G = G + gamma_w[w] * F * mask
    return fwht(G) / float(E.shape[-1])


def apply_H(psi: torch.Tensor, n: int) -> torch.Tensor:
    h = unitary_hadamard(n, psi.dtype, psi.device)
    return psi @ h


def apply_rx(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """⊗ exp(-i β X / 2), matching explore_OCT29.lib.qaoa."""
    c = torch.cos(beta / 2).to(dtype=psi.dtype)
    s = (-1j * torch.sin(beta / 2)).to(dtype=psi.dtype)
    B, N = psi.shape
    for q in range(n):
        v = psi.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        a0 = v[:, :, 0, :]
        a1 = v[:, :, 1, :]
        psi = torch.stack((c * a0 + s * a1, s * a0 + c * a1), dim=2).reshape(B, N)
    return psi


def phi_from_c(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    k = torch.arange(4, device=c1.device, dtype=c1.dtype)
    return c1[:, None] * k + c2[:, None] * (k ** 2) + c3[:, None] * (k ** 3)


def sum_x_expect(psi: torch.Tensor, n: int) -> torch.Tensor:
    B, N = psi.shape
    acc = torch.zeros(B, dtype=psi.real.dtype, device=psi.device)
    for q in range(n):
        v = psi.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        a0 = v[:, :, 0, :]
        a1 = v[:, :, 1, :]
        acc = acc + (2.0 * (a0.conj() * a1).real).sum(dim=(1, 2))
    return acc


def energy_of(psi: torch.Tensor, E: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = psi.real.square() + psi.imag.square()
    ed = (p * E).sum(dim=-1)
    ex = sum_x_expect(psi, n)
    return ed - LAM * ex, ed, ex


def forward_motif(
    E: torch.Tensor,
    gamma_w: torch.Tensor,
    phi: torch.Tensor,
    pc: torch.Tensor,
    design: torch.Tensor,
) -> torch.Tensor:
    """Nov_3rd forward in the bitstring basis (var0 = MSB).

    gamma_w: (L, 4), phi: (L, 4).
    |0> -> [W exp(-i Σ γ_w E_w) W -> shell SNAP]^L -> W
    """
    B, N = E.shape
    n = int(round(math.log2(N)))
    L = int(phi.shape[0])
    psi = torch.zeros(B, N, dtype=torch.complex128, device=E.device)
    psi[:, 0] = 1.0 + 0.0j
    for ell in range(L):
        psi = apply_H(psi, n)
        diag = walsh_diag_batch(E, gamma_w[ell], pc)
        psi = psi * torch.exp(-1j * diag.to(dtype=psi.dtype))
        psi = apply_H(psi, n)
        phase = design @ phi[ell]
        psi = psi * torch.exp(-1j * phase.to(dtype=psi.dtype))
    psi = apply_H(psi, n)
    return psi


def forward_qaoa(E: torch.Tensor, betas: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """|+> ∏_ℓ exp(-i β X/2) exp(-i γ Cn), Cn = E*(n/m) (no centering; global phase)."""
    B, N = E.shape
    n = int(round(math.log2(N)))
    Cn = E * (float(n) / float(clause_m(n)))
    psi = torch.full((B, N), 1.0 / math.sqrt(N), dtype=torch.complex128, device=E.device)
    L = int(betas.shape[0])
    for ell in range(L):
        psi = psi * torch.exp(-1j * gammas[ell].to(dtype=psi.dtype) * Cn.to(dtype=psi.dtype))
        psi = apply_rx(psi, betas[ell], n)
    return psi


def fourier_schedule(coeff: torch.Tensor, L: int) -> torch.Tensor:
    q = int(coeff.shape[0])
    ell = torch.arange(L, device=coeff.device, dtype=coeff.dtype)
    out = torch.zeros(L, device=coeff.device, dtype=coeff.dtype)
    for k in range(q):
        out = out + coeff[k] * torch.sin((k + 0.5) * (ell + 0.5) * math.pi / L)
    return out


def fit_constant_fourier(value: float, L: int, q: int) -> np.ndarray:
    ell = np.arange(L, dtype=np.float64)
    A = np.stack([np.sin((k + 0.5) * (ell + 0.5) * np.pi / L) for k in range(q)], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.full(L, value, dtype=np.float64), rcond=None)
    return coef


def plus_energy(E: torch.Tensor, n: int) -> torch.Tensor:
    return E.mean(dim=-1) - LAM * float(n)


def summarize_gaps(energy: np.ndarray, e0: np.ndarray, ed: np.ndarray, ex: np.ndarray) -> dict:
    gap = energy - e0
    return {
        "B": int(gap.size),
        "med_gap": float(np.median(gap)),
        "mean_gap": float(np.mean(gap)),
        "p25_gap": float(np.quantile(gap, 0.25)),
        "p75_gap": float(np.quantile(gap, 0.75)),
        "med_energy": float(np.median(energy)),
        "mean_energy": float(np.mean(energy)),
        "med_D": float(np.median(ed)),
        "med_X": float(np.median(ex)),
        "min_gap": float(np.min(gap)),
    }


def bitstring_probs_from_ansatz(sim, E_row: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor) -> np.ndarray:
    out = sim.forward_nov3rd(E_row, gamma, phi)
    probs = out["probs"].detach().reshape(-1).cpu().numpy().astype(np.float64)
    grid = sim.grid_to_x.detach().reshape(-1).cpu().numpy().astype(np.int64)
    n = int(sim.cfg.n)
    p = np.zeros(1 << n, dtype=np.float64)
    p[grid] = probs
    return p


def self_check(device) -> dict:
    """Circuit identity vs Nov3rdAnsatz, dual vs QAOA, and the cost-only bound."""
    log("self-check")
    report = {}
    n, L = 6, 2
    E_np, _ = load_E(n)
    E_np = E_np[:4]
    dtype = torch.float64
    pc = popcount_tensor(n, device, dtype)
    design = shell_design(n, device, dtype)
    E = torch.as_tensor(E_np, dtype=dtype, device=device)

    # |+> and |0> expectations
    N = 1 << n
    plus = torch.full((4, N), 1.0 / math.sqrt(N), dtype=torch.complex128, device=device)
    e_plus, d_plus, x_plus = energy_of(plus, E, n)
    e_ref = plus_energy(E, n)
    err_plus = float((e_plus - e_ref).abs().max())
    err_x = float((x_plus - n).abs().max())
    if err_plus > 1e-8 or err_x > 1e-8:
        raise RuntimeError(f"|+> energy mismatch {err_plus} {err_x}")

    # zero-parameter motif is |+>
    g0 = torch.zeros(L, 4, dtype=dtype, device=device)
    p0 = torch.zeros(L, 4, dtype=dtype, device=device)
    psi0 = forward_motif(E, g0, p0, pc, design)
    e0m, _, _ = energy_of(psi0, E, n)
    err_m = float((e0m - e_ref).abs().max())
    if err_m > 1e-8:
        raise RuntimeError(f"zero-param motif != |+> ({err_m})")

    # random identity-SNAP (arm B) cannot beat |+>
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    g_rand = torch.randn(L, 4, dtype=dtype, generator=gen).to(device)
    psi_b = forward_motif(E, g_rand, p0, pc, design)
    e_b, d_b, x_b = energy_of(psi_b, E, n)
    # populations flat => <D> = mean(E)
    if float((d_b - E.mean(-1)).abs().max()) > 1e-6:
        raise RuntimeError("identity SNAP changed <D>")
    if float((e_b - e_ref).min()) < -1e-6:
        raise RuntimeError("identity SNAP beat |+>")

    # gamma=0, random shell (arm C) stays at |+>
    c1 = torch.randn(L, dtype=dtype, generator=gen)
    c2 = torch.randn(L, dtype=dtype, generator=gen)
    c3 = torch.randn(L, dtype=dtype, generator=gen) * 0.3
    phi = phi_from_c(c1, c2, c3).to(device)
    psi_c = forward_motif(E, g0, phi, pc, design)
    e_c, _, _ = energy_of(psi_c, E, n)
    if float((e_c - e_ref).abs().max()) > 1e-6:
        raise RuntimeError("gamma=0 SNAP moved the energy")

    # Walsh equal-weight reconstructs cost up to a constant (degree <= 4)
    f = cost_f(E, n)
    g_eq = torch.tensor([0.37, 0.37, 0.37, 0.37], dtype=dtype, device=device)
    diag = walsh_diag_batch(E, g_eq, pc)
    centered = 0.37 * (f - f.mean(dim=-1, keepdim=True))
    recon = float((diag - centered).abs().max())
    if recon > 1e-6:
        raise RuntimeError(f"Walsh degree<=4 reconstruction error {recon}")

    # dual identity: linear shell + equal Walsh == QAOA energies
    betas = torch.tensor([0.31, -0.22], dtype=dtype, device=device)
    gammas = torch.tensor([-0.45, 0.18], dtype=dtype, device=device)
    psi_q = forward_qaoa(E, betas, gammas)
    e_q, _, _ = energy_of(psi_q, E, n)
    gw = gammas.reshape(L, 1).expand(L, 4).contiguous()
    # centering / degree-0 drop is a global phase vs QAOA's uncentered Cn,
    # and c1=-β matches exp(-i β X/2) because pc = Σ(I-Z)/2.
    c1 = -betas
    z = torch.zeros(L, dtype=dtype, device=device)
    phi_lin = phi_from_c(c1, z, z)
    psi_d = forward_motif(E, gw, phi_lin, pc, design)
    e_d, _, _ = energy_of(psi_d, E, n)
    dual_err = float((e_q - e_d).abs().max())
    if dual_err > 1e-6:
        raise RuntimeError(f"QAOA dual mismatch {dual_err}")

    # match Nov3rdAnsatz probabilities (complex64 reference) on the same angles
    sim = make_sim(n, L, device)
    tv_max = 0.0
    gw32 = gw.detach().to(dtype=torch.float32)
    phi32 = phi_lin.detach().to(dtype=torch.float32)
    for i in range(3):
        p_ref = bitstring_probs_from_ansatz(sim, E[i].to(dtype=torch.float32), gw32, phi32)
        psi = forward_motif(E[i : i + 1], gw, phi_lin, pc, design)
        p = (psi.real.square() + psi.imag.square()).detach().cpu().numpy().reshape(-1)
        tv = 0.5 * float(np.abs(p - p_ref).sum())
        tv_max = max(tv_max, tv)
    if tv_max > 2e-4:
        raise RuntimeError(f"ansatz TV {tv_max}")

    # one nonlinear-shell + split-gamma prob match
    gw_split = torch.tensor([[0.11, -0.07, 0.05, 0.02], [0.04, 0.08, -0.03, 0.01]], dtype=dtype)
    phi_nl = phi_from_c(
        torch.tensor([0.2, -0.1], dtype=dtype),
        torch.tensor([0.05, -0.02], dtype=dtype),
        torch.tensor([0.01, 0.0], dtype=dtype),
    )
    p_ref = bitstring_probs_from_ansatz(
        sim, E[0].to(dtype=torch.float32), gw_split.to(dtype=torch.float32), phi_nl.to(dtype=torch.float32)
    )
    psi = forward_motif(E[0:1], gw_split, phi_nl, pc, design)
    p = (psi.real.square() + psi.imag.square()).detach().cpu().numpy().reshape(-1)
    tv_nl = 0.5 * float(np.abs(p - p_ref).sum())
    if tv_nl > 2e-4:
        raise RuntimeError(f"nonlinear ansatz TV {tv_nl}")

    # E0 below |+> and below classical 0, GS not a bitstring
    e0, peak = ground_state(E_np[0], n)
    ep = float(E_np[0].mean() - LAM * n)
    if not (e0 < ep - 0.2 and peak < 0.5 and E_np[0].min() == 0.0):
        raise RuntimeError(f"GS sanity e0={e0} E+={ep} peak={peak}")

    report = {
        "plus_err": err_plus,
        "zero_motif_err": err_m,
        "walsh_recon_err": recon,
        "dual_energy_err": dual_err,
        "ansatz_tv_linear": tv_max,
        "ansatz_tv_walsh_shell": tv_nl,
        "armB_energy_minus_plus_min": float((e_b - e_ref).min()),
        "sample_E0": e0,
        "sample_Eplus": ep,
        "sample_peak": peak,
    }
    log(
        f"self-check OK  dual={dual_err:.2e} ansatzTV={tv_max:.2e}/{tv_nl:.2e} "
        f"recon={recon:.2e} E0={e0:.4f} E+={ep:.4f} peak={peak:.3f}"
    )
    return report


def precompute_spectra(bundles: dict) -> None:
    cache_path = OUT / "e0_cache.npz"
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        for k in z.files:
            cache[k] = z[k]
    for name, bun in bundles.items():
        n = bun["n"]
        E = bun["E_np"]
        key_e = f"{name}_e0"
        key_p = f"{name}_peak"
        if key_e in cache and cache[key_e].shape[0] == E.shape[0]:
            bun["e0"] = cache[key_e]
            bun["peak"] = cache[key_p]
            log(f"E0 cache hit {name} B={E.shape[0]}")
            continue
        e0s = np.empty(E.shape[0], dtype=np.float64)
        peaks = np.empty(E.shape[0], dtype=np.float64)
        t0 = time.time()
        for i in range(E.shape[0]):
            e0s[i], peaks[i] = ground_state(E[i], n)
        log(f"E0 {name} n={n} B={E.shape[0]} med={np.median(e0s):.4f} "
            f"peak_med={np.median(peaks):.3f} {time.time()-t0:.2f}s")
        bun["e0"] = e0s
        bun["peak"] = peaks
        cache[key_e] = e0s
        cache[key_p] = peaks
        np.savez(cache_path, **cache)


def make_bundles(device) -> dict:
    packs = {}
    seed_sets = {}
    for n in (6, 9, 12):
        E, seeds = load_E(n)
        packs[n] = E
        seed_sets[n] = seeds
    E6s, s6s = load_scale6()
    if set(seed_sets[6].tolist()) & set(s6s.tolist()):
        raise RuntimeError("n6 train pack overlaps scale1000")
    # slices
    bundles = {
        "n6_train": _bun(6, packs[6][:TRAIN_N], seed_sets[6][:TRAIN_N], device),
        "n6_val": _bun(6, packs[6][TRAIN_N : TRAIN_N + VAL_N], seed_sets[6][TRAIN_N : TRAIN_N + VAL_N], device),
        "n9_train": _bun(9, packs[9][:TRAIN_N], seed_sets[9][:TRAIN_N], device),
        "n9_val": _bun(9, packs[9][TRAIN_N : TRAIN_N + VAL_N], seed_sets[9][TRAIN_N : TRAIN_N + VAL_N], device),
        "n6_ext": _bun(6, E6s[:N6_EXT], s6s[:N6_EXT], device),
        "n12": _bun(12, packs[12][:N12_PRIMARY], seed_sets[12][:N12_PRIMARY], device),
        "n12b": _bun(12, packs[12][N12_PRIMARY : N12_PRIMARY + N12_CONFIRM],
                    seed_sets[12][N12_PRIMARY : N12_PRIMARY + N12_CONFIRM], device),
    }
    # satisfiable?
    for name, b in bundles.items():
        emin = b["E_np"].min(axis=1)
        b["frac_sat"] = float(np.mean(emin == 0.0))
        b["med_Eplus"] = float(np.median(b["E_np"].mean(axis=1) - LAM * b["n"]))
    return bundles


def _bun(n, E_np, seeds, device):
    dtype = torch.float64
    return {
        "n": n,
        "E_np": np.asarray(E_np, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "E": torch.as_tensor(E_np, dtype=dtype, device=device),
        "pc": popcount_tensor(n, device, dtype),
        "design": shell_design(n, device, dtype),
    }


@torch.no_grad()
def eval_motif(bun, gamma_w, phi) -> dict:
    psi = forward_motif(bun["E"], gamma_w, phi, bun["pc"], bun["design"])
    en, ed, ex = energy_of(psi, bun["E"], bun["n"])
    norm = (psi.real.square() + psi.imag.square()).sum(-1)
    if float((norm - 1).abs().max()) > 1e-5:
        raise RuntimeError(f"norm drift {float((norm-1).abs().max())}")
    if float((en - bun["e0_t"]).min()) < -1e-4:
        raise RuntimeError("variational energy below E0")
    out = summarize_gaps(
        en.cpu().numpy(), bun["e0"], ed.cpu().numpy(), ex.cpu().numpy()
    )
    return out


@torch.no_grad()
def eval_qaoa(bun, betas, gammas) -> dict:
    psi = forward_qaoa(bun["E"], betas, gammas)
    en, ed, ex = energy_of(psi, bun["E"], bun["n"])
    if float((en - bun["e0_t"]).min()) < -1e-4:
        raise RuntimeError("QAOA energy below E0")
    return summarize_gaps(en.cpu().numpy(), bun["e0"], ed.cpu().numpy(), ex.cpu().numpy())


def val_key_from_parts(s6: dict, s9: dict) -> float:
    return 0.5 * (s6["med_gap"] + s9["med_gap"])


def loss_motif(b6, b9, gamma_w, phi) -> torch.Tensor:
    e6, _, _ = energy_of(forward_motif(b6["E"], gamma_w, phi, b6["pc"], b6["design"]), b6["E"], 6)
    e9, _, _ = energy_of(forward_motif(b9["E"], gamma_w, phi, b9["pc"], b9["design"]), b9["E"], 9)
    return 0.5 * (e6.mean() + e9.mean())


def loss_qaoa(b6, b9, betas, gammas) -> torch.Tensor:
    e6, _, _ = energy_of(forward_qaoa(b6["E"], betas, gammas), b6["E"], 6)
    e9, _, _ = energy_of(forward_qaoa(b9["E"], betas, gammas), b9["E"], 9)
    return 0.5 * (e6.mean() + e9.mean())


def optimize(params, loss_fn, tag: str, adam_steps: int, lbfgs_rounds: int) -> list[float]:
    hist = []
    if adam_steps > 0 and params:
        opt = torch.optim.Adam(params, lr=ADAM_LR)
        stall = 0
        prev = None
        t0 = time.time()
        for step in range(adam_steps):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{tag} nonfinite Adam loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            val = float(loss.detach())
            hist.append(val)
            if prev is not None and abs(prev - val) < 1e-6:
                stall += 1
            else:
                stall = 0
            prev = val
            if step % 40 == 0 or step == adam_steps - 1:
                log(f"  {tag} Adam {step:3d} loss={val:.6f}")
            if stall >= 25 and step > 40:
                log(f"  {tag} Adam stall at {step} ({time.time()-t0:.1f}s)")
                break
    if lbfgs_rounds > 0 and params:
        opt = torch.optim.LBFGS(
            params,
            lr=0.5,
            max_iter=LBFGS_MAX_ITER,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{tag} nonfinite LBFGS loss")
            loss.backward()
            return loss

        for r in range(lbfgs_rounds):
            try:
                loss = opt.step(closure)
            except Exception as exc:  # line search can fail on a flat arm
                log(f"  {tag} LBFGS stop r={r}: {type(exc).__name__}: {exc}")
                break
            val = float(loss.detach())
            hist.append(val)
            log(f"  {tag} LBFGS {r:2d} loss={val:.6f}")
            if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < 1e-7:
                break
    return hist


def pack_params(gamma_w, c1, c2, c3) -> dict:
    return {
        "gamma_w": gamma_w.detach().cpu().tolist(),
        "c1": c1.detach().cpu().tolist(),
        "c2": c2.detach().cpu().tolist(),
        "c3": c3.detach().cpu().tolist(),
    }


def gamma_spread(gamma_w: torch.Tensor) -> float:
    gw = gamma_w.detach()
    return float((gw - gw.mean(dim=1, keepdim=True)).abs().mean())


def init_subspace(L: int, seed: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.zeros(L, dtype=dtype, device=device)
    c1 = torch.zeros(L, dtype=dtype, device=device)
    if seed == 0:
        g.fill_(-0.01)
        c1.fill_(-0.01)  # c1 = -β with β=0.01
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(10_000 + 17 * seed + L)
        g = torch.randn(L, generator=gen, dtype=dtype).to(device) * 0.35
        c1 = torch.randn(L, generator=gen, dtype=dtype).to(device) * 0.35
    return torch.nn.Parameter(g), torch.nn.Parameter(c1)


def train_arm_A(L, seed, b6, b9, device, dtype, warm=None) -> dict:
    """Stage 1: QAOA-dual subspace. Stage 2: free Walsh γ and c2, c3.

    Both checkpoints live in the full family. Caller picks by validation.
    """
    log(f"A L={L} seed={seed} {'warm' if warm else 'scratch'}")
    if warm is None:
        g, c1 = init_subspace(L, seed, device, dtype)
    else:
        g = torch.nn.Parameter(warm["gamma"].detach().to(device=device, dtype=dtype).reshape(L).clone())
        c1 = torch.nn.Parameter((-warm["beta"]).detach().to(device=device, dtype=dtype).reshape(L).clone())
    z = torch.zeros(L, dtype=dtype, device=device)

    def loss1():
        gw = g.unsqueeze(-1).expand(L, 4)
        phi = phi_from_c(c1, z, z)
        return loss_motif(b6, b9, gw, phi)

    optimize([g, c1], loss1, f"A{seed}/s1", ADAM_STEPS, LBFGS_ROUNDS)
    with torch.no_grad():
        gw1 = g.detach().unsqueeze(-1).repeat(1, 4).clone()
        c1_1 = c1.detach().clone()
    stage1 = {
        "gamma_w": gw1,
        "c1": c1_1,
        "c2": torch.zeros(L, dtype=dtype),
        "c3": torch.zeros(L, dtype=dtype),
        "kind": "subspace",
    }

    gw = torch.nn.Parameter(gw1.clone().to(device))
    c1b = torch.nn.Parameter(c1_1.clone().to(device))
    c2 = torch.nn.Parameter(torch.zeros(L, dtype=dtype, device=device))
    c3 = torch.nn.Parameter(torch.zeros(L, dtype=dtype, device=device))

    def loss2():
        phi = phi_from_c(c1b, c2, c3)
        return loss_motif(b6, b9, gw, phi)

    optimize([gw, c1b, c2, c3], loss2, f"A{seed}/s2", ADAM_STEPS, LBFGS_ROUNDS)
    stage2 = {
        "gamma_w": gw.detach().cpu(),
        "c1": c1b.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "kind": "full",
    }
    stage1 = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in stage1.items()}
    return {"stage1": stage1, "stage2": stage2}


def train_arm_B(L, seed, b6, b9, device, dtype) -> dict:
    log(f"B L={L} seed={seed}")
    if seed == 0:
        gw = torch.nn.Parameter(torch.full((L, 4), -0.01, dtype=dtype, device=device))
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20_000 + 19 * seed + L)
        gw = torch.nn.Parameter(torch.randn(L, 4, generator=gen, dtype=dtype).to(device) * 0.35)
    zc = torch.zeros(L, dtype=dtype, device=device)
    phi = phi_from_c(zc, zc, zc)

    def loss():
        return loss_motif(b6, b9, gw, phi)

    optimize([gw], loss, f"B{seed}", ADAM_STEPS // 2, max(4, LBFGS_ROUNDS // 2))
    return {
        "gamma_w": gw.detach().cpu(),
        "c1": zc.detach().cpu(),
        "c2": zc.detach().cpu(),
        "c3": zc.detach().cpu(),
        "kind": "identity_snap",
    }


def train_arm_C(L, seed, b6, b9, device, dtype) -> dict:
    log(f"C L={L} seed={seed}")
    gw = torch.zeros(L, 4, dtype=dtype, device=device)
    if seed == 0:
        c1 = torch.nn.Parameter(torch.full((L,), -0.01, dtype=dtype, device=device))
        c2 = torch.nn.Parameter(torch.zeros(L, dtype=dtype, device=device))
        c3 = torch.nn.Parameter(torch.zeros(L, dtype=dtype, device=device))
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(30_000 + 23 * seed + L)
        c1 = torch.nn.Parameter(torch.randn(L, generator=gen, dtype=dtype).to(device) * 0.4)
        c2 = torch.nn.Parameter(torch.randn(L, generator=gen, dtype=dtype).to(device) * 0.15)
        c3 = torch.nn.Parameter(torch.randn(L, generator=gen, dtype=dtype).to(device) * 0.05)

    def loss():
        phi = phi_from_c(c1, c2, c3)
        return loss_motif(b6, b9, gw, phi)

    optimize([c1, c2, c3], loss, f"C{seed}", 30, 2)
    z = torch.zeros(L, dtype=dtype)
    return {
        "gamma_w": gw.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "kind": "snap_only",
    }


def train_arm_D(L, seed, b6, b9, device, dtype) -> dict:
    log(f"D L={L} seed={seed}")
    q = L  # complete Fourier basis at this depth
    if seed == 0:
        ug = torch.nn.Parameter(torch.as_tensor(fit_constant_fourier(-0.01, L, q), dtype=dtype, device=device))
        ub = torch.nn.Parameter(torch.as_tensor(fit_constant_fourier(0.01, L, q), dtype=dtype, device=device))
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(40_000 + 29 * seed + L)
        ug = torch.nn.Parameter(torch.randn(q, generator=gen, dtype=dtype).to(device) * 0.35)
        ub = torch.nn.Parameter(torch.randn(q, generator=gen, dtype=dtype).to(device) * 0.35)

    def loss():
        return loss_qaoa(b6, b9, fourier_schedule(ub, L), fourier_schedule(ug, L))

    # Match A's two-stage step budget.
    optimize([ug, ub], loss, f"D{seed}", ADAM_STEPS * 2, LBFGS_ROUNDS * 2)
    betas = fourier_schedule(ub, L).detach()
    gammas = fourier_schedule(ug, L).detach()
    return {
        "u_gamma": ug.detach().cpu(),
        "u_beta": ub.detach().cpu(),
        "beta": betas.cpu(),
        "gamma": gammas.cpu(),
        "kind": "fourier_qaoa",
    }


def score_motif_ckpt(ckpt, bundles, device, dtype) -> dict:
    gw = ckpt["gamma_w"].to(device=device, dtype=dtype)
    phi = phi_from_c(
        ckpt["c1"].to(device=device, dtype=dtype),
        ckpt["c2"].to(device=device, dtype=dtype),
        ckpt["c3"].to(device=device, dtype=dtype),
    )
    scored = {}
    for name in ("n6_train", "n6_val", "n9_train", "n9_val", "n6_ext", "n12", "n12b"):
        scored[name] = eval_motif(bundles[name], gw, phi)
    scored["val_med"] = val_key_from_parts(scored["n6_val"], scored["n9_val"])
    scored["spread"] = gamma_spread(gw)
    scored["c2_abs"] = float(ckpt["c2"].abs().mean())
    scored["c3_abs"] = float(ckpt["c3"].abs().mean())
    return scored


def score_qaoa_ckpt(ckpt, bundles, device, dtype) -> dict:
    betas = ckpt["beta"].to(device=device, dtype=dtype)
    gammas = ckpt["gamma"].to(device=device, dtype=dtype)
    scored = {}
    for name in ("n6_train", "n6_val", "n9_train", "n9_val", "n6_ext", "n12", "n12b"):
        scored[name] = eval_qaoa(bundles[name], betas, gammas)
    scored["val_med"] = val_key_from_parts(scored["n6_val"], scored["n9_val"])
    return scored


def score_reference(bundles) -> dict:
    """Analytic |+> and classical bitstring (energy 0 on satisfiable instances)."""
    out = {}
    for name, bun in bundles.items():
        e_plus = bun["E_np"].mean(axis=1) - LAM * bun["n"]
        e_bit = bun["E_np"].min(axis=1)  # <X>=0
        out[name] = {
            "plus": summarize_gaps(e_plus, bun["e0"], bun["E_np"].mean(axis=1), np.full(len(e_plus), bun["n"])),
            "bitstring": summarize_gaps(e_bit, bun["e0"], e_bit, np.zeros(len(e_bit))),
            "med_peak": float(np.median(bun["peak"])),
            "frac_sat": bun["frac_sat"],
            "med_E0": float(np.median(bun["e0"])),
        }
    return out


def judge(rB: float, rC: float, rD: float) -> str:
    if rB >= KILL_RATIO_B:
        return "KILL"
    if rB <= PASS_RATIO and rC <= PASS_RATIO and rD <= PASS_RATIO:
        return "PASS"
    return "needs_edit"


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def write_results(payload: dict) -> None:
    lines = []
    lines.append("# NEW_SAFE cost-only ablation — U_c^X / SNAP mixer")
    lines.append("")
    lines.append(f"_UTC {payload['finished_utc']}_")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "On NEW_SAFE, does a diagonal cost pulse by itself reach the ground state, "
        "or does the Hamming-shell SNAP mixer have to sit in the circuit with the cost pulse?"
    )
    lines.append("")
    lines.append("## Hamiltonian and circuits")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"H = \mathrm{diag}(E_{\mathrm{unsat}}) - 0.75 \sum_{i=1}^{n} X_i")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        r"Motif (arm A): \(U_c^X = W \exp(-i \sum_{w=1}^{4} \gamma_{\ell,w} E_w) W\) "
        r"with \(W = H^{\otimes 3}\) per cavity, \(E_w\) the degree-\(w\) Walsh piece of `cost_Cn`, "
        r"then a shared Hamming-shell SNAP \(\varphi_k = c_1 k + c_2 k^2 + c_3 k^3\), \(k=0..3\), "
        r"and a final \(W\) into the Fock/bitstring readout. "
        r"Equal \(\gamma_w\) and \(c_2=c_3=0\), \(c_1=-\beta\) reproduces Fourier/free QAOA "
        r"with the repo mixer \(\exp(-i\beta X/2)\) (checked, energy error \(<10^{-6}\))."
    )
    lines.append("")
    lines.append("| arm | what is trained | what is frozen |")
    lines.append("|-----|-----------------|----------------|")
    lines.append(r"| A | Walsh \(\gamma_{\ell,w}\) and shell \(c_1,c_2,c_3\) | — |")
    lines.append(r"| B | Walsh \(\gamma\) only | shell phases = 0 (identity SNAP) |")
    lines.append(r"| C | shell \(c_1,c_2,c_3\) | \(\gamma = 0\) |")
    lines.append(r"| D | Zhou Fourier modes of \((\beta,\gamma)\), \(q=L\) | standard QAOA, no SNAP |")
    lines.append("")
    lines.append("## Why a diagonal pulse cannot win by itself")
    lines.append("")
    lines.append(
        r"With identity SNAP the layer is \(U_c^X(\gamma_\ell)=W e^{-i\phi_\ell} W\). "
        "These factors commute and collapse, for any number of layers and any diagonal waveform "
        r"\(\phi(z)\) (scalar \(\gamma\), Walsh grading, or a full GRAPE schedule of \(H_c\)), to"
    )
    lines.append("")
    lines.append(r"\[")
    lines.append(r"|\psi\rangle = e^{-i\phi(Z)}\,|+\rangle^{\otimes n}.")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        r"Z-populations stay uniform, so \(\langle E_{\mathrm{unsat}}\rangle=\mathrm{mean}(E)\). "
        r"Each \(\langle X_i\rangle=\mathrm{mean}_z\cos(\phi(z)-\phi(z\oplus e_i))\le 1\), "
        r"with equality on every qubit only for constant \(\phi\). "
        r"Minimizing \(\langle H\rangle=\langle E\rangle-0.75\sum\langle X_i\rangle\) inside this family "
        r"lands on \(|+\rangle\), energy \(\mathrm{mean}(E)-0.75\,n\). "
        r"Arm C is the same state: \(\gamma=0\) returns the register to \(|0\rangle\) before each "
        r"Z-diagonal SNAP, the shell multiplies only a global phase, and the final \(W\) reads out \(|+\rangle\)."
    )
    lines.append("")
    lines.append(
        "The training runs are a numerical check of that bound. "
        "The reported arm-B and arm-C gaps are the optimized circuits; "
        r"they sit on the \(|+\rangle\) gap."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    p = payload["protocol"]
    lines.append(
        f"- L ∈ {{2, 4}}; seeds {{0,1,2,3,4}} plus one A warm-start from the val-best D angles."
    )
    lines.append(
        f"- Train objective: mean ⟨H⟩ on n6[{0}:{p['train_n']}] and n9[{0}:{p['train_n']}] "
        f"(equal average). Optimizer: Adam lr={p['adam_lr']} then L-BFGS, same step budget for A and D."
    )
    lines.append(
        f"- Model selection uses the median gap on a disjoint val slice "
        f"(n6/n9 indices [{p['train_n']}:{p['train_n']+p['val_n']}]). n12 never enters the loss or the selection."
    )
    lines.append(
        f"- Frozen eval: `train_packs/train_n12.npz` indices [0:{p['n12']}] primary and "
        f"[{p['n12']}:{p['n12']+p['n12b']}] confirm "
        "(scale1000 n12 is not in this tree; these instances never enter the loss). "
        f"Extra: scale1000 n6[0:{p['n6_ext']}], seed-disjoint from the n6 train pack."
    )
    lines.append(
        r"- Arm D uses \(q=L\) Zhou modes, which span every \((\beta,\gamma)\) schedule, so D is full-angle QAOA. "
        r"The QAOA-dual point inside A (equal \(\gamma_w\), \(c_2=c_3=0\)) reproduces D."
    )
    lines.append(
        "- Metric: Medε = median over instances of (⟨H⟩ − E0). "
        f"Ratios r_B = Medε_A/Medε_B, and the same for C and D. "
        f"**PASS** if r_B, r_C, and r_D are all ≤ {PASS_RATIO} on primary n12. "
        f"**KILL** if r_B ≥ {KILL_RATIO_B}. Otherwise **needs_edit**."
    )
    lines.append("- E0: dense `eigh` at n≤9, Lanczos `eigsh` at n=12 (residual checked).")
    lines.append("- n21 ladder, `settings.py` locks, and `qaoa_angles/` were not touched.")
    lines.append("")
    lines.append("## Ground state")
    lines.append("")
    lines.append("| slice | med E0 | med E(|+⟩) | med ε(|+⟩) | med ε(best bitstring) | med GS bitstring peak | frac sat |")
    lines.append("|-------|--------|------------|------------|------------------------|-----------------------|----------|")
    ref = payload["reference"]
    for name in ("n6_val", "n9_val", "n6_ext", "n12", "n12b"):
        r = ref[name]
        lines.append(
            f"| {name} | {_fmt(r['med_E0'])} | {_fmt(r['plus']['med_energy'])} | "
            f"{_fmt(r['plus']['med_gap'])} | {_fmt(r['bitstring']['med_gap'])} | "
            f"{_fmt(r['med_peak'])} | {_fmt(r['frac_sat'])} |"
        )
    lines.append("")
    lines.append(
        "Every eval instance is satisfiable, so the best computational-basis state has energy 0. "
        "The ground state is lower, and its largest bitstring weight is well below 1/2, "
        "so the ground state is a superposition."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "Headline rows are the val-selected checkpoint of each arm. "
        "Medε is the median instance gap. Lower is closer to E0."
    )
    lines.append("")
    for L in payload["Ls"]:
        block = payload["by_L"][str(L)]
        lines.append(f"### L={L}")
        lines.append("")
        lines.append("| arm | seed | val Medε | n6 ext Medε | n12 Medε | n12b Medε | n12 ⟨D⟩ med | n12 ⟨ΣX⟩ med |")
        lines.append("|-----|------|----------|-------------|----------|-----------|-------------|--------------|")
        for arm in ("A", "B", "C", "D"):
            row = block["selected"][arm]
            lines.append(
                f"| {arm} | {row['seed']} | {_fmt(row['val_med'])} | {_fmt(row['n6_ext']['med_gap'])} | "
                f"**{_fmt(row['n12']['med_gap'])}** | {_fmt(row['n12b']['med_gap'])} | "
                f"{_fmt(row['n12']['med_D'])} | {_fmt(row['n12']['med_X'])} |"
            )
        lines.append(
            f"| |+> | analytic | {_fmt(block['plus_val'])} | {_fmt(block['plus_n6ext'])} | "
            f"{_fmt(block['plus_n12'])} | {_fmt(block['plus_n12b'])} |  |  |"
        )
        lines.append("")
        rat = block["ratios_n12"]
        lines.append(
            f"Primary n12 ratios: r_B={_fmt(rat['rB'])}, r_C={_fmt(rat['rC'])}, r_D={_fmt(rat['rD'])}. "
            f"Confirm n12b: r_B={_fmt(block['ratios_n12b']['rB'])}, r_C={_fmt(block['ratios_n12b']['rC'])}, "
            f"r_D={_fmt(block['ratios_n12b']['rD'])}."
        )
        lines.append("")
        lines.append(
            f"Median-over-seeds n12 Medε: "
            + ", ".join(f"{a}={_fmt(block['seed_median_n12'][a])}" for a in ("A", "B", "C", "D"))
            + f". Ratios of those medians: r_B={_fmt(block['seed_median_ratios']['rB'])}, "
            f"r_C={_fmt(block['seed_median_ratios']['rC'])}, r_D={_fmt(block['seed_median_ratios']['rD'])}."
        )
        lines.append("")
        lines.append(f"**L={L} verdict: {block['verdict']}.** {block['verdict_note']}")
        lines.append("")
        # per seed
        lines.append("<details><summary>all seeds, n12 Medε</summary>")
        lines.append("")
        lines.append(r"| arm | seed | kind | val Medε | n12 Medε | γ spread | mean |c2| | mean |c3| |")
        lines.append("|-----|------|------|----------|----------|----------|-------------|-------------|")
        for arm in ("A", "B", "C", "D"):
            for s in block["seeds"][arm]:
                lines.append(
                    f"| {arm} | {s['seed']} | {s.get('kind','')} | {_fmt(s['val_med'])} | "
                    f"{_fmt(s['n12']['med_gap'])} | {_fmt(s.get('spread'))} | "
                    f"{_fmt(s.get('c2_abs'))} | {_fmt(s.get('c3_abs'))} |"
                )
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{payload['overall']}**")
    lines.append("")
    lines.append(payload["overall_note"])
    lines.append("")
    lines.append("## Repro")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 Nov_3rd/scripts/cost_only_ablation_NEW_SAFE.py")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Self-check (dual energy error {payload['self_check']['dual_energy_err']:.2e}, "
        f"ansatz TV {payload['self_check']['ansatz_tv_linear']:.2e} / "
        f"{payload['self_check']['ansatz_tv_walsh_shell']:.2e}) runs at startup. "
        f"Wall {payload['wall_s']:.0f}s. Machine-readable: `results.json`."
    )
    lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "RESULTS.md").write_text(text)
    log(f"wrote {OUT / 'RESULTS.md'}")


def tensor_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): tensor_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_tree(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run(args):
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device("cpu")
    dtype = torch.float64
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    check = self_check(device)
    if args.self_check_only:
        (OUT / "self_check.json").write_text(json.dumps(check, indent=2) + "\n")
        return
    global ADAM_STEPS, LBFGS_ROUNDS, TRAIN_N, VAL_N, N12_PRIMARY, N12_CONFIRM
    if args.smoke:
        ADAM_STEPS = 40
        LBFGS_ROUNDS = 3
        # smoke still uses the module-level slice sizes already captured in make_bundles
        # so rebuild with smaller counts via args
    seeds = (0, 1) if args.smoke else SEEDS
    Ls = (2,) if args.smoke else (2, 4)
    if args.smoke:
        # temporary smaller bundles
        global TRAIN_N, VAL_N, N12_PRIMARY, N12_CONFIRM, N6_EXT
        TRAIN_N, VAL_N, N12_PRIMARY, N12_CONFIRM, N6_EXT = 16, 12, 16, 16, 16
    bundles = make_bundles(device)
    precompute_spectra(bundles)
    for bun in bundles.values():
        bun["e0_t"] = torch.as_tensor(bun["e0"], dtype=dtype, device=device)

    ref = score_reference(bundles)
    by_L = {}
    for L in Ls:
        log(f"===== L={L} =====")
        store = {"A": [], "B": [], "C": [], "D": []}
        # D first so A can warm-start from the best D
        d_ckpts = []
        for seed in seeds:
            ckpt = train_arm_D(L, seed, bundles["n6_train"], bundles["n9_train"], device, dtype)
            scored = score_qaoa_ckpt(ckpt, bundles, device, dtype)
            row = {
                "seed": seed,
                "kind": "fourier",
                "val_med": scored["val_med"],
                "n12": scored["n12"],
                "n12b": scored["n12b"],
                "n6_ext": scored["n6_ext"],
                "n6_val": scored["n6_val"],
                "n9_val": scored["n9_val"],
                "spread": None,
                "c2_abs": None,
                "c3_abs": None,
                "params": {
                    "u_beta": ckpt["u_beta"].tolist(),
                    "u_gamma": ckpt["u_gamma"].tolist(),
                    "beta": ckpt["beta"].tolist(),
                    "gamma": ckpt["gamma"].tolist(),
                },
            }
            store["D"].append(row)
            d_ckpts.append(ckpt)
            log(f"  D seed {seed} val={scored['val_med']:.4f} n12={scored['n12']['med_gap']:.4f}")

        best_D = min(store["D"], key=lambda r: r["val_med"])
        warm = d_ckpts[store["D"].index(best_D)]
        # The QAOA dual of the val-best D angles is a legal point of family A
        # (equal Walsh weights, linear shell). Score it before any further steps.
        dual = {
            "gamma_w": warm["gamma"].reshape(L, 1).repeat(1, 4).contiguous(),
            "c1": (-warm["beta"]).contiguous(),
            "c2": torch.zeros(L, dtype=dtype),
            "c3": torch.zeros(L, dtype=dtype),
            "kind": "dual-of-D",
        }
        scored = score_motif_ckpt(dual, bundles, device, dtype)
        dual_row = _motif_row("dual-of-D", "dual-of-D", dual, scored)
        store["A"].append(dual_row)
        log(
            f"  A dual-of-D val={scored['val_med']:.4f} n12={scored['n12']['med_gap']:.4f} "
            f"(D val={best_D['val_med']:.4f} n12={best_D['n12']['med_gap']:.4f})"
        )

        a_jobs = [(seed, None) for seed in seeds] + [(-1, warm)]
        for seed, warm_ckpt in a_jobs:
            trained = train_arm_A(
                L, seed, bundles["n6_train"], bundles["n9_train"], device, dtype, warm=warm_ckpt
            )
            for stage_name, ckpt in (("stage1", trained["stage1"]), ("stage2", trained["stage2"])):
                scored = score_motif_ckpt(ckpt, bundles, device, dtype)
                kind = "subspace" if stage_name == "stage1" else "full"
                if seed == -1:
                    kind = "warm-" + kind
                row = _motif_row(seed if seed >= 0 else f"warm-{stage_name}", kind, ckpt, scored)
                # keep stage identity in the seed label; selection uses val
                row["seed"] = f"{seed}:{stage_name}" if seed >= 0 else f"warm:{stage_name}"
                store["A"].append(row)
                log(
                    f"  A {row['seed']} {kind} val={scored['val_med']:.4f} "
                    f"n12={scored['n12']['med_gap']:.4f} spread={scored['spread']:.4f}"
                )

        for seed in seeds:
            ckpt = train_arm_B(L, seed, bundles["n6_train"], bundles["n9_train"], device, dtype)
            scored = score_motif_ckpt(ckpt, bundles, device, dtype)
            store["B"].append(_motif_row(seed, "identity_snap", ckpt, scored))
            log(f"  B seed {seed} val={scored['val_med']:.4f} n12={scored['n12']['med_gap']:.4f}")
            ckpt = train_arm_C(L, seed, bundles["n6_train"], bundles["n9_train"], device, dtype)
            scored = score_motif_ckpt(ckpt, bundles, device, dtype)
            store["C"].append(_motif_row(seed, "snap_only", ckpt, scored))
            log(f"  C seed {seed} val={scored['val_med']:.4f} n12={scored['n12']['med_gap']:.4f}")

        selected = {}
        for arm, rows in store.items():
            selected[arm] = min(rows, key=lambda r: r["val_med"])
        ratios = _ratios(selected, "n12")
        ratios_b = _ratios(selected, "n12b")
        seed_med = {}
        for arm, rows in store.items():
            # one number per random seed: for A, take the better stage on val, then median over seeds
            by_seed = {}
            for r in rows:
                key = str(r["seed"]).split(":")[0]
                if key not in by_seed or r["val_med"] < by_seed[key]["val_med"]:
                    by_seed[key] = r
            vals = [r["n12"]["med_gap"] for r in by_seed.values()]
            seed_med[arm] = float(np.median(vals))
        seed_ratios = {
            "rB": seed_med["A"] / seed_med["B"],
            "rC": seed_med["A"] / seed_med["C"],
            "rD": seed_med["A"] / seed_med["D"],
        }
        verdict = judge(ratios["rB"], ratios["rC"], ratios["rD"])
        note_bits = []
        if verdict == "PASS" and (
            seed_ratios["rD"] > 0.95 or seed_ratios["rB"] > 0.90 or ratios_b["rD"] > PASS_RATIO
            or ratios_b["rB"] > PASS_RATIO or ratios_b["rC"] > PASS_RATIO
        ):
            verdict = "needs_edit"
            note_bits.append(
                "Primary slice cleared the margin, but the confirm slice or the median seed did not."
            )
        if verdict == "PASS":
            note_bits.append(
                "Full Walsh SNAP + U_c^X is closer to E0 than the diagonal pulse, the SNAP-only circuit, and Fourier QAOA."
            )
        elif verdict == "KILL":
            note_bits.append(
                "Identity-SNAP cost pulse matches the full motif, so a diagonal GRAPE of H_c already captures the gain."
            )
        else:
            note_bits.append(
                "Margins are soft against the pre-registered 0.85 ratio, or the confirm slice disagrees."
            )
        # describe whether selected A left the QAOA subspace
        a_sel = selected["A"]
        note_bits.append(
            f"Selected A is `{a_sel['seed']}` ({a_sel['kind']}), "
            f"γ spread {a_sel['spread']:.4f}, mean |c2|={a_sel['c2_abs']:.4f}, mean |c3|={a_sel['c3_abs']:.4f}."
        )
        by_L[str(L)] = {
            "selected": {a: _public_row(selected[a]) for a in selected},
            "seeds": {a: [_public_row(r) for r in store[a]] for a in store},
            "ratios_n12": ratios,
            "ratios_n12b": ratios_b,
            "seed_median_n12": seed_med,
            "seed_median_ratios": seed_ratios,
            "verdict": verdict,
            "verdict_note": " ".join(note_bits),
            "plus_val": val_key_from_parts(ref["n6_val"]["plus"], ref["n9_val"]["plus"]),
            "plus_n12": ref["n12"]["plus"]["med_gap"],
            "plus_n12b": ref["n12b"]["plus"]["med_gap"],
            "plus_n6ext": ref["n6_ext"]["plus"]["med_gap"],
        }
        log(f"L={L} verdict {verdict} ratios {ratios}")

    # overall: PASS only if every L passes; KILL if any L kills and none pass with a clean story
    verdicts = [by_L[str(L)]["verdict"] for L in Ls]
    ratio_bits = []
    for L in Ls:
        r = by_L[str(L)]["ratios_n12"]
        ratio_bits.append(
            f"L={L} n12 Medε A/B/C/D = {r['medA']:.4f}/{r['medB']:.4f}/{r['medC']:.4f}/{r['medD']:.4f} "
            f"(r_B={r['rB']:.3f}, r_C={r['rC']:.3f}, r_D={r['rD']:.3f}, verdict {by_L[str(L)]['verdict']})"
        )
    ratio_line = " ".join(ratio_bits)
    if all(v == "PASS" for v in verdicts):
        overall = "PASS"
        overall_note = (
            ratio_line + ". "
            "Val-selected Walsh SNAP + U_c^X cuts the frozen-n12 median gap to E0 by at least "
            "the pre-registered 15% versus the identity-SNAP cost pulse, versus SNAP with the cost "
            "pulse removed, and versus same-depth Fourier QAOA, at both depths. "
            "A GRAPE schedule of diagonal H_c is the arm-B family. Its best state is |+>, "
            "and that gap stays well above the interleaved SNAP mixer. "
            "The mixer is required together with the cost pulse."
        )
    elif any(v == "KILL" for v in verdicts):
        overall = "KILL"
        overall_note = (
            ratio_line + ". "
            "On at least one depth the identity-SNAP cost pulse reaches essentially the same "
            "median gap as the full motif. That kills the claim that a diagonal pulse is not the solution. "
            "The |+> bound still caps every diagonal pulse; a KILL means the full motif "
            "failed to move materially below that cap."
        )
    else:
        overall = "needs_edit"
        overall_note = (
            ratio_line + ". "
            "Identity SNAP is capped at |+> (arm B), and SNAP with gamma frozen at 0 is the same state (arm C). "
            "Where the full motif beats that cap, the cost pulse alone is not the ground state. "
            "PASS also requires a 15% median-gap win over Fourier QAOA on frozen n12 at both L=2 and L=4, "
            "including the confirm slice and a stable seed median. That bar is not met cleanly."
        )
    payload = {
        "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "wall_s": time.time() - t_all,
        "hamiltonian": "diag(unsat) - 0.75 * sum_i X_i",
        "self_check": check,
        "protocol": {
            "train_n": TRAIN_N,
            "val_n": VAL_N,
            "n12": N12_PRIMARY,
            "n12b": N12_CONFIRM,
            "n6_ext": N6_EXT,
            "seeds": list(seeds),
            "Ls": list(Ls),
            "adam_steps_per_stage": ADAM_STEPS,
            "lbfgs_rounds_per_stage": LBFGS_ROUNDS,
            "adam_lr": ADAM_LR,
            "pass_ratio": PASS_RATIO,
            "kill_ratio_B": KILL_RATIO_B,
            "smoke": bool(args.smoke),
        },
        "reference": ref,
        "Ls": list(Ls),
        "by_L": by_L,
        "overall": overall,
        "overall_note": overall_note,
    }
    # reference contains numpy via summarize — tensor_tree handles it
    (OUT / "results.json").write_text(json.dumps(tensor_tree(payload), indent=2) + "\n")
    write_results(tensor_tree(payload))
    log(f"OVERALL {overall} in {time.time()-t_all:.0f}s")


def _motif_row(seed, kind, ckpt, scored) -> dict:
    return {
        "seed": seed,
        "kind": kind,
        "val_med": scored["val_med"],
        "n12": scored["n12"],
        "n12b": scored["n12b"],
        "n6_ext": scored["n6_ext"],
        "n6_val": scored["n6_val"],
        "n9_val": scored["n9_val"],
        "spread": scored["spread"],
        "c2_abs": scored["c2_abs"],
        "c3_abs": scored["c3_abs"],
        "params": pack_params(ckpt["gamma_w"], ckpt["c1"], ckpt["c2"], ckpt["c3"]),
    }


def _public_row(row: dict) -> dict:
    keep = (
        "seed", "kind", "val_med", "n12", "n12b", "n6_ext", "n6_val", "n9_val",
        "spread", "c2_abs", "c3_abs", "params",
    )
    return {k: row[k] for k in keep if k in row}


def _ratios(selected, slice_name: str) -> dict:
    a = selected["A"][slice_name]["med_gap"]
    return {
        "rB": a / selected["B"][slice_name]["med_gap"],
        "rC": a / selected["C"][slice_name]["med_gap"],
        "rD": a / selected["D"][slice_name]["med_gap"],
        "medA": a,
        "medB": selected["B"][slice_name]["med_gap"],
        "medC": selected["C"][slice_name]["med_gap"],
        "medD": selected["D"][slice_name]["med_gap"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
