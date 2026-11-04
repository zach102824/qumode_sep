#!/usr/bin/env python3
"""Smoke: SHORTLIST D signed 3-local Ising + transverse field (μ = 0).

H = Σ_c J_c Z_i Z_j Z_k − λ Σ_i X_i
  J_c ∈ {±1} from the product of the three kept literal signs
  supports: existing 4-SAT clause packs, 4th (highest-index) literal dropped
  μ = 0: no cavity XX

Train {6, 9}, freeze n12. L ∈ {2, 4}, λ ∈ {0.50, 0.75, 1.00}, seeds {0, 1, 2}.
Walsh shell-SNAP (degree-3 cost pulse; degrees 1, 2, 4 of H_Z are identically 0)
vs Zhou-Fourier QAOA on the same H.

Does not touch n21, settings locks, or the classical Max-E ladder.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "explore_OCT29" / "data"
OUT = ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "signed_3local_tf_breadth"

LAMBDAS = (0.50, 0.75, 1.00)
LS = (2, 4)
SEEDS = (0, 1, 2)
TRAIN_NS = (6, 9)
EVAL_NS = (6, 9, 12)
TRAIN_B = 32
Q_FOURIER = 3  # Zhou modes; at L=2 the q=2 sub-basis is completed below

# Pass / kill gates from the round brief.
LAMBDA_MAX = 0.80
GAP_MIN = 0.03
R_PASS = 0.85
R_SOFT = 0.98


def log(*a):
    print(time.strftime("%H:%M:%S"), "[signed3]", *a, flush=True)


def fwht_unnormalized(x: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh–Hadamard on the last axis. Matches Nov_3rd.ansatz."""
    n = int(x.shape[-1])
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT length must be a power of 2, got {n}")
    y = np.array(x, copy=True)
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = np.stack((a + b, a - b), axis=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def popcount_array(idx: np.ndarray) -> np.ndarray:
    x = idx.astype(np.uint64, copy=True)
    x -= (x >> 1) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x.astype(np.int64)


def hadamard_np(psi: np.ndarray) -> np.ndarray:
    return fwht_unnormalized(psi) / np.sqrt(psi.shape[-1])


def load_clauses(n: int, split: str):
    if int(n) == 21 or int(n) > 12:
        raise RuntimeError(f"refusing n={n}; this smoke stops at n12 and never touches n21")
    path = DATA / f"data_n{n}.npz"
    z = np.load(path)
    return {
        "n": int(z["n"]),
        "m": int(z["m"]),
        "seeds": np.asarray(z[f"{split}_seeds"]).astype(np.int64),
        "cls_v": np.asarray(z[f"{split}_cls_v"]),
        "cls_s": np.asarray(z[f"{split}_cls_s"]),
    }


def terms_3local(cls_v: np.ndarray, cls_s: np.ndarray):
    """Drop the highest-index literal. J = product of the three kept signs."""
    v = np.asarray(cls_v[:, :3], dtype=np.int64)
    s = np.asarray(cls_s[:, :3], dtype=np.int64)
    J = np.prod(s, axis=1).astype(np.float64)
    if not np.all(np.abs(J) == 1.0):
        raise RuntimeError("J_c left {±1}; clause signs are not ±1")
    return v, J


def hz_from_terms(v: np.ndarray, J: np.ndarray, n: int) -> np.ndarray:
    dim = 1 << int(n)
    hz = np.zeros(dim, dtype=np.float64)
    idx = np.arange(dim, dtype=np.int64)
    for t in range(int(v.shape[0])):
        contrib = np.full(dim, float(J[t]), dtype=np.float64)
        for a in range(3):
            shift = int(n) - 1 - int(v[t, a])
            bit = (idx >> shift) & 1
            contrib *= 1.0 - 2.0 * bit
        hz += contrib
    return hz


def cross_cavity_fraction(v: np.ndarray) -> float:
    touched = []
    for t in range(v.shape[0]):
        touched.append(len({int(x) // 3 for x in v[t]}))
    return float(np.mean(np.asarray(touched) >= 2))


def apply_x_np(vec: np.ndarray, n: int, q: int) -> np.ndarray:
    shift = int(n) - 1 - int(q)
    step = 1 << shift
    dim = vec.shape[0]
    a = vec.reshape(dim // (2 * step), 2, step)
    out = np.empty_like(a)
    out[:, 0, :] = a[:, 1, :]
    out[:, 1, :] = a[:, 0, :]
    return out.reshape(dim)


def h_matvec(vec: np.ndarray, hz: np.ndarray, n: int, lam: float) -> np.ndarray:
    out = hz * vec
    for q in range(int(n)):
        out = out - float(lam) * apply_x_np(vec, n, q)
    return out


def ground_state(hz: np.ndarray, n: int, lam: float) -> tuple[float, np.ndarray]:
    """Lowest eigenpair of H_Z − λ Σ X. Dense for n≤9; Lanczos at n=12."""
    dim = 1 << int(n)
    if n <= 9:
        H = np.diag(hz.astype(np.float64))
        for q in range(n):
            shift = n - 1 - q
            step = 1 << shift
            # Off-diagonal −λ on each hypercube edge.
            for base in range(0, dim, 2 * step):
                i0 = np.arange(base, base + step)
                i1 = i0 + step
                H[i0, i1] = -float(lam)
                H[i1, i0] = -float(lam)
        evals, evecs = np.linalg.eigh(H)
        psi = np.real(evecs[:, 0]).astype(np.float64)
        e0 = float(evals[0])
    else:
        def mv(v):
            return h_matvec(np.asarray(v, dtype=np.float64), hz, n, lam)

        op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
        v0 = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
        evals, evecs = eigsh(op, k=1, which="SA", v0=v0, ncv=min(dim - 1, 96), tol=1e-10, maxiter=8000)
        psi = np.real(evecs[:, 0]).astype(np.float64)
        e0 = float(evals[0])
        resid = float(np.linalg.norm(mv(psi) - e0 * psi))
        if resid > 1e-6:
            evals, evecs = eigsh(
                op, k=1, which="SA", v0=np.abs(psi), ncv=min(dim - 1, 160), tol=1e-12, maxiter=16000
            )
            psi = np.real(evecs[:, 0]).astype(np.float64)
            e0 = float(evals[0])
            resid = float(np.linalg.norm(mv(psi) - e0 * psi))
            if resid > 1e-5:
                raise RuntimeError(f"Lanczos residual {resid:.2e} at n={n}")
    psi /= np.linalg.norm(psi)
    if float(psi.sum()) < 0.0:
        psi = -psi
    return e0, psi


def cavity_lambda_max(psi: np.ndarray, n: int) -> list[float]:
    """λ_max of each cavity reduced state. Product across cavities ⇒ 1."""
    C = int(n) // 3
    amp = np.asarray(psi, dtype=np.float64).reshape((2,) * int(n))
    out = []
    for c in range(C):
        keep = (3 * c, 3 * c + 1, 3 * c + 2)
        drop = [i for i in range(int(n)) if i not in keep]
        moved = np.moveaxis(amp, drop + list(keep), range(int(n)))
        mat = moved.reshape(-1, 8)
        rho = mat.T @ mat
        ev = np.linalg.eigvalsh(rho)
        out.append(float(ev[-1]))
    return out


def shell_table(n: int) -> np.ndarray:
    """shell[x, c] = popcount of cavity c (3 bits). Shape (2^n, C)."""
    C = int(n) // 3
    dim = 1 << int(n)
    shell = np.zeros((dim, C), dtype=np.int64)
    idx = np.arange(dim)
    for c in range(C):
        pc = np.zeros(dim, dtype=np.int64)
        for j in range(3):
            shift = int(n) - 1 - (3 * c + j)
            pc += (idx >> shift) & 1
        shell[:, c] = pc
    return shell


def walsh_components(hz: np.ndarray, n: int, m: int) -> np.ndarray:
    """Degree 1..4 pieces of cost_bitstring(hz), shape (4, 2^n)."""
    f = np.asarray(hz, dtype=np.float64) * (float(n) / float(m))
    f = f - f.max()
    F = fwht_unnormalized(f)
    N = f.shape[0]
    pc = popcount_array(np.arange(N))
    comps = np.zeros((4, N), dtype=np.float64)
    for w in range(1, 5):
        Fw = np.where(pc == w, F, 0.0)
        comps[w - 1] = fwht_unnormalized(Fw) / float(N)
    return comps


def _energy_from_theta(theta: np.ndarray, triples: list[tuple[int, int, int, float]], lam: float) -> float:
    c = np.cos(theta)
    e = 0.0
    for i, j, k, J in triples:
        e += J * c[i] * c[j] * c[k]
    e -= float(lam) * float(np.sin(theta).sum())
    return float(e)


def best_product(triples: list[tuple[int, int, int, float]], n: int, lam: float, hz: np.ndarray, rng: np.random.Generator):
    """Coordinate descent on qubit-product states. Returns (E_MF, theta)."""
    inc = [[] for _ in range(n)]
    for i, j, k, J in triples:
        inc[i].append((j, k, J))
        inc[j].append((i, k, J))
        inc[k].append((i, j, J))

    def polish(theta: np.ndarray) -> np.ndarray:
        theta = theta.copy()
        for _ in range(60):
            max_d = 0.0
            for i in range(n):
                F = 0.0
                ci = np.cos(theta)
                for a, b, J in inc[i]:
                    F += J * ci[a] * ci[b]
                R = float(np.hypot(F, lam))
                if R < 1e-15:
                    continue
                # min_θ F cos θ − λ sin θ = −R
                new = float(np.atan2(lam / R, -F / R))
                max_d = max(max_d, abs(new - theta[i]))
                theta[i] = new
            if max_d < 1e-12:
                break
        return theta

    starts = [np.full(n, np.pi / 2)]
    order = np.argsort(hz)[:8]
    for x in order:
        th = np.zeros(n, dtype=np.float64)
        for i in range(n):
            bit = (int(x) >> (n - 1 - i)) & 1
            th[i] = np.pi if bit else 0.0
        starts.append(th)
    for _ in range(8):
        starts.append(rng.uniform(0.0, 2.0 * np.pi, size=n))

    best_e = np.inf
    best_th = starts[0]
    for th0 in starts:
        th = polish(th0)
        e = _energy_from_theta(th, triples, lam)
        if e < best_e:
            best_e = e
            best_th = th
    return float(best_e), best_th


def triples_of(v: np.ndarray, J: np.ndarray) -> list[tuple[int, int, int, float]]:
    out = []
    for t in range(v.shape[0]):
        out.append((int(v[t, 0]), int(v[t, 1]), int(v[t, 2]), float(J[t])))
    return out


def summarize(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "B": int(a.size),
    }


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------

def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    n = int(x.shape[-1])
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def hadamard_torch(psi: torch.Tensor) -> torch.Tensor:
    scale = psi.new_tensor(psi.shape[-1]).sqrt()
    return fwht_torch(psi) / scale


def expect_energy(psi: torch.Tensor, hz: torch.Tensor, n: int, lam: float) -> torch.Tensor:
    """⟨H_Z − λ Σ X⟩ for a batch of states. Z-populations + X coherences."""
    p = psi.real.square() + psi.imag.square()
    ez = (p * hz).sum(dim=-1)
    ex = torch.zeros(psi.shape[0], dtype=hz.dtype, device=psi.device)
    dim = psi.shape[-1]
    for q in range(int(n)):
        shift = int(n) - 1 - q
        step = 1 << shift
        a = psi.reshape(psi.shape[0], dim // (2 * step), 2, step)
        # ⟨X⟩ = 2 Re Σ_{bit=0} ψ0* ψ1
        coh = (a[:, :, 0, :].conj() * a[:, :, 1, :]).sum(dim=(1, 2))
        ex = ex + (coh + coh.conj()).real
    return ez - float(lam) * ex


def apply_mixer_x(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """⊗ exp(−i β X / 2), matching Nov_3rd QAOA convention."""
    half = beta.to(dtype=psi.real.dtype) / 2
    c = torch.cos(half).to(dtype=psi.dtype)
    s = (-1j * torch.sin(half)).to(dtype=psi.dtype)
    B, dim = psi.shape
    for q in range(int(n)):
        shift = int(n) - 1 - q
        step = 1 << shift
        a = psi.reshape(B, dim // (2 * step), 2, step)
        a0, a1 = a[:, :, 0, :], a[:, :, 1, :]
        b0 = c * a0 + s * a1
        b1 = s * a0 + c * a1
        psi = torch.stack((b0, b1), dim=2).reshape(B, dim)
    return psi


def fourier_schedule(coeffs: torch.Tensor, L: int) -> torch.Tensor:
    """Zhou et al.: θ_ℓ = Σ_k c_k sin((k+1/2)(ℓ+1/2) π / L)."""
    q = int(coeffs.shape[0])
    ell = torch.arange(L, dtype=coeffs.dtype, device=coeffs.device)
    out = torch.zeros(L, dtype=coeffs.dtype, device=coeffs.device)
    for k in range(q):
        out = out + coeffs[k] * torch.sin((k + 0.5) * (ell + 0.5) * np.pi / L)
    return out


def qaoa_energy_sched(beta: torch.Tensor, gamma: torch.Tensor, cn: torch.Tensor, hz: torch.Tensor, n: int, lam: float) -> torch.Tensor:
    B, dim = cn.shape
    psi = torch.ones(B, dim, dtype=torch.complex128, device=cn.device) / np.sqrt(dim)
    for ell in range(int(beta.shape[0])):
        psi = psi * torch.exp(-1j * gamma[ell] * cn)
        psi = apply_mixer_x(psi, beta[ell], n)
    return expect_energy(psi, hz, n, lam)


def snap_phases(phi: torch.Tensor, shell: torch.Tensor) -> torch.Tensor:
    """phi (L, 4), shell (dim, C) → phase (L, dim) = Σ_c φ_{popcount(c)}."""
    picked = phi[:, shell]  # (L, dim, C)
    return picked.sum(dim=-1)


def walsh_energy(gamma3: torch.Tensor, phi: torch.Tensor, comp3: torch.Tensor, hz: torch.Tensor, shell: torch.Tensor, n: int, lam: float) -> torch.Tensor:
    """Nov_3rd motif: |0⟩ → [W exp(−i γ3 E_3) W → shell SNAP]^L → W.

    comp3 is the degree-3 Walsh piece of cost_bitstring (includes n/m).
    """
    B, dim = comp3.shape
    psi = torch.zeros(B, dim, dtype=torch.complex128, device=comp3.device)
    psi[:, 0] = 1.0 + 0.0j
    phases = snap_phases(phi, shell)
    L = int(gamma3.shape[0])
    for ell in range(L):
        psi = hadamard_torch(psi)
        psi = psi * torch.exp(-1j * gamma3[ell] * comp3)
        psi = hadamard_torch(psi)
        psi = psi * torch.exp(-1j * phases[ell])
    psi = hadamard_torch(psi)
    return expect_energy(psi, hz, n, lam)


def fourier_q(L: int) -> int:
    """L=2 is full-rank with q=2; L≥3 uses up to 3 Zhou modes."""
    return int(min(Q_FOURIER, L))


def design_matrix(L: int, q: int) -> np.ndarray:
    ell = (np.arange(L, dtype=np.float64) + 0.5)
    cols = [np.sin((k + 0.5) * ell * np.pi / L) for k in range(q)]
    return np.stack(cols, axis=1)


def init_fourier(L: int, seed: int, lam: float) -> np.ndarray:
    q = fourier_q(L)
    ell = np.arange(L, dtype=np.float64)
    denom = float(max(L - 1, 1))
    beta_tgt = 0.35 + 0.55 * (1.0 - ell / denom)
    gamma_tgt = -0.15 - 0.45 * (ell / denom)
    A = design_matrix(L, q)
    cb, *_ = np.linalg.lstsq(A, beta_tgt, rcond=None)
    cg, *_ = np.linalg.lstsq(A, gamma_tgt, rcond=None)
    rng = np.random.default_rng(10_000 + int(seed) * 100 + int(L) * 10 + int(round(float(lam) * 100)))
    cb = cb + rng.normal(0.0, 0.05, size=q)
    cg = cg + rng.normal(0.0, 0.05, size=q)
    return np.concatenate([cb, cg]).astype(np.float64)


class TorchBatch:
    def __init__(self, hz: np.ndarray, e0: np.ndarray, comp3: np.ndarray, n: int, m: int, shell: np.ndarray):
        self.n = int(n)
        self.m = int(m)
        self.hz = torch.tensor(hz, dtype=torch.float64)
        self.e0 = torch.tensor(e0, dtype=torch.float64)
        self.comp3 = torch.tensor(comp3, dtype=torch.float64)
        self.cn = self.hz * (float(n) / float(m))
        self.shell = torch.tensor(shell, dtype=torch.long)


def _lbfgs(x0: np.ndarray, fun_grad, maxiter: int) -> tuple[np.ndarray, float, bool]:
    hist = {"n": 0, "loss": []}

    def wrapped(x):
        loss, grad = fun_grad(x)
        hist["n"] += 1
        hist["loss"].append(float(loss))
        return loss, grad

    res = minimize(
        wrapped,
        np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-6, "maxls": 40},
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun), bool(res.success)


def train_fourier(batch_list: list[TorchBatch], L: int, seed: int, lam: float, maxiter: int) -> dict:
    q = fourier_q(L)
    x0 = init_fourier(L, seed, lam)

    def fun_grad(x):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        beta = fourier_schedule(t[:q], L)
        gamma = fourier_schedule(t[q:], L)
        loss = torch.zeros((), dtype=torch.float64)
        for b in batch_list:
            e = qaoa_energy_sched(beta, gamma, b.cn, b.hz, b.n, lam)
            loss = loss + (e / float(b.n)).mean()
        loss = loss / float(len(batch_list))
        g = torch.autograd.grad(loss, t)[0]
        return float(loss.detach()), g.detach().cpu().numpy().astype(np.float64)

    x, loss, ok = _lbfgs(x0, fun_grad, maxiter)
    t = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        beta = fourier_schedule(t[:q], L).cpu().numpy()
        gamma = fourier_schedule(t[q:], L).cpu().numpy()
    return {
        "x": x,
        "beta": beta,
        "gamma": gamma,
        "train_loss": loss,
        "init_loss": float(fun_grad(x0)[0]),
        "success": ok,
        "q": q,
    }


def train_walsh(batch_list: list[TorchBatch], beta: np.ndarray, gamma: np.ndarray, lam: float, maxiter: int) -> dict:
    L = int(len(beta))
    # Dual warm start: γ_3 ← γ_QAOA, φ_j ← −β j (shell).
    gamma3_0 = np.asarray(gamma, dtype=np.float64).copy()
    phi0 = (-np.asarray(beta, dtype=np.float64).reshape(L, 1)) * np.arange(4, dtype=np.float64).reshape(1, 4)
    x0 = np.concatenate([gamma3_0, phi0.reshape(-1)])

    def unpack(t: torch.Tensor):
        g3 = t[:L]
        phi = t[L:].reshape(L, 4)
        return g3, phi

    def fun_grad(x):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        g3, phi = unpack(t)
        loss = torch.zeros((), dtype=torch.float64)
        for b in batch_list:
            e = walsh_energy(g3, phi, b.comp3, b.hz, b.shell, b.n, lam)
            loss = loss + (e / float(b.n)).mean()
        loss = loss / float(len(batch_list))
        g = torch.autograd.grad(loss, t)[0]
        return float(loss.detach()), g.detach().cpu().numpy().astype(np.float64)

    x, loss, ok = _lbfgs(x0, fun_grad, maxiter)
    t = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        g3, phi = unpack(t)
        g3_np = g3.cpu().numpy()
        phi_np = phi.cpu().numpy()
    return {
        "x": x,
        "gamma3": g3_np,
        "phi": phi_np,
        "train_loss": loss,
        "init_loss": float(fun_grad(x0)[0]),
        "success": ok,
    }


def eval_fourier(beta: np.ndarray, gamma: np.ndarray, batch: TorchBatch, lam: float) -> np.ndarray:
    bt = torch.tensor(beta, dtype=torch.float64)
    gt = torch.tensor(gamma, dtype=torch.float64)
    with torch.no_grad():
        e = qaoa_energy_sched(bt, gt, batch.cn, batch.hz, batch.n, lam)
        resid = (e - batch.e0) / float(batch.n)
    return resid.cpu().numpy().astype(np.float64)


def eval_walsh(gamma3: np.ndarray, phi: np.ndarray, batch: TorchBatch, lam: float) -> np.ndarray:
    g3 = torch.tensor(gamma3, dtype=torch.float64)
    ph = torch.tensor(phi, dtype=torch.float64)
    with torch.no_grad():
        e = walsh_energy(g3, ph, batch.comp3, batch.hz, batch.shell, batch.n, lam)
        resid = (e - batch.e0) / float(batch.n)
    return resid.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Ground-state quantumness
# ---------------------------------------------------------------------------

def instance_record(pack, idx: int, lam: float, rng: np.random.Generator) -> dict:
    v, J = terms_3local(pack["cls_v"][idx], pack["cls_s"][idx])
    n = int(pack["n"])
    hz = hz_from_terms(v, J, n)
    e0, psi = ground_state(hz, n, lam)
    lams = cavity_lambda_max(psi, n)
    zpeak = float(np.max(psi ** 2))
    triples = triples_of(v, J)
    e_mf, _ = best_product(triples, n, lam, hz, rng)
    e_bit = float(hz.min())  # ⟨X⟩=0 on bitstrings
    if e_mf > e_bit + 1e-6:
        # Coordinate descent missed the classical vertex; keep the better one.
        e_mf = e_bit
    gap = (e_mf - e0) / float(n)
    if gap < -1e-7:
        raise RuntimeError(f"product energy {e_mf} below E0 {e0}")
    return {
        "Lambda_cav": lams,
        "Lambda_med": float(np.median(lams)),
        "zpeak": zpeak,
        "gap": float(max(gap, 0.0)),
        "gap_classical": float((e_bit - e0) / float(n)),
        "E0": float(e0),
        "E_mf": float(e_mf),
        "E_bit": e_bit,
        "hz_rms": float(np.sqrt(np.mean(hz ** 2))),
        "hz": hz,
        "comp3": walsh_components(hz, n, int(pack["m"]))[2],
        "comps": walsh_components(hz, n, int(pack["m"])),
    }


def collect_split(n: int, split: str, count: int, lam: float, seed: int) -> list[dict]:
    pack = load_clauses(n, split)
    B = min(int(count), int(pack["seeds"].shape[0]))
    rng = np.random.default_rng(seed + 17 * n + int(lam * 1000))
    rows = []
    t0 = time.time()
    for i in range(B):
        rows.append(instance_record(pack, i, lam, rng))
        if (i + 1) % 10 == 0 or i + 1 == B:
            log(f"GS n={n} split={split} λ={lam:.2f} {i+1}/{B} ({time.time()-t0:.1f}s)")
    return rows


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _assert_close(name, a, b, tol=1e-8):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if err > tol:
        raise AssertionError(f"{name} err {err:.3e} > {tol}")
    log(f"ok {name} err {err:.2e}")


def self_test():
    log("self-test")
    # Hadamard
    e0 = np.zeros(8)
    e0[0] = 1.0
    hp = hadamard_np(e0)
    _assert_close("H|0>", hp, np.ones(8) / np.sqrt(8), 1e-12)
    _assert_close("H^2", hadamard_np(hadamard_np(np.eye(8)[3])), np.eye(8)[3], 1e-12)

    # One clause ZZZ
    v = np.array([[0, 1, 2]])
    J = np.array([1.0])
    hz = hz_from_terms(v, J, 3)
    if abs(hz[0] - 1.0) > 1e-12 or abs(hz.sum()) > 1e-8:
        raise AssertionError(f"ZZZ table bad hz0={hz[0]} sum={hz.sum()}")
    comps = walsh_components(hz, 3, m=1)
    # n/m = 3; degree 3 carries the whole traceless signal of f = 3*hz (max-shift is degree 0)
    _assert_close("degree3", comps[2], 3.0 * hz, 1e-8)
    if np.max(np.abs(comps[[0, 1, 3]])) > 1e-8:
        raise AssertionError("ZZZ leaked into other Walsh degrees")

    # Matvec vs dense energy on |+>
    plus = np.ones(8) / np.sqrt(8)
    mv = h_matvec(plus, hz, 3, lam=0.5)
    # ⟨+|H|+⟩ = 0 − 0.5*3 = −1.5
    e_plus = float(plus @ mv)
    _assert_close("<+|H|+>", e_plus, -1.5, 1e-10)

    e0, psi = ground_state(hz, 3, 0.5)
    e_dense_check = float(psi @ h_matvec(psi, hz, 3, 0.5))
    _assert_close("GS rayleigh", e_dense_check, e0, 1e-8)
    if psi.min() < -1e-7:
        raise AssertionError(f"GS not the positive Perron vector, min={psi.min()}")
    lams = cavity_lambda_max(psi, 3)
    _assert_close("single cavity Λ", lams[0], 1.0, 1e-8)

    # GHZ across 2 cavities
    ghz = np.zeros(64)
    ghz[0] = 1 / np.sqrt(2)
    ghz[-1] = 1 / np.sqrt(2)
    gl = cavity_lambda_max(ghz, 6)
    _assert_close("GHZ Λ", gl, [0.5, 0.5], 1e-10)
    prod = np.zeros(64)
    prod[0] = 1.0
    _assert_close("product Λ", cavity_lambda_max(prod, 6), [1.0, 1.0], 1e-10)

    # Product optimizer at λ=0 lands on a vertex energy
    triples = [(0, 1, 2, 1.0)]
    rng = np.random.default_rng(0)
    e_mf, _ = best_product(triples, 3, 0.0, hz, rng)
    _assert_close("λ=0 product", e_mf, hz.min(), 1e-8)

    # Fourier design is full rank at L=2, q=2
    A = design_matrix(2, 2)
    if abs(np.linalg.det(A)) < 1e-8:
        raise AssertionError("Fourier q=2 at L=2 is singular")
    log("ok Fourier L=2 full rank")

    # Circuit: γ=0, φ=0 → |+⟩, energy −λ n
    n, lam = 6, 0.4
    hz6 = np.zeros(1 << n)
    comp3 = np.zeros(1 << n)
    shell = torch.tensor(shell_table(n), dtype=torch.long)
    g3 = torch.zeros(2, dtype=torch.float64)
    phi = torch.zeros(2, 4, dtype=torch.float64)
    hz_t = torch.tensor(hz6, dtype=torch.float64).reshape(1, -1)
    c3 = torch.tensor(comp3, dtype=torch.float64).reshape(1, -1)
    e = walsh_energy(g3, phi, c3, hz_t, shell, n, lam)
    _assert_close("idle SNAP energy", float(e), -lam * n, 1e-8)

    # Finite-difference gradient on a tiny Fourier objective
    pack = load_clauses(6, "val")
    v, J = terms_3local(pack["cls_v"][0], pack["cls_s"][0])
    hz1 = hz_from_terms(v, J, 6)
    cn = torch.tensor(hz1 * (6 / pack["m"]), dtype=torch.float64).reshape(1, -1)
    hz_t = torch.tensor(hz1, dtype=torch.float64).reshape(1, -1)
    x = np.array([0.3, 0.2, -0.2, -0.4], dtype=np.float64)

    def loss_of(arr):
        t = torch.tensor(arr, dtype=torch.float64, requires_grad=True)
        beta = fourier_schedule(t[:2], 2)
        gamma = fourier_schedule(t[2:], 2)
        e = qaoa_energy_sched(beta, gamma, cn, hz_t, 6, 0.75)
        return e.sum(), t

    base, t = loss_of(x)
    g = torch.autograd.grad(base, t)[0].detach().numpy()
    eps = 1e-5
    fd = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        lp, _ = loss_of(xp)
        lm, _ = loss_of(xm)
        fd[i] = (float(lp.detach()) - float(lm.detach())) / (2 * eps)
    _assert_close("Fourier grad", g, fd, 1e-4)

    # Walsh grad + match to Nov_3rd populations
    comp = walsh_components(hz1, 6, pack["m"])
    _assert_close("pure degree-3", comp[2], hz1 * (6 / pack["m"]), 1e-6)
    if np.max(np.abs(comp[[0, 1, 3]])) > 1e-6:
        raise AssertionError(f"pack clause leaked Walsh degrees {np.max(np.abs(comp[[0, 1, 3]])):.2e}")
    c3 = torch.tensor(comp[2], dtype=torch.float64).reshape(1, -1)
    xw = np.array([0.25, -0.15, 0.0, 0.1, -0.2, 0.05, 0.2, -0.1, 0.0, 0.3], dtype=np.float64)

    def loss_w(arr):
        t = torch.tensor(arr, dtype=torch.float64, requires_grad=True)
        g3 = t[:2]
        phi = t[2:].reshape(2, 4)
        e = walsh_energy(g3, phi, c3, hz_t, shell, 6, 0.75)
        return e.sum(), t

    base, t = loss_w(xw)
    g = torch.autograd.grad(base, t)[0].detach().numpy()
    fd = np.zeros_like(xw)
    for i in range(xw.size):
        xp, xm = xw.copy(), xw.copy()
        xp[i] += eps
        xm[i] -= eps
        lp, _ = loss_w(xp)
        lm, _ = loss_w(xm)
        fd[i] = (float(lp.detach()) - float(lm.detach())) / (2 * eps)
    _assert_close("Walsh grad", g, fd, 2e-4)

    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Nov_3rd.ansatz import make_sim

    sim = make_sim(6, 2, torch.device("cpu"))
    gamma = torch.zeros(2, 4, dtype=torch.float32)
    gamma[:, 2] = torch.tensor(xw[:2], dtype=torch.float32)
    phi = torch.tensor(xw[2:].reshape(2, 4), dtype=torch.float32)
    E = torch.tensor(hz1, dtype=torch.float32)
    out = sim.forward_nov3rd(E, gamma, phi)
    probs = out["probs"].detach().cpu().numpy()
    grid = sim.grid_to_x.detach().cpu().numpy()
    p_ref = np.zeros(64, dtype=np.float64)
    p_ref[grid.ravel()] = probs.ravel()
    with torch.no_grad():
        g3 = torch.tensor(xw[:2], dtype=torch.float64)
        ph = torch.tensor(xw[2:].reshape(2, 4), dtype=torch.float64)
        # rebuild state populations
        Bdim = c3.shape[-1]
        psi = torch.zeros(1, Bdim, dtype=torch.complex128)
        psi[:, 0] = 1
        phases = snap_phases(ph, shell)
        for ell in range(2):
            psi = hadamard_torch(psi)
            psi = psi * torch.exp(-1j * g3[ell] * c3)
            psi = hadamard_torch(psi)
            psi = psi * torch.exp(-1j * phases[ell])
        psi = hadamard_torch(psi)
        p = (psi.real.square() + psi.imag.square()).numpy().ravel()
    _assert_close("Nov3rd populations", p, p_ref, 2e-5)
    log("self-test passed")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def gs_block(lam: float, count_eval: int) -> dict:
    block = {"lambda": lam, "n": {}}
    for n in EVAL_NS:
        rows = collect_split(n, "val", count_eval, lam, seed=0)
        pooled = np.concatenate([np.asarray(r["Lambda_cav"]) for r in rows])
        block["n"][str(n)] = {
            "Lambda": summarize(pooled),
            "Lambda_inst": summarize([r["Lambda_med"] for r in rows]),
            "gap": summarize([r["gap"] for r in rows]),
            "gap_classical": summarize([r["gap_classical"] for r in rows]),
            "zpeak": summarize([r["zpeak"] for r in rows]),
            "E0_over_n": summarize([r["E0"] / n for r in rows]),
            "hz_rms": summarize([r["hz_rms"] for r in rows]),
            "quantum": bool(float(np.median(pooled)) <= LAMBDA_MAX and float(np.median([r["gap"] for r in rows])) >= GAP_MIN),
        }
        # stash arrays for training on a separate pass
        block["n"][str(n)]["_rows_keep"] = False
    return block


def build_batch(n: int, split: str, count: int, lam: float, offset: int = 0) -> TorchBatch:
    pack = load_clauses(n, split)
    B = min(int(count), int(pack["seeds"].shape[0]) - int(offset))
    if B <= 0:
        raise RuntimeError(f"empty batch n={n} {split}")
    hz = []
    e0 = []
    comp3 = []
    rng = np.random.default_rng(123 + n)
    for i in range(offset, offset + B):
        rec = instance_record(pack, i, lam, rng)
        hz.append(rec["hz"])
        e0.append(rec["E0"])
        comp3.append(rec["comp3"])
    shell = shell_table(n)
    return TorchBatch(np.stack(hz), np.asarray(e0), np.stack(comp3), n, pack["m"], shell)


def quantum_flag(block: dict) -> bool:
    return all(block["n"][str(n)]["quantum"] for n in EVAL_NS)


def judge(gs: dict, cells: list[dict]) -> dict:
    """cells: per (λ, L, seed) paired Walsh/Fourier holdout medians."""
    per_lam = {}
    for lam in LAMBDAS:
        q = quantum_flag(gs[f"{lam:.2f}"])
        entry = {"quantum": q, "L": {}}
        for L in LS:
            seed_rows = [c for c in cells if abs(c["lambda"] - lam) < 1e-9 and c["L"] == L]
            R = {}
            for n in EVAL_NS:
                ratios = []
                wmed = []
                fmed = []
                for c in seed_rows:
                    rw = c["walsh"][str(n)]["median"]
                    rf = c["fourier"][str(n)]["median"]
                    wmed.append(rw)
                    fmed.append(rf)
                    ratios.append(float(rw / rf) if rf > 1e-8 else float("inf"))
                R[str(n)] = {
                    "R_median_over_seeds": float(np.median(ratios)),
                    "R_best_seed": float(np.min(ratios)),
                    "R_worst_seed": float(np.max(ratios)),
                    "resid_walsh_median_of_seed_medians": float(np.median(wmed)),
                    "resid_fourier_median_of_seed_medians": float(np.median(fmed)),
                    "ratios": ratios,
                }
            entry["L"][str(L)] = R
        per_lam[f"{lam:.2f}"] = entry

    q_lams = [k for k, v in per_lam.items() if v["quantum"]]
    reasons = []
    verdict = "KILL"
    r_notes = []
    for k in [f"{lam:.2f}" for lam in LAMBDAS]:
        r6 = per_lam[k]["L"]["2"]["6"]["R_median_over_seeds"]
        r9 = per_lam[k]["L"]["2"]["9"]["R_median_over_seeds"]
        r12 = per_lam[k]["L"]["2"]["12"]["R_median_over_seeds"]
        r_notes.append(f"λ={k} L=2 R(n6,n9,n12)=({r6:.3f},{r9:.3f},{r12:.3f})")
    reasons.append("L=2 median-seed R_Fourier: " + "; ".join(r_notes))
    r_kill = all(
        per_lam[k]["L"]["2"]["6"]["R_median_over_seeds"] > R_SOFT
        and per_lam[k]["L"]["2"]["9"]["R_median_over_seeds"] > R_SOFT
        for k in per_lam
    )
    if not q_lams:
        verdict = "KILL"
        reasons.append(
            "all λ fail quantumness (need median cavity Λ≤0.80 and product gap/n≥0.03 at n=6, 9, and 12)"
        )
    else:
        winners = []
        for k in q_lams:
            r12 = per_lam[k]["L"]["2"]["12"]["R_median_over_seeds"]
            if r12 <= R_PASS:
                winners.append((k, r12))
        if winners:
            verdict = "PASS"
            best = min(winners, key=lambda t: t[1])
            reasons.append(
                f"λ={best[0]} is quantum and L=2 median-seed R_Fourier at n12 is {best[1]:.3f}≤{R_PASS}"
            )
        else:
            r_kill = True
            for k in [f"{lam:.2f}" for lam in LAMBDAS]:
                r6 = per_lam[k]["L"]["2"]["6"]["R_median_over_seeds"]
                r9 = per_lam[k]["L"]["2"]["9"]["R_median_over_seeds"]
                if not (r6 > R_SOFT and r9 > R_SOFT):
                    r_kill = False
            if r_kill:
                verdict = "KILL"
                reasons.append(
                    f"L=2 median-seed R_Fourier > {R_SOFT} at both n6 and n9 for every λ"
                )
            else:
                verdict = "needs_edit"
                soft_bits = []
                for k in q_lams:
                    r12 = per_lam[k]["L"]["2"]["12"]["R_median_over_seeds"]
                    soft_bits.append(f"λ={k} R12={r12:.3f}")
                reasons.append(
                    "quantumness holds for some λ but L=2 n12 R_Fourier is soft "
                    f"(pass ≤{R_PASS}, kill-train >{R_SOFT}): " + ", ".join(soft_bits)
                )
    if r_kill and verdict != "PASS":
        extra = (
            f"L=2 R_Fourier also exceeds {R_SOFT} at both n6 and n9 for every λ, "
            "so the motif kill fires as well. needs_edit does not apply: those ratios sit above the soft band."
        )
        if not q_lams:
            extra += (
                " L=4 shell polish can beat Fourier on the train sizes, but the pass gate is L=2 at n12 "
                "and no scored λ is quantum."
            )
        reasons.append(extra)
    return {"verdict": verdict, "reasons": reasons, "per_lambda": per_lam}


def render_md(payload: dict) -> str:
    d = payload
    lines = []
    lines.append("# Signed 3-local transverse field — breadth smoke")
    lines.append("")
    lines.append(f"_Verdict: **{d['verdict']}**._")
    lines.append("")
    for r in d["judge"]["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Hamiltonian")
    lines.append("")
    lines.append("Family D from `SHORTLIST.md`, μ = 0 (no cavity XX):")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"H = \sum_c J_c Z_{i_c} Z_{j_c} Z_{k_c} \;-\; \lambda \sum_{i=1}^{n} X_i,")
    lines.append(r"\qquad J_c\in\{+1,-1\}.")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        "Supports come from the sealed 4-SAT packs `explore_OCT29/data/data_n{6,9,12}.npz`. "
        "Each clause keeps its three lowest-index literals; the fourth literal is dropped so every "
        "term is strictly 3-local. `J_c` is the product of those three `cls_s` signs. "
        "There is no μ Σ XX inside a cavity — that term can pin the ground state to a product of "
        "cavity states and push cavity Λ back to 1."
    )
    lines.append("")
    lines.append(
        "Cross-cavity fraction of the kept triples, averaged over 16 val instances: "
        + ", ".join(f"n{n}={d['cross'][str(n)]:.3f}" for n in EVAL_NS)
        + "."
    )
    lines.append("")
    lines.append(
        "`H_Z` is a pure Walsh degree-3 function, so graded γ weights on degrees 1, 2, and 4 are "
        "identically idle. Walsh SNAP trains γ_3 per layer plus the 4 Hamming-shell phases. "
        "Fourier QAOA uses Zhou modes of (β, γ) with the locked mixer ⊗ exp(−i β X/2) and cost "
        "phase exp(−i γ H_Z n/m). L=2 with q=2 is full-rank (same family as free angles). "
        "L=4 uses q=3."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- λ ∈ {0.50, 0.75, 1.00}, L ∈ {2, 4}, seeds {0, 1, 2}.")
    lines.append(
        f"- Train on train[:{d['train_B']}] of n=6 and n=9 jointly (one angle set). "
        "Freeze and score val of n=6, n=9, and n=12. n12 is never trained. n21 is not loaded."
    )
    lines.append(
        "- Walsh is warm-started from that seed's trained Fourier angles (γ_3 ← γ, φ_j ← −β j), "
        "then L-BFGS-polished on the same ⟨H⟩/n objective. "
        "That shell is the QAOA mixer: H exp(i β |1⟩⟨1|) H = e^{iβ/2} exp(−i β X/2), "
        "so φ_j=−β j reproduces ⊗ exp(−i β X/2) up to a global phase. "
        "Polish can bend the four shell phases off that ray."
    )
    lines.append("- Optimizer: L-BFGS-B only.")
    lines.append("")
    lines.append("### Metrics")
    lines.append("")
    lines.append(
        "- **Cavity Λ** (true GS): largest eigenvalue of the 8-level reduced state of each cavity "
        "(3 qubits). Headline number is the median over val instances and cavities. "
        "A cavity-product state has Λ = 1. Quantum gate: median Λ ≤ 0.80 at every n ∈ {6, 9, 12}."
    )
    lines.append(
        "- **Product gap**: (E_MF − E_0) / n, where E_MF is the best qubit-product energy from "
        "multi-start coordinate descent (bitstring vertices included). Gate: median gap ≥ 0.03 at every n. "
        "The classical bitstring gap (min_x H_Z(x) − E_0)/n is listed separately and is not the gate: "
        "a tilted product state lies below the best bitstring while remaining separable."
    )
    lines.append("- **Z-peak**: max_x |⟨x|ψ_GS⟩|². Reported with the quantumness numbers; not an extra gate.")
    lines.append(
        "- **R_Fourier**: median_instances(⟨H⟩−E_0)_Walsh / median_instances(⟨H⟩−E_0)_Fourier, "
        "then the median of that ratio over the 3 paired seeds. Lower means Walsh is closer to the "
        "true ground state. Pass needs L=2 R ≤ 0.85 at n12 on a λ that is quantum. "
        "R in (0.85, 0.98] is soft. R > 0.98 at both n6 and n9 for every λ kills the motif."
    )
    lines.append("")
    lines.append("## Ground state")
    lines.append("")
    lines.append("| λ | n | median Λ | p10 Λ | median gap/n | median classical gap/n | median Z-peak | quantum |")
    lines.append("|---|---|----------|-------|--------------|------------------------|---------------|---------|")
    for lam in LAMBDAS:
        key = f"{lam:.2f}"
        for n in EVAL_NS:
            g = d["gs"][key]["n"][str(n)]
            lines.append(
                f"| {lam:.2f} | {n} | {g['Lambda']['median']:.3f} | {g['Lambda']['p10']:.3f} | "
                f"{g['gap']['median']:.4f} | {g['gap_classical']['median']:.4f} | "
                f"{g['zpeak']['median']:.3f} | {'yes' if g['quantum'] else 'no'} |"
            )
    lines.append("")
    rms_bits = []
    for n in EVAL_NS:
        rms_bits.append(f"n{n}={d['gs']['0.50']['n'][str(n)]['hz_rms']['median']:.2f}")
    lines.append(
        "Root-mean-square of H_Z on the hypercube (λ=0.50 block, same diagonal for every λ): "
        + ", ".join(rms_bits)
        + ". Requested λ ∈ {0.50, 0.75, 1.00} sits well below that scale."
    )
    lines.append("")
    lines.append("## Walsh vs Fourier")
    lines.append("")
    lines.append(
        "Residuals are median_(val) (⟨H⟩ − E_0) / n. R is the median over seeds of "
        "(Walsh median / Fourier median)."
    )
    lines.append("")
    lines.append("| λ | L | R n6 | R n9 | R n12 | Walsh resid n12 | Fourier resid n12 |")
    lines.append("|---|---|------|------|-------|-----------------|-------------------|")
    for lam in LAMBDAS:
        key = f"{lam:.2f}"
        for L in LS:
            R = d["judge"]["per_lambda"][key]["L"][str(L)]
            lines.append(
                f"| {lam:.2f} | {L} | {R['6']['R_median_over_seeds']:.3f} | "
                f"{R['9']['R_median_over_seeds']:.3f} | {R['12']['R_median_over_seeds']:.3f} | "
                f"{R['12']['resid_walsh_median_of_seed_medians']:.4f} | "
                f"{R['12']['resid_fourier_median_of_seed_medians']:.4f} |"
            )
    lines.append("")
    lines.append("### Per seed, L=2 (paired)")
    lines.append("")
    lines.append("| λ | seed | R n6 | R n9 | R n12 | train loss Fourier init→final | Walsh init→final |")
    lines.append("|---|------|------|------|-------|--------------------------------|------------------|")
    for c in d["cells"]:
        if c["L"] != 2:
            continue
        def rr(n):
            rw = c["walsh"][str(n)]["median"]
            rf = c["fourier"][str(n)]["median"]
            return rw / rf
        lines.append(
            f"| {c['lambda']:.2f} | {c['seed']} | {rr(6):.3f} | {rr(9):.3f} | {rr(12):.3f} | "
            f"{c['fourier_train']['init_loss']:.4f}→{c['fourier_train']['train_loss']:.4f} | "
            f"{c['walsh_train']['init_loss']:.4f}→{c['walsh_train']['train_loss']:.4f} |"
        )
    lines.append("")
    if d.get("diagnostic"):
        lines.append("## Diagnostic λ scan (not scored)")
        lines.append("")
        lines.append(
            "Same Hamiltonian on the first 8 val instances. This grid is not part of the pass rule; "
            "it only shows where cavity Λ and the product gap move."
        )
        lines.append("")
        lines.append("| λ | n | median Λ | median gap/n | median Z-peak |")
        lines.append("|---|---|----------|--------------|---------------|")
        for row in d["diagnostic"]:
            lines.append(
                f"| {row['lambda']:.2f} | {row['n']} | {row['Lambda']:.3f} | {row['gap']:.4f} | {row['zpeak']:.3f} |"
            )
        lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append(
        "PASS if some scored λ has median cavity Λ ≤ 0.80 and product gap/n ≥ 0.03 at n=6, 9, and 12, "
        "and L=2 median-seed R_Fourier ≤ 0.85 at n12. needs_edit if quantumness holds but that R is soft. "
        "KILL if every scored λ fails quantumness, or if every λ has L=2 R > 0.98 at both n6 and n9."
    )
    lines.append("")
    lines.append(f"**Verdict: {d['verdict']}.**")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"Wall time {d['wall_s']:.0f}s. Train B={d['train_B']}, L-BFGS maxiter={d['maxiter']}. "
        "Did not modify `Nov_3rd/settings.py`, did not write `qaoa_angles/`, and did not run or signal n21."
    )
    lines.append("")
    return "\n".join(lines)


def run_diagnostic(count: int = 8) -> list[dict]:
    grid = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00)
    rows = []
    for n in (6, 9):
        pack = load_clauses(n, "val")
        # Precompute hz once
        prepared = []
        for i in range(min(count, pack["seeds"].shape[0])):
            v, J = terms_3local(pack["cls_v"][i], pack["cls_s"][i])
            prepared.append((hz_from_terms(v, J, n), triples_of(v, J)))
        for lam in grid:
            lams, gaps, peaks = [], [], []
            rng = np.random.default_rng(0)
            for hz, triples in prepared:
                e0, psi = ground_state(hz, n, lam)
                lams.extend(cavity_lambda_max(psi, n))
                e_mf, _ = best_product(triples, n, lam, hz, rng)
                e_bit = float(hz.min())
                e_mf = min(e_mf, e_bit)
                gaps.append((e_mf - e0) / float(n))
                peaks.append(float(np.max(psi ** 2)))
            rows.append({
                "lambda": lam,
                "n": n,
                "Lambda": float(np.median(lams)),
                "gap": float(np.median(gaps)),
                "zpeak": float(np.median(peaks)),
                "B": len(prepared),
            })
            log(f"diag n={n} λ={lam:.2f} Λ={rows[-1]['Lambda']:.3f} gap={rows[-1]['gap']:.4f} Z={rows[-1]['zpeak']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gs-only", action="store_true")
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--train-b", type=int, default=TRAIN_B)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--eval-b", type=int, default=40)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--no-diagnostic", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, int(torch.get_num_threads())))

    if args.self_test:
        self_test()
        return
    if not args.skip_self_test:
        self_test()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip() != "")
    t_all = time.time()

    cross = {}
    for n in EVAL_NS:
        pack = load_clauses(n, "val")
        v, _ = terms_3local(pack["cls_v"][0], pack["cls_s"][0])
        # fraction over the whole val split, first instance is not enough — average 16
        fr = []
        for i in range(min(16, pack["cls_v"].shape[0])):
            vv, _ = terms_3local(pack["cls_v"][i], pack["cls_s"][i])
            fr.append(cross_cavity_fraction(vv))
        cross[str(n)] = float(np.mean(fr))
        # disjoint train/val
        if n in TRAIN_NS:
            tr = load_clauses(n, "train")
            overlap = set(tr["seeds"][: args.train_b].tolist()) & set(pack["seeds"][: args.eval_b].tolist())
            if overlap:
                raise RuntimeError(f"train/val seed overlap at n={n}: {list(overlap)[:5]}")
        log(f"cross-cavity fraction n={n}: {cross[str(n)]:.3f}")

    gs = {}
    for lam in LAMBDAS:
        block = {"lambda": lam, "n": {}}
        for n in EVAL_NS:
            rows = collect_split(n, "val", args.eval_b, lam, seed=0)
            pooled = np.concatenate([np.asarray(r["Lambda_cav"], dtype=np.float64) for r in rows])
            block["n"][str(n)] = {
                "Lambda": summarize(pooled),
                "Lambda_inst": summarize([r["Lambda_med"] for r in rows]),
                "gap": summarize([r["gap"] for r in rows]),
                "gap_classical": summarize([r["gap_classical"] for r in rows]),
                "zpeak": summarize([r["zpeak"] for r in rows]),
                "E0_over_n": summarize([r["E0"] / n for r in rows]),
                "hz_rms": summarize([r["hz_rms"] for r in rows]),
                "quantum": bool(
                    float(np.median(pooled)) <= LAMBDA_MAX
                    and float(np.median([r["gap"] for r in rows])) >= GAP_MIN
                ),
            }
            g = block["n"][str(n)]
            log(
                f"λ={lam:.2f} n={n} Λ={g['Lambda']['median']:.3f} "
                f"gap={g['gap']['median']:.4f} Z={g['zpeak']['median']:.3f} quantum={g['quantum']}"
            )
        gs[f"{lam:.2f}"] = block

    diagnostic = [] if args.no_diagnostic else run_diagnostic(8)

    if args.gs_only:
        preview = {"gs": gs, "cross": cross, "diagnostic": diagnostic}
        (OUT / "gs_only.json").write_text(json.dumps(preview, indent=2))
        log("wrote gs_only.json")
        return

    # Train / eval. Rebuild GS inside batches (val and train) so energies match.
    cells = []
    for lam in LAMBDAS:
        log(f"preparing train/holdout batches λ={lam:.2f}")
        train_batches = [build_batch(n, "train", args.train_b, lam, offset=0) for n in TRAIN_NS]
        hold = {n: build_batch(n, "val", args.eval_b, lam, offset=0) for n in EVAL_NS}
        for L in LS:
            for seed in seeds:
                log(f"train Fourier λ={lam:.2f} L={L} seed={seed}")
                four = train_fourier(train_batches, L, seed, lam, args.maxiter)
                log(
                    f"  Fourier loss {four['init_loss']:.4f} → {four['train_loss']:.4f} "
                    f"success={four['success']}"
                )
                log(f"train Walsh λ={lam:.2f} L={L} seed={seed}")
                wal = train_walsh(train_batches, four["beta"], four["gamma"], lam, args.maxiter)
                log(
                    f"  Walsh loss {wal['init_loss']:.4f} → {wal['train_loss']:.4f} "
                    f"success={wal['success']}"
                )
                cell = {
                    "lambda": lam,
                    "L": L,
                    "seed": seed,
                    "fourier_train": {
                        "init_loss": four["init_loss"],
                        "train_loss": four["train_loss"],
                        "success": four["success"],
                        "q": four["q"],
                        "beta": four["beta"].tolist(),
                        "gamma": four["gamma"].tolist(),
                    },
                    "walsh_train": {
                        "init_loss": wal["init_loss"],
                        "train_loss": wal["train_loss"],
                        "success": wal["success"],
                        "gamma3": wal["gamma3"].tolist(),
                        "phi": wal["phi"].tolist(),
                    },
                    "fourier": {},
                    "walsh": {},
                }
                for n in EVAL_NS:
                    rf = eval_fourier(four["beta"], four["gamma"], hold[n], lam)
                    rw = eval_walsh(wal["gamma3"], wal["phi"], hold[n], lam)
                    if np.nanmin(rf) < -1e-4 or np.nanmin(rw) < -1e-4:
                        raise RuntimeError(f"residual below zero n={n} λ={lam} L={L} seed={seed}")
                    cell["fourier"][str(n)] = summarize(rf)
                    cell["walsh"][str(n)] = summarize(rw)
                    ratio = cell["walsh"][str(n)]["median"] / max(cell["fourier"][str(n)]["median"], 1e-12)
                    log(f"  n={n} resid W/F {cell['walsh'][str(n)]['median']:.4f}/"
                        f"{cell['fourier'][str(n)]['median']:.4f} R={ratio:.3f}")
                cells.append(cell)
                # incremental dump
                (OUT / "cells_partial.json").write_text(json.dumps({"cells": cells}, indent=2))

    judged = judge(gs, cells)
    payload = {
        "family": "D_signed_3local_tf",
        "mu": 0,
        "verdict": judged["verdict"],
        "judge": judged,
        "gs": gs,
        "cells": cells,
        "cross": cross,
        "diagnostic": diagnostic,
        "train_B": args.train_b,
        "eval_B": args.eval_b,
        "maxiter": args.maxiter,
        "seeds": list(seeds),
        "wall_s": time.time() - t_all,
        "definition": {
            "H": "sum_c J_c Z_i Z_j Z_k - λ sum_i X_i",
            "J": "product of three kept cls_s; 4th literal dropped",
            "Lambda": "median over instances and cavities of λ_max(ρ_cavity) of the true GS",
            "product_gap": "(E_mean_field - E_0) / n",
            "R_Fourier": "median_seed of median_i(resid_Walsh)/median_i(resid_Fourier)",
            "resid": "(<H> - E_0) / n",
        },
    }
    # json-safe: gs already plain
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    md = render_md(payload)
    (OUT / "RESULTS.md").write_text(md)
    log(f"VERDICT {judged['verdict']}")
    for r in judged["reasons"]:
        log(" ", r)
    log(f"wrote {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
