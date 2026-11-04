"""Rank-2 deformation of Arm A: intra-cavity X products on Max-E3LIN2.

    H = Σ_E J_abc Z_a Z_b Z_c − λ Σ_i X_i
        + μ Σ_cav Σ_{i<j} X_i X_j + κ Σ_cav X_1 X_2 X_3

The gate sequence is unchanged (BinH ``U_c^X`` ↔ Hamming-shell SNAP).
``H_c = Σ J ZZZ`` is still the only diagonal piece inside ``U_c``.
The field and the intra-cavity X products are scored from the X histogram:

  * Z histogram (after the final W) → ⟨H_c⟩
  * per-cavity 8-outcome X histogram (final W omitted) → ⟨X_i⟩, ⟨X_i X_j⟩, ⟨XXX⟩

Those operators are cavity-local X strings, so the joint X histogram of each
3-qubit cavity is sufficient. No Y correlator enters H.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from Nov_3rd.quantum_h_tf import (
    FockSim,
    best_product_energy,
    cavity_entropy_bits,
    dense_hamiltonian,
    exact_ground,
    fidelity_susceptibility,
    load_train_instances,
    zzz_diag,
)

# Small grid. (0, 0) is Arm A.
MU_GRID = (-0.25, 0.0, 0.25)
KAPPA_GRID = (-0.25, 0.0, 0.25)


def grid_points() -> list[tuple[float, float]]:
    pts = [(float(mu), float(ka)) for mu in MU_GRID for ka in KAPPA_GRID]
    pts.sort(key=lambda mk: (abs(mk[0]) + abs(mk[1]), abs(mk[1]), mk[0], mk[1]))
    return pts


def tag_mk(mu: float, kappa: float) -> str:
    return f"m{mu:+.2f}_k{kappa:+.2f}"


def plus_energy(n: int, lam: float, mu: float, kappa: float) -> float:
    """⟨+|H|+⟩. Each cavity contributes 3 XX pairs and one XXX, all +1 on |+⟩."""
    n = int(n)
    if n % 3 != 0:
        raise ValueError(n)
    return float(-lam * n + mu * n + kappa * (n / 3.0))


def _pair_masks(n: int) -> list[int]:
    masks = []
    for c in range(n // 3):
        bits = (3 * c, 3 * c + 1, 3 * c + 2)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            masks.append((1 << (n - 1 - bits[a])) | (1 << (n - 1 - bits[b])))
    return masks


def _triple_masks(n: int) -> list[int]:
    masks = []
    for c in range(n // 3):
        mask = 0
        for j in range(3):
            mask |= 1 << (n - 1 - (3 * c + j))
        masks.append(mask)
    return masks


def dense_rank2(diag: np.ndarray, n: int, lam: float, mu: float, kappa: float) -> np.ndarray:
    """Real symmetric H in the Z basis (var 0 = MSB)."""
    n = int(n)
    dim = 1 << n
    if diag.shape != (dim,):
        raise ValueError(f"diag {diag.shape} != {(dim,)}")
    if abs(mu) < 1e-15 and abs(kappa) < 1e-15:
        return dense_hamiltonian(diag, n, lam)
    H = np.zeros((dim, dim), dtype=np.float64)
    H[np.diag_indices(dim)] = np.asarray(diag, dtype=np.float64)
    idx = np.arange(dim, dtype=np.int64)
    for i in range(n):
        H[idx, idx ^ (1 << (n - 1 - i))] -= float(lam)
    if mu != 0.0:
        for mask in _pair_masks(n):
            H[idx, idx ^ mask] += float(mu)
    if kappa != 0.0:
        for mask in _triple_masks(n):
            H[idx, idx ^ mask] += float(kappa)
    return H


def local_x_eigs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues of Σ X, Σ_{i<j} XX, XXX on one cavity, X-basis index 0..7.

    Local integer MSB is the first qubit of the cavity. Bit 0 → eigenvalue +1.
    """
    sx = np.zeros(8, dtype=np.float64)
    xx = np.zeros(8, dtype=np.float64)
    xxx = np.zeros(8, dtype=np.float64)
    for f in range(8):
        s = [1.0 - 2.0 * ((f >> (2 - j)) & 1) for j in range(3)]
        sx[f] = s[0] + s[1] + s[2]
        xx[f] = s[0] * s[1] + s[0] * s[2] + s[1] * s[2]
        xxx[f] = s[0] * s[1] * s[2]
    return sx, xx, xxx


def product_energy_rank2(theta: np.ndarray, clauses, n: int, lam: float, mu: float, kappa: float) -> float:
    """XZ product state: ⟨Z⟩=cos θ, ⟨X⟩=sin θ, ⟨Y⟩=0."""
    cz = np.cos(theta)
    sx = np.sin(theta)
    e = 0.0
    for a, b, c, J in clauses:
        e += float(J) * float(cz[a] * cz[b] * cz[c])
    e -= float(lam) * float(sx.sum())
    for c0 in range(int(n) // 3):
        i, j, k = 3 * c0, 3 * c0 + 1, 3 * c0 + 2
        e += float(mu) * float(sx[i] * sx[j] + sx[i] * sx[k] + sx[j] * sx[k])
        e += float(kappa) * float(sx[i] * sx[j] * sx[k])
    return float(e)


def _coordinate_descent(theta: np.ndarray, clauses, n: int, lam: float, mu: float, kappa: float) -> np.ndarray:
    """Exact single-spin updates. α cos θ_i + β sin θ_i, others held fixed."""
    th = np.array(theta, dtype=np.float64, copy=True)
    cz = np.cos(th)
    sx = np.sin(th)
    for _ in range(48):
        max_d = 0.0
        for i in range(n):
            alpha = 0.0
            beta = -float(lam)
            for a, b, c, J in clauses:
                if i == a:
                    alpha += float(J) * float(cz[b] * cz[c])
                elif i == b:
                    alpha += float(J) * float(cz[a] * cz[c])
                elif i == c:
                    alpha += float(J) * float(cz[a] * cz[b])
            cav = i // 3
            mates = [3 * cav + j for j in range(3) if 3 * cav + j != i]
            beta += float(mu) * float(sum(sx[m] for m in mates))
            if len(mates) == 2:
                beta += float(kappa) * float(sx[mates[0]] * sx[mates[1]])
            new = math.atan2(-beta, -alpha)
            # α=β=0 → atan2(0,0)=0, which is a fine gauge choice
            d = abs(math.atan2(math.sin(new - th[i]), math.cos(new - th[i])))
            if d > max_d:
                max_d = d
            th[i] = new
            cz[i] = math.cos(new)
            sx[i] = math.sin(new)
        if max_d < 1e-12:
            break
    return th


def best_product_rank2(clauses, n: int, lam: float, mu: float, kappa: float, seed: int = 0) -> float:
    from scipy.optimize import minimize

    diag = zzz_diag(n, clauses)
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
    rng = np.random.default_rng(int(seed) + 17 * int(n) + int(round(1000 * lam)) + int(round(100 * mu)) + int(round(10 * kappa)))
    for _ in range(4):
        starts.append(rng.uniform(-np.pi, np.pi, size=n))
    polished = []
    for th0 in starts:
        polished.append(_coordinate_descent(th0, clauses, n, lam, mu, kappa))
    best = min(product_energy_rank2(th, clauses, n, lam, mu, kappa) for th in polished)
    # Polish the distinct coordinate-descent minima. Order is deterministic.
    order = sorted(range(len(polished)), key=lambda i: product_energy_rank2(polished[i], clauses, n, lam, mu, kappa))
    seen = []
    for i in order:
        th = polished[i]
        if any(np.max(np.abs(np.sin(0.5 * (th - s)))) < 0.05 for s in seen):
            continue
        seen.append(th)
        if len(seen) > 3:
            break
        res = minimize(
            lambda t: product_energy_rank2(t, clauses, n, lam, mu, kappa),
            th,
            method="L-BFGS-B",
        )
        if np.isfinite(res.fun):
            best = min(best, float(res.fun))
    return float(best)


def exact_ground_rank2(diag: np.ndarray, clauses, n: int, lam: float, mu: float, kappa: float, seed: int, product: bool) -> dict:
    H = dense_rank2(diag, n, lam, mu, kappa)
    evals, evecs = np.linalg.eigh(H)
    psi = evecs[:, 0]
    out = {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "gap": float(evals[1] - evals[0]),
        "peak": float(np.max(psi ** 2)),
        "entropy": cavity_entropy_bits(psi, n),
        "chi": fidelity_susceptibility(evals, evecs, n),
        "E_cl": float(np.min(diag)),
        "E_plus": plus_energy(n, lam, mu, kappa),
        "lam": float(lam),
        "mu": float(mu),
        "kappa": float(kappa),
        "seed": int(seed),
        "n": int(n),
    }
    if product:
        eprod = best_product_rank2(clauses, n, lam, mu, kappa, seed=seed)
        out["E_prod"] = float(eprod)
        denom = max(abs(out["E0"]), 1e-8)
        out["prod_gap"] = float((eprod - out["E0"]) / denom)
        out["prod_gap_density"] = float((eprod - out["E0"]) / float(n))
    return out


class Rank2Sim(FockSim):
    """FockSim twin. Energy adds intra-cavity XX and XXX from the X histogram."""

    def __init__(self, n: int, device=None, dtype=torch.float64):
        super().__init__(n, device=device, dtype=dtype)
        sx, xx, xxx = local_x_eigs()
        self.sx8 = torch.as_tensor(sx, dtype=self.rdtype, device=self.device)
        self.xx8 = torch.as_tensor(xx, dtype=self.rdtype, device=self.device)
        self.xxx8 = torch.as_tensor(xxx, dtype=self.rdtype, device=self.device)

    def x_correlators(self, pX: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-cavity X histogram → Σ⟨X⟩, Σ_cav Σ_{i<j}⟨XX⟩, Σ_cav ⟨XXX⟩.

        ``pX`` is the joint X-basis distribution in bitstring order (var 0 = MSB).
        Cavity-local operators use only the 8-outcome marginal of that cavity.
        """
        B = int(pX.shape[0])
        if int(pX.shape[-1]) != self.dim:
            raise ValueError(f"pX {tuple(pX.shape)} != dim {self.dim}")
        marg = pX.reshape(B, *([8] * self.C))
        e_sx = torch.zeros(B, dtype=self.rdtype, device=pX.device)
        e_xx = torch.zeros(B, dtype=self.rdtype, device=pX.device)
        e_xxx = torch.zeros(B, dtype=self.rdtype, device=pX.device)
        for c in range(self.C):
            axes = tuple(i for i in range(1, self.C + 1) if i != c + 1)
            m = marg.sum(dim=axes) if axes else marg
            e_sx = e_sx + m @ self.sx8
            e_xx = e_xx + m @ self.xx8
            e_xxx = e_xxx + m @ self.xxx8
        return e_sx, e_xx, e_xxx

    def energy(
        self,
        Hc_x: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        lam: float,
        mu: float = 0.0,
        kappa: float = 0.0,
    ) -> torch.Tensor:
        out = self.run(Hc_x, gamma, phi)
        e_sx, e_xx, e_xxx = self.x_correlators(out["pX"])
        return out["e_c"] - float(lam) * e_sx + float(mu) * e_xx + float(kappa) * e_xxx

    def state_bit_Z(self, Hc_x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Amplitudes after the final W, bitstring order. For the ⟨ψ|H|ψ⟩ check."""
        B, dim = Hc_x.shape
        if dim != self.dim:
            raise ValueError(dim)
        L = int(gamma.shape[0])
        Hc_f = self.fock_of_bit_diag(Hc_x).to(self.rdtype)
        psi = torch.zeros(B, self.dim, dtype=self.cdtype, device=self.device)
        psi[:, 0] = 1.0 + 0.0j
        g = gamma.to(self.rdtype)
        for ell in range(L):
            psi = self._apply_W(psi)
            psi = psi * torch.exp((-1j * g[ell]).to(self.cdtype) * Hc_f.to(self.cdtype))
            psi = self._apply_W(psi)
            psi = self._apply_snap(psi, phi[ell])
        psi = self._apply_W(psi)
        return psi.index_select(-1, self.inv)


def _brute_x_strings(p: np.ndarray, n: int) -> tuple[float, float, float]:
    dim = 1 << int(n)
    idx = np.arange(dim, dtype=np.int64)
    e_sx = 0.0
    e_xx = 0.0
    e_xxx = 0.0
    for c in range(n // 3):
        s = []
        for j in range(3):
            bit = (idx >> (n - 1 - (3 * c + j))) & 1
            s.append(1.0 - 2.0 * bit.astype(np.float64))
        e_sx += float(p @ (s[0] + s[1] + s[2]))
        e_xx += float(p @ (s[0] * s[1] + s[0] * s[2] + s[1] * s[2]))
        e_xxx += float(p @ (s[0] * s[1] * s[2]))
    return e_sx, e_xx, e_xxx


def self_check(pack_n6=None) -> dict:
    """Raises on failure. Proves the X-product readout matches ⟨ψ|H|ψ⟩ and Arm A at μ=κ=0."""
    from pathlib import Path

    from Nov_3rd import settings as S

    report: dict = {"needs_YY": False}
    sx, xx, xxx = local_x_eigs()
    # |+++⟩ is local index 0
    assert abs(sx[0] - 3) < 1e-12 and abs(xx[0] - 3) < 1e-12 and abs(xxx[0] - 1) < 1e-12
    # one minus on the last qubit: local bits 001 → integer 1
    assert abs(xxx[1] + 1) < 1e-12

    rng = np.random.default_rng(0)
    for _ in range(6):
        lam, mu, ka = [float(x) for x in rng.uniform(-0.7, 0.7, size=3)]
        spec = -lam * sx + mu * xx + ka * xxx
        H = dense_rank2(np.zeros(8), 3, lam, mu, ka)
        ev = np.linalg.eigvalsh(H)
        assert np.allclose(np.sort(ev), np.sort(spec), atol=1e-10), (ev, spec)
    # μ=κ=0 matches Arm A dense builder, including a ZZZ diagonal
    clauses = [(0, 1, 2, 1), (1, 2, 3, -1), (0, 2, 4, 1), (3, 4, 5, -1)]
    # n=6 needs a full diag
    d6 = zzz_diag(6, [(0, 1, 4, 1), (1, 2, 5, -1), (0, 3, 5, 1)])
    for lam in (0.0, 0.5, 1.25):
        a = dense_hamiltonian(d6, 6, lam)
        b = dense_rank2(d6, 6, lam, 0.0, 0.0)
        assert np.allclose(a, b, atol=1e-12)
    g_a = exact_ground(d6, 6, 0.5)
    g_b = exact_ground_rank2(d6, [], 6, 0.5, 0.0, 0.0, seed=0, product=False)
    assert abs(g_a["E0"] - g_b["E0"]) < 1e-9, (g_a["E0"], g_b["E0"])

    if pack_n6 is None:
        pack_n6 = Path(S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    insts = load_train_instances(6, 2, pack_n6)
    # product optimizer agrees with Arm A at μ=κ=0 (equal or lower)
    for ins in insts:
        e_ref = best_product_energy(ins.clauses, ins.n, 0.5, seed=ins.seed)
        e_new = best_product_rank2(ins.clauses, ins.n, 0.5, 0.0, 0.0, seed=ins.seed)
        assert e_new <= e_ref + 1e-4, (e_new, e_ref, ins.seed)
    report["product_matches_arm_A"] = True

    sim = Rank2Sim(6, dtype=torch.float64)
    # brute-force X strings vs per-cavity marginal
    p = rng.random(sim.dim)
    p = p / p.sum()
    pt = torch.as_tensor(p, dtype=torch.float64).reshape(1, -1)
    got = sim.x_correlators(pt)
    brute = _brute_x_strings(p, 6)
    for g, b in zip(got, brute):
        assert abs(float(g) - b) < 1e-10, (float(g), b)

    Hc = torch.as_tensor(insts[0].diag, dtype=torch.float64).reshape(1, -1)
    # |+⟩ (L=0)
    zero_g = torch.zeros(0, dtype=torch.float64)
    zero_p = torch.zeros(0, 4, dtype=torch.float64)
    for mu, ka in ((0.0, 0.0), (0.25, -0.15), (-0.4, 0.2)):
        e = float(sim.energy(Hc, zero_g, zero_p, 0.5, mu, ka))
        expect = plus_energy(6, 0.5, mu, ka)  # ⟨H_c⟩_+ = 0
        assert abs(e - expect) < 1e-8, (e, expect, mu, ka)

    gamma = torch.tensor([0.17, -0.11, 0.05], dtype=torch.float64)
    phi = torch.tensor(
        [[0.0, -0.2, 0.05, 0.01], [0.0, 0.15, -0.07, 0.02], [0.0, 0.04, -0.02, 0.03]],
        dtype=torch.float64,
    )
    # single-qubit X marginal matches Arm A xexp
    base = sim.run(Hc, gamma, phi)
    e_sx, _, _ = sim.x_correlators(base["pX"])
    assert torch.allclose(e_sx, base["xexp"].sum(dim=-1), atol=1e-8)

    for mu, ka, lam in ((0.0, 0.0, 0.5), (0.25, 0.25, 0.5), (-0.3, 0.15, 0.75), (0.0, -0.25, 0.5)):
        e_hist = float(sim.energy(Hc, gamma, phi, lam, mu, ka))
        psi = sim.state_bit_Z(Hc, gamma, phi)[0].detach().cpu().numpy()
        H = dense_rank2(insts[0].diag, 6, lam, mu, ka)
        e_psi = float(np.real(np.vdot(psi, H @ psi)))
        assert abs(e_hist - e_psi) < 1e-8, (e_hist, e_psi, mu, ka, lam)
        assert abs(float(psi.real @ psi.real + psi.imag @ psi.imag) - 1.0) < 1e-8
    report["statevector_max_abs"] = 1e-8

    # autograd sees μ and κ
    theta = torch.nn.Parameter(torch.tensor([0.2, -0.1, -0.15, 0.05, 0.02, -0.01, 0.0, 0.01], dtype=torch.float64))

    def loss_of(th: torch.Tensor) -> torch.Tensor:
        g = th[:2]
        c1, c2, c3 = th[2:4], th[4:6], th[6:8]
        k = torch.arange(4, dtype=torch.float64)
        powers = torch.stack([k, k * k, k * k * k], 0)
        C = torch.stack([c1, c2, c3], 1)
        phi_l = C @ powers
        return sim.energy(Hc, g, phi_l, 0.5, 0.25, -0.2).sum()

    loss_of(theta).backward()
    g_auto = float(theta.grad[0])
    eps = 1e-6
    with torch.no_grad():
        theta[0] += eps
        lp = float(loss_of(theta))
        theta[0] -= 2 * eps
        lm = float(loss_of(theta))
        theta[0] += eps
    g_fd = (lp - lm) / (2 * eps)
    assert abs(g_auto - g_fd) < 1e-4, (g_auto, g_fd)
    report["grad_abs"] = abs(g_auto - g_fd)
    report["ok"] = True
    return report
