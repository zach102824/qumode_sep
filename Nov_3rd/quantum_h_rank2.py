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


def _add_symmetric_flip(rows, cols, vals, idx: np.ndarray, mask: int, weight: float) -> None:
    """One real off-diagonal of weight ``weight`` on every pair z ↔ z⊕mask."""
    if weight == 0.0 or mask == 0:
        return
    flipped = idx ^ np.int64(mask)
    keep = idx < flipped
    r = idx[keep]
    c = flipped[keep]
    v = np.full(r.shape[0], float(weight), dtype=np.float64)
    rows.extend((r, c))
    cols.extend((c, r))
    vals.extend((v, v.copy()))


def sparse_rank2(diag: np.ndarray, n: int, lam: float, mu: float, kappa: float):
    """CSR of the same matrix as ``dense_rank2``. XX/XXX edges are cavity-local flips.

    μ = κ = 0 stores only the diagonal and the single-qubit X edges, matching
    the Arm A transverse-field sparsity.
    """
    from scipy.sparse import coo_matrix

    n = int(n)
    diag = np.asarray(diag, dtype=np.float64)
    dim = 1 << n
    if diag.shape != (dim,):
        raise ValueError(f"diag {diag.shape} != {(dim,)}")
    idx = np.arange(dim, dtype=np.int64)
    rows = [idx]
    cols = [idx]
    vals = [diag]
    for i in range(n):
        _add_symmetric_flip(rows, cols, vals, idx, 1 << (n - 1 - i), -float(lam))
    if mu != 0.0:
        for mask in _pair_masks(n):
            _add_symmetric_flip(rows, cols, vals, idx, mask, float(mu))
    if kappa != 0.0:
        for mask in _triple_masks(n):
            _add_symmetric_flip(rows, cols, vals, idx, mask, float(kappa))
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    ).tocsr()


def sparse_ground_rank2(diag: np.ndarray, n: int, lam: float, mu: float, kappa: float, tol: float = 1e-9) -> dict:
    """Lowest eigenpair via Lanczos. ψ is in bitstring order (var 0 = MSB)."""
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    H = sparse_rank2(diag, n, lam, mu, kappa)
    dim = int(H.shape[0])
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
    psi = psi / float(np.linalg.norm(psi))
    if float(psi.sum()) < 0.0:
        psi = -psi
    ev = float(evals[0])
    resid = float(np.linalg.norm(H @ psi - ev * psi))
    if resid > 1e-5:
        raise RuntimeError(f"n={n} μ={mu} κ={kappa} Lanczos residual {resid:.2e}")
    return {
        "E0": ev,
        "peak": float(np.max(psi ** 2)),
        "entropy": cavity_entropy_bits(psi, n),
        "residual": resid,
    }


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
        """⟨H_c⟩ from the Z histogram, ⟨X⟩ / ⟨XX⟩ / ⟨XXX⟩ from the X histogram.

        The field term uses the same single-qubit reduction as Arm A (``xexp``),
        which the cavity marginal reproduces. μ = κ = 0 therefore takes the
        Arm A graph exactly: the XX and XXX reductions are not added.
        """
        out = self.run(Hc_x, gamma, phi)
        e = out["e_c"] - float(lam) * out["xexp"].sum(dim=-1)
        if mu != 0.0 or kappa != 0.0:
            _sx, e_xx, e_xxx = self.x_correlators(out["pX"])
            e = e + float(mu) * e_xx + float(kappa) * e_xxx
        return e

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


# Locked before the μ=−0.25 transfer run. PR #15 published the μ=0 freeze.
PUBLISHED_MU0_N12_R = 0.924
BEAT_R = 0.98
TIE_R = 1.02
CLEAR_DROP = 0.02
CONTROL_TOL = 0.01


def paired_clear(R_new: float, R_base: float, r_new, r_base, min_drop: float = CLEAR_DROP) -> dict:
    """Whether R_new is clearly below R_base.

    The headline drop is the ratio-of-medians. A seed whose baseline per-seed
    ratio stays at the Fourier point (R ≥ 0.995) is stuck on the warm start:
    it does not count as a win and it does not veto. The remaining seeds must
    drop by at least ``min_drop`` at the median, and that median must be at
    least the std of those paired drops.
    """
    r_new = [float(x) for x in r_new]
    r_base = [float(x) for x in r_base]
    if len(r_new) != len(r_base) or len(r_new) < 1:
        raise ValueError("per-seed ratios")
    drop = float(R_base) - float(R_new)
    paired = [b - a for a, b in zip(r_new, r_base)]
    active = [d for d, b in zip(paired, r_base) if b < 0.995]
    if len(active) >= 2:
        med = float(np.median(active))
        std = float(np.std(active, ddof=1)) if len(active) > 1 else 0.0
        clear = bool(drop >= min_drop and med >= min_drop and med + 1e-12 >= std)
    elif len(active) == 0:
        med = float(np.median(paired))
        std = float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
        clear = bool(drop >= max(min_drop, std) and med >= min_drop)
    else:
        # One seed left the Fourier point. That is not a multi-seed gain.
        med = float(active[0])
        std = 0.0
        clear = False
    return {
        "drop": float(drop),
        "paired": paired,
        "active": active,
        "n_active": int(len(active)),
        "active_median": float(med),
        "active_std": float(std),
        "clear": clear,
    }


def decide_rank2_transfer(
    *,
    R_mu: float,
    R_mu0: float,
    per_seed_R_mu,
    per_seed_R_mu0,
    peak: float,
    prod_gap: float,
    R_mu_n15: float | None = None,
    published_R: float = PUBLISHED_MU0_N12_R,
) -> dict:
    """PASS / needs_edit / KILL for the frozen n=12 L=16 power_p comparison.

    PASS requires R(μ=−0.25) ≤ 0.98 against Fourier QAOA, a quantum n=12
    ground state (peak < 0.5 and product gap ≥ 2%), a clearly lower ratio
    than the in-run μ=0 transfer, and a ratio no higher than the published
    μ=0 transfer (PR #15, 0.924). The in-run μ=0 control must itself beat
    QAOA and land within 0.01 of 0.924; otherwise the comparison is not the
    published one and the call is needs_edit.
    """
    clear = paired_clear(R_mu, R_mu0, per_seed_R_mu, per_seed_R_mu0)
    beat = float(R_mu) <= BEAT_R
    quantum_pass = float(peak) < 0.5 and float(prod_gap) >= 0.02
    classical = float(peak) >= 0.5
    below_published = float(R_mu) <= float(published_R) + 1e-12
    control_beat = float(R_mu0) <= BEAT_R
    control_match = abs(float(R_mu0) - float(published_R)) <= CONTROL_TOL
    n15_flip = R_mu_n15 is not None and float(R_mu_n15) >= 1.10
    info = {
        "clear": clear,
        "beat_qaoa": beat,
        "quantum_pass": quantum_pass,
        "classical_gs": classical,
        "below_published": below_published,
        "control_beat": control_beat,
        "control_match": control_match,
        "n15_flip": bool(n15_flip),
        "published_R": float(published_R),
        "R_mu": float(R_mu),
        "R_mu0": float(R_mu0),
        "drop_vs_published": float(published_R) - float(R_mu),
    }
    if classical:
        info["verdict"] = "KILL"
        info["reason"] = "n=12 ground state is a bitstring (peak ≥ 0.5)"
        return info
    reverses = float(R_mu) >= 1.05 and not (float(R_mu) < float(R_mu0) - 1e-6)
    loses_both = float(R_mu) > TIE_R and float(R_mu) >= float(R_mu0) - 1e-9
    if reverses or loses_both:
        info["verdict"] = "KILL"
        info["reason"] = (
            "n=12 does not beat Fourier QAOA and does not improve on the μ=0 transfer"
            if loses_both and not reverses
            else "n=12 advantage reverses and μ=−0.25 is not below the μ=0 transfer"
        )
        return info
    if beat and quantum_pass and clear["clear"] and below_published and control_beat and control_match and not n15_flip:
        info["verdict"] = "PASS"
        info["reason"] = (
            f"frozen n=12 R={float(R_mu):.3f} ≤ 0.98 and clearly below the μ=0 transfer "
            f"(in-run {float(R_mu0):.3f}, published {float(published_R):.3f})"
        )
        return info
    missed = []
    if not beat:
        missed.append(f"n=12 R={float(R_mu):.3f} is above 0.98")
    if not quantum_pass:
        missed.append(f"n=12 quantumness fails (peak={float(peak):.3f}, prod gap={float(prod_gap):.3f})")
    if not clear["clear"]:
        missed.append(
            f"drop vs in-run μ=0 is {clear['drop']:.3f}, not a clear gain "
            f"(active median {clear['active_median']:.3f}, std {clear['active_std']:.3f})"
        )
    if not below_published:
        missed.append(f"n=12 R={float(R_mu):.3f} is above the published μ=0 transfer {float(published_R):.3f}")
    if not control_beat or not control_match:
        missed.append(
            f"in-run μ=0 R={float(R_mu0):.3f} does not reproduce the published transfer {float(published_R):.3f}"
        )
    if n15_flip:
        missed.append(f"n=15 R={float(R_mu_n15):.3f} flips a ratio past 1.10")
    info["verdict"] = "needs_edit"
    info["reason"] = "; ".join(missed) if missed else "bars not all met"
    return info
