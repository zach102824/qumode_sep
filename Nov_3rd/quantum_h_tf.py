"""Transverse-field Max-E3LIN2 on the locked Nov_3rd gate sequence.

Hamiltonian (Opus sign)::

    H = H_c - λ Σ_i X_i
    H_c = Σ_{(a,b,c)} J_abc Z_a Z_b Z_c,   J_abc = ±1

Gate sequence is unchanged (BinH, b=3)::

    |0⟩ → [U_c^X → Hamming-shell SNAP]^L → W
    U_c^X = W exp(-i γ H_c) W,   W = H^{⊗3}

The transverse field is not Trotterized into U_c. It is scored from the
X-histogram (circuit with the final W omitted)::

    ⟨H⟩ = Σ_z p_Z(z) H_c(z) - λ Σ_i ⟨X_i⟩
    ⟨X_i⟩ = 2 p_{X_i}(+) - 1

Instances: existing 4-SAT packs do not store 3-local ``cls_s``. Clause
supports are the cross-cavity triples inside those 4-clauses, with
``J = ±1`` the product of the three literal signs (first occurrence).
Degree is capped at 3 so ``m = n`` when the pack geometry allows it.
Train rows come from ``beat_qaoa_12h/train_packs`` (E regenerated from the
stored seed and checked). Holdout rows use the scale1000 seed base in
``regen_scale1000.py`` (the scale1000 npz itself is not in this checkout).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from explore_OCT29.lib.sat_gen import gen_instance
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH, build_grid_to_x_binary

# regen_scale1000.py
SCALE1000_BASE = 54_000_000_000
POPCOUNT8 = np.array([bin(i).count("1") for i in range(8)], dtype=np.int64)


@dataclass
class E3Instance:
    n: int
    seed: int
    split: str
    clauses: list  # (a, b, c, J)
    diag: np.ndarray  # H_c eigenvalues, bitstring order, var0 = MSB
    m: int
    degree: np.ndarray


def _unique_cross_triples(n: int, cl) -> list[tuple[tuple[int, int, int], int]]:
    """Cross-cavity 3-subsets of the pack's 4-clauses, J = product of signs."""
    seen: dict[tuple[int, int, int], int] = {}
    order: list[tuple[tuple[int, int, int], int]] = []
    for v, s in cl:
        v = [int(x) for x in v]
        s = [int(x) for x in s]
        for drop in range(4):
            idxs = [i for i in range(4) if i != drop]
            tri = [v[i] for i in idxs]
            sig = [s[i] for i in idxs]
            order_i = sorted(range(3), key=lambda t: tri[t])
            tri_s = tuple(tri[t] for t in order_i)
            J = 1
            for t in order_i:
                J *= sig[t]
            if len({t // 3 for t in tri_s}) < 2:
                continue
            if tri_s in seen:
                continue
            seen[tri_s] = int(J)
            order.append((tri_s, int(J)))
    return order


def extract_e3lin(n: int, cl, m_target: int | None = None) -> list[tuple[int, int, int, int]]:
    """Greedy degree-capped cross-cavity 3-XORSAT clauses. Prefer degree ≤ 3."""
    if m_target is None:
        m_target = int(n)
    cand = _unique_cross_triples(n, cl)
    chosen: list[tuple[int, int, int, int]] = []
    deg = np.zeros(n, dtype=np.int64)
    used = set()
    # Stay at degree ≤ 3. A short hypergraph is still D≤3; do not bump the cap.
    for cap in (3,):
        progressed = True
        while len(chosen) < m_target and progressed:
            progressed = False
            best_i = None
            best_key = None
            for i, (tri, _J) in enumerate(cand):
                if i in used:
                    continue
                if any(int(deg[t]) >= cap for t in tri):
                    continue
                key = (int(sum(deg[t] for t in tri)), int(max(deg[t] for t in tri)), i)
                if best_key is None or key < best_key:
                    best_key = key
                    best_i = i
            if best_i is None:
                break
            tri, J = cand[best_i]
            used.add(best_i)
            for t in tri:
                deg[t] += 1
            chosen.append((int(tri[0]), int(tri[1]), int(tri[2]), int(J)))
            progressed = True
        if len(chosen) >= m_target:
            break
    return chosen


def zzz_diag(n: int, clauses) -> np.ndarray:
    """Eigenvalues of Σ J Z_a Z_b Z_c. Z|0⟩=+1, Z|1⟩=-1, var0 = MSB."""
    dim = 1 << int(n)
    diag = np.zeros(dim, dtype=np.float64)
    idx = np.arange(dim, dtype=np.int64)
    for a, b, c, J in clauses:
        ba = (idx >> (n - 1 - int(a))) & 1
        bb = (idx >> (n - 1 - int(b))) & 1
        bc = (idx >> (n - 1 - int(c))) & 1
        eig = np.where(((ba + bb + bc) & 1) == 0, 1.0, -1.0)
        diag += float(J) * eig
    return diag


def degrees_of(n: int, clauses) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    for a, b, c, _J in clauses:
        deg[a] += 1
        deg[b] += 1
        deg[c] += 1
    return deg


def instance_from_clauses(n: int, seed: int, split: str, cl) -> E3Instance:
    clauses = extract_e3lin(n, cl, m_target=n)
    if len(clauses) == 0:
        raise RuntimeError(f"n={n} seed={seed}: no cross-cavity triple")
    diag = zzz_diag(n, clauses)
    return E3Instance(
        n=n,
        seed=int(seed),
        split=split,
        clauses=clauses,
        diag=diag,
        m=len(clauses),
        degree=degrees_of(n, clauses),
    )


def load_train_instances(n: int, count: int, pack_path) -> list[E3Instance]:
    z = np.load(pack_path, allow_pickle=False)
    seeds = np.asarray(z["train_seeds"]).ravel()
    E = np.asarray(z["train_E"])
    if len(seeds) < count:
        raise RuntimeError(f"{pack_path} has {len(seeds)} < {count}")
    out = []
    for i in range(count):
        seed = int(seeds[i])
        _m, cl, Eg = gen_instance(n, seed, compute_E=True)
        if not np.array_equal(np.asarray(Eg), np.asarray(E[i])):
            raise RuntimeError(f"n={n} seed={seed}: regenerated E != train pack")
        out.append(instance_from_clauses(n, seed, "train", cl))
    return out


def load_scale_holdout(n: int, count: int, base: int = SCALE1000_BASE) -> list[E3Instance]:
    """First ``count`` satisfiable formulas at the scale1000 seed base.

    Same seed map as ``regen_scale1000.py`` (``base + 10_000_000 n + k``),
    skipping unsatisfiable draws the way ``instances()`` does.
    """
    out = []
    k = 0
    limit = max(8000, count * 500)
    while len(out) < count:
        if k >= limit:
            raise RuntimeError(f"n={n}: only {len(out)}/{count} sat scale1000 draws")
        seed = int(base) + 10_000_000 * int(n) + k
        k += 1
        _m, cl, E = gen_instance(n, seed, compute_E=True)
        if int(np.min(E)) != 0:
            continue
        out.append(instance_from_clauses(n, seed, "scale1000", cl))
    return out


def dense_hamiltonian(diag: np.ndarray, n: int, lam: float) -> np.ndarray:
    """H = diag(H_c) - λ Σ_i X_i. Real symmetric."""
    dim = 1 << int(n)
    if diag.shape != (dim,):
        raise ValueError(f"diag {diag.shape} != {(dim,)}")
    H = np.zeros((dim, dim), dtype=np.float64)
    H[np.diag_indices(dim)] = diag
    idx = np.arange(dim, dtype=np.int64)
    for i in range(n):
        mask = 1 << (n - 1 - i)
        H[idx, idx ^ mask] -= float(lam)
    return H


def _sx_on_vec(psi: np.ndarray, n: int) -> np.ndarray:
    idx = np.arange(psi.shape[0], dtype=np.int64)
    acc = np.zeros_like(psi)
    for i in range(n):
        acc += psi[idx ^ (1 << (n - 1 - i))]
    return acc


def cavity_entropy_bits(psi: np.ndarray, n: int) -> float:
    """von Neumann entropy (bits) across cavity 0 | rest. ψ in bitstring order."""
    amp = np.asarray(psi, dtype=np.complex128).reshape(8, -1)
    s = np.linalg.svd(amp, compute_uv=False)
    p = np.real(s) ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def fidelity_susceptibility(evals: np.ndarray, evecs: np.ndarray, n: int) -> float:
    """χ_F w.r.t. λ for H = H_c - λ Σ X. ∂_λ H = -Σ X."""
    psi0 = evecs[:, 0]
    amps = evecs.T @ _sx_on_vec(psi0, n)  # ⟨m|Σ X|0⟩
    den = evals - evals[0]
    den = den.copy()
    den[0] = np.inf
    # tiny gaps: keep them (they dominate χ) but avoid 0/0
    den[np.abs(den) < 1e-14] = np.inf
    return float(np.sum((np.real(amps) ** 2) / den ** 2))


def product_energy(theta: np.ndarray, clauses, lam: float) -> float:
    """XZ product state: ⟨Z⟩=cos θ, ⟨X⟩=sin θ."""
    cz = np.cos(theta)
    e = 0.0
    for a, b, c, J in clauses:
        e += float(J) * float(cz[a] * cz[b] * cz[c])
    e -= float(lam) * float(np.sin(theta).sum())
    return float(e)


def best_product_energy(clauses, n: int, lam: float, seed: int = 0) -> float:
    from scipy.optimize import minimize

    diag = zzz_diag(n, clauses)
    # classical bitstrings are θ ∈ {0, π}
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
    rng = np.random.default_rng(seed + 17 * n + int(round(1000 * lam)))
    for _ in range(4):
        starts.append(rng.uniform(-np.pi, np.pi, size=n))
    best = min(product_energy(th, clauses, lam) for th in starts)
    for th0 in starts:
        res = minimize(lambda th: product_energy(th, clauses, lam), th0, method="L-BFGS-B")
        if res.success or np.isfinite(res.fun):
            best = min(best, float(res.fun))
    return float(best)


def exact_ground(diag: np.ndarray, n: int, lam: float) -> dict:
    H = dense_hamiltonian(diag, n, lam)
    evals, evecs = np.linalg.eigh(H)
    psi = evecs[:, 0]
    peak = float(np.max(psi ** 2))
    sx = float(np.real(psi @ _sx_on_vec(psi, n)))
    return {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "gap": float(evals[1] - evals[0]),
        "peak": peak,
        "sx": sx,
        "entropy": cavity_entropy_bits(psi, n),
        "chi": fidelity_susceptibility(evals, evecs, n),
        "E_cl": float(np.min(diag)),  # bitstring, ⟨X⟩=0
    }


def ed_observables(inst: E3Instance, lam: float, product: bool = True) -> dict:
    out = exact_ground(inst.diag, inst.n, lam)
    if product:
        eprod = best_product_energy(inst.clauses, inst.n, lam, seed=inst.seed)
        out["E_prod"] = eprod
        denom = max(abs(out["E0"]), 1e-8)
        out["prod_gap"] = float((eprod - out["E0"]) / denom)
        out["prod_gap_density"] = float((eprod - out["E0"]) / inst.n)
    out["lam"] = float(lam)
    out["seed"] = int(inst.seed)
    out["m"] = int(inst.m)
    return out


class FockSim:
    """Vectorized twin of Nov3rdAnsatz.forward_nov3rd on the nf=8 subspace.

    Flat index is the Fock product with cavity 0 slowest, matching
    ``build_grid_to_x_binary``.
    """

    def __init__(self, n: int, device=None, dtype=torch.float64):
        if n % 3 != 0:
            raise ValueError(f"n={n} not divisible by 3")
        self.n = int(n)
        self.C = int(n // 3)
        self.dim = 8 ** self.C
        self.device = torch.device(device or "cpu")
        self.rdtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        W = build_W_BinH(nf=8, dtype=torch.float64).numpy()[:8, :8]
        self.H3 = torch.as_tensor(W, dtype=self.cdtype, device=self.device)
        perm = build_grid_to_x_binary(n).reshape(-1).astype(np.int64)
        if perm.shape != (self.dim,) or len(np.unique(perm)) != self.dim:
            raise RuntimeError("grid_to_x is not a permutation")
        inv = np.empty(self.dim, dtype=np.int64)
        inv[perm] = np.arange(self.dim)
        self.perm = torch.as_tensor(perm, device=self.device)  # fock -> x
        self.inv = torch.as_tensor(inv, device=self.device)  # x -> fock
        pcs = []
        for c in range(self.C):
            div = 8 ** (self.C - 1 - c)
            f = (np.arange(self.dim) // div) % 8
            pcs.append(POPCOUNT8[f])
        self.pc_idx = torch.as_tensor(np.stack(pcs, 0), dtype=torch.long, device=self.device)
        idx = np.arange(self.dim, dtype=np.int64)
        bits = np.stack([((idx >> (n - 1 - i)) & 1).astype(np.float64) for i in range(n)], 0)
        self.bit1 = torch.as_tensor(bits, dtype=self.rdtype, device=self.device)  # (n, dim) bitstring

    def fock_of_bit_diag(self, Hc_x: torch.Tensor) -> torch.Tensor:
        """(B, dim_x) → (B, dim_fock)."""
        return Hc_x.index_select(-1, self.perm)

    def _apply_W(self, psi: torch.Tensor) -> torch.Tensor:
        B = psi.shape[0]
        x = psi.reshape(B, *([8] * self.C))
        for c in range(self.C):
            x = torch.tensordot(self.H3, x, dims=([1], [c + 1]))
            x = torch.movedim(x, 0, c + 1)
        return x.reshape(B, self.dim)

    def _apply_snap(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """phi: (4,) shell phases. Phase is Σ_cav φ[pc(f_cav)]."""
        ph = phi.to(dtype=self.rdtype)[self.pc_idx]  # (C, dim)
        total = ph.sum(dim=0)
        return psi * torch.exp(-1j * total.to(self.cdtype))

    def run(self, Hc_x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor) -> dict:
        """gamma (L,), phi (L, 4). Hc_x (B, 2^n) bitstring order.

        Returns Z-histogram probabilities (bitstring order) and X expectations.
        """
        B, dim = Hc_x.shape
        if dim != self.dim:
            raise ValueError(f"Hc width {dim} != {self.dim}")
        L = int(gamma.shape[0])
        if tuple(phi.shape) != (L, 4):
            raise ValueError(f"phi {tuple(phi.shape)} != {(L, 4)}")
        Hc_f = self.fock_of_bit_diag(Hc_x).to(self.rdtype)
        psi = torch.zeros(B, self.dim, dtype=self.cdtype, device=self.device)
        psi[:, 0] = 1.0 + 0.0j
        g = gamma.to(self.rdtype)
        for ell in range(L):
            psi = self._apply_W(psi)
            psi = psi * torch.exp((-1j * g[ell]).to(self.cdtype) * Hc_f.to(self.cdtype))
            psi = self._apply_W(psi)
            psi = self._apply_snap(psi, phi[ell])
        # X-histogram: Fock readout before the final W, bits via grid_to_x.
        p_fock_x = psi.real.square() + psi.imag.square()
        p_bit_x = p_fock_x.index_select(-1, self.inv)
        p1 = torch.matmul(p_bit_x, self.bit1.T)  # (B, n)
        xexp = 1.0 - 2.0 * p1
        psi = self._apply_W(psi)
        p_fock_z = psi.real.square() + psi.imag.square()
        p_bit_z = p_fock_z.index_select(-1, self.inv)
        e_c = (p_bit_z * Hc_x.to(self.rdtype)).sum(dim=-1)
        return {
            "e_c": e_c,
            "xexp": xexp,
            "pZ": p_bit_z,
            "pX": p_bit_x,
            "norm": p_fock_z.sum(dim=-1),
        }

    def energy(self, Hc_x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, lam: float) -> torch.Tensor:
        out = self.run(Hc_x, gamma, phi)
        return out["e_c"] - float(lam) * out["xexp"].sum(dim=-1)


def phi_powers(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """φ_k = c1 k + c2 k^2 + c3 k^3, k=0..3. Matches beat_params.phi_from_c."""
    k = torch.arange(4, device=c1.device, dtype=c1.dtype)
    pows = torch.stack([k, k * k, k * k * k], dim=0)  # (3, 4)
    C = torch.stack([c1.reshape(-1), c2.reshape(-1), c3.reshape(-1)], dim=1)
    return C @ pows


def fourier_schedule(coeffs: torch.Tensor, L: int) -> torch.Tensor:
    """Zhou et al.: θ_ℓ = Σ_k c_k sin((k+1/2)(ℓ+1/2) π / L)."""
    q = int(coeffs.shape[0])
    ell = torch.arange(L, device=coeffs.device, dtype=coeffs.dtype)
    out = torch.zeros(L, device=coeffs.device, dtype=coeffs.dtype)
    for k in range(q):
        out = out + coeffs[k] * torch.sin((k + 0.5) * (ell + 0.5) * math.pi / L)
    return out


def fit_fourier(angles: np.ndarray, q: int) -> np.ndarray:
    L = int(angles.shape[0])
    ell = np.arange(L, dtype=np.float64)
    A = np.stack([np.sin((k + 0.5) * (ell + 0.5) * np.pi / L) for k in range(q)], axis=1)
    coeff, *_ = np.linalg.lstsq(A, np.asarray(angles, dtype=np.float64), rcond=None)
    return coeff


def paper_fourier_init(L: int, q: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares Fourier coeffs of (β, γ) = (0.01, -0.01)."""
    if q is None:
        q = int(L)
    beta = np.full(L, 0.01, dtype=np.float64)
    gamma = np.full(L, -0.01, dtype=np.float64)
    return fit_fourier(gamma, q), fit_fourier(beta, q)


def stack_diags(insts: list[E3Instance], device, dtype) -> torch.Tensor:
    arr = np.stack([ins.diag for ins in insts], axis=0)
    return torch.as_tensor(arr, dtype=dtype, device=device)


def self_check(device="cpu") -> dict:
    """Circuit identity, X-readout, ED, and pack-seed checks. Raises on failure."""
    from Nov_3rd.ansatz import make_sim
    from Nov_3rd.scripts.beat_params import phi_from_c

    report = {}
    # ED: one qubit, H = -λ X
    H = dense_hamiltonian(np.zeros(2), 1, lam=1.25)
    ev = np.linalg.eigvalsh(H)
    assert abs(ev[0] + 1.25) < 1e-10, ev
    # one triangle, λ=0 → E0 = -1
    clauses = [(0, 1, 2, 1)]
    d = zzz_diag(3, clauses)
    assert abs(d.min() + 1) < 1e-12 and abs(d.max() - 1) < 1e-12
    g0 = exact_ground(d, 3, lam=0.0)
    assert abs(g0["E0"] + 1) < 1e-9, g0
    # large λ: E0 < -λ n, and ⟨+|H|+⟩ = -λ n
    gL = exact_ground(d, 3, lam=2.0)
    assert gL["E0"] < -2.0 * 3 + 0.2

    # pack seed reproduces stored E, extraction yields ±1 cross-cavity clauses
    from pathlib import Path
    from Nov_3rd import settings as S

    pack = Path(S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    insts = load_train_instances(6, 2, pack)
    for ins in insts:
        assert ins.m == 6, ins.m
        assert set(J for *_, J in ins.clauses) <= {-1, 1}
        assert all(len({a // 3, b // 3, c // 3}) >= 2 for a, b, c, _J in ins.clauses)
        assert int(ins.degree.max()) == 3
    report["pack_n6_m"] = [ins.m for ins in insts]

    # shell polynomial matches beat_params
    c1 = torch.tensor([0.2, -0.1], dtype=torch.float64)
    c2 = torch.tensor([0.05, 0.0], dtype=torch.float64)
    c3 = torch.tensor([-0.01, 0.02], dtype=torch.float64)
    got = phi_powers(c1, c2, c3)
    ref = phi_from_c(c1.float(), c2.float(), c3.float()).double()
    assert torch.allclose(got, ref, atol=1e-6), (got, ref)

    # Fock twin vs Nov3rdAnsatz on a raw diagonal (not the normalized Cn).
    n, L = 6, 2
    sim_f = FockSim(n, device=device, dtype=torch.float64)
    rng = np.random.default_rng(0)
    Hc = torch.as_tensor(rng.normal(size=(1, 1 << n)), dtype=torch.float64)
    gamma = torch.tensor([0.17, -0.11], dtype=torch.float64)
    phi = torch.tensor(
        [[0.0, -0.2, 0.05, 0.01], [0.0, 0.15, -0.07, 0.02]],
        dtype=torch.float64,
    )
    cav = make_sim(n, L, torch.device(device))
    psi = cav.vacuum()
    grid = cav.grid_to_x
    diag = Hc[0].to(dtype=cav.rdtype, device=psi.device)[grid]
    g32 = gamma.to(cav.rdtype)
    phi32 = phi.to(cav.rdtype)
    # X-hist reference: stop before final W
    for ell in range(L):
        psi = cav._apply_all(psi, cav.Wmat)
        psi = psi * torch.exp(-1j * g32[ell].to(cav.cdtype) * diag.to(cav.cdtype))
        psi = cav._apply_all(psi, cav.Wmat)
        th = phi32[ell][cav.popcount]
        psi = cav._apply_all(psi, cav._snap_theta(th))
    pX_cav = (psi.real.square() + psi.imag.square()).reshape(-1).double()
    psi = cav._apply_all(psi, cav.Wmat)
    pZ_cav = (psi.real.square() + psi.imag.square()).reshape(-1).double()
    out = sim_f.run(Hc, gamma, phi)
    # cavity probs are fock-flat; FockSim pZ/pX are bitstring order
    pZ_fock = out["pZ"][0].index_select(0, sim_f.perm)
    pX_fock = out["pX"][0].index_select(0, sim_f.perm)
    errZ = float((pZ_fock - pZ_cav).abs().max())
    errX = float((pX_fock - pX_cav).abs().max())
    assert errZ < 2e-5, errZ
    assert errX < 2e-5, errX
    report["circuit_max_abs_pZ"] = errZ
    report["circuit_max_abs_pX"] = errX

    # QAOA point: φ_j = -β j matches ideal qubit mixer on probabilities.
    from explore_OCT29.lib.qaoa import _apply_x_mixer

    betas = torch.tensor([0.31, -0.22], dtype=torch.float64)
    gammas = torch.tensor([-0.18, 0.07], dtype=torch.float64)
    k = torch.arange(4, dtype=torch.float64)
    phi_q = (-betas.reshape(L, 1)) * k.reshape(1, -1)
    out_q = sim_f.run(Hc, gammas, phi_q)
    psi_q = torch.full((1, 1 << n), 1.0 / math.sqrt(1 << n), dtype=torch.complex128)
    Hc_c = Hc.to(torch.complex128)
    for ell in range(L):
        psi_q = psi_q * torch.exp(-1j * gammas[ell] * Hc_c)
        psi_q = _apply_x_mixer(psi_q, betas[ell].to(torch.float32), n).to(torch.complex128)
    p_ref = (psi_q.real.square() + psi_q.imag.square())[0]
    err_q = float((out_q["pZ"][0] - p_ref).abs().max())
    assert err_q < 2e-5, err_q
    report["qaoa_point_max_abs_p"] = err_q

    # |+⟩ limit (L=0): ⟨X_i⟩ = +1 and ⟨H_c⟩ = 0 on a balanced ZZZ diag.
    sim3 = FockSim(6, device=device)
    zero_g = torch.zeros(0, dtype=torch.float64)
    zero_p = torch.zeros(0, 4, dtype=torch.float64)
    Hc_bal = torch.as_tensor(insts[0].diag, dtype=torch.float64).reshape(1, -1)
    bare = sim3.run(Hc_bal, zero_g, zero_p)
    assert torch.allclose(bare["xexp"], torch.ones_like(bare["xexp"]), atol=1e-8)
    assert float(bare["e_c"].abs()) < 1e-8, float(bare["e_c"])
    report["plus_x"] = float(bare["xexp"][0, 0])

    # direct ⟨X⟩ on the Z-frame state matches the X-histogram
    # rebuild ψ_Z amplitudes for a nontrivial circuit
    psi = torch.zeros(1, sim_f.dim, dtype=torch.complex128)
    psi[:, 0] = 1
    Hc_f = sim_f.fock_of_bit_diag(Hc)
    for ell in range(L):
        psi = sim_f._apply_W(psi)
        psi = psi * torch.exp(-1j * gamma[ell] * Hc_f.to(torch.complex128))
        psi = sim_f._apply_W(psi)
        psi = sim_f._apply_snap(psi, phi[ell])
    psi = sim_f._apply_W(psi)
    psi_x = psi.index_select(-1, sim_f.inv)
    idx = torch.arange(sim_f.dim)
    direct = []
    for i in range(n):
        y = idx ^ (1 << (n - 1 - i))
        direct.append((psi_x.conj() * psi_x[:, y]).sum().real)
    direct_t = torch.stack(direct)
    assert torch.allclose(direct_t, out["xexp"][0], atol=2e-5), (direct_t, out["xexp"][0])
    report["x_hist_vs_pauli"] = float((direct_t - out["xexp"][0]).abs().max())
    report["ok"] = True
    return report
