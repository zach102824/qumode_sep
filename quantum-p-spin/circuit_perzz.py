"""Per-triple (per-ZZZ) trainable U_c phases on the Nov_3rd Hamming-shell stack.

Baseline U_c^X = W D W with D = exp(-i γ_ℓ H_c), H_c = -s ∑_t J_t ZZZ_t.
Here each triple t gets its own γ_{ℓ,t} (or a factored form).

Recipes (term construction):
  xj   : phase angle = ∑_t γ_{ℓ,t} * (-s J_t ZZZ_t)   # γ_t≡γ_ℓ recovers baseline
  free : phase angle = ∑_t γ_{ℓ,t} * ZZZ_t             # ignores |J|

Paramizations of γ_{ℓ,t}:
  full     : free (L, T)
  factor1p : γ_ℓ * (1 + ε_t)     ε shared across layers
  factorth : γ_ℓ * tanh(ε_t)     ε shared across layers
  sharede  : γ_ℓ * (1 + ε_t)     alias of factor1p (Zach recipe D)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from circuit_snap import B_SHELL, make_ansatz
from circuit_fourier import shell_phi_from_betas
from explore_OCT29.lib.packing import NU
from pspin import scale_p3

__all__ = [
    "B_SHELL",
    "make_ansatz",
    "n_triples",
    "zzz_bitstrings",
    "build_terms",
    "terms_on_grid",
    "forward_state_perzz",
    "fidelity_perzz",
    "expand_gamma",
    "init_perzz",
    "n_params_perzz",
    "selfcheck_recover_baseline",
]


def n_triples(n: int) -> int:
    return n * (n - 1) * (n - 2) // 6


def zzz_bitstrings(n: int) -> np.ndarray:
    """ZZZ eigenvalues for every triple on every bitstring. Shape (T, 2**n)."""
    triples = list(combinations(range(n), 3))
    T = len(triples)
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    out = np.empty((T, dim), dtype=np.float64)
    for ti, (i, j, k) in enumerate(triples):
        def z(bit):
            shift = n - 1 - bit
            return np.where(((x >> shift) & 1) == 0, 1.0, -1.0)
        out[ti] = z(i) * z(j) * z(k)
    return out


def build_terms(n: int, J: np.ndarray, kind: str) -> np.ndarray:
    """Per-triple diagonal contributions on bitstrings. Shape (T, 2**n).

    kind='xj'  → -s J_t ZZZ_t  (sum_t = H_c)
    kind='free'→ ZZZ_t
    """
    zzz = zzz_bitstrings(n)
    if kind == "xj":
        s = scale_p3(n)
        return (-s * J.astype(np.float64))[:, None] * zzz
    if kind == "free":
        return zzz
    raise ValueError(kind)


def terms_on_grid(sim, terms_bit: torch.Tensor) -> torch.Tensor:
    """Map (T, 2**n) → (T, *grid) via sim.grid_to_x."""
    gtx = sim.grid_to_x  # (*grid,) long
    return terms_bit[:, gtx]  # (T, *grid)


def expand_gamma(recipe: str, raw: dict, L: int, T: int) -> torch.Tensor:
    """Return γ of shape (L, T) from recipe-specific raw params."""
    if recipe in ("full_xj", "full_free", "full_xj_snap", "full_free_snap", "full_xj_cold", "full_free_cold", "full_xj_snap_warm", "full_xj_snap_adam"):
        g = raw["gamma_full"]
        if tuple(g.shape) != (L, T):
            raise ValueError(f"gamma_full {tuple(g.shape)} != {(L, T)}")
        return g
    if recipe in (
        "factor1p_xj", "factorth_xj", "sharede_xj",
        "factor1p_xj_snap", "factorth_xj_snap", "sharede_xj_snap",
    ):
        gL = raw["gamma_L"].reshape(L, 1)  # (L,1)
        eps = raw["eps"].reshape(1, T)     # (1,T)
        if "factorth" in recipe:
            return gL * torch.tanh(eps)
        return gL * (1.0 + eps)
    if recipe in ("lowrank_xj", "lowrank_xj_snap"):
        # γ_{ℓ,t} = (A @ B^T)_{ℓ,t}  A:(L,R) B:(T,R)
        return raw["A"] @ raw["B"].T
    raise ValueError(recipe)


def forward_state_perzz(
    sim,
    terms_grid: torch.Tensor,
    gamma_LT: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Z-basis statevector. terms_grid (T,*grid), gamma_LT (L,T), phi (L,4)."""
    L = sim.L
    device = sim.lam.device
    gamma_LT = gamma_LT.to(dtype=sim.rdtype, device=device)
    phi = phi.to(dtype=sim.rdtype, device=device)
    if tuple(phi.shape) != (L, B_SHELL):
        raise ValueError(f"phi shape {tuple(phi.shape)} != {(L, B_SHELL)}")
    T = terms_grid.shape[0]
    if tuple(gamma_LT.shape) != (L, T):
        raise ValueError(f"gamma_LT {tuple(gamma_LT.shape)} != {(L, T)}")

    terms_flat = terms_grid.reshape(T, -1).to(dtype=sim.rdtype, device=device)  # (T, G)
    psi = sim.vacuum()
    sl = tuple(slice(0, NU) for _ in range(sim.C))

    for ell in range(L):
        psi = sim._apply_all(psi, sim.Wmat)
        Cn = (gamma_LT[ell] @ terms_flat).reshape(terms_grid.shape[1:])
        Cn = Cn - Cn.mean()
        phase = torch.exp(-1j * Cn.to(sim.cdtype))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        psi = psi_new
        psi = sim._apply_all(psi, sim.Wmat)
        th = phi[ell][sim.popcount]
        psi = sim._apply_all(psi, sim._snap_theta(th))

    psi = sim._apply_all(psi, sim.Wmat)
    n = sim.cfg.n
    dim = 1 << n
    amps_grid = psi[sl].reshape(-1)
    x_flat = sim.grid_to_x.reshape(-1)
    out = torch.zeros(dim, dtype=sim.cdtype, device=device)
    out[x_flat] = amps_grid
    nrm = torch.linalg.vector_norm(out)
    out = out / nrm.clamp(min=1e-12)
    return out


def fidelity_perzz(sim, terms_grid, gs, gamma_LT, phi) -> torch.Tensor:
    psi = forward_state_perzz(sim, terms_grid, gamma_LT, phi)
    ov = torch.dot(gs.conj().to(psi.dtype), psi)
    return ov.real.square() + ov.imag.square()


def n_params_perzz(recipe: str, L: int, T: int, rank: int = 4) -> int:
    snap_qaoa = L          # β
    snap_free = 4 * L      # φ_0..3
    free_snap = ("_snap" in recipe)
    snap = snap_free if free_snap else snap_qaoa
    # strip suffixes for family match
    base = recipe
    for suf in ("_snap_warm", "_snap_adam", "_snap", "_cold", "_warm", "_adam"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if base in ("full_xj", "full_free"):
        return L * T + snap
    if base in ("factor1p_xj", "factorth_xj", "sharede_xj"):
        return L + T + snap
    if base == "lowrank_xj":
        return L * rank + T * rank + snap
    raise ValueError(recipe)


def init_perzz(recipe: str, L: int, T: int, seed: int = 0, rank: int = 4) -> dict:
    torch.manual_seed(seed)
    out: dict = {}
    base = recipe
    for suf in ("_snap_warm", "_snap_adam", "_snap", "_cold", "_warm", "_adam"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if base in ("full_xj", "full_free"):
        if "cold" in recipe:
            out["gamma_full"] = 0.01 * torch.randn(L, T)
        else:
            out["gamma_full"] = 0.01 + 0.01 * torch.randn(L, T)
    elif base in ("factor1p_xj", "sharede_xj"):
        out["gamma_L"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
        out["eps"] = 0.01 * torch.randn(T)
    elif base == "factorth_xj":
        out["gamma_L"] = torch.full((L,), 0.05) + 0.001 * torch.randn(L)
        out["eps"] = 0.2 * torch.randn(T)
    elif base == "lowrank_xj":
        out["A"] = 0.01 * torch.randn(L, rank)
        out["B"] = 0.01 * torch.randn(T, rank)
    else:
        raise ValueError(recipe)

    if "_snap" in recipe:
        pcs = torch.arange(B_SHELL, dtype=torch.float32)
        phi = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        phi = phi + 0.001 * torch.randn(L, B_SHELL)
        out["phi"] = phi
    else:
        beta = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
        out["beta"] = beta
    return out


def term_kind(recipe: str) -> str:
    if "free" in recipe:
        return "free"
    return "xj"


def uses_free_snap(recipe: str) -> bool:
    return "_snap" in recipe


def selfcheck_recover_baseline(n: int = 6, L: int = 2, seed: int = 0) -> None:
    """When γ_{ℓ,t}=γ_ℓ (xj), forward must match circuit_snap.forward_state."""
    from pspin import make_instance, hc_torch, gs_torch
    from circuit_snap import forward_state, init_angles

    inst = make_instance(n, 0.5, seed=seed)
    device = torch.device("cpu")
    sim = make_ansatz(n, L, device)
    g, phi = init_angles(L, seed=seed)
    Hc = hc_torch(inst, device)
    gs = gs_torch(inst, device)

    terms = torch.as_tensor(build_terms(n, inst.J, "xj"), dtype=torch.float32)
    tgrid = terms_on_grid(sim, terms)
    # constant γ across triples
    gLT = g.reshape(L, 1).expand(L, terms.shape[0]).contiguous()

    psi_base = forward_state(sim, Hc, g, phi)
    psi_pz = forward_state_perzz(sim, tgrid, gLT, phi)
    ov = torch.dot(psi_base.conj(), psi_pz)
    Fov = float((ov.real.square() + ov.imag.square()).clamp(max=1.0))
    # also fidelity vs GS should match
    from circuit_snap import fidelity as fid_base
    Fb = float(fid_base(sim, Hc, gs, g, phi))
    Fp = float(fidelity_perzz(sim, tgrid, gs, gLT, phi))
    print(f"selfcheck overlap^2={Fov:.10f} F_base={Fb:.6f} F_perzz={Fp:.6f}")
    assert Fov > 1.0 - 1e-5, Fov
    assert abs(Fb - Fp) < 1e-5, (Fb, Fp)
    print("PERZZ SELFCHECK OK")
