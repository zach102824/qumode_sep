"""Non-diagonal U_c recipes on Nov_3rd Hamming-shell stack.

Baseline (diagonal D only): U_c^X = W D(Hc) W, then SNAP.
Per-ZZZ reweighting of D collapsed to QAOA (WHY_NOT_BEAT_QAOA.md).

Champ dbl_hc_snap: W Dγ W · SNAP1 · W Dδ W · SNAP2  (same Hc twice, SNAP between)

Push / ablations toward stretch ≥0.55:
  tri_hc_snap       : γ/δ/ε + 3 free SNAPs
  dbl_hc_qaoa_snap  : same sandwich but SNAP φ_k = -β·k (QAOA mixer form)
  dbl_hc_nomid      : no mid-SNAP → collapses to single D_{γ+δ} (ablation)
  dbl_hc_delta0     : δ≡0, keep mid+final SNAP (ablation: is 2nd Hc needed?)
  dbl_hc_onesnap    : mid-SNAP only, no final SNAP2 (ablation: one vs two SNAPs)
  dbl_hc_snap       : original champ (free φ)
"""
from __future__ import annotations

import numpy as np
import torch

from explore_OCT29.lib.packing import NU
from circuit_snap import B_SHELL, make_ansatz, hc_on_grid
from circuit_fourier import shell_phi_from_betas
from pspin import scale_p3

__all__ = [
    "B_SHELL",
    "make_ansatz",
    "ND_RECIPES",
    "n_params_nd",
    "init_nd",
    "warm_from_qaoa",
    "forward_state_nd",
    "fidelity_nd",
    "uses_free_snap",
    "selfcheck_reduce_baseline",
    "N_GIVENS_PLANES",
]

N_GIVENS_PLANES = 4  # pairs (0,1),(2,3),(4,5),(6,7) on NU=8
ND_RECIPES = (
    "dbl_hc_snap",
    "tri_hc_snap",
    "tri_hc_qaoa_snap",
    "dbl_hc_qaoa_snap",
    "dbl_hc_nomid",
    "dbl_hc_delta0",
    "dbl_hc_onesnap",
    "dbl_shift_snap",
    "dbl_mag_snap",
    "givens_inner_snap",
    "givens_after_snap",
    "xx_match_snap",
    "xx_path_snap",
)


def uses_free_snap(recipe: str) -> bool:
    if recipe in ("dbl_hc_qaoa_snap", "tri_hc_qaoa_snap"):
        return False
    return "_snap" in recipe or recipe in (
        "dbl_hc_nomid",
        "dbl_hc_delta0",
        "dbl_hc_onesnap",
        "tri_hc_snap",
    )


def _base(recipe: str) -> str:
    base = recipe
    for suf in ("_snap_warm", "_snap_adam", "_snap", "_warm", "_adam"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    # keep special ablation names intact
    if recipe in ("dbl_hc_nomid", "dbl_hc_delta0", "dbl_hc_onesnap", "dbl_hc_qaoa_snap", "tri_hc_qaoa_snap", "tri_hc_snap"):
        return recipe
    return base


def mag_z_diag(n: int) -> np.ndarray:
    """∑_i Z_i on bitstrings (var0=MSB)."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    out = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        shift = n - 1 - i
        out += np.where(((x >> shift) & 1) == 0, 1.0, -1.0)
    return out


def shift_hc(Hc: np.ndarray) -> np.ndarray:
    """Hc(x ⊕ (2^n-1)) — all-bit complement."""
    dim = Hc.shape[0]
    mask = dim - 1
    idx = np.arange(dim) ^ mask
    return Hc[idx]


def matching_edges(n: int) -> list[tuple[int, int]]:
    """Perfect-ish matching: (0,1),(2,3),..."""
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def path_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def n_xx_edges(recipe: str, n: int) -> int:
    base = _base(recipe)
    if base == "xx_match":
        return len(matching_edges(n))
    if base == "xx_path":
        return len(path_edges(n))
    return 0


def n_params_nd(recipe: str, L: int, n: int = 9) -> int:
    base = _base(recipe)
    snap4 = 4 * L
    if base == "dbl_hc":
        # γ, δ, phi1, phi2
        return 2 * L + 2 * snap4
    if base == "tri_hc_snap":
        # γ, δ, ε, phi1,2,3
        return 3 * L + 3 * snap4
    if base == "dbl_hc_qaoa_snap":
        # γ, δ, β1, β2
        return 4 * L
    if base == "tri_hc_qaoa_snap":
        # γ, δ, ε, β1, β2, β3
        return 6 * L
    if base == "dbl_hc_nomid":
        # γ, δ, phi (final only) — theoretically collapses to single D
        return 2 * L + snap4
    if base == "dbl_hc_delta0":
        # γ, phi1, phi2 (δ≡0)
        return L + 2 * snap4
    if base == "dbl_hc_onesnap":
        # γ, δ, phi1 (mid only)
        return 2 * L + snap4
    if base in ("dbl_shift", "dbl_mag"):
        return 2 * L + 2 * snap4
    if base in ("givens_inner", "givens_after"):
        return L + L * N_GIVENS_PLANES + snap4
    if base in ("xx_match", "xx_path"):
        E = n_xx_edges(recipe, n)
        return L + L * E + snap4
    raise ValueError(recipe)


def init_nd(recipe: str, L: int, n: int = 9, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    out: dict = {}
    base = _base(recipe)
    out["gamma"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
    pcs = torch.arange(B_SHELL, dtype=torch.float32)

    if base == "dbl_hc":
        out["delta"] = 0.001 * torch.randn(L)
        out["phi1"] = 0.001 * torch.randn(L, B_SHELL)
        out["phi2"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi2"] = out["phi2"] + 0.001 * torch.randn(L, B_SHELL)
    elif base == "tri_hc_snap":
        out["delta"] = 0.001 * torch.randn(L)
        out["eps"] = 0.001 * torch.randn(L)
        out["phi1"] = 0.001 * torch.randn(L, B_SHELL)
        out["phi2"] = 0.001 * torch.randn(L, B_SHELL)
        out["phi3"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi3"] = out["phi3"] + 0.001 * torch.randn(L, B_SHELL)
    elif base == "dbl_hc_qaoa_snap":
        out["delta"] = 0.001 * torch.randn(L)
        out["beta1"] = 0.001 * torch.randn(L)  # mid SNAP ~ I
        out["beta2"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
    elif base == "tri_hc_qaoa_snap":
        out["delta"] = 0.001 * torch.randn(L)
        out["eps"] = 0.001 * torch.randn(L)
        out["beta1"] = 0.001 * torch.randn(L)
        out["beta2"] = 0.001 * torch.randn(L)
        out["beta3"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
    elif base == "dbl_hc_nomid":
        out["delta"] = 0.001 * torch.randn(L)
        out["phi"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi"] = out["phi"] + 0.001 * torch.randn(L, B_SHELL)
    elif base == "dbl_hc_delta0":
        # no delta param
        out["phi1"] = 0.001 * torch.randn(L, B_SHELL)
        out["phi2"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi2"] = out["phi2"] + 0.001 * torch.randn(L, B_SHELL)
    elif base == "dbl_hc_onesnap":
        out["delta"] = 0.001 * torch.randn(L)
        out["phi1"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi1"] = out["phi1"] + 0.001 * torch.randn(L, B_SHELL)
    elif base in ("dbl_shift", "dbl_mag"):
        out["delta"] = 0.001 * torch.randn(L)
        out["phi1"] = 0.001 * torch.randn(L, B_SHELL)
        out["phi2"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi2"] = out["phi2"] + 0.001 * torch.randn(L, B_SHELL)
    elif base in ("givens_inner", "givens_after"):
        out["theta"] = 0.001 * torch.randn(L, N_GIVENS_PLANES)
        out["phi"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi"] = out["phi"] + 0.001 * torch.randn(L, B_SHELL)
    elif base in ("xx_match", "xx_path"):
        E = n_xx_edges(recipe, n)
        out["theta_xx"] = 0.001 * torch.randn(L, E)
        out["phi"] = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi"] = out["phi"] + 0.001 * torch.randn(L, B_SHELL)
    else:
        raise ValueError(recipe)
    return out


def warm_from_qaoa(recipe: str, raw: dict, L: int, ckpt: dict) -> dict:
    g = ckpt["gamma"].float().reshape(-1)
    if g.numel() != L:
        if g.numel() < L:
            g = g.repeat((L + g.numel() - 1) // g.numel())[:L]
        else:
            g = g[:L]
    raw["gamma"] = g.clone() + 0.001 * torch.randn(L)
    base = _base(recipe)
    if "delta" in raw:
        raw["delta"] = 0.001 * torch.randn(L)
    if "eps" in raw:
        raw["eps"] = 0.001 * torch.randn(L)

    b = None
    if "beta" in ckpt:
        b = ckpt["beta"].float().reshape(-1)
        if b.numel() < L:
            b = b.repeat((L + b.numel() - 1) // b.numel())[:L]
        else:
            b = b[:L]
        phi_q = shell_phi_from_betas(b)
        if "phi2" in raw:
            raw["phi2"] = phi_q + 0.001 * torch.randn_like(phi_q)
            if "phi1" in raw:
                raw["phi1"] = 0.001 * torch.randn_like(phi_q)
        if "phi3" in raw:
            raw["phi3"] = phi_q + 0.001 * torch.randn_like(phi_q)
            raw["phi1"] = 0.001 * torch.randn_like(phi_q)
            raw["phi2"] = 0.001 * torch.randn_like(phi_q)
        if "phi" in raw:
            raw["phi"] = phi_q + 0.001 * torch.randn_like(phi_q)
        if "beta2" in raw:
            raw["beta2"] = b.clone() + 0.001 * torch.randn(L)
            raw["beta1"] = 0.001 * torch.randn(L)
        if "beta3" in raw:
            raw["beta3"] = b.clone() + 0.001 * torch.randn(L)
            raw["beta1"] = 0.001 * torch.randn(L)
            raw["beta2"] = 0.001 * torch.randn(L)
        if "phi1" in raw and "phi2" not in raw and "phi3" not in raw:
            # onesnap: mid SNAP gets QAOA warm
            raw["phi1"] = phi_q + 0.001 * torch.randn_like(phi_q)
    if "theta" in raw:
        raw["theta"] = 0.001 * torch.randn_like(raw["theta"])
    if "theta_xx" in raw:
        raw["theta_xx"] = 0.001 * torch.randn_like(raw["theta_xx"])
    return raw


def _givens_mat(theta: torch.Tensor, device, dtype, cdtype) -> torch.Tensor:
    """NU×NU real Givens product on pairs (0,1),(2,3),(4,5),(6,7). theta:(P,)."""
    P = N_GIVENS_PLANES
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    G = torch.eye(NU, dtype=cdtype, device=device)
    th = theta.to(dtype=dtype, device=device).reshape(P)
    for p, (a, b) in enumerate(pairs):
        c = torch.cos(th[p]).to(cdtype)
        s = torch.sin(th[p]).to(cdtype)
        Gp = torch.eye(NU, dtype=cdtype, device=device)
        Gp[a, a] = c
        Gp[b, b] = c
        Gp[a, b] = s
        Gp[b, a] = -s
        G = Gp @ G
    return G


def _apply_phase_diag(sim, psi, Cn_grid, scale: torch.Tensor):
    """psi ← diag(exp(-i scale * Cn)) psi on computational grid."""
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    phase = torch.exp(-1j * (scale * Cn_grid).to(sim.cdtype))
    psi_new = psi.clone()
    psi_new[sl] = psi[sl] * phase
    return psi_new


def _apply_snap(sim, psi, phi_row: torch.Tensor):
    th = phi_row[sim.popcount]
    return sim._apply_all(psi, sim._snap_theta(th))


def _bitstring_from_grid(sim, psi) -> torch.Tensor:
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    n = sim.cfg.n
    dim = 1 << n
    amps = psi[sl].reshape(-1)
    x = sim.grid_to_x.reshape(-1)
    out = torch.zeros(dim, dtype=sim.cdtype, device=sim.lam.device)
    out[x] = amps
    return out


def _grid_from_bitstring(sim, full: torch.Tensor, psi_template):
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    x = sim.grid_to_x.reshape(-1)
    psi = psi_template.clone()
    psi[sl] = full[x].reshape(psi[sl].shape)
    return psi


def _apply_xx_edges(full: torch.Tensor, edges: list[tuple[int, int]], thetas: torch.Tensor, n: int):
    """Apply ∏_e exp(-i θ_e XX_e) on Z-basis statevector (var0=MSB)."""
    dim = full.shape[0]
    idx = torch.arange(dim, device=full.device)
    out = full
    th = thetas.reshape(-1)
    for e, (i, j) in enumerate(edges):
        si = n - 1 - i
        sj = n - 1 - j
        partner = idx ^ (1 << si) ^ (1 << sj)
        c = torch.cos(th[e]).to(full.dtype)
        s = torch.sin(th[e]).to(full.dtype)
        a = out
        b = out[partner]
        out = c * a - 1j * s * b
    return out


def forward_state_nd(
    sim,
    Hc_diag: torch.Tensor,
    raw: dict,
    recipe: str,
    *,
    Hc2_diag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Z-basis statevector for a non-diagonal recipe."""
    L = sim.L
    n = sim.cfg.n
    device = sim.lam.device
    base = _base(recipe)
    gamma = raw["gamma"].to(dtype=sim.rdtype, device=device).reshape(L)

    Cn = hc_on_grid(sim, Hc_diag)
    Cn = Cn - Cn.mean()
    if Hc2_diag is not None:
        Cn2 = hc_on_grid(sim, Hc2_diag)
        Cn2 = Cn2 - Cn2.mean()
    else:
        Cn2 = None

    psi = sim.vacuum()
    sl = tuple(slice(0, NU) for _ in range(sim.C))

    for ell in range(L):
        if base == "dbl_hc":
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi1 = raw["phi1"].to(dtype=sim.rdtype, device=device)
            phi2 = raw["phi2"].to(dtype=sim.rdtype, device=device)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi2[ell])

        elif base == "tri_hc_snap":
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            eps = raw["eps"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi1 = raw["phi1"].to(dtype=sim.rdtype, device=device)
            phi2 = raw["phi2"].to(dtype=sim.rdtype, device=device)
            phi3 = raw["phi3"].to(dtype=sim.rdtype, device=device)
            # W Dγ W · SNAP1 · W Dδ W · SNAP2 · W Dε W · SNAP3
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi2[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, eps[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi3[ell])

        elif base == "dbl_hc_qaoa_snap":
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            beta1 = raw["beta1"].to(dtype=sim.rdtype, device=device).reshape(L)
            beta2 = raw["beta2"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi1 = shell_phi_from_betas(beta1)
            phi2 = shell_phi_from_betas(beta2)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi2[ell])

        elif base == "tri_hc_qaoa_snap":
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            eps = raw["eps"].to(dtype=sim.rdtype, device=device).reshape(L)
            beta1 = raw["beta1"].to(dtype=sim.rdtype, device=device).reshape(L)
            beta2 = raw["beta2"].to(dtype=sim.rdtype, device=device).reshape(L)
            beta3 = raw["beta3"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi1 = shell_phi_from_betas(beta1)
            phi2 = shell_phi_from_betas(beta2)
            phi3 = shell_phi_from_betas(beta3)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi2[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, eps[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi3[ell])

        elif base == "dbl_hc_nomid":
            # W Dγ W · W Dδ W · SNAP  → collapses to W D_{γ+δ} W · SNAP
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi = raw["phi"].to(dtype=sim.rdtype, device=device)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi[ell])

        elif base == "dbl_hc_delta0":
            # W Dγ W · SNAP1 · SNAP2  (δ≡0)
            phi1 = raw["phi1"].to(dtype=sim.rdtype, device=device)
            phi2 = raw["phi2"].to(dtype=sim.rdtype, device=device)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = _apply_snap(sim, psi, phi2[ell])

        elif base == "dbl_hc_onesnap":
            # W Dγ W · SNAP1 · W Dδ W   (no final SNAP)
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi1 = raw["phi1"].to(dtype=sim.rdtype, device=device)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)

        elif base in ("dbl_shift", "dbl_mag"):
            delta = raw["delta"].to(dtype=sim.rdtype, device=device).reshape(L)
            phi1 = raw["phi1"].to(dtype=sim.rdtype, device=device)
            phi2 = raw["phi2"].to(dtype=sim.rdtype, device=device)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi1[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            C2 = Cn2
            psi = _apply_phase_diag(sim, psi, C2, delta[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi2[ell])

        elif base == "givens_inner":
            phi = raw["phi"].to(dtype=sim.rdtype, device=device)
            th = raw["theta"].to(dtype=sim.rdtype, device=device)
            G = _givens_mat(th[ell], device, sim.rdtype, sim.cdtype)
            Gd = G.mH
            psi = sim._apply_all(psi, sim.Wmat)
            psi = sim._apply_all(psi, Gd)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, G)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi[ell])

        elif base == "givens_after":
            phi = raw["phi"].to(dtype=sim.rdtype, device=device)
            th = raw["theta"].to(dtype=sim.rdtype, device=device)
            G = _givens_mat(th[ell], device, sim.rdtype, sim.cdtype)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, G)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_snap(sim, psi, phi[ell])

        elif base in ("xx_match", "xx_path"):
            phi = raw["phi"].to(dtype=sim.rdtype, device=device)
            thx = raw["theta_xx"].to(dtype=sim.rdtype, device=device)
            edges = matching_edges(n) if base == "xx_match" else path_edges(n)
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase_diag(sim, psi, Cn, gamma[ell])
            psi = sim._apply_all(psi, sim.Wmat)
            full = _bitstring_from_grid(sim, psi)
            full = _apply_xx_edges(full, edges, thx[ell], n)
            psi = _grid_from_bitstring(sim, full, psi)
            psi = _apply_snap(sim, psi, phi[ell])
        else:
            raise ValueError(recipe)

    psi = sim._apply_all(psi, sim.Wmat)
    dim = 1 << n
    amps_grid = psi[sl].reshape(-1)
    x_flat = sim.grid_to_x.reshape(-1)
    out = torch.zeros(dim, dtype=sim.cdtype, device=device)
    out[x_flat] = amps_grid
    nrm = torch.linalg.vector_norm(out)
    out = out / nrm.clamp(min=1e-12)
    return out


def fidelity_nd(sim, Hc_diag, gs, raw, recipe, Hc2_diag=None) -> torch.Tensor:
    psi = forward_state_nd(sim, Hc_diag, raw, recipe, Hc2_diag=Hc2_diag)
    ov = torch.dot(gs.conj().to(psi.dtype), psi)
    return ov.real.square() + ov.imag.square()


def pack_hc2(inst, recipe: str, device) -> torch.Tensor | None:
    base = _base(recipe)
    if base == "dbl_shift":
        arr = shift_hc(inst.Hc_diag)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    if base == "dbl_mag":
        arr = mag_z_diag(inst.n)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    return None


def selfcheck_reduce_baseline(n: int = 6, L: int = 2, seed: int = 0) -> None:
    """δ=0, phi1=0, G=I, θ_xx=0 → overlap with circuit_snap baseline ≈ 1."""
    from pspin import make_instance, hc_torch, gs_torch
    from circuit_snap import forward_state, init_angles

    inst = make_instance(n, 0.5, seed=seed)
    device = torch.device("cpu")
    sim = make_ansatz(n, L, device)
    g, phi = init_angles(L, seed=seed)
    Hc = hc_torch(inst, device)
    gs = gs_torch(inst, device)
    psi_base = forward_state(sim, Hc, g, phi)

    def ov2(a, b):
        o = torch.dot(a.conj(), b)
        return float((o.real.square() + o.imag.square()).clamp(max=1.0))

    raw = {
        "gamma": g.clone(),
        "delta": torch.zeros(L),
        "phi1": torch.zeros(L, B_SHELL),
        "phi2": phi.clone(),
    }
    psi = forward_state_nd(sim, Hc, raw, "dbl_hc_snap")
    f = ov2(psi_base, psi)
    print(f"dbl_hc reduce overlap^2={f:.10f}")
    assert f > 1.0 - 1e-5, f

    # delta0 with phi1=0, phi2=phi → baseline
    raw_d0 = {"gamma": g.clone(), "phi1": torch.zeros(L, B_SHELL), "phi2": phi.clone()}
    psi = forward_state_nd(sim, Hc, raw_d0, "dbl_hc_delta0")
    f = ov2(psi_base, psi)
    print(f"dbl_hc_delta0 reduce overlap^2={f:.10f}")
    assert f > 1.0 - 1e-5, f

    # nomid with delta=0 → baseline
    raw_nm = {"gamma": g.clone(), "delta": torch.zeros(L), "phi": phi.clone()}
    psi = forward_state_nd(sim, Hc, raw_nm, "dbl_hc_nomid")
    f = ov2(psi_base, psi)
    print(f"dbl_hc_nomid reduce overlap^2={f:.10f}")
    assert f > 1.0 - 1e-5, f

    # qaoa with delta=0, beta1=0, beta2 from phi=-βk → need matching beta
    # skip exact unless phi is Fourier; just check runs
    raw_q = {
        "gamma": g.clone(),
        "delta": torch.zeros(L),
        "beta1": torch.zeros(L),
        "beta2": torch.full((L,), 0.01),
    }
    psi = forward_state_nd(sim, Hc, raw_q, "dbl_hc_qaoa_snap")
    assert psi.shape[0] == (1 << n)

    # tri with δ=ε=0, phi1=phi2=0, phi3=phi → baseline
    raw_t = {
        "gamma": g.clone(),
        "delta": torch.zeros(L),
        "eps": torch.zeros(L),
        "phi1": torch.zeros(L, B_SHELL),
        "phi2": torch.zeros(L, B_SHELL),
        "phi3": phi.clone(),
    }
    psi = forward_state_nd(sim, Hc, raw_t, "tri_hc_snap")
    f = ov2(psi_base, psi)
    print(f"tri_hc reduce overlap^2={f:.10f}")
    assert f > 1.0 - 1e-5, f

    raw_g = {"gamma": g.clone(), "theta": torch.zeros(L, N_GIVENS_PLANES), "phi": phi.clone()}
    psi = forward_state_nd(sim, Hc, raw_g, "givens_inner_snap")
    f = ov2(psi_base, psi)
    print(f"givens_inner reduce overlap^2={f:.10f}")
    assert f > 1.0 - 1e-5, f

    print("NONDIAG SELFCHECK OK")


if __name__ == "__main__":
    selfcheck_reduce_baseline()
