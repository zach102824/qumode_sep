"""Adapt-VQE-style growth from QAOA base on Nov_3rd stack.

Base (locked): U = [W D(γ H_c) W · SNAP(φ_k=-β k)]^L · final W
  → 2L params (β, γ).

Pool candidates = single-parameter-family unitaries appended per layer
(after QAOA mixer) or as a global trailer. Screening uses |∂F/∂θ| at θ=0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from explore_OCT29.lib.packing import NU
from circuit_snap import B_SHELL, make_ansatz, hc_on_grid
from circuit_fourier import shell_phi_from_betas
from circuit_nondiag import (
    mag_z_diag,
    shift_hc,
    matching_edges,
    path_edges,
    _givens_mat,
    N_GIVENS_PLANES,
)

__all__ = [
    "POOL",
    "make_ansatz",
    "B_SHELL",
    "init_qaoa",
    "forward_adapt",
    "fidelity_adapt",
    "n_params_adapt",
    "pool_edges",
    "grad_pool_at_zero",
    "OpSpec",
]


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str          # xx_match|xx_path|yy_match|xy_match|x_total|mag_chan|shift_chan|hc2_chan|givens
    n_angles_per_layer: int  # usually 1 (shared across edges)
    desc: str


POOL: tuple[OpSpec, ...] = (
    OpSpec("xx_match", "xx_match", 1, "matching XX edges, shared θ per layer"),
    OpSpec("xx_path", "xx_path", 1, "path XX edges, shared θ per layer"),
    OpSpec("yy_match", "yy_match", 1, "matching YY edges, shared θ per layer"),
    OpSpec("xy_match", "xy_match", 1, "matching (XY+YX)/2, shared θ per layer"),
    OpSpec("x_total", "x_total", 1, "exp(-iθ ∑X_i) Mag-X mixer boost"),
    OpSpec("hc2_chan", "hc2_chan", 1, "2nd X-channel W D(δ Hc) W after mid-SNAP(β_mid=0→train)"),
    OpSpec("mag_chan", "mag_chan", 1, "2nd channel W D(α Mag_Z) W"),
    OpSpec("shift_chan", "shift_chan", 1, "2nd channel W D(δ Hc_shift) W"),
    OpSpec("givens", "givens", 1, "shared Fock Givens angle (all planes)"),
)


def pool_edges(kind: str, n: int) -> list[tuple[int, int]]:
    if kind in ("xx_match", "yy_match", "xy_match"):
        return matching_edges(n)
    if kind == "xx_path":
        return path_edges(n)
    return []


def init_qaoa(L: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "beta": torch.full((L,), 0.01) + 0.001 * torch.randn(L),
        "gamma": torch.full((L,), 0.01) + 0.001 * torch.randn(L),
        "adapt_ops": [],          # list of OpSpec.name in append order
        "adapt_theta": torch.zeros(0),  # flat (n_ops * L,) or per-op blocks
    }


def warm_qaoa_from_ckpt(raw: dict, L: int, ckpt: dict) -> dict:
    g = ckpt["gamma"].float().reshape(-1)
    b = ckpt["beta"].float().reshape(-1)
    if g.numel() < L:
        g = g.repeat((L + g.numel() - 1) // g.numel())[:L]
    else:
        g = g[:L]
    if b.numel() < L:
        b = b.repeat((L + b.numel() - 1) // b.numel())[:L]
    else:
        b = b[:L]
    raw["gamma"] = g.clone()
    raw["beta"] = b.clone()
    return raw


def n_params_adapt(raw: dict) -> int:
    L = int(raw["beta"].numel())
    n_ops = len(raw.get("adapt_ops", []))
    return 2 * L + n_ops * L  # each op adds 1 angle / layer


def _bit_phase_y(bit_val: torch.Tensor) -> torch.Tensor:
    """Amplitude factor for Y: Y|0⟩=-i|1⟩, Y|1⟩=+i|0⟩ → factor when leaving state."""
    # bit_val 0 or 1
    return torch.where(bit_val == 0, -1j * torch.ones_like(bit_val, dtype=torch.complex64),
                       1j * torch.ones_like(bit_val, dtype=torch.complex64))


def apply_pauli_prod(
    full: torch.Tensor,
    edges: list[tuple[int, int]],
    theta: torch.Tensor,
    n: int,
    kind: str,
) -> torch.Tensor:
    """Apply ∏_e exp(-i θ Pauli_e) with shared θ on all edges (kind xx|yy|xy)."""
    if abs(float(theta.detach())) < 1e-12 and not theta.requires_grad:
        # still need grad path if requires_grad
        pass
    dim = full.shape[0]
    idx = torch.arange(dim, device=full.device)
    th = theta.to(dtype=full.real.dtype if full.is_complex() else full.dtype)
    # use real dtype for cos/sin
    rdtype = torch.float32 if full.dtype == torch.complex64 else torch.float64
    th = theta.to(dtype=rdtype)
    c = torch.cos(th).to(full.dtype)
    s = torch.sin(th).to(full.dtype)
    out = full
    for i, j in edges:
        si = n - 1 - i
        sj = n - 1 - j
        partner = idx ^ (1 << si) ^ (1 << sj)
        bi = (idx >> si) & 1
        bj = (idx >> sj) & 1
        if kind == "xx":
            # XX real flip
            a = out
            b = out[partner]
            out = c * a - 1j * s * b
        elif kind == "yy":
            # YY: flip with phase σ_i σ_j where σ= -i (bit0) or +i (bit1)
            # phase_from = Yfactor(bi)*Yfactor(bj); applied as out[p] gets contrib from out[src]
            # exp(-iθ YY)= c I - i s YY
            phase = _bit_phase_y(bi) * _bit_phase_y(bj)
            phase = phase.to(full.dtype)
            a = out
            b = phase * out[partner]
            # YY|x⟩ = phase_xy |x⊕ei⊕ej⟩ where phase_xy = Yamp_i Yamp_j
            # But partner state's YY back: need consistent
            # Standard: ⟨x⊕|YY|x⟩ = (−1)^{x_i+x_j} for common convention with Y=[[0,-i],[i,0]]?
            # Use: YY|x⟩ = (Y_i amp)(Y_j amp)|partner⟩
            out = c * a - 1j * s * (phase * out[partner])
        elif kind == "xy":
            # (XY + YX)/2 is Hermitian; XY|x⟩ = X_i Y_j
            # X amp = 1; Y amp as above on j
            # Use Hermitian H = (XY+YX)/2: simpler apply exp(-iθ XX) style with phases
            # XY|x⟩ = Yfactor(bj) |x⊕ei⊕ej⟩ (X on i has factor 1)
            # YX|x⟩ = Yfactor(bi) |partner⟩
            # H|x⟩ = 0.5 (Yfactor(bj)+Yfactor(bi)) |partner⟩ — not always ±1
            # Cleaner: apply exp(-iθ XY) alone (not Hermitian if θ real — use Hermitian XY+YX)
            yj = _bit_phase_y(bj).to(full.dtype)
            yi = _bit_phase_y(bi).to(full.dtype)
            # (XY+YX)|x⟩ = (yj + yi)|partner⟩
            coup = 0.5 * (yi + yj)
            a = out
            b = coup * out[partner]
            out = c * a - 1j * s * b
        else:
            raise ValueError(kind)
    return out


def apply_x_total(full: torch.Tensor, theta: torch.Tensor, n: int) -> torch.Tensor:
    """exp(-i θ ∑_i X_i) = ∏_i exp(-i θ X_i)."""
    dim = full.shape[0]
    idx = torch.arange(dim, device=full.device)
    rdtype = torch.float32 if full.dtype == torch.complex64 else torch.float64
    th = theta.to(dtype=rdtype)
    c = torch.cos(th).to(full.dtype)
    s = torch.sin(th).to(full.dtype)
    out = full
    for i in range(n):
        si = n - 1 - i
        partner = idx ^ (1 << si)
        a = out
        b = out[partner]
        out = c * a - 1j * s * b
    return out


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


def _apply_phase(sim, psi, Cn, scale):
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    phase = torch.exp(-1j * (scale * Cn).to(sim.cdtype))
    psi_new = psi.clone()
    psi_new[sl] = psi[sl] * phase
    return psi_new


def _apply_snap_beta(sim, psi, beta_ell):
    phi_row = shell_phi_from_betas(beta_ell.reshape(1))[0]
    th = phi_row[sim.popcount]
    return sim._apply_all(psi, sim._snap_theta(th))


def forward_adapt(
    sim,
    Hc_diag: torch.Tensor,
    raw: dict,
    *,
    Hc_shift: torch.Tensor | None = None,
    Mag: torch.Tensor | None = None,
    probe: dict | None = None,
) -> torch.Tensor:
    """QAOA base + appended adapt ops.

    probe: optional {name, theta_per_layer:(L,)} screened at append time (not in raw yet).
    """
    L = sim.L
    n = sim.cfg.n
    device = sim.lam.device
    beta = raw["beta"].to(dtype=sim.rdtype, device=device).reshape(L)
    gamma = raw["gamma"].to(dtype=sim.rdtype, device=device).reshape(L)
    adapt_ops: list[str] = list(raw.get("adapt_ops", []))
    # theta layout: (n_ops, L)
    n_ops = len(adapt_ops)
    if n_ops == 0:
        ath = None
    else:
        ath = raw["adapt_theta"].to(dtype=sim.rdtype, device=device).reshape(n_ops, L)

    Cn = hc_on_grid(sim, Hc_diag)
    Cn = Cn - Cn.mean()
    Cn_shift = None
    Cn_mag = None
    if Hc_shift is not None:
        Cn_shift = hc_on_grid(sim, Hc_shift)
        Cn_shift = Cn_shift - Cn_shift.mean()
    if Mag is not None:
        Cn_mag = hc_on_grid(sim, Mag)
        Cn_mag = Cn_mag - Cn_mag.mean()

    psi = sim.vacuum()

    def apply_op(psi, name: str, theta_ell: torch.Tensor):
        kind = name
        # channel ops: W D W with tiny mid identity (no extra SNAP — WDW at δ inserts X-channel)
        if name in ("hc2_chan", "mag_chan", "shift_chan"):
            if name == "hc2_chan":
                C = Cn
            elif name == "mag_chan":
                C = Cn_mag
            else:
                C = Cn_shift
            psi = sim._apply_all(psi, sim.Wmat)
            psi = _apply_phase(sim, psi, C, theta_ell)
            psi = sim._apply_all(psi, sim.Wmat)
            return psi
        if name == "givens":
            # shared angle on all planes
            thvec = theta_ell.reshape(1).expand(N_GIVENS_PLANES).contiguous()
            # build from scalar
            thvec = theta_ell + torch.zeros(N_GIVENS_PLANES, dtype=sim.rdtype, device=device)
            G = _givens_mat(thvec, device, sim.rdtype, sim.cdtype)
            return sim._apply_all(psi, G)
        # bitstring Pauli ops after mapping
        full = _bitstring_from_grid(sim, psi)
        if name == "x_total":
            full = apply_x_total(full, theta_ell, n)
        elif name in ("xx_match", "xx_path"):
            edges = pool_edges(name, n)
            full = apply_pauli_prod(full, edges, theta_ell, n, "xx")
        elif name == "yy_match":
            edges = pool_edges(name, n)
            full = apply_pauli_prod(full, edges, theta_ell, n, "yy")
        elif name == "xy_match":
            edges = pool_edges(name, n)
            full = apply_pauli_prod(full, edges, theta_ell, n, "xy")
        else:
            raise ValueError(name)
        return _grid_from_bitstring(sim, full, psi)

    for ell in range(L):
        # QAOA base layer
        psi = sim._apply_all(psi, sim.Wmat)
        psi = _apply_phase(sim, psi, Cn, gamma[ell])
        psi = sim._apply_all(psi, sim.Wmat)
        psi = _apply_snap_beta(sim, psi, beta[ell])
        # appended ops for this layer (same order)
        for oi, name in enumerate(adapt_ops):
            psi = apply_op(psi, name, ath[oi, ell])
        # probe op (screening) — applied after existing adapt ops
        if probe is not None:
            psi = apply_op(psi, probe["name"], probe["theta"][ell])

    psi = sim._apply_all(psi, sim.Wmat)
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    dim = 1 << n
    amps = psi[sl].reshape(-1)
    x = sim.grid_to_x.reshape(-1)
    out = torch.zeros(dim, dtype=sim.cdtype, device=device)
    out[x] = amps
    nrm = torch.linalg.vector_norm(out)
    return out / nrm.clamp(min=1e-12)


def fidelity_adapt(sim, Hc, gs, raw, Hc_shift=None, Mag=None, probe=None) -> torch.Tensor:
    psi = forward_adapt(sim, Hc, raw, Hc_shift=Hc_shift, Mag=Mag, probe=probe)
    ov = torch.dot(gs.conj().to(psi.dtype), psi)
    return ov.real.square() + ov.imag.square()


def append_op(raw: dict, name: str, L: int) -> dict:
    ops = list(raw.get("adapt_ops", []))
    ops.append(name)
    old = raw.get("adapt_theta", torch.zeros(0))
    if torch.is_tensor(old) and old.numel() > 0:
        prev = old.detach().reshape(-1, L)
        neu = torch.zeros(1, L, dtype=prev.dtype)
        ath = torch.cat([prev, neu], dim=0).reshape(-1)
    else:
        ath = torch.zeros(len(ops) * L)
    raw = dict(raw)
    raw["adapt_ops"] = ops
    raw["adapt_theta"] = ath
    return raw


@torch.no_grad()
def pack_aux(inst, device):
    Hc = torch.as_tensor(inst.Hc_diag, dtype=torch.float32, device=device)
    Hs = torch.as_tensor(shift_hc(inst.Hc_diag), dtype=torch.float32, device=device)
    Mag = torch.as_tensor(mag_z_diag(inst.n), dtype=torch.float32, device=device)
    gs = torch.as_tensor(inst.gs, dtype=torch.complex64, device=device)
    return Hc, gs, Hs, Mag


def grad_pool_at_zero(
    sim,
    raw: dict,
    packed_train: list,
    pool: tuple[OpSpec, ...] = POOL,
    max_inst: int = 16,
) -> list[dict]:
    """Mean |∂F/∂θ̄| at θ=0 for each pool op not already in adapt_ops.

    Uses a single shared θ across layers for screening (θ·1_L), returns |dF/dθ|.
    """
    L = sim.L
    have = set(raw.get("adapt_ops", []))
    rows = []
    insts = packed_train[:max_inst]
    for spec in pool:
        if spec.name in have:
            continue
        # shared scalar θ, broadcast to layers
        theta0 = torch.zeros((), requires_grad=True)
        losses = []
        Fs = []
        for Hc, gs, Hs, Mag in insts:
            probe = {"name": spec.name, "theta": theta0.reshape(()).expand(L) + torch.zeros(L)}
            # expand keeps grad to theta0
            probe = {"name": spec.name, "theta": theta0 + torch.zeros(L, dtype=torch.float32)}
            F = fidelity_adapt(sim, Hc, gs, raw, Hc_shift=Hs, Mag=Mag, probe=probe)
            Fs.append(F)
        Fmean = torch.stack(Fs).mean()
        g = torch.autograd.grad(Fmean, theta0, allow_unused=True)[0]
        gabs = float(g.detach().abs()) if g is not None else 0.0
        rows.append({
            "name": spec.name,
            "kind": spec.kind,
            "grad_abs": gabs,
            "F_at_0": float(Fmean.detach()),
            "desc": spec.desc,
        })
    rows.sort(key=lambda r: -r["grad_abs"])
    return rows
