"""Ansatz — step 0 locked skeleton.

|0⟩ → D(α₀) → [SNAP → D → U_Φ]^L
U_Φ = exp(-i γ_ℓ Φ),  Φ = H_cost  (step 0; later + εΔ)
SNAP+D params shared across cavities; only γ_ℓ changes with layer.
Ban: free U_phase^q / per-cavity private mixers / nonlocal Δ.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from .packing import K_BITS, NU, n_cavities, default_nf
from .gray_grid import build_grid_to_x


@dataclass
class AnsatzConfig:
    n: int
    L: int = 5
    nf: Optional[int] = None
    amax: float = 1.8
    mode: str = "A0"  # A0 exact cost; A1 reserved

    def __post_init__(self):
        self.C = n_cavities(self.n)
        self.nf = int(self.nf or default_nf(self.n))
        assert self.nf >= NU


# --- deprecated helpers (kept for reference; not used in step-0 forward) ---
def q_from_logit(u: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """DEPRECATED: free continuous q. Step 0 uses exact H_cost only."""
    if isinstance(u, torch.Tensor):
        return 1.0 + 9.0 * torch.sigmoid(u)
    return 1.0 + 9.0 / (1.0 + np.exp(-float(u)))


def integer_q(u: Union[float, torch.Tensor]) -> int:
    """DEPRECATED."""
    q = q_from_logit(u)
    if isinstance(q, torch.Tensor):
        q = float(q.detach().cpu())
    return int(np.clip(np.round(q), 1, 10))


def _soft_clip_alpha(re: torch.Tensor, im: torch.Tensor, amax: float):
    r_raw = torch.sqrt(re * re + im * im + 1e-20)
    if amax > 0:
        sc = amax * torch.tanh(r_raw / amax) / r_raw
        re = re * sc
        im = im * sc
        r = torch.sqrt(re * re + im * im + 1e-20)
    else:
        r = r_raw
    ph = torch.atan2(im, re)
    return re, im, r, ph


class MultiCavityAnsatz(torch.nn.Module):
    """Torch multi-cavity Fock simulator: shared SNAP+D + U_Φ (Φ=H_cost)."""

    def __init__(self, cfg: AnsatzConfig, device: Optional[torch.device] = None, dtype=torch.complex64):
        super().__init__()
        self.cfg = cfg
        self.C = cfg.C
        self.nf = int(cfg.nf)
        self.L = cfg.L
        self.amax = float(cfg.amax)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        device = device or torch.device("cpu")

        grid = build_grid_to_x(cfg.n)
        self.register_buffer("grid_to_x", torch.as_tensor(grid, dtype=torch.long, device=device))

        a = torch.diag(torch.sqrt(torch.arange(1, self.nf, dtype=self.rdtype)), 1)
        X = 1j * (a.T.contiguous() - a)
        lam_np, V_np = np.linalg.eigh(X.cpu().numpy().astype(np.complex128))
        self.register_buffer("lam", torch.as_tensor(lam_np.real, dtype=self.rdtype, device=device))
        self.register_buffer("V", torch.as_tensor(V_np, dtype=self.cdtype, device=device))
        self.register_buffer("nv", torch.arange(self.nf, dtype=self.rdtype, device=device))

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def _D(self, re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
        """nf×nf displacement matrix; re, im are 0-dim or scalar tensors."""
        re = re.to(dtype=self.rdtype)
        im = im.to(dtype=self.rdtype)
        _, _, r, ph = _soft_clip_alpha(re, im, self.amax)
        phase_n = torch.exp(1j * ph.to(self.cdtype) * self.nv.to(self.cdtype))
        W = phase_n[:, None] * self.V
        eigs = torch.exp(-1j * r.to(self.cdtype) * self.lam.to(self.cdtype))
        return (W * eigs[None, :]) @ W.conj().T

    def _apply_local(self, psi: torch.Tensor, M: torch.Tensor, cav: int) -> torch.Tensor:
        out = torch.tensordot(M, psi, dims=([1], [cav]))
        return out.movedim(0, cav)

    def vacuum(self) -> torch.Tensor:
        psi = torch.zeros((self.nf,) * self.C, dtype=self.cdtype, device=self.lam.device)
        psi[(0,) * self.C] = 1.0 + 0.0j
        return psi

    def cost_Cn(self, E: torch.Tensor) -> torch.Tensor:
        """Relative cost on Gray product grid, shape (NU,)*C. This is Φ = H_cost."""
        n = self.cfg.n
        m = int(round(9.93 * n))
        E = E.to(device=self.lam.device)
        if E.dtype not in (torch.float32, torch.float64):
            E = E.to(self.rdtype)
        E_grid = E[self.grid_to_x]
        Cn = E_grid * (n / m)
        Cn = Cn - Cn.amax()
        return Cn.to(self.rdtype)

    def forward(
        self,
        E: torch.Tensor,
        alpha0: torch.Tensor,
        A: torch.Tensor,
        TH: torch.Tensor,
        gamma: torch.Tensor,
    ) -> dict:
        """Forward one instance.

        Shapes (step 0 shared mixer):
          alpha0: (2,) shared — OR (C, 2) tiled (both accepted)
          A:      (L, 2) shared displacement per layer
          TH:     (L, NU) shared SNAP phases per layer
          gamma:  (L,) one γ_ℓ per layer
        """
        C, L, nf, NU_ = self.C, self.L, self.nf, NU
        # accept shared (2,) or per-cavity (C,2) for alpha0
        if tuple(alpha0.shape) == (2,):
            alpha0 = alpha0.unsqueeze(0).expand(C, 2)
        assert tuple(alpha0.shape) == (C, 2), f"alpha0 shape {tuple(alpha0.shape)}"
        assert tuple(A.shape) == (L, 2), f"A shape {tuple(A.shape)} want ({L}, 2)"
        assert tuple(TH.shape) == (L, NU_), f"TH shape {tuple(TH.shape)} want ({L}, {NU_})"
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.full((L,), float(gamma), dtype=self.rdtype, device=self.lam.device)
        else:
            gamma = gamma.to(dtype=self.rdtype, device=self.lam.device)
            if gamma.ndim == 0:
                gamma = gamma.expand(L)
        assert tuple(gamma.shape) == (L,), f"gamma shape {tuple(gamma.shape)} want ({L},)"

        psi = self.vacuum()
        # α₀: shared path tiles (2,)→(C,2) so all cavities get the same D
        for c in range(C):
            psi = self._apply_local(psi, self._D(alpha0[c, 0], alpha0[c, 1]), c)

        Cn = self.cost_Cn(E)  # Φ = H_cost on Gray product grid

        for ell in range(L):
            # shared SNAP → every cavity
            ph = torch.zeros(nf, dtype=self.rdtype, device=self.lam.device)
            ph = ph.clone()
            ph[:NU_] = TH[ell].to(self.rdtype)
            snap = torch.diag(torch.exp(-1j * ph.to(self.cdtype)))
            for c in range(C):
                psi = self._apply_local(psi, snap, c)
            # shared D → every cavity
            Dell = self._D(A[ell, 0], A[ell, 1])
            for c in range(C):
                psi = self._apply_local(psi, Dell, c)
            # U_Φ = exp(-i γ_ℓ Φ), Φ = H_cost
            phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
            sl = tuple(slice(0, NU_) for _ in range(C))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new

        probs = psi.real.square() + psi.imag.square()
        comp_sl = tuple(slice(0, NU_) for _ in range(C))
        p_comp = probs[comp_sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        E_grid = E.to(device=self.lam.device)[self.grid_to_x]
        succ = p_comp[E_grid == 0].sum()
        p_succ = succ.clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        return {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": inv_p,
            "norm": probs.sum(),
            "psi": psi,
            "freedom_step": 0,
        }


def random_params(cfg: AnsatzConfig, rng: np.random.Generator | None = None, device=None) -> dict:
    """Shared-mixer random params (step 0 shapes)."""
    rng = rng or np.random.default_rng(0)
    L = cfg.L
    device = device or torch.device("cpu")
    return {
        "alpha0": torch.as_tensor(rng.normal(0, 0.3, size=(2,)), dtype=torch.float32, device=device),
        "A": torch.as_tensor(rng.normal(0, 0.3, size=(L, 2)), dtype=torch.float32, device=device),
        "TH": torch.as_tensor(rng.normal(0, 0.3, size=(L, NU)), dtype=torch.float32, device=device),
        "gamma": torch.as_tensor(rng.normal(0, 0.3, size=(L,)), dtype=torch.float32, device=device),
    }
