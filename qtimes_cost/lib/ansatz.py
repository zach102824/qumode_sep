"""Ansatz: |0> -> D(alpha0) -> [SNAP -> D -> U_phase^q]^L  (multi-cavity, K=3).

PyTorch simulator with autograd through SNAP / displacement / cost phase.
A0: U_phase from exact cost Hamiltonian on the product Gray grid.
q in {1..10}: continuous q_tilde = 1 + 9*sigmoid(u).
q≡1 freeze: q_from_logit(..., freeze_q1=True) returns exact 1 and zeroes the logit.
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


Q_MODES = ("learned", "freeze1")


def check_q_mode(q_mode: str) -> str:
    if q_mode not in Q_MODES:
        raise ValueError(f"q_mode must be one of {Q_MODES}, got {q_mode!r}")
    return q_mode


def q_from_logit(
    u: Union[float, torch.Tensor],
    freeze_q1: bool = False,
) -> Union[float, torch.Tensor]:
    """Continuous q in (1, 10), or exact 1.0 when freeze_q1.

    Learned path (default): q = 1 + 9 * sigmoid(u).
    Freeze path: multiply the logit by zero and add 1, so q is exactly 1 and
    the logit contributes nothing (gradient w.r.t. u is identically 0).
    """
    if freeze_q1:
        if isinstance(u, torch.Tensor):
            return u * 0 + 1.0
        return 1.0
    if isinstance(u, torch.Tensor):
        return 1.0 + 9.0 * torch.sigmoid(u)
    return 1.0 + 9.0 / (1.0 + np.exp(-float(u)))


def integer_q(u: Union[float, torch.Tensor], freeze_q1: bool = False) -> int:
    if freeze_q1:
        return 1
    q = q_from_logit(u, freeze_q1=False)
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
    """Torch multi-cavity Fock simulator for D + SNAP + U_phase^q."""

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

        # Displacement eigenbasis of i(a† - a) — fixed, independent of alpha
        a = torch.diag(torch.sqrt(torch.arange(1, self.nf, dtype=self.rdtype)), 1)
        X = 1j * (a.T.contiguous() - a)  # Hermitian
        # Use complex eigh via numpy for stability at init, then buffer
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
        # W = exp(i ph n)[:,None] * V
        phase_n = torch.exp(1j * ph.to(self.cdtype) * self.nv.to(self.cdtype))
        W = phase_n[:, None] * self.V
        eigs = torch.exp(-1j * r.to(self.cdtype) * self.lam.to(self.cdtype))
        return (W * eigs[None, :]) @ W.conj().T

    def _apply_local(self, psi: torch.Tensor, M: torch.Tensor, cav: int) -> torch.Tensor:
        """Apply single-cavity matrix M on axis `cav`."""
        # tensordot M @ psi along cav, then move new axis back
        out = torch.tensordot(M, psi, dims=([1], [cav]))
        # out axes: (nf, ... without cav ...)
        return out.movedim(0, cav)

    def vacuum(self) -> torch.Tensor:
        psi = torch.zeros((self.nf,) * self.C, dtype=self.cdtype, device=self.lam.device)
        psi[(0,) * self.C] = 1.0 + 0.0j
        return psi

    def cost_Cn(self, E: torch.Tensor) -> torch.Tensor:
        """Relative cost on Gray product grid, shape (NU,)*C."""
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
        q_logits: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 1.0,
        q_mode: str = "learned",
    ) -> dict:
        """Forward one instance (all tensors on same device as module).

        q_mode='learned': q = 1+9*sigmoid(logit) per layer.
        q_mode='freeze1': zero the logits, then q_from_logit forces q≡1.
        """
        C, L, nf, NU_ = self.C, self.L, self.nf, NU
        assert tuple(alpha0.shape) == (C, 2)
        assert tuple(A.shape) == (C, L, 2)
        assert tuple(TH.shape) == (C, L, NU_)
        assert tuple(q_logits.shape) == (L,)
        freeze = check_q_mode(q_mode) == "freeze1"
        if freeze:
            # Drop whatever the q head emitted. q_from_logit still forces 1.
            q_logits = torch.zeros_like(q_logits)

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, dtype=self.rdtype, device=self.lam.device)
        else:
            gamma = gamma.to(dtype=self.rdtype, device=self.lam.device)

        psi = self.vacuum()
        for c in range(C):
            psi = self._apply_local(psi, self._D(alpha0[c, 0], alpha0[c, 1]), c)

        Cn = self.cost_Cn(E)  # (NU,)*C real

        q_cont = []
        q_int = []
        for ell in range(L):
            for c in range(C):
                ph = torch.zeros(nf, dtype=self.rdtype, device=self.lam.device)
                ph = ph.clone()
                ph[:NU_] = TH[c, ell].to(self.rdtype)
                snap = torch.diag(torch.exp(-1j * ph.to(self.cdtype)))
                psi = self._apply_local(psi, snap, c)
            for c in range(C):
                psi = self._apply_local(psi, self._D(A[c, ell, 0], A[c, ell, 1]), c)
            q = q_from_logit(q_logits[ell], freeze_q1=freeze)
            q_cont.append(q)
            q_int.append(integer_q(q_logits[ell], freeze_q1=freeze))
            phase = torch.exp(-1j * (gamma * q).to(self.cdtype) * Cn.to(self.cdtype))
            sl = tuple(slice(0, NU_) for _ in range(C))
            # Avoid in-place on view that breaks autograd: clone then assign
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new

        probs = psi.real.square() + psi.imag.square()
        # computational mask
        # build via indexing
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
            "q_int": q_int,
            "q_cont": q_cont,
            "psi": psi,
        }


def random_params(cfg: AnsatzConfig, rng: np.random.Generator | None = None, device=None) -> dict:
    rng = rng or np.random.default_rng(0)
    C, L = cfg.C, cfg.L
    device = device or torch.device("cpu")
    return {
        "alpha0": torch.as_tensor(rng.normal(0, 0.3, size=(C, 2)), dtype=torch.float32, device=device),
        "A": torch.as_tensor(rng.normal(0, 0.3, size=(C, L, 2)), dtype=torch.float32, device=device),
        "TH": torch.as_tensor(rng.normal(0, 0.3, size=(C, L, NU)), dtype=torch.float32, device=device),
        "q_logits": torch.as_tensor(rng.normal(0, 0.5, size=(L,)), dtype=torch.float32, device=device),
        "gamma": torch.tensor(1.0, dtype=torch.float32, device=device),
    }
