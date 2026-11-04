"""Nov_3rd motif: |0⟩ → [Uc^X → bare SNAP]^L → final W → Fock.

Built on explore GadgetAnsatz hardware (BinH W), but forward is local to this package
so Nov/ explorers cannot silently change the circuit.
"""
from __future__ import annotations

import torch

from explore_OCT29.lib.ansatz import AnsatzConfig
from explore_OCT29.lib.packing import NU
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH
from explore_OCT29.scripts.gadget_Uc_explore import GadgetAnsatz

from Nov_3rd import settings as S


class Nov3rdAnsatz(GadgetAnsatz):
    """Locked Nov_3rd forward. SNAP head is always shell; shared across cavities."""

    def forward_nov3rd(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
    ) -> dict:
        """phi: (L, b+1) shell phases; gamma: (L,)."""
        L = self.L
        if L != int(gamma.numel()):
            raise ValueError(f"gamma len {gamma.numel()} != L={L}")
        if tuple(phi.shape) != (L, S.B_BITS + 1):
            raise ValueError(f"phi shape {tuple(phi.shape)} != {(L, S.B_BITS + 1)}")

        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()  # |0⟩ — no prep W
        Cn = self.cost_Cn(E)
        sl = tuple(slice(0, NU) for _ in range(self.C))

        for ell in range(L):
            # Uc^X = W D W
            psi = self._apply_all(psi, self.Wmat)
            phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new
            psi = self._apply_all(psi, self.Wmat)
            # bare SNAP shell (same on every cavity via _apply_all)
            th = phi[ell][self.popcount]
            psi = self._apply_all(psi, self._snap_theta(th))

        # final W → Fock
        psi = self._apply_all(psi, self.Wmat)

        probs = psi.real.square() + psi.imag.square()
        p_comp = probs[sl]
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
            "arm": "Nov_3rd",
            "probs": probs,
        }

    def forward_nov3rd_graded(
        self,
        E: torch.Tensor,
        gamma_w: torch.Tensor,
        phi: torch.Tensor,
        Cn_parts: torch.Tensor,
    ) -> dict:
        """Graded H_c U_c^X = W exp(-i Σ_w γ_{ℓ,w} C_w) W, then Hamming-shell SNAP.

        gamma_w: (L, K)  — K Walsh degrees (w=1..4)
        Cn_parts: (K, *grid) already scaled by n/m, NOT amax-centered
        Equal γ_w recovers forward_nov3rd (global phase after amax).
        Success is still scored on the original unsat E (solutions unchanged
        only when the γ_w are equal; otherwise the cost is a reweighted H_c).
        """
        L = self.L
        K = int(Cn_parts.shape[0])
        if tuple(gamma_w.shape) != (L, K):
            raise ValueError(f"gamma_w shape {tuple(gamma_w.shape)} != {(L, K)}")
        if tuple(phi.shape) != (L, S.B_BITS + 1):
            raise ValueError(f"phi shape {tuple(phi.shape)} != {(L, S.B_BITS + 1)}")

        gw = gamma_w.to(dtype=self.rdtype, device=self.lam.device)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        parts = Cn_parts.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()
        sl = tuple(slice(0, NU) for _ in range(self.C))
        ones = (1,) * self.C

        for ell in range(L):
            psi = self._apply_all(psi, self.Wmat)
            arg = (gw[ell].reshape(K, *ones) * parts).sum(dim=0)
            arg = arg - arg.amax()
            phase = torch.exp(-1j * arg.to(self.cdtype))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new
            psi = self._apply_all(psi, self.Wmat)
            th = phi[ell][self.popcount]
            psi = self._apply_all(psi, self._snap_theta(th))

        psi = self._apply_all(psi, self.Wmat)
        probs = psi.real.square() + psi.imag.square()
        p_comp = probs[sl]
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
            "arm": "Nov_3rd_graded_Hc",
            "probs": probs,
        }


def make_W():
    S.assert_locked(W_frame=S.W_FRAME, b=S.B_BITS)
    if S.W_FRAME != "BinH":
        raise RuntimeError("Nov_3rd requires BinH")
    return build_W_BinH(nf=S.NF)


def make_sim(n: int, L: int, device):
    if n % S.B_BITS != 0:
        raise ValueError(f"n={n} not divisible by b={S.B_BITS}")
    W = make_W()
    cfg = AnsatzConfig(n=n, L=L, nf=S.NF)
    return Nov3rdAnsatz(cfg, W, device=device).to(device)
