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
from Nov_3rd.cavity_grade import split_E_torch


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

    def forward_nov3rd_cavity(
        self,
        E: torch.Tensor,
        gamma_in: torch.Tensor,
        gamma_x: torch.Tensor,
        phi: torch.Tensor,
        E_in: torch.Tensor | None = None,
        E_x: torch.Tensor | None = None,
    ) -> dict:
        """Graded-H_c U_c^X: W exp(-i[γ_in E_in + γ_x E_x]) W, then Hamming-shell SNAP.

        E_in = intra-cavity Walsh strings (incl. DC); E_x = strings touching ≥2 cavities.
        Warm γ_in = γ_x = γ recovers forward_nov3rd exactly (global phase via amax).
        """
        L = self.L
        if L != int(gamma_in.numel()) or L != int(gamma_x.numel()):
            raise ValueError(
                f"gamma_in/x lens {gamma_in.numel()}/{gamma_x.numel()} != L={L}"
            )
        if tuple(phi.shape) != (L, S.B_BITS + 1):
            raise ValueError(f"phi shape {tuple(phi.shape)} != {(L, S.B_BITS + 1)}")

        gamma_in = gamma_in.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        gamma_x = gamma_x.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        n = self.cfg.n
        m = int(round(9.93 * n))
        scale = float(n) / float(m)
        E = E.to(device=self.lam.device)
        if E_in is None or E_x is None:
            E_in, E_x = split_E_torch(E.reshape(-1).to(dtype=torch.float32), n)
        E_in = E_in.to(device=self.lam.device, dtype=self.rdtype).reshape(-1)
        E_x = E_x.to(device=self.lam.device, dtype=self.rdtype).reshape(-1)

        Cn_in = E_in[self.grid_to_x] * scale
        Cn_x = E_x[self.grid_to_x] * scale

        psi = self.vacuum()
        sl = tuple(slice(0, NU) for _ in range(self.C))
        for ell in range(L):
            psi = self._apply_all(psi, self.Wmat)
            arg = gamma_in[ell] * Cn_in + gamma_x[ell] * Cn_x
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
            "arm": "Nov_3rd_cavity_graded_Hc",
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
