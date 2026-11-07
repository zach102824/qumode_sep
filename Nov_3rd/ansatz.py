"""Nov_3rd motif: |0⟩ → [Uc^X → bare SNAP]^L → final W → Fock.

Built on explore GadgetAnsatz hardware (BinH W), but forward is local to this package
so Nov/ explorers cannot silently change the circuit.

Supports graded-H_c Walsh-degree γ: gamma shape (L,) or (L, 4) with
U_c^X = W exp(-i Σ_{w=1..4} γ_{ℓ,w} E_w) W, E_w = degree-w Walsh piece of cost_Cn.
Equal γ_w recovers scalar γ·E (up to global phase from w=0).
"""
from __future__ import annotations

import torch

from explore_OCT29.lib.ansatz import AnsatzConfig
from explore_OCT29.lib.packing import NU
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH
from explore_OCT29.scripts.gadget_Uc_explore import GadgetAnsatz

from Nov_3rd import settings as S
from Nov_3rd.cavity_grade import split_E_torch


def fwht_unnormalized(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Fast Walsh–Hadamard transform along the last axis."""
    n = int(x.shape[-1])
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT length must be power of 2, got {n}")
    y = x.contiguous().clone()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def popcount_arange(n_bits: int, device, dtype=torch.int64) -> torch.Tensor:
    """popcount(0..2^n_bits-1)."""
    N = 1 << int(n_bits)
    idx = torch.arange(N, device=device, dtype=torch.int64)
    # SWAR popcount
    pc = idx - ((idx >> 1) & 0x5555_5555_5555_5555)
    pc = (pc & 0x3333_3333_3333_3333) + ((pc >> 2) & 0x3333_3333_3333_3333)
    pc = (pc + (pc >> 4)) & 0x0F0F_0F0F_0F0F_0F0F
    pc = (pc * 0x0101_0101_0101_0101) >> 56
    return pc.to(dtype=dtype)


class Nov3rdAnsatz(GadgetAnsatz):
    """Locked Nov_3rd forward. SNAP head is always shell; shared across cavities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = int(self.cfg.n)
        pc = popcount_arange(n, device=self.lam.device)
        self.register_buffer("_walsh_popcount", pc)
        # degree masks w=1..4 as bool (N,)
        masks = torch.stack([(pc == w) for w in range(1, 5)], dim=0)  # (4, N)
        self.register_buffer("_walsh_degree_masks", masks)

    def cost_bitstring(self, E: torch.Tensor) -> torch.Tensor:
        """Normalized relative cost in bitstring order (length 2^n), same as cost_Cn."""
        n = self.cfg.n
        m = int(round(9.93 * n))
        E = E.to(device=self.lam.device)
        if E.dtype not in (torch.float32, torch.float64):
            E = E.to(self.rdtype)
        f = E.to(self.rdtype) * (n / m)
        f = f - f.amax()
        return f

    def graded_cost_diag(self, E: torch.Tensor, gamma_w: torch.Tensor) -> torch.Tensor:
        """Σ_{w=1..4} γ_w E_w on Gray/binary product grid, shape (NU,)*C.

        gamma_w: (4,) layer weights for degrees 1..4.
        """
        f = self.cost_bitstring(E)  # (N,)
        N = f.numel()
        F = fwht_unnormalized(f)
        gw = gamma_w.to(dtype=self.rdtype, device=f.device).reshape(4)
        # Walsh-domain mix: G[S] = γ_{|S|} F[S] for |S|∈{1..4}
        G = (gw.reshape(4, 1) * torch.where(self._walsh_degree_masks, F.unsqueeze(0), 0.0)).sum(dim=0)
        diag = fwht_unnormalized(G) / float(N)
        return diag[self.grid_to_x]

    def walsh_degree_components_grid(self, E: torch.Tensor) -> list[torch.Tensor]:
        """Return [E_1..E_4] on product grid (for diagnostics)."""
        f = self.cost_bitstring(E)
        N = f.numel()
        F = fwht_unnormalized(f)
        out = []
        for w in range(4):
            Fw = torch.where(self._walsh_degree_masks[w], F, torch.zeros_like(F))
            fw = fwht_unnormalized(Fw) / float(N)
            out.append(fw[self.grid_to_x])
        return out

    def forward_nov3rd(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
    ) -> dict:
        """phi: (L, b+1) shell phases; gamma: (L,) or (L, 4) Walsh-degree graded."""
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device)
        if gamma.ndim == 1:
            if int(gamma.numel()) != L:
                raise ValueError(f"gamma len {gamma.numel()} != L={L}")
            graded = False
            gamma = gamma.reshape(L)
        elif gamma.ndim == 2:
            if tuple(gamma.shape) != (L, 4):
                raise ValueError(f"gamma shape {tuple(gamma.shape)} != {(L, 4)}")
            graded = True
        else:
            raise ValueError(f"gamma ndim {gamma.ndim} not in {{1,2}}")
        if tuple(phi.shape) != (L, S.B_BITS + 1):
            raise ValueError(f"phi shape {tuple(phi.shape)} != {(L, S.B_BITS + 1)}")

        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()  # |0⟩ — no prep W
        sl = tuple(slice(0, NU) for _ in range(self.C))

        if not graded:
            Cn = self.cost_Cn(E)
        else:
            Cn = None  # per-layer graded diag

        for ell in range(L):
            # Uc^X = W D W
            psi = self._apply_all(psi, self.Wmat)
            if graded:
                diag = self.graded_cost_diag(E, gamma[ell])
            else:
                diag = Cn * gamma[ell]
            # for graded, diag already includes γ_w; for scalar, Cn*γ
            if graded:
                phase = torch.exp(-1j * diag.to(self.cdtype))
            else:
                phase = torch.exp(-1j * diag.to(self.cdtype))
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
            "graded_Hc": graded,
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
