"""Nov_3rd motif: |0⟩ → [Uc^X → bare SNAP]^L → final W → Fock.

Built on explore GadgetAnsatz hardware (BinH W), but forward is local to this package
so Nov/ explorers cannot silently change the circuit.

Supports graded-H_c Walsh-degree γ: gamma shape (L,) or (L, 4) with
U_c^X = W exp(-i Σ_{w=1..4} γ_{ℓ,w} E_w) W, E_w = degree-w Walsh piece of cost_Cn.
Equal γ_w recovers scalar γ·E (up to global phase from w=0).

Optional fcws= dict turns on footprint-class grading (Nov_3rd.fc_ws): class
boosts (1+δ_C) on U_c^X and a commutator-matched rescaling of shell orders.
fcws is None on every existing champ / power_p / Walsh path.
"""
from __future__ import annotations

import torch

from explore_OCT29.lib.ansatz import AnsatzConfig
from explore_OCT29.lib.packing import NU
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH
from explore_OCT29.scripts.gadget_Uc_explore import GadgetAnsatz

from Nov_3rd import settings as S
from Nov_3rd.fc_ws import (
    CLASS_DEG_INDEX,
    effective_delta,
    footprint_class_ids,
    o_tilde,
    rescale_shell_phi,
)


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
        fp_ids = footprint_class_ids(n, device=self.lam.device)
        self.register_buffer("_fp_class", fp_ids)
        fp_masks = torch.stack([(fp_ids == c) for c in range(6)], dim=0)
        self.register_buffer("_fp_masks", fp_masks)
        self.register_buffer("_fp_rest", fp_ids == -2)

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

    @torch.no_grad()
    def footprint_parts(self, E: torch.Tensor) -> dict:
        """Walsh pieces of H_c on the product grid, plus class masses w_C.

        comps[c] = Σ_{S∈C} ŵ_S (-1)^{S·x} gathered onto the Fock grid.
        rest = degree≥4 piece (no footprint δ). Degree 0 is omitted.
        w[c] = Σ_{S∈C} |ŵ_S|. Detached: H_c is data, not a parameter.
        """
        f = self.cost_bitstring(E)
        N = int(f.numel())
        F = fwht_unnormalized(f)
        Fw = torch.where(self._fp_masks, F.unsqueeze(0), torch.zeros_like(F))
        w = Fw.abs().sum(dim=-1) / float(N)
        spatial = fwht_unnormalized(Fw) / float(N)
        Fr = torch.where(self._fp_rest, F, torch.zeros_like(F))
        rest = fwht_unnormalized(Fr) / float(N)
        g = self.grid_to_x
        return {"comps": spatial[:, g], "rest": rest[g], "w": w}

    def _fcws_layer_diag(self, parts: dict, gamma_ell: torch.Tensor, delta_eff: torch.Tensor) -> torch.Tensor:
        """Spatial diagonal of Σ_C γ_{ℓ,deg(C)} (1+δ_C) E_C + γ_rest E_rest."""
        comps = parts["comps"]
        rest = parts["rest"]
        dlt = delta_eff.to(dtype=self.rdtype, device=comps.device).reshape(6)
        if gamma_ell.ndim == 0 or int(gamma_ell.numel()) == 1:
            g = gamma_ell.to(dtype=self.rdtype, device=comps.device).reshape(())
            acc = rest * g
            weights = g * (1.0 + dlt)
        else:
            gw = gamma_ell.to(dtype=self.rdtype, device=comps.device).reshape(4)
            acc = rest * gw[3]
            deg_idx = torch.tensor(CLASS_DEG_INDEX, device=comps.device)
            weights = gw[deg_idx] * (1.0 + dlt)
        view = weights.reshape(-1, *([1] * (comps.ndim - 1)))
        return acc + (view * comps).sum(dim=0)

    def forward_nov3rd(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        *,
        fcws: dict | None = None,
    ) -> dict:
        """phi: (L, b+1) shell phases; gamma: (L,) or (L, 4) Walsh-degree graded.

        fcws, when set, is graded H_c footprint-class FC-WS:
          delta: (6,) or pooled (5,) class boosts
          kappa: scalar shell-order coupling
          pool_111: tie δ_(1,1,1)=δ_(1,1)
          parts: optional precomputed footprint_parts(E)
        """
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
        use_fc = fcws is not None
        if use_fc:
            parts = fcws.get("parts")
            if parts is None:
                parts = self.footprint_parts(E)
            delta_eff = effective_delta(fcws["delta"], bool(fcws.get("pool_111", False)))
            delta_eff = delta_eff.to(device=self.lam.device)
            kappa = fcws["kappa"]
            if not torch.is_tensor(kappa):
                kappa = torch.tensor(float(kappa), device=self.lam.device)
            ot = o_tilde(parts["w"], delta_eff)
            phi = rescale_shell_phi(phi, ot, kappa).to(dtype=self.rdtype)
            graded = False  # diag comes from _fcws_layer_diag, not scalar Cn
        psi = self.vacuum()  # |0⟩ — no prep W
        sl = tuple(slice(0, NU) for _ in range(self.C))

        if use_fc:
            Cn = None
        elif not graded:
            Cn = self.cost_Cn(E)
        else:
            Cn = None  # per-layer graded diag

        for ell in range(L):
            # Uc^X = W D W
            psi = self._apply_all(psi, self.Wmat)
            if use_fc:
                diag = self._fcws_layer_diag(parts, gamma[ell], delta_eff)
            elif graded:
                diag = self.graded_cost_diag(E, gamma[ell])
            else:
                diag = Cn * gamma[ell]
            # for graded / fcws, diag already includes γ; for scalar, Cn*γ
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
            "graded_Hc": bool(graded or use_fc),
            "fc_ws": bool(use_fc),
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
