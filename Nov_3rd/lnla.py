"""Load-normalized graded H_c × load-adaptive cavity SNAP (Opus Idea 2, LN-LA).

Honest graded H_c: every Walsh weight is positive and comes from the cavity
load of the instance. This is not a projector and not a renamed cost.

Symbols (4-SAT TEB packs in this repo: m = round(9.93 n), 4 distinct vars):
  ρ_j  clause incidences on the 3 bits of cavity j (var v sits in cavity v // 3)
  ρ̄   mean of ρ_j over the n/3 cavities
  r_j  ρ_j / ρ̄
  cav(S)  cavities touched by Walsh mode S (bits of S, var0 = MSB)
  γ_S  γ_ℓ · Π_{j∈cav(S)} r_j^{−α / |cav(S)|}     (empty product = 1)
  a_n  (8 / (n − 1))^p     frozen power_p champ
  φ_k  c_{ℓ,1} k + a_n (c_{ℓ,2} k² + c_{ℓ,3} k³)   shared champ shell
  φ_k^{(j)}  r_j^β · φ_k     cavity-j Hamming-shell SNAP

Warm start: α = β = 0 recovers scalar-γ power_p Hamming-shell SNAP
(same U_c^X = W D W and the same shared shell on every cavity).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from explore_OCT29.lib.packing import NU
from explore_OCT29.lib.sat_gen import clause_count

_TOUCH: dict[int, np.ndarray] = {}


def a_of_n(n: int, p: float) -> float:
    """power_p scale. a_9 = 1 for every p."""
    return (8.0 / float(n - 1)) ** float(p)


def shell_phi(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """φ_k = c1 k + c2 k² + c3 k³, k = 0..3. c*: (L,) → (L, 4)."""
    k = torch.arange(4, dtype=c1.dtype, device=c1.device)
    return (
        c1.reshape(-1, 1) * k
        + c2.reshape(-1, 1) * (k ** 2)
        + c3.reshape(-1, 1) * (k ** 3)
    )


def angles_from_champ(state: dict, n: int, device=None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Frozen power_p champ → (gamma (L,), phi (L, 4), a_n) at this n.

    φ already includes a_n on c2, c3. LN-LA then multiplies by r_j^β per cavity.
    """
    device = device or torch.device("cpu")
    gamma = state["gamma"].to(device=device, dtype=torch.float32).reshape(-1)
    c1 = state["c1"].to(device=device, dtype=torch.float32).reshape(-1)
    c2 = state["c2"].to(device=device, dtype=torch.float32).reshape(-1)
    c3 = state["c3"].to(device=device, dtype=torch.float32).reshape(-1)
    if not (gamma.numel() == c1.numel() == c2.numel() == c3.numel()):
        raise ValueError("champ gamma/c1/c2/c3 length mismatch")
    p = state.get("p")
    if p is None:
        raise ValueError("LN-LA freezes a power_p champ (missing p)")
    a = a_of_n(int(n), float(p))
    phi = shell_phi(c1, c2 * a, c3 * a)
    return gamma, phi, a


def cavity_loads(n: int, clause_vars) -> tuple[np.ndarray, np.ndarray]:
    """ρ_j and r_j = ρ_j / mean(ρ).

    clause_vars: (m, k) variable indices, or a flat list of those rows.
    Each entry is one clause incidence on that variable's cavity.
    """
    if int(n) % 3 != 0:
        raise ValueError(f"n={n} is not a multiple of 3")
    C = int(n) // 3
    vars_arr = np.asarray(clause_vars, dtype=np.int64)
    if vars_arr.size == 0:
        raise ValueError("no clause variables")
    cav = vars_arr.reshape(-1) // 3
    if int(cav.min()) < 0 or int(cav.max()) >= C:
        raise ValueError(f"variable outside 0..{n - 1}")
    rho = np.bincount(cav, minlength=C).astype(np.float64)
    mean = float(rho.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"mean cavity load is {mean}")
    if np.any(rho <= 0.0):
        raise ValueError(f"cavity with nonpositive load ρ={rho.tolist()}")
    return rho, rho / mean


def cavity_touch(n: int) -> np.ndarray:
    """Cavity-set bitmask for every Walsh index x in 0..2^n-1 (var0 = MSB).

    Bit j of the mask is set when cavity j owns at least one variable in mode x.
    """
    n = int(n)
    cached = _TOUCH.get(n)
    if cached is not None:
        return cached
    if n % 3 != 0:
        raise ValueError(f"n={n} is not a multiple of 3")
    N = 1 << n
    x = np.arange(N, dtype=np.uint32)
    touch = np.zeros(N, dtype=np.uint16)
    for v in range(n):
        bit = ((x >> np.uint32(n - 1 - v)) & np.uint32(1)).astype(np.uint16)
        touch |= bit << np.uint16(v // 3)
    _TOUCH[n] = touch
    return touch


def subset_multipliers(r: np.ndarray, alpha: float) -> np.ndarray:
    """m[mask] = Π_{j in mask} r_j^{−α / |mask|}. m[0] = 1."""
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if np.any(r <= 0.0) or not np.all(np.isfinite(r)):
        raise ValueError(f"load ratios must be positive and finite, got {r}")
    C = int(r.shape[0])
    nmask = 1 << C
    masks = np.arange(nmask, dtype=np.int64)
    log_r = np.log(r)
    acc = np.zeros(nmask, dtype=np.float64)
    cnt = np.zeros(nmask, dtype=np.float64)
    for j in range(C):
        bit = ((masks >> j) & 1).astype(np.float64)
        acc += bit * log_r[j]
        cnt += bit
    out = np.ones(nmask, dtype=np.float64)
    nz = cnt > 0.0
    out[nz] = np.exp((-float(alpha) / cnt[nz]) * acc[nz])
    return out


def load_multipliers(r: np.ndarray, alpha: float, touch: np.ndarray) -> np.ndarray:
    """Per-mode Walsh weights, length 2^n, aligned with bitstring index."""
    return subset_multipliers(r, alpha)[np.asarray(touch, dtype=np.int64)]


def fwht_unnormalized(x: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh–Hadamard along the last axis. Length must be 2^k."""
    n = int(x.shape[-1])
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT length must be a power of 2, got {n}")
    y = np.array(x, dtype=np.float64, copy=True)
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = np.array(y[..., 0, :], copy=True)
        b = np.array(y[..., 1, :], copy=True)
        y = np.stack((a + b, a - b), axis=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def normalized_cost_bitstring(E, n: int) -> np.ndarray:
    """Same normalization as cost_Cn, in bitstring order (var0 = MSB): E·n/m − max."""
    m = clause_count(int(n))
    if torch.is_tensor(E):
        e = E.detach().cpu().numpy()
    else:
        e = np.asarray(E)
    f = np.asarray(e, dtype=np.float64).reshape(-1) * (float(n) / float(m))
    if f.size != (1 << int(n)):
        raise ValueError(f"E length {f.size} != 2^{n}")
    f = f - f.max()
    return f


def graded_cost_bitstring(E, n: int, r: np.ndarray, alpha: float) -> np.ndarray:
    """Bitstring-order cost after load-normalized Walsh weights.

    α = 0 returns the normalized cost (FWHT round-trip in float64).
    """
    f = normalized_cost_bitstring(E, n)
    if abs(float(alpha)) == 0.0:
        return f
    touch = cavity_touch(int(n))
    spec = fwht_unnormalized(f)
    weighted = spec * load_multipliers(r, float(alpha), touch)
    return fwht_unnormalized(weighted) / float(f.size)


def cost_diag_grid(sim, E: torch.Tensor, r: np.ndarray, alpha: float) -> torch.Tensor:
    """γ-free cost diagonal on the product grid, shape (8,)^(n/3).

    α = 0 uses cost_Cn so the warm cost matches the champ operator exactly.
    """
    if abs(float(alpha)) == 0.0:
        return sim.cost_Cn(E)
    g = graded_cost_bitstring(E, int(sim.cfg.n), r, float(alpha))
    t = torch.as_tensor(g, dtype=sim.rdtype, device=sim.lam.device)
    return t[sim.grid_to_x]


def _measure(sim, psi: torch.Tensor, E: torch.Tensor) -> dict:
    probs = psi.real.square() + psi.imag.square()
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    p_comp = probs[sl]
    p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
    E_grid = E.to(device=sim.lam.device)[sim.grid_to_x]
    p_succ = p_comp[E_grid == 0].sum().clamp(min=0.0)
    inv_p = 1.0 / p_succ.clamp(min=1e-12)
    return {
        "p_succ": p_succ,
        "p_leak": p_leak,
        "inv_p": inv_p,
        "norm": probs.sum(),
        "arm": "Nov_3rd",
        "label": "load_normalized_graded_Hc",
        "graded_Hc": True,
    }


@torch.no_grad()
def forward_lnla(
    sim,
    E: torch.Tensor,
    gamma: torch.Tensor,
    phi: torch.Tensor,
    r,
    alpha: float,
    beta: float,
    *,
    exact_warm: bool = True,
) -> dict:
    """One instance of LN-LA.

    alpha = beta = 0 calls forward_nov3rd (exact champ), unless exact_warm is False.
    SNAP is shared across cavities when β = 0, and φ_k^{(j)} = r_j^β φ_k otherwise.
    """
    alpha = float(alpha)
    beta = float(beta)
    L = int(sim.L)
    gamma = gamma.to(dtype=sim.rdtype, device=sim.lam.device).reshape(L)
    phi = phi.to(dtype=sim.rdtype, device=sim.lam.device).reshape(L, 4)
    r_np = np.asarray(r, dtype=np.float64).reshape(-1)
    if r_np.shape != (sim.C,):
        raise ValueError(f"r has shape {r_np.shape}, want ({sim.C},)")

    warm = abs(alpha) == 0.0 and abs(beta) == 0.0
    if exact_warm and warm:
        out = sim.forward_nov3rd(E, gamma, phi)
        out = dict(out)
        out["label"] = "load_normalized_graded_Hc"
        out["lnla_warm"] = True
        out["alpha"] = 0.0
        out["beta"] = 0.0
        return out

    diag = cost_diag_grid(sim, E, r_np, alpha)
    scales = None
    if abs(beta) != 0.0:
        scales = torch.as_tensor(
            np.power(r_np, beta), dtype=sim.rdtype, device=sim.lam.device
        )

    psi = sim.vacuum()
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    for ell in range(L):
        psi = sim._apply_all(psi, sim.Wmat)
        phase = torch.exp(-1j * (diag * gamma[ell]).to(sim.cdtype))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        psi = psi_new
        psi = sim._apply_all(psi, sim.Wmat)
        if scales is None:
            th = phi[ell][sim.popcount]
            psi = sim._apply_all(psi, sim._snap_theta(th))
        else:
            for c in range(sim.C):
                th = scales[c] * phi[ell][sim.popcount]
                psi = sim._apply_local(psi, sim._snap_theta(th), c)
    psi = sim._apply_all(psi, sim.Wmat)
    out = _measure(sim, psi, E)
    out["lnla_warm"] = False
    out["alpha"] = alpha
    out["beta"] = beta
    return out
