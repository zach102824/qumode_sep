"""WT: shared Walsh-degree tilt on top of the power_p champ.

U_c^X = W exp(-i Σ_{w=1..4} γ_ℓ · τ_w · f_w) W
    f_w = degree-w Walsh component of the normalized cost f
    τ_1 ≡ 1
    τ_2, τ_3, τ_4 shared across layers and across n

SNAP is the champ power_p cubic, unchanged:
    φ_k = c1·k + a(n)·(c2·k² + c3·k³)
    a(n) = (8/(n−1))^p

τ = (1,1,1,1) reproduces scalar γ · f up to a global phase (the degree-0
Walsh piece). That is the champ-recovery point.

Graded Fourier QAOA uses the same τ on the qubit cost and the FM-WS DST/DCT
schedule. The basis below matches ``fm_ws.py`` on the FM-WS branch so a
published (A, B) rebuilds the same (β, γ).
"""
from __future__ import annotations

import math

import torch

from Nov_3rd import settings as S
from Nov_3rd.ansatz import fwht_unnormalized, popcount_arange

# Free tilt collapses when every |τ_w − 1| is below this (Opus kill, like κ).
COLLAPSE_EPS = 0.05
TAU_LO = -8.0
TAU_HI = 8.0
P_LO = -1.0
P_HI = 2.0


def phi_from_c(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """φ_k = c1·k + c2·k² + c3·k³, k = 0..3. Same map as scripts/beat_params.py."""
    L = int(c1.reshape(-1).numel())
    k = torch.arange(S.B_BITS + 1, device=c1.device, dtype=c1.dtype)
    pows = torch.stack([k, k ** 2, k ** 3], dim=0)
    C = torch.stack([c1.reshape(L), c2.reshape(L), c3.reshape(L)], dim=1)
    return C @ pows


def a_of_n(n: int, p: torch.Tensor) -> torch.Tensor:
    """a(n) = (8/(n−1))^p. Equals 1 at n = 9 for every p."""
    base = 8.0 / float(int(n) - 1)
    p = p if torch.is_tensor(p) else torch.tensor(float(p))
    return torch.pow(torch.tensor(base, device=p.device, dtype=p.dtype), p)


def tau_full(tau_free: torch.Tensor) -> torch.Tensor:
    """(τ_1, τ_2, τ_3, τ_4) with τ_1 locked at 1. ``tau_free`` is (3,)."""
    free = tau_free.reshape(3)
    one = free.new_ones(1)
    return torch.cat([one, free], dim=0)


def graded_gamma(gamma: torch.Tensor, tau_free: torch.Tensor) -> torch.Tensor:
    """γ_{ℓ,w} = γ_ℓ · τ_w. gamma (L,) → (L, 4)."""
    tau = tau_full(tau_free)
    return gamma.reshape(-1, 1) * tau.reshape(1, 4)


def snap_angles(gamma: torch.Tensor, c1, c2, c3, p, tau_free, n: int):
    """Graded γ (L, 4) and shell φ (L, 4) at size n."""
    a = a_of_n(n, p)
    phi = phi_from_c(c1, c2 * a, c3 * a)
    return graded_gamma(gamma, tau_free), phi


def walsh_degree_components(f: torch.Tensor) -> torch.Tensor:
    """Degree-1..4 Walsh pieces of ``f``.

    f: (..., 2^n) in bitstring order (var0 = MSB). Returns (4, ..., 2^n).
    Inverse FWHT is the unnormalized transform divided by 2^n, matching
    ``Nov3rdAnsatz.graded_cost_diag``.
    """
    n_bits = int(round(math.log2(int(f.shape[-1]))))
    if (1 << n_bits) != int(f.shape[-1]):
        raise ValueError(f"last dim {f.shape[-1]} is not a power of 2")
    N = 1 << n_bits
    F = fwht_unnormalized(f)
    pc = popcount_arange(n_bits, device=f.device)
    pieces = []
    for w in range(1, 5):
        Fw = torch.where(pc == w, F, torch.zeros_like(F))
        pieces.append(fwht_unnormalized(Fw) / float(N))
    return torch.stack(pieces, dim=0)


def graded_spectrum(f: torch.Tensor, tau4: torch.Tensor) -> torch.Tensor:
    """Σ_{w=1..4} τ_w f_w. ``tau4`` is (4,) with τ_1 included."""
    comps = walsh_degree_components(f)
    tau = tau4.reshape(4, *([1] * (comps.ndim - 1))).to(dtype=comps.dtype, device=comps.device)
    return (tau * comps).sum(dim=0)


def normalized_cost(E: torch.Tensor, n: int) -> torch.Tensor:
    """f = E · n / m, m = round(9.93 n). No max subtraction (degree ≥ 1 is invariant to it)."""
    m = int(round(9.93 * int(n)))
    return E.to(dtype=torch.float32) * (float(n) / float(m))


def tilt_collapsed(tau_free: torch.Tensor, eps: float = COLLAPSE_EPS) -> bool:
    """True when every free |τ_w − 1| < eps. τ_1 is identically 1."""
    dev = (tau_free.detach().reshape(-1).float() - 1.0).abs()
    if int(dev.numel()) != 3:
        raise ValueError(f"tau_free must have 3 entries, got {int(dev.numel())}")
    return bool(torch.all(dev < float(eps)).item())


def max_tau_dev(tau_free: torch.Tensor) -> float:
    return float((tau_free.detach().reshape(3).float() - 1.0).abs().max().item())


# --- Fourier schedule (DST-IV / DCT-IV), same as FM-WS ---


def mode_matrix(L: int, Q: int, kind: str, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Basis (L, Q). kind 'dst' → u_q (sine), 'dct' → v_q (cosine)."""
    L = int(L)
    Q = int(Q)
    if L < 1 or Q < 1:
        raise ValueError(f"L and Q must be ≥ 1, got L={L} Q={Q}")
    if Q > L:
        raise ValueError(f"Q={Q} exceeds L={L}")
    if kind not in ("dst", "dct"):
        raise ValueError(f"kind must be 'dst' or 'dct', got {kind!r}")
    ell = torch.arange(1, L + 1, device=device, dtype=dtype)
    q = torch.arange(1, Q + 1, device=device, dtype=dtype)
    ang = (q - 0.5).unsqueeze(0) * (ell - 0.5).unsqueeze(1) * (math.pi / float(L))
    if kind == "dst":
        return torch.sin(ang)
    return torch.cos(ang)


def analyze(schedule: torch.Tensor, kind: str) -> torch.Tensor:
    """Full DST-IV / DCT-IV coefficients, length L."""
    x = schedule.detach().reshape(-1).to(dtype=torch.float64)
    L = int(x.numel())
    M = mode_matrix(L, L, kind, device=x.device, dtype=torch.float64)
    return (2.0 / float(L)) * (M.transpose(0, 1) @ x)


def synthesize(coeffs: torch.Tensor, L: int, kind: str) -> torch.Tensor:
    """Σ_{q=1..Q} coeff_q · mode_q(ℓ)."""
    M = mode_matrix(L, int(coeffs.shape[0]), kind, device=coeffs.device, dtype=coeffs.dtype)
    return M @ coeffs


def qaoa_angles_from_modes(A: torch.Tensor, B: torch.Tensor, L: int):
    """β_ℓ = Σ B_q v_q(ℓ), γ_ℓ = Σ A_q u_q(ℓ)."""
    if A.ndim != 1 or B.ndim != 1:
        raise ValueError("Fourier A, B must be 1-D")
    if int(A.shape[0]) != int(B.shape[0]):
        raise ValueError("A and B must share Q")
    return synthesize(B, L, "dct"), synthesize(A, L, "dst")


def qaoa_modes_from_angles(betas: torch.Tensor, gammas: torch.Tensor, Q: int) -> dict:
    b = betas.detach().reshape(-1)
    g = gammas.detach().reshape(-1)
    if int(b.numel()) != int(g.numel()):
        raise ValueError("betas and gammas length mismatch")
    L = int(b.numel())
    Q = int(Q)
    if Q < 1 or Q > L:
        raise ValueError(f"Q={Q} not in 1..L={L}")
    return {
        "A": analyze(g, "dst")[:Q].to(dtype=torch.float32).contiguous(),
        "B": analyze(b, "dct")[:Q].to(dtype=torch.float32).contiguous(),
    }


def qaoa_psucc_graded(betas, gammas, E: torch.Tensor, n: int, tau4: torch.Tensor) -> torch.Tensor:
    """Qubit QAOA success probability on Σ_w τ_w f_w.

    τ = 1 matches ``qaoa_psucc`` up to the degree-0 global phase.
    """
    from explore_OCT29.lib.qaoa import _apply_x_mixer

    if E.ndim == 1:
        E = E.reshape(1, -1)
    Bsz, dim = E.shape
    if dim != (1 << int(n)):
        raise ValueError(f"E width {dim} != 2^{n}")
    device = E.device
    f = normalized_cost(E, n)
    Cn = graded_spectrum(f, tau4.to(device=device, dtype=torch.float32))
    betas = betas.to(device=device, dtype=torch.float32).reshape(-1)
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    if betas.numel() != gammas.numel():
        raise ValueError("L mismatch between betas and gammas")
    psi = torch.full((Bsz, dim), 1.0 / math.sqrt(dim), dtype=torch.complex64, device=device)
    for ell in range(int(gammas.numel())):
        phase = torch.exp(-1j * gammas[ell].to(torch.complex64) * Cn.to(torch.complex64))
        psi = psi * phase
        psi = _apply_x_mixer(psi, betas[ell], int(n))
    probs = psi.real.square() + psi.imag.square()
    sat = (E == 0).to(dtype=probs.dtype)
    return (probs * sat).sum(dim=-1)


def _gap(qaoa_med: float, snap_med: float) -> float:
    """QAOA Med − SNAP Med. Larger means SNAP beats its matched QAOA by more."""
    return float(qaoa_med) - float(snap_med)


def wt_verdict(
    *,
    l2_n15: list[float],
    walsh_n15: float,
    l16: list[dict],
    fourier_n15: float,
    champ_n15: float,
    eps: float = COLLAPSE_EPS,
) -> dict:
    """Ship / kill / needs-edit for the WT smoke.

    Each ``l16`` entry needs ``snap_n15``, ``graded_n15``, ``tau_dev`` (max |τ_w−1|
    over the three free tilts).

    Kill (Opus), judged on the whole seed set:
      * every L=2 seed has n15 Med worse than Walsh seed1, or
      * every L=16 seed has all |τ_w−1| < eps, or
      * every L=16 seed has SNAP−QAOA gap with τ no bigger than the τ=1 gap.
    Ship only if none of those fire, the best L=2 seed beats Walsh seed1 at n15,
    and at least one L=16 seed moves τ, grows the gap, and beats both graded
    Fourier QAOA (same τ) and ungraded Fourier QAOA at n15.
    """
    if not l2_n15:
        raise ValueError("l2_n15 is empty")
    if not l16:
        raise ValueError("l16 is empty")
    gap0 = _gap(fourier_n15, champ_n15)
    l2_best = min(float(x) for x in l2_n15)
    l2_worse = [float(x) > float(walsh_n15) for x in l2_n15]
    l2_kill = all(l2_worse)
    l2_beat = l2_best < float(walsh_n15)

    rows = []
    for row in l16:
        dev = float(row["tau_dev"])
        collapsed = dev < float(eps)
        gap = _gap(row["graded_n15"], row["snap_n15"])
        rows.append(
            {
                "snap_n15": float(row["snap_n15"]),
                "graded_n15": float(row["graded_n15"]),
                "tau_dev": dev,
                "collapsed": collapsed,
                "gap_with": gap,
                "gap_grew": gap > gap0,
                "beats_graded": float(row["snap_n15"]) < float(row["graded_n15"]),
                "beats_ungraded": float(row["snap_n15"]) < float(fourier_n15),
            }
        )
    collapse_kill = all(r["collapsed"] for r in rows)
    gap_kill = all(not r["gap_grew"] for r in rows)
    clear = [
        r
        for r in rows
        if (not r["collapsed"]) and r["gap_grew"] and r["beats_graded"] and r["beats_ungraded"]
    ]
    reasons = []
    if l2_kill:
        reasons.append("every L=2 seed is worse than Walsh seed1 at n15")
    if collapse_kill:
        reasons.append(f"every L=16 seed has all |τ_w−1| < {eps}")
    if gap_kill:
        reasons.append("every L=16 seed has SNAP−QAOA gap with τ no bigger than without τ")

    if l2_kill or collapse_kill or gap_kill:
        label = "kill"
    elif l2_beat and len(clear) >= 1:
        label = "ship"
        reasons.append(
            f"{len(clear)}/{len(rows)} L=16 seeds beat graded and ungraded Fourier QAOA with a larger gap"
        )
    else:
        label = "needs-edit"
        if not l2_beat:
            reasons.append("no L=2 seed beats Walsh seed1 at n15")
        if not clear:
            reasons.append(
                "no L=16 seed both moves τ and beats graded and ungraded Fourier QAOA with a larger gap"
            )

    return {
        "verdict": label,
        "reasons": reasons,
        "l2_best_n15": l2_best,
        "l2_beats_walsh_n15": l2_beat,
        "l2_kill_all_worse": l2_kill,
        "collapse_kill": collapse_kill,
        "gap_kill": gap_kill,
        "gap_without": gap0,
        "n_l16_clear": len(clear),
        "l16": rows,
        "rule": (
            "kill if every L=2 seed is worse than Walsh seed1 n15, "
            "or every L=16 seed has all |τ−1|<0.05, "
            "or every L=16 seed fails to grow the SNAP−QAOA gap; "
            "ship only if the best L=2 seed beats Walsh seed1 n15 and at least one "
            "L=16 seed moves τ, grows that gap, and beats both graded Fourier QAOA "
            "(same τ) and ungraded Fourier QAOA at n15; otherwise needs-edit"
        ),
    }
