"""FM-WS: shared Fourier modes + locked n-scaling (Opus Idea 3).

Hamming-shell SNAP + Walsh-graded U_c^X. Symbols (1-based layer ℓ, mode q):

    u_q(ℓ) = sin[(q − ½)(ℓ − ½) π / L]
    v_q(ℓ) = cos[(q − ½)(ℓ − ½) π / L]

    s_n = 8 / (n − 1)
    p_1 = 0 (locked),  p_2, p_3, λ free

    γ_{ℓ,d}(n) = s_n^{λ p_d} Σ_{q=1..Q} A_{q,d} u_q(ℓ)     d = 1, 2, 3
    c_{ℓ,k}(n) = s_n^{p_k}     Σ_{q=1..Q} B_{q,k} v_q(ℓ)     k = 1, 2, 3

φ_k = c_1 k + c_2 k² + c_3 k³. Both sides share Q, the INTERP rule, and the
exponents p_d (the cost side uses λ p_d).

u_q is a DST-IV mode and v_q is a DCT-IV mode. With Q = L the transform is
involutory up to 2/L, so the full-mode DST/DCT of a champ schedule is an exact
warm start: A_{q,d} equal across d, λ = 0, p_2 = p_3 = p_champ.

INTERP (pad): keep (A, B, p_2, p_3, λ) and rebuild the basis at the new depth.
Modes q > Q stay zero.

The cost is random 4-SAT (m = round(9.93 n)): Walsh degree ≤ 4, and degrees
≥ 5 are identically 0. Opus writes d = 1..3. The degree-4 slot is tied to
degree 1 (γ_{ℓ,4} = γ_{ℓ,1}) and adds no parameter. With that tie, λ = 0 and
A_{q,d} equal across d reproduces scalar γ · cost_Cn exactly (degree 0 is a
global phase). Zeroing γ_4 would drop a real piece of H_c and miss the champ.
"""
from __future__ import annotations

import math

import torch

from Nov_3rd import settings as S

# Degrees / polynomial orders that carry a free exponent. p_1 stays 0.
N_DEGREE = 3
P_LO = -1.0
P_HI = 2.0
LAM_LO = -3.0
LAM_HI = 3.0


def phi_from_c(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """φ_k = c1·k + c2·k² + c3·k³, k = 0..3. Same map as scripts/beat_params.py."""
    L = int(c1.reshape(-1).numel())
    k = torch.arange(S.B_BITS + 1, device=c1.device, dtype=c1.dtype)
    pows = torch.stack([k, k ** 2, k ** 3], dim=0)  # (3, 4)
    C = torch.stack([c1.reshape(L), c2.reshape(L), c3.reshape(L)], dim=1)
    return C @ pows


def s_of_n(n: int) -> float:
    """s_n = 8/(n−1). Equals 1 at n = 9, so stored schedules are the n=9 angles."""
    n = int(n)
    if n <= 1:
        raise ValueError(f"n must be > 1, got {n}")
    return 8.0 / float(n - 1)


def mode_matrix(
    L: int,
    Q: int,
    kind: str,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Basis (L, Q). kind 'dst' → u_q (sine), 'dct' → v_q (cosine)."""
    L = int(L)
    Q = int(Q)
    if L < 1 or Q < 1:
        raise ValueError(f"L and Q must be ≥ 1, got L={L} Q={Q}")
    if Q > L:
        raise ValueError(f"Q={Q} exceeds L={L} (pad does not invent modes above L)")
    if kind not in ("dst", "dct"):
        raise ValueError(f"kind must be 'dst' or 'dct', got {kind!r}")
    ell = torch.arange(1, L + 1, device=device, dtype=dtype)
    q = torch.arange(1, Q + 1, device=device, dtype=dtype)
    ang = (q - 0.5).unsqueeze(0) * (ell - 0.5).unsqueeze(1) * (math.pi / float(L))
    if kind == "dst":
        return torch.sin(ang)
    return torch.cos(ang)


def analyze(schedule: torch.Tensor, kind: str) -> torch.Tensor:
    """Full DST-IV / DCT-IV coefficients, length L. Inverse of synthesize at Q=L.

    coeff = (2/L) M schedule, and M is symmetric with M² = (L/2) I.
    """
    x = schedule.detach().reshape(-1).to(dtype=torch.float64)
    L = int(x.numel())
    M = mode_matrix(L, L, kind, device=x.device, dtype=torch.float64)
    return (2.0 / float(L)) * (M.transpose(0, 1) @ x)


def synthesize(coeffs: torch.Tensor, L: int, kind: str) -> torch.Tensor:
    """Σ_{q=1..Q} coeff_q · mode_q(ℓ) at depth L. coeffs: (Q,) or (Q, K)."""
    if coeffs.ndim == 1:
        M = mode_matrix(L, int(coeffs.shape[0]), kind, device=coeffs.device, dtype=coeffs.dtype)
        return M @ coeffs
    if coeffs.ndim == 2:
        M = mode_matrix(L, int(coeffs.shape[0]), kind, device=coeffs.device, dtype=coeffs.dtype)
        return M @ coeffs
    raise ValueError(f"coeffs ndim {coeffs.ndim} not in {{1, 2}}")


def n_scale(exponents: torch.Tensor, n: int) -> torch.Tensor:
    """s_n^{exponent} with grad through the exponent. Shape follows ``exponents``."""
    s = s_of_n(n)
    log_s = math.log(s)
    return torch.exp(exponents.to(dtype=torch.float32) * float(log_s))


def exponents_snap(p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """(p_1, p_2, p_3) with p_1 locked at 0."""
    zero = p2 * 0.0
    return torch.stack([zero, p2, p3])


def exponents_cost(p2: torch.Tensor, p3: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """λ p_d, still with p_1 = 0 so degree 1 does not scale with n."""
    return lam * exponents_snap(p2, p3)


def schedules_from_modes(
    A: torch.Tensor,
    B: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    lam: torch.Tensor,
    L: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return γ (L, 4), c1, c2, c3 each (L,).

    A, B: (Q, 3) Fourier amplitudes (degrees / polynomial orders).
    γ_{:,3} (degree 4) is tied to degree 1.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != N_DEGREE or B.shape[1] != N_DEGREE:
        raise ValueError(f"A,B must be (Q, 3), got {tuple(A.shape)} {tuple(B.shape)}")
    if int(A.shape[0]) != int(B.shape[0]):
        raise ValueError("A and B must share Q")
    g = synthesize(A, L, "dst")  # (L, 3)
    c = synthesize(B, L, "dct")  # (L, 3)
    g = g * n_scale(exponents_cost(p2, p3, lam), n).reshape(1, N_DEGREE)
    c = c * n_scale(exponents_snap(p2, p3), n).reshape(1, N_DEGREE)
    gamma = torch.cat([g, g[:, :1]], dim=1)  # degree 4 ← degree 1
    return gamma, c[:, 0], c[:, 1], c[:, 2]


def gamma_phi_from_modes(
    A: torch.Tensor,
    B: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    lam: torch.Tensor,
    L: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """γ (L, 4) and shell φ (L, 4) at size n and depth L."""
    gamma, c1, c2, c3 = schedules_from_modes(A, B, p2, p3, lam, L, n)
    return gamma, phi_from_c(c1, c2, c3)


def param_count(Q: int) -> int:
    """6Q + p_2 + p_3 + λ."""
    return 6 * int(Q) + 3


def warm_modes_from_champ(
    gamma: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    p: float,
    Q: int,
) -> dict[str, torch.Tensor | float]:
    """Exact DST/DCT warm start, then truncate to Q.

    A_{q,d} is the same DST of scalar γ for every degree. λ = 0.
    p_2 = p_3 = p (power_p). B columns are the DCT of unscaled c_k.
    """
    g = gamma.detach().reshape(-1).float()
    cols = [c1, c2, c3]
    L = int(g.numel())
    if any(int(c.numel()) != L for c in cols):
        raise ValueError("champ γ and c1–c3 must share length L")
    Q = int(Q)
    if Q < 1 or Q > L:
        raise ValueError(f"Q={Q} not in 1..L={L}")
    a = analyze(g, "dst")[:Q].to(dtype=torch.float32)
    A = a.reshape(Q, 1).expand(Q, N_DEGREE).contiguous().clone()
    B = torch.stack([analyze(c.detach().reshape(-1), "dct")[:Q].to(dtype=torch.float32) for c in cols], dim=1)
    return {
        "A": A,
        "B": B,
        "p2": float(p),
        "p3": float(p),
        "lam": 0.0,
        "Q": Q,
        "source_L": L,
    }


def qaoa_modes_from_angles(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    Q: int,
) -> dict[str, torch.Tensor]:
    """Fourier QAOA warm start: β from DCT (v_q), γ from DST (u_q), truncate to Q."""
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
        "Q": Q,
        "source_L": L,
    }


def qaoa_angles_from_modes(A: torch.Tensor, B: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """β_ℓ = Σ B_q v_q(ℓ), γ_ℓ = Σ A_q u_q(ℓ). Returns (betas, gammas), each (L,)."""
    if A.ndim != 1 or B.ndim != 1:
        raise ValueError("QAOA Fourier A,B must be 1-D")
    if int(A.shape[0]) != int(B.shape[0]):
        raise ValueError("QAOA A and B must share Q")
    gammas = synthesize(A, L, "dst")
    betas = synthesize(B, L, "dct")
    return betas, gammas


def truncation_rel_l2(schedule: torch.Tensor, Q: int, kind: str) -> float:
    """‖sched − P_Q sched‖ / ‖sched‖ for the low-mode projection at the same L."""
    x = schedule.detach().reshape(-1).to(dtype=torch.float64)
    coeff = analyze(x, kind)[: int(Q)]
    recon = synthesize(coeff, int(x.numel()), kind)
    num = torch.linalg.norm(x - recon)
    den = torch.linalg.norm(x).clamp(min=1e-12)
    return float((num / den).item())


def clamp_exponents_(p2: torch.Tensor, p3: torch.Tensor, lam: torch.Tensor) -> None:
    with torch.no_grad():
        p2.clamp_(P_LO, P_HI)
        p3.clamp_(P_LO, P_HI)
        lam.clamp_(LAM_LO, LAM_HI)


def degree_mad(gamma: torch.Tensor) -> float:
    """Mean |γ_{ℓ,d} − mean_d γ_{ℓ,d}| over layers and d=1..3. Zero iff degrees match."""
    g = gamma.detach()[:, :3].float()
    mu = g.mean(dim=1, keepdim=True)
    return float((g - mu).abs().mean().item())


def _rel_drop(new: float, old: float) -> float:
    """(old − new) / old. Positive when ``new`` is a better (lower) Med."""
    if not (old == old and new == new) or old == 0.0:
        return float("nan")
    return (float(old) - float(new)) / float(old)


def fmws_verdict(
    *,
    q6_l16: dict[int, float],
    champ_l16: dict[int, float],
    q6_l32_n15: float,
    q6_l48_n15: float,
    champ_l32_n15: float,
    fq_l32_n15: float,
    fq_l48_n15: float,
    q4_l32_n15: float | None = None,
    q4_l48_n15: float | None = None,
) -> dict:
    """Ship / kill / needs edit from the Opus smoke gates.

    Lower Med is better. Gates use raw Med on CORE B=1000:

    * Gate A — Q=6 at L=16 is not worse than the free L=16 champ by more than
      0.5% at n=15. Every other CORE n is reported; a >0.5% regression on any
      of them blocks ``ship`` (``needs_edit``) but is not by itself a kill.
    * Gate B — Q=6 Fourier at L=32 and at L=48 each beats the free L=32 champ
      by more than 1% Med at n=15.
    * Gate C — that same Q=6 model also beats Fourier-matched QAOA at n=15
      (otherwise the gain is the Fourier optimizer, which Opus names as the
      main failure mode).

    ``kill`` if gate A fails, or if neither deep depth clears gate B.
    ``ship`` only if A, both B depths, C, and the all-CORE 0.5% check pass.
    ``needs_edit`` covers a split (one depth, or Q=4 instead of Q=6, or a
    Fourier-QAOA tie).
    """
    core = (6, 9, 12, 15)
    rel16 = {n: _rel_drop(q6_l16[n], champ_l16[n]) for n in core}
    # rel_drop > -0.005 means "not worse than 0.5%" (a gain is positive).
    gate_a = rel16[15] >= -0.005 - 1e-12
    core_ok = all(rel16[n] >= -0.005 - 1e-12 for n in core)
    drop32 = _rel_drop(q6_l32_n15, champ_l32_n15)
    drop48 = _rel_drop(q6_l48_n15, champ_l32_n15)
    gate_b32 = drop32 > 0.01
    gate_b48 = drop48 > 0.01
    gate_c32 = q6_l32_n15 < fq_l32_n15
    gate_c48 = q6_l48_n15 < fq_l48_n15
    q4_b32 = None if q4_l32_n15 is None else _rel_drop(q4_l32_n15, champ_l32_n15) > 0.01
    q4_b48 = None if q4_l48_n15 is None else _rel_drop(q4_l48_n15, champ_l32_n15) > 0.01

    if not gate_a or (not gate_b32 and not gate_b48):
        label = "kill"
    elif gate_b32 and gate_b48 and gate_c32 and gate_c48 and core_ok:
        label = "ship"
    else:
        label = "needs_edit"

    return {
        "verdict": label,
        "gate_A_q6_l16_n15_within_0.5pct": gate_a,
        "gate_A_all_core_within_0.5pct": core_ok,
        "rel_vs_champ_l16": {str(n): rel16[n] for n in core},
        "gate_B_l32_n15_drop": drop32,
        "gate_B_l48_n15_drop": drop48,
        "gate_B_l32": gate_b32,
        "gate_B_l48": gate_b48,
        "gate_C_beats_fourier_qaoa_l32_n15": gate_c32,
        "gate_C_beats_fourier_qaoa_l48_n15": gate_c48,
        "q4_also_clears_B_l32": q4_b32,
        "q4_also_clears_B_l48": q4_b48,
        "rule": (
            "kill if Q=6 L16 n15 is worse than the free L16 champ by >0.5%, "
            "or if neither L32 nor L48 beats the free L32 champ by >1% Med at n15; "
            "ship only if both depths clear that 1% bar, both beat Fourier QAOA at n15, "
            "and Q=6 L16 stays within 0.5% of the champ on every CORE n; "
            "otherwise needs_edit"
        ),
    }
