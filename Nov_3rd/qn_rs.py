"""QN-RS: a SNAP residual stacked on matched Fourier QAOA (Opus rank 1).

Ungraded cost, Hamming-shell SNAP. Layer ℓ and mode q are 1-based.
t_ℓ = (ℓ − ½) / L.

    γ_ℓ = Σ_{q=1..Q}  u_q sin((q − ½) π t_ℓ)          # DST-IV, ungraded H_c
    β_ℓ = Σ_{q=1..Q}  v_q cos((q − ½) π t_ℓ)          # DCT-IV
    r2_ℓ = Σ_{q=1..Q'} s_q cos((q − ½) π t_ℓ)
    r3_ℓ = Σ_{q=1..Q'} w_q cos((q − ½) π t_ℓ)
    a(n) = (8 / (n − 1))^p
    φ_{ℓ,k} = −β_ℓ k + a(n) (r2_ℓ k² + r3_ℓ k³)       k = 0..3

U_c^X = W exp(−i γ_ℓ f) W with f the normalized cost (same as qubit QAOA).
Initialization (u, v) = the matched Fourier-QAOA solution and s = w = 0, so
φ_{ℓ,k} = −β_ℓ k and the circuit is that Fourier QAOA. The only terms qubit
QAOA cannot represent are r2 and r3.

Parameter count is 2Q + 2Q' + 1 (the +1 is p). Q = Q' = 6 → 25.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from Nov_3rd import settings as S

P_LO = -1.0
P_HI = 2.0
# Pass bar: scaled residual stays a real fraction of the linear mixer.
RESIDUAL_ALIVE = 0.10
# Below this, r2 and r3 have collapsed (the FC-WS / λ failure mode).
RESIDUAL_DEAD = 0.02


def mode_matrix(
    L: int,
    Q: int,
    kind: str,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Basis (L, Q). 'dst' is sin((q−½)π t_ℓ); 'dct' is cos((q−½)π t_ℓ)."""
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


def synthesize(coeffs: torch.Tensor, L: int, kind: str) -> torch.Tensor:
    """Σ_q coeff_q · mode_q(ℓ). coeffs is (Q,)."""
    c = coeffs.reshape(-1)
    M = mode_matrix(int(L), int(c.numel()), kind, device=c.device, dtype=c.dtype)
    return M @ c


def a_of_n(n: int, p: torch.Tensor | float) -> torch.Tensor:
    """a(n) = (8/(n−1))^p. Equals 1 at n = 9 for every p. Grad flows through p."""
    n = int(n)
    if n <= 1:
        raise ValueError(f"n must be > 1, got {n}")
    log_base = math.log(8.0 / float(n - 1))
    if torch.is_tensor(p):
        return torch.exp(p.to(dtype=torch.float32) * float(log_base))
    return torch.tensor(math.exp(float(p) * log_base), dtype=torch.float32)


def schedules(
    u: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor | float,
    L: int,
    n: int,
) -> dict[str, torch.Tensor]:
    """γ (L,), β, r2, r3 (L,), φ (L, 4) at size n."""
    gamma = synthesize(u, L, "dst")
    beta = synthesize(v, L, "dct")
    r2 = synthesize(s, L, "dct")
    r3 = synthesize(w, L, "dct")
    a = a_of_n(n, p).to(device=gamma.device, dtype=gamma.dtype)
    k = torch.arange(S.B_BITS + 1, device=gamma.device, dtype=gamma.dtype)  # 0..3
    phi = -beta.reshape(-1, 1) * k.reshape(1, -1) + a * (
        r2.reshape(-1, 1) * k.reshape(1, -1) ** 2 + r3.reshape(-1, 1) * k.reshape(1, -1) ** 3
    )
    return {"gamma": gamma, "beta": beta, "r2": r2, "r3": r3, "phi": phi, "a": a}


def param_count(Q: int, Qp: int) -> int:
    """2Q + 2Q' + 1."""
    return 2 * int(Q) + 2 * int(Qp) + 1


def residual_ratio(beta: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, a: torch.Tensor | float) -> float:
    """‖a·(r2, r3)‖ / ‖β‖. The pass bar is ≥ 0.10 at n = 15."""
    scale = a if torch.is_tensor(a) else torch.tensor(float(a), dtype=beta.dtype, device=beta.device)
    resid = torch.cat([(scale * r2).reshape(-1), (scale * r3).reshape(-1)])
    den = beta.reshape(-1).norm().clamp(min=1e-12)
    return float((resid.norm() / den).detach())


def load_fourier_modes(path: str, Q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, v) from a Fourier-QAOA params.pt. u = A (γ DST), v = B (β DCT)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    A = blob["A"].detach().reshape(-1).float()
    B = blob["B"].detach().reshape(-1).float()
    if int(A.numel()) != int(Q) or int(B.numel()) != int(Q):
        raise ValueError(f"Fourier ckpt Q mismatch: A {A.numel()} B {B.numel()} want {Q} @ {path}")
    return A.contiguous(), B.contiguous()


def paired_median_ci(
    ours: np.ndarray,
    base: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap of median(ours) − median(base). Negative means ours is better.

    Resamples instance indices with replacement and recomputes both medians.
    The 95% percentile interval excludes 0 when ``excludes_0`` is true.
    """
    a = np.asarray(ours, dtype=np.float64).reshape(-1)
    b = np.asarray(base, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"paired arrays must share a nonempty shape, got {a.shape} {b.shape}")
    point = float(np.median(a) - np.median(b))
    rng = np.random.default_rng(int(seed))
    n = int(a.size)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    deltas = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    lo = float(np.quantile(deltas, alpha / 2.0))
    hi = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    return {
        "delta_med": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_0": bool(hi < 0.0 or lo > 0.0),
        "ours_better": bool(point < 0.0 and hi < 0.0),
        "n_boot": int(n_boot),
        "B": n,
    }


def seed_passes(n12: dict, n15: dict, ratio: float) -> bool:
    """One seed: n12 and n15 Med below Fourier QAOA, both CIs exclude 0 on the better side, residual alive."""
    return bool(n12.get("ours_better") and n15.get("ours_better") and float(ratio) >= RESIDUAL_ALIVE - 1e-12)


def qnrs_verdict(seed_rows: list[dict], *, depth_tag: str) -> dict:
    """ship / kill / needs_edit for one depth.

    ``seed_rows`` entries need keys med_n12, med_n15, fq_n12, fq_n15, boot_n12,
    boot_n15 (paired_median_ci dicts), residual_ratio.

    L=16 is the decisive smoke. A later caller maps an L=16 pass plus an L=32
    failure to needs_edit, and both passes to ship. This function judges one depth:

    * pass_depth — ≥ 2/3 seeds clear seed_passes
    * kill — no seed has n15 Med below the Fourier QAOA median, or every seed's
      residual ratio is < RESIDUAL_DEAD
    * needs_edit — anything between (a real but unreplicated gain, or a gain
      whose residual is below the 10% bar)
    """
    if not seed_rows:
        raise ValueError("no seeds")
    flags = []
    n15_better = []
    ratios = []
    for row in seed_rows:
        ratio = float(row["residual_ratio"])
        ratios.append(ratio)
        b12 = row["boot_n12"]
        b15 = row["boot_n15"]
        ok = seed_passes(b12, b15, ratio)
        flags.append(ok)
        n15_better.append(bool(row["med_n15"] < row["fq_n15"]))
    n_pass = int(sum(flags))
    n_seeds = len(seed_rows)
    need = max(2, (2 * n_seeds + 2) // 3)  # ≥2/3, rounded up; 3 → 2
    # For exactly 3 seeds Opus says ≥2/3, which is 2.
    if n_seeds == 3:
        need = 2
    pass_depth = n_pass >= need
    no_n15_gain = not any(n15_better)
    residuals_dead = all(r < RESIDUAL_DEAD for r in ratios)
    if pass_depth:
        label = "pass"
    elif no_n15_gain or residuals_dead:
        label = "kill"
    else:
        label = "needs_edit"
    return {
        "depth": depth_tag,
        "verdict": label,
        "n_pass_seeds": n_pass,
        "n_seeds": n_seeds,
        "need": need,
        "seed_pass": flags,
        "n15_better_than_fourier": n15_better,
        "residual_ratios": ratios,
        "residuals_dead": residuals_dead,
        "no_n15_gain": no_n15_gain,
        "rule": (
            "pass if ≥2/3 seeds have n12 and n15 Med below Fourier QAOA with paired "
            "bootstrap 95% CI of (Med−Med_FQ) excluding 0 on the better side, and "
            f"‖a(15)·(r2,r3)‖/‖β‖ ≥ {RESIDUAL_ALIVE:.2f}; "
            "kill if n15 is no better than Fourier QAOA on every seed, or every "
            f"residual ratio is < {RESIDUAL_DEAD:.2f}; otherwise needs_edit"
        ),
    }


def combine_verdicts(v16: dict | None, v32: dict | None) -> str:
    """Overall ship / kill / needs_edit. L=32 is only required after an L=16 pass."""
    if v16 is None:
        return "needs_edit"
    if v16["verdict"] == "kill":
        return "kill"
    if v16["verdict"] != "pass":
        return "needs_edit"
    if v32 is None:
        return "needs_edit"
    if v32["verdict"] == "pass":
        return "ship"
    if v32["verdict"] == "kill":
        return "needs_edit"
    return "needs_edit"
