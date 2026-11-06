"""Power-law long-range Ising on the cavity chain, plus the X / XX driver.

    H = Σ_{i<j} (J_ij / |i-j|^α) Z_i Z_j − λ Σ_i X_i
        + μ Σ_c Σ_{i<j in cavity c} X_i X_j

J_ij ∈ {±1} is an open-chain spin glass. Qubit i sits at site i, qubit 0 is
the MSB, and cavity c owns sites [3c, 3c+3). Pairs that cross a cavity
boundary are ordinary chain pairs; they are not dropped. α = 1.5.

The physical diagonal is that ZZ sum. The circuit phases use Cn = H_ZZ / n,
which is the same family {exp(−i γ H_ZZ)} with γ rescaled by n. ⟨H⟩ and E_0
always use the un-normalized H above.

Readout is the Z histogram (every ZZ) and the X histogram (⟨X⟩ and the
intra-cavity ⟨XX⟩). There is no Y.

Walsh spectrum of H_ZZ is exactly degree 2, so the graded cost has one live
weight γ_2 per layer. The Hamming shell is the rest of Walsh SNAP.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    dense_H,
    expectation_H,
    fourier_state,
    hamiltonian_matvec,
    hva_state,
    init_angles,
    prepare_state,
    snap_state,
    state_fidelity,
    walsh_power_by_degree,
)

ALPHA = 1.5
SEED_BASE = 1_500_003_017
SEED_STRIDE = 100_003

# Decision bars for the breadth smoke. R is Medε_SNAP / Medε_baseline.
PASS_R_DRIVER = 0.90
PASS_R_PURE = 0.98
KILL_R = 0.98
TRANSFER_SLACK = 0.10
PEAK_LT = 0.5
PROD_GAP_MIN = 0.03
# A 5-seed seal is warranted when the smoke call sits inside this margin.
SEAL_MARGIN = 0.03


def instance_seed(n: int, index: int) -> int:
    """Disjoint streams per n. Index 0.. is the whole pool for that n."""
    return int(SEED_BASE + int(n) * SEED_STRIDE + int(index))


def couplings(n: int, index: int, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Return (J, K). J_ij ∈ {±1, 0}, K_ij = J_ij / |i-j|^α, both symmetric."""
    if int(n) % 3 != 0:
        raise ValueError(f"n={n} is not a multiple of the cavity size 3")
    rng = np.random.default_rng(instance_seed(n, index))
    J = np.zeros((n, n), dtype=np.int8)
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sign = int(rng.integers(0, 2)) * 2 - 1
            J[i, j] = J[j, i] = sign
            K[i, j] = K[j, i] = float(sign) / float(j - i) ** float(alpha)
    return J, K


def zz_diag_from_K(K: np.ndarray) -> np.ndarray:
    """H_ZZ(z) for every bitstring. Qubit 0 is the MSB of the integer index."""
    n = int(K.shape[0])
    N = 1 << n
    idx = np.arange(N, dtype=np.int64)
    z = np.empty((n, N), dtype=np.float64)
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1
        z[q] = 1.0 - 2.0 * bit
    ii, jj = np.triu_indices(n, 1)
    return (K[ii, jj][:, None] * z[ii] * z[jj]).sum(axis=0)


def make_instance(n: int, index: int, alpha: float = ALPHA) -> dict:
    J, K = couplings(n, index, alpha=alpha)
    diag = zz_diag_from_K(K)
    return {"n": int(n), "index": int(index), "seed": instance_seed(n, index), "J": J, "K": K, "diag": diag}


def cross_cavity_weight(K: np.ndarray) -> float:
    """Fraction of Σ |K_ij| whose endpoints lie in different cavities."""
    n = int(K.shape[0])
    total = 0.0
    cross = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(float(K[i, j]))
            total += w
            if i // 3 != j // 3:
                cross += w
    if total <= 0.0:
        return 0.0
    return float(cross / total)


def cost_diag(diag: torch.Tensor, n: int) -> torch.Tensor:
    """Cn = H_ZZ / n. γ then labels the same unitaries as a raw ZZ phase."""
    return diag / float(n)


def field_coeff(lam: float) -> float:
    """Coefficient of Σ X in H. The grid stores λ > 0 from the −λ Σ X term."""
    return -float(lam)


def product_energy(theta: np.ndarray, K: np.ndarray, lam: float, mu: float) -> float:
    """XZ product state. ⟨Z⟩=cos θ, ⟨X⟩=sin θ, ⟨Y⟩=0."""
    cz = np.cos(theta)
    sx = np.sin(theta)
    e = 0.5 * float(cz @ K @ cz)
    e -= float(lam) * float(sx.sum())
    n = int(theta.shape[0])
    for c in range(n // 3):
        s = sx[3 * c : 3 * c + 3]
        e += float(mu) * float(s[0] * s[1] + s[0] * s[2] + s[1] * s[2])
    return float(e)


def _coordinate_descent(theta: np.ndarray, K: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """Exact single-spin updates. Each spin sees α cos θ + β sin θ."""
    th = np.array(theta, dtype=np.float64, copy=True)
    n = int(th.shape[0])
    cz = np.cos(th)
    sx = np.sin(th)
    mates = []
    for i in range(n):
        cav = i // 3
        mates.append([3 * cav + j for j in range(3) if 3 * cav + j != i])
    for _ in range(64):
        max_d = 0.0
        for i in range(n):
            alpha = float(K[i] @ cz)
            beta = -float(lam) + float(mu) * float(sum(sx[m] for m in mates[i]))
            new = math.atan2(-beta, -alpha)
            d = abs(math.atan2(math.sin(new - th[i]), math.cos(new - th[i])))
            if d > max_d:
                max_d = d
            th[i] = new
            cz[i] = math.cos(new)
            sx[i] = math.sin(new)
        if max_d < 1e-14:
            break
    return th


def best_product(K: np.ndarray, diag: np.ndarray, lam: float, mu: float, seed: int) -> float:
    """Lowest product energy found. Includes the exact best bitstring."""
    from scipy.optimize import minimize

    n = int(K.shape[0])
    x_best = int(np.argmin(diag))
    bits = np.array([(x_best >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.float64)
    starts = [
        np.zeros(n),
        np.full(n, np.pi),
        np.full(n, 0.5 * np.pi),
        np.full(n, -0.5 * np.pi),
        bits * np.pi,
        (1.0 - bits) * np.pi,
    ]
    rng = np.random.default_rng(int(seed) + 17 + int(round(1000 * float(lam))) + int(round(100 * float(mu))))
    for _ in range(6):
        starts.append(rng.uniform(-np.pi, np.pi, size=n))
    polished = [_coordinate_descent(th0, K, lam, mu) for th0 in starts]
    best = min(product_energy(th, K, lam, mu) for th in polished)
    order = sorted(range(len(polished)), key=lambda i: product_energy(polished[i], K, lam, mu))
    seen: list[np.ndarray] = []
    for i in order:
        th = polished[i]
        if any(np.max(np.abs(np.sin(0.5 * (th - s)))) < 0.05 for s in seen):
            continue
        seen.append(th)
        if len(seen) > 3:
            break
        res = minimize(lambda t: product_energy(t, K, lam, mu), th, method="L-BFGS-B")
        if np.isfinite(res.fun):
            best = min(best, float(res.fun))
    return float(best)


def ground_state(diag: np.ndarray, n: int, lam: float, mu: float) -> dict:
    """Lowest eigenpair. ψ is real, bitstring order, qubit 0 = MSB."""
    diag = np.asarray(diag, dtype=np.float64).reshape(-1)
    N = 1 << int(n)
    if diag.size != N:
        raise ValueError(f"diag {diag.size} != 2^{n}")
    lx = field_coeff(lam)
    mx = float(mu)
    if int(n) <= 9:
        H = dense_H(diag, n, lambda_x=lx, mu_xx=mx)
        evals, evecs = np.linalg.eigh(H)
        e0 = float(evals[0])
        psi = np.real(np.asarray(evecs[:, 0], dtype=np.float64))
    else:
        from scipy.sparse.linalg import LinearOperator, eigsh

        def mv(v):
            return hamiltonian_matvec(np.asarray(v, dtype=np.float64).reshape(-1), diag, n, lambda_x=lx, mu_xx=mx)

        op = LinearOperator((N, N), matvec=mv, dtype=np.float64)
        v0 = np.full(N, 1.0 / math.sqrt(N))
        evals, evecs = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=max(4000, N), ncv=min(N - 1, 40), v0=v0)
        e0 = float(evals[0])
        psi = np.real(np.asarray(evecs[:, 0], dtype=np.float64))
        resid = float(np.linalg.norm(mv(psi) - e0 * psi))
        if resid > 1e-6 * max(1.0, abs(e0)):
            raise RuntimeError(f"eigsh residual {resid:.3e} at n={n}")
    psi = psi / float(np.linalg.norm(psi))
    if float(psi.sum()) < 0.0:
        psi = -psi
    if e0 > float(diag.min()) + 1e-6:
        raise RuntimeError(f"E0 {e0} above a diagonal entry {float(diag.min())}")
    return {"E0": e0, "psi": psi, "peak": float(np.max(psi ** 2))}


def quantum_observables(K: np.ndarray, diag: np.ndarray, n: int, lam: float, mu: float, seed: int) -> dict:
    gs = ground_state(diag, n, lam, mu)
    e_prod = best_product(K, diag, lam, mu, seed=seed)
    e0 = float(gs["E0"])
    if e_prod < e0 - 1e-5:
        raise RuntimeError(f"product {e_prod} below E0 {e0} at n={n}")
    denom = max(abs(e0), 1e-8)
    return {
        "E0": e0,
        "peak": float(gs["peak"]),
        "E_prod": float(e_prod),
        "E_classical": float(np.min(diag)),
        "prod_gap": float((e_prod - e0) / denom),
        "prod_gap_density": float((e_prod - e0) / float(n)),
    }


def quantum_summary(peaks: np.ndarray, gaps: np.ndarray) -> dict:
    """Median bitstring peak and median relative product gap on one pool."""
    peaks = np.asarray(peaks, dtype=np.float64).reshape(-1)
    gaps = np.asarray(gaps, dtype=np.float64).reshape(-1)
    peak_med = float(np.median(peaks))
    gap_med = float(np.median(gaps))
    return {
        "B": int(peaks.size),
        "peak_med": peak_med,
        "peak_max": float(np.max(peaks)),
        "frac_peak_below": float(np.mean(peaks < PEAK_LT)),
        "gap_med": gap_med,
        "gap_min": float(np.min(gaps)),
        "frac_gap_ok": float(np.mean(gaps >= PROD_GAP_MIN - 1e-15)),
        "pass": bool(peak_med < PEAK_LT and gap_med >= PROD_GAP_MIN),
    }


def snap_angles_from_gamma2(gamma2: torch.Tensor, phi: torch.Tensor) -> dict:
    """Build the (L, 4) Walsh table. Only degree 2 is nonzero."""
    if gamma2.ndim != 1:
        raise ValueError("gamma2 must be (L,)")
    if phi.ndim != 2 or int(phi.shape[1]) != 4:
        raise ValueError(f"phi shape {tuple(phi.shape)}")
    if int(phi.shape[0]) != int(gamma2.shape[0]):
        raise ValueError("gamma2/phi length mismatch")
    gw = gamma2.new_zeros(int(gamma2.shape[0]), 4)
    gw[:, 1] = gamma2
    return {"gamma_w": gw, "phi": phi}


def init_snap(L: int, dtype=torch.float64, device=None) -> dict:
    """Paper shell init. γ_2 = −0.01; other Walsh weights stay 0."""
    L = int(L)
    k = torch.arange(4, dtype=dtype, device=device)
    phi = (-INIT_BETA * k).reshape(1, 4).expand(L, 4).contiguous().clone()
    return {
        "gamma2": torch.full((L,), INIT_GAMMA, dtype=dtype, device=device),
        "phi": phi,
    }


def prepare_snap(Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    table = snap_angles_from_gamma2(angles["gamma2"], angles["phi"])
    return snap_state(Cn, table["gamma_w"], table["phi"])


def prepare(method: str, Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    if method == "snap":
        return prepare_snap(Cn, angles)
    return prepare_state(method, Cn, angles)


def init_method(method: str, L: int, dtype=torch.float64, device=None) -> dict:
    if method == "snap":
        return init_snap(L, dtype=dtype, device=device)
    if method not in ("fourier", "hva"):
        raise ValueError(method)
    return init_angles(method, L, dtype=dtype, device=device)


def parameter_list(method: str, angles: dict) -> list[torch.nn.Parameter]:
    keys = {
        "fourier": ("betas", "gammas"),
        "hva": ("beta_x", "beta_xx", "gammas"),
        "snap": ("gamma2", "phi"),
    }[method]
    params = []
    for k in keys:
        p = torch.nn.Parameter(angles[k].detach().clone())
        angles[k] = p
        params.append(p)
    return params


def expectation(psi: torch.Tensor, diag: torch.Tensor, n: int, lam: float, mu: float) -> torch.Tensor:
    return expectation_H(psi, diag, n, lambda_x=field_coeff(lam), mu_xx=float(mu))


@torch.no_grad()
def instance_eps(
    method: str,
    angles: dict,
    diag: torch.Tensor,
    e0: torch.Tensor,
    n: int,
    lam: float,
    mu: float,
    *,
    chunk: int = 8,
) -> torch.Tensor:
    """ε = ⟨H⟩ − E_0 per instance. diag is the raw ZZ table, shape (B, 2^n)."""
    out = []
    B = int(diag.shape[0])
    for s in range(0, B, int(chunk)):
        d = diag[s : s + chunk]
        psi = prepare(method, cost_diag(d, n), angles)
        h = expectation(psi, d, n, lam, mu)
        out.append(h - e0[s : s + chunk].to(dtype=h.dtype, device=h.device))
    return torch.cat(out, dim=0)


def shell_nonlinearity(phi: np.ndarray) -> float:
    """Median over layers of the RMS residual of φ_k against a + b k."""
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != 4:
        raise ValueError(f"phi shape {phi.shape}")
    k = np.arange(4, dtype=np.float64)
    rms = []
    for row in phi:
        slope, intercept = np.polyfit(k, row, 1)
        pred = intercept + slope * k
        rms.append(float(np.sqrt(np.mean((row - pred) ** 2))))
    return float(np.median(rms))


def ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 1e-12:
        return float("inf")
    return float(num) / float(den)


def transfer_collapses(r12: float, r6: float, r9: float, slack: float = TRANSFER_SLACK) -> bool:
    """The frozen ratio ran away from both training sizes."""
    if not all(np.isfinite(x) for x in (r12, r6, r9)):
        return True
    return float(r12) > max(float(r6), float(r9)) + float(slack)


def _baseline_ok(rec: dict, bar: float) -> bool:
    """n12 under `bar`, train sizes still a win at 0.98, transfer held."""
    r6 = float(rec["6"])
    r9 = float(rec["9"])
    r12 = float(rec["12"])
    if not (r12 <= float(bar) and r6 <= KILL_R and r9 <= KILL_R):
        return False
    if transfer_collapses(r12, r6, r9):
        return False
    return True


def cell_call(mu: float, quantum_pass: bool, ratios: dict) -> dict:
    """One (λ, μ, L) cell. `ratios` maps baseline name → {6, 9, 12} R."""
    reasons = []
    driver = abs(float(mu)) > 1e-15
    passed_by = []
    if not quantum_pass:
        reasons.append("quantumness fails (median peak or product gap)")
    else:
        if driver:
            for name, bar in (("fourier", PASS_R_DRIVER), ("hva", PASS_R_DRIVER)):
                rec = ratios.get(name)
                if rec is None:
                    continue
                if _baseline_ok(rec, bar):
                    passed_by.append(name)
            if not passed_by:
                reasons.append(
                    f"no baseline has n12 R≤{PASS_R_DRIVER:.2f} with train R≤{KILL_R:.2f} and transfer held"
                )
        else:
            rec = ratios.get("fourier")
            if rec is not None and _baseline_ok(rec, PASS_R_PURE):
                passed_by.append("fourier")
            else:
                reasons.append(
                    f"pure family misses n12 R≤{PASS_R_PURE:.2f} with train R≤{KILL_R:.2f} and transfer held"
                )
    kind = "driver" if driver else "pure"
    ok = bool(quantum_pass and passed_by)
    return {
        "pass": ok,
        "kind": kind,
        "passed_by": passed_by,
        "reason": "ok" if ok else "; ".join(reasons),
    }


def kill_grid(l4_cells: list[dict]) -> bool:
    """KILL when every (λ, μ) has Fourier R>0.98 at both n6 and n9, L=4."""
    if not l4_cells:
        return False
    for cell in l4_cells:
        r = cell["ratios"]["fourier"]
        if not (float(r["6"]) > KILL_R and float(r["9"]) > KILL_R):
            return False
    return True


def decide(cells: list[dict]) -> dict:
    """Overall smoke call.

    `cells` entries need: lam, mu, L, quantum_pass, ratios, and optionally
    per-seed n12 ratios under per_seed[baseline] = list of R.
    A driver PASS (μ≠0, R≤0.90) and a pure-family PASS (μ=0, R≤0.98) are
    each enough. KILL is only the train-size sweep failure at L=4.
    """
    calls = []
    for c in cells:
        call = cell_call(float(c["mu"]), bool(c["quantum_pass"]), c["ratios"])
        calls.append({**c, "call": call})
    driver_hits = [c for c in calls if c["call"]["pass"] and c["call"]["kind"] == "driver"]
    pure_hits = [c for c in calls if c["call"]["pass"] and c["call"]["kind"] == "pure"]
    l4 = [c for c in calls if int(c["L"]) == 4]
    killed = kill_grid(l4) if l4 else False
    if driver_hits:
        verdict = "PASS"
        reason = "μ≠0 driver clears n12 at L=2 or L=4"
    elif pure_hits:
        verdict = "PASS"
        reason = "μ=0 pure family clears n12 at the 0.98 bar"
    elif killed:
        verdict = "KILL"
        reason = "L=4 Fourier R>0.98 at both n6 and n9 for every (λ, μ)"
    else:
        verdict = "needs_edit"
        reason = "no cell clears its bar, and the L=4 train sweep is not a KILL"
    return {
        "verdict": verdict,
        "reason": reason,
        "driver_passes": [_cell_tag(c) for c in driver_hits],
        "pure_passes": [_cell_tag(c) for c in pure_hits],
        "kill": bool(killed),
        "seal_warranted": seal_warranted(verdict, calls),
        "cells": calls,
    }


def _cell_tag(c: dict) -> str:
    bases = ",".join(c["call"]["passed_by"])
    return f"λ={float(c['lam']):.2f} μ={float(c['mu']):.2f} L={int(c['L'])} via {bases}"


def seal_warranted(verdict: str, calls: list[dict]) -> dict:
    """Yes when a PASS should be sealed, or a near-miss could flip."""
    if verdict == "PASS":
        close = _seed_dispute(calls, only_passing=True)
        return {
            "warranted": True,
            "why": (
                "3-seed PASS. A 5-seed seal is the next step before freezing this H"
                + ("; at least one passing cell has seeds that disagree or sit near the bar" if close else
                   "; the passing seeds sit together under the bar")
            ),
        }
    if verdict == "KILL":
        return {"warranted": False, "why": "L=4 train ratios are past 0.98 on every (λ, μ). Another two seeds would not reopen a killed family."}
    close = _near_miss(calls)
    if close:
        return {"warranted": True, "why": close}
    return {
        "warranted": False,
        "why": (
            "No quantum cell is within 0.03 of its n12 bar with the lead still holding at larger n. "
            "A 5-seed seal would not change the smoke call."
        ),
    }


def _deeper_reverses(c: dict, calls: list[dict]) -> bool:
    """A deeper L of the same (λ, μ) has Fourier R>1, so this depth is not a family."""
    for other in calls:
        if float(other["lam"]) != float(c["lam"]) or float(other["mu"]) != float(c["mu"]):
            continue
        if int(other["L"]) <= int(c["L"]):
            continue
        rf = (other.get("ratios") or {}).get("fourier", {}).get("12")
        if rf is not None and np.isfinite(rf) and float(rf) > 1.0:
            return True
    return False


def _near_miss(calls: list[dict]) -> str:
    best = None
    for c in calls:
        if not c.get("quantum_pass"):
            continue
        if _deeper_reverses(c, calls):
            continue
        driver = abs(float(c["mu"])) > 1e-15
        bar = PASS_R_DRIVER if driver else PASS_R_PURE
        names = ("fourier", "hva") if driver else ("fourier",)
        for name in names:
            rec = c["ratios"].get(name)
            if rec is None or not np.isfinite(rec.get("12", np.nan)):
                continue
            if transfer_collapses(rec["12"], rec["6"], rec["9"]):
                continue
            gap = float(rec["12"]) - bar
            if best is None or gap < best[0]:
                best = (gap, c, name, bar)
    if best is None:
        return ""
    gap, c, name, bar = best
    if gap <= SEAL_MARGIN:
        return (
            f"closest quantum cell λ={float(c['lam']):.2f} μ={float(c['mu']):.2f} "
            f"L={int(c['L'])} {name} n12 R={float(c['ratios'][name]['12']):.3f} "
            f"is within {SEAL_MARGIN:.2f} of {bar:.2f}, and the lead has not reversed at deeper L"
        )
    return ""


def _seed_dispute(calls: list[dict], only_passing: bool) -> bool:
    for c in calls:
        if only_passing and not c["call"]["pass"]:
            continue
        per = c.get("per_seed") or {}
        driver = abs(float(c["mu"])) > 1e-15
        for name in c["call"]["passed_by"] or (("fourier", "hva") if driver else ("fourier",)):
            bar = PASS_R_DRIVER if driver else PASS_R_PURE
            rs = per.get(name) or []
            finite = [float(r) for r in rs if np.isfinite(r)]
            if not finite:
                continue
            if min(finite) <= bar < max(finite):
                return True
            if min(abs(r - bar) for r in finite) <= SEAL_MARGIN:
                return True
    return False


def self_check() -> None:
    """Pauli identity, Walsh degree, mixer nesting, classical λ=μ=0 limit."""
    torch.manual_seed(0)
    n = 3
    inst = make_instance(n, 0)
    J, K, diag = inst["J"], inst["K"], inst["diag"]
    if set(np.unique(J).tolist()) - {0} != {-1, 1}:
        raise AssertionError(f"J values {np.unique(J)}")
    # Hand-built Pauli H. Kronecker order is qubit 0 (MSB) first.
    I = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    def embed(ops: dict[int, np.ndarray]) -> np.ndarray:
        mats = [ops.get(q, I) for q in range(n)]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    lam, mu = 0.5, -0.5
    H = np.zeros((8, 8), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            H = H + (J[i, j] / float(j - i) ** ALPHA) * embed({i: Z, j: Z})
        H = H + (-lam) * embed({i: X})
    for i, j in ((0, 1), (0, 2), (1, 2)):
        H = H + mu * embed({i: X, j: X})
    H_code = dense_H(diag, n, lambda_x=field_coeff(lam), mu_xx=mu)
    if np.max(np.abs(H - H_code)) > 1e-10:
        raise AssertionError(f"Pauli H mismatch {np.max(np.abs(H - H_code))}")
    # Site 2 and 3 are the cavity cut on n=6. That bond is a chain bond.
    K6 = couplings(6, 1)[1]
    if abs(K6[2, 3]) <= 0.0 or cross_cavity_weight(K6) <= 0.2:
        raise AssertionError("cross-cavity chain bond missing")
    pwr = walsh_power_by_degree(diag)
    if float(pwr[2]) < 1.0 - 1e-10 or float(pwr.sum() - pwr[2]) > 1e-10:
        raise AssertionError(f"ZZ Walsh power is not pure degree 2: {pwr}")

    Et = torch.tensor(diag, dtype=torch.float64).reshape(1, -1)
    psi = torch.tensor(np.full((1, 8), 1.0 / math.sqrt(8)), dtype=torch.complex128)
    h_plus = float(expectation(psi, Et, n, lam, mu))
    # |+⟩: ⟨ZZ⟩=0, ⟨X⟩=1, ⟨XX⟩=1, three XX pairs.
    h_formula = float(-lam * n + mu * 3)
    if abs(h_plus - h_formula) > 1e-8:
        raise AssertionError(f"|+⟩ {h_plus} != {h_formula}")
    vv = np.linalg.eigh(H)[1][:, 0]
    v = torch.tensor(vv, dtype=torch.complex128).reshape(1, -1)
    h_hist = float(expectation(v, Et, n, lam, mu))
    h_dense = float(np.vdot(vv, H @ vv).real)
    if abs(h_hist - h_dense) > 1e-8:
        raise AssertionError(f"hist {h_hist} != dense {h_dense}")

    # λ=μ=0 is a classical bitstring ground state. Product gap must vanish.
    obs0 = quantum_observables(K, diag, n, 0.0, 0.0, seed=0)
    if obs0["peak"] < 1.0 - 1e-8 or obs0["prod_gap"] > 1e-8:
        raise AssertionError(f"classical limit failed {obs0}")

    # Paper init: Fourier, HVA (β^XX=0), and degree-2 shell SNAP are one state.
    Cn = cost_diag(Et, n)
    L = 2
    g = torch.full((L,), INIT_GAMMA, dtype=torch.float64)
    b = torch.full((L,), INIT_BETA, dtype=torch.float64)
    sf = fourier_state(Cn, b, g)
    sh = hva_state(Cn, b, torch.zeros(L, dtype=torch.float64), g)
    snap = init_snap(L)
    ss = prepare_snap(Cn, snap)
    if float(state_fidelity(sf, sh)[0]) < 1 - 1e-10:
        raise AssertionError("HVA init is not Fourier")
    if float(state_fidelity(sf, ss)[0]) < 1 - 1e-10:
        raise AssertionError("SNAP init is not Fourier on a degree-2 cost")

    # A bent shell must move off that state, and β^XX must have a gradient.
    phi = snap["phi"].clone()
    phi[0, 2] = 0.4
    sb = prepare_snap(Cn, {"gamma2": snap["gamma2"], "phi": phi})
    if float(state_fidelity(sf, sb)[0]) > 1 - 1e-6:
        raise AssertionError("bent shell did not change the state")
    bx = torch.tensor([0.2], dtype=torch.float64)
    bxx = torch.tensor([0.15], dtype=torch.float64, requires_grad=True)
    gg = torch.tensor([-0.05], dtype=torch.float64)
    loss = expectation(hva_state(Cn, bx, bxx, gg), Et, n, lam, mu).sum()
    loss.backward()
    if bxx.grad is None or abs(float(bxx.grad)) < 1e-8:
        raise AssertionError("β^XX gradient vanished")

    # n=6 dense ground state agrees with the matvec, and E_prod ≥ E0.
    inst6 = make_instance(6, 2)
    obs6 = quantum_observables(inst6["K"], inst6["diag"], 6, 1.0, -0.5, seed=inst6["seed"])
    e_dense = float(np.linalg.eigvalsh(dense_H(inst6["diag"], 6, lambda_x=-1.0, mu_xx=-0.5))[0])
    if abs(obs6["E0"] - e_dense) > 1e-8:
        raise AssertionError(f"n6 E0 {obs6['E0']} != {e_dense}")
    if obs6["E_prod"] + 1e-8 < obs6["E0"]:
        raise AssertionError("product below the ground state")
    # Single-spin update does not raise the product energy.
    th = np.linspace(-1.0, 1.3, 6)
    e_before = product_energy(th, inst6["K"], 1.0, -0.5)
    th2 = _coordinate_descent(th, inst6["K"], 1.0, -0.5)
    if product_energy(th2, inst6["K"], 1.0, -0.5) > e_before + 1e-8:
        raise AssertionError("coordinate descent raised the energy")
