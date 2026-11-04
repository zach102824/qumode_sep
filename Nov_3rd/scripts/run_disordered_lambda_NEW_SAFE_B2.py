#!/usr/bin/env python3
"""Disordered-λ robustness of B=2 Walsh SNAP at NEW_SAFE.

Same Hamiltonian and protocol as the Hamming-shell run (PR #49):

    H = diag(unsat) − Σ_i λ_i X_i
    λ_i = 0.75 (1 + δ u_i),   u_i ~ Unif[-1, 1]

One shared pattern u (NumPy seed 20261105), δ ∈ {0.15, 0.30}, μ = 0.
Train n ∈ {6, 9} only. Freeze n = 12. Five seeds. Matched Fourier QAOA.
L = 2 is the gate. L = 4 is reported and is not the gate.

The SNAP arm is the B=2 parameterization from the coefficient ablation
(PR #57), not the four-phase Hamming shell:

    φ_k = c1 k + c2 k²    on popcount k = 0..3

Coordinates are the orthonormal basis of that span, so L-BFGS sees the
same Euclidean metric on φ as the locked shell. The cost pulse stays the
Walsh grading γ_{w=1..4}. n = 21 is not loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    angles_to_lists,
    cn_from_unsat,
    expectation_H,
    ground_bundle,
    init_angles,
    lambda_of_state,
    method_parameter_list,
    prepare_state,
    self_check,
    snap_state,
    snap_state_from_components,
    summarize_eps,
    walsh_degree_components,
)

RUN = S.OUT / "quantum_h_explore" / "disordered_lambda_NEW_SAFE_B2"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "B2")
DELTAS = (0.15, 0.30)
DEPTHS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_N = 12
FREEZE_B = 128
DISORDER_SEED = 20261105
N_PATTERN = 48
LAMBDA0 = 0.75
MU = 0.0
PROTOCOL = "disordered_lambda_NEW_SAFE_B2"
R_PASS = 0.85
R_KILL = 0.98
LAMBDA_PASS = 0.80
LAMBDA_KILL = 0.90

# Clean NEW_SAFE, B=2, from PR #57. Not remeasured here.
B2_CLEAN = {
    "lambda_med_n12": 0.7477,
    2: {"fourier": 0.6876, "B2": 0.4261, "R": 0.6197},
    4: {"fourier": 0.3172, "B2": 0.1958, "R": 0.6174},
}
# Same disorder, Hamming shell (B=4 φ), from PR #49. Not remeasured here.
SHELL49 = {
    0.15: {
        "lambda_med_n12": 0.7586,
        2: {"fourier": 0.6772, "snap": 0.4151, "R": 0.6130},
        4: {"fourier": 0.3125, "snap": 0.2578, "R": 0.8249},
    },
    0.30: {
        "lambda_med_n12": 0.7610,
        2: {"fourier": 0.6694, "snap": 0.4128, "R": 0.6167},
        4: {"fourier": 0.3130, "snap": 0.3036, "R": 0.9699},
    },
}

# Orthonormal rows spanning {k, k^2} on popcount k=0..3. φ = c @ B2_BASIS.
_K = np.arange(4, dtype=np.float64)
_RAW = np.stack([_K, _K ** 2], axis=1)
_Q, _R = np.linalg.qr(_RAW, mode="reduced")
for _j in range(2):
    if float(_Q[:, _j] @ _RAW[:, _j]) < 0.0:
        _Q[:, _j] *= -1.0
B2_BASIS = np.ascontiguousarray(_Q.T)  # (2, 4)
_PAPER_PHI = -INIT_BETA * _K
B2_INIT = np.ascontiguousarray(_PAPER_PHI @ B2_BASIS.T)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[ns-dlam-b2]", *a, flush=True)


def tag_delta(delta: float) -> str:
    return f"d{delta:.2f}".replace(".", "p")


def unit_pattern() -> np.ndarray:
    """Shared u_i ~ Unif[-1, 1]. δ_i = δ u_i is then Unif[-δ, δ]."""
    return np.random.default_rng(DISORDER_SEED).uniform(-1.0, 1.0, size=N_PATTERN)


def positive_lambda(n: int, delta: float, unit: np.ndarray) -> np.ndarray:
    """λ_i = 0.75 (1 + δ u_i) > 0 for |δ u_i| < 1."""
    if int(n) == 21:
        raise RuntimeError("n=21 is out of scope for this run")
    lam = LAMBDA0 * (1.0 + float(delta) * np.asarray(unit[:n], dtype=np.float64))
    if np.any(lam <= 0.0):
        raise RuntimeError(f"nonpositive λ at δ={delta}: {lam}")
    return lam


def x_coeff(n: int, delta: float, unit: np.ndarray) -> np.ndarray:
    """Coefficients in H = diag(unsat) + Σ c_i X_i, c_i = −λ_i."""
    return -positive_lambda(n, delta, unit)


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    if int(n) == 21:
        raise RuntimeError("n=21 is out of scope for this run")
    if int(n) not in (6, 9, 12):
        raise RuntimeError(f"refusing n={n}; this run loads only 6, 9, 12")
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def phi_from_b2(c: torch.Tensor) -> torch.Tensor:
    """c (L, 2) → popcount phases (L, 4) in span{k, k²}."""
    basis = c.new_tensor(B2_BASIS)
    return c @ basis


def shell_rms(phi) -> float:
    """RMS distance of φ_k from the best line a + b k, pooled over layers.

    Zero means the trained shell is a linear X mixer. B=2 can leave that
    line only through the k² coefficient.
    """
    arr = np.asarray(phi, dtype=np.float64).reshape(-1, 4)
    k = np.arange(4.0)
    design = np.stack([np.ones(4), k], axis=1)
    resid = []
    for row in arr:
        coef, *_ = np.linalg.lstsq(design, row, rcond=None)
        resid.append(row - design @ coef)
    r = np.concatenate(resid)
    return float(np.sqrt(np.mean(r * r)))


def physical_shell_gap(phi) -> float:
    """Mean over layers of RMS residual outside span{1, k, k²}.

    A constant on all popcounts is a global phase. B=2 lives in span{k, k²},
    so this gap is zero by construction. It is recorded so a nonzero value
    means the parameterization leaked.
    """
    arr = np.asarray(phi, dtype=np.float64).reshape(-1, 4)
    k = np.arange(4.0)
    design = np.stack([np.ones(4), k, k ** 2], axis=1)
    acc = []
    for row in arr:
        coef, *_ = np.linalg.lstsq(design, row, rcond=None)
        resid = row - design @ coef
        acc.append(float(np.sqrt(np.mean(resid ** 2))))
    return float(np.mean(acc)) if acc else 0.0


def ratio(num: float, den: float) -> float:
    if den == 0.0:
        return float("inf")
    return float(num) / float(den)


def _le(value: float, bar: float) -> bool:
    return float(value) <= float(bar) + 1e-12


def _gt(value: float, bar: float) -> bool:
    return float(value) > float(bar) + 1e-12


def _ge(value: float, bar: float) -> bool:
    return float(value) >= float(bar) - 1e-12


def verdict_delta(r_l2: float, lam_med: float) -> tuple[str, str]:
    """Gate one δ. Same bars as PR #49. L=4 is not an input.

    PASS when L=2 R_Fourier ≤ 0.85 and median cavity Λ ≤ 0.80.
    That is B=2 keeping the disorder lead the Hamming shell had.
    needs_edit when R is soft, in (0.85, 0.98], or Λ sits in (0.80, 0.90).
    A soft R is B=2 losing that lead.
    KILL when L=2 R > 0.98 or Λ ≥ 0.90. A KILL clause wins over a PASS clause.
    """
    r = float(r_l2)
    lam = float(lam_med)
    kills = []
    if _gt(r, R_KILL):
        kills.append(f"L=2 R_Fourier {r:.4f} > 0.98")
    if _ge(lam, LAMBDA_KILL):
        kills.append(f"median Λ {lam:.4f} ≥ 0.90")
    if kills:
        return "KILL", "; ".join(kills)
    r_pass = _le(r, R_PASS)
    lam_pass = _le(lam, LAMBDA_PASS)
    if r_pass and lam_pass:
        return "PASS", "L=2 R_Fourier ≤ 0.85 and median Λ ≤ 0.80; B=2 keeps the disorder lead"
    bits = []
    if not r_pass:
        bits.append(
            f"B=2 loses the disorder lead: L=2 R_Fourier {r:.4f} is soft, in (0.85, 0.98]"
        )
    if not lam_pass:
        bits.append(f"median Λ {lam:.4f} is inside (0.80, 0.90)")
    return "needs_edit", "; ".join(bits)


def family_verdict(calls: list[str]) -> str:
    if any(c == "KILL" for c in calls):
        return "KILL"
    if calls and all(c == "PASS" for c in calls):
        return "PASS"
    if any(c == "INCOMPLETE" for c in calls):
        return "INCOMPLETE"
    return "needs_edit"


def _b2_grad_check() -> None:
    """Autograd through φ = c @ basis matches a finite difference on ⟨H⟩."""
    n = 6
    E = np.zeros(1 << n, dtype=np.float64)
    for x in range(1 << n):
        bits = [(x >> (n - 1 - q)) & 1 for q in range(4)]
        if bits == [1, 1, 1, 1]:
            E[x] += 1.0
    Et = torch.tensor(E, dtype=torch.float64).reshape(1, -1)
    Cn = cn_from_unsat(Et, n)
    comps = walsh_degree_components(Cn).detach()
    c = torch.tensor([[0.12, -0.04]], dtype=torch.float64, requires_grad=True)
    gw = torch.tensor([[-0.02, 0.01, -0.03, 0.015]], dtype=torch.float64, requires_grad=True)
    coeff = np.array([-0.80, -0.70, -0.90, -0.60, -0.85, -0.65], dtype=np.float64)

    def energy(c_t, g_t) -> torch.Tensor:
        psi = snap_state_from_components(comps, g_t, phi_from_b2(c_t))
        return expectation_H(psi, Et, n, lambda_x=coeff, mu_xx=MU).sum()

    loss = energy(c, gw)
    loss.backward()
    if c.grad is None or float(c.grad.abs().max()) < 1e-8:
        raise AssertionError(f"B=2 coefficient grad vanished: {None if c.grad is None else c.grad}")
    eps = 1e-6
    with torch.no_grad():
        def at(delta: float) -> float:
            shift = torch.zeros_like(c)
            shift[0, 1] = delta
            return float(energy(c.detach() + shift, gw.detach()))

        fd = (at(eps) - at(-eps)) / (2.0 * eps)
    if abs(fd - float(c.grad[0, 1])) > 1e-4 * max(1.0, abs(fd)):
        raise AssertionError(f"B=2 grad {float(c.grad[0, 1])} != fd {fd}")
    # Paper init is exactly the linear shell, so Fourier and B=2 start together.
    ang = init_method("B2", 2)
    phi = phi_from_b2(ang["c"])
    if float(torch.max(torch.abs(phi[0] - torch.tensor(_PAPER_PHI)))) > 1e-12:
        raise AssertionError("B=2 init is not the paper shell")
    sf = prepare_state("fourier", Cn, init_angles("fourier", 2, dtype=torch.float64))
    sb = snap_state(Cn, ang["gamma_w"], phi)
    fid = float(torch.sum(sf.conj() * sb, dim=-1).abs()[0])
    if fid < 1.0 - 1e-8:
        raise AssertionError(f"B=2 init fidelity vs Fourier {fid}")
    if physical_shell_gap(phi.detach().numpy()) > 1e-12:
        raise AssertionError("B=2 init leaked out of span{1, k, k^2}")


def decision_self_test() -> None:
    """The written gate, the shared disorder pattern, and the B=2 span."""
    if verdict_delta(0.62, 0.75)[0] != "PASS":
        raise AssertionError("uniform-like cell")
    if verdict_delta(0.85, 0.80)[0] != "PASS":
        raise AssertionError("bars are inclusive on the pass side")
    soft = verdict_delta(0.90, 0.75)
    if soft[0] != "needs_edit" or "loses the disorder lead" not in soft[1]:
        raise AssertionError(soft)
    between = verdict_delta(0.70, 0.85)
    if between[0] != "needs_edit" or "0.85" not in between[1]:
        raise AssertionError(between)
    if verdict_delta(0.98, 0.70)[0] != "needs_edit":
        raise AssertionError("R=0.98 is not above the kill line")
    if verdict_delta(0.9800001, 0.50)[0] != "KILL":
        raise AssertionError("R just above 0.98")
    if verdict_delta(0.40, 0.90)[0] != "KILL":
        raise AssertionError("Λ=0.90 is a kill")
    if verdict_delta(0.99, 0.95)[0] != "KILL":
        raise AssertionError("both kill clauses")
    if family_verdict(["PASS", "PASS"]) != "PASS":
        raise AssertionError("both deltas")
    if family_verdict(["PASS", "needs_edit"]) != "needs_edit":
        raise AssertionError("one soft delta")
    if family_verdict(["PASS", "KILL"]) != "KILL":
        raise AssertionError("one kill delta")
    if family_verdict(["INCOMPLETE", "PASS"]) != "INCOMPLETE":
        raise AssertionError("incomplete")
    u = unit_pattern()
    expect = np.array(
        [0.524555, 0.919803, 0.880181, 0.083294, 0.198537, -0.635247],
        dtype=np.float64,
    )
    if np.max(np.abs(u[:6] - expect)) > 5e-7:
        raise AssertionError(f"disorder pattern drifted {u[:6]}")
    lam = positive_lambda(6, 0.30, u)
    if abs(float(lam[0]) - 0.75 * (1.0 + 0.30 * float(u[0]))) > 1e-12:
        raise AssertionError("λ formula")
    try:
        positive_lambda(21, 0.15, u)
    except RuntimeError:
        pass
    else:
        raise AssertionError("n=21 must be refused")
    try:
        load_unsat(21, slice(0, 1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("n=21 pack must be refused")
    recon = B2_INIT @ B2_BASIS
    if float(np.max(np.abs(recon - _PAPER_PHI))) > 1e-12:
        raise AssertionError(f"B=2 basis does not reconstruct the paper shell {recon}")
    phi_lin = np.array([[0.0, -0.2, -0.4, -0.6], [0.0, -0.1, -0.2, -0.3]])
    if shell_rms(phi_lin) > 1e-12:
        raise AssertionError("linear shell")
    quad = np.stack([_K ** 2, 0.5 * _K ** 2], axis=0)
    if shell_rms(quad) < 1e-6:
        raise AssertionError("quadratic shell should leave the linear mixer")
    if physical_shell_gap(quad) > 1e-10:
        raise AssertionError("k² is inside span{1, k, k²}")
    _b2_grad_check()


def _gs_one(task: tuple) -> tuple:
    row, n, coeff = task
    bundle = ground_bundle(np.asarray(row, dtype=np.float64), int(n), lambda_x=np.asarray(coeff), mu_xx=MU)
    lam_c = np.asarray(lambda_of_state(bundle["psi"], int(n)), dtype=np.float64).reshape(-1)
    if lam_c.size != int(n) // 3:
        raise RuntimeError(f"Λ length {lam_c.size} at n={n}")
    return (
        float(bundle["E0"]),
        float(bundle["peak"]),
        lam_c,
        float(bundle["residual"]),
    )


def _pool_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def gs_path(tag: str, n: int, sl: slice) -> Path:
    return RUN / f"gs_{tag}_n{n}_{sl.start}_{sl.stop}.npz"


def ensure_gs(
    tag: str,
    n: int,
    sl: slice,
    E: np.ndarray,
    seeds: np.ndarray,
    coeff: np.ndarray,
    jobs: int,
) -> dict:
    """E0, Z-peak, and per-cavity Λ of the true ground state."""
    RUN.mkdir(parents=True, exist_ok=True)
    path = gs_path(tag, n, sl)
    if path.exists():
        z = np.load(path)
        same = (
            np.array_equal(z["seeds"], seeds)
            and z["e0"].shape[0] == E.shape[0]
            and np.allclose(z["coeff"], coeff)
            and abs(float(z["mu"])) < 1e-15
            and str(z["protocol"]) == PROTOCOL
        )
        if same:
            log(f"GS cache hit {tag} n{n} [{sl.start}:{sl.stop}]")
            return {k: z[k] for k in z.files}
    log(f"GS {tag} n{n} [{sl.start}:{sl.stop}] B={E.shape[0]} jobs={jobs}")
    t0 = time.time()
    tasks = [(E[i], n, coeff) for i in range(E.shape[0])]
    if jobs <= 1 or E.shape[0] == 1:
        rows = [_gs_one(t) for t in tasks]
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ctx.Pool(int(jobs), initializer=_pool_init) as pool:
            rows = list(pool.imap(_gs_one, tasks, chunksize=1))
    e0 = np.array([r[0] for r in rows], dtype=np.float64)
    peak = np.array([r[1] for r in rows], dtype=np.float64)
    lam_c = np.stack([r[2] for r in rows], axis=0)
    resid = np.array([r[3] for r in rows], dtype=np.float64)
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above diag min at {tag} n{n}")
    if np.any(lam_c < -1e-8) or np.any(lam_c > 1.0 + 1e-6):
        raise RuntimeError(f"Λ out of range at {tag} n{n}")
    np.savez_compressed(
        path,
        seeds=seeds,
        e0=e0,
        peak=peak,
        lambda_c=lam_c,
        residual=resid,
        n=np.int64(n),
        coeff=np.asarray(coeff, dtype=np.float64),
        mu=np.float64(MU),
        delta_tag=tag,
        protocol=PROTOCOL,
    )
    log(
        f"GS {tag} n{n} {time.time()-t0:.1f}s  medE0={float(np.median(e0)):.4f} "
        f"Λ={float(np.median(lam_c)):.4f} peak={float(np.median(peak)):.4f}"
    )
    return {
        "seeds": seeds,
        "e0": e0,
        "peak": peak,
        "lambda_c": lam_c,
        "residual": resid,
        "coeff": np.asarray(coeff, dtype=np.float64),
        "mu": np.float64(MU),
    }


def lambda_summary(lam_c: np.ndarray, peak: np.ndarray, e0: np.ndarray) -> dict:
    flat = np.asarray(lam_c, dtype=np.float64).reshape(-1)
    pk = np.asarray(peak, dtype=np.float64).reshape(-1)
    arr = np.asarray(lam_c, dtype=np.float64)
    per_c = [float(np.median(arr[:, c])) for c in range(arr.shape[1])]
    return {
        "B": int(pk.size),
        "n_cav": int(arr.shape[1]),
        "lambda_med": float(np.median(flat)),
        "lambda_min": float(np.min(flat)),
        "lambda_max": float(np.max(flat)),
        "frac_gt_080": float(np.mean(flat > 0.80)),
        "frac_ge_090": float(np.mean(flat >= 0.90)),
        "per_cavity_med": per_c,
        "peak_med": float(np.median(pk)),
        "peak_max": float(np.max(pk)),
        "frac_classical": float(np.mean(pk >= 0.5)),
        "e0_med": float(np.median(e0)),
    }


def init_method(method: str, L: int) -> dict:
    L = int(L)
    if method == "fourier":
        return init_angles("fourier", L, dtype=torch.float64)
    if method == "B2":
        c0 = torch.tensor(B2_INIT, dtype=torch.float64)
        return {
            "gamma_w": torch.full((L, 4), INIT_GAMMA, dtype=torch.float64),
            "c": c0.reshape(1, 2).expand(L, 2).contiguous().clone(),
        }
    raise ValueError(method)


def bind_parameters(method: str, angles: dict) -> list[torch.nn.Parameter]:
    if method == "fourier":
        return method_parameter_list("fourier", angles)
    if method != "B2":
        raise ValueError(method)
    params = []
    for k in ("gamma_w", "c"):
        p = torch.nn.Parameter(angles[k].detach().clone())
        angles[k] = p
        params.append(p)
    return params


def _as_f64(angles: dict) -> dict:
    return {k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v) for k, v in angles.items()}


def _prepare(method: str, Cn: torch.Tensor, angles: dict, comps=None) -> torch.Tensor:
    if method == "fourier":
        return prepare_state("fourier", Cn, angles)
    if method == "B2":
        phi = phi_from_b2(angles["c"])
        if comps is None:
            return snap_state(Cn, angles["gamma_w"], phi)
        return snap_state_from_components(comps, angles["gamma_w"], phi)
    raise ValueError(method)


def eval_method(method: str, angles: dict, bundle: dict, coeff_of: dict) -> dict:
    """bundle[n] = (unsat, e0, seeds). Returns medians and the ε list."""
    ang = _as_f64(angles)
    chunk = {6: 32, 9: 16, 12: 8}
    out = {}
    with torch.no_grad():
        for n, (E, e0, seeds) in bundle.items():
            Eu = torch.tensor(np.asarray(E, dtype=np.float64))
            e0t = torch.tensor(np.asarray(e0, dtype=np.float64))
            coeff = np.asarray(coeff_of[int(n)], dtype=np.float64)
            parts = []
            step = chunk[int(n)]
            for s in range(0, Eu.shape[0], step):
                sl = slice(s, s + step)
                Cn = cn_from_unsat(Eu[sl], int(n))
                psi = _prepare(method, Cn, ang)
                h = expectation_H(psi, Eu[sl], int(n), lambda_x=coeff, mu_xx=MU)
                parts.append((h - e0t[sl]).detach().cpu().numpy())
            eps = np.concatenate(parts, axis=0)
            if np.any(eps < -1e-4):
                raise RuntimeError(f"{method} ε below E0 by {float(eps.min()):.3e} at n={n}")
            rec = summarize_eps(eps)
            rec.update(
                {
                    "n": int(n),
                    "eps": [float(x) for x in eps.tolist()],
                    "seeds": [int(x) for x in np.asarray(seeds).tolist()],
                    "med_e0": float(np.median(e0)),
                }
            )
            out[str(n)] = rec
    return out


def _loss_on(method, angles, Eu, e0b, n: int, coeff: np.ndarray, comps) -> torch.Tensor:
    if method == "B2":
        psi = _prepare(method, None, angles, comps)
    else:
        Cn = cn_from_unsat(Eu, n)
        psi = _prepare(method, Cn, angles)
    h = expectation_H(psi, Eu, n, lambda_x=coeff, mu_xx=MU)
    return ((h - e0b) / float(n)).mean()


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _angle_payload(method: str, angles: dict) -> dict:
    payload = angles_to_lists(method, angles)
    if method == "B2":
        phi = phi_from_b2(angles["c"]).detach().cpu().to(torch.float64).numpy()
        payload["phi"] = phi.tolist()
        payload["shell_rms"] = shell_rms(phi)
        payload["phys_rms"] = physical_shell_gap(phi)
    return payload


def train_one(method: str, L: int, seed: int, delta: float, packs: dict, rounds: int) -> dict:
    """Batch stream depends on the seed, not on δ, method, or L.

    That is the PR #49 stream: NumPy Generator(seed) chooses the two
    halves of the batch, and torch.manual_seed(seed) is set before init.
    Fourier and B=2 share it.
    """
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    angles = init_method(method, int(L))
    params = bind_parameters(method, angles)
    E6, e06, c6 = packs["E6"], packs["e06"], packs["c6"]
    E9, e09, c9 = packs["E9"], packs["e09"], packs["c9"]
    coeff6, coeff9 = packs["coeff6"], packs["coeff9"]
    n6, n9 = int(E6.shape[0]), int(E9.shape[0])
    b6 = min(S.LBFGS_BATCH // 2, n6)
    b9 = min(S.LBFGS_BATCH - b6, n9)
    hold = {9: packs["hold9"]}

    def hold_med() -> float:
        return float(eval_method(method, angles, hold, packs["coeff"])["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(
        f"δ={delta:.2f} L{L} {method} s{seed} init n9-hold Medε={init_med:.5f} "
        f"nparams={sum(p.numel() for p in params)}"
    )
    for r in range(int(rounds)):
        opt = torch.optim.LBFGS(
            params,
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        i6 = torch.tensor(rng.choice(n6, size=b6, replace=False), dtype=torch.long)
        i9 = torch.tensor(rng.choice(n9, size=b9, replace=False), dtype=torch.long)

        def closure(i6=i6, i9=i9):
            opt.zero_grad()
            loss = 0.5 * (
                _loss_on(
                    method,
                    angles,
                    E6.index_select(0, i6),
                    e06.index_select(0, i6),
                    6,
                    coeff6,
                    c6.index_select(1, i6),
                )
                + _loss_on(
                    method,
                    angles,
                    E9.index_select(0, i9),
                    e09.index_select(0, i9),
                    9,
                    coeff9,
                    c9.index_select(1, i9),
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as exc:
            log(f"δ={delta:.2f} L{L} {method} s{seed} r{r} LBFGS skip: {exc}")
            hist.append({"round": r, "skip": str(exc)})
            continue
        if not _finite(loss_f):
            log(f"δ={delta:.2f} L{L} {method} s{seed} r{r} nonfinite — restore best")
            with torch.no_grad():
                for k, v in best["angles"].items():
                    angles[k].data.copy_(v.to(dtype=angles[k].dtype))
            hist.append({"round": r, "loss": None})
            continue
        med = hold_med()
        hist.append({"round": r, "loss": loss_f, "hold_med9": med})
        mark = ""
        if med < best["med9"]:
            best = {
                "med9": med,
                "round": r,
                "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
            }
            mark = " *"
        if mark or r % 10 == 0 or r + 1 == int(rounds):
            log(
                f"δ={delta:.2f} L{L} {method} s{seed} r{r}/{rounds} "
                f"loss={loss_f:.5f} hold={med:.5f} best={best['med9']:.5f}{mark}"
            )
    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    payload = _angle_payload(method, best["angles"])
    ev = eval_method(method, best["angles"], packs["eval"], packs["coeff"])
    return {
        "done": True,
        "protocol": PROTOCOL,
        "delta": float(delta),
        "tag": tag_delta(delta),
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "mu": MU,
        "lambda0": LAMBDA0,
        "snap": "B2" if method == "B2" else None,
        "n_params": int(sum(p.numel() for p in params)),
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "init_hold_med9": float(init_med),
        "wall_s": time.time() - t0,
        "rounds": int(rounds),
        "max_iter": int(S.LBFGS_ITER),
        "lr": float(S.LBFGS_LR),
        "batch": int(S.LBFGS_BATCH),
        "shell_rms": payload.get("shell_rms"),
        "phys_rms": payload.get("phys_rms"),
        "angles": payload,
        "eval": ev,
        "hist": [{k: h[k] for k in h if k in ("round", "loss", "hold_med9", "skip")} for h in hist],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def seed_path(tag: str, L: int, method: str, seed: int) -> Path:
    return RUN / f"{tag}_L{L}_{method}_seed{seed}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _build_packs(delta: float, unit: np.ndarray) -> dict:
    """Train tensors plus the holdout / freeze bundles. GS caches must exist."""
    tag = tag_delta(delta)
    coeff = {n: x_coeff(n, delta, unit) for n in (6, 9, 12)}

    def take(n: int, sl: slice):
        E, seeds = load_unsat(n, sl)
        z = np.load(gs_path(tag, n, sl))
        if not np.array_equal(z["seeds"], seeds):
            raise RuntimeError(f"seed mismatch {tag} n{n} {sl}")
        return E, np.asarray(z["e0"], dtype=np.float64), seeds

    E6, e06, s6 = take(6, slice(0, HOLD_HI))
    E9, e09, s9 = take(9, slice(0, HOLD_HI))
    E12, e012, s12 = take(12, slice(0, FREEZE_B))
    E6t = torch.tensor(E6[TRAIN_LO:TRAIN_HI], dtype=torch.float64)
    E9t = torch.tensor(E9[TRAIN_LO:TRAIN_HI], dtype=torch.float64)
    return {
        "coeff": coeff,
        "coeff6": coeff[6],
        "coeff9": coeff[9],
        "E6": E6t,
        "e06": torch.tensor(e06[TRAIN_LO:TRAIN_HI], dtype=torch.float64),
        "c6": walsh_degree_components(cn_from_unsat(E6t, 6)).detach(),
        "E9": E9t,
        "e09": torch.tensor(e09[TRAIN_LO:TRAIN_HI], dtype=torch.float64),
        "c9": walsh_degree_components(cn_from_unsat(E9t, 9)).detach(),
        "hold9": (E9[HOLD_LO:HOLD_HI], e09[HOLD_LO:HOLD_HI], s9[HOLD_LO:HOLD_HI]),
        "eval": {
            6: (E6[HOLD_LO:HOLD_HI], e06[HOLD_LO:HOLD_HI], s6[HOLD_LO:HOLD_HI]),
            9: (E9[HOLD_LO:HOLD_HI], e09[HOLD_LO:HOLD_HI], s9[HOLD_LO:HOLD_HI]),
            12: (E12, e012, s12),
        },
    }


def _train_worker(spec: dict) -> str:
    try:
        _pool_init()
        delta = float(spec["delta"])
        method = spec["method"]
        L = int(spec["L"])
        seed = int(spec["seed"])
        rounds = int(spec["rounds"])
        tag = tag_delta(delta)
        path = seed_path(tag, L, method, seed)
        if path.exists() and not spec.get("force"):
            row = json.loads(path.read_text())
            if (
                row.get("done")
                and row.get("protocol") == PROTOCOL
                and int(row.get("rounds", -1)) == rounds
                and abs(float(row.get("mu", 1.0))) < 1e-15
                and row.get("method") == method
                and abs(float(row.get("delta", -1.0)) - delta) < 1e-12
            ):
                log(f"train cache hit {path.name}")
                return str(path)
        unit = unit_pattern()
        packs = _build_packs(delta, unit)
        row = train_one(method, L, seed, delta, packs, rounds)
        write_json(path, row)
        log(
            f"saved {path.name} best_r={row['best_round']} "
            f"n12 Medε={row['eval']['12']['med_eps']:.4f} wall={row['wall_s']:.1f}s"
        )
        return str(path)
    except Exception:
        log(f"TRAIN FAIL {spec}\n{traceback.format_exc()}")
        raise


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v or abs(v) == float("inf"):
        return "—"
    return f"{v:.{nd}f}"


def _load_rows(deltas, depths, seeds) -> tuple[dict, list]:
    rows = {}
    missing = []
    for delta in deltas:
        tag = tag_delta(delta)
        for L in depths:
            for method in METHODS:
                for seed in seeds:
                    path = seed_path(tag, L, method, seed)
                    if not path.exists():
                        missing.append(path.name)
                        continue
                    rows[(tag, int(L), method, int(seed))] = json.loads(path.read_text())
    return rows, missing


def _paired(arm_rows: list[dict], base_rows: list[dict], n: str) -> tuple[int, int]:
    base_by = {int(r["seed"]): r for r in base_rows}
    win = total = 0
    for arow in arm_rows:
        brow = base_by[int(arow["seed"])]
        es = np.asarray(arow["eval"][n]["eps"], dtype=np.float64)
        eb = np.asarray(brow["eval"][n]["eps"], dtype=np.float64)
        if es.shape != eb.shape:
            raise RuntimeError(f"paired shape {es.shape} vs {eb.shape}")
        win += int(np.sum(es < eb - 1e-12))
        total += int(es.size)
    return win, total


def _gs_block(tag: str, n: int, sl: slice) -> dict | None:
    path = gs_path(tag, n, sl)
    if not path.exists():
        return None
    z = np.load(path)
    return lambda_summary(z["lambda_c"], z["peak"], z["e0"])


def aggregate(deltas, depths, seeds, unit: np.ndarray) -> dict:
    rows, missing = _load_rows(deltas, depths, seeds)
    summary = {
        "protocol": PROTOCOL,
        "deltas": [float(d) for d in deltas],
        "depths": [int(L) for L in depths],
        "seeds": [int(s) for s in seeds],
        "disorder_seed": DISORDER_SEED,
        "mu": MU,
        "lambda0": LAMBDA0,
        "snap": "B2",
        "b2_basis": B2_BASIS.tolist(),
        "b2_init": B2_INIT.tolist(),
        "b2_recon_err": float(np.max(np.abs(B2_INIT @ B2_BASIS - _PAPER_PHI))),
        "shell49": {
            tag_delta(d): {
                "lambda_med_n12": SHELL49[d]["lambda_med_n12"],
                "L2_R": SHELL49[d][2]["R"],
                "L2_med_snap": SHELL49[d][2]["snap"],
                "L2_med_fourier": SHELL49[d][2]["fourier"],
                "L4_R": SHELL49[d][4]["R"],
            }
            for d in SHELL49
        },
        "b2_clean": {
            "lambda_med_n12": B2_CLEAN["lambda_med_n12"],
            "L2_R": B2_CLEAN[2]["R"],
            "L4_R": B2_CLEAN[4]["R"],
        },
        "by_delta": {},
        "missing": missing,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    calls = []
    for delta in deltas:
        tag = tag_delta(delta)
        lam12 = positive_lambda(FREEZE_N, delta, unit)
        gs12 = _gs_block(tag, 12, slice(0, FREEZE_B))
        gs6 = _gs_block(tag, 6, slice(0, HOLD_HI))
        gs9 = _gs_block(tag, 9, slice(0, HOLD_HI))
        block = {
            "delta": float(delta),
            "lambda_n12": [float(x) for x in lam12],
            "gs": {"6": gs6, "9": gs9, "12": gs12},
            "depths": {},
        }
        r_l2 = None
        for L in depths:
            arm, four = [], []
            for seed in seeds:
                key_s = (tag, int(L), "B2", int(seed))
                key_f = (tag, int(L), "fourier", int(seed))
                if key_s in rows and key_f in rows:
                    arm.append(rows[key_s])
                    four.append(rows[key_f])
            if len(arm) != len(seeds):
                block["depths"][str(L)] = {"complete": False}
                continue
            by_n = {}
            for n in ("6", "9", "12"):
                arm_meds = [float(r["eval"][n]["med_eps"]) for r in arm]
                four_meds = [float(r["eval"][n]["med_eps"]) for r in four]
                win, total = _paired(arm, four, n)
                by_n[n] = {
                    "med_B2": float(np.median(np.asarray(arm_meds))),
                    "med_fourier": float(np.median(np.asarray(four_meds))),
                    "R": ratio(float(np.median(arm_meds)), float(np.median(four_meds))),
                    "paired_win": win,
                    "paired_n": total,
                    "B2_meds": arm_meds,
                    "fourier_meds": four_meds,
                }
            per_seed = []
            n_clear = 0
            order_a = sorted(arm, key=lambda r: int(r["seed"]))
            order_f = sorted(four, key=lambda r: int(r["seed"]))
            for rs, rf in zip(order_a, order_f):
                rr = ratio(float(rs["eval"]["12"]["med_eps"]), float(rf["eval"]["12"]["med_eps"]))
                if _le(rr, R_PASS):
                    n_clear += 1
                per_seed.append(
                    {
                        "seed": int(rs["seed"]),
                        "B2": float(rs["eval"]["12"]["med_eps"]),
                        "fourier": float(rf["eval"]["12"]["med_eps"]),
                        "R": rr,
                        "best_round_B2": int(rs["best_round"]),
                        "best_round_fourier": int(rf["best_round"]),
                        "wall_B2": float(rs["wall_s"]),
                        "wall_fourier": float(rf["wall_s"]),
                        "shell_rms": rs.get("shell_rms"),
                        "phys_rms": rs.get("phys_rms"),
                    }
                )
            shell_vals = [float(p["shell_rms"]) for p in per_seed if p["shell_rms"] is not None]
            phys_vals = [float(p["phys_rms"]) for p in per_seed if p["phys_rms"] is not None]
            block["depths"][str(L)] = {
                "complete": True,
                "by_n": by_n,
                "per_seed": per_seed,
                "n_clear": n_clear,
                "shell_rms_med": float(np.median(shell_vals)) if shell_vals else None,
                "phys_rms_max": float(np.max(phys_vals)) if phys_vals else None,
            }
            if int(L) == 2:
                r_l2 = by_n["12"]["R"]
        if gs12 is None or r_l2 is None:
            call, detail = "INCOMPLETE", "missing ground state or L=2 rows"
        else:
            call, detail = verdict_delta(r_l2, gs12["lambda_med"])
        block["call"] = call
        block["detail"] = detail
        block["R_L2"] = r_l2
        calls.append(call)
        summary["by_delta"][tag] = block
    summary["calls"] = calls
    summary["verdict"] = family_verdict(calls) if calls else "INCOMPLETE"
    text = render(summary, unit)
    (RUN / "RESULTS.md").write_text(text)
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUN / "lambda_u.json").write_text(
        json.dumps(
            {
                "seed": DISORDER_SEED,
                "u": unit.tolist(),
                "note": "delta_i = delta * u_i; same generator as disordered_lambda_NEW_SAFE (PR #49)",
            },
            indent=2,
        )
    )
    log(f"wrote RESULTS.md verdict={summary['verdict']}")
    return summary


def render(summary: dict, unit: np.ndarray) -> str:
    verdict = summary["verdict"]
    lines = [
        "# Disordered λ at NEW_SAFE, B=2 Walsh SNAP (μ = 0)",
        "",
        f"**Overall: {verdict}.**",
        "",
        "Same disorder as PR #49 (one pattern, seed 20261105, δ ∈ {0.15, 0.30}, μ = 0). "
        "The SNAP arm is the B=2 parameterization from PR #57: two coefficients on "
        "popcount, φ_k = c1 k + c2 k², not the four-phase Hamming shell. "
        "Fourier QAOA is matched. Λ is the ground state and does not depend on B.",
        "",
        "## Hamiltonian",
        "",
        r"\[",
        r"H = \mathrm{diag}(\mathrm{unsat}) - \sum_i \lambda_i X_i",
        r"\]",
        "",
        r"\[",
        r"\lambda_i = 0.75\,(1 + \delta u_i),\qquad u_i\sim\mathrm{Unif}[-1,1]",
        r"\]",
        "",
        f"μ = 0. One pattern u (NumPy seed {DISORDER_SEED}), shared by both strengths "
        "and by every n (qubit i uses u_i, so the n=6 pattern is the prefix of n=12). "
        "All λ_i stay positive. Readout is the Z histogram (unsat) plus the X histogram. "
        r"The field is \(\sum_i \lambda_i\langle X_i\rangle\). No Y, no XX.",
        "",
        "Λ_c = ⟨(Σ_{i∈c} X_i / 3)²⟩ from the X histogram of the true ground state. "
        "The reported median pools cavities and instances. Clean NEW_SAFE (δ = 0) has "
        f"n=12 median Λ = {B2_CLEAN['lambda_med_n12']:.4f}.",
        "",
        "## Ansatz",
        "",
        "| method | angles / layer | unitary |",
        "|---|---|---|",
        r"| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), same β on every qubit |",
        r"| B=2 Walsh SNAP | 6: γ_{w=1..4}, c1, c2 | Walsh degree-1..4 cost, then φ_k = c1 k + c2 k² on every cavity |",
        "",
        "B=2 coordinates are an orthonormal basis of span{k, k²} on k = 0..3, the same "
        "metric the locked shell uses on φ. A constant added to every popcount is a "
        "global phase and is not a coefficient. The paper init φ_k = −β k sits in this "
        f"span exactly (reconstruction error {summary.get('b2_recon_err', float('nan')):.3e}). "
        f"Paper init β={INIT_BETA}, γ=γ_w={INIT_GAMMA}, so Fourier and B=2 start on the same state. "
        "No per-qubit angles were added. Disorder is only in H. "
        r"Cost family is still \(C_n=\mathrm{unsat}\cdot n/m\).",
        "",
        "The PR #49 arm was the Hamming shell, four free phases φ_0..φ_3 (8 angles per "
        "layer with the four γ weights). This run replaces only that mixer budget. "
        "PR #57 found B=2 matches that shell at clean NEW_SAFE, L=2 "
        f"(R = {B2_CLEAN[2]['R']:.4f}, Medε {B2_CLEAN[2]['B2']:.4f} / {B2_CLEAN[2]['fourier']:.4f}).",
        "",
        "## Protocol",
        "",
        f"- Packs: `beat_qaoa_12h/train_packs`. Train indices [{TRAIN_LO}:{TRAIN_HI}) on n=6 and n=9. "
        f"Checkpoint on the n=9 holdout [{HOLD_LO}:{HOLD_HI}). "
        f"Reported n=6 and n=9 numbers are that holdout. Freeze n=12, first {FREEZE_B} instances.",
        f"- Seeds {{{', '.join(str(s) for s in summary['seeds'])}}}. "
        "The seed only reshuffles batches. Every δ, method, and depth shares that stream "
        "(NumPy Generator(seed), the PR #49 stream).",
        f"- L-BFGS matches `Nov_3rd.settings`: {S.LBFGS_ROUNDS} rounds × max_iter {S.LBFGS_ITER}, "
        f"lr {S.LBFGS_LR}, history 16, strong Wolfe, grad clip 5, batch {S.LBFGS_BATCH} "
        "(half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. No Adam.",
        "- Each seed restores the round with the best n=9-holdout Med ε.",
        "- R_Fourier = median over seeds of Medε_B2, divided by the same median for Fourier.",
        "- n=21 was not loaded. `settings.py`, `qaoa_angles/`, and the phase-2 ladder were not written.",
        "",
        "## Gate",
        "",
        "- **PASS** a δ when frozen-n12 L=2 R_Fourier ≤ 0.85 and the n=12 median cavity Λ ≤ 0.80. "
        "That is B=2 keeping the disorder lead.",
        "- **needs_edit** when B=2 loses the disorder lead: L=2 R is soft, in (0.85, 0.98], "
        "or median Λ sits in (0.80, 0.90).",
        "- **KILL** when L=2 R_Fourier > 0.98 or median Λ ≥ 0.90. A KILL clause wins.",
        "- Both δ=0.15 and δ=0.30 have to pass. One KILL makes the run KILL. "
        "L=4 is reported and is not the gate. "
        f"The shell's disordered L=2 ratios were {SHELL49[0.15][2]['R']:.3f} and {SHELL49[0.30][2]['R']:.3f}. "
        f"Clean B=2 L=2 was {B2_CLEAN[2]['R']:.3f}.",
        "",
        "## References (not remeasured)",
        "",
        "δ = 0 is clean NEW_SAFE, B=2, from PR #57. The shell columns are PR #49 on this same disorder.",
        "",
        "| source | δ | L | Medε Fourier | Medε SNAP | R_Fourier | n12 median Λ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| PR #57 B=2 | 0 | 2 | {_fmt(B2_CLEAN[2]['fourier'])} | {_fmt(B2_CLEAN[2]['B2'])} | {_fmt(B2_CLEAN[2]['R'])} | {_fmt(B2_CLEAN['lambda_med_n12'])} |",
        f"| PR #57 B=2 | 0 | 4 | {_fmt(B2_CLEAN[4]['fourier'])} | {_fmt(B2_CLEAN[4]['B2'])} | {_fmt(B2_CLEAN[4]['R'])} | {_fmt(B2_CLEAN['lambda_med_n12'])} |",
        f"| PR #49 shell | 0.15 | 2 | {_fmt(SHELL49[0.15][2]['fourier'])} | {_fmt(SHELL49[0.15][2]['snap'])} | {_fmt(SHELL49[0.15][2]['R'])} | {_fmt(SHELL49[0.15]['lambda_med_n12'])} |",
        f"| PR #49 shell | 0.15 | 4 | {_fmt(SHELL49[0.15][4]['fourier'])} | {_fmt(SHELL49[0.15][4]['snap'])} | {_fmt(SHELL49[0.15][4]['R'])} | {_fmt(SHELL49[0.15]['lambda_med_n12'])} |",
        f"| PR #49 shell | 0.30 | 2 | {_fmt(SHELL49[0.30][2]['fourier'])} | {_fmt(SHELL49[0.30][2]['snap'])} | {_fmt(SHELL49[0.30][2]['R'])} | {_fmt(SHELL49[0.30]['lambda_med_n12'])} |",
        f"| PR #49 shell | 0.30 | 4 | {_fmt(SHELL49[0.30][4]['fourier'])} | {_fmt(SHELL49[0.30][4]['snap'])} | {_fmt(SHELL49[0.30][4]['R'])} | {_fmt(SHELL49[0.30]['lambda_med_n12'])} |",
        "",
        "## λ patterns",
        "",
    ]
    for delta in summary["deltas"]:
        tag = tag_delta(delta)
        block = summary["by_delta"].get(tag, {})
        lam = np.asarray(block.get("lambda_n12", positive_lambda(12, delta, unit)), dtype=np.float64)
        lines.append(f"### δ={delta:.2f} ({tag})")
        lines.append("")
        lines.append("n=12 λ = " + ", ".join(f"{v:.4f}" for v in lam) + ".")
        lines.append("")
        lines.append("| cavity | λ | range |")
        lines.append("|---|---|---:|")
        for c in range(FREEZE_N // 3):
            sl = lam[3 * c : 3 * c + 3]
            lines.append(
                f"| {c} | {', '.join(f'{v:.4f}' for v in sl)} | {_fmt(float(sl.max() - sl.min()))} |"
            )
        lines.append("")
    lines += [
        "## Ground state",
        "",
        "Median Λ pools every cavity of every instance. "
        "n=6 and n=9 use the full pack (B=250). n=12 is the freeze pool (B=128). "
        "Λ does not depend on the SNAP budget, so these rows should sit on the PR #49 values.",
        "",
        "| δ | n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | frac Λ≥0.90 | per-cavity median | median Z-peak | frac peak≥0.5 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for delta in summary["deltas"]:
        tag = tag_delta(delta)
        gs = summary["by_delta"][tag]["gs"]
        for n in ("6", "9", "12"):
            rec = gs.get(n)
            if not rec:
                lines.append(f"| {delta:.2f} | {n} | — | — | — | — | — | — | — | — | — |")
                continue
            cav = ", ".join(_fmt(v) for v in rec["per_cavity_med"])
            lines.append(
                f"| {delta:.2f} | {n} | {rec['B']} | {_fmt(rec['lambda_med'])} | "
                f"{_fmt(rec['lambda_min'])} | {_fmt(rec['lambda_max'])} | "
                f"{_fmt(rec['frac_gt_080'], 3)} | {_fmt(rec['frac_ge_090'], 3)} | {cav} | "
                f"{_fmt(rec['peak_med'])} | {_fmt(rec['frac_classical'], 3)} |"
            )
    lines += [
        "",
        f"Clean n=12 median Λ was {_fmt(B2_CLEAN['lambda_med_n12'])}. "
        f"PR #49 reported disordered n=12 median Λ {_fmt(SHELL49[0.15]['lambda_med_n12'])} "
        f"and {_fmt(SHELL49[0.30]['lambda_med_n12'])}.",
        "",
        "## Ratios",
        "",
        "Holdout at n=6 and n=9 is the checkpoint distribution (B=50 per seed). "
        "The decision row is n=12, L=2. Medε is the median over seeds of each seed's median excess.",
        "",
        "| δ | L | n | Medε Fourier | Medε B=2 | R_Fourier | paired B2<Fourier | seeds ≤0.85 | #49 shell R |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for delta in summary["deltas"]:
        tag = tag_delta(delta)
        depths = summary["by_delta"][tag]["depths"]
        for L in summary["depths"]:
            info = depths.get(str(L), {})
            if not info.get("complete"):
                lines.append(f"| {delta:.2f} | {L} | — | — | — | — | incomplete | — | — |")
                continue
            nseed = len(summary["seeds"])
            shell_r = SHELL49.get(float(delta), {}).get(int(L), {}).get("R")
            for n in ("6", "9", "12"):
                rec = info["by_n"][n]
                clear = f"{info['n_clear']}/{nseed}" if n == "12" else ""
                shell_cell = _fmt(shell_r) if n == "12" else ""
                lines.append(
                    f"| {delta:.2f} | {L} | {n} | {_fmt(rec['med_fourier'])} | {_fmt(rec['med_B2'])} | "
                    f"{_fmt(rec['R'])} | {rec['paired_win']}/{rec['paired_n']} | {clear} | {shell_cell} |"
                )
    lines += ["", "### Seeds at n=12", ""]
    lines.append(
        "| δ | L | seed | Medε B=2 | Medε Fourier | R | shell RMS | phys RMS | best round B=2 | wall s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for delta in summary["deltas"]:
        tag = tag_delta(delta)
        for L in summary["depths"]:
            info = summary["by_delta"][tag]["depths"].get(str(L), {})
            if not info.get("complete"):
                continue
            for p in info["per_seed"]:
                lines.append(
                    f"| {delta:.2f} | {L} | {p['seed']} | {_fmt(p['B2'])} | {_fmt(p['fourier'])} | "
                    f"{_fmt(p['R'])} | {_fmt(p['shell_rms'])} | {_fmt(p.get('phys_rms'), 2)} | "
                    f"{p['best_round_B2']} | {p['wall_B2']:.1f} |"
                )
    lines += ["", "## Calls", ""]
    lines.append("| δ | call | L=2 R | n12 median Λ | #49 shell L=2 R | detail |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for delta in summary["deltas"]:
        tag = tag_delta(delta)
        block = summary["by_delta"][tag]
        gs12 = block["gs"].get("12") or {}
        shell_r = SHELL49.get(float(delta), {}).get(2, {}).get("R")
        lines.append(
            f"| {delta:.2f} | {block['call']} | {_fmt(block.get('R_L2'))} | "
            f"{_fmt(gs12.get('lambda_med'))} | {_fmt(shell_r)} | {block['detail']} |"
        )
    lines += ["", "## Reading", ""]
    lines.extend(_reading(summary))
    lines += [
        "",
        "Per-seed JSON is `{d0p15,d0p30}_L{L}_{method}_seed{s}.json`. "
        "Ground-state arrays are `gs_{tag}_n{n}_{start}_{stop}.npz`. "
        "`lambda_u.json` is the shared pattern. `summary.json` is the table source.",
        "",
    ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _reading(summary: dict) -> list[str]:
    lines = []
    for delta in summary["deltas"]:
        tag = tag_delta(delta)
        block = summary["by_delta"][tag]
        gs = block["gs"].get("12") or {}
        d2 = block["depths"].get("2", {})
        d4 = block["depths"].get("4", {})
        if not gs or not d2.get("complete"):
            lines.append(f"δ={delta:.2f} is incomplete.")
            continue
        r2 = d2["by_n"]["12"]["R"]
        med_b = d2["by_n"]["12"]["med_B2"]
        med_f = d2["by_n"]["12"]["med_fourier"]
        r4 = d4["by_n"]["12"]["R"] if d4.get("complete") else None
        shell = SHELL49[float(delta)]
        cav = gs.get("per_cavity_med") or []
        cav_s = ", ".join(f"{float(v):.4f}" for v in cav)
        r4_note = _fmt(r4)
        if r4 is not None and r4 > R_PASS:
            r4_note += " (above 0.85; L=4 is not the gate)"
        lead = "keeps the disorder lead" if block["call"] == "PASS" else "does not keep both pass bars"
        lines.append(
            f"δ={delta:.2f}: n12 median Λ={gs['lambda_med']:.4f} "
            f"(PR #49 {shell['lambda_med_n12']:.4f}, clean {B2_CLEAN['lambda_med_n12']:.4f}, "
            f"shift from clean {gs['lambda_med'] - B2_CLEAN['lambda_med_n12']:+.4f}). "
            f"Per-cavity medians {cav_s}. "
            f"L=2 Medε B=2={med_b:.4f}, Fourier={med_f:.4f}, R_Fourier={r2:.4f} "
            f"(shell R {shell[2]['R']:.4f}, shell Medε {shell[2]['snap']:.4f}; "
            f"clean B=2 R {B2_CLEAN[2]['R']:.4f}). "
            f"{d2['n_clear']}/{len(summary['seeds'])} seeds ≤ 0.85. "
            f"L=4 R_Fourier={r4_note} (shell L=4 R {shell[4]['R']:.4f}). "
            f"{gs['frac_gt_080']:.1%} of n=12 cavity-instance pairs sit above 0.80, "
            f"{gs['frac_ge_090']:.1%} at or above 0.90. "
            f"Median Z-peak={gs['peak_med']:.4f}. "
            f"B=2 {lead}. Call **{block['call']}**."
        )
        lines.append("")
    if summary["verdict"] == "PASS":
        lines.append(
            "B=2 keeps the disorder lead at both strengths. "
            "The L=2 Fourier ratio stays at or under 0.85, on the same side of the bar as the "
            f"Hamming shell ({SHELL49[0.15][2]['R']:.3f} / {SHELL49[0.30][2]['R']:.3f}) and the "
            f"clean B=2 cell ({B2_CLEAN[2]['R']:.3f}). "
            "The n=12 median cavity Λ stays at or under 0.80. "
            "Two SNAP coefficients are enough for the disordered transverse field that four phases handled."
        )
    elif summary["verdict"] == "KILL":
        lines.append(
            "A kill clause fired: either the L=2 Fourier ratio went above 0.98, "
            "or the ground-state cavity Λ reached 0.90. B=2 does not carry NEW_SAFE through this disorder."
        )
    elif summary["verdict"] == "needs_edit":
        lines.append(
            "B=2 loses the disorder lead, or the cavity lock moved into the soft band. "
            "The Hamming shell held L=2 R near 0.61 with median Λ near 0.76. "
            "Two coefficients do not keep both of those bars. "
            "It is not a kill."
        )
    else:
        lines.append("The run is incomplete.")
    lines.append("")
    l4_bits = []
    for delta in summary["deltas"]:
        info = summary["by_delta"][tag_delta(delta)]["depths"].get("4", {})
        if not info.get("complete"):
            continue
        r12 = info["by_n"]["12"]["R"]
        r6 = info["by_n"]["6"]["R"]
        over = sum(1 for p in info["per_seed"] if float(p["R"]) > 1.0)
        extra = ""
        if over == 1:
            extra = ", 1 seed above 1"
        elif over > 1:
            extra = f", {over} seeds above 1"
        l4_bits.append(
            f"δ={delta:.2f} n12 R={r12:.3f} ({info['n_clear']}/{len(summary['seeds'])} seeds ≤ 0.85{extra}), "
            f"n6 holdout R={r6:.3f}"
        )
    if l4_bits:
        lines.append(
            "L=4 is reported and is not the gate. Clean B=2 at L=4 was a win "
            f"(R={B2_CLEAN[4]['R']:.3f}), while the disordered shell was softer "
            f"({SHELL49[0.15][4]['R']:.3f} / {SHELL49[0.30][4]['R']:.3f}). "
            + ". ".join(l4_bits)
            + "."
        )
        lines.append("")
    shells = []
    for delta in summary["deltas"]:
        info = summary["by_delta"][tag_delta(delta)]["depths"].get("2", {})
        if info.get("shell_rms_med") is not None:
            gap = info.get("phys_rms_max")
            shells.append(
                f"δ={delta:.2f} L=2 shell RMS {info['shell_rms_med']:.4f} "
                f"(max phys RMS {_fmt(gap, 2)})"
            )
    if shells:
        lines.append(
            "B=2 shells left the linear X mixer (" + "; ".join(shells) + "). "
            "Shell RMS is the distance of φ_k from the best line a + b k, which for this "
            "family is the k² content. Phys RMS is the distance outside span{1, k, k²} and is "
            "zero when the parameterization holds. Zero shell RMS would be φ_k = b k."
        )
    return lines


def _check_init(delta: float, unit: np.ndarray, depths) -> None:
    packs = _build_packs(delta, unit)
    for L in depths:
        Cn = cn_from_unsat(torch.tensor(packs["eval"][12][0][:1], dtype=torch.float64), 12)
        sf = prepare_state("fourier", Cn, init_angles("fourier", L, dtype=torch.float64))
        ang = init_method("B2", L)
        sb = _prepare("B2", Cn, ang)
        fid = float(torch.sum(sf.conj() * sb, dim=-1).abs()[0])
        log(f"δ={delta:.2f} L{L} init fidelity Fourier vs B=2 = {fid:.8f}")
        if fid < 1.0 - 1e-6:
            raise RuntimeError(f"init states differ at L={L}: fid={fid}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=str, default="0.15,0.30")
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--rounds", type=int, default=S.LBFGS_ROUNDS)
    ap.add_argument("--jobs", type=int, default=4, help="GS and train worker processes")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for n in depths:
        if n not in DEPTHS:
            raise SystemExit(f"depth {n} is outside {DEPTHS}")
    for d in deltas:
        if not any(abs(d - x) < 1e-12 for x in DELTAS):
            raise SystemExit(f"delta {d} is outside {DELTAS}")
    if any(s < 0 for s in seeds) or len(seeds) < 3 or len(seeds) > 5:
        raise SystemExit("use 3 to 5 nonnegative seeds")

    decision_self_test()
    unit = unit_pattern()
    RUN.mkdir(parents=True, exist_ok=True)
    log("pattern u[:12]", " ".join(f"{v:.4f}" for v in unit[:12]))
    log(
        "B2 recon",
        f"{float(np.max(np.abs(B2_INIT @ B2_BASIS - _PAPER_PHI))):.3e}",
        "torch",
        torch.__version__,
        "numpy",
        np.__version__,
    )

    if args.aggregate_only:
        aggregate(deltas, depths, seeds, unit)
        return

    if not args.skip_self_check:
        log("self-check")
        E6, _ = load_unsat(6, slice(0, 1))
        self_check(E6)
        log("self-check ok")

    jobs = max(1, int(args.jobs))
    for delta in deltas:
        tag = tag_delta(delta)
        for n, sl in (
            (6, slice(0, HOLD_HI)),
            (9, slice(0, HOLD_HI)),
            (12, slice(0, FREEZE_B)),
        ):
            E, sd = load_unsat(n, sl)
            ensure_gs(tag, n, sl, E, sd, x_coeff(n, delta, unit), jobs=jobs)
        _check_init(delta, unit, depths)

    specs = []
    for delta in deltas:
        for L in depths:
            for method in METHODS:
                for seed in seeds:
                    specs.append(
                        {
                            "delta": delta,
                            "L": L,
                            "method": method,
                            "seed": seed,
                            "rounds": int(args.rounds),
                            "force": bool(args.force),
                        }
                    )
    log(f"training jobs {len(specs)} workers={jobs} rounds={args.rounds}")
    if jobs <= 1 or len(specs) == 1:
        for spec in specs:
            _train_worker(spec)
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ctx.Pool(jobs, initializer=_pool_init) as pool:
            list(pool.imap_unordered(_train_worker, specs, chunksize=1))
    summary = aggregate(deltas, depths, seeds, unit)
    log("VERDICT", summary["verdict"], summary.get("calls"))


if __name__ == "__main__":
    main()
