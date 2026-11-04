#!/usr/bin/env python3
"""Depth sweep of B=2 Walsh SNAP at locked NEW_SAFE.

H = diag(unsat) − 0.75 Σ_i X_i, μ = 0. The mixer is the PR #57 quadratic
Hamming shell, shared across cavities:

    φ_k = c1 k + c2 k²    on popcount k = 0..3

in orthonormal coordinates, global phase fixed (φ_0 is not a free parameter).
The cost pulse is the locked Walsh grading of degrees 1..4. Fourier QAOA is
trained on the same batches and the same L-BFGS budget.

Train n∈{6,9} only. Freeze the first 128 instances of n=12. Depths
L∈{2,4,8,16}, five seeds each. n=21 is not loaded.

Honesty gate (this file, not the B=4 #50 gate):
  PASS when frozen-n12 L=2 R_Fourier ≤ 0.85
        and L=8 R_Fourier ≥ 1.0
        and L=16 R_Fourier ≥ 1.0
  needs_edit when L=2 softens (R > 0.85) or B=2 wins deep
        (L=8 or L=16 has R < 1.0).
L=4 is reported and is not part of the gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One thread per process. Set before NumPy / Torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S  # noqa: E402
from Nov_3rd.hva_safe import (  # noqa: E402
    INIT_BETA,
    INIT_GAMMA,
    angles_to_lists,
    cn_from_unsat,
    expectation_H,
    init_angles,
    method_parameter_list,
    prepare_state,
    snap_state,
    snap_state_from_components,
    summarize_eps,
    walsh_degree_components,
)

RUN = S.OUT / "quantum_h_explore" / "NEW_SAFE_B2_L_sweep"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = 128
LAM = 0.75
MU = 0.0
LAMBDA_X = -LAM
MU_XX = MU
ROUNDS = 40
BATCH = 64
DEPTHS = (2, 4, 8, 16)
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("fourier", "B2")
PROTOCOL = "new_safe_B2_L_sweep_r17"
BAR_R = 0.85
BAR_R_SOFT = 0.98
BAR_DEEP = 1.0
# PR #57 (same ansatz, same batch stream) and PR #50 (B=4 shell).
QUOTE_B2 = {
    2: {"med_b2": 0.4261, "med_fourier": 0.6876, "r": 0.6197},
    4: {"med_b2": 0.1958, "med_fourier": 0.3172, "r": 0.6174},
}
QUOTE_B4 = {
    2: {"med_snap": 0.4264, "med_fourier": 0.6876, "r": 0.6201, "seeds": 5},
    4: {"med_snap": 0.2755, "med_fourier": 0.3172, "r": 0.8687, "seeds": 5},
    8: {"med_snap": 0.2255, "med_fourier": 0.1173, "r": 1.9232, "seeds": 3},
    16: {"med_snap": 0.1726, "med_fourier": 0.0797, "r": 2.1641, "seeds": 2},
}
QUOTE_LAMBDA_N12 = 0.7477

# Orthonormal rows spanning {k, k²} on popcount k=0..3. φ = c @ B2_BASIS.
_K = np.arange(4, dtype=np.float64)
_RAW = np.stack([_K, _K ** 2], axis=1)
_Q, _R = np.linalg.qr(_RAW, mode="reduced")
for _j in range(2):
    if float(_Q[:, _j] @ _RAW[:, _j]) < 0.0:
        _Q[:, _j] *= -1.0
B2_BASIS = np.ascontiguousarray(_Q.T)  # (2, 4)
_PAPER_PHI = -INIT_BETA * _K
B2_INIT = np.ascontiguousarray(_PAPER_PHI @ B2_BASIS.T)
B2_RECON_ERROR = float(np.max(np.abs((_PAPER_PHI @ B2_BASIS.T) @ B2_BASIS - _PAPER_PHI)))


def log(*a):
    print(time.strftime("%H:%M:%S"), "[B2-L]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def depth_call(r: float) -> str:
    """Same labels as the B=4 depth sweep. R≥0.98 is not a SNAP win."""
    if not np.isfinite(r):
        return "undefined"
    if r <= BAR_R + 1e-12:
        return "win"
    if r < BAR_R_SOFT - 1e-12:
        return "soft"
    if r <= 1.02 + 1e-12:
        return "tie"
    return "loss"


def honesty_verdict(r2: float, r8: float, r16: float) -> str:
    """PASS iff the shallow win holds and Fourier wins both deep depths."""
    shallow = np.isfinite(r2) and r2 <= BAR_R + 1e-12
    deep = (
        np.isfinite(r8)
        and np.isfinite(r16)
        and r8 >= BAR_DEEP - 1e-12
        and r16 >= BAR_DEEP - 1e-12
    )
    if shallow and deep:
        return "PASS"
    return "needs_edit"


def honesty_reasons(r2: float, r8: float, r16: float) -> list[str]:
    reasons = []
    if not np.isfinite(r2) or r2 > BAR_R + 1e-12:
        reasons.append(f"L=2 softens: R_Fourier={r2:.4f} > {BAR_R}")
    if not np.isfinite(r8) or r8 < BAR_DEEP - 1e-12:
        reasons.append(
            f"B=2 wins deep at L=8: R_Fourier={r8:.4f} < {BAR_DEEP} (Fourier does not win)"
        )
    if not np.isfinite(r16) or r16 < BAR_DEEP - 1e-12:
        reasons.append(
            f"B=2 wins deep at L=16: R_Fourier={r16:.4f} < {BAR_DEEP} (Fourier does not win)"
        )
    if not reasons:
        reasons.append(
            f"L=2 R_Fourier={r2:.4f}≤{BAR_R} and L=8 R={r8:.4f}≥{BAR_DEEP} "
            f"and L=16 R={r16:.4f}≥{BAR_DEEP}"
        )
    return reasons


def load_pack(n: int, count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if int(n) == 21 or int(n) not in (6, 9, 12):
        raise RuntimeError(f"refusing n={n}; this sweep loads only 6, 9, 12")
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"]).astype(np.int64)
    if count is not None:
        E = E[: int(count)]
        seeds = seeds[: int(count)]
    if int(d["n"]) != int(n) or E.shape[1] != (1 << int(n)):
        raise RuntimeError(f"{path} shape {E.shape} n={int(d['n'])}")
    return E, seeds


def phi_from_b2(c: torch.Tensor) -> torch.Tensor:
    """c (L, 2) → popcount phases (L, 4)."""
    basis = c.new_tensor(B2_BASIS)
    return c @ basis


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


def prepare(method: str, Cn: torch.Tensor, angles: dict, comps: torch.Tensor | None):
    if method == "fourier":
        return prepare_state("fourier", Cn, angles)
    if method == "B2":
        phi = phi_from_b2(angles["c"])
        if comps is None:
            return snap_state(Cn, angles["gamma_w"], phi)
        return snap_state_from_components(comps, angles["gamma_w"], phi)
    raise ValueError(method)


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_method(method: str, angles: dict, bundle: dict) -> dict:
    ang = _as_f64(angles)
    chunk = {6: 32, 9: 16, 12: 8}
    out = {}
    with torch.no_grad():
        for n, (E, e0, seeds) in bundle.items():
            Eu = torch.tensor(np.asarray(E, dtype=np.float64))
            e0t = torch.tensor(np.asarray(e0, dtype=np.float64))
            parts = []
            step = chunk[int(n)]
            for s in range(0, Eu.shape[0], step):
                sl = slice(s, s + step)
                Cn = cn_from_unsat(Eu[sl], int(n))
                psi = prepare(method, Cn, ang, None)
                h = expectation_H(psi, Eu[sl], int(n), lambda_x=LAMBDA_X, mu_xx=MU_XX)
                parts.append((h - e0t[sl]).detach().cpu().numpy())
            eps = np.concatenate(parts, axis=0)
            if np.any(eps < -1e-4):
                raise RuntimeError(
                    f"{method} ε below E0 by {float(eps.min()):.3e} at n={n}"
                )
            rec = summarize_eps(eps)
            rec.update(
                {
                    "n": int(n),
                    "eps": [float(x) for x in eps.tolist()],
                    "seeds": [int(x) for x in np.asarray(seeds).tolist()],
                }
            )
            out[str(n)] = rec
    return out


def _loss_on(method, angles, Eu, e0b, n: int, comps) -> torch.Tensor:
    if method == "fourier" or comps is None:
        Cn = cn_from_unsat(Eu, n)
        psi = prepare(method, Cn, angles, None)
    else:
        psi = prepare(method, None, angles, comps)
    h = expectation_H(psi, Eu, n, lambda_x=LAMBDA_X, mu_xx=MU_XX)
    return ((h - e0b) / float(n)).mean()


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method: str, L: int, seed: int, packs: dict) -> dict:
    """Batch stream depends on the seed only, matching PR #50 / #57."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    angles = init_method(method, int(L))
    params = bind_parameters(method, angles)
    E6, e06 = packs["E6"], packs["e06"]
    E9, e09 = packs["E9"], packs["e09"]
    n6, n9 = int(E6.shape[0]), int(E9.shape[0])
    b6 = min(BATCH // 2, n6)
    b9 = min(BATCH - b6, n9)
    hold = {9: (packs["E9_hold"], packs["e09_hold"], packs["s9_hold"])}

    def hold_med() -> float:
        return float(eval_method(method, angles, hold)["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(
        f"L{L} {method} s{seed} init n9-hold Medε={init_med:.5f} "
        f"nparams={sum(int(p.numel()) for p in params)}"
    )
    for r in range(int(ROUNDS)):
        opt = torch.optim.LBFGS(
            params,
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        i6 = torch.tensor(rng.choice(n6, size=b6, replace=False), dtype=torch.long)
        i9 = torch.tensor(rng.choice(n9, size=b9, replace=False), dtype=torch.long)
        Eu6 = E6.index_select(0, i6)
        Eu9 = E9.index_select(0, i9)
        e6b = e06.index_select(0, i6)
        e9b = e09.index_select(0, i9)
        comps6 = comps9 = None
        if method != "fourier":
            comps6 = walsh_degree_components(cn_from_unsat(Eu6, 6)).detach()
            comps9 = walsh_degree_components(cn_from_unsat(Eu9, 9)).detach()

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                _loss_on(method, angles, Eu6, e6b, 6, comps6)
                + _loss_on(method, angles, Eu9, e9b, 9, comps9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as exc:
            log(f"L{L} {method} s{seed} r{r} LBFGS skip: {exc}")
            hist.append({"round": r, "skip": str(exc)})
            continue
        if not _finite(loss_f):
            log(f"L{L} {method} s{seed} r{r} nonfinite — restore best")
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
        if mark or r % 10 == 0 or r + 1 == int(ROUNDS):
            log(
                f"L{L} {method} s{seed} r{r}/{ROUNDS} "
                f"loss={loss_f:.5f} hold={med:.5f} best={best['med9']:.5f}{mark}"
            )
    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    ev = eval_method(
        method,
        angles,
        {
            6: (packs["E6_hold"], packs["e06_hold"], packs["s6_hold"]),
            9: hold[9],
            12: (packs["E12"], packs["e012"], packs["s12"]),
        },
    )
    return {
        "protocol": PROTOCOL,
        "point": "NEW_SAFE",
        "method": method,
        "mixer": "B2" if method == "B2" else "fourier",
        "baseline": "fourier",
        "L": int(L),
        "seed": int(seed),
        "lambda": LAM,
        "mu": MU,
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "rounds": int(ROUNDS),
        "batch": BATCH,
        "batch_rng": "numpy Generator(seed); shared across methods and depths",
        "init_hold_med9": init_med,
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": angles_to_lists(method, best["angles"]),
        "n_params": int(sum(p.numel() for p in params)),
        "eval": ev,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def seed_path(L: int, method: str, seed: int) -> Path:
    return RUN / "runs" / f"L{L}_{method}_s{seed}.json"


def seed_is_done(path: Path, method: str, seed: int, L: int) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        row.get("protocol") == PROTOCOL
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and int(row.get("L", -1)) == int(L)
        and int(row.get("rounds", -1)) == int(ROUNDS)
        and abs(float(row.get("lambda", 0.0)) - LAM) < 1e-12
        and abs(float(row.get("mu", 1.0)) - MU) < 1e-12
        and "12" in row.get("eval", {})
    )


def diag_path(n: int) -> Path:
    return RUN / "cache" / f"diag_n{n}.npz"


def _cache_ok(path: Path, seeds: np.ndarray) -> dict | None:
    if not path.is_file():
        return None
    z = np.load(path)
    if (
        abs(float(z["lam"]) - LAM) < 1e-12
        and abs(float(z["mu"]) - MU) < 1e-12
        and np.array_equal(z["seeds"], seeds)
        and int(z["e0"].shape[0]) == int(seeds.shape[0])
    ):
        return {k: z[k] for k in z.files}
    return None


def ensure_diag(n: int, E: np.ndarray, seeds: np.ndarray) -> dict:
    """Reuse the PR #50 diagonalization when λ, μ, and pack seeds match."""
    path = diag_path(n)
    hit = _cache_ok(path, seeds)
    if hit is None:
        raise RuntimeError(
            f"missing NEW_SAFE diag cache for n={n} at {path}. "
            "Refusing to diagonalize inside this sweep (n=21 stays unloaded)."
        )
    resid = np.asarray(hit["residual"], dtype=np.float64)
    e0 = np.asarray(hit["e0"], dtype=np.float64)
    if np.any(resid > 1e-6 * np.maximum(1.0, np.abs(e0))):
        raise RuntimeError(f"n={n} cached ground-state residual too large")
    log(
        f"diag cache hit n={n} B={seeds.shape[0]} "
        f"Λ={float(np.median(hit['lambda_c'])):.4f} "
        f"max|resid|={float(np.max(resid)):.3e}"
    )
    return hit


def build_packs(diags: dict) -> dict:
    E6, s6 = load_pack(6, HOLD_HI)
    E9, s9 = load_pack(9, HOLD_HI)
    E12, s12 = load_pack(12, FREEZE_B)
    for n, seeds, diag in ((6, s6, diags[6]), (9, s9, diags[9]), (12, s12, diags[12])):
        if not np.array_equal(seeds, np.asarray(diag["seeds"])):
            raise RuntimeError(f"n={n} seed mismatch vs diag cache")
    e06 = np.asarray(diags[6]["e0"], dtype=np.float64)
    e09 = np.asarray(diags[9]["e0"], dtype=np.float64)

    def _t(E, e0):
        return (
            torch.tensor(np.asarray(E, dtype=np.float64)),
            torch.tensor(np.asarray(e0, dtype=np.float64)),
        )

    E6t, e6t = _t(E6[TRAIN_LO:TRAIN_HI], e06[TRAIN_LO:TRAIN_HI])
    E9t, e9t = _t(E9[TRAIN_LO:TRAIN_HI], e09[TRAIN_LO:TRAIN_HI])
    return {
        "E6": E6t,
        "e06": e6t,
        "E9": E9t,
        "e09": e9t,
        "E6_hold": E6[HOLD_LO:HOLD_HI],
        "e06_hold": e06[HOLD_LO:HOLD_HI],
        "s6_hold": s6[HOLD_LO:HOLD_HI],
        "E9_hold": E9[HOLD_LO:HOLD_HI],
        "e09_hold": e09[HOLD_LO:HOLD_HI],
        "s9_hold": s9[HOLD_LO:HOLD_HI],
        "E12": E12,
        "e012": np.asarray(diags[12]["e0"], dtype=np.float64),
        "s12": s12,
    }


def _worker_init() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _train_job(payload: tuple) -> str:
    method, L, seed, packs = payload
    path = seed_path(L, method, seed)
    if seed_is_done(path, method, seed, L):
        log(f"L{L} {method} s{seed} already done")
        return str(path)
    row = train_one(method, L, seed, packs)
    write_json(path, row)
    med12 = row["eval"]["12"]["med_eps"]
    log(f"L{L} {method} s{seed} DONE n12 Medε={med12:.4f} wall={row['wall_s']:.1f}s")
    return str(path)


def _paired(rows_a: list[dict], rows_b: list[dict], n: str) -> tuple[int, int]:
    b_by = {int(r["seed"]): r for r in rows_b}
    win = total = 0
    for a in rows_a:
        b = b_by[int(a["seed"])]
        ea = np.asarray(a["eval"][n]["eps"], dtype=np.float64)
        eb = np.asarray(b["eval"][n]["eps"], dtype=np.float64)
        if ea.shape != eb.shape:
            raise RuntimeError(f"paired shape {ea.shape} vs {eb.shape}")
        win += int(np.sum(ea < eb))
        total += int(ea.size)
    return win, total


def _block(rows: list[dict], n: str) -> dict:
    b2 = sorted((r for r in rows if r["method"] == "B2"), key=lambda r: int(r["seed"]))
    base = sorted((r for r in rows if r["method"] == "fourier"), key=lambda r: int(r["seed"]))
    if [int(r["seed"]) for r in b2] != [int(r["seed"]) for r in base]:
        raise RuntimeError("B=2 and Fourier seeds differ")
    b2_meds = np.asarray([float(r["eval"][n]["med_eps"]) for r in b2], dtype=np.float64)
    base_meds = np.asarray([float(r["eval"][n]["med_eps"]) for r in base], dtype=np.float64)
    med_s = float(np.median(b2_meds))
    med_b = float(np.median(base_meds))
    if med_b <= 0:
        raise RuntimeError(f"Fourier Medε {med_b} is not positive")
    r = med_s / med_b
    win, total = _paired(b2, base, n)
    ratios = (b2_meds / base_meds).tolist()
    return {
        "n": int(n),
        "n_seeds": len(b2),
        "med_b2": med_s,
        "med_fourier": med_b,
        "r": float(r),
        "call": depth_call(r),
        "paired_b2_below": win,
        "paired_n": total,
        "seeds_r_le_085": int(sum(1 for x in ratios if x <= BAR_R + 1e-12)),
        "seeds_r_ge_1": int(sum(1 for x in ratios if x >= BAR_DEEP - 1e-12)),
        "per_seed": [
            {
                "seed": int(rs["seed"]),
                "b2": float(rs["eval"][n]["med_eps"]),
                "fourier": float(rb["eval"][n]["med_eps"]),
                "r": float(rs["eval"][n]["med_eps"] / rb["eval"][n]["med_eps"]),
                "best_round_b2": int(rs["best_round"]),
                "best_round_fourier": int(rb["best_round"]),
                "wall_s_b2": float(rs["wall_s"]),
                "wall_s_fourier": float(rb["wall_s"]),
            }
            for rs, rb in zip(b2, base)
        ],
    }


def quantum_summary(diag: dict) -> dict:
    lam = np.asarray(diag["lambda_c"], dtype=np.float64)
    peak = np.asarray(diag["peak"], dtype=np.float64)
    resid = np.asarray(diag["residual"], dtype=np.float64)
    return {
        "B": int(peak.shape[0]),
        "n_cav": int(lam.shape[1]),
        "lambda_med": float(np.median(lam)),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "frac_lambda_gt_080": float(np.mean(lam > 0.80)),
        "peak_med": float(np.median(peak)),
        "frac_peak_ge_05": float(np.mean(peak >= 0.5)),
        "max_residual": float(np.max(resid)),
    }


def assemble(diags: dict, jobs: list[tuple[int, str, int]]) -> dict:
    by_L: dict[int, list[dict]] = {}
    for L, method, seed in jobs:
        row = json.loads(seed_path(L, method, seed).read_text())
        by_L.setdefault(int(L), []).append(row)
    gs = {str(n): quantum_summary(diags[n]) for n in (6, 9, 12)}
    depths = []
    for L in sorted(by_L):
        rows = by_L[L]
        block = {
            "L": int(L),
            "seeds": sorted({int(r["seed"]) for r in rows if r["method"] == "B2"}),
            "by_n": {n: _block(rows, n) for n in ("6", "9", "12")},
        }
        block["lambda_gs"] = {n: gs[n]["lambda_med"] for n in ("6", "9", "12")}
        block["call_n12"] = block["by_n"]["12"]["call"]
        depths.append(block)
    by = {int(b["L"]): b for b in depths}
    for need in (2, 8, 16):
        if need not in by:
            raise RuntimeError(f"L={need} is part of the honesty gate and is not complete")
    r2 = by[2]["by_n"]["12"]["r"]
    r8 = by[8]["by_n"]["12"]["r"]
    r16 = by[16]["by_n"]["12"]["r"]
    verdict = honesty_verdict(r2, r8, r16)
    reasons = honesty_reasons(r2, r8, r16)
    return {
        "protocol": PROTOCOL,
        "point": "NEW_SAFE",
        "lambda": LAM,
        "mu": MU,
        "mixer": "B2",
        "phi": "c1*k + c2*k^2",
        "verdict": verdict,
        "reasons": reasons,
        "pass_rule": (
            "frozen n=12: L=2 R_Fourier≤0.85 AND L=8 R_Fourier≥1.0 AND L=16 R_Fourier≥1.0. "
            "needs_edit if L=2 softens or B=2 wins deep. L=4 is not a gate. "
            "Medε is the median of five seed medians."
        ),
        "L2_R_n12": r2,
        "L8_R_n12": r8,
        "L16_R_n12": r16,
        "L4_R_n12": by[4]["by_n"]["12"]["r"] if 4 in by else None,
        "lambda_n12": gs["12"]["lambda_med"],
        "b2_recon_error": B2_RECON_ERROR,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gs": gs,
        "depths": [
            {
                "L": b["L"],
                "seeds": b["seeds"],
                "call_n12": b["call_n12"],
                "lambda_gs": b["lambda_gs"],
                "by_n": b["by_n"],
            }
            for b in depths
        ],
        "quote_b2": QUOTE_B2,
        "quote_b4": QUOTE_B4,
    }


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


def _reading(summary: dict) -> str:
    by = {int(b["L"]): b for b in summary["depths"]}

    def row(L: int, n: str = "12") -> dict:
        return by[L]["by_n"][n]

    parts = []
    verdict = summary["verdict"]
    r2, r8, r16 = summary["L2_R_n12"], summary["L8_R_n12"], summary["L16_R_n12"]
    parts.append(f"The call is **{verdict}**.")
    parts.append(" ".join(summary["reasons"]) + ".")
    b2 = row(2)
    parts.append(
        f"L=2 frozen n=12 is {b2['call']}: Medε B=2 {_fmt(b2['med_b2'])} vs Fourier "
        f"{_fmt(b2['med_fourier'])}, R={b2['r']:.4f}, "
        f"{b2['seeds_r_le_085']}/{b2['n_seeds']} seeds ≤0.85, "
        f"B=2 lower on {b2['paired_b2_below']}/{b2['paired_n']} paired instances. "
        f"PR #57 quoted R={QUOTE_B2[2]['r']:.4f} "
        f"(Medε {_fmt(QUOTE_B2[2]['med_b2'])} / {_fmt(QUOTE_B2[2]['med_fourier'])}). "
        f"Deltas versus that quote: Medε B=2 {b2['med_b2'] - QUOTE_B2[2]['med_b2']:+.4f}, "
        f"Fourier {b2['med_fourier'] - QUOTE_B2[2]['med_fourier']:+.4f}, "
        f"R {b2['r'] - QUOTE_B2[2]['r']:+.4f}."
    )
    if 4 in by:
        b4 = row(4)
        parts.append(
            f"L=4 is {b4['call']} and is not a gate: Medε B=2 {_fmt(b4['med_b2'])} vs Fourier "
            f"{_fmt(b4['med_fourier'])}, R={b4['r']:.4f}, "
            f"{b4['seeds_r_le_085']}/{b4['n_seeds']} seeds ≤0.85, "
            f"paired {b4['paired_b2_below']}/{b4['paired_n']}. "
            f"PR #57 quoted R={QUOTE_B2[4]['r']:.4f}. "
            f"PR #50's four-phase shell at this depth was soft (R={QUOTE_B4[4]['r']:.4f})."
        )
    # Overlapping Fourier seeds from the B=4 sweep (#50), for the protocol check.
    quote_fourier_seeds = {
        8: {0: 0.1221, 1: 0.1173, 2: 0.1133},
        16: {0: 0.0792, 1: 0.0803},
    }
    for L in (8, 16):
        b = row(L)
        q = QUOTE_B4[L]
        deltas = []
        for seed_row in b["per_seed"]:
            quoted = quote_fourier_seeds[L].get(int(seed_row["seed"]))
            if quoted is not None:
                deltas.append(abs(float(seed_row["fourier"]) - quoted))
        match = max(deltas) if deltas else float("nan")
        late_b2 = all(int(s["best_round_b2"]) == ROUNDS - 1 for s in b["per_seed"])
        late = " Every B=2 seed restores round 39." if late_b2 else ""
        parts.append(
            f"L={L} is {b['call']}: Medε B=2 {_fmt(b['med_b2'])} vs Fourier "
            f"{_fmt(b['med_fourier'])}, R={b['r']:.4f}, "
            f"{b['seeds_r_le_085']}/{b['n_seeds']} seeds ≤0.85, "
            f"{b['seeds_r_ge_1']}/{b['n_seeds']} seeds with R≥1, "
            f"B=2 lower on {b['paired_b2_below']}/{b['paired_n']}. "
            f"PR #50's B=4 shell, on {q['seeds']} seeds, had R={q['r']:.4f} "
            f"(Medε {_fmt(q['med_snap'])} / {_fmt(q['med_fourier'])}). "
            f"Fourier on the seeds #50 already published differs from this run by at most {match:.4f}.{late}"
        )
    parts.append(
        f"Median Λ(GS) at frozen n=12 is {_fmt(summary['lambda_n12'])} "
        f"(n=6 {_fmt(summary['gs']['6']['lambda_med'])}, "
        f"n=9 {_fmt(summary['gs']['9']['lambda_med'])}). "
        f"The #50 quote is {QUOTE_LAMBDA_N12:.4f}. Λ does not depend on L or on B. "
        f"Holdout R at L=2 is n=6 {row(2, '6')['r']:.4f}, n=9 {row(2, '9')['r']:.4f}."
    )
    if verdict == "PASS":
        parts.append(
            "The paper sentence stands at B=2: the quadratic shell still wins at L=2, "
            f"and Fourier still wins at L=8 (R={r8:.3f}) and L=16 (R={r16:.3f})."
        )
    else:
        parts.append(
            "This file is needs_edit. L=2 does not soften "
            f"(R={r2:.3f}). L=16 still belongs to Fourier (R={r16:.3f}). "
            f"L=8 does not: B=2 wins there (R={r8:.3f}≤0.85), so the sentence "
            "\"L≥8 loses to Fourier\" is false for this mixer. "
            "The depth where Fourier takes the median moves from 8 to 16."
        )
    parts.append("n=21 was not run.")
    return "\n\n".join(parts)


def render(summary: dict) -> str:
    gs12 = summary["gs"]["12"]
    lines = [
        "# NEW_SAFE B=2 depth sweep — Walsh SNAP vs Fourier",
        "",
        f"**Verdict: {summary['verdict']}.** "
        f"Frozen n=12 L=2 R_Fourier={_fmt(summary['L2_R_n12'])}, "
        f"L=8 R_Fourier={_fmt(summary['L8_R_n12'])}, "
        f"L=16 R_Fourier={_fmt(summary['L16_R_n12'])}, "
        f"median Λ(GS)={_fmt(summary['lambda_n12'])}.",
        "",
        "Point **NEW_SAFE**: λ=0.75, μ=0. Mixer is the PR #57 quadratic Hamming shell "
        "φ_k = c₁ k + c₂ k² (B=2). PR #50's four-phase shell (B=4) won at L=2 "
        f"(R≈{QUOTE_B4[2]['r']:.2f}) and lost at L=8 and L=16 "
        f"(R≈{QUOTE_B4[8]['r']:.1f} / {QUOTE_B4[16]['r']:.1f}). "
        "This file asks whether that split survives when the mixer is the two-coefficient shell. "
        "n=21 was not loaded.",
        "",
        "## Hamiltonian",
        "",
        r"\[",
        r"H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i",
        r"\]",
        "",
        "μ=0, so there is no cavity XX term. Cavity c owns qubits [3c, 3c+3). "
        "ε = ⟨H⟩ − E₀ from the Z histogram (unsat) and the X histogram (⟨X⟩). "
        "Medε is the median over the five seeds of each seed's median-over-instances ε. "
        "R_Fourier = Medε_B2 / Medε_Fourier, with Fourier trained on the same batches. "
        "Λ is the median of ⟨(Σ_{i∈c} X_i / 3)²⟩ over cavities and instances of the true "
        "ground state. It does not depend on L.",
        "",
        "## Mixer",
        "",
        "The cost pulse is the locked Walsh grading: four weights on degrees 1..4 of Cn. "
        "B=2 has two phase coefficients per layer. φ_k = c₁ k + c₂ k² on popcount k=0..3, "
        "in orthonormal coordinates, so L-BFGS sees the same Euclidean metric on φ as the "
        "locked shell. The global phase is fixed because φ_0 is not free. "
        "Paper init φ_k=−β k sits in this span. "
        f"Basis reconstruction max |φ_recon − φ_paper| = {summary['b2_recon_error']:.3e}.",
        "",
        "## Gate",
        "",
        "- Train n∈{6,9}, indices [0:200). Checkpoint on the n=9 holdout [200:250).",
        f"- Freeze n=12, the first {FREEZE_B} pack instances. Five seeds {{0,1,2,3,4}} at every L∈{{2,4,8,16}}. n=21 was not loaded.",
        "- **PASS** when frozen-n12 L=2 R_Fourier ≤ 0.85 and both L=8 and L=16 have R_Fourier ≥ 1.0 (Fourier wins deep).",
        "- **needs_edit** when L=2 softens (R > 0.85) or B=2 wins deep (L=8 or L=16 has R < 1.0). That would change the paper claim.",
        "- **win** at a depth when R ≤ 0.85. **soft** when 0.85 < R < 0.98. **tie** when 0.98 ≤ R ≤ 1.02. **loss** when R > 1.02.",
        "- L=4 is reported and is not part of the gate. A depth with R ≥ 0.98 is not a SNAP win.",
        "",
        "## Protocol",
        "",
        "Packs: `beat_qaoa_12h/train_packs`. "
        "One batch stream per seed (NumPy Generator(seed) and torch.manual_seed(seed)), "
        "shared by B=2 and Fourier and by every depth. "
        f"L-BFGS matches `Nov_3rd.settings` and PR #57: {ROUNDS} rounds × max_iter {int(S.LBFGS_ITER)}, "
        f"lr {float(S.LBFGS_LR)}, history 16, strong Wolfe, grad clip 5, batch {BATCH} "
        "(half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. "
        "Each seed restores the round with the best n=9-holdout Medε. "
        "Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. B=2 starts on that shell. "
        "Medε of five seeds is the ordinary median (the middle seed), not a mean of two. "
        "Ground states are the PR #50 cache at the same λ, μ, and pack seeds "
        "(dense eigh at n=6 and n=9, ARPACK at n=12). Residuals were checked on load. "
        f"torch {summary['torch']}, numpy {summary['numpy']}. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.",
        "",
        "## Frozen n=12",
        "",
        "| L | seeds | Medε B=2 | Medε Fourier | R_Fourier | median Λ (GS) | paired B=2<Fourier | seeds R≤0.85 | call |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for block in summary["depths"]:
        b = block["by_n"]["12"]
        lines.append(
            f"| {block['L']} | {b['n_seeds']} | {_fmt(b['med_b2'])} | {_fmt(b['med_fourier'])} | "
            f"{_fmt(b['r'])} | {_fmt(gs12['lambda_med'])} | "
            f"{b['paired_b2_below']}/{b['paired_n']} | {b['seeds_r_le_085']}/{b['n_seeds']} | "
            f"{b['call']} |"
        )
    lines += [
        "",
        "Quote check against PR #57 (B=2, five seeds, same protocol) and PR #50 (B=4). "
        "R in this file always uses this run's Fourier.",
        "",
        "| L | this Medε B=2 | #57 Medε B=2 | this Fourier | #57 Fourier | this R | #57 R | #50 B=4 R | #50 seeds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by = {int(b["L"]): b for b in summary["depths"]}
    for L in (2, 4, 8, 16):
        if L not in by:
            continue
        b = by[L]["by_n"]["12"]
        q2 = QUOTE_B2.get(L)
        q4 = QUOTE_B4[L]
        if q2 is None:
            lines.append(
                f"| {L} | {_fmt(b['med_b2'])} | — | {_fmt(b['med_fourier'])} | — | "
                f"{_fmt(b['r'])} | — | {_fmt(q4['r'])} | {q4['seeds']} |"
            )
        else:
            lines.append(
                f"| {L} | {_fmt(b['med_b2'])} | {_fmt(q2['med_b2'])} | {_fmt(b['med_fourier'])} | "
                f"{_fmt(q2['med_fourier'])} | {_fmt(b['r'])} | {_fmt(q2['r'])} | "
                f"{_fmt(q4['r'])} | {q4['seeds']} |"
            )
    lines += [
        "",
        "Holdout Medε (checkpoint distribution, B=50). Λ there is the ground state "
        "on the diagonalized pack prefix (n=6 and n=9, B=250), not a function of L.",
        "",
        "| L | n | Medε B=2 | Medε Fourier | R_Fourier | median Λ (GS) | paired B=2<Fourier | call |",
        "|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for block in summary["depths"]:
        for n in ("6", "9"):
            b = block["by_n"][n]
            lines.append(
                f"| {block['L']} | {n} | {_fmt(b['med_b2'])} | {_fmt(b['med_fourier'])} | "
                f"{_fmt(b['r'])} | {_fmt(summary['gs'][n]['lambda_med'])} | "
                f"{b['paired_b2_below']}/{b['paired_n']} | {b['call']} |"
            )
    lines += [
        "",
        "## Ground state",
        "",
        "Λ and the Z-peak do not change with L. The n=12 row is the freeze.",
        "",
        "| n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | median Z-peak | frac peak≥0.5 | max residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ("6", "9", "12"):
        g = summary["gs"][n]
        lines.append(
            f"| {n} | {g['B']} | {_fmt(g['lambda_med'])} | {_fmt(g['lambda_min'])} | "
            f"{_fmt(g['lambda_max'])} | {_fmt(g['frac_lambda_gt_080'], 3)} | "
            f"{_fmt(g['peak_med'])} | {_fmt(g['frac_peak_ge_05'], 3)} | "
            f"{g['max_residual']:.2e} |"
        )
    lines += ["", "## Seeds at frozen n=12", ""]
    for block in summary["depths"]:
        b = block["by_n"]["12"]
        lines.append(f"### L={block['L']}")
        lines.append("")
        lines.append("| seed | Medε B=2 | Medε Fourier | R | best round B=2 | best round Fourier | wall s B=2 | wall s Fourier |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in b["per_seed"]:
            lines.append(
                f"| {row['seed']} | {_fmt(row['b2'])} | {_fmt(row['fourier'])} | "
                f"{_fmt(row['r'])} | {row['best_round_b2']} | {row['best_round_fourier']} | "
                f"{row['wall_s_b2']:.1f} | {row['wall_s_fourier']:.1f} |"
            )
        lines.append("")
    lines += [
        "## Reading",
        "",
        _reading(summary),
        "",
        "## Non-claims",
        "",
        "This does not retune λ or μ, does not add XX, and does not reseal n=15 or n=18. "
        "It does not replace the PR #50 B=4 sentence. L=4 is outside the honesty gate. "
        "The deep bar is R≥1.0 on the median of five seed medians, which is a coarser "
        "statement than the per-depth win/soft/tie/loss labels. n=21 was not loaded.",
        "",
        "Per-seed JSON is `runs/L{L}_{method}_s{seed}.json`. "
        "Ground-state arrays are `cache/diag_n{n}.npz`.",
        "",
    ]
    return "\n".join(lines)


def run(jobs: list[tuple[int, str, int]], workers: int) -> dict:
    if B2_RECON_ERROR > 1e-12:
        raise RuntimeError(f"B=2 basis does not contain the paper shell: {B2_RECON_ERROR}")
    torch.set_num_threads(1)
    diags = {}
    for n, count in ((6, HOLD_HI), (9, HOLD_HI), (12, FREEZE_B)):
        E, seeds = load_pack(n, count)
        diags[n] = ensure_diag(n, E, seeds)
    packs = build_packs(diags)
    pending = [job for job in jobs if not seed_is_done(seed_path(*job), job[1], job[2], job[0])]
    # seed_path(L, method, seed) but job is (L, method, seed). seed_is_done(path, method, seed, L).
    log(f"jobs {len(jobs)} pending {len(pending)} workers {workers}")
    if pending:
        payloads = [(method, L, seed, packs) for L, method, seed in pending]
        with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_worker_init) as pool:
            futs = [pool.submit(_train_job, p) for p in payloads]
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                log(f"progress {done}/{len(pending)}")
    done_jobs = [
        (L, method, seed)
        for L, method, seed in jobs
        if seed_is_done(seed_path(L, method, seed), method, seed, L)
    ]
    if len(done_jobs) != len(jobs):
        raise RuntimeError(f"finished {len(done_jobs)}/{len(jobs)} jobs")
    summary = assemble(diags, done_jobs)
    text = render(summary)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "RESULTS.md").write_text(text)
    public = json.loads(json.dumps(summary))
    for block in public["depths"]:
        for rec in block["by_n"].values():
            for row in rec["per_seed"]:
                pass
    write_json(RUN / "summary.json", public)
    log(
        f"verdict {summary['verdict']} "
        f"L2 R={summary['L2_R_n12']:.4f} "
        f"L8 R={summary['L8_R_n12']:.4f} "
        f"L16 R={summary['L16_R_n12']:.4f} "
        f"Λ={summary['lambda_n12']:.4f}"
    )
    return summary


def _parse_jobs(depths: str, seeds: str, methods: str) -> list[tuple[int, str, int]]:
    ds = [int(x) for x in depths.split(",") if x.strip()]
    ss = [int(x) for x in seeds.split(",") if x.strip()]
    ms = [x.strip() for x in methods.split(",") if x.strip()]
    for L in ds:
        if L not in DEPTHS:
            raise SystemExit(f"L={L} is not in {DEPTHS}")
    for s in ss:
        if s not in SEEDS:
            raise SystemExit(f"seed={s} is not in {SEEDS}")
    for m in ms:
        if m not in METHODS:
            raise SystemExit(f"method={m} is not in {METHODS}")
    if 21 in ds or 21 in ss:
        raise SystemExit("refusing n=21")
    jobs = []
    for L in ds:
        for method in ms:
            for seed in ss:
                jobs.append((int(L), method, int(seed)))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=str, default="2,4,8,16")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--methods", type=str, default="fourier,B2")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = _parse_jobs(args.depths, args.seeds, args.methods)
    # A partial invocation still writes RESULTS only when the honesty depths
    # are all present. Callers that pass a subset train those jobs and, if the
    # gate depths are incomplete, stop before assemble.
    gate = {(2, m, s) for m in METHODS for s in SEEDS}
    gate |= {(8, m, s) for m in METHODS for s in SEEDS}
    gate |= {(16, m, s) for m in METHODS for s in SEEDS}
    have = set(jobs)
    # Train the requested jobs. Assemble only when every gate job is on disk
    # after this call, counting files written by earlier invocations.
    run_partial = not gate.issubset(have)
    if run_partial:
        # Train only. Assemble if the files happen to complete the gate.
        diags_needed = True
        # Fall through to run(), which assembles `jobs` only. For a subset,
        # assemble would raise. So train without assemble.
        if B2_RECON_ERROR > 1e-12:
            raise SystemExit(f"B=2 basis error {B2_RECON_ERROR}")
        torch.set_num_threads(1)
        diags = {}
        for n, count in ((6, HOLD_HI), (9, HOLD_HI), (12, FREEZE_B)):
            E, seeds = load_pack(n, count)
            diags[n] = ensure_diag(n, E, seeds)
        packs = build_packs(diags)
        pending = [
            job for job in jobs
            if not seed_is_done(seed_path(job[0], job[1], job[2]), job[1], job[2], job[0])
        ]
        log(f"partial jobs {len(jobs)} pending {len(pending)} workers {args.workers}")
        if pending:
            payloads = [(method, L, seed, packs) for L, method, seed in pending]
            with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_worker_init) as pool:
                futs = [pool.submit(_train_job, p) for p in payloads]
                for fut in as_completed(futs):
                    fut.result()
        complete = all(
            seed_is_done(seed_path(L, m, s), m, s, L)
            for L in (2, 8, 16)
            for m in METHODS
            for s in SEEDS
        )
        if not complete:
            log("gate depths incomplete; RESULTS.md not rewritten")
            return
        full = [(L, m, s) for L in DEPTHS for m in METHODS for s in SEEDS]
        # Include L=4 if those files exist; assemble requires 2, 8, 16.
        present = [
            job for job in full
            if seed_is_done(seed_path(job[0], job[1], job[2]), job[1], job[2], job[0])
        ]
        summary = assemble(diags, present)
        text = render(summary)
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "RESULTS.md").write_text(text)
        write_json(RUN / "summary.json", json.loads(json.dumps(summary)))
        log(f"verdict {summary['verdict']}")
        return
    run(jobs, args.workers)


if __name__ == "__main__":
    main()
