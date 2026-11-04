#!/usr/bin/env python3
"""Walsh-shell coefficient ablation at locked NEW_SAFE.

H = diag(unsat) - 0.75 * sum_i X_i,  μ = 0.

The cost pulse stays the locked Walsh grading (degrees 1..4). 4-SAT unsat
has no Walsh power above degree 4, so that grading is already the full cost
spectrum. What still compresses is the SNAP phase inside each 3-bit cavity.
B is that phase budget:

  B=2   two coefficients, span{k, k^2} on popcount k=0..3
        (codebase shell polynomial with the cubic dropped; Walsh degrees ≤2
        modulo a fixed global phase). Orthonormal coordinates so L-BFGS sees
        the same Euclidean metric on φ as the locked shell.
  B=4   four popcount phases φ_0..φ_3. This is the locked Walsh shell
        (PR #50 / #51 / #52). Same tensors and same L-BFGS path as that trainer.
  full  eight Fock phases on the cavity, one per computational basis state.
        The Hamming shell is the popcount-tied subspace of this family.

Matched baseline: Fourier QAOA, same L, same batches, same L-BFGS budget.
Train n∈{6,9} only. Freeze the first 128 instances of n=12. L=2 is the gate,
L=4 is a soft check. n=21 is not loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# One thread per process. The pool sets the width.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S  # noqa: E402
from Nov_3rd.hva_safe import (  # noqa: E402
    INIT_BETA,
    INIT_GAMMA,
    angles_to_lists,
    apply_all_cavities,
    cn_from_unsat,
    expectation_H,
    fwht_unnormalized,
    ground_bundle,
    hadamard3,
    init_angles,
    lambda_of_state,
    method_parameter_list,
    plus_state,
    popcount_index,
    self_check,
    shell_unitary,
    snap_state,
    snap_state_from_components,
    summarize_eps,
    walsh_degree_components,
    walsh_power_by_degree,
)

RUN = S.OUT / "quantum_h_explore" / "NEW_SAFE_B_ablation"
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
DEPTHS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("fourier", "B2", "B4", "full")
PROTOCOL = "new_safe_B_ablation_r15"
BAR_R = 0.85
BAR_R_SOFT = 0.98
COMPRESS_MAX = 1.3
# Quote from PR #50 (same packs, same budget). #51 sits 0.0008 higher in R.
QUOTE_L2 = {"med_snap": 0.4264, "med_fourier": 0.6876, "r": 0.6201, "lambda": 0.7477}

# Orthonormal rows spanning {k, k^2} on popcount k=0..3. φ = c @ B2_BASIS.
_K = np.arange(4, dtype=np.float64)
_RAW = np.stack([_K, _K ** 2], axis=1)  # (4, 2)
_Q, _R = np.linalg.qr(_RAW, mode="reduced")
for _j in range(2):
    if float(_Q[:, _j] @ _RAW[:, _j]) < 0.0:
        _Q[:, _j] *= -1.0
B2_BASIS = np.ascontiguousarray(_Q.T)  # (2, 4)
_PAPER_PHI = -INIT_BETA * _K
B2_INIT = np.ascontiguousarray(_PAPER_PHI @ B2_BASIS.T)  # (2,)
_PC8 = np.asarray((0, 1, 1, 2, 1, 2, 2, 3), dtype=np.int64)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[B-abl]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def r_call(r: float) -> str:
    if not np.isfinite(r):
        return "undefined"
    if r <= BAR_R + 1e-12:
        return "win"
    if r < BAR_R_SOFT - 1e-12:
        return "soft"
    if r <= 1.02 + 1e-12:
        return "tie"
    return "loss"


def load_pack(n: int) -> tuple[np.ndarray, np.ndarray]:
    if int(n) not in (6, 9, 12):
        raise RuntimeError(f"refusing n={n}; this ablation loads only 6, 9, 12")
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"]).astype(np.int64)
    if int(d["n"]) != int(n) or E.shape[1] != (1 << int(n)):
        raise RuntimeError(f"{path} shape {E.shape} n={int(d['n'])}")
    return E, seeds


def phi_from_b2(c: torch.Tensor) -> torch.Tensor:
    """c (L, 2) → popcount phases (L, 4)."""
    basis = c.new_tensor(B2_BASIS)
    return c @ basis


def theta_from_phi(phi: torch.Tensor) -> torch.Tensor:
    """Popcount phases (L, 4) → Fock phases (L, 8), tied on Hamming weight."""
    pc = torch.as_tensor(_PC8, device=phi.device)
    return phi.index_select(1, pc)


def fock_unitary(theta8: torch.Tensor, H3: torch.Tensor) -> torch.Tensor:
    """H^{⊗3} diag(e^{−i θ}) H^{⊗3}. theta8 is (8,)."""
    phase = torch.exp(-1j * theta8.to(dtype=H3.dtype))
    return H3 @ torch.diag(phase) @ H3


def snap_fock_from_components(
    comps: torch.Tensor,
    gamma_w: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Walsh-graded cost, then a shared 8-phase SNAP. theta is (L, 8)."""
    if theta.ndim != 2 or int(theta.shape[1]) != 8:
        raise ValueError(f"theta shape {tuple(theta.shape)} != (L, 8)")
    if gamma_w.shape[0] != theta.shape[0] or int(gamma_w.shape[1]) != 4:
        raise ValueError("gamma_w must be (L, 4) aligned with theta")
    B, dim = int(comps.shape[1]), int(comps.shape[2])
    n = int(round(np.log2(dim)))
    H3 = hadamard3(comps.dtype, comps.device)
    psi = plus_state(B, n, comps.dtype, comps.device)
    for ell in range(int(theta.shape[0])):
        diag = torch.tensordot(gamma_w[ell], comps, dims=([0], [0]))
        psi = psi * torch.exp(-1j * diag.to(dtype=psi.dtype))
        psi = apply_all_cavities(psi, fock_unitary(theta[ell], H3), n)
    return psi


def init_method(method: str, L: int) -> dict:
    L = int(L)
    if method == "fourier":
        return init_angles("fourier", L, dtype=torch.float64)
    if method == "B4":
        return init_angles("snap", L, dtype=torch.float64)
    if method == "B2":
        c0 = torch.tensor(B2_INIT, dtype=torch.float64)
        return {
            "gamma_w": torch.full((L, 4), INIT_GAMMA, dtype=torch.float64),
            "c": c0.reshape(1, 2).expand(L, 2).contiguous().clone(),
        }
    if method == "full":
        pc = torch.as_tensor(_PC8, dtype=torch.float64)
        theta0 = (-INIT_BETA * pc).reshape(1, 8).expand(L, 8).contiguous().clone()
        return {
            "gamma_w": torch.full((L, 4), INIT_GAMMA, dtype=torch.float64),
            "theta": theta0,
        }
    raise ValueError(method)


def bind_parameters(method: str, angles: dict) -> list[torch.nn.Parameter]:
    if method == "fourier":
        return method_parameter_list("fourier", angles)
    if method == "B4":
        return method_parameter_list("snap", angles)
    keys = {"B2": ("gamma_w", "c"), "full": ("gamma_w", "theta")}[method]
    params = []
    for k in keys:
        p = torch.nn.Parameter(angles[k].detach().clone())
        angles[k] = p
        params.append(p)
    return params


def shell_phases(method: str, angles: dict) -> torch.Tensor | None:
    """Popcount phases when the arm is a shell. None for the 8-phase arm."""
    if method == "B4":
        return angles["phi"]
    if method == "B2":
        return phi_from_b2(angles["c"])
    if method == "full":
        return None
    raise ValueError(method)


def prepare(method: str, Cn: torch.Tensor, angles: dict, comps: torch.Tensor | None):
    if method == "fourier":
        from Nov_3rd.hva_safe import prepare_state

        return prepare_state("fourier", Cn, angles)
    if method in ("B2", "B4"):
        phi = shell_phases(method, angles)
        if comps is None:
            return snap_state(Cn, angles["gamma_w"], phi)
        return snap_state_from_components(comps, angles["gamma_w"], phi)
    if method == "full":
        if comps is None:
            comps = walsh_degree_components(Cn)
        return snap_fock_from_components(comps, angles["gamma_w"], angles["theta"])
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


def physical_shell_gap(method: str, angles: dict) -> float:
    """Mean over layers of the RMS residual of popcount phases outside span{1, k, k²}.

    A constant on all popcounts is a global phase of each cavity unitary and does
    not change ⟨H⟩. span{k, k²} is the B=2 family. The distance that matters is
    therefore the residual against span{1, k, k²}, not against span{k, k²} alone.
    """
    if method in ("fourier", "B2"):
        return 0.0
    if method == "B4":
        phi = np.asarray(angles["phi"], dtype=np.float64)
    elif method == "full":
        theta = np.asarray(angles["theta"], dtype=np.float64)
        phi = np.stack(
            [[float(theta[ell, _PC8 == k].mean()) for k in range(4)] for ell in range(theta.shape[0])]
        )
    else:
        return float("nan")
    k = np.arange(4.0)
    design = np.stack([np.ones(4), k, k ** 2], axis=1)
    acc = []
    for ell in range(phi.shape[0]):
        coef, *_ = np.linalg.lstsq(design, phi[ell], rcond=None)
        resid = phi[ell] - design @ coef
        acc.append(float(np.sqrt(np.mean(resid ** 2))))
    return float(np.mean(acc)) if acc else 0.0


def phase_diagnostics(method: str, angles: dict) -> dict:
    """How far the trained phases sit from a Hamming shell, and from B=2."""
    if method == "fourier":
        return {}
    with torch.no_grad():
        if method == "full":
            theta = angles["theta"].detach().to(dtype=torch.float64).cpu().numpy()
        else:
            phi = shell_phases(method, angles).detach().to(dtype=torch.float64).cpu().numpy()
            theta = phi[:, _PC8]
    # Within each popcount, the std over the Fock states that share it.
    stds = []
    for ell in range(theta.shape[0]):
        for k in range(4):
            vals = theta[ell, _PC8 == k]
            stds.append(float(np.std(vals)))
    # Distance of the popcount-mean phase to the B=2 span, after removing nothing
    # (B=2 already contains the paper gauge φ_0 = 0 only as a subspace, so compare
    # the four popcount means to their projection onto span{k, k^2}).
    means = np.stack(
        [np.array([theta[ell, _PC8 == k].mean() for k in range(4)]) for ell in range(theta.shape[0])]
    )
    proj = (means @ B2_BASIS.T) @ B2_BASIS
    resid = means - proj
    return {
        "max_within_popcount_std": float(np.max(stds)),
        "mean_within_popcount_std": float(np.mean(stds)),
        "rms_shell_to_B2": float(np.sqrt(np.mean(resid ** 2))),
    }


def train_one(method: str, L: int, seed: int, packs: dict) -> dict:
    """Batch stream depends on the seed only, matching the Round-12 / #50 lock."""
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
    log(f"L{L} {method} s{seed} init n9-hold Medε={init_med:.5f} nparams={sum(p.numel() for p in params)}")
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
        "phase_diag": phase_diagnostics(method, angles),
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


def _gs_job(payload: tuple) -> dict:
    idx, n, unsat = payload
    rec = ground_bundle(unsat, int(n), lambda_x=LAMBDA_X, mu_xx=MU_XX)
    psi = np.asarray(rec["psi"], dtype=np.float64)
    lam = np.asarray(lambda_of_state(psi, int(n)), dtype=np.float64).reshape(-1)
    return {
        "i": int(idx),
        "E0": float(rec["E0"]),
        "peak": float(rec["peak"]),
        "residual": float(rec["residual"]),
        "lambda_c": lam,
    }


def ensure_diag(n: int, E: np.ndarray, seeds: np.ndarray, workers: int) -> dict:
    path = RUN / "cache" / f"diag_n{n}.npz"
    if path.is_file():
        z = np.load(path)
        if (
            abs(float(z["lam"]) - LAM) < 1e-12
            and abs(float(z["mu"]) - MU) < 1e-12
            and np.array_equal(z["seeds"], seeds)
            and int(z["e0"].shape[0]) == int(seeds.shape[0])
        ):
            log(f"diag n={n} cache hit B={seeds.shape[0]}")
            return {k: z[k] for k in z.files}
    log(f"diag n={n} B={E.shape[0]} workers={workers}")
    jobs = [(i, n, np.asarray(E[i], dtype=np.float64)) for i in range(E.shape[0])]
    rows = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_gs_job, job) for job in jobs]
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows[row["i"]] = row
            done += 1
            if done == len(jobs) or done % 16 == 0:
                log(f"diag n={n} {done}/{len(jobs)} E0={row['E0']:.4f}")
    e0 = np.array([r["E0"] for r in rows], dtype=np.float64)
    peak = np.array([r["peak"] for r in rows], dtype=np.float64)
    residual = np.array([r["residual"] for r in rows], dtype=np.float64)
    lam = np.stack([r["lambda_c"] for r in rows], axis=0)
    payload = {
        "lam": np.float64(LAM),
        "mu": np.float64(MU),
        "lambda_x": np.float64(LAMBDA_X),
        "mu_xx": np.float64(MU_XX),
        "seeds": seeds.astype(np.int64),
        "e0": e0,
        "peak": peak,
        "residual": residual,
        "lambda_c": lam,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return payload


def build_packs(diags: dict) -> dict:
    def sl(n, a, b):
        E = diags[n]["E"]
        return E[a:b], diags[n]["e0"][a:b], diags[n]["seeds"][a:b]

    E6, e06, s6 = sl(6, TRAIN_LO, TRAIN_HI)
    E9, e09, s9 = sl(9, TRAIN_LO, TRAIN_HI)
    E6h, e06h, s6h = sl(6, HOLD_LO, HOLD_HI)
    E9h, e09h, s9h = sl(9, HOLD_LO, HOLD_HI)
    E12, e012, s12 = sl(12, 0, FREEZE_B)
    return {
        "E6": torch.tensor(E6, dtype=torch.float64),
        "e06": torch.tensor(e06, dtype=torch.float64),
        "s6": s6,
        "E9": torch.tensor(E9, dtype=torch.float64),
        "e09": torch.tensor(e09, dtype=torch.float64),
        "s9": s9,
        "E6_hold": E6h,
        "e06_hold": e06h,
        "s6_hold": s6h,
        "E9_hold": E9h,
        "e09_hold": e09h,
        "s9_hold": s9h,
        "E12": E12,
        "e012": e012,
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
        win += int(np.sum(ea < eb - 1e-12))
        total += int(ea.size)
    return win, total


def _median_block(rows: list[dict], n: str) -> dict:
    rows = sorted(rows, key=lambda r: int(r["seed"]))
    meds = [float(r["eval"][n]["med_eps"]) for r in rows]
    return {
        "n_seeds": len(rows),
        "med": float(np.median(np.asarray(meds, dtype=np.float64))),
        "per_seed_med": meds,
        "seeds": [int(r["seed"]) for r in rows],
    }


def _mean_walsh_power(diags: dict) -> dict:
    """Mean normalized Walsh power by degree on the packs that were diagonalized."""
    out = {}
    for n, rec in diags.items():
        E = np.asarray(rec["E"], dtype=np.float64)
        acc = np.zeros(int(n) + 1, dtype=np.float64)
        for i in range(E.shape[0]):
            acc += walsh_power_by_degree(E[i])
        acc /= float(E.shape[0])
        out[str(n)] = [float(x) for x in acc.tolist()]
    return out


def quantum_summary(diag: dict) -> dict:
    lam = np.asarray(diag["lambda_c"], dtype=np.float64)
    peak = np.asarray(diag["peak"], dtype=np.float64)
    return {
        "B": int(peak.shape[0]),
        "n_cav": int(lam.shape[1]),
        "lambda_med": float(np.median(lam)),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "frac_lambda_gt_080": float(np.mean(lam > 0.80)),
        "peak_med": float(np.median(peak)),
        "frac_peak_ge_05": float(np.mean(peak >= 0.5)),
        "max_residual": float(np.max(diag["residual"])),
    }


def assemble(diags: dict, jobs: list[tuple[str, int, int]]) -> dict:
    rows = []
    for method, L, seed in jobs:
        row = json.loads(seed_path(L, method, seed).read_text())
        rows.append(row)
    gs = {str(n): quantum_summary(diags[n]) for n in (6, 9, 12)}
    depths = []
    for L in DEPTHS:
        at = [r for r in rows if int(r["L"]) == int(L)]
        by = {m: [r for r in at if r["method"] == m] for m in METHODS}
        if any(len(by[m]) != len(SEEDS) for m in METHODS):
            raise RuntimeError(f"L={L} seed coverage { {m: len(by[m]) for m in METHODS} }")
        blocks = {m: _median_block(by[m], "12") for m in METHODS}
        med_f = blocks["fourier"]["med"]
        if med_f <= 0:
            raise RuntimeError(f"Fourier Medε {med_f} is not positive")
        arms = {}
        for m in ("B2", "B4", "full"):
            med = blocks[m]["med"]
            r = med / med_f
            win, total = _paired(by[m], by["fourier"], "12")
            seed_rs = [
                float(rm["eval"]["12"]["med_eps"]) / float(rf["eval"]["12"]["med_eps"])
                for rm, rf in zip(
                    sorted(by[m], key=lambda z: int(z["seed"])),
                    sorted(by["fourier"], key=lambda z: int(z["seed"])),
                )
            ]
            arms[m] = {
                "med_eps": med,
                "r_fourier": float(r),
                "call": r_call(r),
                "paired_below_fourier": win,
                "paired_n": total,
                "seeds_r_le_085": int(sum(x <= BAR_R + 1e-12 for x in seed_rs)),
                "per_seed": [
                    {
                        "seed": int(rm["seed"]),
                        "med_eps": float(rm["eval"]["12"]["med_eps"]),
                        "fourier": float(rf["eval"]["12"]["med_eps"]),
                        "r": float(rm["eval"]["12"]["med_eps"] / rf["eval"]["12"]["med_eps"]),
                        "best_round": int(rm["best_round"]),
                        "wall_s": float(rm["wall_s"]),
                        "n_params": int(rm["n_params"]),
                        "phase_diag": rm.get("phase_diag", {}),
                        "phys_rms": physical_shell_gap(m, rm["angles"]),
                    }
                    for rm, rf in zip(
                        sorted(by[m], key=lambda z: int(z["seed"])),
                        sorted(by["fourier"], key=lambda z: int(z["seed"])),
                    )
                ],
            }
        compress = arms["B2"]["med_eps"] / arms["full"]["med_eps"]
        compress_b4 = arms["B2"]["med_eps"] / arms["B4"]["med_eps"]
        hold = {}
        for n in ("6", "9"):
            hold[n] = {}
            for m in METHODS:
                hb = _median_block(by[m], n)
                hold[n][m] = {
                    "med_eps": hb["med"],
                    "r_fourier": float(hb["med"] / _median_block(by["fourier"], n)["med"]),
                }
        b2_vs_full, b2_vs_full_n = _paired(by["B2"], by["full"], "12")
        depths.append(
            {
                "L": int(L),
                "fourier_med_eps": med_f,
                "arms": arms,
                "compress_B2_over_full": float(compress),
                "compress_B2_over_B4": float(compress_b4),
                "paired_B2_below_full": b2_vs_full,
                "paired_B2_full_n": b2_vs_full_n,
                "holdout": hold,
                "lambda_gs": {n: gs[n]["lambda_med"] for n in ("6", "9", "12")},
            }
        )
    gate = next(d for d in depths if d["L"] == 2)
    r_full = gate["arms"]["full"]["r_fourier"]
    r_b2 = gate["arms"]["B2"]["r_fourier"]
    r_b4 = gate["arms"]["B4"]["r_fourier"]
    compress = gate["compress_B2_over_full"]
    full_win = r_full <= BAR_R + 1e-12
    compress_ok = compress <= COMPRESS_MAX + 1e-12
    b2_soft = BAR_R < r_b2 < BAR_R_SOFT
    if full_win and compress_ok and r_b2 <= BAR_R + 1e-12:
        verdict = "PASS"
    else:
        verdict = "needs_edit"
    reasons = []
    if not full_win:
        reasons.append(f"full R_Fourier={r_full:.4f} is not ≤ {BAR_R}")
    if not compress_ok:
        reasons.append(
            f"B=2 Medε is {compress:.3f}× full, above the {COMPRESS_MAX} compression bar"
        )
    if b2_soft:
        reasons.append(f"B=2 R_Fourier={r_b2:.4f} is soft (in ({BAR_R}, {BAR_R_SOFT}))")
    elif r_b2 > BAR_R_SOFT:
        reasons.append(f"B=2 R_Fourier={r_b2:.4f} is not a win against Fourier")
    if verdict == "PASS":
        reasons.append(
            f"full R_Fourier={r_full:.4f}≤{BAR_R} and B=2 Medε is {compress:.3f}× full (≤{COMPRESS_MAX}), with B=2 itself at R={r_b2:.4f}"
        )
    return {
        "protocol": PROTOCOL,
        "point": "NEW_SAFE",
        "lambda": LAM,
        "mu": MU,
        "verdict": verdict,
        "reasons": reasons,
        "pass_rule": (
            "L=2 frozen n=12: full R_Fourier≤0.85 AND Medε(B=2)≤1.3×Medε(full) "
            "AND B=2 itself is not soft (R_Fourier≤0.85). L=4 is reported and is not the gate. "
            "needs_edit otherwise."
        ),
        "L2_R_full": r_full,
        "L2_R_B4": r_b4,
        "L2_R_B2": r_b2,
        "L2_compress_B2_over_full": compress,
        "lambda_n12": gs["12"]["lambda_med"],
        "quote_L2": QUOTE_L2,
        "b2_definition": {
            "span": "popcount phases in span{k, k^2}, k=0..3",
            "basis_rows": B2_BASIS.tolist(),
            "init_coeff": B2_INIT.tolist(),
            "paper_phi": _PAPER_PHI.tolist(),
            "recon_error": float(np.max(np.abs(B2_INIT @ B2_BASIS - _PAPER_PHI))),
            "note": (
                "Orthonormal basis of the codebase polynomial φ=c1*k+c2*k^2. "
                "Same unitaries as symmetric Walsh degrees ≤2 with the global phase fixed by φ_0. "
                "B=4 is free φ_0..φ_3 (locked shell). full is eight Fock phases."
            ),
        },
        "gs": gs,
        "depths": depths,
        "rounds": ROUNDS,
        "lbfgs_iter": int(S.LBFGS_ITER),
        "lbfgs_lr": float(S.LBFGS_LR),
        "batch": BATCH,
        "freeze_b": FREEZE_B,
        "train": [TRAIN_LO, TRAIN_HI],
        "hold": [HOLD_LO, HOLD_HI],
        "n21": "not loaded",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "walsh_power_mean": _mean_walsh_power(diags),
    }


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def render(summary: dict) -> str:
    g2 = next(d for d in summary["depths"] if d["L"] == 2)
    lam12 = summary["lambda_n12"]
    lines = [
        "# NEW_SAFE Walsh-shell coefficient ablation",
        "",
        f"**Verdict: {summary['verdict']}.** "
        f"L=2 frozen n=12 full R_Fourier={_fmt(summary['L2_R_full'])}, "
        f"B=4 R_Fourier={_fmt(summary['L2_R_B4'])}, "
        f"B=2 R_Fourier={_fmt(summary['L2_R_B2'])}, "
        f"Medε(B=2)/Medε(full)={_fmt(summary['L2_compress_B2_over_full'], 3)}, "
        f"median Λ(GS)={_fmt(lam12, 3)}.",
        "",
        "Point **NEW_SAFE**: λ=0.75, μ=0. The quote from PR #50 / #51 / #52 is L=2 R≈0.62 "
        f"(#50 Medε SNAP/Fourier {_fmt(QUOTE_L2['med_snap'])}/{_fmt(QUOTE_L2['med_fourier'])}, "
        f"R={_fmt(QUOTE_L2['r'])}, Λ={_fmt(QUOTE_L2['lambda'])}). "
        "This file retrains that cell and the coefficient sweep around it. n=21 was not loaded.",
        "",
        "## What B is",
        "",
        "The cost pulse is the locked Walsh grading in every SNAP arm: four weights on degrees 1..4 of Cn. "
        "On these 4-SAT packs the Walsh power above degree 4 is numerical noise, so four degree weights already are the full cost spectrum. "
        "B counts free SNAP phase coefficients of the shared cavity mixer.",
        "",
        "| arm | phase parameters per layer | what it spans |",
        "|:----|:---------------------------|:--------------|",
        "| B=2 | 2 | φ_k = c1 k + c2 k² on popcount k=0..3, in orthonormal coordinates. Walsh degrees ≤2 of a symmetric phase, global phase fixed because φ_0 is not free. Paper init φ_k=−βk sits in this span exactly. |",
        "| B=4 | 4 | free φ_0..φ_3. The locked Hamming shell (8 parameters per layer once the four γ weights are counted, at L=1). |",
        "| full | 8 | one phase per Fock state of the 3-bit cavity. The Hamming shell is the popcount-tied subspace. |",
        "",
        "Fourier is the matched baseline: two angles per layer, same batches, same L-BFGS budget. "
        "R_Fourier = Medε_arm / Medε_Fourier. Medε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. "
        "Λ is the median of ⟨(Σ_{i∈c} X_i / 3)²⟩ on the true ground state. It does not depend on B or L.",
        "",
        "## Gate",
        "",
        "- Train n∈{6,9}, indices [0:200). Checkpoint on the n=9 holdout [200:250).",
        "- Freeze n=12, the first 128 pack instances. L=2 is the gate. L=4 is a soft check and does not change the verdict.",
        "- **PASS** when, at L=2 on frozen n=12, full has R_Fourier ≤ 0.85, Medε(B=2) ≤ 1.3 × Medε(full), and B=2 itself has R_Fourier ≤ 0.85.",
        "- **needs_edit** otherwise. B=2 is called soft when its R_Fourier sits in (0.85, 0.98). A soft B=2 is needs_edit even if the 1.3× ratio holds.",
        "",
        "Reasons: " + " ".join(summary["reasons"]),
        "",
        "## Protocol",
        "",
        "Packs: `beat_qaoa_12h/train_packs`. Seeds {0,1,2,3,4} at both depths. "
        "One batch stream per seed (NumPy Generator(seed) and torch.manual_seed(seed)), shared by every arm and both depths. "
        f"L-BFGS matches `Nov_3rd.settings`: {ROUNDS} rounds × max_iter {int(S.LBFGS_ITER)}, "
        f"lr {float(S.LBFGS_LR)}, history 16, strong Wolfe, grad clip 5, batch {BATCH}. "
        "New optimizer each round. Loss = mean ε/n on a 32+32 batch. "
        "Each seed restores the round with the best n=9-holdout Medε. "
        "Paper init β=0.01, γ=γ_w=−0.01, φ_k=−βk, and the full arm starts on that same shell. "
        f"Ground states: dense eigh at n=6 and n=9, ARPACK at n=12. torch {summary['torch']}, numpy {summary['numpy']}. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.",
        "",
        f"B=2 basis reconstruction of the paper shell: max |φ_recon − φ_paper| = {summary['b2_definition']['recon_error']:.3e}.",
        "",
        "Mean Walsh power of Cn's unsat table by degree, on the diagonalized packs. Degree 0 is a global phase of the cost and is dropped by the grading. Power above degree 4 is the part four γ weights cannot see.",
        "",
    ]
    lines.append("| n | " + " | ".join(f"d={d}" for d in range(7)) + " |")
    lines.append("|--:|" + "|".join("---:" for _ in range(7)) + "|")
    for n in ("6", "9", "12"):
        p = summary["walsh_power_mean"][n]
        # Pad/trim to degrees 0..6 for the table. Higher degrees are summed into a note.
        cells = []
        for d in range(7):
            cells.append(_fmt(p[d], 4) if d < len(p) else "—")
        lines.append("| " + n + " | " + " | ".join(cells) + " |")
    hi5 = max(float(sum(summary["walsh_power_mean"][n][5:])) for n in ("6", "9", "12"))
    lines += [
        "",
        f"Degree ≥5 power, summed and then taking the worst of these three packs, is {hi5:.3e}. "
        "Four γ weights already see the whole cost spectrum on 4-SAT, which is why B does not truncate γ.",
        "",
        "## Frozen n=12",
        "",
        "| L | arm | Medε | Medε Fourier | R_Fourier | Medε / Medε(full) | median Λ (GS) | paired < Fourier | seeds R≤0.85 | call |",
        "|--:|:----|-----:|-------------:|----------:|------------------:|--------------:|-----------------:|-------------:|:-----|",
    ]
    for d in summary["depths"]:
        for m in ("B2", "B4", "full"):
            a = d["arms"][m]
            ratio_full = a["med_eps"] / d["arms"]["full"]["med_eps"]
            lines.append(
                f"| {d['L']} | {m} | {_fmt(a['med_eps'])} | {_fmt(d['fourier_med_eps'])} | "
                f"{_fmt(a['r_fourier'])} | {_fmt(ratio_full, 3)} | {_fmt(d['lambda_gs']['12'])} | "
                f"{a['paired_below_fourier']}/{a['paired_n']} | {a['seeds_r_le_085']}/5 | {a['call']} |"
            )
    lines += [
        "",
        "B=2 against the full arm on paired frozen instances "
        f"(L=2): {g2['paired_B2_below_full']}/{g2['paired_B2_full_n']} have a lower excess than full. "
        f"Medε(B=2)/Medε(B=4) at L=2 is {_fmt(g2['compress_B2_over_B4'], 3)}. "
        "B=4 is the locked shell, so that ratio is the compression claim against the published ansatz rather than against the 8-phase arm.",
        "",
        "## Quote check (B=4 and Fourier, L=2, n=12)",
        "",
        "| source | Medε shell | Medε Fourier | R | Λ |",
        "|:-------|----------:|-------------:|--:|--:|",
        f"| PR #50 quote | {_fmt(QUOTE_L2['med_snap'])} | {_fmt(QUOTE_L2['med_fourier'])} | {_fmt(QUOTE_L2['r'])} | {_fmt(QUOTE_L2['lambda'])} |",
        f"| this run, B=4 | {_fmt(g2['arms']['B4']['med_eps'])} | {_fmt(g2['fourier_med_eps'])} | {_fmt(g2['arms']['B4']['r_fourier'])} | {_fmt(lam12)} |",
        "",
    ]
    d_med = g2["arms"]["B4"]["med_eps"] - QUOTE_L2["med_snap"]
    d_f = g2["fourier_med_eps"] - QUOTE_L2["med_fourier"]
    d_r = g2["arms"]["B4"]["r_fourier"] - QUOTE_L2["r"]
    d_l = lam12 - QUOTE_L2["lambda"]
    lines.append(
        f"Deltas (this run minus #50): Medε B=4 {d_med:+.4f}, Fourier {d_f:+.4f}, R {d_r:+.4f}, Λ {d_l:+.4f}. "
        "R in this file always uses this run's Fourier, trained on the same batches as the shells."
    )
    lines += [
        "",
        "## Holdout (checkpoint distribution, B=50)",
        "",
        "Λ here is the ground state on the diagonalized pack (n=6 and n=9, B=250), not a property of the circuit.",
        "",
        "| L | n | arm | Medε | R_Fourier | median Λ (GS) |",
        "|--:|--:|:----|-----:|----------:|--------------:|",
    ]
    for d in summary["depths"]:
        for n in ("6", "9"):
            for m in ("B2", "B4", "full"):
                h = d["holdout"][n][m]
                lines.append(
                    f"| {d['L']} | {n} | {m} | {_fmt(h['med_eps'])} | {_fmt(h['r_fourier'])} | {_fmt(d['lambda_gs'][n])} |"
                )
    lines += [
        "",
        "## Ground state",
        "",
        "| n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | median Z-peak | frac peak≥0.5 | max residual |",
        "|--:|--:|--------:|------:|------:|------------:|--------------:|--------------:|-------------:|",
    ]
    for n in ("6", "9", "12"):
        g = summary["gs"][n]
        lines.append(
            f"| {n} | {g['B']} | {_fmt(g['lambda_med'])} | {_fmt(g['lambda_min'])} | {_fmt(g['lambda_max'])} | "
            f"{_fmt(g['frac_lambda_gt_080'], 3)} | {_fmt(g['peak_med'])} | {_fmt(g['frac_peak_ge_05'], 3)} | {g['max_residual']:.2e} |"
        )
    lines += ["", "## Seeds at frozen n=12", ""]
    for d in summary["depths"]:
        lines.append(f"### L={d['L']}")
        lines.append("")
        lines.append("| arm | seed | Medε | Medε Fourier | R | best round | wall s | shell std | phys rms |")
        lines.append("|:----|-----:|-----:|-------------:|--:|----------:|-------:|----------:|---------:|")
        for m in ("B2", "B4", "full"):
            for s in d["arms"][m]["per_seed"]:
                pd = s.get("phase_diag") or {}
                std = pd.get("max_within_popcount_std", 0.0)
                rms = float(s.get("phys_rms", float("nan")))
                lines.append(
                    f"| {m} | {s['seed']} | {_fmt(s['med_eps'])} | {_fmt(s['fourier'])} | {_fmt(s['r'])} | "
                    f"{s['best_round']} | {s['wall_s']:.1f} | {std:.2e} | {rms:.4f} |"
                )
        lines.append("")
    # L=4 narrative, not a gate
    g4 = next(d for d in summary["depths"] if d["L"] == 4)
    def _mean_key(block, arm, key):
        vals = []
        for s in block["arms"][arm]["per_seed"]:
            if key == "phys_rms":
                vals.append(float(s.get("phys_rms", float("nan"))))
                continue
            pd = s.get("phase_diag") or {}
            if key in pd:
                vals.append(float(pd[key]))
        return float(np.mean(vals)) if vals else float("nan")

    lines += [
        "## Reading",
        "",
        f"L=2 full is a {g2['arms']['full']['call']} against Fourier "
        f"(R={_fmt(g2['arms']['full']['r_fourier'])}, Medε {_fmt(g2['arms']['full']['med_eps'])} vs {_fmt(g2['fourier_med_eps'])}). "
        f"L=2 B=4, the locked shell, is a {g2['arms']['B4']['call']} "
        f"(R={_fmt(g2['arms']['B4']['r_fourier'])}, Medε {_fmt(g2['arms']['B4']['med_eps'])}). "
        f"L=2 B=2 is a {g2['arms']['B2']['call']} "
        f"(R={_fmt(g2['arms']['B2']['r_fourier'])}, Medε {_fmt(g2['arms']['B2']['med_eps'])}), "
        f"{_fmt(g2['compress_B2_over_full'], 3)}× the full arm and {_fmt(g2['compress_B2_over_B4'], 3)}× the locked shell. "
        f"On paired instances B=2 is lower than full on {g2['paired_B2_below_full']}/{g2['paired_B2_full_n']}, a tie rather than a sweep.",
        "",
        "The three L=2 arms tie, and B=4 matches the PR #50 quote at four decimals on both Medε columns, on R, and on Λ. "
        "A constant added to every popcount phase is a global phase of the cavity unitary, so it is not a real coefficient. "
        f"After removing that constant, B=4's phases sit {_mean_key(g2, 'B4', 'phys_rms'):.4f} RMS from span{{1, k, k²}} "
        f"(full: {_mean_key(g2, 'full', 'phys_rms'):.4f}). "
        "The third shell direction is barely used, which is why two coefficients match four. "
        f"full's largest within-popcount phase std, averaged over seeds, is {_mean_key(g2, 'full', 'max_within_popcount_std'):.4f}, "
        "so the eight Fock phases stay close to a shell.",
        "",
        f"L=4 is the soft check, not the gate. full R={_fmt(g4['arms']['full']['r_fourier'])} ({g4['arms']['full']['call']}), "
        f"B=4 R={_fmt(g4['arms']['B4']['r_fourier'])} ({g4['arms']['B4']['call']}), "
        f"B=2 R={_fmt(g4['arms']['B2']['r_fourier'])} ({g4['arms']['B2']['call']}), "
        f"Medε(B=2)/Medε(full)={_fmt(g4['compress_B2_over_full'], 3)}, "
        f"Medε(B=2)/Medε(B=4)={_fmt(g4['compress_B2_over_B4'], 3)}. "
        "B=4 at this depth is the published soft result (Medε 0.2755 / Fourier 0.3172 / R 0.8687). "
        "B=2 and full are wins there. Two coefficients are better than the locked four-phase shell under the same budget, which is the opposite of a compression failure. "
        f"The physical gap of B=4 off span{{1, k, k²}} grows to {_mean_key(g4, 'B4', 'phys_rms'):.4f} RMS "
        f"(full {_mean_key(g4, 'full', 'phys_rms'):.4f}), so L=4 does use a bit of the extra shell direction, and that use does not improve Medε. "
        "L=4 stays outside the locked NEW_SAFE sentence.",
        "",
        f"Median Λ at frozen n=12 is {_fmt(lam12)}. The expectation from the seal is about 0.75, and the pass line used there was 0.80. "
        f"n=6 / n=9 medians are {_fmt(summary['gs']['6']['lambda_med'])} / {_fmt(summary['gs']['9']['lambda_med'])}.",
        "",
        "Shell std is the max, over layers and popcount groups, of the std of the eight Fock phases inside one popcount. "
        "It is zero for B=2 and B=4 by construction. "
        "Phys rms is the layer-averaged RMS of the popcount phases outside span{1, k, k²}. "
        "That span is the B=2 family plus the unphysical global phase. "
        "A raw distance to span{k, k²} alone is not reported: it is dominated by the global phase.",
        "",
        "## Non-claims",
        "",
        "This does not retune μ, does not add XX, and does not train L≥8. "
        "It does not say that two coefficients beat four on every instance. "
        "The 1.3× bar is on Medε, which is a median of medians. "
        "n=15 and n=18 were not resealed. n=21 was not loaded.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_self_check(E6: np.ndarray) -> None:
    """Identity checks before any training budget is spent."""
    self_check(E6[:1])
    recon = B2_INIT @ B2_BASIS
    err = float(np.max(np.abs(recon - _PAPER_PHI)))
    if err > 1e-12:
        raise AssertionError(f"B=2 basis does not contain the paper shell, err {err}")
    # Orthonormal rows.
    gram = B2_BASIS @ B2_BASIS.T
    if np.max(np.abs(gram - np.eye(2))) > 1e-12:
        raise AssertionError(f"B=2 basis not orthonormal: {gram}")
    pwr = walsh_power_by_degree(E6[0])
    hi = float(pwr[5:].sum()) if pwr.size > 5 else 0.0
    if hi > 1e-8:
        raise AssertionError(f"n6 instance0 Walsh degree>4 power {hi}")
    log(f"Walsh n6[0] power by degree {np.array2string(pwr, precision=4)}")

    torch.manual_seed(0)
    n, B, L = 6, 4, 2
    E = torch.tensor(E6[:B], dtype=torch.float64)
    Cn = cn_from_unsat(E, n)
    # B=4 path matches snap_state on random phases.
    gamma = torch.randn(L, 4, dtype=torch.float64)
    phi = torch.randn(L, 4, dtype=torch.float64)
    psi_a = snap_state(Cn, gamma, phi)
    comps = walsh_degree_components(Cn)
    psi_b = snap_state_from_components(comps, gamma, phi)
    tv = (psi_a - psi_b).abs().mean().item()
    if tv > 1e-8:
        raise AssertionError(f"B=4 component path TV {tv}")
    # B=2 init matches the paper shell state.
    ang4 = init_method("B4", L)
    ang2 = init_method("B2", L)
    angf = init_method("full", L)
    psi4 = snap_state(Cn, ang4["gamma_w"], ang4["phi"])
    psi2 = snap_state(Cn, ang2["gamma_w"], phi_from_b2(ang2["c"]))
    psif = snap_fock_from_components(comps, angf["gamma_w"], angf["theta"])
    # Global-phase-free: probabilities, and also the state itself for B=2 (exact span).
    if (psi4 - psi2).abs().max().item() > 1e-8:
        raise AssertionError(f"B=2 init state err {(psi4 - psi2).abs().max().item()}")
    if (psi4 - psif).abs().max().item() > 1e-8:
        raise AssertionError(f"full init state err {(psi4 - psif).abs().max().item()}")
    # A shell embedded in full matches B=4.
    theta = theta_from_phi(phi)
    psi_shell = snap_fock_from_components(comps, gamma, theta)
    psi_ref = snap_state(Cn, gamma, phi)
    if (psi_shell - psi_ref).abs().max().item() > 1e-8:
        raise AssertionError("full embedding of a shell disagrees with snap_state")
    # Energies at init agree, μ=0.
    e0_dummy = torch.zeros(B, dtype=torch.float64)
    for name, psi in (("B4", psi4), ("B2", psi2), ("full", psif)):
        h = expectation_H(psi, E, n, lambda_x=LAMBDA_X, mu_xx=MU_XX)
        log(f"init energy {name} mean {float(h.mean()):.6f}")
    h4 = expectation_H(psi4, E, n, lambda_x=LAMBDA_X, mu_xx=0.0)
    h2 = expectation_H(psi2, E, n, lambda_x=LAMBDA_X, mu_xx=0.0)
    if (h4 - h2).abs().max().item() > 1e-8:
        raise AssertionError("init energies diverged")
    # One dense ground state is finite and below the mean diagonal.
    gs = ground_bundle(E6[0], 6, lambda_x=LAMBDA_X, mu_xx=MU_XX)
    if not np.isfinite(gs["E0"]) or gs["E0"] > float(E6[0].mean()):
        raise AssertionError(f"ground state looks wrong: {gs['E0']}")
    log(f"self-check ok. n6[0] E0={gs['E0']:.4f} peak={gs['peak']:.4f} degree>4 power={hi:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--methods", type=str, default="fourier,B2,B4,full")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()
    depths = tuple(int(x) for x in args.depths.split(",") if x)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    methods = tuple(x for x in args.methods.split(",") if x)
    for d in depths:
        if d not in DEPTHS:
            raise SystemExit(f"depth {d} is outside the registered sweep {DEPTHS}")
    for s in seeds:
        if s not in SEEDS:
            raise SystemExit(f"seed {s} is outside {SEEDS}")
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"method {m} is outside {METHODS}")

    torch.set_num_threads(1)
    E6_all, s6_all = load_pack(6)
    E9_all, s9_all = load_pack(9)
    E12_all, s12_all = load_pack(12)
    if not args.skip_self_check:
        run_self_check(E6_all)

    # Diagonalize the slices we actually score: full n=6 and n=9 packs (train+hold
    # is the first 250, which is the whole file) and the n=12 freeze prefix.
    # n=21 is not opened.
    slices = {
        6: (E6_all, s6_all),
        9: (E9_all, s9_all),
        12: (E12_all[:FREEZE_B], s12_all[:FREEZE_B]),
    }
    diags = {}
    for n, (E, seeds_n) in slices.items():
        rec = ensure_diag(n, E, seeds_n, args.workers)
        rec["E"] = E
        diags[n] = rec
        log(
            f"Λ n={n} median={float(np.median(rec['lambda_c'])):.4f} "
            f"peak_med={float(np.median(rec['peak'])):.4f}"
        )

    packs = build_packs(diags)
    jobs = [(m, L, s) for L in depths for m in methods for s in seeds]
    pending = [j for j in jobs if not seed_is_done(seed_path(j[1], j[0], j[2]), j[0], j[2], j[1])]
    log(f"jobs {len(jobs)} pending {len(pending)} workers {args.workers}")
    if pending:
        payloads = [(m, L, s, packs) for m, L, s in pending]
        # A small pool. Each job is one L-BFGS chain and not internally parallel.
        with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_worker_init) as pool:
            futs = {pool.submit(_train_job, p): p[:3] for p in payloads}
            for fut in as_completed(futs):
                tag = futs[fut]
                try:
                    fut.result()
                except Exception:
                    log(f"JOB FAILED {tag}")
                    traceback.print_exc()
                    raise

    # Assemble only when the full registered grid is on disk. A partial
    # --seeds/--methods invocation still leaves its JSON behind.
    full_jobs = [(m, L, s) for L in DEPTHS for m in METHODS for s in SEEDS]
    if all(seed_is_done(seed_path(L, m, s), m, s, L) for m, L, s in full_jobs):
        summary = assemble(diags, full_jobs)
        write_json(RUN / "summary.json", summary)
        text = render(summary)
        (RUN / "RESULTS.md").write_text(text)
        log(f"verdict {summary['verdict']} — {summary['reasons']}")
    else:
        log("partial grid; RESULTS.md not rewritten")


if __name__ == "__main__":
    main()
