#!/usr/bin/env python3
"""Fidelity transfer for the power-law TFIM at λ ∈ {0.75, 1.00}, μ = 0.

H = Σ_{i<j} J_ij / |i-j|^{1.5} Z_i Z_j − λ Σ_i X_i

Same open cavity chain as Nov_3rd.powerlaw_tfim (PR #42). J_ij ∈ {±1},
α = 1.5, qubit 0 is the MSB, cavity c owns sites [3c, 3c+3).

The objective is the state fidelity to the exact ground state, not the
energy ratio against QAOA. F = |⟨ψ(θ)|GS⟩|².

Ansatzes, matched budget:
  b2       Walsh SNAP. One cost weight γ_2 per layer (H_ZZ is pure Walsh
           degree 2) and a two-coefficient Hamming shell
           φ_k = c · basis{k, k²} on popcount k = 0..3.
  fourier  Standard Fourier QAOA on Cn = H_ZZ / n. Reference column.

Train jointly on n ∈ {6, 9}. Pools: 64 train (indices 0..63), 32 hold
(64..95). Freeze the decision depth on fresh n=12 and n=15 draws
(indices 0..), no retrain.

L* is the smallest L in {1, 2, 4, 8, 12, 16} whose median-over-seeds
train and hold fidelities are all ≥ 0.95 on both n=6 and n=9.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S  # noqa: E402
from Nov_3rd.hva_safe import (  # noqa: E402
    INIT_BETA,
    INIT_GAMMA,
    state_fidelity,
    walsh_degree_diag,
)
from Nov_3rd.powerlaw_tfim import (  # noqa: E402
    ALPHA,
    best_product,
    cost_diag,
    couplings,
    cross_cavity_weight,
    expectation,
    init_method,
    instance_seed,
    make_instance,
    prepare,
    prepare_snap,
    quantum_summary,
    self_check,
    shell_nonlinearity,
    snap_angles_from_gamma2,
    zz_diag_from_K,
)

RUN = S.OUT / "quantum_h_explore" / "powerlaw_fidelity_transfer" / "lam_075_100"
CACHE = RUN / "cache"
PROTOCOL = "powerlaw_fidelity_lam075_100_v1"
MU = 0.0
LAMS = (0.75, 1.0)
DEPTHS = (1, 2, 4, 8, 12, 16)
SEEDS = (0, 1, 2)
METHODS = ("b2", "fourier")
TRAIN_B = 64
HOLD_B = 32
POOL_B = TRAIN_B + HOLD_B
FREEZE12_B = 64
FREEZE15_B = 64
ROUNDS = 40
BATCH = 32
LBFGS_ITER = int(S.LBFGS_ITER)
LBFGS_LR = float(S.LBFGS_LR)
F_BAR = 0.95
PASS_N12 = 0.70
PASS_DROP = 0.25
# Published PR #42 breadth holdout (indices 64..95) at λ=1, μ=0.
REF_L1 = {
    6: {"peak_med": 0.16456590775753027, "gap_med": 0.04737419807123598},
    9: {"peak_med": 0.09169306843304864, "gap_med": 0.036229311401873285},
}

# Orthonormal rows spanning {k, k²} on popcount k=0..3. φ = c @ B2_BASIS.
_K4 = np.arange(4, dtype=np.float64)
_RAW = np.stack([_K4, _K4 ** 2], axis=1)
_Q, _R = np.linalg.qr(_RAW, mode="reduced")
for _j in range(2):
    if float(_Q[:, _j] @ _RAW[:, _j]) < 0.0:
        _Q[:, _j] *= -1.0
B2_BASIS = np.ascontiguousarray(_Q.T)  # (2, 4)
B2_INIT = np.ascontiguousarray((-INIT_BETA * _K4) @ B2_BASIS.T)  # (2,)

LOG_PATH: Path | None = None


def log(*parts) -> None:
    msg = time.strftime("%H:%M:%S") + " " + " ".join(str(p) for p in parts)
    print(msg, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a") as handle:
            handle.write(msg + "\n")


def lam_tag(lam: float) -> str:
    return f"{float(lam):.2f}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def flips_for(n: int) -> list[np.ndarray]:
    idx = np.arange(1 << int(n), dtype=np.int64)
    return [idx ^ (1 << (int(n) - 1 - q)) for q in range(int(n))]


def mean_x(psi: np.ndarray, flips: list[np.ndarray], n: int) -> float:
    acc = 0.0
    for f in flips:
        acc += float(np.dot(psi, psi[f]))
    return float(acc / float(n))


def ground_bundle(diag: np.ndarray, n: int, lam: float, flips: list[np.ndarray] | None = None) -> dict:
    """Exact lowest eigenpair of H_ZZ − λ Σ X. ψ is real and normalized."""
    from Nov_3rd.hva_safe import dense_H
    from scipy.sparse.linalg import LinearOperator, eigsh

    diag = np.asarray(diag, dtype=np.float64).reshape(-1)
    n = int(n)
    N = 1 << n
    if diag.size != N:
        raise ValueError(f"diag {diag.size} != 2^{n}")
    lx = -float(lam)
    flips = flips if flips is not None else flips_for(n)

    def mv(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        out = diag * vv
        for f in flips:
            out = out + lx * vv[f]
        return out

    if n <= 9:
        evals, evecs = np.linalg.eigh(dense_H(diag, n, lambda_x=lx, mu_xx=0.0))
        e0 = float(evals[0])
        e1 = float(evals[1])
        psi = np.real(np.asarray(evecs[:, 0], dtype=np.float64))
    else:
        op = LinearOperator((N, N), matvec=mv, dtype=np.float64)
        starts = [np.full(N, 1.0 / math.sqrt(N))]
        bit = int(np.argmin(diag))
        classical = np.zeros(N)
        classical[bit] = 1.0
        starts.append(classical)
        best = None
        last_resid = None
        for attempt, v0 in enumerate(starts):
            evals, evecs = eigsh(
                op,
                k=2,
                which="SA",
                tol=1e-10,
                maxiter=max(4000, 4 * N),
                ncv=min(N - 1, 48),
                v0=np.asarray(v0, dtype=np.float64),
            )
            order = np.argsort(np.real(evals))
            e0 = float(np.real(evals[order[0]]))
            e1 = float(np.real(evals[order[1]]))
            psi = np.real(np.asarray(evecs[:, order[0]], dtype=np.float64))
            psi = psi / float(np.linalg.norm(psi))
            resid = float(np.linalg.norm(mv(psi) - e0 * psi))
            last_resid = resid
            ok = resid <= 1e-6 * max(1.0, abs(e0)) and e0 <= float(diag.min()) + 1e-6
            if ok and (best is None or e0 < best["E0"]):
                best = {"E0": e0, "E1": e1, "psi": psi, "resid": resid}
            if best is not None and attempt == 0:
                break
        if best is None:
            raise RuntimeError(f"eigsh failed at n={n} λ={lam} resid={last_resid}")
        e0, e1, psi = best["E0"], best["E1"], best["psi"]
    psi = psi / float(np.linalg.norm(psi))
    if float(psi.sum()) < 0.0:
        psi = -psi
    if e0 > float(diag.min()) + 1e-6:
        raise RuntimeError(f"E0 {e0} above a diagonal entry {float(diag.min())}")
    return {
        "E0": float(e0),
        "E1": float(e1),
        "psi": psi,
        "peak": float(np.max(psi ** 2)),
        "x": mean_x(psi, flips, n),
    }


def gs_path(n: int, lam: float, count: int) -> Path:
    return CACHE / f"gs_n{n}_l{lam_tag(lam)}_B{count}_mu0.npz"


def build_gs_pool(n: int, count: int, lam: float, workers: int) -> dict:
    path = gs_path(n, lam, count)
    if path.exists():
        z = np.load(path)
        same = (
            int(z["n"]) == int(n)
            and int(z["count"]) == int(count)
            and abs(float(z["lam"]) - float(lam)) < 1e-15
            and abs(float(z["mu"]) - MU) < 1e-15
            and abs(float(z["alpha"]) - ALPHA) < 1e-15
        )
        if same:
            log(f"gs cache hit n{n} λ={lam_tag(lam)} B={count}")
            return {k: np.asarray(z[k]) for k in ("diag", "psi", "e0", "e1", "peak", "x", "cross", "seeds")}
    log(f"gs build n{n} λ={lam_tag(lam)} B={count}")
    t0 = time.time()
    diags = np.empty((count, 1 << int(n)), dtype=np.float64)
    seeds = np.empty(count, dtype=np.int64)
    cross = np.empty(count, dtype=np.float64)
    for i in range(count):
        _J, K = couplings(n, i)
        diags[i] = zz_diag_from_K(K)
        seeds[i] = instance_seed(n, i)
        cross[i] = cross_cavity_weight(K)
    flips = flips_for(n)
    e0 = np.empty(count)
    e1 = np.empty(count)
    peak = np.empty(count)
    xmean = np.empty(count)
    psi = np.empty_like(diags)
    if int(n) <= 9 or int(workers) <= 1:
        for i in range(count):
            gs = ground_bundle(diags[i], n, lam, flips)
            e0[i], e1[i], peak[i], xmean[i], psi[i] = gs["E0"], gs["E1"], gs["peak"], gs["x"], gs["psi"]
            if (i + 1) == count or (i + 1) % 8 == 0:
                log(f"  n{n} λ={lam_tag(lam)} {i+1}/{count} E0={e0[i]:.4f} peak={peak[i]:.4f} ⟨X⟩={xmean[i]:.3f}")
    else:
        jobs = [(diags[i], n, lam) for i in range(count)]
        done = 0
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(_gs_job, job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futs):
                i = futs[fut]
                gs = fut.result()
                e0[i], e1[i], peak[i], xmean[i], psi[i] = gs["E0"], gs["E1"], gs["peak"], gs["x"], gs["psi"]
                done += 1
                if done == count or done % 4 == 0:
                    log(f"  n{n} λ={lam_tag(lam)} {done}/{count} E0={e0[i]:.4f} peak={peak[i]:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        diag=diags,
        psi=psi,
        e0=e0,
        e1=e1,
        peak=peak,
        x=xmean,
        cross=cross,
        seeds=seeds,
        n=int(n),
        count=int(count),
        lam=float(lam),
        mu=float(MU),
        alpha=float(ALPHA),
    )
    log(f"gs n{n} λ={lam_tag(lam)} B={count} in {time.time()-t0:.1f}s")
    return {
        "diag": diags,
        "psi": psi,
        "e0": e0,
        "e1": e1,
        "peak": peak,
        "x": xmean,
        "cross": cross,
        "seeds": seeds,
    }


def _gs_job(job: tuple) -> dict:
    diag, n, lam = job
    return ground_bundle(diag, int(n), float(lam), flips_for(int(n)))


def _gap_job(job: tuple) -> float:
    n, index, lam, seed, diag = job
    _J, K = couplings(int(n), int(index))
    return float(best_product(K, np.asarray(diag, dtype=np.float64), float(lam), MU, seed=int(seed)))


def product_gaps(n: int, lam: float, diag: np.ndarray, seeds: np.ndarray, e0: np.ndarray, workers: int) -> np.ndarray:
    count = int(diag.shape[0])
    jobs = [(n, i, lam, int(seeds[i]), diag[i]) for i in range(count)]
    e_prod = np.empty(count)
    t0 = time.time()
    if workers <= 1 or count < 4:
        for i, job in enumerate(jobs):
            e_prod[i] = _gap_job(job)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for i, val in enumerate(pool.map(_gap_job, jobs)):
                e_prod[i] = val
    gap = (e_prod - e0) / np.maximum(np.abs(e0), 1e-8)
    if float(np.min(e_prod - e0)) < -1e-5:
        raise RuntimeError(f"product below E0 at n={n} λ={lam}")
    log(f"product gaps n{n} λ={lam_tag(lam)} B={count} med={float(np.median(gap)):.4f} in {time.time()-t0:.1f}s")
    return gap


def phase_path(lam: float) -> Path:
    return RUN / f"phase_l{lam_tag(lam)}.json"


def build_phase(lam: float, pool6: dict, pool9: dict, workers: int) -> dict:
    path = phase_path(lam)
    if path.exists():
        payload = read_json(path)
        if payload.get("protocol") == PROTOCOL and payload.get("complete") is True:
            log(f"phase cache hit λ={lam_tag(lam)}")
            return payload
    blocks = {}
    for n, pool in ((6, pool6), (9, pool9)):
        gaps = product_gaps(n, lam, pool["diag"], pool["seeds"], pool["e0"], workers)
        rel = (pool["e1"] - pool["e0"]) / np.maximum(np.abs(pool["e0"]), 1e-8)
        hold = slice(TRAIN_B, POOL_B)
        train = slice(0, TRAIN_B)
        blocks[str(n)] = {
            "all": _phase_block(pool, gaps, rel, slice(0, POOL_B)),
            "train": _phase_block(pool, gaps, rel, train),
            "hold": _phase_block(pool, gaps, rel, hold),
            "cross_med": float(np.median(pool["cross"])),
        }
    payload = {
        "protocol": PROTOCOL,
        "complete": True,
        "lam": float(lam),
        "mu": MU,
        "alpha": ALPHA,
        "n": blocks,
    }
    write_json(path, payload)
    return payload


def _phase_block(pool: dict, gaps: np.ndarray, rel_gap: np.ndarray, sl: slice) -> dict:
    peaks = np.asarray(pool["peak"][sl], dtype=np.float64)
    g = np.asarray(gaps[sl], dtype=np.float64)
    x = np.asarray(pool["x"][sl], dtype=np.float64)
    e0 = np.asarray(pool["e0"][sl], dtype=np.float64)
    rg = np.asarray(rel_gap[sl], dtype=np.float64)
    q = quantum_summary(peaks, g)
    q.update(
        {
            "x_med": float(np.median(x)),
            "x_mean": float(np.mean(x)),
            "E0_med": float(np.median(e0)),
            "spec_gap_med": float(np.median(rg)),
            "spec_gap_min": float(np.min(rg)),
        }
    )
    return q


def check_published_phase(phase: dict) -> None:
    """λ=1 holdout must reproduce the PR #42 breadth cache."""
    if abs(float(phase["lam"]) - 1.0) > 1e-15:
        return
    for n, ref in REF_L1.items():
        hold = phase["n"][str(n)]["hold"]
        dp = abs(float(hold["peak_med"]) - ref["peak_med"])
        dg = abs(float(hold["gap_med"]) - ref["gap_med"])
        log(
            f"λ=1 n{n} hold vs PR#42 peak {hold['peak_med']:.6f} "
            f"(Δ{dp:.2e}) gap {hold['gap_med']:.6f} (Δ{dg:.2e})"
        )
        if dp > 1e-6:
            raise RuntimeError(f"n{n} bitstring peak disagrees with PR #42")
        if dg > 1e-4:
            raise RuntimeError(f"n{n} product gap disagrees with PR #42")


def init_b2(L: int) -> dict:
    L = int(L)
    c0 = torch.tensor(B2_INIT, dtype=torch.float64)
    return {
        "gamma2": torch.full((L,), INIT_GAMMA, dtype=torch.float64),
        "c": c0.reshape(1, 2).expand(L, 2).contiguous().clone(),
    }


def bind_b2(angles: dict) -> list[torch.nn.Parameter]:
    params = []
    for key in ("gamma2", "c"):
        p = torch.nn.Parameter(angles[key].detach().clone())
        angles[key] = p
        params.append(p)
    return params


def prepare_b2(Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    phi = angles["c"] @ angles["c"].new_tensor(B2_BASIS)
    return prepare_snap(Cn, {"gamma2": angles["gamma2"], "phi": phi})


def prepare_method(method: str, Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    if method == "b2":
        return prepare_b2(Cn, angles)
    if method == "fourier":
        return prepare("fourier", Cn, angles)
    raise ValueError(method)


def init_train(method: str, L: int) -> tuple[dict, list[torch.nn.Parameter]]:
    if method == "b2":
        angles = init_b2(L)
        return angles, bind_b2(angles)
    if method == "fourier":
        angles = init_method("fourier", L, dtype=torch.float64)
        params = []
        for key in ("betas", "gammas"):
            p = torch.nn.Parameter(angles[key].detach().clone())
            angles[key] = p
            params.append(p)
        return angles, params
    raise ValueError(method)


def fidelity_sq(psi: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    ov = torch.sum(psi.conj() * gs.to(dtype=psi.dtype), dim=-1)
    return ov.real.square() + ov.imag.square()


def batch_metrics(
    method: str,
    angles: dict,
    diag: torch.Tensor,
    gs: torch.Tensor,
    e0: torch.Tensor,
    n: int,
    lam: float,
    chunk: int,
) -> dict:
    fs: list[np.ndarray] = []
    epss: list[np.ndarray] = []
    B = int(diag.shape[0])
    with torch.no_grad():
        for s in range(0, B, int(chunk)):
            d = diag[s : s + chunk]
            g = gs[s : s + chunk]
            psi = prepare_method(method, cost_diag(d, n), angles)
            F = fidelity_sq(psi, g)
            h = expectation(psi, d, n, lam, MU)
            fs.append(F.detach().cpu().numpy())
            epss.append((h - e0[s : s + chunk]).detach().cpu().numpy())
    f = np.concatenate(fs)
    e = np.concatenate(epss)
    if float(np.max(f)) > 1.0 + 1e-6:
        raise RuntimeError(f"fidelity {float(np.max(f))} above 1")
    return {
        "B": int(f.size),
        "med_F": float(np.median(f)),
        "mean_F": float(np.mean(f)),
        "min_F": float(np.min(f)),
        "med_eps": float(np.median(e)),
        "mean_eps": float(np.mean(e)),
        "med_E0": float(np.median(e0.detach().cpu().numpy())),
        "F": [float(v) for v in f.tolist()],
        "eps": [float(v) for v in e.tolist()],
    }


def split_score(method: str, angles: dict, packs: dict) -> dict[str, float]:
    """Median fidelity of each split. Energy is left to the final eval."""
    out: dict[str, float] = {}
    with torch.no_grad():
        for name in ("train6", "train9", "hold6", "hold9"):
            n = 6 if name.endswith("6") else 9
            pack = packs[name]
            psi = prepare_method(method, cost_diag(pack["diag"], n), angles)
            med = float(np.median(fidelity_sq(psi, pack["gs"]).detach().cpu().numpy()))
            out[name] = med
    out["score"] = float(min(out.values()))
    return out


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method: str, L: int, seed: int, lam: float, rounds: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(int(seed) + 1000 * int(L))
    rng = np.random.default_rng([int(seed), int(L), 750100])
    pool = build_gs_pool(6, POOL_B, lam, workers=1)
    pool9 = build_gs_pool(9, POOL_B, lam, workers=1)
    packs = _torch_packs(lam, pool, pool9)
    angles, params = init_train(method, L)
    n6 = int(packs["train6"]["diag"].shape[0])
    n9 = int(packs["train9"]["diag"].shape[0])
    b6 = min(BATCH // 2, n6)
    b9 = min(BATCH - b6, n9)
    t0 = time.time()
    init_sc = split_score(method, angles, packs)
    best = {
        "score": init_sc["score"],
        "round": -1,
        "split": init_sc,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, **{k: float(v) for k, v in init_sc.items()}, "loss": None}]
    log(
        f"L{L} {method} s{seed} λ={lam_tag(lam)} init score={init_sc['score']:.4f} "
        f"nparams={sum(p.numel() for p in params)}"
    )
    sat = 0
    early = False
    for r in range(int(rounds)):
        opt = torch.optim.LBFGS(
            params,
            lr=LBFGS_LR,
            max_iter=LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        i6 = torch.tensor(rng.choice(n6, size=b6, replace=False), dtype=torch.long)
        i9 = torch.tensor(rng.choice(n9, size=b9, replace=False), dtype=torch.long)

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                _fid_loss(method, angles, packs["train6"], i6, 6)
                + _fid_loss(method, angles, packs["train9"], i9, 9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as exc:
            log(f"L{L} {method} s{seed} r{r} LBFGS skip: {exc}")
            _restore(angles, best["angles"])
            hist.append({"round": r, "skip": str(exc)})
            sat = 0
            continue
        if not _finite(loss_f):
            log(f"L{L} {method} s{seed} r{r} nonfinite — restore best")
            _restore(angles, best["angles"])
            hist.append({"round": r, "loss": None})
            sat = 0
            continue
        sc = split_score(method, angles, packs)
        hist.append({"round": r, "loss": loss_f, **{k: float(v) for k, v in sc.items()}})
        mark = ""
        if sc["score"] > best["score"] + 1e-12:
            best = {
                "score": sc["score"],
                "round": r,
                "split": sc,
                "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
            }
            mark = " *"
        if sc["score"] >= 0.99:
            sat += 1
        else:
            sat = 0
        if mark or r % 5 == 0 or r + 1 == int(rounds) or sat >= 3:
            log(
                f"L{L} {method} s{seed} λ={lam_tag(lam)} r{r}/{rounds} "
                f"loss={loss_f:.4f} score={sc['score']:.4f} best={best['score']:.4f}@r{best['round']}{mark}"
            )
        if sat >= 3:
            early = True
            break
    _restore(angles, best["angles"])
    ev = {}
    for name, n in (("train6", 6), ("hold6", 6), ("train9", 9), ("hold9", 9)):
        ev[name] = batch_metrics(
            method,
            angles,
            packs[name]["diag"],
            packs[name]["gs"],
            packs[name]["e0"],
            n,
            lam,
            chunk=32,
        )
    neg = min(float(np.min(ev[name]["eps"])) for name in ev)
    if neg < -1e-4:
        raise RuntimeError(f"negative excess {neg} at {method} L{L} s{seed}")
    return {
        "protocol": PROTOCOL,
        "complete": True,
        "objective": "fidelity",
        "F_def": "|<psi|GS>|^2",
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "mu": MU,
        "alpha": ALPHA,
        "early_stop": bool(early),
        "rounds_config": int(rounds),
        "rounds_ran": int(hist[-1]["round"] + 1 if hist else 0),
        "best_round": int(best["round"]),
        "best_score": float(best["score"]),
        "init_score": float(init_sc["score"]),
        "wall_s": time.time() - t0,
        "n_params": int(sum(int(v.numel()) for v in best["angles"].values())),
        "budget": _budget(rounds),
        "angles": {k: v.detach().cpu().numpy().tolist() for k, v in best["angles"].items()},
        "phase_diag": _angle_diag(method, best["angles"]),
        "eval": {k: _public_metrics(v) for k, v in ev.items()},
        "hist": hist,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _fid_loss(method: str, angles: dict, pack: dict, idx: torch.Tensor, n: int) -> torch.Tensor:
    d = pack["diag"].index_select(0, idx)
    g = pack["gs"].index_select(0, idx)
    psi = prepare_method(method, cost_diag(d, n), angles)
    return (1.0 - fidelity_sq(psi, g)).mean()


def _restore(angles: dict, saved: dict) -> None:
    with torch.no_grad():
        for k, v in saved.items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))


def _torch_packs(lam: float, pool6: dict, pool9: dict) -> dict:
    def take(pool: dict, sl: slice) -> dict:
        return {
            "diag": torch.tensor(np.asarray(pool["diag"][sl]), dtype=torch.float64),
            "gs": torch.tensor(np.asarray(pool["psi"][sl]), dtype=torch.float64),
            "e0": torch.tensor(np.asarray(pool["e0"][sl]), dtype=torch.float64),
        }

    return {
        "lam": float(lam),
        "train6": take(pool6, slice(0, TRAIN_B)),
        "hold6": take(pool6, slice(TRAIN_B, POOL_B)),
        "train9": take(pool9, slice(0, TRAIN_B)),
        "hold9": take(pool9, slice(TRAIN_B, POOL_B)),
    }


def _budget(rounds: int) -> dict:
    return {
        "rounds": int(rounds),
        "lbfgs_iter": LBFGS_ITER,
        "lbfgs_lr": LBFGS_LR,
        "batch": BATCH,
        "train_b": TRAIN_B,
        "hold_b": HOLD_B,
        "grad_clip": 5.0,
        "checkpoint": "max min(med train F n6, med train F n9, med hold F n6, med hold F n9)",
        "batch_rng": "numpy Generator([seed, L, 750100]); shared by b2 and fourier",
        "init": "paper β=0.01, γ=γ2=−0.01, φ_k=−β k inside the B=2 span",
    }


def _public_metrics(m: dict) -> dict:
    return {
        "B": m["B"],
        "med_F": m["med_F"],
        "mean_F": m["mean_F"],
        "min_F": m["min_F"],
        "med_eps": m["med_eps"],
        "mean_eps": m["mean_eps"],
        "med_E0": m["med_E0"],
        "F": m["F"],
        "eps": m["eps"],
    }


def _angle_diag(method: str, angles: dict) -> dict:
    if method == "b2":
        phi = np.asarray(angles["c"].detach().cpu().numpy() @ B2_BASIS, dtype=np.float64)
        return {
            "shell": "B2",
            "shell_rms_vs_linear": shell_nonlinearity(phi),
            "mean_abs_gamma2": float(np.mean(np.abs(angles["gamma2"].detach().cpu().numpy()))),
            "mean_abs_c": float(np.mean(np.abs(angles["c"].detach().cpu().numpy()))),
        }
    if method == "fourier":
        beta = np.asarray(angles["betas"].detach().cpu().numpy())
        gamma = np.asarray(angles["gammas"].detach().cpu().numpy())
        return {
            "mean_abs_beta": float(np.mean(np.abs(beta))),
            "mean_abs_gamma": float(np.mean(np.abs(gamma))),
        }
    return {}


def seed_path(L: int, method: str, seed: int, lam: float) -> Path:
    return RUN / f"L{L}_{method}_s{seed}_l{lam_tag(lam)}.json"


def train_done(path: Path, rounds: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return False
    budget = payload.get("budget") or {}
    return (
        payload.get("protocol") == PROTOCOL
        and payload.get("complete") is True
        and payload.get("objective") == "fidelity"
        and int(budget.get("rounds", -1)) == int(rounds)
        and "eval" in payload
        and "angles" in payload
    )


def _train_job(job: tuple) -> str:
    L, method, seed, lam, rounds = job
    path = seed_path(L, method, seed, lam)
    if train_done(path, rounds):
        return str(path)
    payload = train_one(method, L, seed, lam, rounds)
    write_json(path, payload)
    ev = payload["eval"]
    log(
        f"done L{L} {method} s{seed} λ={lam_tag(lam)} "
        f"train {ev['train6']['med_F']:.4f}/{ev['train9']['med_F']:.4f} "
        f"hold {ev['hold6']['med_F']:.4f}/{ev['hold9']['med_F']:.4f} "
        f"wall={payload['wall_s']:.0f}s"
    )
    return str(path)


def lists_to_angles(method: str, raw: dict) -> dict:
    keys = ("gamma2", "c") if method == "b2" else ("betas", "gammas")
    return {k: torch.tensor(raw[k], dtype=torch.float64) for k in keys}


def freeze_path(L: int, method: str, seed: int, lam: float) -> Path:
    return RUN / f"freeze_L{L}_{method}_s{seed}_l{lam_tag(lam)}.json"


def freeze_one(method: str, L: int, seed: int, lam: float, n15: int) -> dict:
    trained = read_json(seed_path(L, method, seed, lam))
    angles = lists_to_angles(method, trained["angles"])
    out = {"12": None, "15": None}
    specs = ((12, FREEZE12_B, 8), (15, int(n15), 4))
    for n, count, chunk in specs:
        pool = build_gs_pool(n, count, lam, workers=1)
        diag = torch.tensor(pool["diag"], dtype=torch.float64)
        gs = torch.tensor(pool["psi"], dtype=torch.float64)
        e0 = torch.tensor(pool["e0"], dtype=torch.float64)
        out[str(n)] = batch_metrics(method, angles, diag, gs, e0, n, lam, chunk=chunk)
    payload = {
        "protocol": PROTOCOL,
        "complete": True,
        "kind": "freeze",
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "mu": MU,
        "n12_B": FREEZE12_B,
        "n15_B": int(n15),
        "eval": {k: _public_metrics(v) for k, v in out.items()},
    }
    write_json(freeze_path(L, method, seed, lam), payload)
    log(
        f"freeze L{L} {method} s{seed} λ={lam_tag(lam)} "
        f"n12 F={payload['eval']['12']['med_F']:.4f} n15 F={payload['eval']['15']['med_F']:.4f}"
    )
    return payload


def _freeze_job(job: tuple) -> str:
    L, method, seed, lam, n15 = job
    path = freeze_path(L, method, seed, lam)
    if path.exists():
        payload = read_json(path)
        if (
            payload.get("complete") is True
            and int(payload.get("n15_B", -1)) == int(n15)
            and int(payload.get("L", -1)) == int(L)
        ):
            return str(path)
    freeze_one(method, L, seed, lam, n15)
    return str(path)


def _med(rows: list[dict], *keys: str) -> float:
    cur: list | float = rows
    for key in keys:
        if isinstance(cur, list):
            cur = [row[key] for row in cur]
        else:
            raise TypeError(key)
    return float(np.median(np.asarray(cur, dtype=np.float64)))


def collect_train(lams, depths, seeds, methods, rounds: int) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for lam in lams:
        for L in depths:
            for method in methods:
                rows = []
                for seed in seeds:
                    path = seed_path(L, method, seed, lam)
                    if not train_done(path, rounds):
                        raise RuntimeError(f"missing train result {path.name}")
                    rows.append(read_json(path))
                groups[(float(lam), method, int(L))] = rows
    return groups


def decision_for(lam: float, groups: dict, depths, seeds) -> dict:
    rows_by_L = {}
    fourier_by_L = {}
    for L in depths:
        snap = groups[(float(lam), "b2", int(L))]
        four = groups[(float(lam), "fourier", int(L))]
        rows_by_L[int(L)] = _ladder_point(snap)
        fourier_by_L[int(L)] = _ladder_point(four)
    strict = None
    for L in depths:
        if _clears(rows_by_L[int(L)]):
            strict = int(L)
            break
    best = max(depths, key=lambda L: rows_by_L[int(L)]["score"])
    train_hit_L = None
    for L in depths:
        pt = rows_by_L[int(L)]
        if pt["train6"] >= F_BAR - 1e-12 and pt["train9"] >= F_BAR - 1e-12:
            train_hit_L = int(L)
            break
    return {
        "lam": float(lam),
        "L_star": strict,
        "best_L": int(best),
        "train_hit_L": train_hit_L,
        "decision_L": int(strict if strict is not None else best),
        "b2": {str(L): rows_by_L[int(L)] for L in depths},
        "fourier": {str(L): fourier_by_L[int(L)] for L in depths},
        "seeds": list(seeds),
    }


def _ladder_point(rows: list[dict]) -> dict:
    point = {
        "train6": _med(rows, "eval", "train6", "med_F"),
        "train9": _med(rows, "eval", "train9", "med_F"),
        "hold6": _med(rows, "eval", "hold6", "med_F"),
        "hold9": _med(rows, "eval", "hold9", "med_F"),
        "train6_eps": _med(rows, "eval", "train6", "med_eps"),
        "train9_eps": _med(rows, "eval", "train9", "med_eps"),
        "hold6_eps": _med(rows, "eval", "hold6", "med_eps"),
        "hold9_eps": _med(rows, "eval", "hold9", "med_eps"),
        "per_seed": [
            {
                "seed": int(r["seed"]),
                "best_round": int(r["best_round"]),
                "early_stop": bool(r.get("early_stop")),
                "train6": float(r["eval"]["train6"]["med_F"]),
                "train9": float(r["eval"]["train9"]["med_F"]),
                "hold6": float(r["eval"]["hold6"]["med_F"]),
                "hold9": float(r["eval"]["hold9"]["med_F"]),
            }
            for r in rows
        ],
    }
    point["score"] = float(min(point["train6"], point["train9"], point["hold6"], point["hold9"]))
    point["train_min"] = float(min(point["train6"], point["train9"]))
    return point


def _clears(point: dict) -> bool:
    return all(float(point[k]) >= F_BAR - 1e-12 for k in ("train6", "train9", "hold6", "hold9"))


def attach_freeze(decision: dict, methods, seeds, n15: int) -> dict:
    L = int(decision["decision_L"])
    lam = float(decision["lam"])
    frozen = {}
    for method in methods:
        rows = []
        for seed in seeds:
            path = freeze_path(L, method, seed, lam)
            payload = read_json(path)
            rows.append(payload)
        inst12 = np.median(np.stack([np.asarray(r["eval"]["12"]["F"], dtype=np.float64) for r in rows]), axis=0)
        inst15 = np.median(np.stack([np.asarray(r["eval"]["15"]["F"], dtype=np.float64) for r in rows]), axis=0)
        frozen[method] = {
            "L": L,
            "n12_F": _med(rows, "eval", "12", "med_F"),
            "n15_F": _med(rows, "eval", "15", "med_F"),
            "n12_eps": _med(rows, "eval", "12", "med_eps"),
            "n15_eps": _med(rows, "eval", "15", "med_eps"),
            "n12_E0": _med(rows, "eval", "12", "med_E0"),
            "n15_E0": _med(rows, "eval", "15", "med_E0"),
            "n12_q25": float(np.quantile(inst12, 0.25)),
            "n12_min": float(np.min(inst12)),
            "n12_frac_ge_070": float(np.mean(inst12 >= 0.70)),
            "n15_q25": float(np.quantile(inst15, 0.25)),
            "n15_min": float(np.min(inst15)),
            "per_seed": [
                {
                    "seed": int(r["seed"]),
                    "n12_F": float(r["eval"]["12"]["med_F"]),
                    "n15_F": float(r["eval"]["15"]["med_F"]),
                    "n12_eps": float(r["eval"]["12"]["med_eps"]),
                    "n15_eps": float(r["eval"]["15"]["med_eps"]),
                }
                for r in rows
            ],
        }
    snap = frozen["b2"]
    train_ref = float(decision["b2"][str(L)]["train_min"])
    drop = float(train_ref - snap["n12_F"])
    train_hit = decision["train_hit_L"] is not None
    strict = decision["L_star"] is not None
    freeze_ok = snap["n12_F"] >= PASS_N12 - 1e-12 and drop <= PASS_DROP + 1e-12
    if not train_hit:
        verdict = "KILL"
        reason = (
            f"median train F never reached {F_BAR:.2f} on both n=6 and n=9 at L≤16. "
            f"Best L={decision['best_L']} score={decision['b2'][str(decision['best_L'])]['score']:.4f}."
        )
    elif strict and freeze_ok:
        verdict = "PASS"
        reason = (
            f"L*={decision['L_star']} clears train and hold F≥{F_BAR:.2f}. "
            f"Frozen n12 F={snap['n12_F']:.4f}, drop={drop:.4f}."
        )
    else:
        verdict = "needs_edit"
        if not strict:
            why = "train reached 0.95 but hold stayed under 0.95, so there is no L*"
        elif not freeze_ok:
            why = (
                f"train/hold cleared 0.95 at L*={decision['L_star']} but the freeze missed "
                f"the n12 bar (F={snap['n12_F']:.4f}, drop={drop:.4f})"
            )
        else:
            why = "train reached 0.95 and the freeze is inside the n12 bar, but L* was not met"
        reason = why
    decision = dict(decision)
    decision["freeze"] = frozen
    decision["train_ref"] = train_ref
    decision["drop12"] = drop
    decision["verdict"] = verdict
    decision["reason"] = reason
    if int(L) != 16:
        decision["side_L16"] = _side_block(float(lam), 16, decision, methods, seeds)
    return decision


def _pack_rows(rows: list[dict], L: int) -> dict:
    inst12 = np.median(np.stack([np.asarray(r["eval"]["12"]["F"], dtype=np.float64) for r in rows]), axis=0)
    inst15 = np.median(np.stack([np.asarray(r["eval"]["15"]["F"], dtype=np.float64) for r in rows]), axis=0)
    return {
        "L": int(L),
        "n12_F": _med(rows, "eval", "12", "med_F"),
        "n15_F": _med(rows, "eval", "15", "med_F"),
        "n12_eps": _med(rows, "eval", "12", "med_eps"),
        "n15_eps": _med(rows, "eval", "15", "med_eps"),
        "n12_q25": float(np.quantile(inst12, 0.25)),
        "n12_frac_ge_070": float(np.mean(inst12 >= 0.70)),
        "n15_q25": float(np.quantile(inst15, 0.25)),
    }


def _side_block(lam: float, L: int, decision: dict, methods, seeds) -> dict | None:
    packed = {}
    for method in methods:
        rows = []
        for seed in seeds:
            path = freeze_path(L, method, seed, lam)
            if not path.exists():
                return None
            rows.append(read_json(path))
        packed[method] = _pack_rows(rows, L)
    packed["drop12"] = float(decision["b2"][str(L)]["train_min"] - packed["b2"]["n12_F"])
    return packed


def render(summary: dict) -> str:
    lines = []
    lines.append("# Power-law TFIM fidelity transfer — λ ∈ {0.75, 1.00}")
    lines.append("")
    bits = []
    for dec in summary["decisions"]:
        star = "none" if dec["L_star"] is None else str(dec["L_star"])
        bits.append(f"λ={dec['lam']:.2f} **{dec['verdict']}** (L\\*={star})")
    lines.append("**Verdicts:** " + "; ".join(bits) + ".")
    lines.append("")
    lines.append(
        "Fidelity to the exact ground state, not an energy race against QAOA. "
        "F = |⟨ψ(θ)|GS⟩|². A depth clears the train bar when the median-over-seeds "
        "of the median-over-instances fidelity is at least 0.95 on the n=6 train split, "
        "the n=9 train split, and both holdouts. L\\* is the smallest such depth. "
        "PASS requires that L\\* and a frozen n=12 fidelity ≳ 0.70 whose drop from "
        "min(train n6, train n9) is ≲ 0.25. needs_edit means train hit ~95% and the "
        "holdout or the freeze did not. KILL means no depth through L=16 put both "
        "training sizes at ~95%."
    )
    lines.append("")
    lines.append("## Hamiltonian")
    lines.append("")
    lines.append("$$")
    lines.append(r"H = \sum_{i<j} \frac{J_{ij}}{|i-j|^{1.5}} Z_i Z_j - \lambda \sum_i X_i")
    lines.append("$$")
    lines.append("")
    lines.append(
        r"μ = 0, α = 1.5, \(J_{ij}\in\{\pm 1\}\) on the open chain. "
        r"Qubit 0 is the MSB. Cavity \(c\) owns \([3c, 3c+3)\). "
        r"Bonds that cross a cavity stay in the sum. "
        r"Circuit phases use \(C_n = H_{ZZ}/n\); \(\langle H\rangle\) and \(E_0\) use the formula above. "
        r"\(H_{ZZ}\) is pure Walsh degree 2."
    )
    lines.append("")
    cross = summary.get("cross_med") or {}
    if cross:
        bits = ", ".join(f"n{n}={cross[n]:.3f}" for n in sorted(cross, key=lambda s: int(s)))
        lines.append(f"Median cross-cavity share of Σ|K_ij| on the pools used here: {bits}.")
        lines.append("")
    lines.append("## Phase diagnostic")
    lines.append("")
    lines.append(
        "Bitstring peak is max_z |⟨z|GS⟩|². Product gap is (E_prod − E_0)/|E_0| "
        "with the same product-state search as the breadth smoke. "
        "⟨X⟩ is the ground-state mean of Σ_i ⟨X_i⟩ / n. "
        "The spectral gap is (E_1 − E_0)/|E_0|. "
        "Holdout indices 64..95 at λ=1 reproduce the PR #42 breadth cache. "
        "The quantumness bit is that smoke's rule, median peak < 0.5 and median product gap ≥ 0.03. "
        "It is not the fidelity verdict."
    )
    lines.append("")
    lines.append("| λ | n | split | B | peak | product gap | ⟨X⟩ | (E1−E0)/|E0| | E0 | quantumness |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for phase in summary["phases"]:
        for n in ("6", "9"):
            for split in ("train", "hold", "all"):
                b = phase["n"][n][split]
                lines.append(
                    f"| {phase['lam']:.2f} | {n} | {split} | {b['B']} | {b['peak_med']:.4f} | "
                    f"{b['gap_med']:.4f} | {b['x_med']:.4f} | {b['spec_gap_med']:.4f} | "
                    f"{b['E0_med']:.3f} | {'pass' if b['pass'] else 'fail'} |"
                )
    lines.append("")
    lines.append("## Ansatz and budget")
    lines.append("")
    lines.append(
        "Walsh SNAP (b2): each layer has γ_2 on Walsh degree 2 of C_n and two shell "
        "coefficients. φ_k = c₁ u_k + c₂ v_k, where {u, v} is the orthonormal basis of "
        "span{k, k²} on popcount k=0..3 used by the B=2 ablation. Paper init φ_k = −β k "
        "sits in that span, so the init matches Fourier. The full four-phase Hamming shell "
        "was not needed for the parameterization. Fourier is the reference column on the "
        "same C_n, same batches, same L-BFGS budget. It does not decide the verdict."
    )
    lines.append("")
    lines.append("| knob | value |")
    lines.append("|---|---|")
    lines.append(f"| λ | {list(summary['lams'])} |")
    lines.append("| μ, α | 0, 1.5 |")
    lines.append(f"| L | {list(summary['depths'])} |")
    lines.append(f"| seeds | {list(summary['seeds'])} |")
    lines.append(
        f"| pools | train B={TRAIN_B} indices 0..63, hold B={HOLD_B} indices 64..95, "
        f"joint n=6 and n=9 |"
    )
    lines.append(
        f"| freeze | n=12 B={FREEZE12_B} indices 0..63, n=15 B={summary['n15']} indices 0.., no retrain |"
    )
    lines.append(
        f"| optimizer | L-BFGS {summary['rounds']}×{LBFGS_ITER}, lr {LBFGS_LR}, "
        f"batch {BATCH} (half n6 / half n9), grad clip 5 |"
    )
    lines.append("| loss | mean of per-size mean(1 − F) |")
    lines.append("| checkpoint | highest min of the four split median fidelities |")
    lines.append("| early stop | three rounds in a row with that min ≥ 0.99 |")
    lines.append("| instance seed | 1500003017 + n · 100003 + index |")
    lines.append("")
    lines.append("## Ladder")
    lines.append("")
    lines.append(
        "Each entry is the median over the three seeds. The four fidelities are "
        "train n6, train n9, hold n6, hold n9. score is their minimum."
    )
    lines.append("")
    for dec in summary["decisions"]:
        lines.append(f"### λ = {dec['lam']:.2f}")
        lines.append("")
        lines.append(
            "| L | b2 train6 | b2 train9 | b2 hold6 | b2 hold9 | b2 score | "
            "Fourier train6 | Fourier train9 | Fourier hold6 | Fourier hold9 |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for L in summary["depths"]:
            b = dec["b2"][str(L)]
            f = dec["fourier"][str(L)]
            lines.append(
                f"| {L} | {b['train6']:.4f} | {b['train9']:.4f} | {b['hold6']:.4f} | {b['hold9']:.4f} | "
                f"{b['score']:.4f} | {f['train6']:.4f} | {f['train9']:.4f} | {f['hold6']:.4f} | {f['hold9']:.4f} |"
            )
        lines.append("")
        lines.append(
            "Med ε = ⟨H⟩ − E_0 at the fidelity checkpoint (b2 / fourier), holdout."
        )
        lines.append("")
        lines.append("| L | n6 b2 | n6 Fourier | n9 b2 | n9 Fourier |")
        lines.append("|---:|---:|---:|---:|---:|")
        for L in summary["depths"]:
            b = dec["b2"][str(L)]
            f = dec["fourier"][str(L)]
            lines.append(
                f"| {L} | {b['hold6_eps']:.4f} | {f['hold6_eps']:.4f} | "
                f"{b['hold9_eps']:.4f} | {f['hold9_eps']:.4f} |"
            )
        lines.append("")
        star = "none" if dec["L_star"] is None else str(dec["L_star"])
        fstar = next((int(L) for L in summary["depths"] if _clears(dec["fourier"][str(L)])), None)
        lines.append(
            f"L\\* = {star}. Best score is L={dec['best_L']} "
            f"({dec['b2'][str(dec['best_L'])]['score']:.4f}). "
            f"Smallest L with both train fidelities ≥ 0.95: "
            f"{'none' if dec['train_hit_L'] is None else dec['train_hit_L']}. "
            f"Frozen depth L={dec['decision_L']}. "
            f"{'Fourier does not clear the same four-way 0.95 bar at any L≤16.' if fstar is None else 'Fourier first clears the same four-way 0.95 bar at L=' + str(fstar) + '.'}"
        )
        lines.append("")
        lines.append("Per seed at the frozen depth, b2 then Fourier (train6, train9, hold6, hold9):")
        lines.append("")
        for method, label in (("b2", "b2"), ("fourier", "Fourier")):
            pt = dec[method][str(dec["decision_L"])]
            for row in pt["per_seed"]:
                lines.append(
                    f"- {label} seed {row['seed']}: "
                    f"{row['train6']:.4f}, {row['train9']:.4f}, {row['hold6']:.4f}, {row['hold9']:.4f} "
                    f"(best round {row['best_round']}"
                    f"{', early stop' if row['early_stop'] else ''})"
                )
        lines.append("")
    lines.append("## Freeze")
    lines.append("")
    lines.append(
        "Angles are the fidelity checkpoint from n∈{6, 9}. "
        "n=12 and n=15 are new draws from the same seed formula. No retrain. "
        "Drop = min(train n6, train n9) − n12 F, using the median-over-seeds numbers "
        "at the frozen depth."
    )
    lines.append("")
    lines.append("| λ | verdict | L | n12 F b2 | drop | n15 F b2 | n12 F Fourier | n15 F Fourier | n12 Medε b2 | n12 E0 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dec in summary["decisions"]:
        b = dec["freeze"]["b2"]
        f = dec["freeze"]["fourier"]
        lines.append(
            f"| {dec['lam']:.2f} | {dec['verdict']} | {dec['decision_L']} | "
            f"{b['n12_F']:.4f} | {dec['drop12']:.4f} | {b['n15_F']:.4f} | "
            f"{f['n12_F']:.4f} | {f['n15_F']:.4f} | {b['n12_eps']:.4f} | {b['n12_E0']:.3f} |"
        )
    lines.append("")
    for dec in summary["decisions"]:
        lines.append(f"**λ={dec['lam']:.2f}: {dec['verdict']}.** {dec['reason']}")
        lines.append("")
        b = dec["freeze"]["b2"]
        lines.append(
            f"n12 Medε = {b['n12_eps']:.4f} against median E0 = {b['n12_E0']:.3f}; "
            f"n15 Medε = {b['n15_eps']:.4f} against median E0 = {b['n15_E0']:.3f}. "
            f"Across instances, the seed-median n12 fidelity has lower quartile {b['n12_q25']:.3f}, "
            f"minimum {b['n12_min']:.3f}, and fraction ≥ 0.70 equal to {b['n12_frac_ge_070']:.2f}. "
            f"n15 lower quartile is {b['n15_q25']:.3f}."
        )
        lines.append("")
        side = dec.get("side_L16")
        if side:
            sb = side["b2"]
            sf = side["fourier"]
            lines.append(
                f"Side measurement at L=16, not used for the verdict. "
                f"That depth also clears train and hold. Frozen b2 n12 F={sb['n12_F']:.4f}, "
                f"drop={side['drop12']:.4f}, n15 F={sb['n15_F']:.4f} "
                f"(lower quartile {sb['n12_q25']:.3f}, fraction ≥ 0.70 is {sb['n12_frac_ge_070']:.2f}). "
                f"Fourier at the same angles-depth has n12 F={sf['n12_F']:.4f} and n15 F={sf['n15_F']:.4f}."
            )
            lines.append("")
        for row in b["per_seed"]:
            lines.append(
                f"- b2 seed {row['seed']}: n12 F={row['n12_F']:.4f} (ε={row['n12_eps']:.4f}), "
                f"n15 F={row['n15_F']:.4f} (ε={row['n15_eps']:.4f})"
            )
        lines.append("")
    lines.append("## Repro")
    lines.append("")
    lines.append(
        "Helpers are `Nov_3rd/powerlaw_tfim.py` and `Nov_3rd/hva_safe.py` from the PR #42 family, unchanged. "
        f"Runner: `Nov_3rd/scripts/run_powerlaw_fidelity_lam075_100.py`. Protocol `{PROTOCOL}`."
    )
    lines.append("")
    return "\n".join(lines)


def startup_checks() -> None:
    log("self-check")
    self_check()
    gram = B2_BASIS @ B2_BASIS.T
    if np.max(np.abs(gram - np.eye(2))) > 1e-10:
        raise AssertionError(f"B2 basis not orthonormal: {gram}")
    phi = B2_INIT @ B2_BASIS
    if np.max(np.abs(phi + INIT_BETA * _K4)) > 1e-10:
        raise AssertionError(f"B2 init missed the paper shell: {phi}")
    inst = make_instance(6, 0)
    Cn = cost_diag(torch.tensor(inst["diag"], dtype=torch.float64).reshape(1, -1), 6)
    graded = walsh_degree_diag(Cn, torch.tensor([0.0, 0.3, 0.0, 0.0], dtype=torch.float64))
    if float(torch.max(torch.abs(graded - 0.3 * Cn))) > 1e-8:
        raise AssertionError("degree-2 Walsh weight is not scalar γ on this H_ZZ")
    L = 2
    sf = prepare("fourier", Cn, init_method("fourier", L, dtype=torch.float64))
    sb = prepare_b2(Cn, init_b2(L))
    # init_b2 returns plain tensors; prepare_b2 multiplies c @ basis. Good.
    fid = float(state_fidelity(sf, sb)[0])
    if fid < 1.0 - 1e-8:
        raise AssertionError(f"B2 paper init is not Fourier (F_amp={fid})")
    gs = ground_bundle(inst["diag"], 6, 0.75)
    g = torch.tensor(gs["psi"], dtype=torch.float64).reshape(1, -1)
    ang = init_b2(1)
    ang["c"] = torch.nn.Parameter(ang["c"].detach().clone())
    loss = 1.0 - fidelity_sq(prepare_b2(Cn, ang), g)[0]
    grad = torch.autograd.grad(loss, ang["c"])[0].detach().numpy()
    eps = 1e-6
    base = ang["c"].detach().numpy().copy()
    flat = int(np.argmax(np.abs(grad)))
    direction = np.zeros_like(base)
    direction.reshape(-1)[flat] = 1.0
    def loss_at(arr):
        trial = {"gamma2": ang["gamma2"].detach(), "c": torch.tensor(arr, dtype=torch.float64)}
        with torch.no_grad():
            return float((1.0 - fidelity_sq(prepare_b2(Cn, trial), g)[0]).detach())
    fd = (loss_at(base + eps * direction) - loss_at(base - eps * direction)) / (2 * eps)
    analytic = float(grad.reshape(-1)[flat])
    if abs(fd - analytic) > 1e-4 * max(1.0, abs(analytic)):
        raise AssertionError(f"B2 fidelity grad {analytic} != FD {fd}")
    # Degree-2 table the trainer actually builds agrees with a direct γ2.
    table = snap_angles_from_gamma2(
        torch.tensor([0.2, -0.4], dtype=torch.float64),
        torch.tensor(phi, dtype=torch.float64).reshape(1, 4).expand(2, 4).contiguous(),
    )
    if not torch.equal(table["gamma_w"][:, 1], torch.tensor([0.2, -0.4], dtype=torch.float64)):
        raise AssertionError("γ2 was not written onto Walsh degree 2")
    log("startup checks passed")


def run_jobs(jobs: list[tuple], workers: int, fn) -> None:
    if not jobs:
        return
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            fn(job)
        return
    failed = 0
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futs = [pool.submit(fn, job) for job in jobs]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception:
                failed += 1
                traceback.print_exc()
    if failed:
        raise RuntimeError(f"{failed} jobs failed")


def assemble(lams, depths, seeds, methods, rounds: int, n15: int) -> dict:
    groups = collect_train(lams, depths, seeds, methods, rounds)
    decisions = []
    phases = []
    cross = {}
    for n, count in ((6, POOL_B), (9, POOL_B), (12, FREEZE12_B), (15, int(n15))):
        # Geometry does not depend on λ. Read whichever cache is already on disk.
        for lam in lams:
            path = gs_path(n, lam, count)
            if path.exists():
                cross[str(n)] = float(np.median(np.load(path)["cross"]))
                break
    for lam in lams:
        phase = read_json(phase_path(lam))
        phases.append(phase)
        dec = decision_for(lam, groups, depths, seeds)
        decisions.append(attach_freeze(dec, methods, seeds, n15))
    summary = {
        "protocol": PROTOCOL,
        "lams": [float(v) for v in lams],
        "depths": [int(v) for v in depths],
        "seeds": [int(v) for v in seeds],
        "methods": list(methods),
        "rounds": int(rounds),
        "n15": int(n15),
        "mu": MU,
        "alpha": ALPHA,
        "F_bar": F_BAR,
        "pass_n12": PASS_N12,
        "pass_drop": PASS_DROP,
        "b2_basis": B2_BASIS.tolist(),
        "b2_init": B2_INIT.tolist(),
        "cross_med": cross,
        "phases": phases,
        "decisions": decisions,
        "verdicts": {lam_tag(d["lam"]): d["verdict"] for d in decisions},
    }
    write_json(RUN / "summary.json", summary)
    text = render(summary)
    (RUN / "RESULTS.md").write_text(text)
    log("wrote", RUN / "RESULTS.md")
    for dec in decisions:
        log(f"λ={dec['lam']:.2f} {dec['verdict']}: {dec['reason']}")
    return summary


def bench(workers: int) -> None:
    startup_checks()
    for lam in LAMS:
        p6 = build_gs_pool(6, POOL_B, lam, workers)
        p9 = build_gs_pool(9, POOL_B, lam, workers)
        phase = build_phase(lam, p6, p9, workers)
        check_published_phase(phase)
    from Nov_3rd.powerlaw_tfim import ground_state

    t0 = time.time()
    diag12 = zz_diag_from_K(couplings(12, 0)[1])
    gs12 = ground_bundle(diag12, 12, 0.75)
    ref12 = ground_state(diag12, 12, 0.75, 0.0)
    if abs(gs12["E0"] - ref12["E0"]) > 1e-6 or abs(gs12["peak"] - ref12["peak"]) > 1e-6:
        raise RuntimeError(f"n12 GS {gs12['E0']} peak {gs12['peak']} != helper {ref12}")
    log(f"one n12 GS {time.time()-t0:.2f}s E0={gs12['E0']:.4f} peak={gs12['peak']:.4f} matches helper")
    t0 = time.time()
    gs15 = ground_bundle(zz_diag_from_K(couplings(15, 0)[1]), 15, 0.75)
    log(f"one n15 GS {time.time()-t0:.2f}s E0={gs15['E0']:.4f} peak={gs15['peak']:.4f}")
    t0 = time.time()
    row = train_one("b2", 4, 0, 0.75, rounds=2)
    log(f"2-round L4 b2 wall={time.time()-t0:.1f}s score={row['best_score']:.4f}")
    t0 = time.time()
    row = train_one("b2", 16, 0, 0.75, rounds=1)
    log(f"1-round L16 b2 wall={time.time()-t0:.1f}s score={row['best_score']:.4f}")
    t0 = time.time()
    row = train_one("fourier", 16, 0, 0.75, rounds=1)
    log(f"1-round L16 fourier wall={time.time()-t0:.1f}s score={row['best_score']:.4f}")


def main() -> None:
    global LOG_PATH, FREEZE15_B
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="*", default=list(LAMS))
    ap.add_argument("--depths", type=int, nargs="*", default=list(DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n15", type=int, default=FREEZE15_B)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    LOG_PATH = RUN / "train.log"
    if args.bench:
        bench(args.workers)
        return
    startup_checks()
    lams = tuple(float(v) for v in args.lams)
    depths = tuple(int(v) for v in args.depths)
    seeds = tuple(int(v) for v in args.seeds)
    methods = tuple(args.methods)
    for lam in lams:
        p6 = build_gs_pool(6, POOL_B, lam, args.workers)
        p9 = build_gs_pool(9, POOL_B, lam, args.workers)
        phase = build_phase(lam, p6, p9, args.workers)
        check_published_phase(phase)
    if not args.assemble_only and not args.skip_train:
        jobs = []
        for lam in lams:
            for L in depths:
                for method in methods:
                    for seed in seeds:
                        path = seed_path(L, method, seed, lam)
                        if not train_done(path, args.rounds):
                            jobs.append((L, method, seed, lam, args.rounds))
        log(f"training jobs {len(jobs)}")
        run_jobs(jobs, args.workers, _train_job)
    if args.skip_train:
        return
    groups = collect_train(lams, depths, seeds, methods, args.rounds)
    freeze_jobs = []
    for lam in lams:
        dec = decision_for(lam, groups, depths, seeds)
        log(
            f"λ={lam_tag(lam)} decision L={dec['decision_L']} "
            f"L*={dec['L_star']} best={dec['best_L']} train_hit={dec['train_hit_L']}"
        )
        # Ground states before the freeze pool so workers only read the cache.
        build_gs_pool(12, FREEZE12_B, lam, args.workers)
        build_gs_pool(15, int(args.n15), lam, args.workers)
        for method in methods:
            for seed in seeds:
                freeze_jobs.append((dec["decision_L"], method, seed, lam, int(args.n15)))
    if not args.assemble_only:
        missing = [job for job in freeze_jobs if not _freeze_done(job)]
        log(f"freeze jobs {len(missing)}")
        run_jobs(missing, args.workers, _freeze_job)
    assemble(lams, depths, seeds, methods, args.rounds, int(args.n15))


def _freeze_done(job: tuple) -> bool:
    L, method, seed, lam, n15 = job
    path = freeze_path(L, method, seed, lam)
    if not path.exists():
        return False
    payload = read_json(path)
    return (
        payload.get("complete") is True
        and int(payload.get("n15_B", -1)) == int(n15)
        and int(payload.get("L", -1)) == int(L)
    )


if __name__ == "__main__":
    main()
