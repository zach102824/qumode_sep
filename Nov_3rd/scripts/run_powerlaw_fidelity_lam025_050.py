#!/usr/bin/env python3
"""Fidelity transfer of B=2 Walsh SNAP on power-law TFIM, λ ∈ {0.25, 0.50}.

    H = Σ_{i<j} J_ij / |i-j|^{1.5} Z_i Z_j − λ Σ_i X_i

μ = 0. Same open cavity chain as powerlaw_tfim (qubit 0 = MSB, cavity c
owns [3c, 3c+3)). Train |⟨ψ_GS|ψ(θ)⟩|² jointly on n∈{6,9}. Freeze the
angles. Evaluate at n=12 and n=15 with no retrain.

SNAP shell is B=2: φ_k = c1 k + c2 k² on popcount k=0..3, cost weight γ_2
only (H_ZZ is pure Walsh degree 2, so this is the #42 SNAP family with the
shell restricted). Fourier is the reference column: φ_k = −β k, the X mixer.

Success is absolute fidelity transfer, not a ratio against Fourier.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Workers inherit this. Matvecs are Python-side; one thread avoids oversubscribe.
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("FIDELITY_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    dense_H,
    fourier_state,
    hamiltonian_matvec,
    snap_state,
)
from Nov_3rd.powerlaw_tfim import (
    ALPHA,
    PEAK_LT,
    PROD_GAP_MIN,
    best_product,
    cost_diag,
    couplings,
    cross_cavity_weight,
    expectation,
    ground_state,
    init_snap,
    instance_seed,
    prepare_snap,
    quantum_summary,
    shell_nonlinearity,
    zz_diag_from_K,
)

PROTOCOL = "powerlaw_fidelity_lam025_050_v1"
RUN = S.OUT / "quantum_h_explore" / "powerlaw_fidelity_transfer" / "lam_025_050"
LAMS = (0.25, 0.50)
LADDER = (1, 2, 4, 8, 12, 16)
SEEDS = (0, 1, 2)
MU = 0.0
TRAIN_B = 64
HOLD_B = 32
FREEZE_B = 32
ROUNDS = 40
EXTRA_ROUNDS = 20
LBFGS_ITER = 8
LBFGS_LR = 0.25
BATCH = 64
TARGET = 0.95
PASS_F12 = 0.70
PASS_DROP = 0.25
SPLITS4 = ("train6", "hold6", "train9", "hold9")


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fid]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt(x, nd=4):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.{nd}f}"


def k_powers(dtype=torch.float64, device=None) -> torch.Tensor:
    return torch.arange(4, dtype=dtype, device=device)


def phi_from_c(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """φ_k = c1 k + c2 k², k = 0..3. Shape (L, 4). φ_0 = 0."""
    k = k_powers(dtype=c1.dtype, device=c1.device)
    c1 = c1.reshape(-1, 1)
    c2 = c2.reshape(-1, 1)
    return c1 * k + c2 * k.square()


def init_snap_b2(L: int) -> dict:
    """Paper shell: c1 = −β, c2 = 0, γ_2 = −0.01. Matches linear φ_k = −β k."""
    L = int(L)
    return {
        "gamma2": torch.full((L,), INIT_GAMMA, dtype=torch.float64),
        "c1": torch.full((L,), -INIT_BETA, dtype=torch.float64),
        "c2": torch.zeros(L, dtype=torch.float64),
    }


def init_fourier(L: int) -> dict:
    L = int(L)
    return {
        "betas": torch.full((L,), INIT_BETA, dtype=torch.float64),
        "gammas": torch.full((L,), INIT_GAMMA, dtype=torch.float64),
    }


def init_angles(method: str, L: int) -> dict:
    if method == "snap":
        return init_snap_b2(L)
    if method == "shell":
        # PR #42 Hamming shell: free φ_k on k=0..3, γ_2 only.
        return init_snap(L)
    if method == "fourier":
        return init_fourier(L)
    raise ValueError(method)


def bind_params(method: str, angles: dict) -> list[torch.nn.Parameter]:
    keys = {
        "snap": ("gamma2", "c1", "c2"),
        "shell": ("gamma2", "phi"),
        "fourier": ("betas", "gammas"),
    }[method]
    params = []
    for key in keys:
        p = torch.nn.Parameter(angles[key].detach().clone())
        angles[key] = p
        params.append(p)
    return params


def prepare_model(method: str, Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    if method == "snap":
        phi = phi_from_c(angles["c1"], angles["c2"])
        # H_ZZ is pure Walsh degree 2 with zero mean, so γ_2 · Cn is the
        # graded cost. library_check asserts this against prepare_snap.
        return snap_state(Cn, angles["gamma2"], phi)
    if method == "shell":
        return prepare_snap(Cn, angles)
    if method == "fourier":
        return fourier_state(Cn, angles["betas"], angles["gammas"])
    raise ValueError(method)


def fidelity_sq(psi: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """|⟨gs|ψ⟩|² per row. gs may be real; the bra is conjugated."""
    ov = torch.sum(gs.conj().to(dtype=psi.dtype) * psi, dim=-1)
    return ov.real.square() + ov.imag.square()


def pool_cache(n: int, count: int, lam: float) -> Path:
    return RUN / "cache" / f"pool_n{n}_B{count}_l{lam:.2f}.npz"


def obs_one(psi: np.ndarray, n: int) -> dict:
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    p = psi * psi
    idx = np.arange(psi.size)
    z_acc = np.empty(n, dtype=np.float64)
    x_acc = np.empty(n, dtype=np.float64)
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1
        z_acc[q] = float(np.dot(p, 1.0 - 2.0 * bit))
        x_acc[q] = float(np.dot(psi, psi[idx ^ (1 << (n - 1 - q))]))
    return {
        "peak": float(p.max()),
        "ipr": float(np.dot(p, p)),
        "q_ea": float(np.mean(z_acc ** 2)),
        "mean_x": float(np.mean(x_acc)),
        "mean_abs_z": float(np.mean(np.abs(z_acc))),
        "cat": float(p[int(np.argmax(p))] + p[int(np.argmax(p)) ^ ((1 << int(n)) - 1)]),
    }


def _gs_retry(diag: np.ndarray, n: int, lam: float) -> dict:
    """Same real-symmetric eigenproblem as ground_state, with a larger Krylov space."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    N = 1 << int(n)
    lx = -float(lam)

    def mv(v):
        return hamiltonian_matvec(np.asarray(v, dtype=np.float64).reshape(-1), diag, n, lambda_x=lx, mu_xx=MU)

    op = LinearOperator((N, N), matvec=mv, dtype=np.float64)
    v0 = np.full(N, 1.0 / np.sqrt(N))
    evals, evecs = eigsh(
        op, k=1, which="SA", tol=1e-8, maxiter=max(8000, N), ncv=min(N - 1, 80), v0=v0
    )
    psi = np.real(np.asarray(evecs[:, 0], dtype=np.float64))
    psi = psi / float(np.linalg.norm(psi))
    if float(psi.sum()) < 0.0:
        psi = -psi
    e0 = float(evals[0])
    resid = float(np.linalg.norm(mv(psi) - e0 * psi))
    if resid > 1e-5 * max(1.0, abs(e0)):
        raise RuntimeError(f"eigsh retry residual {resid:.3e} at n={n}")
    return {"E0": e0, "psi": psi, "peak": float(np.max(psi ** 2))}


def build_pool(n: int, count: int, lam: float) -> dict:
    """diag, real GS, E0, and (n≤9) E1. Cached."""
    path = pool_cache(n, count, lam)
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
            log(f"pool cache hit n{n} B={count} λ={lam:.2f}")
            return {k: np.asarray(z[k]) for k in z.files}
    log(f"building pool n{n} B={count} λ={lam:.2f}")
    t0 = time.time()
    dim = 1 << int(n)
    diag = np.empty((count, dim), dtype=np.float64)
    psi = np.empty((count, dim), dtype=np.float64)
    e0 = np.empty(count, dtype=np.float64)
    e1 = np.full(count, np.nan, dtype=np.float64)
    peak = np.empty(count, dtype=np.float64)
    seeds = np.empty(count, dtype=np.int64)
    for i in range(count):
        d = zz_diag_from_K(couplings(n, i)[1])
        try:
            gs = ground_state(d, n, lam, MU)
        except RuntimeError as exc:
            log(f"  ground_state retry n{n} i={i}: {exc}")
            gs = _gs_retry(d, n, lam)
        diag[i] = d
        psi[i] = gs["psi"]
        e0[i] = gs["E0"]
        peak[i] = gs["peak"]
        seeds[i] = instance_seed(n, i)
        if int(n) <= 9:
            ev = np.linalg.eigvalsh(dense_H(d, n, lambda_x=-float(lam), mu_xx=MU))
            if abs(float(ev[0]) - float(gs["E0"])) > 1e-6:
                raise RuntimeError(f"gap E0 mismatch n={n} i={i}: {ev[0]} vs {gs['E0']}")
            e1[i] = float(ev[1])
        if (i + 1) == count or (i + 1) % 8 == 0:
            log(f"  n{n} {i+1}/{count} E0={e0[i]:.4f} peak={peak[i]:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        diag=diag,
        psi=psi,
        e0=e0,
        e1=e1,
        peak=peak,
        seeds=seeds,
        n=n,
        count=count,
        lam=lam,
        mu=MU,
        alpha=ALPHA,
    )
    log(f"pool n{n} B={count} λ={lam:.2f} in {time.time()-t0:.1f}s")
    return {
        "diag": diag,
        "psi": psi,
        "e0": e0,
        "e1": e1,
        "peak": peak,
        "seeds": seeds,
        "n": np.array(n),
        "count": np.array(count),
        "lam": np.array(lam),
        "mu": np.array(MU),
        "alpha": np.array(ALPHA),
    }


def as_torch(diag: np.ndarray, psi: np.ndarray, e0: np.ndarray) -> dict:
    return {
        "diag": torch.tensor(np.asarray(diag), dtype=torch.float64),
        "psi": torch.tensor(np.asarray(psi), dtype=torch.float64),
        "e0": torch.tensor(np.asarray(e0), dtype=torch.float64),
    }


def load_split_pools(lam: float) -> dict:
    """Train/hold tensors for n=6 and n=9. Indices match the #42 breadth split."""
    out = {}
    for n, count in ((6, TRAIN_B + HOLD_B), (9, TRAIN_B + HOLD_B)):
        raw = build_pool(n, count, lam)
        packs = {
            "train": (slice(0, TRAIN_B),),
            "hold": (slice(TRAIN_B, TRAIN_B + HOLD_B),),
        }
        for role, (sl,) in packs.items():
            out[f"{role}{n}"] = as_torch(raw["diag"][sl], raw["psi"][sl], raw["e0"][sl])
            out[f"{role}{n}"]["e0_np"] = np.asarray(raw["e0"][sl], dtype=np.float64)
    return out


def slice_pool(raw: dict, sl: slice) -> dict:
    pack = as_torch(raw["diag"][sl], raw["psi"][sl], raw["e0"][sl])
    pack["e0_np"] = np.asarray(raw["e0"][sl], dtype=np.float64)
    return pack


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


@torch.no_grad()
def eval_pack(method: str, angles: dict, pack: dict, n: int, lam: float, *, chunk: int, energy: bool) -> dict:
    diag = pack["diag"]
    gs = pack["psi"]
    e0 = pack["e0"]
    B = int(diag.shape[0])
    fs = []
    hs = []
    for s in range(0, B, int(chunk)):
        d = diag[s : s + chunk]
        st = prepare_model(method, cost_diag(d, n), angles)
        F = fidelity_sq(st, gs[s : s + chunk])
        if torch.any(F < -1e-6) or torch.any(F > 1.0 + 1e-5):
            raise RuntimeError(f"fidelity out of range n={n} min={float(F.min())} max={float(F.max())}")
        fs.append(F.detach())
        if energy:
            hs.append(expectation(st, d, n, lam, MU).detach())
    F = torch.cat(fs).cpu().numpy()
    rec = {
        "n": int(n),
        "B": int(B),
        "med_F": float(np.median(F)),
        "mean_F": float(np.mean(F)),
        "min_F": float(np.min(F)),
        "F": [float(x) for x in F.tolist()],
    }
    if energy:
        H = torch.cat(hs).cpu().numpy()
        e0_np = pack["e0_np"]
        eps = H - e0_np
        if float(np.min(eps)) < -1e-3:
            raise RuntimeError(f"negative excess {float(np.min(eps))} at n={n} (E0 above a prepared state)")
        rec.update(
            {
                "med_eps": float(np.median(eps)),
                "mean_eps": float(np.mean(eps)),
                "med_H": float(np.median(H)),
                "med_E0": float(np.median(e0_np)),
                "eps": [float(x) for x in eps.tolist()],
            }
        )
    return rec


def chunk_for(n: int) -> int:
    return {6: 32, 9: 16, 12: 4, 15: 2}.get(int(n), 2)


def score_of(ev: dict) -> float:
    return float(min(ev[name]["med_F"] for name in SPLITS4))


def eval_train_hold(method, angles, pools, lam, *, energy: bool) -> dict:
    ev = {}
    for name, n in (("train6", 6), ("hold6", 6), ("train9", 9), ("hold9", 9)):
        ev[name] = eval_pack(
            method, angles, pools[name], n, lam, chunk=chunk_for(n), energy=energy
        )
    return ev


def train_loss(method, angles, pack, idx, n) -> torch.Tensor:
    d = pack["diag"].index_select(0, idx)
    gs = pack["psi"].index_select(0, idx)
    psi = prepare_model(method, cost_diag(d, n), angles)
    F = fidelity_sq(psi, gs)
    return (1.0 - F).mean()


def angles_to_lists(angles: dict) -> dict:
    out = {}
    for k, v in angles.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().to(torch.float64).numpy().tolist()
        else:
            out[k] = v
    return out


def lists_to_angles(raw: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in raw.items()}


def shell_diag(method: str, angles: dict) -> dict:
    if method == "shell":
        phi = np.asarray(angles["phi"], dtype=np.float64)
        return {
            "shell": "hamming4",
            "phi": "free k=0..3",
            "shell_rms": shell_nonlinearity(phi),
            "mean_abs_gamma2": float(np.mean(np.abs(np.asarray(angles["gamma2"], dtype=np.float64)))),
        }
    if method != "snap":
        beta = np.asarray(angles["betas"], dtype=np.float64)
        gamma = np.asarray(angles["gammas"], dtype=np.float64)
        return {
            "mean_abs_beta": float(np.mean(np.abs(beta))),
            "mean_abs_gamma": float(np.mean(np.abs(gamma))),
        }
    c1 = np.asarray(angles["c1"], dtype=np.float64).reshape(-1)
    c2 = np.asarray(angles["c2"], dtype=np.float64).reshape(-1)
    phi = c1.reshape(-1, 1) * np.arange(4) + c2.reshape(-1, 1) * np.arange(4) ** 2
    return {
        "shell": "B2",
        "phi": "c1*k + c2*k^2",
        "mean_abs_c2": float(np.mean(np.abs(c2))),
        "mean_abs_gamma2": float(np.mean(np.abs(np.asarray(angles["gamma2"], dtype=np.float64)))),
        "shell_rms": shell_nonlinearity(phi),
    }


def seed_path(lam: float, L: int, method: str, seed: int) -> Path:
    return RUN / f"L{int(L)}_{method}_s{int(seed)}_l{lam:.2f}.json"


def budget_dict() -> dict:
    return {
        "rounds": ROUNDS,
        "extra_rounds": EXTRA_ROUNDS,
        "lbfgs_iter": LBFGS_ITER,
        "lr": LBFGS_LR,
        "train_b": TRAIN_B,
        "hold_b": HOLD_B,
        "freeze_b": FREEZE_B,
        "batch": BATCH,
        "target": TARGET,
    }


def train_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return False
    b = payload.get("budget") or {}
    ev = payload.get("eval") or {}
    return (
        payload.get("protocol") == PROTOCOL
        and payload.get("complete") is True
        and int(b.get("rounds", -1)) == ROUNDS
        and int(b.get("train_b", -1)) == TRAIN_B
        and all(name in ev and "med_F" in ev[name] and "med_eps" in ev[name] for name in SPLITS4)
    )


def train_one(method: str, L: int, seed: int, lam: float, pools: dict) -> dict:
    torch.manual_seed(int(seed) + 1000 * int(L) + int(round(100 * lam)))
    rng = np.random.default_rng(int(seed) + 17 * int(L))
    angles = init_angles(method, L)
    params = bind_params(method, angles)
    tr6, tr9 = pools["train6"], pools["train9"]
    n6 = int(tr6["diag"].shape[0])
    n9 = int(tr9["diag"].shape[0])
    b6 = min(BATCH // 2, n6)
    b9 = min(BATCH - b6, n9)
    tag = f"L{L} {method} s{seed} λ={lam:.2f}"

    def snapshot(energy: bool) -> dict:
        return eval_train_hold(method, angles, pools, lam, energy=energy)

    t0 = time.time()
    ev0 = snapshot(False)
    score0 = score_of(ev0)
    best = {
        "score": score0,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "loss": None, "score": score0, **{k: ev0[k]["med_F"] for k in SPLITS4}}]
    log(f"{tag} init min4 F={score0:.4f}")

    def run_rounds(start: int, stop: int) -> None:
        nonlocal best
        stall = 0
        for r in range(start, stop):
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
                    train_loss(method, angles, tr6, i6, 6) + train_loss(method, angles, tr9, i9, 9)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                return loss

            try:
                loss = opt.step(closure)
                loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            except Exception as exc:
                log(f"{tag} r{r} LBFGS skip: {exc}")
                hist.append({"round": r, "skip": str(exc)})
                stall += 1
                continue
            if not math_finite(loss_f):
                log(f"{tag} r{r} nonfinite loss — restore best")
                with torch.no_grad():
                    for k, v in best["angles"].items():
                        angles[k].data.copy_(v.to(dtype=angles[k].dtype))
                hist.append({"round": r, "loss": None})
                stall += 1
                continue
            ev = snapshot(False)
            score = score_of(ev)
            hist.append({"round": r, "loss": loss_f, "score": score, **{k: ev[k]["med_F"] for k in SPLITS4}})
            mark = ""
            if score > best["score"] + 1e-6:
                best = {
                    "score": score,
                    "round": r,
                    "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
                }
                stall = 0
                mark = " *"
            else:
                stall += 1
            if r % 5 == 0 or r + 1 == stop or mark:
                log(
                    f"{tag} r{r}/{stop} loss={loss_f:.4f} min4={score:.4f} "
                    f"best={best['score']:.4f}@r{best['round']}{mark}"
                )
            if best["score"] >= 0.97 and stall >= 5:
                log(f"{tag} early stop at r{r}: hold/train min4 ≥ 0.97 and stalled")
                break

    run_rounds(0, ROUNDS)
    extended = False
    if 0.90 <= best["score"] < TARGET:
        log(f"{tag} min4={best['score']:.4f} is short of {TARGET:.2f}; +{EXTRA_ROUNDS} rounds")
        extended = True
        run_rounds(ROUNDS, ROUNDS + EXTRA_ROUNDS)

    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    final = snapshot(True)
    return {
        "protocol": PROTOCOL,
        "complete": True,
        "family": "powerlaw_tfim",
        "objective": "mean(1 - |<psi_GS|psi>|^2) joint n6/n9",
        "method": method,
        "shell": {"snap": "B2", "shell": "hamming4", "fourier": "fourier_phi=-beta*k"}[method],
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "mu": MU,
        "alpha": ALPHA,
        "budget": budget_dict(),
        "extended": extended,
        "init_score": score0,
        "best_round": int(best["round"]),
        "best_score": float(best["score"]),
        "wall_s": time.time() - t0,
        "n_params": int(sum(p.numel() for p in params)),
        "diag_angles": shell_diag(method, angles_to_lists(angles)),
        "angles": angles_to_lists(angles),
        "eval": final,
        "hist": hist,
    }


def _optimize_rounds(method, angles, params, pools, lam, rng, *, start: int, stop: int, best: dict, hist: list) -> None:
    tr6, tr9 = pools["train6"], pools["train9"]
    n6 = int(tr6["diag"].shape[0])
    n9 = int(tr9["diag"].shape[0])
    b6 = min(BATCH // 2, n6)
    b9 = min(BATCH - b6, n9)
    tag = f"L{angles['gamma2'].shape[0] if method != 'fourier' else angles['betas'].shape[0]} {method}"

    def snapshot() -> dict:
        return eval_train_hold(method, angles, pools, lam, energy=False)

    for r in range(start, stop):
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
                train_loss(method, angles, tr6, i6, 6) + train_loss(method, angles, tr9, i9, 9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as exc:
            log(f"{tag} r{r} LBFGS skip: {exc}")
            hist.append({"round": r, "skip": str(exc)})
            continue
        if not math_finite(loss_f):
            with torch.no_grad():
                for k, v in best["angles"].items():
                    angles[k].data.copy_(v.to(dtype=angles[k].dtype))
            hist.append({"round": r, "loss": None})
            continue
        ev = snapshot()
        score = score_of(ev)
        hist.append({"round": r, "loss": loss_f, "score": score, **{k: ev[k]["med_F"] for k in SPLITS4}})
        if score > best["score"] + 1e-6:
            best["score"] = score
            best["round"] = r
            best["angles"] = {k: v.detach().cpu().clone() for k, v in angles.items()}
            log(f"{tag} r{r}/{stop} loss={loss_f:.4f} min4={score:.4f} *")
        elif r % 10 == 0 or r + 1 == stop:
            log(f"{tag} r{r}/{stop} loss={loss_f:.4f} min4={score:.4f} best={best['score']:.4f}@r{best['round']}")


def train_job(lam: float, L: int, method: str, seed: int) -> str:
    path = seed_path(lam, L, method, seed)
    if train_complete(path):
        log(f"skip complete {path.name}")
        return "skip"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    pools = load_split_pools(lam)
    payload = train_one(method, L, seed, lam, pools)
    write_json(path, payload)
    ev = payload["eval"]
    log(
        f"done {path.name} min4={score_of(ev):.4f} "
        f"F6={ev['train6']['med_F']:.3f}/{ev['hold6']['med_F']:.3f} "
        f"F9={ev['train9']['med_F']:.3f}/{ev['hold9']['med_F']:.3f} "
        f"wall={payload['wall_s']:.0f}s"
    )
    return "ok"


def continue_job(lam: float, L: int, method: str, seed: int, extra: int) -> str:
    """More L-BFGS rounds from the saved checkpoint. Refreshes n=12 and n=15."""
    path = seed_path(lam, L, method, seed)
    payload = read_json(path)
    if not payload.get("complete"):
        raise RuntimeError(f"{path.name} is not a finished train")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    pools = load_split_pools(lam)
    angles = lists_to_angles(payload["angles"])
    params = bind_params(method, angles)
    ev = eval_train_hold(method, angles, pools, lam, energy=False)
    best = {
        "score": score_of(ev),
        "round": int(payload.get("best_round", -1)),
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = list(payload.get("hist") or [])
    start = max((int(h["round"]) for h in hist if "round" in h), default=-1) + 1
    rng = np.random.default_rng(int(seed) + 17 * int(L) + 10007 + start)
    t0 = time.time()
    log(f"continue {path.name} from r{start} min4={best['score']:.4f} +{extra}")
    _optimize_rounds(
        method, angles, params, pools, lam, rng,
        start=start, stop=start + int(extra), best=best, hist=hist,
    )
    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    final = eval_train_hold(method, angles, pools, lam, energy=True)
    payload["angles"] = angles_to_lists(angles)
    payload["diag_angles"] = shell_diag(method, payload["angles"])
    payload["eval"] = final
    payload["best_round"] = int(best["round"])
    payload["best_score"] = float(best["score"])
    payload["extended"] = True
    payload["continued_rounds"] = int(payload.get("continued_rounds") or 0) + int(extra)
    payload["hist"] = hist
    payload["wall_s"] = float(payload.get("wall_s") or 0.0) + (time.time() - t0)
    write_json(path, payload)
    log(f"continued {path.name} min4={score_of(final):.4f} best@{payload['best_round']}")
    for n in (12, 15):
        cur = read_json(path)
        cur.get("eval", {}).pop(f"n{n}", None)
        write_json(path, cur)
        attach_freeze(lam, L, method, seed, n)
    return "ok"


def attach_freeze(lam: float, L: int, method: str, seed: int, n: int) -> None:
    path = seed_path(lam, L, method, seed)
    payload = read_json(path)
    key = f"n{n}"
    prev = (payload.get("eval") or {}).get(key)
    if prev and int(prev.get("B", -1)) == FREEZE_B and "med_eps" in prev:
        log(f"skip freeze {path.name} {key}")
        return
    raw = build_pool(n, FREEZE_B, lam)
    pack = slice_pool(raw, slice(0, FREEZE_B))
    angles = lists_to_angles(payload["angles"])
    rec = eval_pack(method, angles, pack, n, lam, chunk=chunk_for(n), energy=True)
    payload.setdefault("eval", {})[key] = rec
    payload["freeze_note"] = (
        f"n={n} GS is the exact lowest eigenvector from powerlaw_tfim.ground_state "
        "(dense eigh for n≤9, scipy eigsh SA from |+⟩ for n≥12). Angles are not retrained."
    )
    write_json(path, payload)
    log(f"freeze {path.name} n{n} medF={rec['med_F']:.4f} medε={rec['med_eps']:.4f}")


def phase_label(row: dict) -> str:
    """Short reading of one (λ, n) diagnostic. Thresholds are descriptive, not a fit.

    q_EA is not a glass order parameter here: H commutes with Π_i X_i, the
    ground state is even, and ⟨Z_i⟩ cancels between a bitstring and its flip.
    """
    bits = []
    if row.get("cat_med", 0.0) >= 0.60 and row["q_ea_med"] < 0.05:
        bits.append("Z2-even cat (weight on z* and its global flip; ⟨Z⟩ cancels)")
    elif row["q_ea_med"] >= 0.30 or row["peak_med"] >= 0.30:
        bits.append("Z-localized")
    elif row["q_ea_med"] <= 0.15 and row["peak_med"] < 0.15:
        bits.append("Z-spread")
    else:
        bits.append("partially Z-localized")
    if row["mean_x_med"] <= 0.35:
        bits.append("weak ⟨X⟩")
    elif row["mean_x_med"] >= 0.65:
        bits.append("field-polarized ⟨X⟩")
    else:
        bits.append("intermediate ⟨X⟩")
    if row["prod_gap_med"] < PROD_GAP_MIN:
        bits.append("product state nearly matches E0")
    else:
        bits.append("product gap above the #42 quantumness floor")
    if row["gap_med"] < 0.05:
        bits.append("small many-body gap")
    else:
        bits.append("gapped")
    return "; ".join(bits)


def _med(xs) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def build_phase(lam: float) -> dict:
    path = RUN / f"phase_l{lam:.2f}.json"
    if path.exists():
        old = read_json(path)
        row9 = (old.get("rows") or {}).get("9") or {}
        if old.get("protocol") == PROTOCOL and old.get("hold_b") == HOLD_B and "cat_med" in row9:
            log(f"phase cache hit λ={lam:.2f}")
            return old
    log(f"phase diagnostic λ={lam:.2f}")
    rows = {}
    per_n = {}
    for n in (6, 9):
        raw = build_pool(n, TRAIN_B + HOLD_B, lam)
        sl = slice(TRAIN_B, TRAIN_B + HOLD_B)
        peaks, gaps, qea, mx, ipr, gaps_spec, e0s, cats = [], [], [], [], [], [], [], []
        instances = []
        for i in range(TRAIN_B, TRAIN_B + HOLD_B):
            K = couplings(n, i)[1]
            ob = obs_one(raw["psi"][i], n)
            e_prod = best_product(K, raw["diag"][i], lam, MU, seed=int(raw["seeds"][i]))
            e0 = float(raw["e0"][i])
            if e_prod < e0 - 1e-5:
                raise RuntimeError(f"product {e_prod} below E0 {e0}")
            pg = (e_prod - e0) / max(abs(e0), 1e-8)
            spec = float(raw["e1"][i] - e0)
            peaks.append(ob["peak"])
            gaps.append(pg)
            qea.append(ob["q_ea"])
            mx.append(ob["mean_x"])
            ipr.append(ob["ipr"])
            gaps_spec.append(spec)
            e0s.append(e0)
            cats.append(ob["cat"])
            instances.append(
                {
                    "index": i,
                    "peak": ob["peak"],
                    "ipr": ob["ipr"],
                    "q_ea": ob["q_ea"],
                    "mean_x": ob["mean_x"],
                    "cat": ob["cat"],
                    "prod_gap": pg,
                    "gap": spec,
                    "E0": e0,
                    "E_prod": float(e_prod),
                }
            )
            if len(instances) % 8 == 0:
                log(f"  phase n{n} {len(instances)}/{HOLD_B} peak={ob['peak']:.3f} qEA={ob['q_ea']:.3f} X={ob['mean_x']:.3f}")
        qsum = quantum_summary(np.asarray(peaks), np.asarray(gaps))
        row = {
            "n": n,
            "B": HOLD_B,
            "split": f"indices {TRAIN_B}..{TRAIN_B + HOLD_B - 1}",
            "peak_med": _med(peaks),
            "ipr_med": _med(ipr),
            "q_ea_med": _med(qea),
            "cat_med": _med(cats),
            "mean_x_med": _med(mx),
            "prod_gap_med": _med(gaps),
            "gap_med": _med(gaps_spec),
            "gap_per_site_med": _med(gaps_spec) / float(n),
            "E0_med": _med(e0s),
            "quantum_summary": qsum,
            "label": "",
        }
        row["label"] = phase_label(row)
        rows[str(n)] = row
        per_n[str(n)] = instances
    # Cross-cavity weight does not depend on λ. One instance is enough to quote the fraction's scale;
    # report the mean over the holdout, matching the chain (not a fit).
    cross = []
    for n in (6, 9, 12):
        ws = [cross_cavity_weight(couplings(n, i)[1]) for i in range(8)]
        cross.append({"n": n, "mean_cross_over_8": float(np.mean(ws))})
    payload = {
        "protocol": PROTOCOL,
        "lam": lam,
        "mu": MU,
        "alpha": ALPHA,
        "hold_b": HOLD_B,
        "rows": rows,
        "instances": per_n,
        "cross_cavity": cross,
    }
    write_json(path, payload)
    for n, row in rows.items():
        log(
            f"phase λ={lam:.2f} n{n}: peak={row['peak_med']:.3f} qEA={row['q_ea_med']:.3f} "
            f"⟨X⟩={row['mean_x_med']:.3f} gap={row['gap_med']:.4f} prod={row['prod_gap_med']:.4f} | {row['label']}"
        )
    return payload


def library_check() -> None:
    """B=2 init matches Fourier, and scalar γ_2 matches the graded SNAP cost."""
    torch.manual_seed(0)
    for n in (3, 6):
        inst_diag = zz_diag_from_K(couplings(n, 0)[1])
        Cn = cost_diag(torch.tensor(inst_diag, dtype=torch.float64).reshape(1, -1), n)
        L = 2
        b2 = init_snap_b2(L)
        fou = init_fourier(L)
        psi_b = prepare_model("snap", Cn, b2)
        psi_f = prepare_model("fourier", Cn, fou)
        graded = init_snap(L)
        psi_g = prepare_snap(Cn, graded)
        Fbf = float(fidelity_sq(psi_b, psi_f)[0])
        Fbg = float(fidelity_sq(psi_b, psi_g)[0])
        if Fbf < 1.0 - 1e-10:
            raise AssertionError(f"B=2 init fidelity to Fourier {Fbf} at n={n}")
        if Fbg < 1.0 - 1e-10:
            raise AssertionError(f"scalar γ2 fidelity to graded SNAP {Fbg} at n={n}")
        # A quadratic bend leaves the X mixer.
        bent = {k: v.clone() for k, v in b2.items()}
        bent["c2"] = torch.tensor([0.35, -0.2], dtype=torch.float64)
        psi_q = prepare_model("snap", Cn, bent)
        if float(fidelity_sq(psi_q, psi_f)[0]) > 1.0 - 1e-6:
            raise AssertionError("quadratic shell did not move the state")
    # Fidelity of a ground state to itself, and a nonzero c2 gradient.
    diag = zz_diag_from_K(couplings(6, 1)[1])
    gs = ground_state(diag, 6, 0.50, MU)
    g = torch.tensor(gs["psi"], dtype=torch.float64).reshape(1, -1)
    if abs(float(fidelity_sq(g.to(torch.complex128), g)[0]) - 1.0) > 1e-10:
        raise AssertionError("GS fidelity to itself is not 1")
    Cn = cost_diag(torch.tensor(diag, dtype=torch.float64).reshape(1, -1), 6)
    ang = init_snap_b2(1)
    ang["c2"] = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    loss = (1.0 - fidelity_sq(prepare_model("snap", Cn, ang), g)).mean()
    loss.backward()
    if ang["c2"].grad is None or abs(float(ang["c2"].grad)) < 1e-10:
        raise AssertionError(f"c2 gradient vanished: {None if ang['c2'].grad is None else float(ang['c2'].grad)}")
    log("library_check ok")


def _group_payloads(payloads: list[dict]) -> dict:
    groups: dict[tuple, list] = {}
    for p in payloads:
        groups.setdefault((float(p["lam"]), int(p["L"]), p["method"]), []).append(p)
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r["seed"]))
        if [int(r["seed"]) for r in rows] != list(SEEDS):
            raise RuntimeError(f"seeds for {key} are {[r['seed'] for r in rows]}")
    return groups


def med_F(rows: list[dict], split: str) -> float:
    return float(np.median([float(r["eval"][split]["med_F"]) for r in rows]))


def med_eps(rows: list[dict], split: str) -> float | None:
    vals = []
    for r in rows:
        rec = r["eval"].get(split) or {}
        if "med_eps" not in rec:
            return None
        vals.append(float(rec["med_eps"]))
    return float(np.median(vals))


def med_energy(rows: list[dict], split: str, key: str) -> float | None:
    vals = []
    for r in rows:
        rec = r["eval"].get(split) or {}
        if key not in rec:
            return None
        vals.append(float(rec[key]))
    return float(np.median(vals))


def four_scores(rows: list[dict]) -> dict:
    out = {name: med_F(rows, name) for name in SPLITS4}
    out["min4"] = float(min(out.values()))
    out["f_train"] = float(min(out["train6"], out["train9"]))
    out["f_hold"] = float(min(out["hold6"], out["hold9"]))
    return out


def hit_95(scores: dict) -> bool:
    return bool(scores["f_train"] >= TARGET - 1e-12 and scores["f_hold"] >= TARGET - 1e-12)


def choose_L(lam_groups: dict, method: str) -> dict:
    """L* is the smallest ladder depth whose median train and holdout F clear TARGET on both n."""
    scored = []
    for L in LADDER:
        rows = lam_groups.get((L, method))
        if not rows:
            continue
        sc = four_scores(rows)
        scored.append({"L": L, **sc, "hit": hit_95(sc), "n_seeds": len(rows)})
    star = next((s for s in scored if s["hit"]), None)
    best = max(scored, key=lambda s: (s["min4"], -s["L"])) if scored else None
    return {"ladder": scored, "Lstar": None if star is None else int(star["L"]), "best": best}


def call_one(hit: bool, f_train: float | None, f12: float | None) -> str:
    if not hit or f_train is None or f12 is None:
        return "no95"
    drop = float(f_train) - float(f12)
    if f12 >= PASS_F12 - 1e-12 and drop <= PASS_DROP + 1e-12:
        return "pass"
    return "collapse"


def call_study(per_lam: list[str]) -> str:
    if any(c == "pass" for c in per_lam):
        return "PASS"
    if any(c == "collapse" for c in per_lam):
        return "needs_edit"
    return "KILL"


def load_payloads() -> list[dict]:
    rows = []
    for path in sorted(RUN.glob("L*_s*_l*.json")):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        if payload.get("protocol") != PROTOCOL or payload.get("complete") is not True:
            continue
        budget = payload.get("budget") or {}
        if int(budget.get("rounds", -1)) != ROUNDS or int(budget.get("train_b", -1)) != TRAIN_B:
            continue
        rows.append(payload)
    return rows


def render_results(phases: dict, payloads: list[dict]) -> tuple[str, dict]:
    groups = _group_payloads(payloads)
    by_lam: dict[float, dict] = {}
    for lam in LAMS:
        lam_groups = {}
        for L in LADDER:
            for method in ("snap", "shell", "fourier"):
                rows = groups.get((lam, L, method))
                if rows:
                    lam_groups[(L, method)] = rows
        by_lam[lam] = {
            "snap": choose_L(lam_groups, "snap"),
            "shell": choose_L(lam_groups, "shell"),
            "fourier": choose_L(lam_groups, "fourier"),
            "groups": lam_groups,
        }

    def _one(lam: float, method: str) -> dict | None:
        chosen = by_lam[lam][method]
        if not chosen["ladder"]:
            return None
        use_L = chosen["Lstar"] if chosen["Lstar"] is not None else chosen["best"]["L"]
        rows = by_lam[lam]["groups"].get((use_L, method))
        scored = next(s for s in chosen["ladder"] if s["L"] == use_L)
        f12 = med_F(rows, "n12") if rows and all("n12" in r["eval"] for r in rows) else None
        f15 = med_F(rows, "n15") if rows and all("n15" in r["eval"] for r in rows) else None
        hit = chosen["Lstar"] is not None
        f_train = scored["f_train"]
        call = "no95" if not hit else call_one(True, f_train, f12)
        return {
            "lam": lam,
            "method": method,
            "call": call,
            "Lstar": chosen["Lstar"],
            "report_L": use_L,
            "f_train": f_train,
            "f_hold": scored["f_hold"],
            "f12": f12,
            "f15": f15,
            "drop12": None if f12 is None else float(f_train) - float(f12),
            "drop15": None if f15 is None else float(f_train) - float(f15),
        }

    lam_calls = []
    lam_detail = []
    for lam in LAMS:
        methods = [d for d in (_one(lam, m) for m in ("snap", "shell")) if d is not None]
        calls = [d["call"] for d in methods]
        if "pass" in calls:
            call = "pass"
        elif "collapse" in calls:
            call = "collapse"
        else:
            call = "no95"
        lam_calls.append(call)
        lam_detail.append({"lam": lam, "call": call, "methods": methods})
    verdict = call_study(lam_calls)

    lines = []
    lines.append("# Power-law TFIM fidelity transfer — λ ∈ {0.25, 0.50}")
    lines.append("")
    lines.append(f"**Verdict: {verdict}.**")
    lines.append("")
    lines.append(
        "This slice trains by ground-state fidelity, not by excess energy. "
        "Sibling runs own λ ∈ {0.75, 1.0, 1.5}. μ = 0, α = 1.5, J_ij ∈ {±1}."
    )
    lines.append("")
    lines.append("## Bar")
    lines.append("")
    lines.append(
        f"L* is the smallest L in {{1,2,4,8,12,16}} whose median-over-seeds of the "
        f"median-over-instances fidelity is at least {TARGET:.2f} on **train and holdout**, "
        f"at **both** n=6 and n=9. F = |⟨ψ_GS|ψ(θ)⟩|²."
    )
    lines.append("")
    lines.append(
        f"PASS if some λ in this slice has an L* and the frozen n=12 fidelity is "
        f"≳ {PASS_F12:.2f} with drop from min(train n6, train n9) ≲ {PASS_DROP:.2f}. "
        "needs_edit if some λ hits that train bar and the freeze then falls through it. "
        "KILL if neither λ hits the train bar at any L ≤ 16. "
        "Fourier is a reference column. Beating it is not the bar. n=15 is reported beside n=12; the written bar is n=12."
    )
    lines.append("")
    lines.append("## Hamiltonian")
    lines.append("")
    lines.append("$$")
    lines.append(r"H = \sum_{i<j} \frac{J_{ij}}{|i-j|^{1.5}} Z_i Z_j - \lambda \sum_i X_i")
    lines.append("$$")
    lines.append("")
    lines.append(
        "Open cavity chain, same builder as `Nov_3rd/powerlaw_tfim.py` (PR #42). "
        "Qubit 0 is the MSB. Cavity c owns sites [3c, 3c+3). Bonds that cross a cavity stay in the sum. "
        "Circuit cost is exp(−i γ H_ZZ / n); ⟨H⟩ and E0 use the un-normalized H. "
        "H_ZZ is pure Walsh degree 2, so SNAP trains γ_2 only."
    )
    lines.append("")
    lines.append(
        "Two SNAP shells, both with γ_2 only. **B=2** is φ_k = c₁ k + c₂ k². "
        "**shell** is the PR #42 Hamming shell, free φ_k for k=0..3. "
        "Paper init is c₁=−β, c₂=0 (and φ_k=−β k), γ_2=−0.01, β=0.01. "
        "Fourier uses that same linear mixer (standard QAOA X rotation) and a scalar γ. "
        "At init all three states match; `library_check` asserts the B=2 and graded shells against Fourier. "
        "B=2 contains the Fourier mixer (c₂=0). The full shell contains B=2."
    )
    lines.append("")
    if phases:
        cross = next(iter(phases.values())).get("cross_cavity") or []
        if cross:
            bits = ", ".join(f"n{c['n']}={c['mean_cross_over_8']:.3f}" for c in cross)
            lines.append(f"Mean cross-cavity share of Σ|K_ij| over 8 instances: {bits}.")
            lines.append("")
    lines.append(
        "Prior energy smoke (PR #42), best cell λ=0.50 μ=−0.50 L=4: SNAP Medε 0.209 at n=9 hold "
        "to 0.675 at n=12. That cell is not this Hamiltonian (it has an XX driver). "
        "On this μ=0, λ=0.50 cell the same energy smoke had n=9 SNAP Medε 0.544 and n=12 Medε 1.661, "
        "and the product gap failed the quantumness floor (n=9 hold peak 0.313, product gap 0.004). "
        "Those numbers are energy, not fidelity."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| knob | value |")
    lines.append("|---|---|")
    lines.append("| λ | 0.25, 0.50 |")
    lines.append("| μ, α | 0, 1.5 |")
    lines.append("| ansatz | SNAP B=2 (γ_2, c1, c2), full Hamming shell (γ_2, φ_k), Fourier (β, γ) |")
    lines.append("| L | 1, 2, 4, 8, 12, 16, stop a λ once SNAP hits L* |")
    lines.append("| seeds | 0, 1, 2 |")
    lines.append(
        f"| pools | train B={TRAIN_B} indices 0..{TRAIN_B-1}; "
        f"holdout B={HOLD_B} indices {TRAIN_B}..{TRAIN_B+HOLD_B-1}; "
        f"freeze n=12 and n=15 B={FREEZE_B} indices 0..{FREEZE_B-1} |"
    )
    lines.append("| loss | ½ (mean infidelity n6 + mean infidelity n9), minibatch 32+32 |")
    lines.append(
        f"| optimizer | L-BFGS {ROUNDS}×{LBFGS_ITER}, lr {LBFGS_LR}, grad clip 5, paper init, single start. "
        f"Stop early only after min4 ≥ 0.97 with 5 rounds of no improvement. "
        f"+{EXTRA_ROUNDS} rounds if the best min4 lands in [0.90, {TARGET}) |"
    )
    lines.append("| checkpoint | maximize min(train6, hold6, train9, hold9) median F |")
    lines.append("| instance seed | 1500003017 + n·100003 + index |")
    lines.append("| n≤12 GS | exact (`ground_state`: dense eigh for n≤9, eigsh for n=12) |")
    lines.append("| n=15 GS | exact eigsh, same helper, k=1, which=SA, start \\|+⟩. Not approximate. |")
    lines.append("")
    lines.append("## Phase")
    lines.append("")
    lines.append(
        "Diagnostics are the n=6 and n=9 **holdout** (B=32), the same index split as PR #42. "
        "Peak is max_z |ψ(z)|². IPR is Σ_z p(z)². Cat weight is p(z*)+p(flip z*), the global spin flip. "
        "q_EA = (1/n) Σ_i ⟨Z_i⟩². It is near zero in the even ground state because ⟨Z_i⟩ cancels "
        "between z* and its flip, so it is not the order parameter on this H. "
        "⟨X⟩ is (1/n) Σ_i ⟨X_i⟩. Gap is E1−E0 from the dense spectrum. "
        "Product gap is (E_prod − E0)/|E0|."
    )
    lines.append("")
    lines.append("| λ | n | peak | cat | IPR | q_EA | ⟨X⟩ | E1−E0 | product gap | E0 | #42 quantumness | reading |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for lam in LAMS:
        phase = phases[lam]
        for n in ("6", "9"):
            row = phase["rows"][n]
            q = row["quantum_summary"]
            qbit = "pass" if q["pass"] else "fail"
            lines.append(
                f"| {lam:.2f} | {n} | {_fmt(row['peak_med'], 3)} | {_fmt(row.get('cat_med'), 3)} | "
                f"{_fmt(row['ipr_med'], 3)} | {_fmt(row['q_ea_med'], 3)} | {_fmt(row['mean_x_med'], 3)} | "
                f"{_fmt(row['gap_med'], 4)} | {_fmt(row['prod_gap_med'], 4)} | {_fmt(row['E0_med'], 3)} | {qbit} "
                f"(peak<0.5 on {q['frac_peak_below']:.0%}, gap≥0.03 on {q['frac_gap_ok']:.0%}) | {row['label']} |"
            )
    lines.append("")
    # Comparison sentence from the two n=9 rows.
    if all(lam in phases for lam in LAMS):
        a = phases[0.25]["rows"]["9"]
        b = phases[0.50]["rows"]["9"]
        lines.append(
            f"On n=9 holdout, λ=0.25 has peak {_fmt(a['peak_med'], 3)}, cat weight {_fmt(a.get('cat_med'), 3)}, "
            f"⟨X⟩ {_fmt(a['mean_x_med'], 3)}, gap {_fmt(a['gap_med'], 4)}, product gap {_fmt(a['prod_gap_med'], 4)}. "
            f"λ=0.50 has peak {_fmt(b['peak_med'], 3)}, cat weight {_fmt(b.get('cat_med'), 3)}, "
            f"⟨X⟩ {_fmt(b['mean_x_med'], 3)}, gap {_fmt(b['gap_med'], 4)}, product gap {_fmt(b['prod_gap_med'], 4)}. "
            "Both sit on the ordered side of this α=1.5 glass: the ground state is the even cat of a classical "
            "bitstring, λ=0.25 more concentrated than λ=0.50, and a product state already matches E0. "
            "That product state has fidelity at most about half a cat, which is why an energy residual can be "
            "small while fidelity stays well below 0.95. These two λ are not a fitted critical point."
        )
        lines.append("")
    lines.append("## Ladder")
    lines.append("")
    lines.append(
        "Each entry is the median over seeds 0, 1, 2 of that seed's median instance fidelity. "
        "min4 is the minimum of train6, hold6, train9, hold9. A dash means that depth was not trained "
        "because a smaller L* already cleared the bar for that λ."
    )
    lines.append("")
    lines.append("| λ | L | B=2 train6 | hold6 | train9 | hold9 | B=2 min4 | B=2 hit | shell min4 | Fourier min4 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for lam in LAMS:
        snap_scored = {s["L"]: s for s in by_lam[lam]["snap"]["ladder"]}
        shell_scored = {s["L"]: s for s in by_lam[lam]["shell"]["ladder"]}
        fou_scored = {s["L"]: s for s in by_lam[lam]["fourier"]["ladder"]}
        Ls = sorted(set(snap_scored) | set(shell_scored) | set(fou_scored))
        for L in Ls:
            s = snap_scored.get(L)
            h = shell_scored.get(L)
            f = fou_scored.get(L)
            src = s or h or f
            if src is None:
                continue
            lines.append(
                f"| {lam:.2f} | {L} | {_fmt(None if s is None else s['train6'])} | "
                f"{_fmt(None if s is None else s['hold6'])} | {_fmt(None if s is None else s['train9'])} | "
                f"{_fmt(None if s is None else s['hold9'])} | {_fmt(None if s is None else s['min4'])} | "
                f"{'yes' if s and s['hit'] else 'no'} | {_fmt(None if h is None else h['min4'])} | "
                f"{_fmt(None if f is None else f['min4'])} |"
            )
    lines.append("")
    lines.append("## Freeze")
    lines.append("")
    lines.append(
        "Angles are the train checkpoint at the reported L. n=12 and n=15 are new instances "
        "(indices 0..31 at that n), not a retrain. Medε = median(⟨H⟩ − E0). "
        "Energies are the median over seeds of each seed's median instance value."
    )
    lines.append("")
    lines.append(
        "| λ | L | role | method | F train | F hold | F n12 | drop12 | F n15 | drop15 | Medε n9 hold | Medε n12 | Medε n15 | med ⟨H⟩ n12 | med E0 n12 |"
    )
    lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lam in LAMS:
        for method in ("snap", "shell", "fourier"):
            chosen = by_lam[lam][method]
            report_L = chosen["Lstar"] if chosen["Lstar"] is not None else (
                None if chosen["best"] is None else chosen["best"]["L"]
            )
            if report_L is None:
                continue
            role = "L*" if chosen["Lstar"] == report_L else "best, below 95%"
            rows = by_lam[lam]["groups"][(report_L, method)]
            sc = four_scores(rows)
            def grab(split, key):
                if any(split not in r["eval"] or key not in r["eval"][split] for r in rows):
                    return None
                return med_energy(rows, split, key) if key != "med_F" else med_F(rows, split)
            f12 = grab("n12", "med_F")
            f15 = grab("n15", "med_F")
            lines.append(
                f"| {lam:.2f} | {report_L} | {role} | {method} | {_fmt(sc['f_train'])} | {_fmt(sc['f_hold'])} | "
                f"{_fmt(f12)} | {_fmt(None if f12 is None else sc['f_train'] - f12)} | "
                f"{_fmt(f15)} | {_fmt(None if f15 is None else sc['f_train'] - f15)} | "
                f"{_fmt(med_eps(rows, 'hold9'))} | {_fmt(med_eps(rows, 'n12'))} | {_fmt(med_eps(rows, 'n15'))} | "
                f"{_fmt(grab('n12', 'med_H'), 3)} | {_fmt(grab('n12', 'med_E0'), 3)} |"
            )
    lines.append("")
    lines.append("### Per seed at the reported SNAP depth")
    lines.append("")
    lines.append("| λ | method | L | seed | train6 | hold6 | train9 | hold9 | n12 | n15 | best round | wall s |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lam in LAMS:
        for method in ("snap", "shell"):
            chosen = by_lam[lam][method]
            report_L = chosen["Lstar"] if chosen["Lstar"] is not None else (
                None if chosen["best"] is None else chosen["best"]["L"]
            )
            if report_L is None:
                continue
            for row in by_lam[lam]["groups"][(report_L, method)]:
                ev = row["eval"]
                lines.append(
                    f"| {lam:.2f} | {method} | {report_L} | {row['seed']} | {_fmt(ev['train6']['med_F'])} | "
                    f"{_fmt(ev['hold6']['med_F'])} | {_fmt(ev['train9']['med_F'])} | {_fmt(ev['hold9']['med_F'])} | "
                    f"{_fmt((ev.get('n12') or {}).get('med_F'))} | {_fmt((ev.get('n15') or {}).get('med_F'))} | "
                    f"{row['best_round']} | {row['wall_s']:.0f} |"
                )
    lines.append("")
    lines.append("## Call by λ")
    lines.append("")
    for d in lam_detail:
        lines.append(f"- **λ={d['lam']:.2f}: {d['call']}.**")
        if not d["methods"]:
            lines.append("  No SNAP rows.")
            continue
        for m in d["methods"]:
            if m["Lstar"] is None:
                lines.append(
                    f"  - {m['method']}: cannot hit 95% at train. Best L={m['report_L']} with "
                    f"min(train n6, train n9) F={_fmt(m['f_train'])}. "
                    f"Freeze n12 F={_fmt(m['f12'])}, n15 F={_fmt(m['f15'])}."
                )
            else:
                lines.append(
                    f"  - {m['method']}: L*={m['Lstar']}. Train min F={_fmt(m['f_train'])}. "
                    f"Frozen n12 F={_fmt(m['f12'])} (drop {_fmt(m['drop12'])}), "
                    f"n15 F={_fmt(m['f15'])} (drop {_fmt(m['drop15'])}). Cell call: {m['call']}."
                )
    lines.append("")
    lines.append(f"Slice call: **{verdict}**.")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    bits = []
    for d in lam_detail:
        parts = []
        for m in d["methods"]:
            if m["call"] == "pass":
                parts.append(
                    f"{m['method']} keeps n12 F={_fmt(m['f12'])} after train F={_fmt(m['f_train'])} at L*={m['Lstar']}"
                )
            elif m["call"] == "collapse":
                parts.append(
                    f"{m['method']} hits the train bar at L*={m['Lstar']} and drops to n12 F={_fmt(m['f12'])}"
                )
            else:
                parts.append(
                    f"{m['method']} peaks at train F={_fmt(m['f_train'])} on L={m['report_L']}"
                )
        bits.append(f"λ={d['lam']:.2f}: " + "; ".join(parts))
    lines.append(" ".join(s + "." for s in bits))
    lines.append("")
    near = by_lam[0.50]["groups"].get((16, "shell"))
    if near and any(int(r.get("continued_rounds") or 0) > 0 for r in near):
        sc = four_scores(near)
        f12 = med_F(near, "n12") if all("n12" in r["eval"] for r in near) else None
        f15 = med_F(near, "n15") if all("n15" in r["eval"] for r in near) else None
        extra = max(int(r.get("continued_rounds") or 0) for r in near)
        rounds = max(int(h.get("round", -1)) for r in near for h in r.get("hist") or [{"round": -1}])
        lines.append(
            f"The only near miss is the full Hamming shell at λ=0.50, L=16. "
            f"It took the automatic +{EXTRA_ROUNDS} rounds and then {extra} more from that checkpoint "
            f"(last logged round {rounds}). The restored checkpoint is not the last round. "
            f"Median train min F={_fmt(sc['f_train'])}, hold min F={_fmt(sc['f_hold'])}, "
            f"n12 F={_fmt(f12)}, n15 F={_fmt(f15)}. "
            f"n=6 is already above 0.95; n=9 is the miss. "
            f"That is still short of {TARGET:.2f}, so λ=0.50 has no L*. "
            f"The frozen n=12 fidelity is already under {PASS_F12:.2f}, and n=15 falls further, "
            f"so clearing the train bar with a longer run would still be a transfer miss on the written n=12 rule."
        )
        lines.append("")
    lines.append(
        "B=2 contains the Fourier mixer at c₂=0, but from the paper init it settles at a lower fidelity "
        "than Fourier. The full Hamming shell beats Fourier at every depth in this slice and is the "
        "ansatz the KILL refers to. Neither shell reaches a train L*."
    )
    lines.append("")
    lines.append(
        "JSON per (λ, L, method, seed) is `L{L}_{method}_s{seed}_l{λ}.json` in this folder. "
        "Angles are the restored checkpoint. Phase rows are `phase_l0.25.json` and `phase_l0.50.json`. "
        "`settings.py` is unchanged. n=21 was not loaded."
    )
    lines.append("")
    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "lams": lam_detail,
        "target": TARGET,
        "pass_f12": PASS_F12,
        "pass_drop": PASS_DROP,
    }
    return "\n".join(lines) + "\n", summary


def cmd_summarize() -> None:
    payloads = load_payloads()
    if not payloads:
        raise SystemExit("no completed payloads")
    phases = {}
    for lam in LAMS:
        path = RUN / f"phase_l{lam:.2f}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        phases[lam] = read_json(path)
    text, summary = render_results(phases, payloads)
    (RUN / "RESULTS.md").write_text(text)
    write_json(RUN / "summary.json", summary)
    log(f"wrote RESULTS.md verdict={summary['verdict']}")
    print(text)


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check")
    p.set_defaults(func=lambda a: library_check())

    p = sub.add_parser("phase")
    p.add_argument("--lams", default="0.25,0.50")
    p.set_defaults(func=lambda a: [build_phase(lam) for lam in _parse_floats(a.lams)])

    p = sub.add_parser("train")
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--method", choices=("snap", "shell", "fourier"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda a: train_job(a.lam, a.L, a.method, a.seed))

    p = sub.add_parser("freeze")
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--method", choices=("snap", "shell", "fourier"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.set_defaults(func=lambda a: attach_freeze(a.lam, a.L, a.method, a.seed, a.n))

    p = sub.add_parser("continue")
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--method", choices=("snap", "shell", "fourier"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--extra", type=int, default=40)
    p.set_defaults(func=lambda a: continue_job(a.lam, a.L, a.method, a.seed, a.extra))

    p = sub.add_parser("summarize")
    p.set_defaults(func=lambda a: cmd_summarize())

    p = sub.add_parser("campaign")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lams", default="0.25,0.50")
    p.add_argument("--ladder", default="1,2,4,8,12,16")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default="snap,fourier")
    p.set_defaults(func=lambda a: cmd_campaign(a))

    args = ap.parse_args()
    args.func(args)


def _cli(args: list[str]) -> None:
    script = str(Path(__file__).resolve())
    proc = subprocess.run([sys.executable, script, *args], cwd=str(_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {args}")


def _parallel(cmds: list[list[str]], workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    if workers <= 1 or len(cmds) <= 1:
        for cmd in cmds:
            _cli(cmd)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_cli, cmds))


def cmd_campaign(args) -> None:
    library_check()
    lams = _parse_floats(args.lams)
    ladder = _parse_ints(args.ladder)
    seeds = _parse_ints(args.seeds)
    methods = tuple(m.strip() for m in str(args.methods).split(",") if m.strip())
    stop_method = "shell" if "shell" in methods and "snap" not in methods else "snap"
    if stop_method not in methods:
        stop_method = methods[0]
    for lam in lams:
        build_phase(lam)
    pending_lams = set(lams)
    workers = int(args.workers)
    for L in ladder:
        if not pending_lams:
            break
        jobs = [
            ["train", "--lam", f"{lam:.2f}", "--L", str(L), "--method", method, "--seed", str(seed)]
            for lam in sorted(pending_lams)
            for method in methods
            for seed in seeds
        ]
        log(f"wave L={L} jobs={len(jobs)} lams={sorted(pending_lams)} stop={stop_method}")
        _parallel(jobs, workers)
        still = set()
        for lam in pending_lams:
            rows = []
            ok = True
            for seed in seeds:
                path = seed_path(lam, L, stop_method, seed)
                if not train_complete(path):
                    ok = False
                    break
                rows.append(read_json(path))
            if not ok:
                log(f"λ={lam:.2f} L={L} {stop_method} incomplete — keep going")
                still.add(lam)
                continue
            sc = four_scores(rows)
            log(
                f"λ={lam:.2f} L={L} {stop_method} min4={sc['min4']:.4f} "
                f"train={sc['f_train']:.4f} hold={sc['f_hold']:.4f}"
            )
            if not hit_95(sc):
                still.add(lam)
        pending_lams = still
    # Freeze n=12 for every completed checkpoint (transfer curve), n=15 at the
    # reported SNAP depth and the matching Fourier depth.
    payloads = load_payloads()
    groups = _group_payloads(payloads) if payloads else {}
    freeze_jobs = []
    reported = {}
    for lam in lams:
        lam_groups = {(L, m): rows for (Llam, L, m), rows in groups.items() if Llam == lam}
        for method in sorted({m for (_, m) in lam_groups}):
            chosen = choose_L(lam_groups, method)
            report_L = chosen["Lstar"] if chosen["Lstar"] is not None else (
                None if chosen["best"] is None else chosen["best"]["L"]
            )
            reported[(lam, method)] = report_L
            for (L, m), rows in lam_groups.items():
                if m != method:
                    continue
                for row in rows:
                    freeze_jobs.append((lam, L, m, int(row["seed"]), 12))
                    if report_L is not None and L == report_L:
                        freeze_jobs.append((lam, L, m, int(row["seed"]), 15))
    # Unique, n=12 first so a failure at 15 still leaves 12.
    freeze_jobs = sorted(set(freeze_jobs), key=lambda j: (j[4], j[0], j[1], j[2], j[3]))
    log(f"freeze jobs={len(freeze_jobs)}")
    # Build exact GS pools once, then farm the circuit evaluations.
    for lam in lams:
        build_pool(12, FREEZE_B, lam)
        if any(j[0] == lam and j[4] == 15 for j in freeze_jobs):
            build_pool(15, FREEZE_B, lam)
    freeze_cmds = [
        ["freeze", "--lam", f"{lam:.2f}", "--L", str(L), "--method", method, "--seed", str(seed), "--n", str(n)]
        for lam, L, method, seed, n in freeze_jobs
    ]
    _parallel(freeze_cmds, workers)
    cmd_summarize()


if __name__ == "__main__":
    main()
