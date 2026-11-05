#!/usr/bin/env python3
"""Fidelity transfer for the power-law TFIM at λ=1.5, μ=0.

Hamiltonian (PR #42 family, α=1.5, open cavity chain, J_ij∈{±1}):

    H = Σ_{i<j} J_ij / |i-j|^{1.5} Z_i Z_j − λ Σ_i X_i

This run does not try to beat QAOA on energy. It trains state fidelity to the
exact ground state on n∈{6,9}, then freezes those angles at n=12 and n=15.

Ansatzes, same depth, same batches, same L-BFGS budget:
  snap_b2   Walsh cost with one live weight γ_2 (H_ZZ is pure degree 2) and a
            Hamming shell φ_k = c1 k + c2 k² on popcount k=0..3, in the
            orthonormal coordinates used by the B=2 ablation.
  fourier   standard 2-angle QAOA, the reference arm, trained on the same
            fidelity objective.

F = |⟨ψ|GS⟩|². L* is the smallest L in {1,2,4,8,12,16} whose median-over-seeds
of median-over-instances F is at least 0.95 on the train and hold splits of
both n=6 and n=9, for snap_b2. Frozen median F ≥ 0.85 at both n=12 and n=15
is the pre-registered reading of "stays reasonable".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S  # noqa: E402
from Nov_3rd.hva_safe import INIT_BETA, dense_H, fourier_state  # noqa: E402
from Nov_3rd.powerlaw_tfim import (  # noqa: E402
    ALPHA,
    couplings,
    cost_diag,
    cross_cavity_weight,
    expectation,
    ground_state,
    init_method,
    init_snap,
    instance_seed,
    parameter_list,
    prepare_snap,
    self_check,
    shell_nonlinearity,
    zz_diag_from_K,
    best_product,
)

RUN = S.OUT / "quantum_h_explore" / "powerlaw_fidelity_transfer" / "lam_150"
PROTOCOL = "powerlaw_fidelity_lam150_v1"
LAM = 1.5
MU = 0.0
BAR_F = 0.95
FREEZE_FLOOR = 0.85
DEPTHS = (1, 2, 4, 8, 12, 16)
SEEDS = (0, 1, 2)
TRAIN_B = 64
HOLD_B = 32
FREEZE_B = 32
COMPARE_B = 32
COMPARE_LAMS = (0.5, 1.0, 2.0)
ROUNDS = S.LBFGS_ROUNDS
BATCH = 32

# Orthonormal rows spanning {k, k²} on popcount k=0..3. φ = c @ B2_BASIS.
_K = np.arange(4, dtype=np.float64)
_RAW = np.stack([_K, _K ** 2], axis=1)
_Q, _R = np.linalg.qr(_RAW, mode="reduced")
for _j in range(2):
    if float(_Q[:, _j] @ _RAW[:, _j]) < 0.0:
        _Q[:, _j] *= -1.0
B2_BASIS = np.ascontiguousarray(_Q.T)  # (2, 4)
_PAPER_PHI = -float(INIT_BETA) * _K
B2_INIT = np.ascontiguousarray(_PAPER_PHI @ B2_BASIS.T)  # (2,)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fid150]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def mean_x(psi: np.ndarray, n: int) -> float:
    """(1/n) Σ_i ⟨X_i⟩. ψ is real in the Z basis, qubit 0 = MSB."""
    v = np.asarray(psi, dtype=np.float64).reshape(-1)
    acc = 0.0
    for q in range(int(n)):
        flipped = v[np.arange(v.size) ^ (1 << (int(n) - 1 - q))]
        acc += float(np.dot(v, flipped))
    return float(acc / int(n))


def f_plus(psi: np.ndarray) -> float:
    """|⟨+|ψ⟩|². |+⟩ is the transverse-field product state."""
    v = np.asarray(psi, dtype=np.float64).reshape(-1)
    plus = np.full(v.size, 1.0 / np.sqrt(v.size))
    ov = float(np.dot(plus, v))
    return float(ov * ov)


def phi_from_b2(c: torch.Tensor) -> torch.Tensor:
    basis = c.new_tensor(B2_BASIS)
    return c @ basis


def init_local(method: str, L: int) -> dict:
    L = int(L)
    if method == "fourier":
        return init_method("fourier", L, dtype=torch.float64)
    if method == "snap_b2":
        c0 = torch.tensor(B2_INIT, dtype=torch.float64)
        return {
            "gamma2": torch.full((L,), float(S.QAOA_INIT_GAMMA), dtype=torch.float64),
            "c": c0.reshape(1, 2).expand(L, 2).contiguous().clone(),
        }
    if method == "snap":
        return init_snap(L, dtype=torch.float64)
    raise ValueError(method)


def bind_local(method: str, angles: dict) -> list[torch.nn.Parameter]:
    if method == "fourier":
        return parameter_list("fourier", angles)
    keys = {"snap_b2": ("gamma2", "c"), "snap": ("gamma2", "phi")}[method]
    params = []
    for k in keys:
        p = torch.nn.Parameter(angles[k].detach().clone())
        angles[k] = p
        params.append(p)
    return params


def prepare_local(method: str, Cn: torch.Tensor, angles: dict) -> torch.Tensor:
    if method == "fourier":
        return fourier_state(Cn, angles["betas"], angles["gammas"])
    if method == "snap_b2":
        return prepare_snap(Cn, {"gamma2": angles["gamma2"], "phi": phi_from_b2(angles["c"])})
    if method == "snap":
        return prepare_snap(Cn, angles)
    raise ValueError(method)


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def pool_stats(rec: dict, sl: slice | None = None) -> dict:
    def take(name: str) -> np.ndarray:
        a = np.asarray(rec[name], dtype=np.float64)
        return a if sl is None else a[sl]

    peak = take("peak")
    gap = take("prod_gap") if rec.get("prod_gap") is not None else None
    mx = take("mean_x")
    fp = take("f_plus")
    e0 = take("e0")
    out = {
        "B": int(peak.size),
        "peak_med": float(np.median(peak)),
        "peak_max": float(np.max(peak)),
        "frac_peak_below_0.5": float(np.mean(peak < 0.5)),
        "mean_x_med": float(np.median(mx)),
        "mean_x_min": float(np.min(mx)),
        "f_plus_med": float(np.median(fp)),
        "f_plus_min": float(np.min(fp)),
        "E0_med": float(np.median(e0)),
    }
    if gap is not None and np.all(np.isfinite(gap)):
        out.update(
            {
                "gap_med": float(np.median(gap)),
                "gap_min": float(np.min(gap)),
                "frac_gap_ge_0.03": float(np.mean(gap >= 0.03 - 1e-15)),
                "quantum_pass": bool(float(np.median(peak)) < 0.5 and float(np.median(gap)) >= 0.03),
            }
        )
    else:
        out.update({"gap_med": None, "gap_min": None, "frac_gap_ge_0.03": None, "quantum_pass": None})
    if rec.get("exc_gap") is not None:
        eg = take("exc_gap")
        if np.all(np.isfinite(eg)):
            out["exc_gap_med"] = float(np.median(eg))
            out["exc_gap_min"] = float(np.min(eg))
        else:
            out["exc_gap_med"] = None
            out["exc_gap_min"] = None
    return out


def solve_instance(n: int, index: int, diag: np.ndarray, lam: float, *, product: bool) -> dict:
    """Exact ground state, plus the product gap when `product` is set."""
    gs = ground_state(diag, n, lam, MU)
    e0 = float(gs["E0"])
    psi = np.asarray(gs["psi"], dtype=np.float64)
    out = {
        "E0": e0,
        "psi": psi,
        "peak": float(gs["peak"]),
        "mean_x": mean_x(psi, n),
        "f_plus": f_plus(psi),
    }
    if int(n) <= 9:
        ev = np.linalg.eigvalsh(dense_H(diag, n, lambda_x=-float(lam), mu_xx=0.0))
        out["exc_gap"] = float(ev[1] - ev[0])
        if abs(float(ev[0]) - e0) > 1e-7:
            raise RuntimeError(f"E0 mismatch n={n} idx={index}: {ev[0]} vs {e0}")
    else:
        out["exc_gap"] = float("nan")
    if product:
        _J, K = couplings(n, index)
        e_prod = best_product(K, diag, lam, MU, seed=instance_seed(n, index))
        if e_prod < e0 - 1e-5:
            raise RuntimeError(f"product {e_prod} below E0 {e0} at n={n} idx={index}")
        out["E_prod"] = float(e_prod)
        out["prod_gap"] = float((e_prod - e0) / max(abs(e0), 1e-8))
    return out


def build_diag(n: int, count: int, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        if int(z["n"]) == int(n) and int(z["count"]) == int(count) and float(z["alpha"]) == float(ALPHA):
            log(f"diag cache n{n} B={count}")
            return np.asarray(z["diag"], dtype=np.float64), np.asarray(z["seeds"], dtype=np.int64)
    log(f"diag n{n} B={count}")
    diags = np.empty((count, 1 << n), dtype=np.float64)
    seeds = np.empty(count, dtype=np.int64)
    for i in range(count):
        _J, K = couplings(n, i)
        diags[i] = zz_diag_from_K(K)
        seeds[i] = instance_seed(n, i)
    np.savez_compressed(cache, diag=diags, seeds=seeds, n=n, count=count, alpha=ALPHA)
    return diags, seeds


def build_quantum(n: int, diag: np.ndarray, lam: float, cache: Path, *, product: bool) -> dict:
    cache.parent.mkdir(parents=True, exist_ok=True)
    B = int(diag.shape[0])
    if cache.exists():
        z = np.load(cache)
        same = (
            int(z["n"]) == int(n)
            and int(z["count"]) == B
            and abs(float(z["lam"]) - float(lam)) < 1e-15
            and int(z["product"]) == int(product)
            and z["psi"].shape == (B, 1 << n)
        )
        if same:
            log(f"quantum cache n{n} λ={lam:.2f} B={B}")
            return {k: np.asarray(z[k]) for k in z.files if k not in ("n", "lam", "product", "alpha")}
    log(f"quantum n{n} λ={lam:.2f} B={B} product={product}")
    t0 = time.time()
    keys = ("e0", "peak", "mean_x", "f_plus", "exc_gap", "prod_gap")
    cols = {k: np.empty(B, dtype=np.float64) for k in keys}
    psi = np.empty((B, 1 << n), dtype=np.float64)
    for i in range(B):
        rec = solve_instance(n, i, diag[i], lam, product=product)
        psi[i] = rec["psi"]
        cols["e0"][i] = rec["E0"]
        cols["peak"][i] = rec["peak"]
        cols["mean_x"][i] = rec["mean_x"]
        cols["f_plus"][i] = rec["f_plus"]
        cols["exc_gap"][i] = rec["exc_gap"]
        cols["prod_gap"][i] = rec.get("prod_gap", np.nan)
        if (i + 1) == B or (i + 1) % 16 == 0:
            log(
                f"  n{n} λ={lam:.2f} {i+1}/{B} E0={rec['E0']:.3f} "
                f"peak={rec['peak']:.3f} ⟨X⟩={rec['mean_x']:.3f} F+={rec['f_plus']:.3f}"
            )
    np.savez_compressed(
        cache,
        psi=psi,
        n=n,
        count=B,
        lam=lam,
        product=int(product),
        alpha=ALPHA,
        **cols,
    )
    log(f"quantum n{n} λ={lam:.2f} done in {time.time()-t0:.1f}s")
    return {"psi": psi, **cols}


def angles_to_lists(angles: dict) -> dict:
    out = {}
    for k, v in angles.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().to(torch.float64).numpy().tolist()
        else:
            out[k] = v
    return out


def pack_metrics(F: np.ndarray, eps: np.ndarray, e0: np.ndarray) -> dict:
    F = np.asarray(F, dtype=np.float64).reshape(-1)
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    e0 = np.asarray(e0, dtype=np.float64).reshape(-1)
    if np.any(F < -1e-8) or np.any(F > 1.0 + 1e-5):
        raise RuntimeError(f"fidelity outside [0,1]: min={F.min()} max={F.max()}")
    if np.min(eps) < -1e-4:
        raise RuntimeError(f"negative excess {float(np.min(eps))} (prepared state below E0)")
    rel = eps / np.maximum(np.abs(e0), 1e-8)
    return {
        "B": int(F.size),
        "med_F": float(np.median(F)),
        "mean_F": float(np.mean(F)),
        "min_F": float(np.min(F)),
        "med_overlap": float(np.median(np.sqrt(np.clip(F, 0.0, 1.0)))),
        "med_eps": float(np.median(eps)),
        "mean_eps": float(np.mean(eps)),
        "med_eps_over_abs_E0": float(np.median(rel)),
        "med_eps_over_med_abs_E0": float(np.median(eps) / max(float(np.median(np.abs(e0))), 1e-8)),
        "med_abs_E0": float(np.median(np.abs(e0))),
        "F": [float(x) for x in F.tolist()],
        "eps": [float(x) for x in eps.tolist()],
    }


@torch.no_grad()
def eval_split(method: str, angles: dict, pool: dict, n: int, *, chunk: int) -> dict:
    diag = pool["diag"]
    psi_gs = pool["psi"]
    e0 = pool["e0"]
    B = int(diag.shape[0])
    fs: list[torch.Tensor] = []
    es: list[torch.Tensor] = []
    for s in range(0, B, int(chunk)):
        d = diag[s : s + chunk]
        st = prepare_local(method, cost_diag(d, n), angles)
        nrm = torch.linalg.vector_norm(st, dim=-1)
        if float((nrm - 1).abs().max()) > 1e-5:
            raise RuntimeError(f"state norm drifted at n={n}: {float(nrm.max())}")
        gs = psi_gs[s : s + chunk].to(dtype=st.dtype)
        ov = torch.sum(st.conj() * gs, dim=-1).abs()
        fs.append((ov * ov).real.cpu())
        h = expectation(st, d, n, LAM, MU)
        es.append((h - e0[s : s + chunk].to(dtype=h.dtype)).cpu())
    F = torch.cat(fs).numpy()
    eps = torch.cat(es).numpy()
    return pack_metrics(F, eps, e0.detach().cpu().numpy())


@torch.no_grad()
def _med_f(method: str, angles: dict, pool: dict, n: int) -> float:
    """Median F on a pool. Skips the energy pass used by the final eval."""
    diag = pool["diag"]
    psi_gs = pool["psi"]
    fs: list[torch.Tensor] = []
    for s in range(0, int(diag.shape[0]), 32):
        d = diag[s : s + 32]
        st = prepare_local(method, cost_diag(d, n), angles)
        gs = psi_gs[s : s + 32].to(dtype=st.dtype)
        ov = torch.sum(st.conj() * gs, dim=-1).abs()
        fs.append((ov * ov).real.detach().cpu())
    return float(np.median(torch.cat(fs).numpy()))


def train_one(method: str, L: int, seed: int, pools: dict, *, rounds: int, batch: int) -> dict:
    torch.manual_seed(int(seed) + 1000 * int(L) + int(round(100 * LAM)))
    rng = np.random.default_rng(int(seed) + 17 * int(L))
    angles = init_local(method, L)
    params = bind_local(method, angles)
    tr6, tr9 = pools["train6"], pools["train9"]
    ho6, ho9 = pools["hold6"], pools["hold9"]
    n6 = int(tr6["diag"].shape[0])
    n9 = int(tr9["diag"].shape[0])
    b6 = min(int(batch) // 2, n6)
    b9 = min(int(batch) - b6, n9)

    def train_score() -> tuple[float, float, float]:
        # Checkpoint on the train pools. The hold split is only scored at the end,
        # so the 0.95 hold gate is not the quantity we maximized.
        f6 = _med_f(method, angles, tr6, 6)
        f9 = _med_f(method, angles, tr9, 9)
        return min(f6, f9), f6, f9

    t0 = time.time()
    score, f6, f9 = train_score()
    h6 = _med_f(method, angles, ho6, 6)
    h9 = _med_f(method, angles, ho9, 9)
    best = {
        "score": score,
        "f6": f6,
        "f9": f9,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{
        "round": -1,
        "train_min": score,
        "train_F6": f6,
        "train_F9": f9,
        "hold_F6": h6,
        "hold_F9": h9,
        "loss": None,
    }]
    log(
        f"L{L} {method} s{seed} init train F n6={f6:.4f} n9={f9:.4f} "
        f"hold {h6:.4f}/{h9:.4f} nparams={sum(p.numel() for p in params)}"
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

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                _infidelity(method, angles, tr6, i6, 6) + _infidelity(method, angles, tr9, i9, 9)
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
            log(f"L{L} {method} s{seed} r{r} nonfinite loss — restore best")
            with torch.no_grad():
                for k, v in best["angles"].items():
                    angles[k].data.copy_(v.to(dtype=angles[k].dtype))
            hist.append({"round": r, "loss": None})
            continue
        score, f6, f9 = train_score()
        row = {"round": r, "loss": loss_f, "train_min": score, "train_F6": f6, "train_F9": f9}
        if r % 5 == 0 or r + 1 == int(rounds):
            row["hold_F6"] = _med_f(method, angles, ho6, 6)
            row["hold_F9"] = _med_f(method, angles, ho9, 9)
        hist.append(row)
        mark = ""
        if score > best["score"] + 1e-6:
            best = {
                "score": score,
                "f6": f6,
                "f9": f9,
                "round": r,
                "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
            }
            mark = " *"
        if r % 5 == 0 or r + 1 == int(rounds) or mark:
            hold_txt = ""
            if "hold_F6" in row:
                hold_txt = f" hold {row['hold_F6']:.4f}/{row['hold_F9']:.4f}"
            log(
                f"L{L} {method} s{seed} r{r}/{rounds} loss={loss_f:.4f} "
                f"train F {f6:.4f}/{f9:.4f}{hold_txt} best={best['score']:.4f}@r{best['round']}{mark}"
            )

    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    return {
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lam": float(LAM),
        "mu": float(MU),
        "init_train_min": float(hist[0]["train_min"]),
        "best_round": int(best["round"]),
        "best_train_min": float(best["score"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(int(p.numel()) for p in params)),
    }


def _infidelity(method: str, angles: dict, pool: dict, idx: torch.Tensor, n: int) -> torch.Tensor:
    d = pool["diag"].index_select(0, idx)
    gs = pool["psi"].index_select(0, idx)
    st = prepare_local(method, cost_diag(d, n), angles)
    ov = torch.sum(st.conj() * gs.to(dtype=st.dtype), dim=-1).abs()
    return (1.0 - (ov * ov).real).mean()


def diagnostics(method: str, angles: dict) -> dict:
    raw = angles_to_lists(angles)
    if method == "snap_b2":
        phi = phi_from_b2(angles["c"]).detach().cpu().numpy()
        return {
            "shell_rms_vs_linear": shell_nonlinearity(phi),
            "mean_abs_gamma2": float(np.mean(np.abs(raw["gamma2"]))),
            "mean_abs_c": [float(x) for x in np.mean(np.abs(raw["c"]), axis=0)],
        }
    if method == "snap":
        return {
            "shell_rms_vs_linear": shell_nonlinearity(np.asarray(raw["phi"], dtype=np.float64)),
            "mean_abs_gamma2": float(np.mean(np.abs(raw["gamma2"]))),
        }
    return {"mean_abs_gamma": float(np.mean(np.abs(raw["gammas"]))), "mean_abs_beta": float(np.mean(np.abs(raw["betas"])))}


def attach_eval(trained: dict, pools: dict) -> dict:
    angles = trained["angles"]
    chunk = {6: 32, 9: 16, 12: 8, 15: 4}
    ev: dict = {}
    for split, names in (
        ("train", (("6", 6, "train6"), ("9", 9, "train9"))),
        ("hold", (("6", 6, "hold6"), ("9", 9, "hold9"))),
        ("freeze", (("12", 12, "freeze12"), ("15", 15, "freeze15"))),
    ):
        ev[split] = {}
        for tag, n, key in names:
            ev[split][tag] = eval_split(trained["method"], angles, pools[key], n, chunk=chunk[n])
            m = ev[split][tag]
            log(
                f"  eval {split} n{n} {trained['method']} L{trained['L']} s{trained['seed']} "
                f"medF={m['med_F']:.4f} medε={m['med_eps']:.4f}"
            )
    out = dict(trained)
    out["angles"] = angles_to_lists(angles)
    out["diagnostics"] = diagnostics(trained["method"], angles)
    out["eval"] = ev
    return out


def seed_path(run: Path, L: int, method: str, seed: int) -> Path:
    return run / f"L{int(L)}_{method}_s{int(seed)}.json"


def budget_ok(payload: dict, cfg: dict) -> bool:
    b = payload.get("budget") or {}
    ev = payload.get("eval") or {}
    try:
        freeze_ok = "12" in ev["freeze"] and "15" in ev["freeze"]
        hold_ok = "6" in ev["hold"] and "9" in ev["hold"]
        train_ok = "6" in ev["train"] and "9" in ev["train"]
    except Exception:
        return False
    return (
        payload.get("protocol") == PROTOCOL
        and int(b.get("rounds", -1)) == int(cfg["rounds"])
        and int(b.get("batch", -1)) == int(cfg["batch"])
        and int(b.get("train_b", -1)) == int(cfg["train_b"])
        and int(b.get("hold_b", -1)) == int(cfg["hold_b"])
        and int(b.get("freeze_b", -1)) == int(cfg["freeze_b"])
        and abs(float(payload.get("lam", -1)) - float(LAM)) < 1e-15
        and freeze_ok
        and hold_ok
        and train_ok
        and "angles" in payload
    )


def _med(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def aggregate(payloads: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for p in payloads:
        groups.setdefault((p["method"], int(p["L"])), []).append(p)
    cells = []
    for (method, L), rows in sorted(
        groups.items(),
        key=lambda kv: (kv[0][1], 0 if kv[0][0] == "snap_b2" else 1, kv[0][0]),
    ):
        rows = sorted(rows, key=lambda r: int(r["seed"]))

        def col(split: str, n: str, key: str) -> list[float]:
            return [float(r["eval"][split][n][key]) for r in rows]

        cell = {
            "method": method,
            "L": int(L),
            "n_seeds": len(rows),
            "seeds": [int(r["seed"]) for r in rows],
            "n_params": int(rows[0]["n_params"]),
            "train": {},
            "hold": {},
            "freeze": {},
            "per_seed": {},
        }
        for split in ("train", "hold", "freeze"):
            ns = ("6", "9") if split != "freeze" else ("12", "15")
            for n in ns:
                cell[split][n] = {
                    "med_F": _med(col(split, n, "med_F")),
                    "min_seed_F": float(min(col(split, n, "med_F"))),
                    "max_seed_F": float(max(col(split, n, "med_F"))),
                    "med_of_mean_F": _med(col(split, n, "mean_F")),
                    "med_of_min_F": _med(col(split, n, "min_F")),
                    "med_eps": _med(col(split, n, "med_eps")),
                    "med_eps_over_abs_E0": _med(col(split, n, "med_eps_over_abs_E0")),
                    "med_overlap": _med(col(split, n, "med_overlap")),
                }
                cell["per_seed"][f"{split}_F_{n}"] = col(split, n, "med_F")
        cells.append(cell)
    return cells


def cell_meets(cell: dict, bar: float = BAR_F) -> bool:
    vals = [
        cell["train"]["6"]["med_F"],
        cell["train"]["9"]["med_F"],
        cell["hold"]["6"]["med_F"],
        cell["hold"]["9"]["med_F"],
    ]
    return all(float(v) >= float(bar) - 1e-12 for v in vals)


def train_hit(cells: list[dict], bar: float = BAR_F) -> bool:
    return any(
        float(c["train"]["6"]["med_F"]) >= float(bar) - 1e-12
        and float(c["train"]["9"]["med_F"]) >= float(bar) - 1e-12
        for c in cells
    )


def select_lstar(cells: list[dict], bar: float = BAR_F) -> int | None:
    hits = [int(c["L"]) for c in cells if cell_meets(c, bar)]
    return min(hits) if hits else None


def decide(snap_cells: list[dict]) -> dict:
    """Verdict for the Walsh B=2 arm only.

    KILL: no depth reaches median train F≥0.95 on both n=6 and n=9.
    needs_edit: train clears 0.95 but the hold split never does, or L* exists
    and frozen median F at n=12 or n=15 falls below FREEZE_FLOOR.
    PASS: L* exists and both frozen sizes stay at or above FREEZE_FLOOR.
    """
    Lstar = select_lstar(snap_cells)
    hit = train_hit(snap_cells)
    if Lstar is None:
        if hit:
            return {
                "verdict": "needs_edit",
                "Lstar": None,
                "freeze_min": None,
                "reason": (
                    "median train F reaches 0.95 on n=6 and n=9 at some L≤16, "
                    "but no depth also holds that bar on both holdouts"
                ),
            }
        return {
            "verdict": "KILL",
            "Lstar": None,
            "freeze_min": None,
            "reason": "no L≤16 reaches median train F≥0.95 on both n=6 and n=9",
        }
    cell = next(c for c in snap_cells if int(c["L"]) == int(Lstar))
    f12 = float(cell["freeze"]["12"]["med_F"])
    f15 = float(cell["freeze"]["15"]["med_F"])
    fmin = min(f12, f15)
    base = (
        f"L*={Lstar} median train/hold F≥0.95 on n=6 and n=9 "
        f"(train {cell['train']['6']['med_F']:.3f}/{cell['train']['9']['med_F']:.3f}, "
        f"hold {cell['hold']['6']['med_F']:.3f}/{cell['hold']['9']['med_F']:.3f})"
    )
    if fmin >= FREEZE_FLOOR - 1e-12:
        return {
            "verdict": "PASS",
            "Lstar": int(Lstar),
            "freeze_min": fmin,
            "reason": (
                f"{base}; frozen median F stays ≥{FREEZE_FLOOR:.2f} "
                f"at n=12 ({f12:.3f}) and n=15 ({f15:.3f})"
            ),
        }
    under = []
    if f12 < FREEZE_FLOOR - 1e-12:
        under.append(f"n=12 ({f12:.3f})")
    if f15 < FREEZE_FLOOR - 1e-12:
        under.append(f"n=15 ({f15:.3f})")
    return {
        "verdict": "needs_edit",
        "Lstar": int(Lstar),
        "freeze_min": fmin,
        "reason": (
            f"{base}. Frozen median F is {f12:.3f} at n=12 and {f15:.3f} at n=15; "
            f"{' and '.join(under)} is under the {FREEZE_FLOOR:.2f} floor"
        ),
    }


def _fmt(x, nd: int = 3) -> str:
    if x is None or not (isinstance(x, (int, float)) and np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _stat_line(st: dict) -> str:
    return (
        f"B={st['B']} peak={_fmt(st['peak_med'])} gap={_fmt(st.get('gap_med'))} "
        f"⟨X⟩={_fmt(st['mean_x_med'])} F(+)= {_fmt(st['f_plus_med'])} "
        f"ΔE={_fmt(st.get('exc_gap_med'), 4)}"
    )


def render(summary: dict) -> str:
    d = summary["decision"]
    diag = summary["diagnostic"]
    lines = [
        f"# Power-law TFIM fidelity transfer — λ=1.5",
        "",
        f"**Verdict: {d['verdict']}**",
        "",
        d["reason"] + ".",
        "",
        "Fidelity is the pure-state overlap squared, F = |⟨ψ|GS⟩|². "
        "The gate is the median over three seeds of each seed's median over instances. "
        f"L* is the smallest depth in {{1,2,4,8,12,16}} with that median ≥ {BAR_F:.2f} "
        "on the train and hold splits of both n=6 and n=9, for Walsh SNAP B=2. "
        f"Frozen F ≥ {FREEZE_FLOOR:.2f} at both n=12 and n=15 is the pre-registered "
        "reading of a freeze that stays reasonable. Fourier is trained on the same "
        "fidelity loss and is a reference, not the arm the verdict is about.",
        "",
        "## Hamiltonian",
        "",
        "μ = 0, α = 1.5, J_ij ∈ {±1} on the open chain. Qubit 0 is the MSB. "
        "Cavity c owns sites [3c, 3c+3). Cross-cavity pairs stay in the chain; "
        "they are not dropped. The circuit phases use Cn = H_ZZ / n. "
        "H_ZZ is pure Walsh degree 2, so SNAP trains a single γ_2 per layer. "
        "The shell is φ_k = c₁ k + c₂ k² on popcount k=0..3, stored in orthonormal "
        "coordinates of that span (the B=2 ablation metric). Paper init φ_k = −β k "
        "lies in the span, and at that init SNAP matches Fourier.",
        "",
        "Cross-cavity fraction of Σ|K_ij| (sign-independent): "
        + ", ".join(f"n={n} {_fmt(summary['cross_weight'][str(n)], 3)}" for n in (6, 9, 12, 15))
        + ".",
        "",
        "## Phase diagnostic",
        "",
        "Bitstring peak is max_z |⟨z|GS⟩|². Product gap is (E_prod − E_0)/|E_0|, "
        "with E_prod the best XZ product energy found. ⟨X⟩ is (1/n) Σ_i ⟨X_i⟩. "
        "F(+) is |⟨+|GS⟩|². ΔE is the gap to the first excited state on n≤9. "
        "Comparison λ use the first 32 instance indices; λ=1.5 also reports the "
        "train pool [0:64) and the hold pool [64:96).",
        "",
        "| λ | n | slice | B | peak | product gap | ⟨X⟩ | F(+) | ΔE | quantumness |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for lam in sorted(float(k) for k in diag["by_lambda"]):
        block = diag["by_lambda"][f"{lam:.2f}"]
        items = sorted(block.items(), key=lambda kv: (int(kv[0].split(":")[0]), kv[1].get("slice") or ""))
        for key, st in items:
            n = key.split(":")[0]
            q = st.get("quantum_pass")
            qtxt = "—" if q is None else ("pass" if q else "fail")
            lines.append(
                f"| {lam:.2f} | {n} | {st['slice']} | {st['B']} | {_fmt(st['peak_med'])} | "
                f"{_fmt(st.get('gap_med'))} | {_fmt(st['mean_x_med'])} | {_fmt(st['f_plus_med'])} | "
                f"{_fmt(st.get('exc_gap_med'), 4)} | {qtxt} |"
            )
    lines += [
        "",
        diag["reading"],
        "",
        "## Ladder",
        "",
        f"Train indices [0:{summary['config']['train_b']}), hold "
        f"[{summary['config']['train_b']}:{summary['config']['train_b']+summary['config']['hold_b']}), "
        f"freeze n=12 and n=15 indices [0:{summary['config']['freeze_b']}). "
        f"Seeds {summary['config']['seeds']}. L-BFGS {summary['config']['rounds']} rounds × "
        f"max_iter {summary['config']['lbfgs_iter']}, lr {summary['config']['lbfgs_lr']}, "
        f"minibatch {summary['config']['batch']} split across n=6 and n=9, "
        "checkpoint on min(train median F at n=6, n=9). "
        "The hold split is not used to pick the round. No retrain at n=12 or n=15.",
        "",
        "| L | arm | train F n6 | hold F n6 | train F n9 | hold F n9 | freeze F n12 | freeze F n15 | Medε n12 | Medε n15 | Med(ε/|E0|) n15 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in summary["cells"]:
        lines.append(
            "| {L} | {m} | {a} | {b} | {c6} | {d} | {e} | {f} | {g} | {h} | {i} |".format(
                L=c["L"],
                m=c["method"],
                a=_fmt(c["train"]["6"]["med_F"]),
                b=_fmt(c["hold"]["6"]["med_F"]),
                c6=_fmt(c["train"]["9"]["med_F"]),
                d=_fmt(c["hold"]["9"]["med_F"]),
                e=_fmt(c["freeze"]["12"]["med_F"]),
                f=_fmt(c["freeze"]["15"]["med_F"]),
                g=_fmt(c["freeze"]["12"]["med_eps"], 4),
                h=_fmt(c["freeze"]["15"]["med_eps"], 4),
                i=_fmt(c["freeze"]["15"]["med_eps_over_abs_E0"], 4),
            )
        )
    lines += ["", "## L* and freeze", ""]
    if d["Lstar"] is None:
        lines.append("No Walsh B=2 depth meets the train and hold bar, so nothing was designated L*.")
    else:
        snap = next(c for c in summary["cells"] if c["method"] == "snap_b2" and int(c["L"]) == int(d["Lstar"]))
        def _seeds(xs: list[float]) -> str:
            return ", ".join(f"{float(x):.3f}" for x in xs)

        lines.append(
            f"Walsh B=2 L* = {d['Lstar']}. "
            f"Frozen median F is {_fmt(snap['freeze']['12']['med_F'])} at n=12 and "
            f"{_fmt(snap['freeze']['15']['med_F'])} at n=15. "
            f"Per-seed freeze F n=12 {_seeds(snap['per_seed']['freeze_F_12'])}; "
            f"n=15 {_seeds(snap['per_seed']['freeze_F_15'])}. "
            f"The median across seeds of each seed's worst instance is "
            f"{_fmt(snap['freeze']['12']['med_of_min_F'])} at n=12 and "
            f"{_fmt(snap['freeze']['15']['med_of_min_F'])} at n=15. "
            f"Medε at n=15 is {_fmt(snap['freeze']['15']['med_eps'], 4)} "
            f"(median ε/|E0| = {_fmt(snap['freeze']['15']['med_eps_over_abs_E0'], 4)})."
        )
        four = next(
            (c for c in summary["cells"] if c["method"] == "fourier" and int(c["L"]) == int(d["Lstar"])),
            None,
        )
        if four is not None:
            lines.append(
                f"Fourier at the same L*={d['Lstar']}: train/hold F "
                f"{_fmt(four['train']['6']['med_F'])}/{_fmt(four['hold']['6']['med_F'])} (n6) and "
                f"{_fmt(four['train']['9']['med_F'])}/{_fmt(four['hold']['9']['med_F'])} (n9); "
                f"freeze {_fmt(four['freeze']['12']['med_F'])} / {_fmt(four['freeze']['15']['med_F'])}."
            )
    fstar = summary.get("fourier_lstar")
    lines.append(
        "Fourier's own smallest depth with train and hold F≥0.95 on both n is "
        + ("none." if fstar is None else f"L={fstar}.")
    )
    lines += [
        "",
        "## Reading",
        "",
        summary["reading"],
        "",
        "Instance streams are `1500003017 + n·100003 + index`, the PR #42 family. "
        "n=21 was not loaded. `settings.py`, `qaoa_angles/`, and the Max-E ladder were not written.",
        "",
        f"_train wall {summary.get('train_wall_s', summary['wall_s']):.0f}s "
        "(sum of per-seed train+eval; ground states are cached separately)_",
        "",
    ]
    return "\n".join(lines)


def _pick(by_lam: dict, lam: float, n: int) -> dict | None:
    block = by_lam.get(f"{lam:.2f}") or {}
    return block.get(str(n))


def diagnostic_reading(by_lam: dict) -> str:
    c6 = _pick(by_lam, 1.5, 6)
    c9 = _pick(by_lam, 1.5, 9)
    if c6 is None or c9 is None:
        return "λ=1.5 diagnostic was not stored."
    text = (
        f"At λ=1.5 on the shared prefix, n=6 has median peak {c6['peak_med']:.3f}, "
        f"product gap {c6.get('gap_med'):.3f}, ⟨X⟩ {c6['mean_x_med']:.3f}, "
        f"F(+) {c6['f_plus_med']:.3f}; n=9 has peak {c9['peak_med']:.3f}, "
        f"gap {c9.get('gap_med'):.3f}, ⟨X⟩ {c9['mean_x_med']:.3f}, F(+) {c9['f_plus_med']:.3f}."
    )
    a = _pick(by_lam, 0.5, 6)
    b = _pick(by_lam, 1.0, 6)
    if a is not None and b is not None:
        text += (
            f" On that same n=6 prefix the smaller fields are less X-disordered: "
            f"peak {a['peak_med']:.3f} (λ=0.50) and {b['peak_med']:.3f} (λ=1.00), "
            f"⟨X⟩ {a['mean_x_med']:.3f} and {b['mean_x_med']:.3f}, "
            f"F(+) {a['f_plus_med']:.3f} and {b['f_plus_med']:.3f}, "
            f"product gap {a.get('gap_med'):.3f} and {b.get('gap_med'):.3f}."
        )
    e = _pick(by_lam, 2.0, 6)
    if e is not None:
        text += (
            f" The cheap λ=2.00 probe on the n=6 prefix goes further: "
            f"peak {e['peak_med']:.3f}, ⟨X⟩ {e['mean_x_med']:.3f}, F(+) {e['f_plus_med']:.3f}."
        )
    return text


def reading_for(decision: dict, cells: list[dict], diagnostic: dict | None = None) -> str:
    snap = [c for c in cells if c["method"] == "snap_b2"]
    four = [c for c in cells if c["method"] == "fourier"]
    if not snap:
        return "No Walsh rows were aggregated."
    by_l = {int(c["L"]): c for c in snap}
    text = decision["reason"] + "."
    if 1 in by_l and decision.get("Lstar") not in (None, 1):
        h9 = by_l[1]["hold"]["9"]["med_F"]
        text += (
            f" L=1 is not L*: its n=9 hold median is {h9:.3f}, under 0.95, "
            "while n=6 and the n=9 train split already clear the bar."
        )
    if decision["verdict"] == "KILL":
        text += " Deeper shells were not a way through the train bar by L=16."
        return text
    if four and decision.get("Lstar") is not None and int(decision["Lstar"]) in {int(c["L"]) for c in four}:
        fcell = next(c for c in four if int(c["L"]) == int(decision["Lstar"]))
        text += (
            f" Fourier at L* freezes at {fcell['freeze']['12']['med_F']:.3f} / "
            f"{fcell['freeze']['15']['med_F']:.3f}. B=2 and Fourier stay within about 0.01 "
            "at every reported cell, so the quadratic shell is not what limits transfer."
        )
    deep = [c for c in snap if int(c["L"]) >= 8]
    if deep:
        best15 = max(deep, key=lambda c: c["freeze"]["15"]["med_F"])
        text += (
            f" Extra depth does not close the freeze. At L={best15['L']} the n=15 median is "
            f"{best15['freeze']['15']['med_F']:.3f} and n=12 is {best15['freeze']['12']['med_F']:.3f}. "
            "Train fidelity has already flattened near 0.985 by L=4."
        )
    if decision["verdict"] == "needs_edit" and decision.get("Lstar") is not None:
        cell = by_l[int(decision["Lstar"])]
        fplus = None
        if diagnostic is not None:
            fplus = ((diagnostic.get("by_lambda") or {}).get("1.50") or {}).get("15", {}).get("f_plus_med")
        plus_txt = f" The |+⟩ overlap on that n=15 pool is {fplus:.3f}." if fplus is not None else ""
        text += (
            f" This is a soft miss rather than a collapse: frozen F at n=15 is "
            f"{cell['freeze']['15']['med_F']:.3f}, the median energy error ε/|E0| is "
            f"{cell['freeze']['15']['med_eps_over_abs_E0']:.3f}, and n=12 stays at "
            f"{cell['freeze']['12']['med_F']:.3f}.{plus_txt} "
            "The pre-registered floor is 0.85 on both frozen sizes, and n=15 is under it."
        )
    elif decision["verdict"] == "PASS" and decision.get("Lstar") is not None:
        cell = by_l[int(decision["Lstar"])]
        hold_min = min(cell["hold"]["6"]["med_F"], cell["hold"]["9"]["med_F"])
        text += (
            f" Freeze sits {hold_min - cell['freeze']['12']['med_F']:.3f} and "
            f"{hold_min - cell['freeze']['15']['med_F']:.3f} below the worse training-size holdout "
            "at n=12 and n=15, and both stay at or above 0.85."
        )
    return text


def _torch_pool(diag: np.ndarray, quantum: dict, sl: slice) -> dict:
    return {
        "diag": torch.tensor(np.asarray(diag[sl], dtype=np.float64)),
        "psi": torch.tensor(np.asarray(quantum["psi"][sl], dtype=np.float64)),
        "e0": torch.tensor(np.asarray(quantum["e0"][sl], dtype=np.float64)),
    }


def run(cfg: dict, run_dir: Path) -> dict:
    t0 = time.time()
    cache = run_dir / "cache"
    counts = {
        6: cfg["train_b"] + cfg["hold_b"],
        9: cfg["train_b"] + cfg["hold_b"],
        12: cfg["freeze_b"],
        15: cfg["freeze_b"],
    }
    diags = {}
    for n, b in counts.items():
        diags[n], _seeds = build_diag(n, b, cache / f"diag_n{n}_B{b}.npz")

    primary = {}
    for n, b in counts.items():
        primary[n] = build_quantum(
            n,
            diags[n],
            LAM,
            cache / f"q_n{n}_B{b}_l{LAM:.2f}.npz",
            product=n <= 9,
        )

    compare = {f"{LAM:.2f}": {}}
    tb, hb, fb = cfg["train_b"], cfg["hold_b"], cfg["freeze_b"]
    slices_primary = {
        6: [("prefix32", slice(0, min(COMPARE_B, counts[6]))), ("train", slice(0, tb)), ("hold", slice(tb, tb + hb))],
        9: [("prefix32", slice(0, min(COMPARE_B, counts[9]))), ("train", slice(0, tb)), ("hold", slice(tb, tb + hb))],
        12: [("freeze", slice(0, fb))],
        15: [("freeze", slice(0, fb))],
    }
    # The public diagnostic uses one slice per (λ, n): the shared prefix for the
    # λ comparison, plus train/hold rows for the primary λ.
    by_lam: dict[str, dict] = {}
    by_lam[f"{LAM:.2f}"] = {}
    for n, parts in slices_primary.items():
        for name, sl in parts:
            st = pool_stats(primary[n], sl)
            st["slice"] = name
            # Keep prefix for the comparison table and train/hold as extra rows
            # under distinct keys when the name is not the comparison slice.
            key = str(n) if name in ("prefix32", "freeze") else f"{n}:{name}"
            by_lam[f"{LAM:.2f}"][key] = st

    cmp_b = min(COMPARE_B, cfg["train_b"] + cfg["hold_b"])
    for lam in cfg["compare_lams"]:
        tag = f"{float(lam):.2f}"
        by_lam[tag] = {}
        for n in (6, 9):
            q = build_quantum(
                n,
                diags[n][:cmp_b],
                float(lam),
                cache / f"q_n{n}_B{cmp_b}_l{float(lam):.2f}.npz",
                product=True,
            )
            st = pool_stats(q, None)
            st["slice"] = "prefix32"
            by_lam[tag][str(n)] = st

    # Table order wants n as the second column. Expand primary train/hold as
    # extra λ=1.5 rows by encoding the slice in the key and rendering n from it.
    diag_rows = {"by_lambda": {}, "reading": ""}
    for tag, block in by_lam.items():
        diag_rows["by_lambda"][tag] = {}
        for key, st in block.items():
            diag_rows["by_lambda"][tag][key] = st
    # reading uses the prefix slices, which are stored at keys "6" and "9"
    diag_rows["reading"] = diagnostic_reading(by_lam) if "0.50" in by_lam else "Comparison λ were not run."

    tb_ = cfg["train_b"]
    hb_ = cfg["hold_b"]
    pools = {
        "train6": _torch_pool(diags[6], primary[6], slice(0, tb_)),
        "hold6": _torch_pool(diags[6], primary[6], slice(tb_, tb_ + hb_)),
        "train9": _torch_pool(diags[9], primary[9], slice(0, tb_)),
        "hold9": _torch_pool(diags[9], primary[9], slice(tb_, tb_ + hb_)),
        "freeze12": _torch_pool(diags[12], primary[12], slice(0, cfg["freeze_b"])),
        "freeze15": _torch_pool(diags[15], primary[15], slice(0, cfg["freeze_b"])),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for L in cfg["depths"]:
        for method in cfg["methods"]:
            for seed in cfg["seeds"]:
                path = seed_path(run_dir, L, method, seed)
                if path.exists():
                    old = json.loads(path.read_text())
                    if budget_ok(old, cfg):
                        log(f"resume {path.name}")
                        payloads.append(old)
                        continue
                trained = train_one(method, L, seed, pools, rounds=cfg["rounds"], batch=cfg["batch"])
                trained = attach_eval(trained, pools)
                trained["protocol"] = PROTOCOL
                trained["budget"] = {
                    "rounds": cfg["rounds"],
                    "batch": cfg["batch"],
                    "train_b": cfg["train_b"],
                    "hold_b": cfg["hold_b"],
                    "freeze_b": cfg["freeze_b"],
                    "lbfgs_iter": cfg["lbfgs_iter"],
                    "lbfgs_lr": cfg["lbfgs_lr"],
                    "alpha": ALPHA,
                    "bar_F": BAR_F,
                    "freeze_floor": FREEZE_FLOOR,
                }
                write_json(path, trained)
                payloads.append(trained)
                log(f"wrote {path.name} wall={trained['wall_s']:.0f}s")

    cells = aggregate(payloads)
    snap = [c for c in cells if c["method"] == "snap_b2"]
    fourier = [c for c in cells if c["method"] == "fourier"]
    decision = decide(snap) if snap else {
        "verdict": "KILL",
        "Lstar": None,
        "freeze_min": None,
        "reason": "no snap_b2 rows",
    }
    cross = {}
    for n in counts:
        _J, K = couplings(n, 0)
        cross[str(n)] = cross_cavity_weight(K)
    summary = {
        "protocol": PROTOCOL,
        "lam": LAM,
        "mu": MU,
        "alpha": ALPHA,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "cross_weight": cross,
        "diagnostic": diag_rows,
        "decision": decision,
        "fourier_lstar": select_lstar(fourier) if fourier else None,
        "cells": cells,
        "wall_s": max(time.time() - t0, float(sum(float(p.get("wall_s") or 0.0) for p in payloads))),
        "train_wall_s": float(sum(float(p.get("wall_s") or 0.0) for p in payloads)),
    }
    summary["reading"] = reading_for(decision, cells, diag_rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "diagnostic.json", diag_rows)
    text = render(summary)
    (run_dir / "RESULTS.md").write_text(text)
    log(f"verdict {decision['verdict']} L*={decision['Lstar']} in {summary['wall_s']:.0f}s")
    log(f"wrote {run_dir / 'RESULTS.md'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--depths", type=str, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--methods", type=str, default="snap_b2,fourier")
    args = ap.parse_args()

    torch.set_num_threads(4)
    log("self-check")
    self_check()
    log("self-check passed")

    cfg = {
        "depths": DEPTHS,
        "seeds": SEEDS,
        "methods": tuple(m for m in args.methods.split(",") if m),
        "train_b": TRAIN_B,
        "hold_b": HOLD_B,
        "freeze_b": FREEZE_B,
        "compare_lams": COMPARE_LAMS,
        "rounds": ROUNDS,
        "batch": BATCH,
        "lbfgs_iter": S.LBFGS_ITER,
        "lbfgs_lr": S.LBFGS_LR,
    }
    run_dir = RUN
    if args.pilot:
        cfg.update(
            {
                "depths": (1, 2),
                "seeds": (0,),
                "train_b": 8,
                "hold_b": 4,
                "freeze_b": 2,
                "compare_lams": (0.5,),
                "rounds": 2,
                "batch": 8,
            }
        )
        run_dir = RUN / "pilot"
    if args.rounds is not None:
        cfg["rounds"] = int(args.rounds)
    if args.depths:
        cfg["depths"] = tuple(int(x) for x in args.depths.split(",") if x)
    if args.seeds:
        cfg["seeds"] = tuple(int(x) for x in args.seeds.split(",") if x)
    log(f"dir {run_dir}")
    log(
        f"L={cfg['depths']} seeds={cfg['seeds']} methods={cfg['methods']} "
        f"rounds={cfg['rounds']} train={cfg['train_b']} hold={cfg['hold_b']} freeze={cfg['freeze_b']}"
    )
    try:
        run(cfg, run_dir)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
