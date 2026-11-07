#!/usr/bin/env python3
"""Five-seed shallow seal of Walsh SNAP against HVA at the SAFE point.

PR #36 (3 seeds, L in {2,4,8,16}) found SNAP ahead of HVA only at L=2 and L=4.
This seal reruns those two depths with seeds {0,1,2,3,4}. It does not train
L≥8 and it does not read or write the n=21 ladder.

Same Hamiltonian, ansatz, and L-BFGS schedule as Nov_3rd/scripts/run_hva_fairness_safe.py:

    H = diag(unsat) − 0.75 Σ_i X_i − 0.50 Σ_c Σ_{i<j in c} X_i X_j

    Fourier — (γ, β) per layer
    HVA     — (γ, β^X, β^XX) per layer
    SNAP    — Walsh γ_{w=1..4} + Hamming-shell φ_k, BinH mixer

Train n∈{6,9} only. Freeze n=12 (B=128) and n=15 (B=64). n=18 (B=16) is an
exploratory probe and is not part of the gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    LAMBDA_X,
    MU_XX,
    angles_to_lists,
    cn_from_unsat,
    expectation_H,
    ground_energy,
    init_angles,
    instance_eps,
    method_parameter_list,
    prepare_state,
    self_check,
    summarize_eps,
    walsh_power_by_degree,
)

RUN = S.OUT / "quantum_h_explore" / "hva_shallow_seal"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "hva", "snap")
TRAIN_NS = (6, 9)
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = {12: 128, 15: 64}
N18_B = 16
N18_BASE = 61_000_000_000
DEPTHS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
BEAT_R = 0.85
BOOT_DRAWS = 4000
BOOT_SEED = 5209
PROTOCOL = "hva_shallow_seal_r9"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[hva-shallow]", *a, flush=True)


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def _e0_job(row_n: tuple[np.ndarray, int]) -> float:
    row, n = row_n
    return float(ground_energy(np.asarray(row, dtype=np.float64), int(n)))


def ground_energies_parallel(E: np.ndarray, n: int, workers: int) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    if int(workers) <= 1 or E.shape[0] <= 1:
        out = np.empty(E.shape[0], dtype=np.float64)
        for i in range(E.shape[0]):
            out[i] = _e0_job((E[i], n))
            if (i + 1) == E.shape[0] or (i + 1) % 8 == 0:
                log(f"E0 n{n} {i+1}/{E.shape[0]} last={out[i]:.4f}")
        return out
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    jobs = [(E[i], n) for i in range(E.shape[0])]
    out = np.empty(E.shape[0], dtype=np.float64)
    t0 = time.time()
    with ctx.Pool(int(workers)) as pool:
        for i, val in enumerate(pool.imap(_e0_job, jobs, chunksize=1)):
            out[i] = val
            if (i + 1) == E.shape[0] or (i + 1) % 8 == 0:
                log(f"E0 n{n} {i+1}/{E.shape[0]} last={val:.4f} elapsed={time.time()-t0:.0f}s")
    return out


def ensure_e0(n: int, tag: str, E: np.ndarray, seeds: np.ndarray, workers: int) -> np.ndarray:
    RUN.mkdir(parents=True, exist_ok=True)
    cache = RUN / f"e0_n{n}_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["seeds"], seeds) and z["e0"].shape[0] == E.shape[0]:
            log(f"E0 cache hit n{n} {tag}")
            return np.asarray(z["e0"], dtype=np.float64)
    log(f"E0 n{n} {tag} B={E.shape[0]} workers={workers}")
    t0 = time.time()
    e0 = ground_energies_parallel(E, n, workers)
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above diag min at n{n} {tag}")
    np.savez_compressed(
        cache,
        seeds=seeds,
        e0=e0,
        n=n,
        tag=tag,
        lambda_x=LAMBDA_X,
        mu_xx=MU_XX,
    )
    log(f"E0 n{n} {tag} done in {time.time()-t0:.1f}s  medE0={np.median(e0):.4f}")
    return e0


def build_n18(count: int = N18_B) -> tuple[np.ndarray, np.ndarray]:
    """Satisfiable 4-SAT from the train-pack generator. Not a train index."""
    from explore_OCT29.lib.sat_gen import instances

    cache = RUN / f"probe_n18_B{count}.npz"
    if cache.exists():
        z = np.load(cache)
        E = np.asarray(z["E"], dtype=np.float64)
        seeds = np.asarray(z["seeds"]).astype(np.int64)
        if E.shape == (count, 1 << 18) and seeds.shape == (count,):
            log(f"n18 pack cache hit B={count}")
            return E, seeds
    log(f"generating n18 sat pack B={count} base={N18_BASE}")
    t0 = time.time()
    recs = instances(18, count, N18_BASE, compute_E=True)
    seeds = np.asarray([int(r["seed"]) for r in recs], dtype=np.int64)
    E = np.stack([np.asarray(r["E"], dtype=np.float64) for r in recs], axis=0)
    if E.shape != (count, 1 << 18):
        raise RuntimeError(f"n18 pack shape {E.shape}")
    if np.any(E.min(axis=1) != 0):
        raise RuntimeError("n18 pack contains an unsatisfiable formula")
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, E=E.astype(np.int16), seeds=seeds, base=N18_BASE, n=18)
    log(f"n18 pack written in {time.time()-t0:.1f}s seeds {int(seeds[0])}..{int(seeds[-1])}")
    return E, seeds


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_method(method: str, angles: dict, bundle: dict, *, chunk: int) -> dict:
    ang = _as_f64(angles)
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        use = 4 if int(n) >= 18 else chunk
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        eps = instance_eps(method, ang, Eu, e0t, n, chunk=use).detach().cpu().numpy()
        energy = eps + e0
        rec = summarize_eps(eps)
        rec.update(
            {
                "n": int(n),
                "seeds": seeds.tolist(),
                "eps": [float(x) for x in eps.tolist()],
                "e0_med": float(np.median(e0)),
                "med_energy": float(np.median(energy)),
            }
        )
        out[str(n)] = rec
    return out


def train_loss(method, angles, E, e0, idx, n: int) -> torch.Tensor:
    Eu = E.index_select(0, idx)
    e0b = e0.index_select(0, idx)
    Cn = cn_from_unsat(Eu, n)
    psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n)
    return ((h - e0b) / float(n)).mean()


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method: str, L: int, seed: int, bundles_t, hold_bundle, *, rounds: int, batch: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    angles = init_angles(method, L, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06 = bundles_t[6]
    E9, e09 = bundles_t[9]
    n6 = E6.shape[0]
    n9 = E9.shape[0]
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_method(method, angles, hold_bundle, chunk=16)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"L{L} {method} s{seed} init n9-hold Medε={init_med:.5f}")

    for r in range(rounds):
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
                train_loss(method, angles, E6, e06, i6, 6)
                + train_loss(method, angles, E9, e09, i9, 9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"L{L} {method} s{seed} r{r} LBFGS skip: {e}")
            hist.append({"round": r, "skip": str(e)})
            continue
        if not math_finite(loss_f):
            log(f"L{L} {method} s{seed} r{r} nonfinite loss — restore best")
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
        if r % 5 == 0 or r + 1 == rounds or mark:
            log(
                f"L{L} {method} s{seed} r{r}/{rounds} loss={loss_f:.5f} "
                f"holdMedε={med:.5f} best={best['med9']:.5f}@r{best['round']}{mark}"
            )

    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    return {
        "method": method,
        "L": L,
        "seed": seed,
        "init_hold_med9": init_med,
        "best_round": best["round"],
        "best_hold_med9": best["med9"],
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(p.numel() for p in params)),
    }


def seed_path(L: int, method: str, seed: int) -> Path:
    return RUN / f"L{L}_{method}_seed{seed}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf")
    return float(num) / float(den)


def decide_depth(r_med: float, seed_rs: list[float], ci_hi: float, n_seeds: int = 5) -> str:
    """PASS / needs_edit / KILL for one depth at frozen n12.

    PASS: median R_HVA ≤ 0.85, at least 3/5 seeds BEAT, and the instance
    bootstrap 95% interval ends at or below 0.85.
    KILL: median R_HVA > 1, or at least 3 seeds have R_HVA > 1 (HVA took them).
    needs_edit: anything in between (a soft lead).
    """
    if int(n_seeds) != 5 or len(seed_rs) != 5:
        return "INCOMPLETE"
    n_beat = sum(1 for r in seed_rs if float(r) <= BEAT_R)
    n_hva = sum(1 for r in seed_rs if float(r) > 1.0)
    if (not math.isfinite(r_med)) or r_med > 1.0 or n_hva >= 3:
        return "KILL"
    ci_ok = math.isfinite(ci_hi) and float(ci_hi) <= BEAT_R
    if r_med <= BEAT_R and n_beat >= 3 and ci_ok:
        return "PASS"
    return "needs_edit"


def decide_overall(by_depth: dict[int, str]) -> str:
    if set(by_depth) != set(DEPTHS):
        return "INCOMPLETE"
    vals = [by_depth[L] for L in DEPTHS]
    if any(v == "INCOMPLETE" for v in vals):
        return "INCOMPLETE"
    if any(v == "KILL" for v in vals):
        return "KILL"
    if any(v == "needs_edit" for v in vals):
        return "needs_edit"
    if all(v == "PASS" for v in vals):
        return "PASS"
    return "INCOMPLETE"


def _self_test_gate() -> None:
    five = [0.3, 0.4, 0.5, 0.6, 0.7]
    if decide_depth(0.50, five, 0.80) != "PASS":
        raise AssertionError("clear beat must PASS")
    if decide_depth(0.50, five, 0.90) != "needs_edit":
        raise AssertionError("CI above 0.85 is soft")
    soft_seeds = [0.40, 0.50, 0.90, 0.95, 0.99]
    if decide_depth(0.80, soft_seeds, 0.84) != "needs_edit":
        raise AssertionError("2/5 BEAT is soft even if the median ratio is under 0.85")
    if decide_depth(1.05, five, 0.80) != "KILL":
        raise AssertionError("median R>1 is KILL")
    hva_seeds = [1.1, 1.2, 1.3, 0.4, 0.5]
    if decide_depth(0.70, hva_seeds, 0.80) != "KILL":
        raise AssertionError("3/5 seeds with R>1 is KILL")
    if decide_overall({2: "PASS", 4: "PASS"}) != "PASS":
        raise AssertionError("both depths")
    if decide_overall({2: "PASS", 4: "needs_edit"}) != "needs_edit":
        raise AssertionError("one soft depth")
    if decide_overall({2: "PASS", 4: "KILL"}) != "KILL":
        raise AssertionError("one kill depth")
    if decide_overall({2: "PASS"}) != "INCOMPLETE":
        raise AssertionError("L=4 required")


def paired_counts(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 1:
        raise ValueError("paired shape")
    delta = a - b
    win = int(np.sum(delta < 0))
    tie = int(np.sum(delta == 0))
    loss = int(np.sum(delta > 0))
    return {"win": win, "tie": tie, "loss": loss, "n": int(a.size)}


def _median_ratio(methods: list[np.ndarray], bases: list[np.ndarray]) -> float:
    med_m = float(np.median([float(np.median(a)) for a in methods]))
    med_b = float(np.median([float(np.median(b)) for b in bases]))
    return ratio(med_m, med_b)


def bootstrap_ratio(eps_method, eps_base, *, n_boot: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict:
    """Instance bootstrap of the median-over-seeds ratio. Seeds stay fixed."""
    methods = [np.asarray(a, dtype=np.float64).reshape(-1) for a in eps_method]
    bases = [np.asarray(b, dtype=np.float64).reshape(-1) for b in eps_base]
    if len(methods) != len(bases) or len(methods) < 2:
        raise ValueError("bootstrap seeds")
    width = methods[0].size
    for a, b in zip(methods, bases):
        if a.shape != b.shape or int(a.size) != width:
            raise ValueError("bootstrap instance axis")
    point = _median_ratio(methods, bases)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    n_seed = len(methods)
    for t in range(int(n_boot)):
        ms, bs = [], []
        for s in range(n_seed):
            idx = rng.integers(0, width, size=width)
            ms.append(methods[s][idx])
            bs.append(bases[s][idx])
        draws[t] = _median_ratio(ms, bs)
    ok = draws[np.isfinite(draws)]
    lo, hi = np.quantile(ok, [0.025, 0.975])
    return {
        "R": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(ok.size),
        "n_seeds": int(n_seed),
        "n_inst": int(width),
        "seed": int(seed),
        "mode": "instance",
    }


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _shell_stats(rows: list[dict]) -> dict:
    """How far SNAP leaves the linear X mixer, and whether γ_w stay equal."""
    rms, gstd = [], []
    k = np.arange(4, dtype=np.float64)
    A = np.stack([np.ones(4), k], axis=1)
    for rec in rows:
        if rec.get("method") != "snap":
            continue
        phi = np.asarray(rec["angles"]["phi"], dtype=np.float64)
        gw = np.asarray(rec["angles"]["gamma_w"], dtype=np.float64)
        for ell in range(phi.shape[0]):
            coef, *_ = np.linalg.lstsq(A, phi[ell], rcond=None)
            resid = phi[ell] - A @ coef
            rms.append(float(np.sqrt(np.mean(resid ** 2))))
            gstd.append(float(np.std(gw[ell])))
    return {
        "shell_rms_med": float(np.median(rms)) if rms else None,
        "gamma_w_std_med": float(np.median(gstd)) if gstd else None,
    }


def _beta_xx(rows: list[dict], L: int) -> float | None:
    vals = []
    for rec in rows:
        if rec.get("method") != "hva" or int(rec["L"]) != int(L):
            continue
        bxx = np.asarray(rec["angles"].get("beta_xx", []), dtype=np.float64)
        if bxx.size:
            vals.append(float(np.mean(np.abs(bxx))))
    if not vals:
        return None
    return float(np.mean(vals))


def _load_rows(depths: list[int], seeds: list[int]) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for L in depths:
        for method in METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if not path.exists():
                    missing.append(path.name)
                    continue
                rows.append(json.loads(path.read_text()))
    return rows, missing


def _eps_matrix(block: list[dict], method: str, n: int) -> list[np.ndarray]:
    rows = sorted((r for r in block if r["method"] == method), key=lambda r: int(r["seed"]))
    return [np.asarray(r["eval"][str(n)]["eps"], dtype=np.float64) for r in rows]


def _reading(summary: dict) -> list[str]:
    lines = [
        "The comparison is excess energy above the exact ground state of the SAFE Hamiltonian, "
        "not a success-probability ratio. A seed BEATs when its own frozen-n12 Med ε ratio is ≤ 0.85.",
        "",
        "Seeds 0–2 use the same batches as PR #36 and reproduce that run's frozen n12 Med ε "
        "for Fourier, HVA, and SNAP at both depths. Seeds 3 and 4 are new.",
        "",
    ]
    d4 = summary.get("depths", {}).get("4", {})
    d2 = summary.get("depths", {}).get("2", {})
    if d4.get("complete") and d2.get("complete"):
        seed_r = d4["seed_R_hva_n12"]
        widest = max(range(len(seed_r)), key=lambda i: seed_r[i])
        lines.append(
            f"At L=4 the widest seed ratio is seed {widest} with R_HVA={_fmt(seed_r[widest])} "
            f"(still a BEAT). Instance intervals end at {_fmt(d2['ci95_n12']['hi'])} for L=2 and "
            f"{_fmt(d4['ci95_n12']['hi'])} for L=4, both under 0.85. "
            f"Seeds BEAT {d2['seeds_beat']}/5 at L=2 and {d4['seeds_beat']}/5 at L=4."
        )
        lines.append("")
        r6 = d4.get("R", {}).get("6", {}).get("R_hva")
        if r6 is not None:
            lines.append(
                f"The n=6 holdout at L=4 has R_HVA={_fmt(r6)}. "
                "That split is not the gate. PR #36 reported 1.016 on the same three-seed holdout."
            )
            lines.append("")
        lines.append(
            "Frozen n12 is the gate. n15 and the n18 probe are reported beside it and do not move the call. "
            "β^XX is away from zero at both depths, so the HVA column is not a Fourier run with a stuck mixer."
        )
        lines.append("")
    lines += [
        "L≥8 is outside this seal. PR #36 is the record that HVA takes those depths "
        "(L=8 R_HVA=1.372, L=16 R_HVA=2.275). This file does not reopen them.",
    ]
    return lines


def aggregate(depths: list[int], seeds: list[int]) -> dict:
    rows, missing = _load_rows(depths, seeds)
    summary: dict = {
        "protocol": PROTOCOL,
        "missing": missing,
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "beat_r": BEAT_R,
        "depths": {},
        "scope": "L in {2,4} only. L>=8 was not trained. n21 was not touched.",
    }
    depth_verdicts: dict[int, str] = {}
    sections: list[str] = []

    for L in depths:
        block = [r for r in rows if int(r["L"]) == int(L) and r.get("eval")]
        need = len(METHODS) * len(seeds)
        if len(block) < need:
            summary["depths"][str(L)] = {"complete": False, "n_files": len(block)}
            depth_verdicts[int(L)] = "INCOMPLETE"
            sections.append(f"## L={L}\n\nIncomplete ({len(block)}/{need} seed files).\n")
            continue
        ns = [6, 9, 12, 15]
        if all("18" in r["eval"] for r in block):
            ns.append(18)
        cell: dict[str, dict[int, float]] = {}
        per_seed: dict[str, dict[int, list[float]]] = {}
        for method in METHODS:
            cell[method] = {}
            per_seed[method] = {}
            for n in ns:
                vals = [
                    float(r["eval"][str(n)]["med_eps"])
                    for r in block
                    if r["method"] == method
                ]
                vals_by_seed = [
                    float(r["eval"][str(n)]["med_eps"])
                    for r in sorted(
                        (r for r in block if r["method"] == method),
                        key=lambda r: int(r["seed"]),
                    )
                ]
                cell[method][n] = float(np.median(vals))
                per_seed[method][n] = vals_by_seed
        seed_rs = [
            ratio(s, h) for s, h in zip(per_seed["snap"][12], per_seed["hva"][12])
        ]
        n_beat = sum(1 for r in seed_rs if r <= BEAT_R)
        n_hva = sum(1 for r in seed_rs if r > 1.0)
        snap12 = _eps_matrix(block, "snap", 12)
        hva12 = _eps_matrix(block, "hva", 12)
        r_point = ratio(cell["snap"][12], cell["hva"][12])
        if len(seeds) >= 2:
            boot = bootstrap_ratio(snap12, hva12)
            if abs(boot["R"] - r_point) > 1e-8:
                raise RuntimeError(f"bootstrap point {boot['R']} != table {r_point}")
        else:
            boot = {"lo": float("inf"), "hi": float("inf"), "n_boot": 0, "R": r_point}
        paired = [paired_counts(a, b) for a, b in zip(snap12, hva12)]
        paired_tot = {
            "win": int(sum(p["win"] for p in paired)),
            "tie": int(sum(p["tie"] for p in paired)),
            "loss": int(sum(p["loss"] for p in paired)),
            "n": int(sum(p["n"] for p in paired)),
            "per_seed": paired,
        }
        verdict = decide_depth(r_point, seed_rs, float(boot["hi"]), n_seeds=len(seeds))
        depth_verdicts[int(L)] = verdict
        roles = {
            6: "holdout",
            9: "holdout (select)",
            12: "freeze (gate)",
            15: "freeze",
            18: "probe",
        }
        lines = [f"## L={L}", ""]
        lines.append(
            "| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        rmap = {}
        for n in ns:
            rf = ratio(cell["snap"][n], cell["fourier"][n])
            rh = ratio(cell["snap"][n], cell["hva"][n])
            rmap[n] = {"R_fourier": rf, "R_hva": rh}
            lines.append(
                f"| {n} | {roles[n]} | {_fmt(cell['fourier'][n])} | {_fmt(cell['hva'][n])} | "
                f"{_fmt(cell['snap'][n])} | {_fmt(rf)} | {_fmt(rh)} |"
            )
        lines += [
            "",
            f"**L={L} n12 gate: {verdict}.** "
            f"R_HVA={_fmt(r_point)} "
            f"(SNAP {_fmt(cell['snap'][12])} / HVA {_fmt(cell['hva'][12])}). "
            f"Instance 95% CI {_fmt(boot['lo'])}–{_fmt(boot['hi'])}. "
            f"Seeds BEAT {n_beat}/5 (R≤{BEAT_R:.2f}); seeds with R>1: {n_hva}/5. "
            f"Paired n12 instances, SNAP lower ε than HVA: "
            f"{paired_tot['win']}/{paired_tot['n']}.",
            "",
            "| seed | n12 SNAP | n12 HVA | n12 Fourier | R_HVA | mark | paired SNAP<HVA | best round SNAP | wall SNAP s |",
            "|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        snap_rows = sorted((r for r in block if r["method"] == "snap"), key=lambda r: int(r["seed"]))
        hva_rows = sorted((r for r in block if r["method"] == "hva"), key=lambda r: int(r["seed"]))
        fou_rows = sorted((r for r in block if r["method"] == "fourier"), key=lambda r: int(r["seed"]))
        for i, (sr, hr, fr) in enumerate(zip(snap_rows, hva_rows, fou_rows)):
            mark = "BEAT" if seed_rs[i] <= BEAT_R else ("HVA" if seed_rs[i] > 1.0 else "soft")
            lines.append(
                f"| {sr['seed']} | {_fmt(sr['eval']['12']['med_eps'])} | "
                f"{_fmt(hr['eval']['12']['med_eps'])} | {_fmt(fr['eval']['12']['med_eps'])} | "
                f"{_fmt(seed_rs[i])} | {mark} | {paired[i]['win']}/{paired[i]['n']} | "
                f"{sr['best_round']} | {sr['wall_s']:.0f} |"
            )
        lines.append("")
        bxx = _beta_xx(block, L)
        shell = _shell_stats(block)
        if bxx is not None:
            lines.append(
                f"HVA mean |β^XX| over layers and seeds: {_fmt(bxx, 3)}. "
                f"SNAP shell RMS off the linear φ_k=a+bk fit, median over layers: "
                f"{_fmt(shell['shell_rms_med'], 4)}. "
                f"Median within-layer std of γ_w: {_fmt(shell['gamma_w_std_med'], 4)}."
            )
            lines.append("")
        if 15 in cell["snap"]:
            lines.append(
                f"n15 is not the gate. R_HVA={_fmt(rmap[15]['R_hva'])} "
                f"(SNAP {_fmt(cell['snap'][15])} / HVA {_fmt(cell['hva'][15])})."
            )
            lines.append("")
        if 18 in cell["snap"]:
            lines.append(
                f"n18 probe, B={N18_B}, not part of the gate. "
                f"R_HVA={_fmt(rmap[18]['R_hva'])} "
                f"(SNAP {_fmt(cell['snap'][18])} / HVA {_fmt(cell['hva'][18])}, "
                f"Fourier {_fmt(cell['fourier'][18])})."
            )
            lines.append("")
        sections.append("\n".join(lines))
        summary["depths"][str(L)] = {
            "complete": True,
            "verdict_n12": verdict,
            "R_hva_n12": r_point,
            "ci95_n12": {"lo": boot["lo"], "hi": boot["hi"], "n_boot": boot["n_boot"]},
            "seeds_beat": n_beat,
            "seeds_hva": n_hva,
            "seed_R_hva_n12": seed_rs,
            "paired_n12": paired_tot,
            "med_eps": {m: {str(n): cell[m][n] for n in ns} for m in METHODS},
            "R": {str(n): rmap[n] for n in ns},
            "mean_abs_beta_xx": bxx,
            "snap_shell": shell,
            "per_seed_n12": {
                m: per_seed[m][12] for m in METHODS
            },
        }

    # Depths that were not loaded stay out of the overall call.
    for L in DEPTHS:
        depth_verdicts.setdefault(int(L), "INCOMPLETE")
    overall = decide_overall(depth_verdicts)
    summary["overall"] = overall
    summary["depth_verdicts"] = {str(k): v for k, v in depth_verdicts.items()}

    headline = []
    for L in DEPTHS:
        info = summary["depths"].get(str(L), {})
        if info.get("complete"):
            headline.append(
                f"L={L} → {info['verdict_n12']} "
                f"(R_HVA={_fmt(info['R_hva_n12'])}, {info['seeds_beat']}/5 seeds BEAT, "
                f"CI {_fmt(info['ci95_n12']['lo'])}–{_fmt(info['ci95_n12']['hi'])})"
            )
        else:
            headline.append(f"L={L} → INCOMPLETE")

    text = "\n".join(
        [
            "# Shallow seal — Walsh SNAP vs HVA at the SAFE point",
            "",
            f"**Verdict: {overall}.** " + " ".join(headline) + ".",
            "",
            "This seal is L∈{2, 4} only. PR #36 already has HVA ahead of SNAP at L=8 "
            "(R_HVA=1.372) and L=16 (R_HVA=2.275) under the same Hamiltonian and the same "
            "L-BFGS budget. Those depths were not retrained here and are not claimed.",
            "",
            "## Gate",
            "",
            "The call uses frozen n=12. Med ε is the median over seeds of each seed's "
            "median-over-instances excess energy ⟨H⟩−E₀. R_HVA = Medε_SNAP / Medε_HVA.",
            "",
            "- **PASS** on a depth when R_HVA ≤ 0.85, at least 3 of 5 seeds have their own "
            "n12 R_HVA ≤ 0.85, and the instance-bootstrap 95% interval for R_HVA ends at or below 0.85.",
            "- **needs_edit** when that depth is not a PASS and HVA has not taken it. "
            "That is the soft case: the median lead is past 0.85, fewer than 3 seeds clear 0.85, "
            "or the instance interval crosses 0.85.",
            "- **KILL** when R_HVA > 1 at n12, or at least 3 of 5 seeds have R_HVA > 1.",
            "- Overall is KILL if either depth is KILL, needs_edit if either depth is needs_edit, "
            "and PASS only when both L=2 and L=4 are PASS.",
            "- n=15 is reported and is not the gate. n=18 is a probe and is not the gate.",
            "",
            "## Hamiltonian",
            "",
            r"\[",
            r"H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j",
            r"\]",
            "",
            "SAFE point **λ=0.75, μ=−0.50**. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. "
            "Readout is the Z histogram (unsat) plus the X histogram (⟨X⟩ and ⟨XX⟩). No Y.",
            "",
            "## Ansatz",
            "",
            "| method | angles / layer | mixer |",
            "|---|---|---|",
            r"| Fourier QAOA | γ, β | \(\bigotimes \exp(-i\beta X/2)\) |",
            r"| HVA QAOA | γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\) then \(U_X(\beta^X)\). β^XX=0 is Fourier. |",
            r"| Walsh SNAP | γ_{w=1..4}, φ_k | BinH Hamming shell. Linear φ_k=−βk is the X mixer. |",
            "",
            f"Paper init β=β^X={INIT_BETA}, β^XX=0, γ=γ_w={INIT_GAMMA}, φ_k=−β k. "
            "The three inits are the same state. The seed only changes the L-BFGS batches.",
            "",
            "## Protocol",
            "",
            f"- Packs: `beat_qaoa_12h/train_packs`. Train indices [{TRAIN_LO}:{TRAIN_HI}) on n=6 and n=9. "
            f"Checkpoint on n=9 holdout [{HOLD_LO}:{HOLD_HI}). "
            f"Freeze n12 first {FREEZE_B[12]}, n15 first {FREEZE_B[15]}.",
            f"- n18 probe: {N18_B} satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` "
            f"at base {N18_BASE}. There is no n18 train pack in the tree. "
            "The probe is not in the loss and not in checkpoint selection.",
            f"- L-BFGS matches `Nov_3rd.settings` and PR #36: {S.LBFGS_ROUNDS} rounds × max_iter {S.LBFGS_ITER}, "
            f"lr {S.LBFGS_LR}, history 16, strong Wolfe, grad clip 5, batch {S.LBFGS_BATCH} "
            "(half n6 / half n9). New optimizer each round. Loss = mean ε/n. No Adam.",
            "- Each seed restores the round with the best n9-holdout Med ε.",
            "- Instance 95% CI: 4000 bootstrap draws, resampling instances inside each seed, seeds held fixed. "
            f"RNG seed {BOOT_SEED}.",
            "- n21 phase-2 ladder was not read or written. L≥8 was not trained.",
            "",
            *sections,
            "## Reading",
            "",
            *_reading(summary),
            "",
            "Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. "
            "Angles are the restored n9-holdout checkpoint.",
            "",
        ]
    )
    if not text.endswith("\n"):
        text += "\n"
    write_json(RUN / "summary.json", summary)
    (RUN / "RESULTS.md").write_text(text)
    log(f"wrote RESULTS.md overall={overall}")
    return summary


def _reject_scope(depths: list[int]) -> None:
    bad = [L for L in depths if L not in DEPTHS]
    if bad:
        raise SystemExit(
            f"refusing depths {bad}: this seal is L in {{2,4}} only and does not claim L>=8"
        )


def main():
    _self_test_gate()
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--rounds", type=int, default=S.LBFGS_ROUNDS)
    ap.add_argument("--batch", type=int, default=S.LBFGS_BATCH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--skip-n18", action="store_true")
    ap.add_argument("--n18-limit-s", type=float, default=180.0)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    _reject_scope(depths)
    if any(s < 0 for s in seeds):
        raise SystemExit("bad seed")
    RUN.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(depths, seeds)
        return

    if not args.skip_self_check:
        log("self-check")
        E6_all, _ = load_unsat(6, slice(0, 250))
        self_check(E6_all)
        log("self-check ok")

    data = {}
    for n, sl, tag in (
        (6, slice(TRAIN_LO, TRAIN_HI), "train"),
        (9, slice(TRAIN_LO, TRAIN_HI), "train"),
        (6, slice(HOLD_LO, HOLD_HI), "hold"),
        (9, slice(HOLD_LO, HOLD_HI), "hold"),
        (12, slice(0, FREEZE_B[12]), "freeze"),
        (15, slice(0, FREEZE_B[15]), "freeze"),
    ):
        E, sd = load_unsat(n, sl)
        e0 = ensure_e0(n, tag, E, sd, args.workers if n >= 12 else 1)
        data[(n, tag)] = (E, e0, sd)
        log(
            f"n{n} {tag} B={len(E)} medE0={float(np.median(e0)):.4f} "
            f"sat_frac={float(np.mean(E.min(1)==0)):.2f}"
        )

    n18_status = {"status": "skipped"}
    if not args.skip_n18:
        try:
            E18, sd18 = build_n18(N18_B)
            t0 = time.time()
            e_one = float(ground_energy(E18[0], 18))
            dt = time.time() - t0
            log(f"n18 feasibility one ground state {dt:.1f}s E0={e_one:.4f}")
            if dt > float(args.n18_limit_s):
                n18_status = {
                    "status": "infeasible",
                    "detail": f"one n=18 ground state took {dt:.0f}s, above {args.n18_limit_s:.0f}s",
                }
            else:
                e0_18 = ensure_e0(18, "probe", E18, sd18, args.workers)
                # ensure_e0 recomputes instance 0; that is cached after the first full write.
                # If the cache missed, the parallel call includes instance 0 again. Fine.
                if abs(float(e0_18[0]) - e_one) > 1e-5:
                    raise RuntimeError(f"n18 E0 cache {e0_18[0]} != probe {e_one}")
                data[(18, "probe")] = (E18, e0_18, sd18)
                n18_status = {"status": "ok", "B": N18_B, "one_ground_s": dt, "med_e0": float(np.median(e0_18))}
        except Exception as exc:
            log(f"n18 probe not scored: {exc}")
            n18_status = {"status": "infeasible", "detail": str(exc)}
    write_json(RUN / "n18_status.json", n18_status)
    log("n18", n18_status)

    spec = {}
    for n, tag in ((6, "train"), (9, "train"), (12, "freeze"), (15, "freeze")):
        pwr = walsh_power_by_degree(data[(n, tag)][0][0])
        spec[str(n)] = {"degree_gt4": float(pwr[5:].sum()), "by_degree": pwr.tolist()}
        log(f"Walsh n{n} degree>4 power {spec[str(n)]['degree_gt4']:.3e}")
    write_json(RUN / "walsh_spectrum.json", spec)

    torch.set_num_threads(max(1, int(args.workers)))
    train_tensors = {}
    for n in TRAIN_NS:
        E, e0, _sd = data[(n, "train")]
        train_tensors[n] = (
            torch.tensor(E, dtype=torch.float64),
            torch.tensor(e0, dtype=torch.float64),
        )
    hold_bundle = {9: data[(9, "hold")]}
    eval_bundle = {
        6: data[(6, "hold")],
        9: data[(9, "hold")],
        12: data[(12, "freeze")],
        15: data[(15, "freeze")],
    }
    if (18, "probe") in data:
        eval_bundle[18] = data[(18, "probe")]

    for L in depths:
        init_ev = {}
        for method in METHODS:
            init_ev[method] = eval_method(
                method, init_angles(method, L, dtype=torch.float64), eval_bundle, chunk=8
            )
        base = np.asarray(init_ev["fourier"]["12"]["eps"])
        for method in METHODS:
            err = float(np.max(np.abs(np.asarray(init_ev[method]["12"]["eps"]) - base)))
            log(f"L{L} init {method} vs fourier n12 max|Δε|={err:.3e}")
            if err > 1e-6:
                raise RuntimeError(f"init states differ: {method} Δ={err}")
        init_public = {
            n: {
                "med_eps": init_ev["fourier"][n]["med_eps"],
                "mean_eps": init_ev["fourier"][n]["mean_eps"],
            }
            for n in init_ev["fourier"]
        }

        for method in METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if path.exists() and not args.force:
                    log(f"skip existing {path.name}")
                    continue
                log(f"=== train L{L} {method} seed{seed} ===")
                try:
                    trained = train_one(
                        method,
                        L,
                        seed,
                        train_tensors,
                        hold_bundle,
                        rounds=args.rounds,
                        batch=args.batch,
                    )
                except Exception:
                    log(traceback.format_exc())
                    raise
                ev = eval_method(method, trained["angles"], eval_bundle, chunk=8)
                payload = {
                    "done": True,
                    "protocol": PROTOCOL,
                    "method": method,
                    "L": L,
                    "seed": seed,
                    "n_params": trained["n_params"],
                    "best_round": trained["best_round"],
                    "init_hold_med9": trained["init_hold_med9"],
                    "best_hold_med9_train_dtype": trained["best_hold_med9"],
                    "wall_s": trained["wall_s"],
                    "rounds": args.rounds,
                    "max_iter": S.LBFGS_ITER,
                    "lr": S.LBFGS_LR,
                    "batch": args.batch,
                    "lambda_x": LAMBDA_X,
                    "mu_xx": MU_XX,
                    "angles": angles_to_lists(method, trained["angles"]),
                    "init_eval": init_public,
                    "eval": ev,
                    "hist": trained["hist"],
                    "ansatz": {
                        "fourier": "U_X(beta) U_C(gamma), 2 angles/layer",
                        "hva": "U_XX(beta_xx) U_X(beta_x) U_C(gamma), 3 angles/layer; beta_xx=0 is Fourier",
                        "snap": "Walsh gamma_w=1..4 then Hamming-shell BinH mixer, 8 numbers/layer",
                    }[method],
                }
                write_json(path, payload)
                n18_bit = ""
                if "18" in ev:
                    n18_bit = f" n18 Medε={ev['18']['med_eps']:.4f}"
                log(
                    f"saved {path.name} best_r={trained['best_round']} "
                    f"n12 Medε={ev['12']['med_eps']:.4f} n15 Medε={ev['15']['med_eps']:.4f}"
                    f"{n18_bit} wall={trained['wall_s']:.0f}s"
                )

    summary = aggregate(depths, seeds)
    log("OVERALL", summary.get("overall"))


if __name__ == "__main__":
    main()
