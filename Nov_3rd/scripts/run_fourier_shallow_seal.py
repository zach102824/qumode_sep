#!/usr/bin/env python3
"""Five-seed shallow seal of Walsh SNAP against plain Fourier QAOA.

The gate is R_Fourier = Medε_SNAP / Medε_Fourier at frozen n=12, L∈{2,4}.
Fourier is (γ, β^X) only: the mixer is ⊗ exp(−i β X/2). There is no XX mixer
and this script does not train HVA. HVA numbers in the table are cited from
PR #38 (hva_pr38.json) and do not enter the call.

    H = diag(unsat) − 0.75 Σ_i X_i − 0.50 Σ_c Σ_{i<j in c} X_i X_j

Train n∈{6,9} only. Freeze n=12 (B=128) and report n=15 (B=64). n=18 (B=16)
is a probe. L≥8 is refused. The n=21 ladder is not opened.
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
from Nov_3rd.fourier_shallow_gate import (
    BEAT_R,
    BOOT_DRAWS,
    BOOT_SEED,
    CI_UPPER,
    DEPTHS,
    KILL_R,
    MIN_BEAT_SEEDS,
    bootstrap_ratio,
    decide_depth,
    decide_overall,
    paired_counts,
    ratio,
    self_test,
)
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

RUN = S.OUT / "quantum_h_explore" / "fourier_shallow_seal"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
HVA_REF = RUN / "hva_pr38.json"
TRAIN_METHODS = ("fourier", "snap")
TRAIN_NS = (6, 9)
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = {12: 128, 15: 64}
N18_B = 16
N18_BASE = 61_000_000_000
PROTOCOL = "fourier_shallow_seal_r10"
FOURIER_ANGLE_KEYS = ("betas", "gammas")


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fourier-shallow]", *a, flush=True)


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    if int(n) >= 21:
        raise RuntimeError("n21 is outside this seal")
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
            e0 = np.asarray(z["e0"], dtype=np.float64)
            check = float(ground_energy(E[0], n))
            if abs(check - float(e0[0])) > 1e-5:
                raise RuntimeError(f"E0 cache n{n} {tag} row0 {e0[0]} != fresh {check}")
            log(f"E0 cache hit n{n} {tag} (row0 checked)")
            return e0
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


def assert_fourier_angles(angles: dict) -> None:
    keys = set(angles)
    keys.discard("method")
    if "beta_xx" in keys or "beta_x" in keys:
        raise RuntimeError(f"Fourier angles include an XX/HVA key: {sorted(keys)}")
    if tuple(sorted(keys)) != tuple(sorted(FOURIER_ANGLE_KEYS)):
        raise RuntimeError(f"Fourier angles {sorted(keys)} != {list(FOURIER_ANGLE_KEYS)}")


def eval_method(method: str, angles: dict, bundle: dict, *, chunk: int) -> dict:
    if method == "fourier":
        assert_fourier_angles(angles)
    if method not in TRAIN_METHODS:
        raise RuntimeError(f"this seal does not evaluate {method}")
    ang = _as_f64(angles)
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        if int(n) >= 21:
            raise RuntimeError("n21 is outside this seal")
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
    if method == "hva":
        raise RuntimeError("HVA is cited from PR #38 and is not trained here")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    angles = init_angles(method, L, dtype=torch.float64)
    if method == "fourier":
        assert_fourier_angles(angles)
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


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _shell_stats(rows: list[dict]) -> dict:
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


def _mean_abs_beta(rows: list[dict]) -> float | None:
    vals = []
    for rec in rows:
        if rec.get("method") != "fourier":
            continue
        assert_fourier_angles(rec["angles"])
        b = np.asarray(rec["angles"]["betas"], dtype=np.float64)
        if b.size:
            vals.append(float(np.mean(np.abs(b))))
    if not vals:
        return None
    return float(np.mean(vals))


def _load_rows(depths: list[int], seeds: list[int]) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for L in depths:
        for method in TRAIN_METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if not path.exists():
                    missing.append(path.name)
                    continue
                rec = json.loads(path.read_text())
                if rec.get("method") == "fourier":
                    assert_fourier_angles(rec["angles"])
                rows.append(rec)
    return rows, missing


def _eps_matrix(block: list[dict], method: str, n: int) -> list[np.ndarray]:
    rows = sorted((r for r in block if r["method"] == method), key=lambda r: int(r["seed"]))
    return [np.asarray(r["eval"][str(n)]["eps"], dtype=np.float64) for r in rows]


def _load_hva_ref() -> dict | None:
    if not HVA_REF.exists():
        return None
    ref = json.loads(HVA_REF.read_text())
    if ref.get("source_pr") != 38:
        raise RuntimeError(f"unexpected HVA citation {ref.get('source_pr')}")
    return ref


def _reading(summary: dict) -> list[str]:
    lines = [
        "Excess energy is ⟨H⟩−E₀ on the SAFE Hamiltonian. "
        "A seed BEATs when its own frozen-n12 Medε ratio against Fourier is ≤ 0.85.",
        "",
        "Fourier's mixer is ⊗ exp(−i β X/2). Saved Fourier checkpoints contain "
        "only β and γ. The XX mixer belongs to the HVA column, which is cited from "
        "PR #38 and is not this gate.",
        "",
    ]
    d2 = summary.get("depths", {}).get("2", {})
    d4 = summary.get("depths", {}).get("4", {})
    if d2.get("complete") and d4.get("complete"):
        for L, info in ((2, d2), (4, d4)):
            seed_r = info["seed_R_fourier_n12"]
            widest = max(range(len(seed_r)), key=lambda i: seed_r[i])
            lines.append(
                f"L={L}: widest seed is seed {widest} with R_Fourier={_fmt(seed_r[widest])}. "
                f"Instance interval {_fmt(info['ci95_n12']['lo'])}–{_fmt(info['ci95_n12']['hi'])}. "
                f"Seeds BEAT {info['seeds_beat']}/5."
            )
        lines.append("")
        r6 = d4.get("R", {}).get("6", {}).get("R_fourier")
        if r6 is not None:
            lines.append(
                f"The n=6 holdout at L=4 has R_Fourier={_fmt(r6)}. "
                "That split is not the gate. Frozen n12 is the gate."
            )
            lines.append("")
        deltas = [d2.get("pr38_n12_abs_delta"), d4.get("pr38_n12_abs_delta")]
        if all(isinstance(d, dict) for d in deltas):
            worst = max(max(d.values()) for d in deltas)
            lines.append(
                "Frozen n12 Medε for Fourier and SNAP matches the PR #38 five-seed trains "
                f"(max |Δ|={worst:.3e}). The cited HVA column is the same n12 slice."
            )
            lines.append("")
        lines.append(
            "n15 and the n18 probe sit beside the gate and do not move it. "
            "L≥8 was not trained. PR #39 left the L=16 rescue at needs_edit "
            "(R_HVA=1.006). This file does not claim those depths."
        )
    return lines


def aggregate(depths: list[int], seeds: list[int]) -> dict:
    rows, missing = _load_rows(depths, seeds)
    hva = _load_hva_ref()
    summary: dict = {
        "protocol": PROTOCOL,
        "missing": missing,
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "beat_r": BEAT_R,
        "ci_upper": CI_UPPER,
        "kill_r": KILL_R,
        "min_beat_seeds": MIN_BEAT_SEEDS,
        "gate": "R_Fourier = Med_SNAP / Med_Fourier at frozen n12",
        "hva_source": "PR #38" if hva else None,
        "depths": {},
        "scope": "L in {2,4} only. L>=8 was not trained. n21 was not touched. Fourier has no XX mixer.",
    }
    depth_verdicts: dict[int, str] = {}
    sections: list[str] = []

    for L in depths:
        block = [r for r in rows if int(r["L"]) == int(L) and r.get("eval")]
        need = len(TRAIN_METHODS) * len(seeds)
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
        for method in TRAIN_METHODS:
            cell[method] = {}
            per_seed[method] = {}
            for n in ns:
                ordered = sorted(
                    (r for r in block if r["method"] == method),
                    key=lambda r: int(r["seed"]),
                )
                vals = [float(r["eval"][str(n)]["med_eps"]) for r in ordered]
                cell[method][n] = float(np.median(vals))
                per_seed[method][n] = vals
        seed_rs = [ratio(s, f) for s, f in zip(per_seed["snap"][12], per_seed["fourier"][12])]
        n_beat = sum(1 for r in seed_rs if r <= BEAT_R)
        n_taken = sum(1 for r in seed_rs if r > KILL_R)
        snap12 = _eps_matrix(block, "snap", 12)
        fou12 = _eps_matrix(block, "fourier", 12)
        r_point = ratio(cell["snap"][12], cell["fourier"][12])
        if len(seeds) >= 2:
            boot = bootstrap_ratio(snap12, fou12)
            if abs(boot["R"] - r_point) > 1e-8:
                raise RuntimeError(f"bootstrap point {boot['R']} != table {r_point}")
        else:
            boot = {"lo": float("inf"), "hi": float("inf"), "n_boot": 0, "R": r_point}
        paired = [paired_counts(a, b) for a, b in zip(snap12, fou12)]
        paired_tot = {
            "win": int(sum(p["win"] for p in paired)),
            "tie": int(sum(p["tie"] for p in paired)),
            "loss": int(sum(p["loss"] for p in paired)),
            "n": int(sum(p["n"] for p in paired)),
            "per_seed": paired,
        }
        verdict = decide_depth(r_point, seed_rs, float(boot["hi"]), n_seeds=len(seeds))
        depth_verdicts[int(L)] = verdict
        hva_L = None if hva is None else hva["depths"][str(L)]
        roles = {
            6: "holdout",
            9: "holdout (select)",
            12: "freeze (gate)",
            15: "freeze",
            18: "probe",
        }
        lines = [f"## L={L}", ""]
        lines.append(
            "| n | role | Medε Fourier | Medε HVA (#38) | Medε SNAP | R_Fourier | R SNAP/HVA |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        rmap = {}
        for n in ns:
            rf = ratio(cell["snap"][n], cell["fourier"][n])
            hva_med = None
            rh = None
            if hva_L is not None and str(n) in hva_L["med_eps"]:
                hva_med = float(hva_L["med_eps"][str(n)])
                rh = ratio(cell["snap"][n], hva_med)
            rmap[n] = {"R_fourier": rf, "R_hva_cited": rh, "hva_med": hva_med}
            lines.append(
                f"| {n} | {roles[n]} | {_fmt(cell['fourier'][n])} | {_fmt(hva_med)} | "
                f"{_fmt(cell['snap'][n])} | {_fmt(rf)} | {_fmt(rh)} |"
            )
        lines += [
            "",
            f"**L={L} n12 gate: {verdict}.** "
            f"R_Fourier={_fmt(r_point)} "
            f"(SNAP {_fmt(cell['snap'][12])} / Fourier {_fmt(cell['fourier'][12])}). "
            f"Instance 95% CI {_fmt(boot['lo'])}–{_fmt(boot['hi'])}. "
            f"Seeds BEAT {n_beat}/5 (R≤{BEAT_R:.2f}); seeds with R>{KILL_R:.2f}: {n_taken}/5. "
            f"Paired n12 instances, SNAP lower ε than Fourier: "
            f"{paired_tot['win']}/{paired_tot['n']}.",
            "",
            "| seed | n12 SNAP | n12 Fourier | R_Fourier | mark | paired SNAP<Fourier | n12 HVA (#38) | best round SNAP | wall SNAP s |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        snap_rows = sorted((r for r in block if r["method"] == "snap"), key=lambda r: int(r["seed"]))
        fou_rows = sorted((r for r in block if r["method"] == "fourier"), key=lambda r: int(r["seed"]))
        hva_seeds = [] if hva_L is None else list(hva_L["per_seed_n12"])
        for i, (sr, fr) in enumerate(zip(snap_rows, fou_rows)):
            if seed_rs[i] <= BEAT_R:
                mark = "BEAT"
            elif seed_rs[i] > KILL_R:
                mark = "FOURIER"
            else:
                mark = "soft"
            hva_s = hva_seeds[i] if i < len(hva_seeds) else None
            lines.append(
                f"| {sr['seed']} | {_fmt(sr['eval']['12']['med_eps'])} | "
                f"{_fmt(fr['eval']['12']['med_eps'])} | {_fmt(seed_rs[i])} | {mark} | "
                f"{paired[i]['win']}/{paired[i]['n']} | {_fmt(hva_s)} | "
                f"{sr['best_round']} | {sr['wall_s']:.0f} |"
            )
        lines.append("")
        shell = _shell_stats(block)
        beta = _mean_abs_beta(block)
        bxx = None if hva_L is None else hva_L.get("mean_abs_beta_xx")
        pr38_delta = None
        if hva_L is not None and "snap_med_n12" in hva_L and "fourier_med_n12" in hva_L:
            pr38_delta = {
                "snap": abs(cell["snap"][12] - float(hva_L["snap_med_n12"])),
                "fourier": abs(cell["fourier"][12] - float(hva_L["fourier_med_n12"])),
            }
        lines.append(
            f"Fourier mean |β| over layers and seeds: {_fmt(beta, 4)}. "
            f"SNAP shell RMS off the linear φ_k=a+bk fit, median over layers: "
            f"{_fmt(shell['shell_rms_med'], 4)}. "
            f"Median within-layer std of γ_w: {_fmt(shell['gamma_w_std_med'], 4)}."
        )
        if bxx is not None:
            lines.append(
                f"Cited HVA mean |β^XX| from PR #38: {_fmt(bxx, 3)}. "
                "That mixer is absent from the Fourier column."
            )
        lines.append("")
        if 15 in cell["snap"]:
            lines.append(
                f"n15 is not the gate. R_Fourier={_fmt(rmap[15]['R_fourier'])} "
                f"(SNAP {_fmt(cell['snap'][15])} / Fourier {_fmt(cell['fourier'][15])})."
            )
            lines.append("")
        if 18 in cell["snap"]:
            lines.append(
                f"n18 probe, B={N18_B}, not part of the gate. "
                f"R_Fourier={_fmt(rmap[18]['R_fourier'])} "
                f"(SNAP {_fmt(cell['snap'][18])} / Fourier {_fmt(cell['fourier'][18])})."
            )
            lines.append("")
        sections.append("\n".join(lines))
        summary["depths"][str(L)] = {
            "complete": True,
            "verdict_n12": verdict,
            "R_fourier_n12": r_point,
            "ci95_n12": {"lo": boot["lo"], "hi": boot["hi"], "n_boot": boot["n_boot"]},
            "seeds_beat": n_beat,
            "seeds_fourier_took": n_taken,
            "seed_R_fourier_n12": seed_rs,
            "paired_n12": paired_tot,
            "med_eps": {m: {str(n): cell[m][n] for n in ns} for m in TRAIN_METHODS},
            "hva_med_eps_pr38": None if hva_L is None else hva_L["med_eps"],
            "R": {str(n): rmap[n] for n in ns},
            "mean_abs_beta": beta,
            "hva_mean_abs_beta_xx_pr38": bxx,
            "snap_shell": shell,
            "per_seed_n12": {m: per_seed[m][12] for m in TRAIN_METHODS},
            "pr38_n12_abs_delta": pr38_delta,
        }

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
                f"(R_Fourier={_fmt(info['R_fourier_n12'])}, {info['seeds_beat']}/5 seeds BEAT, "
                f"CI {_fmt(info['ci95_n12']['lo'])}–{_fmt(info['ci95_n12']['hi'])})"
            )
        else:
            headline.append(f"L={L} → INCOMPLETE")

    text = "\n".join(
        [
            "# Shallow seal — Walsh SNAP vs plain Fourier QAOA",
            "",
            f"**Verdict: {overall}.** " + " ".join(headline) + ".",
            "",
            "This seal is L∈{2, 4} only. The call is R_Fourier on frozen n=12. "
            "HVA from PR #38 is a column in the table. PR #39 left L=16 at needs_edit "
            "(R_HVA=1.006). Those depths were not trained here and are not claimed.",
            "",
            "## Gate",
            "",
            "Frozen n=12. Medε is the median over seeds of each seed's median-over-instances "
            "excess energy ⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier.",
            "",
            f"- **PASS** on a depth when R_Fourier ≤ {BEAT_R:.2f}, at least {MIN_BEAT_SEEDS} of 5 seeds "
            f"have their own n12 R_Fourier ≤ {BEAT_R:.2f}, and the instance-bootstrap 95% interval "
            f"ends strictly below {CI_UPPER:.2f}.",
            "- **needs_edit** when that depth is not a PASS and Fourier has not taken it.",
            f"- **KILL** when R_Fourier > {KILL_R:.2f} at n12, or at least {MIN_BEAT_SEEDS} of 5 seeds "
            f"have R_Fourier > {KILL_R:.2f}.",
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
            r"| Fourier QAOA | γ, β | \(\bigotimes_i \exp(-i \beta X_i/2)\). No XX mixer. |",
            r"| Walsh SNAP | γ_{w=1..4}, φ_k | BinH Hamming shell. Linear φ_k=−β k is the X mixer. |",
            r"| HVA (cited, #38) | γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\) then \(U_X(\beta^X)\). Not trained here. |",
            "",
            f"Paper init β={INIT_BETA}, γ=γ_w={INIT_GAMMA}, φ_k=−β k. "
            "Fourier and SNAP start on the same state. The seed only changes the L-BFGS batches. "
            "HVA in PR #38 used the same init with β^XX=0.",
            "",
            "## Protocol",
            "",
            f"- Packs: `beat_qaoa_12h/train_packs`. Train indices [{TRAIN_LO}:{TRAIN_HI}) on n=6 and n=9. "
            f"Checkpoint on n=9 holdout [{HOLD_LO}:{HOLD_HI}). "
            f"Freeze n12 first {FREEZE_B[12]}, n15 first {FREEZE_B[15]}.",
            f"- n18 probe: {N18_B} satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` "
            f"at base {N18_BASE}. The probe is not in the loss and not in checkpoint selection.",
            f"- L-BFGS matches `Nov_3rd.settings`: {S.LBFGS_ROUNDS} rounds × max_iter {S.LBFGS_ITER}, "
            f"lr {S.LBFGS_LR}, history 16, strong Wolfe, grad clip 5, batch {S.LBFGS_BATCH} "
            "(half n6 / half n9). New optimizer each round. Loss = mean ε/n. No Adam.",
            "- Each seed restores the round with the best n9-holdout Medε.",
            "- Instance 95% CI: 4000 bootstrap draws, resampling instances inside each seed, seeds held fixed. "
            f"RNG seed {BOOT_SEED}.",
            "- HVA medians are read from `hva_pr38.json` (PR #38 five-seed shallow seal). "
            "They are not refit.",
            "- n21 was not read or written. L≥8 was not trained.",
            "",
            *sections,
            "## Reading",
            "",
            *_reading(summary),
            "",
            "Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. "
            "Angles are the restored n9-holdout checkpoint. Fourier files have `betas` and `gammas` only.",
            "",
        ]
    )
    if not text.endswith("\n"):
        text += "\n"
    write_json(RUN / "summary.json", summary)
    (RUN / "RESULTS.md").write_text(text)
    log(f"wrote RESULTS.md overall={overall}")
    return summary


def _reject_scope(depths: list[int], methods: list[str]) -> None:
    bad = [L for L in depths if L not in DEPTHS]
    if bad:
        raise SystemExit(f"refusing depths {bad}: this seal is L in {{2,4}} only and does not claim L>=8")
    bad_m = [m for m in methods if m not in TRAIN_METHODS]
    if bad_m:
        raise SystemExit(f"refusing methods {bad_m}: Fourier is γ, β^X only; HVA is not trained")


def main():
    self_test()
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--methods", type=str, default="fourier,snap")
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
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    _reject_scope(depths, methods)
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
                if abs(float(e0_18[0]) - e_one) > 1e-5:
                    raise RuntimeError(f"n18 E0 cache {e0_18[0]} != probe {e_one}")
                data[(18, "probe")] = (E18, e0_18, sd18)
                n18_status = {
                    "status": "ok",
                    "B": N18_B,
                    "one_ground_s": dt,
                    "med_e0": float(np.median(e0_18)),
                }
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
        for method in methods:
            init_ev[method] = eval_method(
                method, init_angles(method, L, dtype=torch.float64), eval_bundle, chunk=8
            )
        base = np.asarray(init_ev["fourier"]["12"]["eps"])
        for method in methods:
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

        for method in methods:
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
                listed = angles_to_lists(method, trained["angles"])
                if method == "fourier":
                    assert_fourier_angles(listed)
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
                    "angles": listed,
                    "init_eval": init_public,
                    "eval": ev,
                    "hist": trained["hist"],
                    "ansatz": {
                        "fourier": "U_X(beta) U_C(gamma), 2 angles/layer, no XX mixer",
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
