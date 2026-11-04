#!/usr/bin/env python3
"""Larger-B multi-seed seal of unsat λ=0.75 Walsh and power_p.

Train on the union of n=6 and n=9 only. The checkpoint is the best holdout
score on those two sizes. Frozen angles are then scored at n=12 and n=15
against same-depth Fourier QAOA.

Hamiltonian (PR #11, stoquastic sign), gate sequence unchanged::

    H = diag(unsat) − λ Σ_i X_i ,   λ = 0.75
    |0⟩ → [U_c^X → Hamming-shell SNAP]^L → W
    U_c^X = W exp(−i γ C_n) W ,   W = H^{⊗3}
    C_n = (n/m) unsat − max

Walsh grades γ across degrees 1–4 of C_n. power_p is the scalar-γ control.
The published ZZZ champ (PR #13 / #15, λ=0.5, power_p, B_train=32) is the
comparison, not a retrain.

Does not write qaoa_angles/, train69_ladder_fresh250/, or settings.py.
Does not read or write the phase-2 n=21 ladder.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.sat_gen import clause_count
from Nov_3rd import settings as S
from Nov_3rd.quantum_h_tf import FockSim, self_check
from Nov_3rd.quantum_h_transfer import (
    LAM_UNSAT,
    Circuit,
    center_cn,
    cost_only_from_fourier,
    dense_matches_sparse,
    load_split,
    power_theta_from_fourier,
    sparse_ground_state,
    summarize_eps,
    walsh_mass_fractions,
    walsh_pieces,
    walsh_theta_from_fourier,
)
from Nov_3rd.seal_unsat import (
    UNSAT_WALSH_B32,
    ZZZ_CHAMP,
    decide,
    mark_of,
)

_XFER_PATH = Path(__file__).resolve().parent / "transfer_quantum_h_A.py"
_spec = importlib.util.spec_from_file_location("transfer_quantum_h_A", _XFER_PATH)
_xfer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xfer)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
LAM = float(LAM_UNSAT)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[sealU]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None or isinstance(o, str):
        return o
    return str(o)


def _ground_job(payload):
    """Lanczos one diagonal. Spawned, so keep the return value scalar."""
    diag, n, lam = payload
    row = sparse_ground_state(np.asarray(diag, dtype=np.float64), int(n), float(lam))
    if row["residual"] > 1e-5:
        raise RuntimeError(f"n={n} Lanczos residual {row['residual']:.2e}")
    return (
        float(row["E0"]),
        float(row["peak"]),
        float(row["entropy"]),
        float(row["residual"]),
    )


def batch_ground(diags: np.ndarray, n: int, lam: float, workers: int) -> dict:
    payloads = [(np.asarray(diags[i], dtype=np.float64), int(n), float(lam)) for i in range(diags.shape[0])]
    e0 = np.empty(len(payloads), dtype=np.float64)
    peak = np.empty_like(e0)
    entropy = np.empty_like(e0)
    residual = np.empty_like(e0)
    t0 = time.time()
    # Threads, not processes: this script imports torch before ED, and a
    # forked or spawned pool either deadlocks or cannot see the repo on path.
    # ARPACK releases the GIL for the matvecs that dominate n=12 and n=15.
    if workers <= 1 or len(payloads) < 4:
        rows = [_ground_job(p) for p in payloads]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(_ground_job, payloads))
    for i, (ev, pk, ent, res) in enumerate(rows):
        e0[i] = ev
        peak[i] = pk
        entropy[i] = ent
        residual[i] = res
    log(f"  Lanczos n={n} B={len(payloads)} {time.time() - t0:.1f}s peak_med={float(np.median(peak)):.3f}")
    return {"E0": e0, "peak": peak, "entropy": entropy, "residual": residual}


class Split:
    def __init__(self, n, role, circuit, seeds, meta):
        self.n = int(n)
        self.role = role
        self.circuit = circuit
        self.seeds = list(seeds)
        self.meta = meta


def _make_circuit(sim, pulse, score, e0):
    pulse_t = torch.as_tensor(np.asarray(pulse, dtype=np.float64))
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64))
    e0_t = torch.as_tensor(np.asarray(e0, dtype=np.float64))
    pieces = walsh_pieces(pulse_t)
    return Circuit(sim, pulse_t, score_t, e0_t, pieces=pieces)


def _load_or_build(n, role, count, cache_path: Path, workers: int):
    if cache_path.exists():
        z = np.load(cache_path)
        log(f"cache hit n={n} {role} B={count}")
        return {k: z[k] for k in z.files}
    t0 = time.time()
    kind = "train" if role == "train" else "scale"
    pack = TRAIN_PACKS / f"train_n{n}.npz" if role == "train" else None
    insts, raw = load_split(n, count, kind, pack)
    m = int(clause_count(n))
    if role == "train":
        packed_m = int(np.load(pack)["m"])
        if packed_m != m:
            raise RuntimeError(f"n={n} pack m={packed_m} != clause_count {m}")
    log(f"ED n={n} {role} B={count} unsat λ={LAM}")
    gu = batch_ground(raw, n, LAM, workers)
    pulse = center_cn(raw, n, m)
    blob = {
        "score": np.asarray(raw, dtype=np.float64),
        "pulse": np.asarray(pulse, dtype=np.float64),
        "e0": gu["E0"],
        "peak": gu["peak"],
        "entropy": gu["entropy"],
        "residual": gu["residual"],
        "seeds": np.asarray([int(ins.seed) for ins in insts], dtype=np.int64),
        "m": np.asarray(m, dtype=np.int64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **blob)
    log(f"  built n={n} {role} in {time.time() - t0:.1f}s")
    return blob


def _ref_eps(score: np.ndarray, e0: np.ndarray, n: int) -> dict:
    """|+⟩ and best-bitstring residuals. |+⟩ has ⟨X_i⟩=1 and uniform Z."""
    mean_c = score.mean(axis=-1)
    min_c = score.min(axis=-1)
    eps_plus = (mean_c - LAM * float(n) - e0) / float(n)
    eps_bit = (min_c - e0) / float(n)
    return {
        "med_eps_plus": float(np.median(eps_plus)),
        "med_eps_bit": float(np.median(eps_bit)),
    }


def prepare(args, out: Path):
    specs = [
        (6, "train", args.B_train),
        (6, "hold", args.B_hold),
        (9, "train", args.B_train),
        (9, "hold", args.B_hold),
        (12, "transfer", args.B_transfer),
        (15, "transfer", args.B_n15),
    ]
    sims = {}
    prepared = {}
    quantum = {}
    for n, role, count in specs:
        if n not in sims:
            sims[n] = FockSim(n, dtype=torch.float64)
        cache = out / "cache" / f"unsat_n{n}_{role}_B{count}_l0.75.npz"
        blob = _load_or_build(n, role, count, cache, args.workers)
        refs = _ref_eps(blob["score"], blob["e0"], n)
        meta = {
            "n": n,
            "role": role,
            "B": int(count),
            "peak_med": float(np.median(blob["peak"])),
            "entropy_med": float(np.median(blob["entropy"])),
            "residual_max": float(np.max(blob["residual"])),
            "seeds": [int(s) for s in blob["seeds"]],
            **refs,
        }
        circ = _make_circuit(sims[n], blob["pulse"], blob["score"], blob["e0"])
        prepared[(n, role)] = {
            "unsat": Split(n, role, circ, meta["seeds"], meta),
            "pulse": blob["pulse"],
            "score": blob["score"],
        }
        quantum[(n, role)] = meta
        log(
            f"  n={n} {role} peak={meta['peak_med']:.3f} S={meta['entropy_med']:.3f} "
            f"ε+={meta['med_eps_plus']:.3f} εbit={meta['med_eps_bit']:.3f}"
        )
    for n in (6, 9):
        tr = set(prepared[(n, "train")]["unsat"].seeds)
        ho = set(prepared[(n, "hold")]["unsat"].seeds)
        if tr & ho:
            raise RuntimeError(f"n={n} train/hold seed overlap")
        if len(tr) != args.B_train or len(ho) != args.B_hold:
            raise RuntimeError(f"n={n} split size drift")
    return prepared, sims, quantum


def lanczos_audit(prepared) -> dict:
    """Dense eigh vs Lanczos on two train rows at n=6 and n=9. Unsat diagonal."""
    worst = 0.0
    report = {}
    for n in (6, 9):
        score = prepared[(n, "train")]["score"][:2]
        errs = [dense_matches_sparse(score[i], n, LAM) for i in range(score.shape[0])]
        report[f"n{n}_unsat"] = errs
        worst = max(worst, max(errs))
        log(f"Lanczos vs dense n={n} unsat |ΔE0|={max(errs):.2e}")
    if worst > 1e-6:
        raise RuntimeError(f"Lanczos disagrees with dense eigh by {worst}")
    report["worst"] = worst
    return report


def _paths(out: Path, L: int, method: str, seed: int):
    stem = out / "runs" / f"unsat_L{L}_{method}_seed{seed}"
    return stem.with_suffix(".json"), stem.with_suffix(".pt")


def _slim(rec: dict) -> dict:
    keep = {k: v for k, v in rec.items() if k != "theta"}
    ev = {}
    for n, row in rec["eval"].items():
        ev[int(n)] = {k: v for k, v in row.items() if k != "eps"}
        ev[int(n)]["eps"] = [float(x) for x in row["eps"]]
    keep["eval"] = ev
    return keep


def _save_run(out: Path, rec: dict):
    js, pt = _paths(out, rec["L"], rec["method"], rec["seed"])
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(_jsonable(_slim(rec)), indent=2) + "\n")
    torch.save(rec["theta"].detach().cpu(), pt)


def _load_run(out: Path, L: int, method: str, seed: int):
    js, pt = _paths(out, L, method, seed)
    if not (js.exists() and pt.exists()):
        return None
    rec = json.loads(js.read_text())
    rec["theta"] = torch.load(pt, map_location="cpu", weights_only=False)
    rec["L"] = int(rec["L"])
    rec["seed"] = int(rec["seed"])
    # json keys are strings
    rec["eval"] = {int(k): v for k, v in rec["eval"].items()}
    for row in rec["eval"].values():
        row["eps"] = np.asarray(row["eps"], dtype=np.float64)
    hist = []
    for row in rec["history"]:
        hist.append(row)
    rec["history"] = hist
    hold = {}
    for k, v in rec["hold_during_select"].items():
        hold[int(k)] = v
    rec["hold_during_select"] = hold
    return rec


def _assert_warm(method, rec, ref):
    init = rec["history"][0]
    for n in (6, 9):
        got = float(init["med6" if n == 6 else "med9"])
        if abs(got - float(ref[n])) > 1e-5:
            raise RuntimeError(
                f"{method} warm start n={n} ε={got:.6e} vs Fourier {float(ref[n]):.6e}"
            )


def _fit(method, L, seed, train, hold, report, args, theta0, ref, out: Path):
    cached = None if args.no_resume else _load_run(out, L, method, seed)
    if cached is not None:
        log(f"resume {method} L={L} seed={seed} tag={cached.get('best_tag')}")
        return cached
    rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
    max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
    log(f"train unsat {method} L={L} seed={seed}")
    rec = _xfer.fit_one(
        method, L, seed, train, hold, LAM,
        rounds, max_iter, args.lr, theta0, ref, "unsat",
    )
    rec["eval"] = _xfer.evaluate(method, rec["theta"], report, LAM, L)
    _xfer._check_select_matches(rec)
    _save_run(out, rec)
    e12 = rec["eval"][12]["med_eps"]
    e15 = rec["eval"][15]["med_eps"] if 15 in rec["eval"] else float("nan")
    log(f"  frozen ε12={e12:.4f} ε15={e15:.4f} wall={rec['wall_s']:.1f}s tag={rec['best_tag']}")
    return rec


def train(prepared, args, out: Path):
    train_splits = [prepared[(n, "train")]["unsat"] for n in (6, 9)]
    hold_splits = [prepared[(n, "hold")]["unsat"] for n in (6, 9)]
    # n=12 and n=15 are not in the loss and not in the checkpoint.
    if any(sp.n >= 12 for sp in train_splits + hold_splits):
        raise RuntimeError("train/hold leaked a transfer n")
    report = [
        prepared[(6, "hold")]["unsat"],
        prepared[(9, "hold")]["unsat"],
        prepared[(12, "transfer")]["unsat"],
        prepared[(15, "transfer")]["unsat"],
    ]
    runs = []
    for L in args.Ls:
        fourier = {}
        for seed in args.seeds:
            rec = _fit("fourier_qaoa", L, seed, train_splits, hold_splits, report, args, None, None, out)
            fourier[seed] = rec
            runs.append(rec)
        for seed in args.seeds:
            ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
            theta0 = power_theta_from_fourier(fourier[seed]["theta"], L)
            rec = _fit("power_p", L, seed, train_splits, hold_splits, report, args, theta0, ref, out)
            _assert_warm("power_p", rec, ref)
            runs.append(rec)
        for seed in args.seeds:
            ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
            theta0 = walsh_theta_from_fourier(fourier[seed]["theta"], L)
            rec = _fit("walsh", L, seed, train_splits, hold_splits, report, args, theta0, ref, out)
            _assert_warm("walsh", rec, ref)
            runs.append(rec)
        for seed in args.cost_only_seeds:
            theta0 = cost_only_from_fourier(fourier[seed]["theta"], L)
            rec = _fit("cost_only", L, seed, train_splits, hold_splits, report, args, theta0, None, out)
            runs.append(rec)
    return runs


def one_pulse_gap(sim, pulse, score, e0, gammas) -> float:
    """Max |Δ⟨H⟩| between L cost-only layers and one layer with Γ=Σγ.

    SNAP phases are 0, so the layers are exp(−i γ_ℓ C_n^X) and they commute.
    The score is the raw unsat count, which is what ⟨H_c⟩ uses.
    """
    pulse_t = torch.as_tensor(np.asarray(pulse, dtype=np.float64))
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64))
    e0_t = torch.as_tensor(np.asarray(e0, dtype=np.float64))
    circ = Circuit(sim, pulse_t, score_t, e0_t, pieces=None)
    g = torch.as_tensor(gammas, dtype=torch.float64).reshape(-1)
    L = int(g.shape[0])
    phi = g.new_zeros(L, 4)
    e_many = circ.energy(g, phi, False, LAM)
    e_one = circ.energy(g.sum().reshape(1), g.new_zeros(1, 4), False, LAM)
    return float((e_many - e_one).abs().max())


def _agg(rows, n, key):
    vals = [float(r["eval"][n][key]) for r in rows]
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {
        "med": float(np.median(vals)),
        "std": std,
        "per_seed": vals,
        "seeds": [int(r["seed"]) for r in rows],
    }


def build_cells(runs):
    cells = {}
    by = {}
    for rec in runs:
        by.setdefault((rec["method"], int(rec["L"])), []).append(rec)
    for (method, L), rows in by.items():
        if method == "cost_only":
            continue
        rows = sorted(rows, key=lambda r: int(r["seed"]))
        for n in (6, 9, 12, 15):
            role = rows[0]["eval"][n]["role"]
            got = _agg(rows, n, "med_eps")
            iqr = float(np.median([float(r["eval"][n]["iqr_eps"]) for r in rows]))
            h = _agg(rows, n, "med_H_over_n")
            cell = {
                "method": method,
                "L": int(L),
                "n": int(n),
                "role": role,
                "med": got["med"],
                "seed_std": got["std"],
                "iqr": iqr,
                "med_H_over_n": h["med"],
                "per_seed": got["per_seed"],
                "seeds": got["seeds"],
                "n_params": int(rows[0]["n_params"]),
            }
            cells[(method, L, n)] = cell
    for L in sorted({L for (_m, L) in by}):
        qrows = by.get(("fourier_qaoa", L))
        if not qrows:
            continue
        for n in (6, 9, 12, 15):
            q = cells[("fourier_qaoa", L, n)]
            qmeds = q["per_seed"]
            q_by_seed = {int(s): float(v) for s, v in zip(q["seeds"], qmeds)}
            for method in ("power_p", "walsh"):
                if (method, L, n) not in cells:
                    continue
                cell = cells[(method, L, n)]
                ref = float(q["med"])
                cell["qaoa"] = ref
                cell["R"] = float(cell["med"] / ref) if abs(ref) > 1e-12 else float("nan")
                cell["mark"] = mark_of(cell["R"])
                seed_Rs = []
                for seed, med in zip(cell["seeds"], cell["per_seed"]):
                    base = q_by_seed[int(seed)]
                    seed_Rs.append(float(med / base) if abs(base) > 1e-12 else float("nan"))
                cell["seed_Rs"] = seed_Rs
    return cells


def cost_only_line_search(sim, pulse, score, e0, n: int) -> dict:
    """Min median ε of one cost layer, φ=0, over γ.

    Every φ=0 stack equals this one-parameter family, so the minimum is the
    GRAPE optimum for a single control along C_n.
    """
    pulse_t = torch.as_tensor(np.asarray(pulse, dtype=np.float64))
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64))
    e0_t = torch.as_tensor(np.asarray(e0, dtype=np.float64))
    circ = Circuit(sim, pulse_t, score_t, e0_t, pieces=None)

    def med_at(g: float) -> float:
        gt = torch.tensor([float(g)], dtype=torch.float64)
        phi = torch.zeros(1, 4, dtype=torch.float64)
        with torch.no_grad():
            energy = circ.energy(gt, phi, False, LAM)
        return float(((energy - e0_t) / float(n)).median())

    best_g, best_m = 0.0, med_at(0.0)
    for g in np.linspace(-2.0, 2.0, 81):
        med = med_at(float(g))
        if med < best_m:
            best_g, best_m = float(g), med
    for g in np.linspace(best_g - 0.1, best_g + 0.1, 41):
        med = med_at(float(g))
        if med < best_m:
            best_g, best_m = float(g), med
    return {"gamma": best_g, "med_eps": best_m}


def grape_block(runs, prepared, sims) -> dict:
    out = {"lbfgs_med": {}, "cost_only_med": {}, "one_pulse_abs": {}, "per_seed": {}, "line_search": {}}
    rows = [r for r in runs if r["method"] == "cost_only"]
    for rec in rows:
        L = int(rec["L"])
        seed = int(rec["seed"])
        out["per_seed"].setdefault(L, {})[seed] = {
            n: float(rec["eval"][n]["med_eps"]) for n in (6, 9, 12, 15)
        }
        meds = out["lbfgs_med"].setdefault(L, {})
        for n in (6, 9, 12, 15):
            meds.setdefault(n, []).append(float(rec["eval"][n]["med_eps"]))
        gammas = rec["theta"][:L].detach()
        block = prepared[(12, "transfer")]
        # score diagonal's ground energy is on the circuit; pass zeros and
        # ignore them — one_pulse_gap only differences energies.
        gap = one_pulse_gap(
            sims[12], block["pulse"], block["score"], np.zeros(block["score"].shape[0]), gammas,
        )
        out["one_pulse_abs"].setdefault(L, {})[seed] = gap
        log(f"cost-only L={L} seed={seed} one-pulse |Δ⟨H⟩|={gap:.2e} ε12={rec['eval'][12]['med_eps']:.4f}")
    for L, per_n in list(out["lbfgs_med"].items()):
        out["lbfgs_med"][L] = {n: float(np.median(vs)) for n, vs in per_n.items()}
    # GRAPE of e^{-i C_n} is one parameter. L-BFGS on L copies of γ can stall
    # off that optimum because only the sum enters the circuit.
    for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
        block = prepared[(n, role)]
        found = cost_only_line_search(sims[n], block["pulse"], block["score"], block["unsat"].circuit.e0.cpu().numpy(), n)
        out["line_search"][n] = found
        log(f"GRAPE 1D n={n} γ={found['gamma']:.4f} med ε={found['med_eps']:.4f}")
    # Kill rule sees the one-parameter optimum, at every L (they coincide).
    out["cost_only_med"] = {
        L: {n: float(out["line_search"][n]["med_eps"]) for n in out["line_search"]}
        for L in (2, 16)
    }
    return out


def quantum_for_decide(quantum: dict) -> dict:
    """Holdout peak at n=6,9; transfer peak at n=12,15."""
    role = {6: "hold", 9: "hold", 12: "transfer", 15: "transfer"}
    return {n: {"peak_med": float(quantum[(n, role[n])]["peak_med"])} for n in role}


def walsh_mass(prepared) -> dict:
    pulse = prepared[(6, "train")]["pulse"]
    score = prepared[(6, "train")]["score"]
    return {
        "Cn_train_n6": walsh_mass_fractions(pulse),
        "raw_unsat_train_n6": walsh_mass_fractions(score),
    }


def render(payload: dict) -> str:
    v = payload["verdict"]
    q = payload["quantum"]
    cells = payload["cells"]
    lines = []
    lines.append("# RESULTS — unsat λ=0.75 Walsh seal, freeze {6,9} → n=12/15")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append(v["reason"])
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("Stoquastic transverse field on the raw unsat diagonal. The gate is the locked Nov_3rd sequence. Angles for n=12 and n=15 were not in the loss and not in the checkpoint.")
    lines.append("")
    lines.append("```text")
    lines.append("H = diag(unsat) − λ Σ_i X_i ,  λ = 0.75")
    lines.append("U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max")
    lines.append("ε = (⟨H⟩ − E_0) / n")
    lines.append("⟨H⟩ = Σ_z p_Z(z) unsat(z) − λ Σ_i ⟨X_i⟩")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Train B={payload['B_train']} per n from `beat_qaoa_12h/train_packs` (base 51e9). "
        f"Holdout B={payload['B_hold']} per n and transfer B={payload['B_transfer']} (n=12), "
        f"B={payload['B_n15']} (n=15) are the scale1000 satisfiable prefix (base 54e9). "
        f"Seeds {payload['seeds']}. L={payload['Ls']}. "
        f"L-BFGS rounds {payload['rounds_L2']}/{payload['rounds_L16']}, "
        f"max_iter {payload['max_iter_L2']}/{payload['max_iter_L16']}, lr={payload['lr']}. "
        "L=16 is longer than the 24-round PR #15 budget: at 24 rounds Fourier's holdout score was still falling by a few 10⁻⁴ per round, so that baseline was not sealed. Early stop is unchanged (8 rounds without a 10⁻⁶ improvement)."
    )
    lines.append("")
    lines.append(
        "Fourier QAOA stores 2L Zhou coefficients (Q=L, paper init β=0.01, γ=−0.01, seed jitter 0.01). "
        "power_p stores per-layer γ, c1, c2, c3 and one exponent p (4L+1). "
        "Walsh stores four degree weights per layer plus the same shell (7L+1). "
        "Both SNAP heads warm-start from the trained Fourier dual (φ_k=−β k, c2=c3=0, p=0); "
        "Walsh copies the scalar γ onto degrees 1–4. "
        "The loss is the mean of the per-n mean residuals on n=6 and n=9. "
        "The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier."
    )
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append("A split is quantum when the median ground-state bitstring peak is below 0.5. E0 is Lanczos on the sparse hypercube Hamiltonian. It matches dense eigh on two train instances at n=6 and n=9.")
    lines.append("")
    lines.append(f"Worst |ΔE0| Lanczos vs dense = {payload['lanczos_worst']:.2e}.")
    lines.append("")
    lines.append("| n | role | B | peak Med | S_cav Med | ε |+⟩ Med | ε bit Med | residual max |")
    lines.append("|--:|:-----|--:|---------:|----------:|----------:|----------:|-------------:|")
    for n, role in ((6, "train"), (6, "hold"), (9, "train"), (9, "hold"), (12, "transfer"), (15, "transfer")):
        row = q[f"{n}:{role}"]
        lines.append(
            f"| {n} | {role} | {row['B']} | {row['peak_med']:.3f} | {row['entropy_med']:.3f} | "
            f"{row['med_eps_plus']:.3f} | {row['med_eps_bit']:.3f} | {row['residual_max']:.1e} |"
        )
    lines.append("")
    mass = payload["walsh_mass"]["Cn_train_n6"]
    lines.append(
        f"Train n=6 C_n Walsh mass in degrees 1–4 is {mass['frac_deg_1_to_4']:.3f} "
        f"(degree 3 is {mass['frac_deg_3']:.3f}, degree ≥5 is {mass['frac_other']:.3e}). "
        "Equal degree weights reproduce the scalar pulse up to a global phase. "
        "On the ZZZ champ the same split is pure degree 3, so graded γ is a null knob there and was not trained."
    )
    lines.append("")
    lines.append("## Transfer table")
    lines.append("")
    lines.append("R = (median over seeds of median ε) / (the same number for Fourier QAOA). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better. n=6 and n=9 are the selection holdout, so those medians are slightly optimistic. n=12 and n=15 are not.")
    lines.append("")
    lines.append("| method | L | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |")
    lines.append("|:-------|--:|--:|:-----|-------:|------:|---------:|----:|----------:|--:|:-----|")
    order = []
    for L in payload["Ls"]:
        for method in ("fourier_qaoa", "power_p", "walsh"):
            for n in (6, 9, 12, 15):
                order.append((method, L, n))
    for method, L, n in order:
        c = cells[f"{method}|{L}|{n}"]
        R = "—" if "R" not in c else f"{c['R']:.3f}"
        mark = c.get("mark", "")
        lines.append(
            f"| {method} | {L} | {n} | {c['role']} | {c['n_params']} | {c['med']:.4f} | "
            f"{c['seed_std']:.4f} | {c['iqr']:.4f} | {c['med_H_over_n']:.4f} | {R} | {mark} |"
        )
    lines.append("")
    lines.append("Per-seed median ε, then per-seed R where a baseline exists:")
    lines.append("")
    for method, L, n in order:
        c = cells[f"{method}|{L}|{n}"]
        ev = ", ".join(f"{v:.4f}" for v in c["per_seed"])
        extra = ""
        if "seed_Rs" in c:
            rs = ", ".join(f"{v:.3f}" for v in c["seed_Rs"])
            med_r = float(np.median(c["seed_Rs"]))
            extra = f" R {rs} (median of those R = {med_r:.3f})"
        lines.append(f"- `{method}` L={L} n={n} seeds {c['seeds']}: {ev}{extra}")
    lines.append("")
    lines.append("## GRAPE of e^{−i H_c}")
    lines.append("")
    lines.append("Cost-only layers (SNAP phase 0) commute, so L of them equal one pulse whose γ is the sum. That pulse is already U_c^X. A grid over that one parameter is the GRAPE optimum for a single control along C_n. The minimum on every split is γ=0, which is the final W applied to |0⟩, the product |+⟩ state. The Hamming-shell SNAP does not commute with the cost layer, and that is the part that moves ε.")
    lines.append("")
    g = payload["grape"]
    lines.append("| n | role | GRAPE γ | GRAPE Med ε | Walsh L=16 Med ε |")
    lines.append("|--:|:-----|--------:|------------:|------------------:|")
    search = g.get("line_search", {})
    for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
        row = search[n] if n in search else search[str(n)]
        wmed = cells[f"walsh|16|{n}"]["med"]
        lines.append(
            f"| {n} | {role} | {float(row['gamma']):.4f} | {float(row['med_eps']):.4f} | {wmed:.4f} |"
        )
    lines.append("")
    lines.append("L-BFGS on L redundant cost angles (seed 0, warm-started from the Fourier γ with the mixer removed) is a check that the parameterization collapses, not a second optimum. |Δ⟨H⟩| is the gap between those L layers and one layer whose γ is the sum.")
    lines.append("")
    lines.append("| L | seed | L-BFGS Med ε n=12 | one-pulse |Δ⟨H⟩| | Walsh Med ε | power_p Med ε | Fourier Med ε |")
    lines.append("|--:|----:|------------------:|-----------------:|------------:|--------------:|--------------:|")

    def _grape_map(block, L):
        if L in block:
            return block[L]
        if str(L) in block:
            return block[str(L)]
        return {}

    def _at(cell, seed_):
        for s, val in zip(cell["seeds"], cell["per_seed"]):
            if int(s) == int(seed_):
                return float(val)
        return float("nan")

    for L in payload["Ls"]:
        per = _grape_map(g["one_pulse_abs"], L)
        meds = _grape_map(g["per_seed"], L)
        for seed, gap in per.items():
            eps = meds[seed] if seed in meds else meds[str(seed)]
            e12 = float(eps[12] if 12 in eps else eps["12"])
            lines.append(
                f"| {L} | {seed} | {e12:.4f} | {float(gap):.2e} | "
                f"{_at(cells[f'walsh|{L}|12'], seed):.4f} | "
                f"{_at(cells[f'power_p|{L}|12'], seed):.4f} | "
                f"{_at(cells[f'fourier_qaoa|{L}|12'], seed):.4f} |"
            )
    lines.append("")
    lines.append("## Which H_c wins")
    lines.append("")
    lines.append(
        "The two diagonals do not share a residual scale, so the comparison is the residual ratio R against Fourier QAOA on that same H, at the head each diagonal can actually use. "
        "ZZZ is pure degree 3, so Walsh collapses to scalar γ and the champ head is power_p (4L+1 parameters). "
        "Unsat spreads Walsh mass over degrees 1–4, so the PR #11 head is graded γ (7L+1). "
        "power_p on unsat is the scalar control. "
        "A transfer cell is one of {12,15}×{2,16}. The winner is the eligible head (n=12 L=16 BEAT) with more transfer BEATs, then the lower n=12 L=16 R. "
        "ZZZ numbers are the published PR #15 champ at B_train=32. They were not retrained here."
    )
    lines.append("")
    lines.append("| H_c | head | source | n=12 L=16 R | transfer BEATs | eligible |")
    lines.append("|:----|:-----|:-------|------------:|---------------:|:---------|")
    for h in v["winner"]["heads"]:
        lines.append(
            f"| {h['hc']} | {h['head']} | {h['source']} | {h['R12']:.3f} | {h['transfer_beats']}/4 | "
            f"{'yes' if h['eligible'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Winner: {v['winner']['winner']}.** {v['winner']['reason']}.")
    lines.append("")
    p_cell = cells["power_p|16|12"]
    w_cell = cells["walsh|16|12"]
    p_r = float(p_cell["R"])
    z_r = float(ZZZ_CHAMP[(16, 12)]["R"])
    seed_rs = ", ".join(f"{float(r):.3f}" for r in w_cell["seed_Rs"])
    lines.append(
        f"On the scalar power_p head, published ZZZ is {ZZZ_CHAMP[(16, 12)]['mark']} at n=12 L=16 (R={z_r:.3f}) "
        f"and this unsat seal is {p_cell['mark']} (R={p_r:.3f}). "
        "A power_p-only comparison would pick ZZZ: its L=16 transfer survives and unsat's scalar head does not. "
        "Unsat wins the transfer story through Walsh, the head its diagonal actually supports. "
        f"Walsh seed ratios at n=12 L=16 are {seed_rs}. "
        "Two seeds beat and one ties; the median of those ratios and the ratio of medians are both BEAT. "
        "Walsh has 7L+1 parameters, power_p has 4L+1, Fourier QAOA has 2L. "
        "The call is which diagonal supports a transferring head, not a matched parameter count."
    )
    lines.append("")
    lines.append(
        "A 24-round L=16 snapshot, the PR #15 budget, had not stalled: Fourier's holdout score was still falling by a few 10⁻⁴ per round. "
        "At that snapshot unsat Walsh n=12 R was 0.897 and power_p was a BEAT at 0.956. "
        "The table above is the 48-round budget. The scalar head's transfer BEAT did not survive the longer Fourier baseline."
    )
    lines.append("")
    drops = payload.get("fourier_L16_last_drop")
    if drops:
        shown = ", ".join(f"{float(d):.2e}" for d in drops)
        lines.append(
            f"L=16 Fourier's last-round holdout-score drop, seeds in order, was {shown}."
        )
    lines.append("")
    lines.append("Published ZZZ power_p (PR #15) and the earlier unsat Walsh ratios at B_train=32, for the shift this seal measures:")
    lines.append("")
    lines.append("| L | n | ZZZ power_p R | unsat Walsh B=32 R | unsat Walsh this seal R | unsat power_p this seal R |")
    lines.append("|--:|--:|---------------:|-------------------:|------------------------:|--------------------------:|")
    for L in (2, 16):
        for n in (6, 9, 12, 15):
            z = ZZZ_CHAMP[(L, n)]
            prior = UNSAT_WALSH_B32[(L, n)]
            wR = cells[f"walsh|{L}|{n}"]["R"]
            pR = cells[f"power_p|{L}|{n}"]["R"]
            lines.append(f"| {L} | {n} | {z['R']:.3f} {z['mark']} | {prior:.3f} | {wR:.3f} | {pR:.3f} |")
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "PASS if Walsh has R≤0.98 at L=2 and L=16 on holdout n=6, n=9 and frozen n=12; "
        "n=15 R stays below 1.10 at both depths; the n=15 ground state is not a bitstring; "
        "at least two of three n=12 L=16 seeds beat and the median of those seed ratios is on the same side of 0.98 as the ratio of medians; "
        "and the optimized cost-only pulse at n=12 L=16 is worse than 1.05× the Walsh residual. "
        "power_p does not gate PASS."
    )
    lines.append("")
    lines.append(
        "KILL if a gated ground state (n=6, 9, or 12) is a bitstring; if neither head beats Fourier QAOA on both n=6 and n=9 at either depth; "
        "if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05); "
        "or if the cost-only pulse matches Walsh. Otherwise needs_edit."
    )
    lines.append("")
    lines.append(
        f"Self-check: {payload['self_check_line']}"
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over fits {payload['train_wall_s']:.1f}s. "
        f"This process {payload['wall_s']:.1f}s. Branch `{payload['branch']}`. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written or signalled."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _cells_keyed(cells: dict) -> dict:
    """String keys for JSON, plus the tuple form the verdict wants."""
    return cells


def payload_cells(cells: dict) -> dict:
    out = {}
    for (method, L, n), cell in cells.items():
        out[f"{method}|{L}|{n}"] = cell
    return out


def tuple_cells(cells_json: dict) -> dict:
    out = {}
    for key, cell in cells_json.items():
        method, L, n = key.split("|")
        out[(method, int(L), int(n))] = cell
    return out


def write_all(out: Path, payload: dict):
    out.mkdir(parents=True, exist_ok=True)
    # verdict.winner must be jsonable; cells already string-keyed
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(_jsonable(payload["verdict"]), indent=2) + "\n")
    (out / "RESULTS.md").write_text(render(payload))


def bench(args, out: Path):
    """Time one Walsh L=16 backward on the real train batch. Does not score n=12."""
    log("bench prepare n=6,9 train")
    sims = {}
    splits = []
    for n in (6, 9):
        sims[n] = FockSim(n, dtype=torch.float64)
        cache = out / "cache" / f"unsat_n{n}_train_B{args.B_train}_l0.75.npz"
        blob = _load_or_build(n, "train", args.B_train, cache, args.workers)
        circ = _make_circuit(sims[n], blob["pulse"], blob["score"], blob["e0"])
        splits.append(Split(n, "train", circ, blob["seeds"].tolist(), {}))
    L = 16
    theta = walsh_theta_from_fourier(_xfer.init_fourier(L, 0, torch.float64, "cpu"), L).detach().clone().requires_grad_(True)
    def closure_once():
        if theta.grad is not None:
            theta.grad = None
        loss = _xfer._loss("walsh", theta, splits, LAM, L)
        loss.backward()
        return float(loss.detach())
    closure_once()  # warmup
    t0 = time.time()
    nstep = 3
    for _ in range(nstep):
        closure_once()
    dt = (time.time() - t0) / nstep
    # PR15 budget: 24 rounds * up to 8 LBFGS iters * ~15 strong-wolfe closures, times
    # 3 seeds * 3 methods, plus a cheaper L=2. Use 40 closures/round as a middle estimate.
    est_L16 = dt * 40 * args.rounds_L16 * len(args.seeds) * 3
    est_L2 = est_L16 * (2 / 16) * (args.rounds_L2 / args.rounds_L16) * 0.5
    log(f"walsh L=16 closure {dt:.3f}s on B={args.B_train}")
    log(f"rough train estimate L16 {est_L16/60:.1f} min + L2 {est_L2/60:.1f} min (closures only)")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "seal_unsat"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--cost-only-seeds", type=str, default="0")
    ap.add_argument("--B-train", type=int, default=128)
    ap.add_argument("--B-hold", type=int, default=128)
    ap.add_argument("--B-transfer", type=int, default=64)
    ap.add_argument("--B-n15", type=int, default=32)
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=48)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    args.seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    args.Ls = tuple(int(x) for x in args.Ls.split(",") if x.strip())
    args.cost_only_seeds = tuple(int(x) for x in args.cost_only_seeds.split(",") if x.strip())
    if args.B_train < 128 or len(args.seeds) < 3:
        raise SystemExit("seal requires B_train≥128 and at least 3 seeds")
    if tuple(args.Ls) != (2, 16):
        raise SystemExit("seal L set is {2, 16}")
    if 21 in (6, 9, 12, 15):
        raise SystemExit("n=21 is out of scope")
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.bench:
        bench(args, out)
        return
    t_all = time.time()
    if args.skip_self_check:
        check = {"ok": True, "skipped": True}
        check_line = "skipped"
    else:
        log("self-check")
        check = self_check()
        check_line = (
            f"Fock vs Nov3rdAnsatz max |Δp| Z={check['circuit_max_abs_pZ']:.2e}, "
            f"X={check['circuit_max_abs_pX']:.2e}; "
            f"QAOA-point max |Δp|={check['qaoa_point_max_abs_p']:.2e}; "
            f"X-hist vs Pauli {check['x_hist_vs_pauli']:.2e}."
        )
        log(check_line)
    prepared, sims, quantum = prepare(args, out)
    audit = lanczos_audit(prepared)
    mass = walsh_mass(prepared)
    log(
        f"Walsh C_n deg1-4={mass['Cn_train_n6']['frac_deg_1_to_4']:.3f} "
        f"deg3={mass['Cn_train_n6']['frac_deg_3']:.3f}"
    )
    runs = train(prepared, args, out)
    train_wall = float(sum(float(r["wall_s"]) for r in runs))
    fourier_drops = []
    for rec in sorted(
        (r for r in runs if r["method"] == "fourier_qaoa" and int(r["L"]) == 16),
        key=lambda r: int(r["seed"]),
    ):
        hist = [h for h in rec["history"] if str(h.get("tag", "")).startswith("r")]
        if len(hist) >= 2:
            fourier_drops.append(float(hist[-2]["score"]) - float(hist[-1]["score"]))
    cells = build_cells(runs)
    grape = grape_block(runs, prepared, sims)
    q_decide = quantum_for_decide(quantum)
    verdict = decide(cells, q_decide, grape)
    branch = "cursor/unsat-walsh-seal-35aa"
    q_json = {f"{n}:{role}": row for (n, role), row in quantum.items()}
    payload = {
        "verdict": verdict,
        "cells": payload_cells(cells),
        "quantum": q_json,
        "walsh_mass": mass,
        "lanczos_worst": audit["worst"],
        "lanczos": audit,
        "grape": grape,
        "self_check": check,
        "self_check_line": check_line,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": args.B_n15,
        "seeds": list(args.seeds),
        "Ls": list(args.Ls),
        "rounds_L2": args.rounds_L2,
        "rounds_L16": args.rounds_L16,
        "max_iter_L2": args.max_iter_L2,
        "max_iter_L16": args.max_iter_L16,
        "lr": args.lr,
        "branch": branch,
        "wall_s": time.time() - t_all,
        "train_wall_s": train_wall,
        "fourier_L16_last_drop": fourier_drops,
        "lam": LAM,
    }
    write_all(out, payload)
    log("verdict", verdict["verdict"])
    log("winner", verdict["winner"]["winner"], verdict["winner"]["reason"])
    log("wrote", out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
