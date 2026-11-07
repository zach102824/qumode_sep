#!/usr/bin/env python3
"""μ=−0.25 intra-cavity XX on the unsat λ=0.75 Walsh champ.

Train on n=6 and n=9 only. Freeze the angles and score n=12 and n=15 against
same-depth Fourier QAOA on the same H, and against the published μ=0 unsat
Walsh ratios (PR #19). power_p is the scalar-γ control.

    H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −0.25

The gate is unchanged (BinH U_c^X, Hamming-shell SNAP). ⟨unsat⟩ is the Z
histogram. ⟨X⟩ and ⟨XX⟩ are the X histogram. n=21 is not loaded.
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

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.sat_gen import clause_count
from Nov_3rd import settings as S
from Nov_3rd.quantum_h_tf import FockSim
from Nov_3rd.quantum_h_transfer import (
    Circuit,
    center_cn,
    decode,
    init_fourier,
    load_split,
    n_params,
    power_theta_from_fourier,
    summarize_eps,
    walsh_mass_fractions,
    walsh_pieces,
    walsh_theta_from_fourier,
)
from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_rank2_xx import (
    LAM,
    MU,
    PUBLISHED_N12_L16,
    PUBLISHED_POWER_R,
    PUBLISHED_WALSH_R,
    ZXScore,
    batch_ground,
    decide_unsat_xx,
    dense_gap,
    plus_energy,
    statevector_energy,
)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[unsatXX]", *a, flush=True)


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


class Split:
    def __init__(self, n, role, circuit, seeds, meta):
        self.n = int(n)
        self.role = role
        self.circuit = circuit
        self.seeds = list(seeds)
        self.meta = meta


def _make_zx(sim, pulse, score, e0):
    pulse_t = torch.as_tensor(np.asarray(pulse, dtype=np.float64))
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64))
    e0_t = torch.as_tensor(np.asarray(e0, dtype=np.float64))
    pieces = walsh_pieces(pulse_t)
    circ = Circuit(sim, pulse_t, score_t, e0_t, pieces=pieces)
    return ZXScore(circ, MU)


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
    raw = np.asarray(raw, dtype=np.float64)
    seeds = np.asarray([int(ins.seed) for ins in insts], dtype=np.int64)
    log(f"ED n={n} {role} B={count} unsat λ={LAM} μ={MU}")
    gu = batch_ground(raw, n, LAM, MU, workers, seeds=seeds)
    pulse = center_cn(raw, n, m)
    blob = {
        "score": raw,
        "pulse": np.asarray(pulse, dtype=np.float64),
        "e0": gu["E0"],
        "peak": gu["peak"],
        "entropy": gu["entropy"],
        "residual": gu["residual"],
        "e_prod": gu["E_prod"],
        "prod_gap": gu["prod_gap"],
        "seeds": seeds,
        "m": np.asarray(m, dtype=np.int64),
        "lam": np.asarray(LAM, dtype=np.float64),
        "mu": np.asarray(MU, dtype=np.float64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **blob)
    log(f"  built n={n} {role} in {time.time() - t0:.1f}s peak_med={float(np.median(gu['peak'])):.3f}")
    return blob


def _ref_eps(score: np.ndarray, e0: np.ndarray, n: int) -> dict:
    mean_c = score.mean(axis=-1)
    min_c = score.min(axis=-1)
    eps_plus = (mean_c - LAM * float(n) + MU * float(n) - e0) / float(n)
    eps_bit = (min_c - e0) / float(n)
    # the formula above is plus_energy − E0, checked against plus_energy itself
    direct = np.array([plus_energy(score[i], n, LAM, MU) for i in range(score.shape[0])])
    if float(np.max(np.abs(direct - (mean_c - LAM * n + MU * n)))) > 1e-8:
        raise RuntimeError("plus energy drift")
    return {
        "med_eps_plus": float(np.median(eps_plus)),
        "med_eps_bit": float(np.median(eps_bit)),
    }


def prepare(args, out: Path):
    if any(int(n) == 21 for n in (6, 9, 12, 15)):
        raise RuntimeError("n=21 is out of scope")
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
        tag = f"m{MU:+.2f}".replace("+", "")
        cache = out / "cache" / f"unsat_xx_n{n}_{role}_B{count}_l0.75_{tag}.npz"
        blob = _load_or_build(n, role, count, cache, args.workers)
        if abs(float(blob["mu"]) - MU) > 1e-12 or abs(float(blob["lam"]) - LAM) > 1e-12:
            raise RuntimeError(f"cache μ/λ drift n={n} {role}")
        refs = _ref_eps(blob["score"], blob["e0"], n)
        denom = np.maximum(np.abs(blob["e0"]), 1e-8)
        gap = (blob["e_prod"] - blob["e0"]) / denom
        meta = {
            "n": n,
            "role": role,
            "B": int(count),
            "peak_med": float(np.median(blob["peak"])),
            "entropy_med": float(np.median(blob["entropy"])),
            "residual_max": float(np.max(blob["residual"])),
            "prod_gap_med": float(np.median(gap)),
            "seeds": [int(s) for s in blob["seeds"]],
            **refs,
        }
        zx = _make_zx(sims[n], blob["pulse"], blob["score"], blob["e0"])
        prepared[(n, role)] = {
            "split": Split(n, role, zx, meta["seeds"], meta),
            "pulse": blob["pulse"],
            "score": blob["score"],
        }
        quantum[(n, role)] = meta
        log(
            f"  n={n} {role} peak={meta['peak_med']:.3f} S={meta['entropy_med']:.3f} "
            f"gap={meta['prod_gap_med']:.3f} ε+={meta['med_eps_plus']:.3f} εbit={meta['med_eps_bit']:.3f}"
        )
    for n in (6, 9):
        tr = set(prepared[(n, "train")]["split"].seeds)
        ho = set(prepared[(n, "hold")]["split"].seeds)
        if tr & ho:
            raise RuntimeError(f"n={n} train/hold seed overlap")
        if len(tr) != args.B_train or len(ho) != args.B_hold:
            raise RuntimeError(f"n={n} split size drift")
    return prepared, sims, quantum


def audit(prepared) -> dict:
    """Lanczos vs dense, and histogram ⟨H⟩ vs ⟨ψ|H|ψ⟩, on two n=6 and n=9 rows."""
    report = {"worst_e0": 0.0, "worst_psi": 0.0}
    for n in (6, 9):
        score = prepared[(n, "train")]["score"][:2]
        errs = [dense_gap(score[i], n, LAM, MU) for i in range(score.shape[0])]
        report[f"n{n}_e0"] = errs
        report["worst_e0"] = max(report["worst_e0"], max(errs))
        log(f"Lanczos vs dense n={n} |ΔE0|={max(errs):.2e}")
    block = prepared[(6, "train")]
    score = block["score"][:2]
    pulse = block["pulse"][:2]
    e0 = np.zeros(2)
    sim = FockSim(6, dtype=torch.float64)
    zx = _make_zx(sim, pulse, score, e0)
    g = torch.tensor([0.21, -0.13, 0.04], dtype=torch.float64)
    phi = torch.tensor(
        [[0.0, 0.11, -0.04, 0.02], [0.0, -0.07, 0.03, 0.01], [0.0, 0.02, -0.05, 0.03]],
        dtype=torch.float64,
    )
    got = zx.energy(g, phi, False, LAM).detach().cpu().numpy()
    psi = zx.circ.final_amps(g, phi, False).detach().cpu().numpy()
    errs = [abs(float(got[i]) - statevector_energy(psi[i], score[i], 6, LAM, MU)) for i in range(2)]
    report["psi"] = errs
    report["worst_psi"] = max(errs)
    log(f"histogram vs ⟨ψ|H|ψ⟩ |Δ|={max(errs):.2e}")
    if report["worst_e0"] > 1e-6 or report["worst_psi"] > 1e-6:
        raise RuntimeError(f"score/ED audit failed {report}")
    # μ=0 histogram field matches the Pauli score on this same pulse
    zx0 = ZXScore(zx.circ, 0.0)
    # rebuild a circuit whose energy() is the Pauli path; zx.circ is that circuit
    pauli = zx.circ.energy(g, phi, False, LAM)
    hist0 = zx0.energy(g, phi, False, LAM)
    gap0 = float((pauli - hist0).abs().max())
    report["mu0_vs_pauli"] = gap0
    log(f"μ=0 histogram vs Pauli |Δ|={gap0:.2e}")
    if gap0 > 1e-7:
        raise RuntimeError(f"μ=0 histogram drifted from the Pauli score by {gap0}")
    return report


def _paths(out: Path, L: int, method: str, seed: int):
    # The μ tag contains a dot. Path.with_suffix would treat ".25_..." as the suffix
    # and collapse every method and seed onto one file.
    stem = out / "runs" / f"mu-0p25_L{int(L)}_{method}_seed{int(seed)}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".pt")


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
    if abs(float(rec.get("mu", 99)) - MU) > 1e-12:
        raise RuntimeError(f"resume μ drift in {js.name}")
    if rec.get("method") != method or int(rec.get("seed")) != int(seed) or int(rec.get("L")) != int(L):
        raise RuntimeError(
            f"resume identity drift in {js.name}: "
            f"{rec.get('method')} L={rec.get('L')} seed={rec.get('seed')}"
        )
    rec["theta"] = torch.load(pt, map_location="cpu", weights_only=False)
    rec["L"] = int(rec["L"])
    rec["seed"] = int(rec["seed"])
    rec["eval"] = {int(k): v for k, v in rec["eval"].items()}
    for row in rec["eval"].values():
        row["eps"] = np.asarray(row["eps"], dtype=np.float64)
    rec["hold_during_select"] = {int(k): v for k, v in rec["hold_during_select"].items()}
    return rec


def _loss(method, theta, batches, lam, L):
    terms = []
    for sp in batches:
        gamma, phi, graded = decode(method, theta, L, sp.n)
        energy = sp.circuit.energy(gamma, phi, graded, lam)
        terms.append(((energy - sp.circuit.e0) / float(sp.n)).mean())
    return torch.stack(terms).mean()


def _hold_view(method, theta, batches, lam, L):
    view = {}
    with torch.no_grad():
        for sp in batches:
            gamma, phi, graded = decode(method, theta, L, sp.n)
            energy = sp.circuit.energy(gamma, phi, graded, lam)
            view[sp.n] = summarize_eps(energy, sp.circuit.e0, sp.n, lam)
    return view


def _select_score(view, ref_med):
    if ref_med is None:
        return 0.5 * (view[6]["med_eps"] + view[9]["med_eps"])
    return max(view[n]["med_eps"] / max(float(ref_med[n]), 1e-8) for n in (6, 9))


def fit_one(method, L, seed, train, hold, lam, rounds, max_iter, lr, theta0, ref_med):
    dtype = torch.float64
    if theta0 is None:
        if method != "fourier_qaoa":
            raise RuntimeError(f"{method} needs a warm start")
        theta0 = init_fourier(L, seed, dtype, "cpu")
    theta = torch.nn.Parameter(theta0.detach().to(dtype=dtype).clone())
    t0 = time.time()
    hist = []

    def consider(tag, loss_f=None):
        view = _hold_view(method, theta, hold, lam, L)
        score = _select_score(view, ref_med)
        row = {"tag": tag, "score": score, "med6": view[6]["med_eps"], "med9": view[9]["med_eps"]}
        if loss_f is not None:
            row["train_loss"] = loss_f
        hist.append(row)
        return score, view

    score, view = consider("init")
    best = {"score": score, "theta": theta.detach().clone(), "view": view, "tag": "init"}
    stall = 0
    log(
        f"  init {method} L={L} s={seed} "
        f"ε6={view[6]['med_eps']:.4f} ε9={view[9]['med_eps']:.4f} score={score:.4f}"
    )
    for r in range(int(rounds)):
        opt = torch.optim.LBFGS(
            [theta],
            lr=float(lr),
            max_iter=int(max_iter),
            history_size=16,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = _loss(method, theta, train, lam, L)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 1e3)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach())
        except Exception as exc:
            log(f"  LBFGS skip {method} L={L} s={seed} r={r}: {exc}")
            stall += 1
            if stall >= 4:
                break
            continue
        if not math.isfinite(loss_f):
            stall += 1
            continue
        score, view = consider(f"r{r}", loss_f)
        improved = score < best["score"] - 1e-6
        if improved:
            best = {"score": score, "theta": theta.detach().clone(), "view": view, "tag": f"r{r}"}
            stall = 0
        else:
            stall += 1
        if r % 3 == 0 or r + 1 == rounds or improved:
            log(
                f"  {method} L={L} s={seed} r={r} "
                f"ε6={view[6]['med_eps']:.4f} ε9={view[9]['med_eps']:.4f} "
                f"score={score:.4f} best={best['score']:.4f}@{best['tag']}"
            )
        if stall >= 8 and r >= 6:
            log(f"  early stop {method} L={L} s={seed} at r={r}")
            break
    with torch.no_grad():
        theta.copy_(best["theta"])
    return {
        "family": "unsat_xx",
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "mu": float(MU),
        "kappa": 0.0,
        "n_params": n_params(method, L),
        "best_tag": best["tag"],
        "best_score": float(best["score"]),
        "hold_during_select": best["view"],
        "history": hist,
        "theta": theta.detach().clone(),
        "wall_s": time.time() - t0,
        "rounds_budget": int(rounds),
        "max_iter": int(max_iter),
        "lr": float(lr),
    }


def evaluate(method, theta, splits, lam, L):
    th = theta.detach()
    out = {}
    with torch.no_grad():
        for sp in splits:
            gamma, phi, graded = decode(method, th, L, sp.n)
            energy = sp.circuit.energy(gamma, phi, graded, lam)
            row = summarize_eps(energy, sp.circuit.e0, sp.n, lam)
            eps = ((energy - sp.circuit.e0) / float(sp.n)).cpu().numpy()
            row["eps"] = eps
            row["role"] = sp.role
            out[sp.n] = row
    return out


def _assert_warm(method, rec, ref):
    init = rec["history"][0]
    for n, key in ((6, "med6"), (9, "med9")):
        got = float(init[key])
        if abs(got - float(ref[n])) > 1e-5:
            raise RuntimeError(
                f"{method} warm start n={n} ε={got:.6e} vs Fourier {float(ref[n]):.6e}"
            )


def _check_select(rec):
    for n in (6, 9):
        a = float(rec["hold_during_select"][n]["med_eps"])
        b = float(rec["eval"][n]["med_eps"])
        if abs(a - b) > 1e-8:
            raise RuntimeError(f"select/eval drift {rec['method']} n={n}: {a} vs {b}")


def _fit(method, L, seed, train, hold, report, args, theta0, ref, out: Path):
    cached = None if args.no_resume else _load_run(out, L, method, seed)
    if cached is not None:
        log(f"resume {method} L={L} seed={seed} tag={cached.get('best_tag')}")
        return cached
    rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
    max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
    log(f"train {method} L={L} seed={seed}")
    rec = fit_one(method, L, seed, train, hold, LAM, rounds, max_iter, args.lr, theta0, ref)
    rec["eval"] = evaluate(method, rec["theta"], report, LAM, L)
    _check_select(rec)
    _save_run(out, rec)
    e12 = rec["eval"][12]["med_eps"]
    e15 = rec["eval"][15]["med_eps"]
    log(f"  frozen ε12={e12:.4f} ε15={e15:.4f} wall={rec['wall_s']:.1f}s tag={rec['best_tag']}")
    return rec


def train(prepared, args, out: Path):
    train_splits = [prepared[(n, "train")]["split"] for n in (6, 9)]
    hold_splits = [prepared[(n, "hold")]["split"] for n in (6, 9)]
    if any(sp.n >= 12 for sp in train_splits + hold_splits):
        raise RuntimeError("train/hold leaked a transfer n")
    report = [
        prepared[(6, "hold")]["split"],
        prepared[(9, "hold")]["split"],
        prepared[(12, "transfer")]["split"],
        prepared[(15, "transfer")]["split"],
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
    return runs


def _agg(rows, n, key):
    vals = [float(r["eval"][n][key]) for r in rows]
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"med": float(np.median(vals)), "std": std, "per_seed": vals, "seeds": [int(r["seed"]) for r in rows]}


def build_cells(runs):
    cells = {}
    by = {}
    for rec in runs:
        by.setdefault((rec["method"], int(rec["L"])), []).append(rec)
    for (method, L), rows in by.items():
        rows = sorted(rows, key=lambda r: int(r["seed"]))
        for n in (6, 9, 12, 15):
            got = _agg(rows, n, "med_eps")
            iqr = float(np.median([float(r["eval"][n]["iqr_eps"]) for r in rows]))
            h = _agg(rows, n, "med_H_over_n")
            cells[(method, L, n)] = {
                "method": method,
                "L": int(L),
                "n": int(n),
                "role": rows[0]["eval"][n]["role"],
                "med": got["med"],
                "seed_std": got["std"],
                "iqr": iqr,
                "med_H_over_n": h["med"],
                "per_seed": got["per_seed"],
                "seeds": got["seeds"],
                "n_params": int(rows[0]["n_params"]),
            }
    for L in sorted({L for (_m, L) in by}):
        qrows = by.get(("fourier_qaoa", L))
        if not qrows:
            continue
        for n in (6, 9, 12, 15):
            q = cells[("fourier_qaoa", L, n)]
            q_by_seed = {int(s): float(v) for s, v in zip(q["seeds"], q["per_seed"])}
            for method in ("power_p", "walsh"):
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
                published = PUBLISHED_WALSH_R if method == "walsh" else PUBLISHED_POWER_R
                cell["published_R"] = float(published[(int(L), int(n))])
                cell["drop_vs_published"] = float(cell["published_R"] - cell["R"])
    return cells


def cost_scan(zx: ZXScore, n: int) -> dict:
    """One-parameter cost pulse, φ=0. The XX term is only in the score."""

    def med_at(g: float) -> float:
        gt = torch.tensor([float(g)], dtype=torch.float64)
        phi = torch.zeros(1, 4, dtype=torch.float64)
        with torch.no_grad():
            energy = zx.energy(gt, phi, False, LAM)
        return float(((energy - zx.e0) / float(n)).median())

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


def quantum_for_decide(quantum: dict) -> dict:
    role = {6: "hold", 9: "hold", 12: "transfer", 15: "transfer"}
    return {
        n: {
            "peak_med": float(quantum[(n, role[n])]["peak_med"]),
            "prod_gap_med": float(quantum[(n, role[n])]["prod_gap_med"]),
        }
        for n in role
    }


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def render(payload: dict) -> str:
    v = payload["verdict"]
    q = payload["quantum"]
    cells = payload["cells"]
    lines = []
    lines.append("# RESULTS — unsat λ=0.75 + μ=−0.25 XX, Walsh freeze {6,9} → n=12/15")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append(v["reason"])
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "Same gate as the μ=0 unsat Walsh seal (PR #19). The intra-cavity XX term is scored, not compiled into U_c. "
        "Angles for n=12 and n=15 were not in the loss and not in the checkpoint. n=21 was not loaded."
    )
    lines.append("")
    lines.append("```text")
    lines.append("H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −0.25")
    lines.append("U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max")
    lines.append("⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩")
    lines.append("ε = (⟨H⟩ − E_0) / n")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Train B={payload['B_train']} per n from `beat_qaoa_12h/train_packs` (base 51e9). "
        f"Holdout B={payload['B_hold']} per n and transfer B={payload['B_transfer']} (n=12), "
        f"B={payload['B_n15']} (n=15) are the scale1000 satisfiable prefix (base 54e9). "
        f"Seeds {payload['seeds']}. L={payload['Ls']}. "
        f"L-BFGS rounds {payload['rounds_L2']}/{payload['rounds_L16']}, "
        f"max_iter {payload['max_iter_L2']}/{payload['max_iter_L16']}, lr={payload['lr']}. "
        "That is the PR #19 budget. Early stop is 8 rounds without a 10⁻⁶ improvement. "
        "Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. "
        "The loss is the mean of the per-n mean residuals on n=6 and n=9. "
        "The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier."
    )
    lines.append("")
    lines.append(
        f"The μ=0 comparison is the published PR #19 ratio of medians, not a retrain. "
        f"n=12 L=16 Walsh there is R={PUBLISHED_N12_L16:.3f}. "
        "A clear drop is at least 0.02 on that ratio and on the median of the three per-seed ratios. "
        "Absolute residuals are not on the same scale once μ changes E_0."
    )
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "E0 is Lanczos on the sparse Hamiltonian, diagonal plus single-X edges plus intra-cavity XX edges. "
        "A split is a bitstring when the median ground-state peak is at least 0.5. "
        "The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state found from |0⟩, |1⟩, |+⟩, the best bitstring, and three random starts, polished with L-BFGS."
    )
    lines.append("")
    lines.append(
        f"Worst |ΔE0| Lanczos vs dense = {payload['audit']['worst_e0']:.2e}. "
        f"Histogram vs ⟨ψ|H|ψ⟩ = {payload['audit']['worst_psi']:.2e}. "
        f"μ=0 histogram vs Pauli field = {payload['audit']['mu0_vs_pauli']:.2e}."
    )
    lines.append("")
    lines.append("| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |")
    lines.append("|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|")
    for n, role in ((6, "train"), (6, "hold"), (9, "train"), (9, "hold"), (12, "transfer"), (15, "transfer")):
        row = q[f"{n}:{role}"]
        lines.append(
            f"| {n} | {role} | {row['B']} | {row['peak_med']:.3f} | {row['entropy_med']:.3f} | "
            f"{row['prod_gap_med']:.3f} | {row['med_eps_plus']:.3f} | {row['med_eps_bit']:.3f} | "
            f"{row['residual_max']:.1e} |"
        )
    mass = payload["walsh_mass"]
    lines.append("")
    lines.append(
        f"Train n=6 C_n Walsh mass in degrees 1–4 is {mass['frac_deg_1_to_4']:.3f} "
        f"(degree 3 is {mass['frac_deg_3']:.3f}, degree ≥5 is {mass['frac_other']:.3e}). "
        "Equal degree weights still reproduce the scalar pulse up to a global phase when the mass above degree 4 vanishes. "
        "The XX term is not on the diagonal, so it does not change that split."
    )
    lines.append("")
    lines.append("## Transfer table")
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "μ=0 R is the published PR #19 unsat seal. Drop = (μ=0 R) − R. Positive means this run is lower. "
        "n=6 and n=9 are the selection holdout. n=12 and n=15 are not."
    )
    lines.append("")
    lines.append("| method | L | n | role | params | Med ε | seed std | R | mark | μ=0 R | drop |")
    lines.append("|:-------|--:|--:|:-----|-------:|------:|---------:|--:|:-----|------:|-----:|")
    order = []
    for L in payload["Ls"]:
        for method in ("fourier_qaoa", "power_p", "walsh"):
            for n in (6, 9, 12, 15):
                order.append((method, int(L), int(n)))
    for key in order:
        cell = cells[f"{key[0]}|{key[1]}|{key[2]}"]
        if key[0] == "fourier_qaoa":
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {cell['role']} | {cell['n_params']} | "
                f"{cell['med']:.4f} | {cell['seed_std']:.4f} | — |  | — | — |"
            )
        else:
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {cell['role']} | {cell['n_params']} | "
                f"{cell['med']:.4f} | {cell['seed_std']:.4f} | {cell['R']:.3f} | {cell['mark']} | "
                f"{cell['published_R']:.3f} | {cell['drop_vs_published']:+.3f} |"
            )
    lines.append("")
    lines.append("Per-seed median ε, then per-seed R where a baseline exists:")
    lines.append("")
    for key in order:
        cell = cells[f"{key[0]}|{key[1]}|{key[2]}"]
        meds = ", ".join(f"{x:.4f}" for x in cell["per_seed"])
        if "seed_Rs" in cell:
            rs = ", ".join(f"{x:.3f}" for x in cell["seed_Rs"])
            med_r = float(np.median(cell["seed_Rs"]))
            lines.append(
                f"- `{key[0]}` L={key[1]} n={key[2]} seeds {cell['seeds']}: {meds} R {rs} "
                f"(median of those R = {med_r:.3f})"
            )
        else:
            lines.append(f"- `{key[0]}` L={key[1]} n={key[2]} seeds {cell['seeds']}: {meds}")
    lines.append("")
    lines.append("## One-parameter cost pulse")
    lines.append("")
    lines.append(
        "SNAP phase 0. A stack of such layers is one pulse along C_n, because they commute. "
        "XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2]."
    )
    lines.append("")
    lines.append("| n | role | γ | Med ε | Walsh L=16 Med ε |")
    lines.append("|--:|:-----|--:|------:|------------------:|")
    for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
        row = payload["cost_scan"][str(n)]
        w = cells[f"walsh|16|{n}"]["med"]
        lines.append(f"| {n} | {role} | {row['gamma']:.4f} | {row['med_eps']:.4f} | {w:.4f} |")
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "PASS if the frozen n=12 L=16 Walsh ratio of medians is ≤ 0.98 and at least 0.02 below "
        f"PR #19's {PUBLISHED_N12_L16:.3f}, the median of the three per-seed ratios drops by that same 0.02, "
        "at least two seeds beat their own Fourier, the n=12 ground state has peak < 0.5 and product gap ≥ 0.02, "
        "and n=15 L=16 Walsh R stays below 1.10 with a non-bitstring ground state. "
        "power_p does not gate PASS."
    )
    lines.append("")
    lines.append(
        "A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. "
        "KILL if a gated ground state (n=6, 9, or 12) is a bitstring; if n=12 loses to Fourier and is not below 0.940; "
        "or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05)."
    )
    lines.append("")
    pp = cells["power_p|16|12"]
    lines.append(
        f"power_p control at n=12 L=16: R={pp['R']:.3f} ({pp['mark']}), "
        f"published μ=0 power_p R={pp['published_R']:.3f}, drop {pp['drop_vs_published']:+.3f}. "
        "PR #19's scalar head was a TIE at 0.990. This control says whether XX moves the scalar pulse, separate from Walsh."
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over fits {payload['opt_s']:.1f}s. "
        f"Branch `{payload['branch']}`. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _cells_public(cells: dict) -> dict:
    out = {}
    for (method, L, n), cell in cells.items():
        out[f"{method}|{L}|{n}"] = cell
    return out


def write_all(out: Path, payload: dict):
    out.mkdir(parents=True, exist_ok=True)
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    verdict = {
        "verdict": payload["verdict"]["verdict"],
        "null": bool(payload["verdict"].get("null", False)),
        "reason": payload["verdict"]["reason"],
        "R12": payload["verdict"].get("R12"),
        "published_R": PUBLISHED_N12_L16,
        "mu": MU,
        "lam": LAM,
    }
    (out / "verdict.json").write_text(json.dumps(_jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']}")


def bench_closure(prepared, args) -> float:
    """One Walsh L=16 loss+backward on the train pair. Returns seconds."""
    train = [prepared[(n, "train")]["split"] for n in (6, 9)]
    theta = torch.nn.Parameter(walsh_theta_from_fourier(init_fourier(16, 0, torch.float64, "cpu"), 16))
    t0 = time.time()
    loss = _loss("walsh", theta, train, LAM, 16)
    loss.backward()
    dt = time.time() - t0
    log(f"walsh L=16 closure {dt:.3f}s loss={float(loss.detach()):.4f} on B={args.B_train}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "unsat_rank2_mu"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--Ls", type=str, default="2,16")
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
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    args.Ls = [int(s) for s in args.Ls.split(",") if s != ""]
    if 21 in args.Ls or any(s == 21 for s in args.seeds):
        raise SystemExit("refusing n=21")
    if args.B_train < 128 or len(args.seeds) < 3:
        raise SystemExit("this comparison needs B_train≥128 and at least 3 seeds")
    if sorted(args.Ls) != [2, 16]:
        raise SystemExit("L must be 2 and 16")
    torch.set_num_threads(4)
    out = Path(args.out)
    log(f"out={out} μ={MU} λ={LAM} seeds={args.seeds}")
    t0 = time.time()
    prepared, _sims, quantum = prepare(args, out)
    audit_rep = audit(prepared)
    dt = bench_closure(prepared, args)
    # strong Wolfe often spends ~10 closures per LBFGS step, 8 steps, times rounds, times 3 methods, times seeds.
    # L=2 is cheaper by about the depth ratio.
    est = dt * 10 * args.max_iter_L16 * args.rounds_L16 * len(args.seeds) * 3
    est += est * (2 / 16) * (args.rounds_L2 / max(args.rounds_L16, 1))
    log(f"rough L=16+L=2 optimizer estimate {est/60:.1f} min if every round runs the full budget")
    if args.bench:
        log(f"bench done in {time.time() - t0:.1f}s")
        return
    runs = train(prepared, args, out)
    cells = build_cells(runs)
    qdec = quantum_for_decide(quantum)
    verdict = decide_unsat_xx(cells, qdec)
    log(f"VERDICT {verdict['verdict']}: {verdict['reason']}")
    scan = {}
    for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
        scan[n] = cost_scan(prepared[(n, role)]["split"].circuit, n)
        log(f"cost scan n={n} γ={scan[n]['gamma']:.4f} med ε={scan[n]['med_eps']:.4f}")
    mass = walsh_mass_fractions(prepared[(6, "train")]["pulse"])
    branch = "cursor/unsat-walsh-rank2-mu-5948"
    payload = {
        "verdict": verdict,
        "quantum": {f"{n}:{role}": row for (n, role), row in quantum.items()},
        "quantum_decide": qdec,
        "cells": _cells_public(cells),
        "cost_scan": {str(k): v for k, v in scan.items()},
        "walsh_mass": mass,
        "audit": audit_rep,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": args.B_n15,
        "seeds": args.seeds,
        "Ls": args.Ls,
        "rounds_L2": args.rounds_L2,
        "rounds_L16": args.rounds_L16,
        "max_iter_L2": args.max_iter_L2,
        "max_iter_L16": args.max_iter_L16,
        "lr": args.lr,
        "mu": MU,
        "lam": LAM,
        "opt_s": float(sum(float(r["wall_s"]) for r in runs)),
        "wall_s": time.time() - t0,
        "branch": branch,
        "published_walsh_R": {f"{L},{n}": r for (L, n), r in PUBLISHED_WALSH_R.items()},
        "published_power_R": {f"{L},{n}": r for (L, n), r in PUBLISHED_POWER_R.items()},
    }
    write_all(out, payload)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
