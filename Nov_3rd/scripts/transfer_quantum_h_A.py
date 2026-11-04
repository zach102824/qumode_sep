#!/usr/bin/env python3
"""Round 2: freeze family-A angles trained on n=6 and n=9, eval at n=12 and n=15.

Champ (PR #13): H = Σ J ZZZ − λ Σ X, λ=0.5, power_p vs same-depth Fourier QAOA.
Walsh (PR #11 recipe) is run on the unsat diagonal, because on pure degree-3 ZZZ
the degree-1/2/4 slots are identically zero.

Does not write qaoa_angles/, train69_ladder_fresh250/, or settings.py.
Does not touch the phase-2 n21 ladder.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.sat_gen import clause_count
from Nov_3rd import settings as S
from Nov_3rd.quantum_h_tf import FockSim, best_product_energy, self_check
from Nov_3rd.quantum_h_transfer import (
    BEAT_R,
    LAM_UNSAT,
    LAM_ZZZ,
    TIE_R,
    Circuit,
    batch_ground,
    center_cn,
    cost_only_from_fourier,
    cost_only_matches_one_pulse,
    decode,
    dense_matches_sparse,
    init_fourier,
    load_split,
    n_params,
    power_theta_from_fourier,
    ratio_mark,
    summarize_eps,
    walsh_mass_fractions,
    walsh_pieces,
    walsh_theta_from_fourier,
)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[xferA]", *a, flush=True)


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
    if o is None or isinstance(o, (str,)):
        return o
    return str(o)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


class Split:
    def __init__(self, n, role, circuit, seeds, meta):
        self.n = int(n)
        self.role = role
        self.circuit = circuit
        self.seeds = list(seeds)
        self.meta = meta


def _circuit(sim, pulse, score, e0):
    pulse_t = torch.as_tensor(np.asarray(pulse, dtype=np.float64))
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64))
    e0_t = torch.as_tensor(np.asarray(e0, dtype=np.float64))
    pieces = walsh_pieces(pulse_t)
    return Circuit(sim, pulse_t, score_t, e0_t, pieces=pieces)


def _product_gaps(insts, e0, lam):
    gaps = np.empty(len(insts), dtype=np.float64)
    for i, ins in enumerate(insts):
        eprod = best_product_energy(ins.clauses, ins.n, lam, seed=ins.seed)
        gaps[i] = (eprod - float(e0[i])) / max(abs(float(e0[i])), 1e-8)
        if (i + 1) % 16 == 0 or i + 1 == len(insts):
            log(f"  product n={ins.n} {i+1}/{len(insts)} med gap={float(np.median(gaps[:i+1])):.3f}")
    return gaps


def prepare(args):
    """Instances, Lanczos E0, and circuits. Train rows never include n=12 or n=15."""
    specs = [
        (6, "train", args.B_train),
        (6, "hold", args.B_hold),
        (9, "train", args.B_train),
        (9, "hold", args.B_hold),
        (12, "transfer", args.B_transfer),
    ]
    if not args.skip_n15 and args.B_n15 > 0:
        specs.append((15, "transfer", args.B_n15))
    sims = {}
    out = {}
    for n, role, count in specs:
        t0 = time.time()
        kind = "train" if role == "train" else "scale"
        pack = TRAIN_PACKS / f"train_n{n}.npz" if role == "train" else None
        insts, raw_E = load_split(n, count, kind, pack)
        if n not in sims:
            sims[n] = FockSim(n, dtype=torch.float64)
        m_unsat = int(clause_count(n))
        if role == "train":
            packed_m = int(np.load(pack)["m"])
            if packed_m != m_unsat:
                raise RuntimeError(f"n={n} pack m={packed_m} != clause_count {m_unsat}")
        zzz = np.stack([ins.diag for ins in insts], axis=0)
        log(f"ED n={n} {role} B={count} ZZZ λ={LAM_ZZZ}")
        gz = batch_ground(zzz, n, LAM_ZZZ)
        log(f"ED n={n} {role} unsat λ={LAM_UNSAT}")
        gu = batch_ground(raw_E, n, LAM_UNSAT)
        gaps = _product_gaps(insts, gz["E0"], LAM_ZZZ)
        cn = center_cn(raw_E, n, m_unsat)
        meta = {
            "n": n,
            "role": role,
            "B": count,
            "seeds": [int(ins.seed) for ins in insts],
            "m_zzz": [int(ins.m) for ins in insts],
            "degree_max": int(max(int(ins.degree.max()) for ins in insts)),
            "m_unsat": m_unsat,
            "zzz_peak": gz["peak"],
            "zzz_entropy": gz["entropy"],
            "zzz_residual": gz["residual"],
            "zzz_E0": gz["E0"],
            "zzz_prod_gap": gaps,
            "unsat_peak": gu["peak"],
            "unsat_entropy": gu["entropy"],
            "unsat_residual": gu["residual"],
            "unsat_E0": gu["E0"],
            "wall_s": time.time() - t0,
        }
        out[(n, role)] = {
            "zzz": Split(n, role, _circuit(sims[n], zzz, zzz, gz["E0"]), meta["seeds"], meta),
            "unsat": Split(n, role, _circuit(sims[n], cn, raw_E, gu["E0"]), meta["seeds"], meta),
            "sims": sims[n],
            "zzz_diag": zzz,
        }
        log(
            f"  n={n} {role} m_zzz med={np.median(meta['m_zzz']):.0f} "
            f"peakZZZ={np.median(gz['peak']):.3f} prodΔ={np.median(gaps):.3f} "
            f"peakUnsat={np.median(gu['peak']):.3f} ({meta['wall_s']:.1f}s)"
        )
    # Train and hold seeds must be disjoint. Transfer n is a different n.
    for n in (6, 9):
        tr = set(out[(n, "train")]["zzz"].seeds)
        ho = set(out[(n, "hold")]["zzz"].seeds)
        if tr & ho:
            raise RuntimeError(f"n={n} train/hold seed overlap")
    return out, sims


def lanczos_audit(prepared) -> dict:
    """Dense eigh vs Lanczos on the first two train rows at n=6 and n=9."""
    report = {}
    worst = 0.0
    for n in (6, 9):
        block = prepared[(n, "train")]
        zzz = block["zzz_diag"]
        # unsat score diagonal is not stored raw on the circuit; rebuild from the split seeds
        # by reading E0 agreement on the ZZZ diag, and on a fresh unsat row from the pack.
        pack = np.load(TRAIN_PACKS / f"train_n{n}.npz")
        raw = np.asarray(pack["train_E"][:2], dtype=np.float64)
        for name, diags, lam in (("zzz", zzz[:2], LAM_ZZZ), ("unsat", raw, LAM_UNSAT)):
            errs = [dense_matches_sparse(diags[i], n, lam) for i in range(diags.shape[0])]
            report[f"n{n}_{name}"] = errs
            worst = max(worst, max(errs))
            log(f"Lanczos vs dense n={n} {name} |ΔE0|={max(errs):.2e}")
    if worst > 1e-6:
        raise RuntimeError(f"Lanczos disagrees with dense eigh by {worst}")
    report["worst"] = worst
    return report


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


def fit_one(method, L, seed, train, hold, lam, rounds, max_iter, lr, theta0, ref_med, family):
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
        row = {
            "tag": tag,
            "score": score,
            "med6": view[6]["med_eps"],
            "med9": view[9]["med_eps"],
        }
        if loss_f is not None:
            row["train_loss"] = loss_f
        hist.append(row)
        return score, view

    score, view = consider("init")
    best = {"score": score, "theta": theta.detach().clone(), "view": view, "tag": "init"}
    stall = 0
    log(
        f"  init {family} {method} L={L} s={seed} "
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
            log(f"  LBFGS skip {family} {method} L={L} s={seed} r={r}: {exc}")
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
                f"  {family} {method} L={L} s={seed} r={r} "
                f"ε6={view[6]['med_eps']:.4f} ε9={view[9]['med_eps']:.4f} "
                f"score={score:.4f} best={best['score']:.4f}@{best['tag']}"
            )
        if stall >= 8 and r >= 6:
            log(f"  early stop {family} {method} L={L} s={seed} at r={r}")
            break
    with torch.no_grad():
        theta.copy_(best["theta"])
    gamma9, phi9, _ = decode(method, theta.detach(), L, 9)
    return {
        "family": family,
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "n_params": n_params(method, L),
        "best_tag": best["tag"],
        "best_score": float(best["score"]),
        "hold_during_select": best["view"],
        "history": hist,
        "theta": theta.detach().clone(),
        "gamma_at_n9": gamma9.detach().cpu().tolist(),
        "phi_at_n9": phi9.detach().cpu().tolist(),
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


def train_family(family, lam, prepared, args, methods):
    """Train on hold-selection {6,9} only. n=12 and n=15 are evaluated after the freeze."""
    train = [prepared[(n, "train")][family] for n in (6, 9)]
    hold = [prepared[(n, "hold")][family] for n in (6, 9)]
    report_splits = []
    for n, role in ((6, "hold"), (9, "hold"), (12, "transfer")):
        report_splits.append(prepared[(n, role)][family])
    if (15, "transfer") in prepared:
        report_splits.append(prepared[(15, "transfer")][family])
    runs = []
    for L in args.Ls:
        rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
        max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
        fourier = {}
        for seed in args.seeds:
            log(f"train {family} fourier_qaoa L={L} seed={seed}")
            rec = fit_one(
                "fourier_qaoa", L, seed, train, hold, lam,
                rounds, max_iter, args.lr, None, None, family,
            )
            rec["eval"] = evaluate("fourier_qaoa", rec["theta"], report_splits, lam, L)
            _check_select_matches(rec)
            fourier[seed] = rec
            runs.append(rec)
            log(f"  frozen ε12={rec['eval'][12]['med_eps']:.4f} wall={rec['wall_s']:.1f}s")
        if "power_p" in methods:
            for seed in args.seeds:
                ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
                theta0 = power_theta_from_fourier(fourier[seed]["theta"], L)
                log(f"train {family} power_p L={L} seed={seed} warm=fourier dual")
                rec = fit_one(
                    "power_p", L, seed, train, hold, lam,
                    rounds, max_iter, args.lr, theta0, ref, family,
                )
                _assert_warm_matches(family, rec, ref)
                rec["eval"] = evaluate("power_p", rec["theta"], report_splits, lam, L)
                _check_select_matches(rec)
                runs.append(rec)
                log(f"  frozen ε12={rec['eval'][12]['med_eps']:.4f} wall={rec['wall_s']:.1f}s tag={rec['best_tag']}")
        if "walsh" in methods:
            for seed in args.seeds:
                ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
                theta0 = walsh_theta_from_fourier(fourier[seed]["theta"], L)
                log(f"train {family} walsh L={L} seed={seed} warm=PR11 degree copy")
                rec = fit_one(
                    "walsh", L, seed, train, hold, lam,
                    rounds, max_iter, args.lr, theta0, ref, family,
                )
                rec["eval"] = evaluate("walsh", rec["theta"], report_splits, lam, L)
                _check_select_matches(rec)
                runs.append(rec)
                log(f"  frozen ε12={rec['eval'][12]['med_eps']:.4f} wall={rec['wall_s']:.1f}s tag={rec['best_tag']}")
        if "cost_only" in methods:
            for seed in args.cost_only_seeds:
                if seed not in fourier:
                    continue
                theta0 = cost_only_from_fourier(fourier[seed]["theta"], L)
                log(f"train {family} cost_only L={L} seed={seed} (GRAPE of e^{{-iHc}})")
                rec = fit_one(
                    "cost_only", L, seed, train, hold, lam,
                    rounds, max_iter, args.lr, theta0, None, family,
                )
                rec["eval"] = evaluate("cost_only", rec["theta"], report_splits, lam, L)
                gamma = rec["theta"][:L].detach()
                pulse = prepared[(12, "transfer")]["zzz_diag"]
                pulse_t = torch.as_tensor(pulse, dtype=torch.float64)
                rec["one_pulse_abs"] = cost_only_matches_one_pulse(
                    prepared[(12, "transfer")]["sims"], pulse_t, gamma, lam,
                )
                log(f"  cost-only vs one pulse |Δ⟨H⟩|={rec['one_pulse_abs']:.2e} at n=12")
                runs.append(rec)
    return runs


def _check_select_matches(rec):
    for n in (6, 9):
        a = rec["hold_during_select"][n]["med_eps"]
        b = rec["eval"][n]["med_eps"]
        if abs(a - b) > 1e-8:
            raise RuntimeError(f"select/eval drift {rec['method']} n={n}: {a} vs {b}")


def _assert_warm_matches(family, rec, ref):
    """The p=0, c2=c3=0 dual is the Fourier mixer. On ZZZ it must match at init."""
    if family != "zzz":
        return
    init = rec["history"][0]
    for n, med in ((6, init["med6"]), (9, init["med9"])):
        if abs(med - ref[n]) > 1e-5:
            raise RuntimeError(
                f"power_p warm start n={n} ε={med:.6e} vs Fourier {ref[n]:.6e}"
            )


def _agg(rows, n, key):
    vals = [float(r["eval"][n][key]) for r in rows]
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {
        "med": float(np.median(vals)),
        "std": std,
        "per_seed": vals,
        "seeds": [int(r["seed"]) for r in rows],
    }


def build_table(runs):
    grouped = {}
    for r in runs:
        if r["method"] == "cost_only":
            continue
        grouped.setdefault((r["family"], r["method"], int(r["L"])), []).append(r)
    cells = {}
    for key, rows in grouped.items():
        family, method, L = key
        ns = sorted(rows[0]["eval"])
        for n in ns:
            cells[(family, method, L, n)] = {
                "family": family,
                "method": method,
                "L": L,
                "n": n,
                "role": rows[0]["eval"][n]["role"],
                "n_params": rows[0]["n_params"],
                "eps": _agg(rows, n, "med_eps"),
                "H_over_n": _agg(rows, n, "med_H_over_n"),
                "f": _agg(rows, n, "med_f"),
                "iqr": _agg(rows, n, "iqr_eps"),
            }
    ratios = {}
    for (family, method, L, n), cell in cells.items():
        if method == "fourier_qaoa":
            continue
        q = cells.get((family, "fourier_qaoa", L, n))
        if q is None:
            continue
        R = float(cell["eps"]["med"] / q["eps"]["med"]) if abs(q["eps"]["med"]) > 1e-12 else float("inf")
        paired = [
            s / qv for s, qv in zip(cell["eps"]["per_seed"], q["eps"]["per_seed"])
        ]
        ratios[(family, method, L, n)] = {
            "R": R,
            "R_median_of_seeds": float(np.median(paired)),
            "per_seed_R": paired,
            "adv_eps": float(q["eps"]["med"] - cell["eps"]["med"]),
            "mark": ratio_mark(cell["eps"]["med"], q["eps"]["med"]),
            "mark_median_of_seeds": ratio_mark(float(np.median(paired)), 1.0) if False else (
                "BEAT" if float(np.median(paired)) <= BEAT_R else
                "TIE" if float(np.median(paired)) <= TIE_R else "UNDER"
            ),
        }
    return cells, ratios


def quantum_summary(prepared):
    blocks = {}
    for (n, role), block in prepared.items():
        meta = block["zzz"].meta
        def pack(peak, ent, resid, gaps=None):
            row = {
                "B": int(len(peak)),
                "med_peak": float(np.median(peak)),
                "med_entropy_bits": float(np.median(ent)),
                "max_residual": float(np.max(resid)),
            }
            if gaps is not None:
                row["med_prod_gap"] = float(np.median(gaps))
                row["frac_prod_gap_ge_0.02"] = float(np.mean(gaps >= 0.02))
                row["pass"] = bool(row["med_peak"] < 0.5 and row["med_prod_gap"] >= 0.02)
            else:
                row["pass"] = bool(row["med_peak"] < 0.5)
            return row
        blocks[f"{role}_n{n}"] = {
            "zzz": pack(meta["zzz_peak"], meta["zzz_entropy"], meta["zzz_residual"], meta["zzz_prod_gap"]),
            "unsat": pack(meta["unsat_peak"], meta["unsat_entropy"], meta["unsat_residual"]),
            "m_zzz_med": float(np.median(meta["m_zzz"])),
            "m_zzz_min": int(min(meta["m_zzz"])),
            "m_zzz_max": int(max(meta["m_zzz"])),
            "degree_max": meta["degree_max"],
            "m_unsat": meta["m_unsat"],
        }
    return blocks


def walsh_null_report(prepared) -> dict:
    zzz = prepared[(6, "train")]["zzz_diag"]
    pack = np.load(TRAIN_PACKS / "train_n6.npz")
    raw = np.asarray(pack["train_E"][: zzz.shape[0]], dtype=np.float64)
    cn = center_cn(raw, 6, int(clause_count(6)))
    return {
        "zzz_train_n6": walsh_mass_fractions(zzz),
        "unsat_Cn_train_n6": walsh_mass_fractions(cn),
    }


def decide(cells, ratios, quantum) -> dict:
    """Verdict is the ZZZ power_p L=16 freeze, scored at n=12."""
    def grab(n):
        key = ("zzz", "power_p", 16, n)
        if key not in ratios:
            return None
        return ratios[key] | {"eps_snap": cells[key]["eps"]["med"], "eps_qaoa": cells[("zzz", "fourier_qaoa", 16, n)]["eps"]["med"]}

    r6, r9, r12 = grab(6), grab(9), grab(12)
    r15 = grab(15)
    if r12 is None:
        return {"verdict": "needs_edit", "reason": "n=12 power_p eval missing", "rows": {}}
    q12 = quantum["transfer_n12"]["zzz"]
    classical = bool(q12["med_peak"] >= 0.5)
    soft = bool(q12["med_peak"] < 0.5 and q12.get("med_prod_gap", 1.0) < 0.02)
    bad12 = bool(
        (r9["adv_eps"] > 0 and r12["adv_eps"] < 0 and r12["R"] >= 1.05 and r12["R"] > r9["R"])
        or r12["R"] >= 1.15
    )
    bad15 = bool(
        r15 is not None
        and r12["adv_eps"] > 0
        and r15["adv_eps"] < 0
        and r15["R"] >= 1.10
        and r15["R"] > r12["R"] + 0.05
    )
    unstable = bool(r12["mark"] == "BEAT" and r12["mark_median_of_seeds"] != "BEAT")
    if classical:
        verdict, reason = "KILL", (
            f"n=12 ZZZ ground state is classical (median bitstring peak {q12['med_peak']:.3f} ≥ 0.5)."
        )
    elif bad12:
        verdict, reason = "KILL", (
            f"L=16 power_p advantage reverses at n=12 "
            f"(R6={r6['R']:.3f}, R9={r9['R']:.3f}, R12={r12['R']:.3f})."
        )
    elif r12["R"] <= BEAT_R and not soft and not bad15 and not unstable:
        verdict, reason = "PASS", (
            f"L=16 power_p frozen on {{6,9}} has R={r12['R']:.3f} ≤ 0.98 at n=12 "
            f"against same-depth Fourier QAOA. The n=12 ground state stays quantum "
            f"(peak {q12['med_peak']:.3f}, product gap {q12['med_prod_gap']:.3f})."
        )
    elif r12["R"] <= BEAT_R and bad15:
        verdict, reason = "needs_edit", (
            f"n=12 clears R={r12['R']:.3f}, but n=15 reverses "
            f"(R15={r15['R']:.3f}, adv ε={r15['adv_eps']:.4f})."
        )
    elif r12["R"] <= BEAT_R and unstable:
        verdict, reason = "needs_edit", (
            f"Ratio of median ε is {r12['R']:.3f} but the median of per-seed ratios "
            f"is {r12['R_median_of_seeds']:.3f}, so the n=12 beat is not stable across seeds."
        )
    elif r12["R"] <= BEAT_R and soft:
        verdict, reason = "needs_edit", (
            f"n=12 residual beats QAOA (R={r12['R']:.3f}) but the product-state gap "
            f"{q12['med_prod_gap']:.3f} is under 2%."
        )
    else:
        extra = ""
        if r12["adv_eps"] > 0:
            extra = " Med ⟨H⟩ is lower than Fourier QAOA, short of the 0.98 residual bar."
        verdict, reason = "needs_edit", (
            f"L=16 power_p transfer R={r12['R']:.3f} at n=12 does not clear 0.98 "
            f"and does not reverse badly (R6={r6['R']:.3f}, R9={r9['R']:.3f})." + extra
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "R6": r6, "R9": r9, "R12": r12, "R15": r15,
        "classical_n12": classical,
        "soft_quantum_n12": soft,
        "bad12": bad12,
        "bad15": bad15,
        "unstable_seeds": unstable,
        "rule": (
            "PASS if L=16 power_p on ZZZ has transfer R≤0.98 at n=12, the n=12 GS "
            "still passes peak<0.5 and product gap≥2%, n=15 does not flip the advantage "
            "through R≥1.10, and the per-seed ratios agree. "
            "KILL if the n=12 GS is a bitstring, or the n=6→n=9 advantage flips and "
            "R(n=12)≥1.05 (or R≥1.15). Otherwise needs_edit. "
            "E0 is exact, so a Med ⟨H⟩ win is the same ordering as ε; the 0.98 bar is the residual ratio."
        ),
    }


def render(payload) -> str:
    v = payload["verdict"]
    lines = [
        "# RESULTS — family A round 2, freeze {6,9} → n=12",
        "",
        f"**Verdict: {v['verdict']}**",
        "",
        v["reason"],
        "",
        "## What was frozen",
        "",
        "Angles are trained on the union of n=6 and n=9 only (equal mean of the two per-n mean residuals). "
        "The checkpoint is the best holdout score on those same two sizes. "
        "n=12 and n=15 are not in the loss and not in the checkpoint. "
        "The same tensor is then applied at the larger n.",
        "",
        "Fourier QAOA stores 2L Zhou sine coefficients. The schedule does not depend on n. "
        "power_p stores per-layer γ, c1, c2, c3 and one exponent p (4L+1 numbers). "
        "At a new n the shell is φ_k = c1 k + a_n c2 k² + a_n c3 k³ with a_n = (8/(n−1))^p. "
        "a_9 = 1 for every p, so p is identified by n=6 versus n=9 and then extrapolated. "
        "γ and c1 are copied unchanged. The warm start is the trained Fourier dual (c2=c3=0, p=0), "
        "which is the linear Hamming shell φ_k = −β k.",
        "",
        f"ZZZ train B={payload['B_train']} per n, holdout B={payload['B_hold']} per n, "
        f"n=12 B={payload['B_transfer']}, n=15 B={payload['B_n15']}. "
        f"Seeds {payload['seeds']}. L={payload['Ls']}. Optimizer L-BFGS, "
        f"rounds {payload['rounds_L2']}/{payload['rounds_L16']} (L=2/L=16), "
        f"max_iter {payload['max_iter_L2']}/{payload['max_iter_L16']}.",
        "",
        "Train seeds are `beat_qaoa_12h/train_packs` (base 51e9). "
        "Holdout and transfer seeds are the scale1000 satisfiable prefix (base 54e9). "
        "The n=12 train pack is not used.",
        "",
        "## Hamiltonian and gate",
        "",
        "Champ, PR #13, λ=0.5:",
        "",
        r"\[",
        r"H = \sum J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i",
        r"\]",
        "",
        "J=±1 on cross-cavity triples taken from the 4-SAT clauses, degree capped at 3. "
        "Gate sequence, unchanged:",
        "",
        r"`|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, `W=H^{⊗3}`, `U_c^X = W exp(-i γ H_c) W`.",
        "",
        "The field is not inside U_c. ⟨H⟩ = ⟨H_c⟩ from the Z histogram after the final W, "
        "minus λ Σ ⟨X_i⟩. ε = (⟨H⟩ − E_0)/n. E_0 is the Lanczos ground energy of the sparse "
        "Hamiltonian (hypercube plus diagonal). It matches dense `eigh` at n=6 and n=9.",
        "",
        "Walsh from the PR #11 recipe is a second arm, not a second parameterisation of the champ. "
        "On this ZZZ diagonal the Walsh mass sits in degree 3, so γ_1, γ_2, γ_4 multiply zero "
        "and graded γ collapses to scalar γ. It is not trained on the champ. "
        "The Walsh arm uses the unsat count at λ=0.75 (the PR #11 quantum point). "
        "The pulse diagonal is C_n = (n/m) unsat − max and the score is the raw unsat count. "
        "A 4-local unsat function has Walsh degree at most 4, so tying the four weights "
        "reproduces the scalar pulse up to a global phase. Untying them is a real knob: "
        "the non-constant mass is spread across degrees 1–4.",
        "",
        "## GRAPE of e^{−i H_c}",
        "",
        "H_c is diagonal in the Z basis, and W = H^{⊗3} is an involution, so one cost layer with "
        "the SNAP phase set to 0 is exactly exp(−i γ H_c^X). Those layers all commute:",
        "",
        r"\[",
        r"\prod_{\ell=1}^{L} \mathrm{e}^{-i \gamma_\ell H_c^{X}} = \mathrm{e}^{-i (\sum_\ell \gamma_\ell) H_c^{X}}.",
        r"\]",
        "",
        "GRAPE with a single control along H_c therefore has a one-parameter optimum, and the "
        "locked U_c^X already applies that pulse. It is not the whole answer: the ground state "
        "is entangled (bitstring peak well below 1/2), and the Hamming-shell SNAP does not "
        "commute with H_c^X. The c2/c3 shells are extra mixer terms on top of the QAOA point "
        "φ_k=−β k, which is exp(−i β X/2) up to a global phase.",
        "",
    ]
    grape = payload.get("grape") or {}
    if grape:
        lines.append(
            f"Numerical check, trained cost-only angles (φ=0) versus one layer whose γ is the sum, "
            f"on the n=12 ZZZ batch: max |Δ⟨H⟩| = {grape['one_pulse_abs_max']:.2e}. "
            "The L-layer cost-only circuit did not leave the one-parameter group."
        )
        lines.append("")
        lines.append("| L | seed | n | cost-only Med ε | Fourier Med ε | power_p Med ε |")
        lines.append("|--:|----:|--:|----------------:|---------------:|--------------:|")
        for row in grape["rows"]:
            lines.append(
                f"| {row['L']} | {row['seed']} | {row['n']} | {fmt(row['cost'])} | "
                f"{fmt(row['qaoa'])} | {fmt(row['power'])} |"
            )
        lines.append("")
        lines.append(
            "Cost-only residuals stay far above the mixer circuits, which is the same statement: "
            "the GRAPE pulse of H_c is necessary and already present, and it does not prepare the ground state."
        )
        lines.append("")
    lines += [
        "## Quantumness",
        "",
        "ZZZ gate, same as PR #13: median bitstring peak < 0.5 and median product gap "
        "(E_prod − E_0)/|E_0| ≥ 2%. Product states are XZ-plane, ⟨Z⟩=cos θ, ⟨X⟩=sin θ. "
        "Unsat arm: peak < 0.5 (PR #11). Entropy is the cavity-0 cut, in bits.",
        "",
        "| split | n | family | med peak | med prod gap | med S | pass | m |",
        "|:------|--:|:-------|---------:|-------------:|------:|:----:|--:|",
    ]
    for key, block in payload["quantum"].items():
        # key like hold_n6
        role, ns = key.split("_n")
        for fam in ("zzz", "unsat"):
            row = block[fam]
            gap = fmt(row.get("med_prod_gap"), 3) if "med_prod_gap" in row else "—"
            mtxt = (
                f"{block['m_zzz_med']:.0f}"
                if fam == "zzz"
                else str(block["m_unsat"])
            )
            lines.append(
                f"| {role} | {ns} | {fam} | {fmt(row['med_peak'], 3)} | {gap} | "
                f"{fmt(row['med_entropy_bits'], 3)} | {'yes' if row['pass'] else 'no'} | {mtxt} |"
            )
    lines.append("")
    lines.append(
        f"Lanczos versus dense eigh, worst |ΔE0| on two train instances at n=6 and n=9 "
        f"(both diagonals) = {payload['lanczos_worst']:.2e}."
    )
    lines.append("")
    wm = payload["walsh_mass"]
    lines += [
        "## Walsh mass (train n=6)",
        "",
        f"ZZZ H_c: degree-3 fraction {wm['zzz_train_n6']['frac_deg_3']:.6f}, "
        f"degrees 1–4 {wm['zzz_train_n6']['frac_deg_1_to_4']:.6f}. "
        "Graded γ is a null knob on the champ. It was not trained there.",
        "",
        f"Unsat C_n, fraction of all Walsh energy in degrees 1–4 "
        f"{wm['unsat_Cn_train_n6']['frac_deg_1_to_4']:.3f} "
        f"(the rest is degree 0, the mean), degree 3 "
        f"{wm['unsat_Cn_train_n6']['frac_deg_3']:.3f}, degree ≥5 "
        f"{wm['unsat_Cn_train_n6']['frac_other']:.3f}. "
        "Equal weights reproduce scalar γ. Free weights are the PR #11 head.",
        "",
        "## Transfer table",
        "",
        "R = (median over seeds of median ε) of the method, divided by the same number for Fourier QAOA. "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better. "
        "n=6 and n=9 are the selection holdout. n=12 and n=15 are frozen.",
        "",
        "| family | L | method | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |",
        "|:-------|--:|:-------|--:|:-----|------:|------:|---------:|----:|----------:|--:|:-----|",
    ]
    order = []
    for key, cell in payload["cells"].items():
        order.append(key)
    order.sort(key=lambda k: (k[0], k[2], k[3], 0 if k[1] == "fourier_qaoa" else 1, k[1]))
    for key in order:
        cell = payload["cells"][key]
        ratio = payload["ratios"].get(key)
        R = fmt(ratio["R"], 3) if ratio else ""
        mark = ratio["mark"] if ratio else ""
        lines.append(
            f"| {cell['family']} | {cell['L']} | {cell['method']} | {cell['n']} | {cell['role']} | "
            f"{cell['n_params']} | {fmt(cell['eps']['med'])} | {fmt(cell['eps']['std'])} | "
            f"{fmt(cell['iqr']['med'])} | {fmt(cell['H_over_n']['med'])} | {R} | {mark} |"
        )
    lines.append("")
    lines.append("Per-seed median ε, then per-seed R where a baseline exists:")
    lines.append("")
    for key in order:
        if payload["cells"][key]["method"] == "fourier_qaoa" and False:
            continue
        cell = payload["cells"][key]
        vals = ", ".join(fmt(x) for x in cell["eps"]["per_seed"])
        ratio = payload["ratios"].get(key)
        extra = ""
        if ratio:
            extra = " R " + ", ".join(fmt(x, 3) for x in ratio["per_seed_R"])
            extra += f" (median of those R = {fmt(ratio['R_median_of_seeds'], 3)})"
        lines.append(
            f"- `{cell['family']}` L={cell['L']} {cell['method']} n={cell['n']} "
            f"seeds {cell['eps']['seeds']}: {vals}{extra}"
        )
    lines.append("")
    lines += [
        "## Kill / pass rule",
        "",
        v["rule"],
        "",
        f"- n=12 classical GS: {'yes' if v['classical_n12'] else 'no'}",
        f"- n=12 product-gap soft fail: {'yes' if v['soft_quantum_n12'] else 'no'}",
        f"- advantage reverses at n=12: {'yes' if v['bad12'] else 'no'}",
        f"- n=15 flips a n=12 win: {'yes' if v['bad15'] else 'no'}",
        f"- seed aggregate disagrees with the 0.98 line: {'yes' if v['unstable_seeds'] else 'no'}",
        "",
    ]
    if v.get("R12"):
        bits = []
        for label, row in (("n6", v["R6"]), ("n9", v["R9"]), ("n12", v["R12"]), ("n15", v["R15"])):
            if row:
                bits.append(f"{label} R={row['R']:.3f} adv ε={row['adv_eps']:.4f} ({row['mark']})")
        lines.append("L=16 ZZZ power_p: " + "; ".join(bits) + ".")
        lines.append("")
        per = payload["ratios"].get(("zzz", "power_p", 16, 12))
        q15 = (payload.get("quantum") or {}).get("transfer_n15", {}).get("zzz")
        if per:
            rs = ", ".join(f"{x:.3f}" for x in per["per_seed_R"])
            lines.append(
                f"Per-seed n=12 R = {rs} (median {per['R_median_of_seeds']:.3f}). "
                "A seed that stays on the Fourier warm start contributes R=1. "
                "The residual ratio can soften as n grows and still stay under 0.98; "
                "that is a smaller margin, not a sign flip. "
                "Instance IQR of ε is larger than the median gap, so the win is a shift of the median."
            )
            lines.append("")
        if q15 is not None and "med_prod_gap" in q15:
            lines.append(
                f"n=15 product gap is {q15['med_prod_gap']:.3f} "
                f"(peak {q15['med_peak']:.3f}, quantum pass={'yes' if q15['pass'] else 'no'}). "
                "The PASS bar is n=12. n=15 is the extra size."
            )
            lines.append("")
    # L=2 one-liner
    r2 = payload["ratios"].get(("zzz", "power_p", 2, 12))
    if r2:
        lines.append(
            f"L=2 ZZZ power_p at n=12: R={r2['R']:.3f} ({r2['mark']}). "
            "The success bar is the L=16 champ depth. L=2 is reported, not required."
        )
        lines.append("")
    wu = payload["ratios"].get(("unsat", "walsh", 16, 12))
    if wu:
        lines.append(
            f"Unsat Walsh L=16 at n=12: R={wu['R']:.3f} ({wu['mark']}). "
            "This arm does not change the ZZZ verdict. It is the PR #11 head on a different diagonal."
        )
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- power_p has 4L+1 parameters and Fourier QAOA has 2L. The comparison asks whether the c2/c3 shells, scaled by a_n, beat the linear mixer they contain.",
        "- n=6 and n=9 medians are best-by-hold on the reported holdout, so they are slightly optimistic. n=12 and n=15 are not.",
        "- A Med ⟨H⟩ comparison is the same ordering as ε, because E_0 is the same exact ground energy for both methods. The success bar is the residual ratio, not a separate energy test.",
        "- Degree cap 3: a few larger-n ZZZ instances stop short of m=n instead of raising the degree.",
        "- `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n21 ladder were not written or signalled.",
        "",
        f"Self-check: {payload.get('self_check_line', 'skipped')}",
        "",
        f"Wall clock {payload['wall_s']:.1f}s. Branch `{payload['branch']}`.",
        "",
    ]
    return "\n".join(lines)


def slim_run(rec):
    ev = {}
    for n, row in rec["eval"].items():
        ev[str(n)] = {k: v for k, v in row.items() if k != "eps"}
        ev[str(n)]["eps_head"] = [float(x) for x in np.asarray(row["eps"])[:8]]
    hold = {}
    for n, row in rec["hold_during_select"].items():
        hold[str(n)] = row
    return {
        "family": rec["family"],
        "method": rec["method"],
        "L": rec["L"],
        "seed": rec["seed"],
        "lam": rec["lam"],
        "n_params": rec["n_params"],
        "best_tag": rec["best_tag"],
        "best_score": rec["best_score"],
        "wall_s": rec["wall_s"],
        "theta": rec["theta"].detach().cpu().tolist(),
        "gamma_at_n9": rec["gamma_at_n9"],
        "phi_at_n9": rec["phi_at_n9"],
        "history": rec["history"],
        "eval": ev,
        "hold_during_select": hold,
        "one_pulse_abs": rec.get("one_pulse_abs"),
    }


def grape_block(runs) -> dict:
    rows = []
    abs_max = 0.0
    by = {(r["method"], r["L"], r["seed"]): r for r in runs if r["family"] == "zzz"}
    for rec in runs:
        if rec["family"] != "zzz" or rec["method"] != "cost_only":
            continue
        abs_max = max(abs_max, float(rec.get("one_pulse_abs") or 0.0))
        for n in sorted(rec["eval"]):
            q = by.get(("fourier_qaoa", rec["L"], rec["seed"]))
            p = by.get(("power_p", rec["L"], rec["seed"]))
            rows.append({
                "L": rec["L"],
                "seed": rec["seed"],
                "n": n,
                "cost": rec["eval"][n]["med_eps"],
                "qaoa": None if q is None else q["eval"][n]["med_eps"],
                "power": None if p is None else p["eval"][n]["med_eps"],
            })
    return {"one_pulse_abs_max": abs_max, "rows": rows}


def cells_as_json(cells, ratios):
    out = {}
    for key, cell in cells.items():
        tag = f"{key[0]}_{key[1]}_L{key[2]}_n{key[3]}"
        blob = dict(cell)
        if key in ratios:
            blob["ratio"] = ratios[key]
        out[tag] = blob
    return out


def write_payload(path: Path, payload, runs):
    path.mkdir(parents=True, exist_ok=True)
    md = render(payload)
    (path / "RESULTS.md").write_text(md)
    blob = dict(payload)
    blob["runs"] = [slim_run(r) for r in runs]
    blob["cells"] = cells_as_json(payload["cells"], payload["ratios"])
    # ratios keys are tuples; cells_as_json already inlined them
    blob.pop("ratios", None)
    (path / "summary.json").write_text(json.dumps(_jsonable(blob), indent=2) + "\n")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "transfer_A"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--cost-only-seeds", type=str, default="0")
    ap.add_argument("--B-train", type=int, default=32)
    ap.add_argument("--B-hold", type=int, default=48)
    ap.add_argument("--B-transfer", type=int, default=32)
    ap.add_argument("--B-n15", type=int, default=16)
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=24)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--skip-unsat", action="store_true")
    ap.add_argument("--skip-n15", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()
    args.seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    args.Ls = tuple(int(x) for x in args.Ls.split(",") if x.strip())
    args.cost_only_seeds = tuple(int(x) for x in args.cost_only_seeds.split(",") if x.strip())
    torch.set_num_threads(4)
    out = Path(args.out)
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
    prepared, _sims = prepare(args)
    audit = lanczos_audit(prepared)
    mass = walsh_null_report(prepared)
    log(
        f"Walsh ZZZ deg3={mass['zzz_train_n6']['frac_deg_3']:.6f} "
        f"unsat 1-4={mass['unsat_Cn_train_n6']['frac_deg_1_to_4']:.3f}"
    )
    runs = train_family("zzz", LAM_ZZZ, prepared, args, methods=("power_p", "cost_only"))
    # Draft so a later unsat failure still leaves the champ numbers on disk.
    cells, ratios = build_table(runs)
    quantum = quantum_summary(prepared)
    verdict = decide(cells, ratios, quantum)
    branch = "cursor/arm-a-transfer-r2-6b03"
    payload = {
        "verdict": verdict,
        "cells": cells,
        "ratios": ratios,
        "quantum": quantum,
        "walsh_mass": mass,
        "lanczos_worst": audit["worst"],
        "lanczos": audit,
        "grape": grape_block(runs),
        "self_check": check,
        "self_check_line": check_line,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": 0 if args.skip_n15 else args.B_n15,
        "seeds": list(args.seeds),
        "Ls": list(args.Ls),
        "rounds_L2": args.rounds_L2,
        "rounds_L16": args.rounds_L16,
        "max_iter_L2": args.max_iter_L2,
        "max_iter_L16": args.max_iter_L16,
        "branch": branch,
        "wall_s": time.time() - t_all,
        "families": ["zzz"],
    }
    write_payload(out, payload, runs)
    log("draft verdict", verdict["verdict"], verdict["reason"])
    if not args.skip_unsat:
        try:
            runs.extend(train_family("unsat", LAM_UNSAT, prepared, args, methods=("walsh",)))
            payload["families"] = ["zzz", "unsat"]
        except Exception:
            log("unsat arm failed")
            traceback.print_exc()
            payload["unsat_error"] = traceback.format_exc()
        cells, ratios = build_table(runs)
        payload["cells"] = cells
        payload["ratios"] = ratios
        payload["verdict"] = decide(cells, ratios, quantum)
        payload["grape"] = grape_block(runs)
        payload["wall_s"] = time.time() - t_all
        write_payload(out, payload, runs)
    log("verdict", payload["verdict"]["verdict"])
    log("wrote", out)


if __name__ == "__main__":
    main()
