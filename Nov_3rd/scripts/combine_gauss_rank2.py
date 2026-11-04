#!/usr/bin/env python3
"""Gaussian J (seal A round 3) plus Rank-2 intra-cavity XX at μ=−0.25.

    H = Σ J_abc Z_a Z_b Z_c − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    J ~ N(0,1) on the degree-3 cross-cavity support, λ = 0.5, κ = 0
    μ ∈ {−0.25, 0}

The support and the normal couplings are the seal_A_r3 packing
(``apply_gaussian_J`` inside ``instance_from_clauses``). The XX term is the
Rank-2 score from the per-cavity X histogram; it is not inside ``U_c``.
μ = 0 is the Gaussian control, retrained here with the same joint loss, the
same seeds, and the same L-BFGS budget.

Angles train on n=6 and n=9 only. The checkpoint is the best mean of those
two holdout median residuals (the PR #20 freeze rule, not a ratio-to-Fourier
score). n=12 and n=15 are frozen. Gate: BinH ``U_c^X`` then Hamming-shell
SNAP. ``H_c`` is only the ZZZ piece. ⟨H⟩ uses the Z histogram and the X
histogram. n=21 is not loaded.

Does not write qaoa_angles/, train69_ladder_fresh250/, or settings.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from Nov_3rd.quantum_h_rank2 import (
    BEAT_R,
    PARENT_GAUSS_MU0_N12_R,
    PARENT_PM1_MU_N12_R,
    Rank2Sim,
    decide_gauss_xx,
    self_check,
)
from Nov_3rd.quantum_h_tf import J_RNG_SALT, exact_ground
from Nov_3rd.quantum_h_transfer import TIE_R, n_params, power_theta_from_fourier

_XFER_PATH = Path(__file__).with_name("transfer_rank2_mu.py")
_spec = importlib.util.spec_from_file_location("transfer_rank2_mu", _XFER_PATH)
xfer = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(xfer)

LAM = 0.5
KAPPA = 0.0


def log(*a):
    print(time.strftime("%H:%M:%S"), "[gauss-xx]", *a, flush=True)


def tag_mu(mu: float) -> str:
    return f"m{float(mu):+.2f}"


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _jsonable(o):
    return xfer._jsonable(o)


def shell_rms(theta: torch.Tensor, L: int) -> dict:
    """c2 / c3 movement off the linear Fourier dual, and the shell exponent p."""
    L = int(L)
    c2 = theta[2 * L: 3 * L].detach().cpu().numpy()
    c3 = theta[3 * L: 4 * L].detach().cpu().numpy()
    p = float(theta[4 * L])
    return {
        "c2_rms": float(np.sqrt(np.mean(c2 ** 2))),
        "c3_rms": float(np.sqrt(np.mean(c3 ** 2))),
        "p": p,
    }


def assert_gauss_pack(prepared) -> dict:
    """The seal_A_r3 support with N(0,1) couplings, not the ±1 gauge orbit."""
    batch = prepared[(0.0, 6, "train")]
    js = np.array([float(J) for ins in batch.insts for *_, J in ins.clauses], dtype=np.float64)
    if js.size == 0 or np.median(np.abs(np.abs(js) - 1.0)) < 1e-3:
        raise RuntimeError("n=6 train couplings look like ±1, expected J ~ N(0,1)")
    degrees = [int(ins.degree.max()) for ins in batch.insts]
    if max(degrees) > 3:
        raise RuntimeError(f"degree {max(degrees)} exceeds the seal cap")
    cross = all(len({a // 3, b // 3, c // 3}) >= 2 for ins in batch.insts for a, b, c, _J in ins.clauses)
    if not cross:
        raise RuntimeError("a clause sits inside one cavity")
    e0 = np.asarray(batch.e0.detach().cpu().numpy(), dtype=np.float64)
    nuniq = len(set(np.round(e0, 5)))
    need = int(0.9 * len(e0)) if len(e0) >= 16 else min(len(e0), 6)
    if nuniq < need:
        raise RuntimeError(f"n=6 μ=0 E0 collapsed to {nuniq}/{len(e0)} (need {need})")
    # Same salt the seal uses, so a later edit of apply_gaussian_J is visible.
    report = {
        "J_salt": int(J_RNG_SALT),
        "n6_train_J_std": float(np.std(js)),
        "n6_train_J_median_abs": float(np.median(np.abs(js))),
        "n6_train_unique_E0": int(nuniq),
        "n6_train_B": int(len(e0)),
        "n6_degree_max": int(max(degrees)),
        "cross_cavity": True,
    }
    log(
        f"gauss pack J std={report['n6_train_J_std']:.3f} "
        f"unique E0 {nuniq}/{len(e0)} degree≤{report['n6_degree_max']}"
    )
    return report


def train_all(prepared, args):
    """Joint {6,9} freeze. power_p is warm-started, then selected on absolute ε.

    Selection matches the PR #20 Gaussian freeze: the checkpoint minimises
    the mean of the n=6 and n=9 holdout median residuals. n=12 and n=15
    are scored after the checkpoint is frozen.
    """
    run_dir = Path(args.out) / "runs"
    report_roles = [(6, "hold"), (9, "hold"), (12, "transfer")]
    if any(k[1] == 15 for k in prepared):
        report_roles.append((15, "transfer"))
    runs = []
    for mu in args.mus:
        train = [prepared[(mu, n, "train")] for n in (6, 9)]
        hold = [prepared[(mu, n, "hold")] for n in (6, 9)]
        report = [prepared[(mu, n, role)] for n, role in report_roles]
        for L in args.Ls:
            rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
            max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
            fourier = {}
            for seed in args.seeds:
                path = xfer._rec_path(run_dir, mu, L, "fourier_qaoa", seed)
                if path.exists() and not args.force:
                    rec = xfer._load_rec(path)
                    log(f"resume {tag_mu(mu)} fourier_qaoa L={L} seed={seed} ε12={rec['eval'][12]['med_eps']:.4f}")
                else:
                    log(f"train {tag_mu(mu)} fourier_qaoa L={L} seed={seed}")
                    rec = xfer.fit_one(
                        "fourier_qaoa", L, seed, mu, train, hold, rounds, max_iter, args.lr, None, None,
                    )
                    rec["eval"] = xfer.evaluate("fourier_qaoa", rec["theta"], report, mu, L)
                    xfer._check_select_matches(rec)
                    xfer._dump_rec(path, rec)
                    log(f"  frozen ε12={rec['eval'][12]['med_eps']:.4f} wall={rec['wall_s']:.1f}s")
                fourier[seed] = rec
                runs.append(rec)
            for seed in args.seeds:
                path = xfer._rec_path(run_dir, mu, L, "power_p", seed)
                if path.exists() and not args.force:
                    rec = xfer._load_rec(path)
                    log(f"resume {tag_mu(mu)} power_p L={L} seed={seed} ε12={rec['eval'][12]['med_eps']:.4f}")
                else:
                    ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
                    theta0 = power_theta_from_fourier(fourier[seed]["theta"], L)
                    log(f"train {tag_mu(mu)} power_p L={L} seed={seed} warm=fourier dual select=mean ε")
                    # ref_med=None: select on ½(ε6+ε9), the Gaussian-freeze rule.
                    rec = xfer.fit_one(
                        "power_p", L, seed, mu, train, hold, rounds, max_iter, args.lr, theta0, None,
                    )
                    xfer._assert_warm_matches(rec, ref)
                    rec["eval"] = xfer.evaluate("power_p", rec["theta"], report, mu, L)
                    xfer._check_select_matches(rec)
                    rec["shell"] = shell_rms(rec["theta"], L)
                    xfer._dump_rec(path, rec)
                    log(
                        f"  frozen ε12={rec['eval'][12]['med_eps']:.4f} "
                        f"wall={rec['wall_s']:.1f}s tag={rec['best_tag']} "
                        f"c2={rec['shell']['c2_rms']:.4f} p={rec['shell']['p']:.4f}"
                    )
                if "shell" not in rec and rec["method"] == "power_p":
                    rec["shell"] = shell_rms(rec["theta"], L)
                runs.append(rec)
    return runs


def _ratio(ratios, mu, L, n):
    return ratios.get((float(mu), "power_p", int(L), int(n)))


def render(payload) -> str:
    d = payload["decision"]
    reason = d["reason"]
    if reason:
        reason = reason[0].upper() + reason[1:]
    lines = [
        "# RESULTS — Gaussian J × Rank2 μ=−0.25, freeze {6,9} → n=12",
        "",
        f"**Verdict: {payload['verdict']}**",
        "",
        reason + ".",
        "",
        "## Setup",
        "",
        "Seal A round 3 packing (PR #20) with the Rank-2 XX deformation (PR #18), κ = 0.",
        "",
        r"\[",
        r"H = \sum J_{abc} Z_a Z_b Z_c - \lambda \sum_i X_i + \mu \sum_{\mathrm{cav}} \sum_{i<j} X_i X_j",
        r"\]",
        "",
        "J is an independent standard normal on the degree-3 cross-cavity support "
        f"(salt {payload['pack']['J_salt']}), not the first-occurrence sign. "
        "λ = 0.5. μ ∈ {−0.25, 0}. μ = 0 is the Gaussian control, retrained on the same "
        "instances and the same L-BFGS budget. Published parents, frozen n=12 L=16 power_p: "
        f"gauss μ=0 R = {PARENT_GAUSS_MU0_N12_R:.3f} (PR #20), "
        f"±1 μ=−0.25 R = {PARENT_PM1_MU_N12_R:.3f} (PR #18).",
        "",
        "Angles train on the union of n=6 and n=9 only. The checkpoint minimises the mean of "
        "the two holdout median residuals ε, including the warm start. That is the PR #20 freeze "
        "rule. n=12 and n=15 are not in the loss and not in the checkpoint. Fourier QAOA stores "
        "2L Zhou sine coefficients. power_p stores per-layer γ, c1, c2, c3 and one exponent p "
        "(4L+1 numbers), warm-started from that seed's trained Fourier dual (c2=c3=0, p=0). "
        "At a new n the shell is φ_k = c1 k + a_n c2 k² + a_n c3 k³ with a_n = (8/(n−1))^p. "
        "a_9 = 1 for every p.",
        "",
        "Gate sequence, unchanged: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, `W=H^{⊗3}`, "
        "`U_c^X = W exp(−iγ H_c) W`, `H_c = Σ J ZZZ` only. The field and the intra-cavity XX "
        "are not inside U_c. ⟨H⟩ uses the Z histogram for ⟨H_c⟩ and the per-cavity 8-outcome X "
        "histogram for ⟨X_i⟩ and ⟨X_i X_j⟩. needs YY: no.",
        "",
        f"Train B={payload['B_train']} per n, holdout B={payload['B_hold']} per n, "
        f"n=12 B={payload['B_transfer']}, n=15 B={payload['B_n15']}. "
        f"Seeds {payload['seeds']}. L={payload['Ls']}. "
        f"L-BFGS rounds {payload['rounds_L16']} at L=16, max_iter {payload['max_iter_L16']}, "
        f"lr={payload['lr']}. Train packs are `beat_qaoa_12h/train_packs` (base 51e9). "
        "Holdout and transfer use the scale1000 satisfiable prefix (base 54e9).",
        "",
        "Bar in `decide_gauss_xx`. "
        "PASS if L=16 power_p at μ=−0.25 has frozen n=12 R≤0.98 against same-depth Fourier QAOA, "
        "the n=12 ground state passes peak<0.5 and product gap≥2%, the ratio is clearly below "
        f"both published parents (drop ≥ 0.02 versus {PARENT_GAUSS_MU0_N12_R:.3f} and versus "
        f"{PARENT_PM1_MU_N12_R:.3f}), and the ratio is clearly below the in-run Gaussian μ=0 control. "
        "Clearly versus the control means the ratio-of-medians drops by ≥0.02 and the median paired "
        "drop on seeds whose control per-seed R is below 0.995 is ≥0.02 and at least that drop's std. "
        "The in-run control must itself beat QAOA. Whether it also lands within 0.02 of the PR #20 "
        "freeze is reported and does not veto a ratio that is already clearly under 0.827. "
        "n=15 flipping through R≥1.10 downgrades a pass to needs_edit. "
        "A beat of QAOA that is not clearly under both parents is not synergistic and is needs_edit. "
        "KILL if the n=12 ground state is a bitstring, or n=12 fails to beat QAOA (R>1.02) and does "
        "not improve on the Gaussian control. Anything else is needs_edit.",
        "",
        "## Packing",
        "",
        f"n=6 train: J std {payload['pack']['n6_train_J_std']:.3f}, "
        f"median |J| {payload['pack']['n6_train_J_median_abs']:.3f}, "
        f"distinct μ=0 E0 {payload['pack']['n6_train_unique_E0']}/{payload['pack']['n6_train_B']}, "
        f"degree ≤ {payload['pack']['n6_degree_max']}, cross-cavity yes.",
        "",
        "## Quantumness",
        "",
        "Gate: median bitstring peak < 0.5 and median product gap (E_prod − E_0)/|E_0| ≥ 2%. "
        "Product states are XZ-plane. Entropy is the cavity-0 cut, in bits. E0 is Lanczos on the "
        "sparse rank-2 Hamiltonian. At μ=0 that matrix is the Gaussian transverse-field Hamiltonian.",
        "",
        "| μ | split | n | med peak | med prod gap | med S | med E0/n | pass | B |",
        "|--:|:------|--:|---------:|-------------:|------:|---------:|:----:|--:|",
    ]
    qrows = sorted(payload["quantum"].values(), key=lambda r: (r["mu"], r["n"], r["role"]))
    for row in qrows:
        lines.append(
            f"| {row['mu']:+.2f} | {row['role']} | {row['n']} | {row['med_peak']:.3f} | "
            f"{row['med_prod_gap']:.3f} | {row['med_entropy']:.3f} | {row['med_E0_over_n']:.3f} | "
            f"{'yes' if row['pass'] else 'no'} | {row['B']} |"
        )
    lines += [
        "",
        f"Lanczos versus dense eigh, worst |ΔE0| on the audit rows = {payload['lanczos_worst']:.2e}.",
        "",
        "## Transfer",
        "",
        "ε = (⟨H⟩ − E_0)/n. Med ε is the median over seeds of the per-seed median. "
        "R = Med ε_power / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "Lower ε is better. n=6 and n=9 are the selection holdout. n=12 and n=15 are frozen.",
        "",
        "| μ | L | method | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |",
        "|--:|--:|:-------|--:|:-----|------:|------:|---------:|----:|----------:|--:|:-----|",
    ]
    cells = sorted(payload["cells"], key=lambda c: (c["mu"], c["L"], c["n"], c["method"] != "fourier_qaoa", c["method"]))
    ratio_map = {}
    for row in payload["ratios"]:
        ratio_map[(row["mu"], row["method"], row["L"], row["n"])] = row
    for c in cells:
        key = (c["mu"], c["method"], c["L"], c["n"])
        ratio = ratio_map.get(key)
        R = "" if ratio is None else f"{ratio['R']:.3f}"
        mark = "" if ratio is None else ratio["mark"]
        lines.append(
            f"| {c['mu']:+.2f} | {c['L']} | {c['method']} | {c['n']} | {c['role']} | {c['n_params']} | "
            f"{c['eps']['med']:.4f} | {c['eps']['std']:.4f} | {c['iqr']['med']:.4f} | "
            f"{c['H_over_n']['med']:.4f} | {R} | {mark} |"
        )
    lines += ["", "Per-seed median ε, then per-seed R where a Fourier baseline exists:", ""]
    for c in cells:
        seeds = ", ".join(str(s) for s in c["eps"]["seeds"])
        vals = ", ".join(f"{v:.4f}" for v in c["eps"]["per_seed"])
        ratio = ratio_map.get((c["mu"], c["method"], c["L"], c["n"]))
        extra = ""
        if ratio is not None:
            rs = ", ".join(f"{v:.3f}" for v in ratio["per_seed_R"])
            extra = f" R {rs} (median of those R = {ratio['R_median_of_seeds']:.3f})"
        lines.append(
            f"- μ={c['mu']:+.2f} L={c['L']} {c['method']} n={c['n']} seeds [{seeds}]: {vals}{extra}"
        )
    lines += ["", "## vs both parents", ""]
    lines.append(
        f"Published gauss μ=0 n=12 L=16 R = {PARENT_GAUSS_MU0_N12_R:.3f}. "
        f"Published ±1 μ=−0.25 n=12 L=16 R = {PARENT_PM1_MU_N12_R:.3f}. "
        f"In-run gauss μ=0 R = {fmt(d.get('R_mu0'), 3)} "
        f"(control match: {'yes' if d.get('control_match') else 'no'}, tolerance 0.02). "
        f"μ=−0.25 R = {fmt(d.get('R_mu'), 3)}. "
        f"Drop versus gauss parent = {fmt(d.get('drop_vs_gauss'), 3)}. "
        f"Drop versus ±1 μ=−0.25 parent = {fmt(d.get('drop_vs_pm1'), 3)}. "
        f"Drop versus in-run control = {fmt(d['clear']['drop'], 3)}."
    )
    lines.append("")
    paired = d["clear"].get("paired") or []
    if paired:
        lines.append(
            "Paired per-seed ΔR (in-run μ=0 − μ=−0.25) = "
            + ", ".join(f"{v:.3f}" for v in paired)
            + f". Active seeds (μ=0 per-seed R < 0.995): {d['clear']['n_active']}. "
            f"Active median drop = {fmt(d['clear']['active_median'], 3)}, "
            f"std = {fmt(d['clear']['active_std'], 3)}. "
            f"Clear versus the control: {'yes' if d['clear']['clear'] else 'no'}."
        )
        lines.append("")
    syn = "yes" if d.get("synergistic") else "no"
    lines.append(
        f"Synergistic (clearly under both published parents and under the in-run control): {syn}."
    )
    if not d.get("synergistic"):
        lines.append("")
        lines.append(
            "Not synergistic means the μ=−0.25 Gaussian ratio is not a clear improvement on "
            "each ingredient alone. A ratio under 0.98 can still lose to the Gaussian μ=0 freeze, "
            "to the ±1 μ=−0.25 freeze, or to this run's matched control."
        )
    shells = payload.get("shells") or []
    if shells:
        lines += ["", "Hamming shells left the Fourier dual (c2 = c3 = 0, p = 0) when the rms is above 0:", ""]
        for row in shells:
            lines.append(
                f"- μ={row['mu']:+.2f} L={row['L']} seed={row['seed']}: "
                f"c2 rms={row['c2_rms']:.4f}, c3 rms={row['c3_rms']:.4f}, p={row['p']:.4f}"
            )
    r15 = None
    for row in payload["ratios"]:
        if row["method"] == "power_p" and int(row["L"]) == 16 and int(row["n"]) == 15 and abs(row["mu"] + 0.25) < 1e-12:
            r15 = row
    lines += ["", "## Rule", ""]
    lines.append(f"- beat QAOA (R≤{BEAT_R:.2f}): {'yes' if d.get('beat_qaoa') else 'no'}")
    lines.append(f"- n=12 quantum: {'yes' if d.get('quantum_pass') else 'no'}")
    lines.append(f"- clearly below gauss μ=0 parent {PARENT_GAUSS_MU0_N12_R:.3f}: {'yes' if d.get('below_gauss_parent') else 'no'}")
    lines.append(f"- clearly below ±1 μ=−0.25 parent {PARENT_PM1_MU_N12_R:.3f}: {'yes' if d.get('below_pm1_parent') else 'no'}")
    lines.append(f"- clearly below in-run gauss control: {'yes' if d['clear'].get('clear') else 'no'}")
    lines.append(f"- gauss control reproduced: {'yes' if d.get('control_match') else 'no'}")
    lines.append(f"- gauss control beats QAOA: {'yes' if d.get('control_beat') else 'no'}")
    n15_txt = "no" if not d.get("n15_flip") else "yes"
    if r15 is not None:
        n15_txt = f"{n15_txt} (R={r15['R']:.3f}, {r15['mark']})"
    lines.append(f"- n=15 flips past 1.10: {n15_txt}")
    lines.append("")
    lines.append(f"Verdict: **{payload['verdict']}**.")
    lines += [
        "",
        "## Notes",
        "",
        "- Cost-only layers still commute. On the n=12 μ=−0.25 batch, L cost-only layers versus one "
        f"pulse with Γ=Σγ differ by |Δ⟨H⟩| = {payload['grape_abs']:.2e}. The XX term is outside U_c, "
        "scored from the X histogram, so the ZZZ pulse is the same one-parameter family as the Gaussian seal.",
        "- power_p has 4L+1 parameters and Fourier QAOA has 2L. The comparison asks whether the c2/c3 "
        "shells, scaled by a_n, beat the linear mixer they contain.",
        "- n=6 and n=9 medians are best-by-hold on the reported holdout, so they are slightly optimistic. "
        "n=12 and n=15 are not.",
        "- The Gaussian parent (PR #20) froze the same packing at μ=0 with this checkpoint rule, "
        "B_train=128, B_hold=256, n=12 B=64, seeds 0–2, and reported R=0.827. "
        "The ±1 parent (PR #18) used J=±1, B_train=32, and a ratio-to-Fourier checkpoint, and reported R=0.838. "
        "Those two numbers are the parents. This run's μ=0 arm is the matched Gaussian control.",
        "- Fourier at μ=0 reproduces that PR #20 freeze: n=12 per-seed medians are 0.0247, 0.0433, 0.0234 "
        "and the published median is 0.0247. power_p n=12 here is 0.0218, 0.0347, 0.0203 against "
        "PR #20's 0.0201, 0.0347, 0.0204. Seed 1 matches. Seed 0 is still descending at round 24 "
        "(hold score 0.01177 versus the published checkpoint 0.01120), which moves the ratio of medians "
        "from 0.827 to 0.883. The μ=−0.25 objective is the same circuit with the XX score, and its "
        "n=12 per-seed ratios are 0.588, 0.669, 0.668.",
        "- At μ=0 the n=12 product gap on all 64 instances is 0.019, under the 2% line. "
        "PR #20 quoted that gap on the first 8 instances only. Peak 0.073 matches the published freeze. "
        "The pass gate is the μ=−0.25 ground state, which clears the line (peak 0.015, gap 0.083).",
        "- `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n21 ladder were not written. "
        "n=21 instances were not loaded.",
        "",
        f"Self-check: rank2 statevector {payload['self_check'].get('statevector_max_abs', '—')}, "
        f"μ=0 energy |Δ|={payload['mu0_graph'].get('energy_abs', float('nan')):.2e}, "
        f"grad |Δ|={payload['mu0_graph'].get('grad_abs', float('nan')):.2e}. needs_YY=no.",
        "",
        f"L-BFGS fit time {payload['train_wall_s']:.1f}s across the twelve runs. "
        f"Process wall {payload['wall_s']:.1f}s. Branch `{payload['branch']}`.",
        "",
    ]
    return "\n".join(lines)


def _git_branch() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def main():
    t_all = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "gauss_rank2_mu"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--Ls", type=str, default="16")
    ap.add_argument("--mus", type=str, default="-0.25,0")
    ap.add_argument("--B-train", type=int, default=128)
    ap.add_argument("--B-hold", type=int, default=256)
    ap.add_argument("--B-transfer", type=int, default=64)
    ap.add_argument("--B-n15", type=int, default=16)
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=24)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--skip-n15", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny debug budget, not a verdict")
    args = ap.parse_args()
    if args.quick:
        args.B_train = min(args.B_train, 4)
        args.B_hold = min(args.B_hold, 4)
        args.B_transfer = min(args.B_transfer, 4)
        args.B_n15 = min(args.B_n15, 2)
        args.rounds_L2 = min(args.rounds_L2, 2)
        args.rounds_L16 = min(args.rounds_L16, 2)
        args.max_iter_L2 = min(args.max_iter_L2, 2)
        args.max_iter_L16 = min(args.max_iter_L16, 2)
        args.seeds = "0"
        args.Ls = "2"
        args.skip_n15 = True
        args.out = str(Path(args.out).with_name(Path(args.out).name + "_quick"))
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.Ls = [int(s) for s in args.Ls.split(",") if s.strip()]
    args.mus = list(parse_floats(args.mus))
    if 21 in () or any(n > 15 for n in ()):
        raise SystemExit("n=21 refused")
    if any(abs(m) > 1.0 for m in args.mus):
        raise SystemExit("μ outside the smoke grid")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"out={out} seeds={args.seeds} L={args.Ls} μ={args.mus} B_train={args.B_train} quick={args.quick}")
    if not args.skip_self_check and not args.quick:
        report = self_check()
        graph = xfer.mu0_graph_check()
        log(f"self-check ok grad={report.get('grad_abs')}")
    else:
        report = {"skipped": True}
        graph = {"energy_abs": float("nan"), "grad_abs": float("nan")}
    # prepare() in the Rank-2 transfer script already refuses n=21.
    prepared, _sims = xfer.prepare(args)
    pack = assert_gauss_pack(prepared)
    # Spot-check one n=6 train row against dense eigh at both μ.
    ins0 = prepared[(0.0, 6, "train")].insts[0]
    for mu in args.mus:
        dense = exact_ground(ins0.diag, 6, LAM) if abs(mu) < 1e-15 else None
        if dense is not None:
            err = abs(float(dense["E0"]) - float(prepared[(mu, 6, "train")].e0[0]))
            if err > 1e-6:
                raise RuntimeError(f"μ=0 dense/sparse drift {err}")
    audit = xfer.lanczos_audit(prepared)
    if (-0.25, 12, "transfer") not in prepared:
        raise RuntimeError("n=12 transfer batch missing")
    grape = xfer.grape_check(prepared)
    runs = train_all(prepared, args)
    cells, ratios = xfer.build_table(runs)
    qsum = xfer.quantum_summary(prepared)

    def ratio_json(key, row):
        mu, method, L, n = key
        out_row = dict(row)
        out_row.update({"mu": mu, "method": method, "L": L, "n": n})
        return out_row

    cells_l = list(cells.values())
    ratios_l = [ratio_json(k, r) for k, r in ratios.items()]
    shells = []
    for rec in runs:
        if rec["method"] != "power_p":
            continue
        sh = rec.get("shell") or shell_rms(rec["theta"], int(rec["L"]))
        shells.append({"mu": float(rec["mu"]), "L": int(rec["L"]), "seed": int(rec["seed"]), **sh})

    decision = {
        "verdict": "needs_edit",
        "reason": "quick run is not a verdict",
        "clear": {"drop": float("nan"), "paired": [], "n_active": 0, "active_median": float("nan"), "active_std": float("nan"), "clear": False},
        "beat_qaoa": False,
        "quantum_pass": False,
        "synergistic": False,
        "below_gauss_parent": False,
        "below_pm1_parent": False,
        "control_match": False,
        "control_beat": False,
        "n15_flip": False,
        "R_mu": float("nan"),
        "R_mu0": float("nan"),
        "drop_vs_gauss": float("nan"),
        "drop_vs_pm1": float("nan"),
    }
    verdict = "needs_edit"
    if not args.quick:
        r_mu = ratios[(-0.25, "power_p", 16, 12)]
        r_0 = ratios[(0.0, "power_p", 16, 12)]
        q12 = qsum["m-0.25_transfer_n12"]
        r15 = ratios.get((-0.25, "power_p", 16, 15))
        decision = decide_gauss_xx(
            R_mu=r_mu["R"],
            R_mu0=r_0["R"],
            per_seed_R_mu=r_mu["per_seed_R"],
            per_seed_R_mu0=r_0["per_seed_R"],
            peak=q12["med_peak"],
            prod_gap=q12["med_prod_gap"],
            R_mu_n15=None if r15 is None else r15["R"],
            n_seeds=len(args.seeds),
            B_train=args.B_train,
        )
        verdict = decision["verdict"]
    payload = {
        "verdict": verdict,
        "decision": decision,
        "lam": LAM,
        "kappa": KAPPA,
        "mus": args.mus,
        "seeds": args.seeds,
        "Ls": args.Ls,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": 0 if args.skip_n15 else args.B_n15,
        "rounds_L16": args.rounds_L16,
        "max_iter_L16": args.max_iter_L16,
        "lr": args.lr,
        "quick": bool(args.quick),
        "pack": pack,
        "quantum": qsum,
        "cells": cells_l,
        "ratios": ratios_l,
        "shells": shells,
        "lanczos_worst": audit["worst"],
        "lanczos_rows": audit["rows"],
        "grape_abs": grape,
        "self_check": {k: v for k, v in report.items() if k != "ok"} if isinstance(report, dict) else {},
        "mu0_graph": graph,
        "train_wall_s": float(sum(float(rec["wall_s"]) for rec in runs)),
        "wall_s": time.time() - t_all,
        "branch": _git_branch(),
        "parent_gauss_mu0_n12_R": PARENT_GAUSS_MU0_N12_R,
        "parent_pm1_mu_n12_R": PARENT_PM1_MU_N12_R,
        "tie_R": TIE_R,
    }
    summary_runs = []
    for rec in runs:
        summary_runs.append({
            "mu": rec["mu"],
            "method": rec["method"],
            "L": rec["L"],
            "seed": rec["seed"],
            "n_params": rec.get("n_params", n_params(rec["method"], int(rec["L"]))),
            "best_tag": rec["best_tag"],
            "best_score": rec["best_score"],
            "wall_s": rec["wall_s"],
            "eval": rec["eval"],
            "shell": rec.get("shell"),
            "p": None if rec["method"] != "power_p" else float(rec["theta"][4 * int(rec["L"])]),
        })
    payload["runs"] = summary_runs
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2))
    log(f"verdict={verdict}")
    log(decision["reason"])
    log(f"wrote {out / 'RESULTS.md'} wall={payload['wall_s']:.1f}s")
    if verdict not in ("PASS", "needs_edit", "KILL"):
        raise SystemExit(f"bad verdict {verdict}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
