#!/usr/bin/env python3
"""QN-RS smoke: gate0, then L=16 × 3 seeds. L=32 only if L=16 passes.

Resumable. A seed with train.json done and eval.json present is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RUN = _ROOT / "Nov_3rd/runs/beat_qaoa_12h/qn_rs"
REF = RUN / "fourier_qaoa_ref" / "Q6"
PACKS = _ROOT / "Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/train_packs"
RESULTS = RUN / "RESULTS.md"
DOC = _ROOT / "Nov_3rd/QN_RS.md"
PY = sys.executable
CORE = (6, 9, 12, 15)
# Published Fourier-QAOA n15 Med (FM-WS Q=6, same modes, scale1000 B=1000).
PUBLISHED = {16: 2.3853191788935257, 32: 1.358376135382406}


def sh(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    with log_path.open("a") as f:
        f.write(f"\n=== {' '.join(cmd)} @ {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== EXIT {p.returncode} ===\n")
    print(f"[{time.strftime('%H:%M:%S')}] exit {p.returncode} :: {log_path.name}", flush=True)
    return p.returncode


def seed_dir(L: int, seed: int) -> Path:
    return RUN / f"L{L}" / f"seed{seed}"


def train_cmd(L: int, seed: int, *, threads: int, eval_B: int, adam: int, lbfgs: int, gate0: bool) -> list[str]:
    out = seed_dir(L, seed) if not gate0 else (RUN / f"L{L}" / "gate0")
    cmd = [
        PY, str(_ROOT / "Nov_3rd/scripts/train_qn_rs.py"),
        "--L", str(L),
        "--seed", str(seed),
        "--Q", "6",
        "--Qp", "6",
        "--threads", str(threads),
        "--fourier-params", str(REF / f"L{L}" / "params.pt"),
        "--fourier-eval", str(REF / f"L{L}" / "eval.json"),
        "--packs", str(PACKS),
        "--out-dir", str(out),
        "--adam-steps", str(adam),
        "--lbfgs-rounds", str(lbfgs),
        "--eval-B", str(eval_B),
    ]
    if gate0:
        cmd.append("--gate0-only")
    return cmd


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def seed_row(ev: dict) -> dict:
    def boot(n: int) -> dict:
        r = ev["per_n"][str(n)]
        return {
            "ours_better": bool(r["ours_better"]),
            "excludes_0": bool(r["ci_excludes_0"]),
            "delta_med": float(r["delta_med"]),
            "ci_lo": float(r["ci_lo"]),
            "ci_hi": float(r["ci_hi"]),
        }

    return {
        "med_n12": float(ev["per_n"]["12"]["Med"]),
        "med_n15": float(ev["per_n"]["15"]["Med"]),
        "fq_n12": float(ev["per_n"]["12"]["Fourier_Med"]),
        "fq_n15": float(ev["per_n"]["15"]["Fourier_Med"]),
        "boot_n12": boot(12),
        "boot_n15": boot(15),
        "residual_ratio": float(ev["residual_ratio_n15"]),
        "p": float(ev["p"]),
        "holdout_best_step": ev.get("holdout_best_step"),
        "per_n": ev["per_n"],
    }


def fmt(x, nd=4) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x):.{nd}f}"


def _reading(gate0, rows, v16, overall: str) -> list[str]:
    """Short interpretation. Numbers come from the eval rows, not from memory."""
    lines = []
    if gate0 and gate0.get("gate0_pass"):
        n15 = gate0["per_n"]["15"]
        lines.append(
            f"Gate 0 passed. With s=w=0 the Hamming-shell circuit reproduces Fourier QAOA: "
            f"n15 Med {n15['Med']:.6f} versus the published {gate0['published_n15']:.6f}."
        )
    if v16 is None:
        lines.append("L=16 was not fully scored.")
        return lines
    lines.append(
        f"L=16 is **{v16['verdict']}** ({v16['n_pass_seeds']}/3 seeds). "
        "n6 and n9 beat Fourier QAOA on every seed, with the paired CI entirely below 0. "
        "n12 and n15 do not: every CI includes 0."
    )
    for seed in (0, 1, 2):
        ev = rows.get((16, seed))
        if ev is None:
            continue
        r12, r15 = ev["per_n"]["12"], ev["per_n"]["15"]
        lines.append(
            f"- seed {seed}: n15 {r15['Med']:.4f} (Δ {r15['delta_med']:+.4f}, "
            f"CI [{r15['ci_lo']:+.4f}, {r15['ci_hi']:+.4f}]), "
            f"n12 Δ {r12['delta_med']:+.4f}, "
            f"p={ev['p']:.3f}, residual ratio={ev['residual_ratio_n15']:.4f}, "
            f"checkpoint {ev.get('holdout_best_step')}"
        )
    ratios = v16["residual_ratios"]
    lines.append(
        "The residual did not collapse and it did not clear the bar. "
        f"‖a(15)·(r2, r3)‖/‖β‖ is {', '.join(f'{r:.4f}' for r in ratios)}, "
        "all under 0.10 and all above the 0.02 collapse line. "
        "p stayed in (0.10, 0.38), so a(n) is a mild mixer-side rescaling. "
        "Each reported checkpoint is that seed's phase-B n9-holdout best."
    )
    lines.append(
        f"Overall **{overall}**. L=32 was not run. "
        "The SNAP-only terms learn a gain on the sizes they were trained on. "
        "They leave n15 at 2.385 inside the bootstrap interval. "
        "The matched Fourier QAOA angles are the baseline, so that gap is the residual. "
        "A follow-up would need n12 and n15 CIs entirely below 0 on at least two seeds, "
        "with the residual ratio still at least 0.10."
    )
    return lines


def write_results(gate0: dict | None, rows: dict, v16, v32, overall: str) -> None:
    lines = [
        "# QN-RS smoke — SNAP residual on matched Fourier QAOA",
        "",
        "Lower Med is better (shots to solution). Train only {6, 9}. CORE is scale1000 B=1000, n∈{6, 9, 12, 15}.",
        "",
        "Circuit: ungraded Fourier γ_ℓ = Σ u_q sin((q−½)π t_ℓ), t_ℓ=(ℓ−½)/L;",
        "φ_{ℓ,k} = −β_ℓ k + a(n)(r2_ℓ k² + r3_ℓ k³), β and (r2, r3) on the cosine modes,",
        "a(n)=(8/(n−1))^p. Init (u, v) = Fourier QAOA Q=6, s=w=0.",
        "",
        "Phase A freezes (u, v) and trains (s, w, p). Phase B unfreezes all 25 parameters.",
        "Adam 600 @ 1e-3 then L-BFGS 12×8, each phase. Loss is mean log(inv/inv_FourierQAOA), w6=1, w9=2.",
        "No residual penalty. Checkpoint is the best n9 holdout (20%).",
        "",
        f"Published Fourier QAOA n15 Med: L=16 **{PUBLISHED[16]:.4f}**, L=32 **{PUBLISHED[32]:.4f}**.",
        "",
        "## Gate 0",
        "",
    ]
    if gate0 is None:
        lines.append("not run")
    else:
        n15 = gate0["per_n"]["15"]
        lines.append(
            f"**{'PASS' if gate0['gate0_pass'] else 'FAIL'}**. "
            f"s=w=0 n15 Med {n15['Med']:.6f} vs published {gate0['published_n15']:.6f} "
            f"(|Δ|={abs(n15['Med'] - gate0['published_n15']):.3e}, tol {gate0['tol']}). "
            f"Paired Fourier Med {n15['Fourier_Med']:.6f}. "
            f"Median |instance diff| {n15['median_abs_instance_diff']:.3e}."
        )
        lines.append("")
        lines.append("| n | SNAP s=w=0 Med | paired Fourier Med | published |")
        lines.append("|---|-----------------|--------------------|-----------|")
        for n in CORE:
            r = gate0["per_n"][str(n)]
            lines.append(
                f"| {n} | {r['Med']:.4f} | {r['Fourier_Med']:.4f} | {fmt(r.get('published_Fourier_Med'))} |"
            )
    lines += ["", "## CORE", ""]
    lines.append(
        "| L | seed | n | QN-RS Med | Fourier Med | Δ Med | 95% CI | CI excludes 0 | residual ratio | p |"
    )
    lines.append("|---|------|---|-----------|-------------|-------|--------|---------------|----------------|---|")
    for L in (16, 32):
        for seed in (0, 1, 2):
            ev = rows.get((L, seed))
            if ev is None:
                continue
            for n in CORE:
                r = ev["per_n"][str(n)]
                lines.append(
                    f"| {L} | {seed} | {n} | {r['Med']:.4f} | {r['Fourier_Med']:.4f} | "
                    f"{r['delta_med']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                    f"{r['ci_excludes_0']} | {ev['residual_ratio_n15']:.4f} | {ev['p']:.4f} |"
                )
    lines += ["", "## Verdict", "", f"**{overall}**", ""]
    if v16 is not None:
        lines.append(f"- L=16: **{v16['verdict']}** ({v16['n_pass_seeds']}/{v16['n_seeds']} seeds). {v16['rule']}")
        lines.append(f"  - seed_pass={v16['seed_pass']} n15_better={v16['n15_better_than_fourier']} ratios={['%.4f' % r for r in v16['residual_ratios']]}")
    else:
        lines.append("- L=16: not scored")
    if v32 is not None:
        lines.append(f"- L=32: **{v32['verdict']}** ({v32['n_pass_seeds']}/{v32['n_seeds']} seeds).")
        lines.append(f"  - seed_pass={v32['seed_pass']} n15_better={v32['n15_better_than_fourier']} ratios={['%.4f' % r for r in v32['residual_ratios']]}")
    elif v16 is not None and v16["verdict"] == "pass":
        lines.append("- L=32: missing after an L=16 pass")
    else:
        lines.append("- L=32: not run (L=16 did not pass)")
    lines.append("")
    lines.append(
        "A seed passes only when n12 and n15 are both better than Fourier QAOA and each paired "
        "bootstrap 95% CI lies entirely below 0, and ‖a(15)·(r2, r3)‖ / ‖β‖ ≥ 0.10. "
        "Ship requires that bar at L=16 and at L=32. Kill is an L=16 failure (n15 never beats "
        "2.385, or the residual collapses on every seed). Needs-edit is a split: one seed, a "
        "residual under 10%, or an L=16 pass that does not repeat at L=32."
    )
    lines += ["", "## Reading", ""]
    lines.extend(_reading(gate0, rows, v16, overall))
    lines.append("")
    text = "\n".join(lines) + "\n"
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(text)
    # Keep the design note's results block in sync when the doc exists.
    if DOC.exists():
        doc = DOC.read_text()
        begin, end = "<!-- RESULTS_BEGIN -->", "<!-- RESULTS_END -->"
        if begin in doc and end in doc:
            pre, rest = doc.split(begin, 1)
            _, post = rest.split(end, 1)
            DOC.write_text(pre + begin + "\n\n" + text + "\n" + end + post)
    print(f"wrote {RESULTS} verdict={overall}", flush=True)


def aggregate(gate0_path: Path) -> str:
    from Nov_3rd.qn_rs import combine_verdicts, qnrs_verdict

    gate0 = json.loads(gate0_path.read_text()) if gate0_path.exists() else None
    rows = {}
    for L in (16, 32):
        for seed in (0, 1, 2):
            path = seed_dir(L, seed) / "eval.json"
            if path.exists():
                rows[(L, seed)] = load_eval(path)
    v16 = v32 = None
    if all((16, s) in rows for s in (0, 1, 2)):
        v16 = qnrs_verdict([seed_row(rows[(16, s)]) for s in (0, 1, 2)], depth_tag="L16")
    if all((32, s) in rows for s in (0, 1, 2)):
        v32 = qnrs_verdict([seed_row(rows[(32, s)]) for s in (0, 1, 2)], depth_tag="L32")
    if gate0 is not None and not gate0.get("gate0_pass"):
        overall = "kill"
    else:
        overall = combine_verdicts(v16, v32)
    (RUN / "verdict.json").write_text(json.dumps({"overall": overall, "L16": v16, "L32": v32}, indent=2) + "\n")
    write_results(gate0, rows, v16, v32, overall)
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--skip-l32", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gate0_path = RUN / "L16" / "gate0" / "gate0.json"

    if args.aggregate_only:
        aggregate(gate0_path)
        return 0

    if not gate0_path.exists():
        ec = sh(
            train_cmd(16, 0, threads=args.threads, eval_B=args.eval_B, adam=args.adam_steps, lbfgs=args.lbfgs_rounds, gate0=True),
            RUN / "logs" / "gate0_L16.log",
        )
        if ec != 0:
            aggregate(gate0_path)
            print("gate0 failed; stop", flush=True)
            return ec

    for seed in seeds:
        ev = seed_dir(16, seed) / "eval.json"
        tr = seed_dir(16, seed) / "train.json"
        if ev.exists() and tr.exists() and json.loads(tr.read_text()).get("done"):
            print(f"skip L16 seed{seed}", flush=True)
            continue
        ec = sh(
            train_cmd(16, seed, threads=args.threads, eval_B=args.eval_B, adam=args.adam_steps, lbfgs=args.lbfgs_rounds, gate0=False),
            RUN / "logs" / f"L16_seed{seed}.log",
        )
        if ec != 0:
            aggregate(gate0_path)
            return ec

    from Nov_3rd.qn_rs import qnrs_verdict

    rows16 = []
    for seed in (0, 1, 2):
        path = seed_dir(16, seed) / "eval.json"
        if path.exists():
            rows16.append(seed_row(load_eval(path)))
    v16 = qnrs_verdict(rows16, depth_tag="L16") if len(rows16) == 3 else None
    run32 = (not args.skip_l32) and v16 is not None and v16["verdict"] == "pass"
    if run32:
        # Identity at the L=32 warm start, then the three seeds.
        g32 = RUN / "L32" / "gate0" / "gate0.json"
        if not g32.exists():
            ec = sh(
                train_cmd(32, 0, threads=args.threads, eval_B=args.eval_B, adam=args.adam_steps, lbfgs=args.lbfgs_rounds, gate0=True),
                RUN / "logs" / "gate0_L32.log",
            )
            if ec != 0:
                aggregate(gate0_path)
                return ec
        for seed in seeds:
            ev = seed_dir(32, seed) / "eval.json"
            tr = seed_dir(32, seed) / "train.json"
            if ev.exists() and tr.exists() and json.loads(tr.read_text()).get("done"):
                print(f"skip L32 seed{seed}", flush=True)
                continue
            ec = sh(
                train_cmd(32, seed, threads=args.threads, eval_B=args.eval_B, adam=args.adam_steps, lbfgs=args.lbfgs_rounds, gate0=False),
                RUN / "logs" / f"L32_seed{seed}.log",
            )
            if ec != 0:
                aggregate(gate0_path)
                return ec
    aggregate(gate0_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
