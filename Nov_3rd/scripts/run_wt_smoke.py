#!/usr/bin/env python3
"""WT smoke: L=2 (3 seeds) then L=16 vs champ, ungraded Fourier QAOA, graded Fourier QAOA.

Train {6, 9} only, fresh250 packs (disjoint from scale1000). CORE B=1000.
Writes Nov_3rd/runs/beat_qaoa_12h/wt/RESULTS.md and summary.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.data_io import log
from Nov_3rd.wt import snap_angles, wt_verdict

PACKS = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250" / "train_packs"
QAOA16 = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250" / "qaoa_angles"
CHAMP = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250" / "power_p" / "params_L16_seed0.pt"
CHAMP_EVAL = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250" / "power_p" / "eval_L16_seed0.json"
WALSH_EVAL = S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a" / "walsh_degree" / "eval_L2_seed1_B1000.json"
OUT = S.OUT / "beat_qaoa_12h" / "wt"
PY = sys.executable
SCR = Path(__file__).resolve().parent


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log("run", " ".join(cmd))
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(_ROOT))
    if proc.returncode != 0:
        tail = log_path.read_text()[-2500:]
        raise SystemExit(f"command failed ec={proc.returncode}\n{tail}")


def _med(payload: dict, n: int) -> float:
    return float(payload["per_n"][str(n)]["Med"])


def ensure_fourier_baseline() -> Path:
    """Copy the published Q=6 L=16 Fourier QAOA checkpoint into this run."""
    dest = OUT / "baselines" / "qaoa_fourier_Q6_L16.pt"
    if dest.exists():
        return dest
    src = Path("/tmp/fq_q6_l16.pt")
    if not src.exists():
        subprocess.check_call(
            [
                "git",
                "show",
                "origin/cursor/fm-ws-idea3-f13e:Nov_3rd/runs/beat_qaoa_12h/fm_ws/qaoa_fourier/Q6/L16/params.pt",
            ],
            cwd=str(_ROOT),
            stdout=open("/tmp/fq_q6_l16.pt", "wb"),
        )
        src = Path("/tmp/fq_q6_l16.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    return dest


def reeval_fourier_fix(threads: int, eval_B: int) -> dict:
    """Re-eval published angles. ``--skip-train`` looks for out-dir/params.pt."""
    out = OUT / "baselines" / "fourier_Q6_L16"
    eval_path = out / "eval.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    ckpt = ensure_fourier_baseline()
    if not (out / "params.pt").exists():
        shutil.copy(ckpt, out / "params.pt")
    run(
        [
            PY,
            str(SCR / "train_qaoa_fourier_graded.py"),
            "--Q", "6",
            "--L", "16",
            "--threads", str(threads),
            "--packs", str(PACKS),
            "--out-dir", str(out),
            "--tau", "1,1,1,1",
            "--skip-train",
            "--eval-B", str(eval_B),
        ],
        OUT / "logs" / "fourier_reeval.log",
    )
    return json.loads(eval_path.read_text())


@torch.no_grad()
def champ_recovery(threads: int, eval_B: int) -> dict:
    """τ=1 on the L=16 champ matches the scalar power_p circuit on CORE."""
    path = OUT / "baselines" / "champ_tau1_recovery.json"
    if path.exists():
        return json.loads(path.read_text())
    from Nov_3rd.eval_scale import eval_n

    torch.set_num_threads(threads)
    device = torch.device("cpu")
    state = torch.load(CHAMP, map_location="cpu", weights_only=False)
    gamma = state["gamma"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    p = torch.tensor(float(state["p"]), device=device)
    free = torch.ones(3, device=device)
    rows = {}
    for n in (6, 9, 12, 15):
        gw, phi = snap_angles(gamma, c1, c2, c3, p, free, n)
        scalar = eval_n(n, 16, gamma, phi, device, max_count=eval_B)
        graded = eval_n(n, 16, gw, phi, device, max_count=eval_B)
        rows[str(n)] = {
            "scalar_Med": float(scalar["median_inv_p"]),
            "tau1_Med": float(graded["median_inv_p"]),
            "abs_diff": abs(float(scalar["median_inv_p"]) - float(graded["median_inv_p"])),
            "B": int(scalar["B"]),
        }
        log(
            f"recovery n{n}: scalar {rows[str(n)]['scalar_Med']:.4f} "
            f"τ=1 {rows[str(n)]['tau1_Med']:.4f} |Δ|={rows[str(n)]['abs_diff']:.3e}"
        )
    published = json.loads(CHAMP_EVAL.read_text())
    payload = {
        "per_n": rows,
        "published_n15": float(published["per_n"]["15"]["Med"]),
        "max_abs_diff": max(v["abs_diff"] for v in rows.values()),
        "recovered": max(v["abs_diff"] for v in rows.values()) < 1e-3,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["recovered"]:
        raise SystemExit(f"τ=1 did not recover the champ: {payload}")
    return payload


def train_wt(L: int, seed: int, threads: int, eval_B: int, *, warm_champ: str = "") -> dict:
    out = OUT / f"L{L}" / f"seed{seed}"
    if (out / "eval.json").exists() and (out / "params.pt").exists():
        log(f"WT L{L} seed{seed} already done")
        return json.loads((out / "eval.json").read_text())
    cmd = [
        PY, str(SCR / "train_wt.py"),
        "--L", str(L),
        "--seed", str(seed),
        "--threads", str(threads),
        "--packs", str(PACKS),
        "--out-dir", str(out),
        "--eval-B", str(eval_B),
        "--adam-steps", "600",
        "--lbfgs-rounds", "12",
        "--batch-per-n", "32",
        "--w6", "1",
        "--w9", "2",
    ]
    if L == 2:
        cmd += ["--qaoa-dir", str(S.QAOA_DIR)]
    else:
        cmd += ["--qaoa-dir", str(QAOA16), "--warm-champ", warm_champ or str(CHAMP)]
    run(cmd, OUT / "logs" / f"wt_L{L}_seed{seed}.log")
    return json.loads((out / "eval.json").read_text())


def train_graded(wt_eval: dict, L: int, seed: int, threads: int, eval_B: int, fourier_ckpt: Path) -> dict:
    out = OUT / f"L{L}" / f"seed{seed}" / "graded_fourier"
    if (out / "eval.json").exists():
        return json.loads((out / "eval.json").read_text())
    tau = ",".join(str(x) for x in wt_eval["tau"])
    run(
        [
            PY, str(SCR / "train_qaoa_fourier_graded.py"),
            "--Q", "6",
            "--L", str(L),
            "--seed", str(seed),
            "--threads", str(threads),
            "--packs", str(PACKS),
            "--out-dir", str(out),
            "--init-ckpt", str(fourier_ckpt),
            "--tau", tau,
            "--eval-B", str(eval_B),
            "--adam-steps", "600",
            "--lbfgs-rounds", "12",
        ],
        OUT / "logs" / f"graded_L{L}_seed{seed}.log",
    )
    return json.loads((out / "eval.json").read_text())


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def write_results(summary: dict) -> None:
    v = summary["verdict"]
    walsh = summary["walsh_seed1"]
    lines = [
        "# RESULTS — WT shared Walsh-degree tilt + power_p",
        "",
        f"**Verdict: {v['verdict']}**",
        "",
        "Formula: U_c^X = W exp(-i sum_{w=1..4} gamma_ell * tau_w * f_w) W, "
        "tau_1 = 1, and tau_2, tau_3, tau_4 shared across layers and n. "
        "SNAP is the power_p cubic a(n)=(8/(n-1))^p, unchanged. "
        "This is graded H_c (Walsh degree), labeled as such.",
        "",
        "Train {6, 9} only (fresh250 packs, disjoint from scale1000). CORE {6, 9, 12, 15} B=1000.",
        "",
        "## Gates",
        "",
        v["rule"],
        "",
        "Reasons:",
        *[f"- {r}" for r in v["reasons"]],
        "",
        f"Gap without τ (ungraded Fourier QAOA Med − champ Med) at n15 = **{v['gap_without']:.4f}**.",
        f"Gap with τ = graded Fourier QAOA Med − WT Med, same τ. Larger means SNAP pulls further ahead of the matched qubit QAOA.",
        "",
        "## Reading",
        "",
        "The L=2 gate compares absolute CORE Med to Walsh seed1. "
        "A kill on that gate is recorded even when L=16 grows the SNAP−QAOA gap "
        "and beats graded and ungraded Fourier QAOA. See the tables below.",
        "",
        "## Champ recovery (L=16, τ=1)",
        "",
        f"Recovered = {summary['recovery']['recovered']} "
        f"(max |Med_τ=1 − Med_scalar| = {summary['recovery']['max_abs_diff']:.3e}).",
        "",
        "| n | scalar Med | τ=1 Med | |Δ| |",
        "|---|------------|---------|-----|",
    ]
    for n in (6, 9, 12, 15):
        r = summary["recovery"]["per_n"][str(n)]
        lines.append(
            f"| {n} | {_fmt(r['scalar_Med'])} | {_fmt(r['tau1_Med'])} | {r['abs_diff']:.3e} |"
        )
    lines += [
        "",
        f"Published champ n15 Med = {_fmt(summary['recovery']['published_n15'])}. "
        f"This-run scalar n15 Med = {_fmt(summary['recovery']['per_n']['15']['scalar_Med'])}.",
        "",
        "## L=2 vs Walsh seed1",
        "",
        f"Walsh seed1 (graded γ_w, smoke-trained) n12 Med = {_fmt(walsh['12'])}, "
        f"n15 Med = {_fmt(walsh['15'])} (knee R ≈ 0.9567 vs sealed QAOA).",
        "",
        "| seed | n6 | n9 | n12 | n15 | n15 vs Walsh | τ | τ dev | collapsed | p |",
        "|------|----|----|-----|-----|--------------|---|-------|-----------|---|",
    ]
    for row in summary["l2"]:
        mark = "better" if row["n15"] < walsh["15"] else ("worse" if row["n15"] > walsh["15"] else "tie")
        tau = ", ".join(f"{t:.3f}" for t in row["tau"])
        lines.append(
            f"| {row['seed']} | {_fmt(row['n6'])} | {_fmt(row['n9'])} | {_fmt(row['n12'])} | "
            f"{_fmt(row['n15'])} | {mark} | {tau} | {row['tau_dev']:.4f} | {row['collapsed']} | {row['p']:.3f} |"
        )
    lines += [
        "",
        "## L=16 vs champ, ungraded Fourier QAOA, graded Fourier QAOA",
        "",
        f"Ungraded Fourier QAOA (published Q=6, re-evaluated here) n15 Med = **{_fmt(summary['fourier_n15'])}** "
        f"(published 2.3853).",
        f"power_p champ n15 Med used for the gap = **{_fmt(summary['champ_n15'])}**.",
        "",
        "| seed | WT n15 | champ n15 | ungraded FQ n15 | graded FQ n15 | gap with τ | gap grew | τ dev | collapsed | beats graded | beats ungraded |",
        "|------|--------|-----------|-----------------|---------------|------------|----------|-------|-----------|--------------|----------------|",
    ]
    for row, gate in zip(summary["l16"], v["l16"]):
        lines.append(
            f"| {row['seed']} | {_fmt(row['n15'])} | {_fmt(summary['champ_n15'])} | "
            f"{_fmt(summary['fourier_n15'])} | {_fmt(row['graded_n15'])} | {gate['gap_with']:.4f} | "
            f"{gate['gap_grew']} | {row['tau_dev']:.4f} | {row['collapsed']} | "
            f"{gate['beats_graded']} | {gate['beats_ungraded']} |"
        )
    lines += [
        "",
        "### L=16 full CORE (WT)",
        "",
        "| seed | n | Med | B90 | fresh QAOA Med | R |",
        "|------|---|-----|-----|----------------|---|",
    ]
    for row in summary["l16"]:
        for n in (6, 9, 12, 15):
            r = row["per_n"][str(n)]
            lines.append(
                f"| {row['seed']} | {n} | {_fmt(r['Med'])} | {_fmt(r['B90'])} | {_fmt(r['QAOA_Med'])} | {r['R']:.4f} |"
            )
    lines += [
        "",
        "### Graded Fourier QAOA CORE",
        "",
        "| seed | n | Med | B90 |",
        "|------|---|-----|-----|",
    ]
    for row in summary["l16"]:
        for n in (6, 9, 12, 15):
            r = row["graded_per_n"][str(n)]
            lines.append(f"| {row['seed']} | {n} | {_fmt(r['Med'])} | {_fmt(r['B90'])} |")
    lines += [
        "",
        "## Protocol",
        "",
        "- L=2: QAOA warm (sealed paper L=2 angles) + power_p (c2=c3=1e-4, p=0) + τ=1. "
        "Adam 600 × 1e-3, L-BFGS 12×8, batch 32, w6=1, w9=2, 20% n9 holdout. The warm point is scored first.",
        "- L=16: warm start is the fresh250 power_p champ with τ=1. Same optimizer. "
        "Ratio loss uses the champ's stored fresh250 QAOA angles.",
        "- Graded Fourier QAOA: Q=6 DST/DCT, frozen τ from that WT seed, warm-started from the published ungraded Fourier modes, "
        "Adam 600 × 0.05 + L-BFGS 12×8, same train packs.",
        "- Ungraded Fourier QAOA number is a re-eval of the FM-WS Q=6 L=16 checkpoint, not a retrain.",
        "",
        f"_Finished {summary['finished']}_",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "RESULTS.md").write_text(text)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Package-level copy so the deliverable is obvious.
    (S.PKG / "WT_RESULTS.md").write_text(text)
    log(f"wrote {OUT / 'RESULTS.md'} verdict={v['verdict']}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--l2-seeds", type=str, default="0,1,2")
    ap.add_argument("--l16-seeds", type=str, default="0,1,2")
    ap.add_argument("--skip-l2", action="store_true")
    ap.add_argument("--skip-l16", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    l2_seeds = [int(x) for x in args.l2_seeds.split(",") if x.strip()]
    l16_seeds = [int(x) for x in args.l16_seeds.split(",") if x.strip()]

    log("WT smoke: champ recovery")
    recovery = champ_recovery(args.threads, args.eval_B)
    fourier_ckpt = ensure_fourier_baseline()
    log("WT smoke: re-eval ungraded Fourier QAOA")
    fourier = reeval_fourier_fix(args.threads, args.eval_B)
    fourier_n15 = _med(fourier, 15)

    walsh_blob = json.loads(WALSH_EVAL.read_text())
    walsh = {int(n): _med(walsh_blob, int(n)) for n in walsh_blob["per_n"]}

    l2_rows = []
    if not args.skip_l2:
        for seed in l2_seeds:
            log(f"WT smoke L=2 seed {seed}")
            ev = train_wt(2, seed, args.threads, args.eval_B)
            l2_rows.append(
                {
                    "seed": seed,
                    "n6": _med(ev, 6),
                    "n9": _med(ev, 9),
                    "n12": _med(ev, 12),
                    "n15": _med(ev, 15),
                    "tau": ev["tau"],
                    "tau_dev": ev["tau_dev"],
                    "collapsed": ev["collapsed"],
                    "p": ev["p"],
                    "per_n": ev["per_n"],
                }
            )

    # Gap baseline: this-run scalar re-eval of the champ, paired with the Fourier re-eval.
    champ_n15 = float(recovery["per_n"]["15"]["scalar_Med"])

    l16_rows = []
    if not args.skip_l16:
        for seed in l16_seeds:
            log(f"WT smoke L=16 seed {seed}")
            ev = train_wt(16, seed, args.threads, args.eval_B, warm_champ=str(CHAMP))
            log(f"WT smoke graded Fourier for L=16 seed {seed}")
            gev = train_graded(ev, 16, seed, args.threads, args.eval_B, fourier_ckpt)
            l16_rows.append(
                {
                    "seed": seed,
                    "n15": _med(ev, 15),
                    "graded_n15": _med(gev, 15),
                    "tau": ev["tau"],
                    "tau_dev": ev["tau_dev"],
                    "collapsed": ev["collapsed"],
                    "p": ev["p"],
                    "per_n": ev["per_n"],
                    "graded_per_n": gev["per_n"],
                }
            )

    if not l2_rows or not l16_rows:
        raise SystemExit("smoke did not produce both L=2 and L=16 rows")

    verdict = wt_verdict(
        l2_n15=[r["n15"] for r in l2_rows],
        walsh_n15=walsh[15],
        l16=[
            {"snap_n15": r["n15"], "graded_n15": r["graded_n15"], "tau_dev": r["tau_dev"]}
            for r in l16_rows
        ],
        fourier_n15=fourier_n15,
        champ_n15=champ_n15,
    )
    summary = {
        "verdict": verdict,
        "recovery": recovery,
        "fourier_n15": fourier_n15,
        "fourier_per_n": fourier["per_n"],
        "champ_n15": champ_n15,
        "walsh_seed1": {"12": walsh[12], "15": walsh[15]},
        "l2": l2_rows,
        "l16": l16_rows,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_results(summary)
    log(f"VERDICT {verdict['verdict']}: {verdict['reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
