#!/usr/bin/env python3
"""L=2 train{6,9} soft-quantile Med(1/p) alignment sweep.

Gate is Med(inv)/QAOA_Med, not median of per-instance ratios. This wrapper
calls train_beat_train69.py --loss med_align across (idea, rho, seed, w) grid.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("PYTHON", sys.executable)
TRAIN = str(Path(__file__).resolve().parent / "train_beat_train69.py")
OUT = ROOT / "Nov_3rd/runs/beat_qaoa_12h/train69"


def run_one(cmd: list[str]) -> int:
    print("[med_align]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=500)
    ap.add_argument("--soft_q", type=float, default=0.5)
    ap.add_argument("--only", type=str, default="", help="comma tags: trust,n9med,ratio,pareto_warm")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip()) or {"trust", "n9med", "ratio"}

    jobs = []
    # trust69 med_align: freeze γ,c1; free c2,c3 in ρ ball
    if "trust" in only:
        for rho in (0.05, 0.08, 0.12):
            for seed in range(4):
                tag = f"med_align_rho{rho:g}"
                jobs.append([
                    PY, TRAIN, "--idea", "trust69", "--L", str(args.L), "--seed", str(seed),
                    "--threads", str(args.threads), "--rho", str(rho),
                    "--adam_steps", str(args.adam_steps), "--lbfgs_rounds", "6",
                    "--loss", "med_align", "--soft_q", str(args.soft_q),
                    "--w6", "1", "--w9", "2", "--holdout_frac", "0.25",
                    "--tag", tag,
                ])
    # n9-only med_align (push the stuck R9≈1.00)
    if "n9med" in only:
        for rho in (0.03, 0.05, 0.08):
            for seed in range(4):
                tag = f"med_align_n9_rho{rho:g}"
                jobs.append([
                    PY, TRAIN, "--idea", "trust69", "--L", str(args.L), "--seed", str(seed),
                    "--threads", str(args.threads), "--rho", str(rho),
                    "--adam_steps", str(args.adam_steps), "--lbfgs_rounds", "6",
                    "--loss", "med_align", "--soft_q", str(args.soft_q),
                    "--w6", "0", "--w9", "1", "--holdout_frac", "0.3",
                    "--tag", tag,
                ])
    # ratio69_w9 free γ,c's with med_align
    if "ratio" in only:
        for seed in range(3):
            tag = "med_align"
            jobs.append([
                PY, TRAIN, "--idea", "ratio69_w9", "--L", str(args.L), "--seed", str(seed),
                "--threads", str(args.threads),
                "--adam_steps", str(args.adam_steps), "--lbfgs_rounds", "8",
                "--loss", "med_align", "--soft_med", "--soft_q", str(args.soft_q),
                "--w6", "1", "--w9", "2", "--holdout_frac", "0.25",
                "--tag", tag,
            ])

    print(f"[med_align] {len(jobs)} jobs soft_q={args.soft_q}", flush=True)
    t0 = time.time()
    fails = 0
    beats = []
    for i, cmd in enumerate(jobs):
        ec = run_one(cmd)
        if ec != 0:
            fails += 1
            print(f"[med_align] FAIL ec={ec} job {i}", flush=True)
            continue
        # scrape latest eval under tag
        # tag is last --tag value
        tag = cmd[cmd.index("--tag") + 1]
        idea = cmd[cmd.index("--idea") + 1]
        seed = cmd[cmd.index("--seed") + 1]
        ev = OUT / idea / tag / f"eval_L{args.L}_seed{seed}.json"
        if ev.exists():
            d = json.loads(ev.read_text())
            pn = d["per_n"]
            row = {k: pn[k]["R"] for k in ("6", "9", "12", "15")}
            ab = all(pn[k]["verdict"] == "BEAT" for k in ("6", "9", "12", "15"))
            print(f"[med_align] RESULT {idea}/{tag}/s{seed} R={row} ALL={ab}", flush=True)
            if ab:
                beats.append(str(ev))
    print(f"[med_align] DONE jobs={len(jobs)} fails={fails} beats={len(beats)} wall={time.time()-t0:.0f}s", flush=True)
    if beats:
        print("[med_align] ALL_CORE_BEAT paths:", *beats, sep="\n  ", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
