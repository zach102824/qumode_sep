#!/usr/bin/env python3
"""L=2 train{6,9} soft-quantile Med(1/p) alignment sweep.

Gate is Med(inv)/QAOA_Med, not median of per-instance ratios. This wrapper
calls train_beat_train69.py --loss med_align across (idea, rho, seed, w) grid.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("PYTHON", sys.executable)
TRAIN = str(Path(__file__).resolve().parent / "train_beat_train69.py")
OUT = ROOT / "Nov_3rd/runs/beat_qaoa_12h/train69"


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def scrape(cmd: list[str], L: int) -> dict | None:
    tag = cmd[cmd.index("--tag") + 1]
    idea = cmd[cmd.index("--idea") + 1]
    seed = cmd[cmd.index("--seed") + 1]
    ev = OUT / idea / tag / f"eval_L{L}_seed{seed}.json"
    if not ev.exists():
        return None
    d = json.loads(ev.read_text())
    pn = d["per_n"]
    row = {k: pn[k]["R"] for k in ("6", "9", "12", "15")}
    ab = all(pn[k]["verdict"] == "BEAT" for k in ("6", "9", "12", "15"))
    return {"path": str(ev), "idea": idea, "tag": tag, "seed": int(seed), "R": row, "ALL_CORE_BEAT": ab,
            "verdict": {k: pn[k]["verdict"] for k in ("6", "9", "12", "15")}}


def run_one(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[med_align]", " ".join(cmd), flush=True)
    with log_path.open("w") as fh:
        return subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1, help="concurrent train processes")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=500)
    ap.add_argument("--soft_q", type=float, default=0.5)
    ap.add_argument("--only", type=str, default="", help="comma tags: trust,n9med,ratio")
    ap.add_argument("--trust_rhos", type=str, default="0.05,0.08,0.12")
    ap.add_argument("--n9_rhos", type=str, default="0.03,0.05,0.08")
    ap.add_argument("--trust_seeds", type=str, default="0-5")
    ap.add_argument("--n9_seeds", type=str, default="0-5")
    ap.add_argument("--ratio_seeds", type=str, default="0-3")
    ap.add_argument("--tag_suffix", type=str, default="", help="appended to tag (e.g. _q40)")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip()) or {"trust", "n9med", "ratio"}
    suf = args.tag_suffix

    jobs = []
    # trust69 med_align: freeze γ,c1; free c2,c3 in ρ ball
    if "trust" in only:
        for rho in parse_floats(args.trust_rhos):
            for seed in parse_seeds(args.trust_seeds):
                tag = f"med_align_rho{rho:g}{suf}"
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
        for rho in parse_floats(args.n9_rhos):
            for seed in parse_seeds(args.n9_seeds):
                tag = f"med_align_n9_rho{rho:g}{suf}"
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
        for seed in parse_seeds(args.ratio_seeds):
            tag = f"med_align{suf}"
            jobs.append([
                PY, TRAIN, "--idea", "ratio69_w9", "--L", str(args.L), "--seed", str(seed),
                "--threads", str(args.threads),
                "--adam_steps", str(args.adam_steps), "--lbfgs_rounds", "8",
                "--loss", "med_align", "--soft_med", "--soft_q", str(args.soft_q),
                "--w6", "1", "--w9", "2", "--holdout_frac", "0.25",
                "--tag", tag,
            ])

    print(f"[med_align] {len(jobs)} jobs soft_q={args.soft_q} workers={args.workers}", flush=True)
    t0 = time.time()
    fails = 0
    beats = []
    rows = []
    log_dir = OUT / "logs_med_align"

    def _finish(i: int, cmd: list[str], ec: int):
        nonlocal fails
        tag = cmd[cmd.index("--tag") + 1]
        seed = cmd[cmd.index("--seed") + 1]
        if ec != 0:
            fails += 1
            print(f"[med_align] FAIL ec={ec} job {i} {tag} s{seed}", flush=True)
            rows.append({"job": i, "tag": tag, "seed": int(seed), "ec": ec})
            return
        rec = scrape(cmd, args.L)
        if rec is None:
            fails += 1
            print(f"[med_align] FAIL missing eval job {i} {tag} s{seed}", flush=True)
            rows.append({"job": i, "tag": tag, "seed": int(seed), "ec": 0, "missing_eval": True})
            return
        rec["job"] = i
        rec["ec"] = 0
        rows.append(rec)
        print(f"[med_align] RESULT {rec['idea']}/{tag}/s{seed} R={rec['R']} ALL={rec['ALL_CORE_BEAT']}", flush=True)
        if rec["ALL_CORE_BEAT"]:
            beats.append(rec["path"])
            print(f"[med_align] ALL_CORE_BEAT {rec['path']}", flush=True)

    if args.workers <= 1:
        for i, cmd in enumerate(jobs):
            tag = cmd[cmd.index("--tag") + 1]
            seed = cmd[cmd.index("--seed") + 1]
            ec = run_one(cmd, log_dir / f"{tag}_s{seed}.log")
            _finish(i, cmd, ec)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for i, cmd in enumerate(jobs):
                tag = cmd[cmd.index("--tag") + 1]
                seed = cmd[cmd.index("--seed") + 1]
                futs[pool.submit(run_one, cmd, log_dir / f"{tag}_s{seed}.log")] = (i, cmd)
            for fut in as_completed(futs):
                i, cmd = futs[fut]
                ec = fut.result()
                _finish(i, cmd, ec)

    summary = {
        "jobs": len(jobs),
        "fails": fails,
        "beats": beats,
        "soft_q": args.soft_q,
        "wall_s": time.time() - t0,
        "rows": rows,
    }
    sum_path = OUT / f"med_align_sweep_summary_q{args.soft_q:g}{suf}.json"
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[med_align] DONE jobs={len(jobs)} fails={fails} beats={len(beats)} wall={time.time()-t0:.0f}s", flush=True)
    print(f"[med_align] summary {sum_path}", flush=True)
    if beats:
        print("[med_align] ALL_CORE_BEAT paths:", *beats, sep="\n  ", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
