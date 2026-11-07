#!/usr/bin/env python3
"""FM-WS smoke: L=16 Q∈{4,6}, INTERP polish at L=32 and 48, matched Fourier QAOA.

Resumable. A job with train.json done and eval.json present is skipped.
Q=6 is scheduled before Q=4 so the Opus kill gates land first.
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
LADDER = _ROOT / "Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250"
RUN = _ROOT / "Nov_3rd/runs/beat_qaoa_12h/fm_ws"
CHAMP_PT = LADDER / "power_p/params_L16_seed0.pt"
PACKS = LADDER / "train_packs"
QAOA_DIR = LADDER / "qaoa_angles"
ANGLES_L16 = QAOA_DIR / "L16/angles.json"
CHAMP_EVAL = {
    16: LADDER / "power_p/eval_L16_seed0.json",
    32: LADDER / "power_p/eval_L32_seed0.json",
}
DOC = _ROOT / "Nov_3rd/FM_WS.md"
PY = sys.executable
CORE = (6, 9, 12, 15)


def sh(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "4")
    with log_path.open("a") as f:
        f.write(f"\n=== {' '.join(cmd)} @ {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== EXIT {p.returncode} ===\n")
    print(f"[{time.strftime('%H:%M:%S')}] exit {p.returncode} :: {log_path.name}", flush=True)
    return p.returncode


def job_dir(kind: str, Q: int, L: int) -> Path:
    return RUN / kind / f"Q{Q}" / f"L{L}"


def train_done(d: Path) -> bool:
    path = d / "train.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get("done"))
    except Exception:
        return False


def eval_done(d: Path) -> bool:
    return (d / "eval.json").exists()


def fm_cmd(Q: int, L: int, *, adam: int, lbfgs: int, threads: int, eval_B: int, skip_train: bool) -> list[str]:
    cmd = [
        PY, str(_ROOT / "Nov_3rd/scripts/train_fm_ws.py"),
        "--Q", str(Q), "--L", str(L),
        "--champ", str(CHAMP_PT),
        "--packs", str(PACKS),
        "--qaoa-dir", str(QAOA_DIR),
        "--out-dir", str(job_dir("fmws", Q, L)),
        "--adam-steps", str(adam),
        "--lbfgs-rounds", str(lbfgs),
        "--threads", str(threads),
        "--eval-B", str(eval_B),
        "--seed", "0",
    ]
    if L != 16:
        src = job_dir("fmws", Q, 16) / "params.pt"
        cmd += ["--init-ckpt", str(src)]
    if skip_train:
        cmd.append("--skip-train")
    return cmd


def qaoa_cmd(Q: int, L: int, *, adam: int, lbfgs: int, threads: int, eval_B: int, skip_train: bool) -> list[str]:
    cmd = [
        PY, str(_ROOT / "Nov_3rd/scripts/train_qaoa_fourier.py"),
        "--Q", str(Q), "--L", str(L),
        "--angles", str(ANGLES_L16),
        "--packs", str(PACKS),
        "--out-dir", str(job_dir("qaoa_fourier", Q, L)),
        "--adam-steps", str(adam),
        "--lbfgs-rounds", str(lbfgs),
        "--threads", str(threads),
        "--eval-B", str(eval_B),
        "--seed", "0",
    ]
    if L != 16:
        src = job_dir("qaoa_fourier", Q, 16) / "params.pt"
        cmd += ["--init-ckpt", str(src)]
    if skip_train:
        cmd.append("--skip-train")
    return cmd


def load_meds_fm_or_q(path: Path) -> dict[int, dict]:
    blob = json.loads(path.read_text())
    out = {}
    for k, v in blob["per_n"].items():
        out[int(k)] = {"Med": float(v["Med"]), "B90": float(v["B90"])}
    return out


def load_champ(L: int) -> dict[int, dict]:
    blob = json.loads(CHAMP_EVAL[L].read_text())
    return {int(k): {"Med": float(v["Med"]), "B90": float(v["B90"])} for k, v in blob["per_n"].items()}


def load_free_qaoa(L: int) -> dict[int, dict] | None:
    path = QAOA_DIR / f"L{L}" / "results.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    ladder = blob.get("ladder") or {}
    out = {}
    for k, v in ladder.items():
        if int(k) not in CORE:
            continue
        out[int(k)] = {"Med": float(v["median_inv_p"]), "B90": float(v["B90_inv_p"])}
    return out


def fmt(x) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.4f}"


def mark_lower(a, b) -> str:
    if a is None or b is None or a != a or b != b:
        return "—"
    if a < b:
        return "BEAT"
    if a == b:
        return "TIE"
    return "UNDER"


def aggregate() -> dict:
    from Nov_3rd.fm_ws import fmws_verdict

    champ16 = load_champ(16)
    champ32 = load_champ(32)
    rows = []
    cache = {}
    for kind in ("fmws", "qaoa_fourier"):
        for Q in (4, 6):
            for L in (16, 32, 48):
                ev = job_dir(kind, Q, L) / "eval.json"
                cache[(kind, Q, L)] = load_meds_fm_or_q(ev) if ev.exists() else None
    for Q in (4, 6):
        for L in (16, 32, 48):
            fm = cache[("fmws", Q, L)]
            fq = cache[("qaoa_fourier", Q, L)]
            free = load_free_qaoa(L)
            champ = champ16 if L == 16 else (champ32 if L == 32 else None)
            for n in CORE:
                fm_m = None if fm is None else fm[n]["Med"]
                fm_b = None if fm is None else fm[n]["B90"]
                fq_m = None if fq is None else fq[n]["Med"]
                ch_m = None if champ is None else champ[n]["Med"]
                fr_m = None if free is None or n not in free else free[n]["Med"]
                rows.append({
                    "Q": Q,
                    "L": L,
                    "n": n,
                    "fm_med": fm_m,
                    "fm_b90": fm_b,
                    "champ_med": ch_m,
                    "free_qaoa_med": fr_m,
                    "fourier_qaoa_med": fq_m,
                    "vs_champ": mark_lower(fm_m, ch_m),
                    "vs_free_qaoa": mark_lower(fm_m, fr_m),
                    "vs_fourier_qaoa": mark_lower(fm_m, fq_m),
                    "champ_l32_med": champ32[n]["Med"],
                })

    verdict = None
    need = [("fmws", 6, 16), ("fmws", 6, 32), ("fmws", 6, 48), ("qaoa_fourier", 6, 32), ("qaoa_fourier", 6, 48)]
    if all(cache[k] is not None for k in need):
        q6_l16 = {n: cache[("fmws", 6, 16)][n]["Med"] for n in CORE}
        ch16 = {n: champ16[n]["Med"] for n in CORE}
        q4_32 = cache[("fmws", 4, 32)]
        q4_48 = cache[("fmws", 4, 48)]
        verdict = fmws_verdict(
            q6_l16=q6_l16,
            champ_l16=ch16,
            q6_l32_n15=cache[("fmws", 6, 32)][15]["Med"],
            q6_l48_n15=cache[("fmws", 6, 48)][15]["Med"],
            champ_l32_n15=champ32[15]["Med"],
            fq_l32_n15=cache[("qaoa_fourier", 6, 32)][15]["Med"],
            fq_l48_n15=cache[("qaoa_fourier", 6, 48)][15]["Med"],
            q4_l32_n15=None if q4_32 is None else q4_32[15]["Med"],
            q4_l48_n15=None if q4_48 is None else q4_48[15]["Med"],
        )
    payload = {
        "rows": rows,
        "verdict": verdict,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "champ_l16": str(CHAMP_EVAL[16]),
        "champ_l32": str(CHAMP_EVAL[32]),
        "note": (
            "Lower Med is better. champ_med is the free power_p champ at the same L "
            "(absent at L=48). The L=48 kill reference is champ L=32, column champ_l32_med. "
            "free_qaoa is the unrestricted fresh250 QAOA at that L (absent at L=48). "
            "fourier_qaoa is the matched Fourier/INTERP QAOA at the same Q and L."
        ),
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_md(payload)
    return payload


def write_md(payload: dict) -> None:
    lines = [
        "# FM-WS smoke — CORE Med",
        "",
        "Lower Med is better (shots to solution, 1/p_succ). CORE B=1000, train {6,9} only.",
        "",
        "Champ = free power_p at that L (fresh250 seed0). There is no free L=48 champ;",
        "L=48 is scored against the free **L=32** champ (Opus kill bar: >1% Med at n=15).",
        "Fourier QAOA uses the same Q, the same DST/DCT modes, and the same INTERP polish.",
        "",
        "| Q | L | n | FM-WS Med | champ Med | vs champ | free QAOA | vs free Q | Fourier QAOA | vs F-QAOA | L32 champ |",
        "|---|---|---|-----------|-----------|----------|-----------|-----------|--------------|-----------|-----------|",
    ]
    for r in payload["rows"]:
        lines.append(
            "| {Q} | {L} | {n} | {fm} | {ch} | {vc} | {fr} | {vf} | {fq} | {vq} | {c32} |".format(
                Q=r["Q"], L=r["L"], n=r["n"],
                fm=fmt(r["fm_med"]), ch=fmt(r["champ_med"]), vc=r["vs_champ"],
                fr=fmt(r["free_qaoa_med"]), vf=r["vs_free_qaoa"],
                fq=fmt(r["fourier_qaoa_med"]), vq=r["vs_fourier_qaoa"],
                c32=fmt(r["champ_l32_med"]),
            )
        )
    v = payload.get("verdict")
    lines += ["", "## Verdict", ""]
    if not v:
        lines.append("Incomplete — Q=6 L=16/32/48 and Fourier QAOA L=32/48 are not all in yet.")
    else:
        lines.append(f"**{v['verdict']}**")
        lines.append("")
        lines.append(f"Rule: {v['rule']}")
        lines.append("")
        lines.append(
            f"- Gate A (Q=6 L16 n15 within 0.5% of champ): {v['gate_A_q6_l16_n15_within_0.5pct']}"
        )
        lines.append(
            f"- Gate A all CORE: {v['gate_A_all_core_within_0.5pct']} rel={v['rel_vs_champ_l16']}"
        )
        lines.append(
            f"- Gate B L32 n15 drop vs L32 champ: {v['gate_B_l32_n15_drop']:.4f} pass={v['gate_B_l32']}"
        )
        lines.append(
            f"- Gate B L48 n15 drop vs L32 champ: {v['gate_B_l48_n15_drop']:.4f} pass={v['gate_B_l48']}"
        )
        lines.append(
            f"- Gate C beats Fourier QAOA n15: L32={v['gate_C_beats_fourier_qaoa_l32_n15']} "
            f"L48={v['gate_C_beats_fourier_qaoa_l48_n15']}"
        )
    lines.append("")
    text = "\n".join(lines) + "\n"
    (RUN / "RESULTS.md").write_text(text)
    # splice into the design doc
    if DOC.exists():
        doc = DOC.read_text()
        begin, end = "<!-- RESULTS_BEGIN -->", "<!-- RESULTS_END -->"
        if begin in doc and end in doc:
            pre, rest = doc.split(begin, 1)
            _, post = rest.split(end, 1)
            DOC.write_text(pre + begin + "\n\n" + text + "\n" + end + post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--only", type=str, default="", help="comma list of job names, e.g. fm_Q6_L16")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    if args.aggregate_only:
        payload = aggregate()
        print(json.dumps(payload.get("verdict"), indent=2), flush=True)
        return 0

    # Q=6 before Q=4. At each Q: both families at L=16, then INTERP depths.
    plan = []
    for Q in (6, 4):
        for L in (16, 32, 48):
            plan.append(("fmws", Q, L))
            plan.append(("qaoa_fourier", Q, L))
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    for kind, Q, L in plan:
        name = f"{'fm' if kind == 'fmws' else 'qaoa'}_Q{Q}_L{L}"
        if only and name not in only:
            continue
        d = job_dir(kind, Q, L)
        d.mkdir(parents=True, exist_ok=True)
        if train_done(d) and eval_done(d) and not args.force:
            print(f"skip {name}", flush=True)
            continue
        if L != 16:
            src_kind = kind
            src = job_dir(src_kind, Q, 16) / "params.pt"
            if not src.exists():
                raise SystemExit(f"{name} needs INTERP source {src}")
        skip_train = train_done(d) and not args.force
        if kind == "fmws":
            cmd = fm_cmd(Q, L, adam=args.adam_steps, lbfgs=args.lbfgs_rounds,
                         threads=args.threads, eval_B=args.eval_B, skip_train=skip_train)
        else:
            cmd = qaoa_cmd(Q, L, adam=args.adam_steps, lbfgs=args.lbfgs_rounds,
                           threads=args.threads, eval_B=args.eval_B, skip_train=skip_train)
        print(f"start {name} skip_train={skip_train}", flush=True)
        ec = sh(cmd, RUN / "logs" / f"{name}.log")
        if ec != 0:
            aggregate()
            raise SystemExit(f"{name} failed ec={ec}")
    payload = aggregate()
    v = (payload.get("verdict") or {}).get("verdict")
    print(f"SMOKE VERDICT {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
