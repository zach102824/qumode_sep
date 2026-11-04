#!/usr/bin/env python3
"""Phase-1 orchestrator: fresh 250/250 packs → per-L QAOA → power_p → RESULTS."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
RUN = _ROOT / "Nov_3rd" / "runs" / "beat_qaoa_12h" / "train69_ladder_fresh250"
PACKS = RUN / "train_packs"
QAOA = RUN / "qaoa_angles"
SNAP_ROOT = RUN  # power_p written under RUN/power_p
STATUS = RUN / "STATUS.md"
RESULTS = RUN / "RESULTS.md"
L_LIST = [1, 4, 8, 12, 16, 20, 24, 28, 32]
EVAL_NS = [6, 9, 12, 15]
PY = os.environ.get("PYTHON", sys.executable)
SCR = _ROOT / "Nov_3rd" / "scripts"


def sh(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("a") as f:
        f.write(f"\n=== CMD {' '.join(cmd)} @ {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== EXIT {p.returncode} ===\n")
    return p.returncode


def update_status(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with STATUS.open("a") as f:
        f.write(f"\n- [{ts} Asia/Singapore] {msg}\n")


def mark_raw(a, b):
    if not (a == a and b == b):
        return "NO_REF"
    if a < b:
        return "BEAT"
    if a == b:
        return "TIE"
    return "UNDER"


def load_results_rows() -> dict:
    path = RUN / "results_rows.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"rows": {}}


def save_results_rows(data: dict):
    (RUN / "results_rows.json").write_text(json.dumps(data, indent=2) + "\n")
    # markdown table
    lines = [
        "# RESULTS — power_p vs QAOA train69 fresh250 (Phase 1 CORE)",
        "",
        "_Beat mark: raw Med < QAOA Med → BEAT; equal → TIE; greater → UNDER. Same for B90. No R column._",
        "",
        "| L | n6 SNAP Med | n6 Q Med | n6 Med | n6 SNAP B90 | n6 Q B90 | n6 B90 | n9 SNAP Med | n9 Q Med | n9 Med | n9 SNAP B90 | n9 Q B90 | n9 B90 | n12 SNAP Med | n12 Q Med | n12 Med | n12 SNAP B90 | n12 Q B90 | n12 B90 | n15 SNAP Med | n15 Q Med | n15 Med | n15 SNAP B90 | n15 Q B90 | n15 B90 |",
        "|---|" + "|".join(["---"] * 24) + "|",
    ]
    for L in L_LIST:
        row = data["rows"].get(str(L))
        if not row:
            continue
        cells = [str(L)]
        for n in EVAL_NS:
            r = row[str(n)]
            cells += [
                f"{r['SNAP_Med']:.4f}",
                f"{r['QAOA_Med']:.4f}",
                r["Med_mark"],
                f"{r['SNAP_B90']:.4f}",
                f"{r['QAOA_B90']:.4f}",
                r["B90_mark"],
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    RESULTS.write_text("\n".join(lines) + "\n")


def ingest_L(L: int):
    eval_path = SNAP_ROOT / "power_p" / f"eval_L{L}_seed0.json"
    qres = QAOA / f"L{L}" / "results.json"
    ev = json.loads(eval_path.read_text())
    qr = json.loads(qres.read_text())
    ladder = {int(k): v for k, v in (qr.get("ladder") or {}).items()}
    row = {}
    for n in EVAL_NS:
        pn = ev["per_n"][str(n)]
        q = ladder[n]
        sm, sb = float(pn["Med"]), float(pn["B90"])
        qm, qb = float(q["median_inv_p"]), float(q["B90_inv_p"])
        row[str(n)] = {
            "SNAP_Med": sm,
            "QAOA_Med": qm,
            "Med_mark": mark_raw(sm, qm),
            "SNAP_B90": sb,
            "QAOA_B90": qb,
            "B90_mark": mark_raw(sb, qb),
        }
    data = load_results_rows()
    data["rows"][str(L)] = row
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_results_rows(data)
    # per-L files
    md = [
        f"# RESULTS L={L}",
        "",
        "| n | SNAP Med | QAOA Med | Med mark | SNAP B90 | QAOA B90 | B90 mark |",
        "|---|----------|----------|----------|----------|----------|----------|",
    ]
    for n in EVAL_NS:
        r = row[str(n)]
        md.append(
            f"| {n} | {r['SNAP_Med']:.4f} | {r['QAOA_Med']:.4f} | {r['Med_mark']} | "
            f"{r['SNAP_B90']:.4f} | {r['QAOA_B90']:.4f} | {r['B90_mark']} |"
        )
    md.append("")
    (RUN / f"RESULTS_L{L}.md").write_text("\n".join(md) + "\n")
    (RUN / f"RESULTS_L{L}.json").write_text(json.dumps({"L": L, "per_n": row}, indent=2) + "\n")


def build_packs():
    cmd = [
        PY, str(SCR / "build_beat_train_packs.py"),
        "--out", str(PACKS),
        "--ns", "6,9",
        "--target", "250",
        "--base", "62000000000",
        "--subsample_seed", "20261104",
    ]
    ec = sh(cmd, RUN / "logs" / "build_packs.log")
    if ec != 0:
        raise SystemExit(f"build packs failed ec={ec}")
    # disjoint + differ from old packs check
    old = _ROOT / "Nov_3rd" / "runs" / "beat_qaoa_12h" / "train_packs"
    import numpy as np
    for n in (6, 9):
        a = set(map(int, np.load(PACKS / f"train_n{n}.npz")["train_seeds"]))
        # scale1000
        sc = set(map(int, np.load(_ROOT / "Nov_3rd" / "data" / "scale1000" / f"data_n{n}.npz")["scale1000_seeds"]))
        assert not (a & sc), f"n{n} overlap scale1000"
        if (old / f"train_n{n}.npz").exists():
            b = set(map(int, np.load(old / f"train_n{n}.npz")["train_seeds"]))
            print(f"n{n}: new vs old overlap={len(a & b)}/{len(a)} (want differ)", flush=True)
    update_status(f"Fresh packs built at {PACKS} (n6+n9=250 each, disjoint scale1000)")


def run_L(L: int, force: bool = False):
    # skip if already done
    if (RUN / f"RESULTS_L{L}.json").exists() and not force:
        update_status(f"L={L} already has RESULTS — skip")
        ingest_L(L)
        return
    update_status(f"L={L}: train QAOA fresh…")
    ec = sh(
        [
            PY, str(SCR / "train_qaoa_train69_fresh.py"),
            "--L", str(L),
            "--packs", str(PACKS),
            "--out_root", str(QAOA),
            "--eval_ns", "6,9,12,15",
            "--batch_per_n", "32",
            "--threads", "2",
        ] + (["--force"] if force else []),
        RUN / "logs" / f"qaoa_L{L}.log",
    )
    if ec != 0:
        raise SystemExit(f"QAOA L{L} failed ec={ec}")
    update_status(f"L={L}: train power_p seed0…")
    ec = sh(
        [
            PY, str(SCR / "train_beat_train69.py"),
            "--idea", "power_p",
            "--L", str(L),
            "--seed", "0",
            "--threads", "2",
            "--batch_per_n", "32",
            "--w6", "1",
            "--w9", "2",
            "--packs_dir", str(PACKS),
            "--out_root", str(SNAP_ROOT),
            "--qaoa_dir", str(QAOA),
            "--eval_ns", "6,9,12,15",
            "--beat_mode", "raw",
        ],
        RUN / "logs" / f"power_p_L{L}.log",
    )
    if ec != 0:
        raise SystemExit(f"power_p L{L} failed ec={ec}")
    ingest_L(L)
    update_status(f"L={L}: DONE — RESULTS_L{L}.md written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_packs", action="store_true")
    ap.add_argument("--only_L", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from_L", type=int, default=None)
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    if not args.skip_packs:
        if (PACKS / "manifest.json").exists() and not args.force:
            update_status("packs already present — skip build")
        else:
            build_packs()
    Ls = L_LIST
    if args.only_L is not None:
        Ls = [args.only_L]
    elif args.from_L is not None:
        Ls = [L for L in L_LIST if L >= args.from_L]
    for L in Ls:
        run_L(L, force=args.force)
    # phase1 complete check
    data = load_results_rows()
    if all(str(L) in data["rows"] for L in L_LIST):
        update_status("PHASE1 COMPLETE — all L in L_LIST have CORE Med/B90. Do NOT start Phase2.")
        (RUN / "PHASE1_COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %z") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
