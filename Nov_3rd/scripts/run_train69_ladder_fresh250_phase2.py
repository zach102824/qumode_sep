#!/usr/bin/env python3
"""Phase 2: extend train69_ladder_fresh250 eval to n=18 then n=21 (no retrain)."""
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
SNAP = RUN / "power_p"
STATUS = RUN / "STATUS.md"
RESULTS = RUN / "RESULTS.md"
ROWS = RUN / "results_rows.json"
L_LIST = [1, 4, 8, 12, 16, 20, 24, 28, 32]
CORE_NS = [6, 9, 12, 15]
PY = os.environ.get("PYTHON", sys.executable)
SCR = _ROOT / "Nov_3rd" / "scripts"
SCALE = _ROOT / "Nov_3rd" / "data" / "scale1000"


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


def load_rows() -> dict:
    if ROWS.exists():
        return json.loads(ROWS.read_text())
    return {"rows": {}}


def save_results_md(data: dict, ns: list[int]):
    title = "# RESULTS — power_p vs QAOA train69 fresh250"
    if ns == CORE_NS:
        title += " (Phase 1 CORE)"
    elif 21 in ns:
        title += " (Phase 1+2 CORE+n18+n21)"
    elif 18 in ns:
        title += " (Phase 1+2 CORE+n18)"
    else:
        title += f" (ns={ns})"
    header_parts = ["L"]
    for n in ns:
        header_parts += [
            f"n{n} SNAP Med", f"n{n} Q Med", f"n{n} Med",
            f"n{n} SNAP B90", f"n{n} Q B90", f"n{n} B90",
        ]
    lines = [
        title,
        "",
        "_Beat mark: raw Med < QAOA Med → BEAT; equal → TIE; greater → UNDER. Same for B90. No R column._",
        "",
        "| " + " | ".join(header_parts) + " |",
        "|---|" + "|".join(["---"] * (len(header_parts) - 1)) + "|",
    ]
    for L in L_LIST:
        row = data["rows"].get(str(L))
        if not row:
            continue
        cells = [str(L)]
        complete = True
        for n in ns:
            r = row.get(str(n))
            if not r:
                complete = False
                cells += ["—"] * 6
                continue
            cells += [
                f"{r['SNAP_Med']:.4f}",
                f"{r['QAOA_Med']:.4f}",
                r["Med_mark"],
                f"{r['SNAP_B90']:.4f}",
                f"{r['QAOA_B90']:.4f}",
                r["B90_mark"],
            ]
        if complete or any(row.get(str(n)) for n in ns):
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    RESULTS.write_text("\n".join(lines) + "\n")


def ensure_pack(n: int) -> bool:
    pack = SCALE / f"data_n{n}.npz"
    if pack.exists():
        update_status(f"scale1000 pack n{n} present: {pack}")
        return True
    update_status(f"scale1000 pack n{n} MISSING — attempting regen…")
    logp = RUN / "logs" / f"regen_scale1000_n{n}.log"
    ec = sh(
        [PY, str(_ROOT / "Nov_3rd" / "runs" / "beat_qaoa_12h" / "train69" / "scale1000_regen" / "regen_scale1000.py"), str(n)],
        logp,
    )
    if ec != 0 or not pack.exists():
        update_status(f"BLOCKER: failed to build scale1000 data_n{n}.npz (ec={ec}). See {logp}")
        return False
    # update DATA_MANIFEST lightly
    man = _ROOT / "Nov_3rd" / "DATA_MANIFEST.md"
    try:
        import hashlib
        h = hashlib.sha256(pack.read_bytes()).hexdigest()[:12]
        with man.open("a") as f:
            f.write(f"\ndata/scale1000/data_n{n}.npz  {pack.stat().st_size}  sha256={h}  # phase2 regen\n")
    except Exception as e:
        update_status(f"manifest append warn: {e}")
    update_status(f"scale1000 pack n{n} built OK ({pack.stat().st_size} bytes)")
    return True


def eval_qaoa_ns(L: int, ns: list[int]):
    ec = sh(
        [
            PY, str(SCR / "train_qaoa_train69_fresh.py"),
            "--L", str(L),
            "--packs", str(PACKS),
            "--out_root", str(QAOA),
            "--eval_ns", ",".join(str(n) for n in ns),
            "--threads", "2",
        ],
        RUN / "logs" / f"phase2_qaoa_L{L}.log",
    )
    if ec != 0:
        raise SystemExit(f"QAOA eval L{L} ns={ns} failed ec={ec}")


def eval_snap_ns(L: int, ns: list[int]):
    """Eval power_p params on ns and MERGE into existing eval JSON (keep CORE)."""
    pt = SNAP / f"params_L{L}_seed0.pt"
    if not pt.exists():
        raise SystemExit(f"missing SNAP params {pt}")
    # run eval_only into a temp tag dir so we don't clobber CORE eval_L*_seed0.json
    tag = f"phase2_n{'_'.join(map(str, ns))}"
    tmp_dir = SNAP / tag
    ec = sh(
        [
            PY, str(SCR / "train_beat_train69.py"),
            "--idea", "power_p",
            "--L", str(L),
            "--seed", "0",
            "--threads", "2",
            "--eval_only", str(pt),
            "--out_root", str(RUN),
            "--qaoa_dir", str(QAOA),
            "--eval_ns", ",".join(str(n) for n in ns),
            "--beat_mode", "raw",
            "--tag", tag,
        ],
        RUN / "logs" / f"phase2_snap_L{L}_n{'_'.join(map(str, ns))}.log",
    )
    # train_beat writes under out_root/idea/tag → RUN/power_p/<tag>/
    # Wait: out_dir = out_root / idea; if tag: out_dir = out_dir / tag
    # So with --out_root RUN and --idea power_p --tag X → RUN/power_p/X/
    # But we also pass --tag after setting idea... Looking at code:
    #   out_dir = out_root / idea
    #   if args.tag: out_dir = out_dir / args.tag
    # And eval_only write_eval uses out_dir. Good.
    # BUT --tag is passed as phase2_... and idea is power_p, out_root is RUN
    # → RUN/power_p/phase2_n18/eval_L{L}_seed0.json
    if ec != 0:
        raise SystemExit(f"SNAP eval L{L} ns={ns} failed ec={ec}")
    # find written file
    written = RUN / "power_p" / tag / f"eval_L{L}_seed0.json"
    # when tag is used with idea power_p under out_root=RUN:
    # Actually wait — train_beat sets out_dir = out_root / idea / tag = RUN/power_p/tag
    # But --idea power_p with out_root=RUN → RUN/power_p/tag. Yes.
    if not written.exists():
        # fallback: maybe tag alone under power_p incorrectly
        alts = list((RUN / "power_p").rglob(f"eval_L{L}_seed0.json"))
        raise SystemExit(f"expected {written}; found alts={[str(a) for a in alts]}")
    fresh = json.loads(written.read_text())
    core_path = SNAP / f"eval_L{L}_seed0.json"
    core = json.loads(core_path.read_text()) if core_path.exists() else {"per_n": {}, "idea": "power_p", "L": L, "seed": 0}
    for k, v in fresh.get("per_n", {}).items():
        core.setdefault("per_n", {})[k] = v
    core["beat_mode"] = "raw"
    core["phase2_updated"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    core_path.write_text(json.dumps(core, indent=2) + "\n")
    # also merge md
    return core


def ingest_ns(L: int, ns: list[int]):
    ev = json.loads((SNAP / f"eval_L{L}_seed0.json").read_text())
    qr = json.loads((QAOA / f"L{L}" / "results.json").read_text())
    ladder = {int(k): v for k, v in (qr.get("ladder") or {}).items()}
    data = load_rows()
    row = data["rows"].setdefault(str(L), {})
    for n in ns:
        pn = ev["per_n"][str(n)]
        q = ladder[n]
        sm, sb = float(pn["Med"]), float(pn["B90"])
        qm, qb = float(q["median_inv_p"]), float(q["B90_inv_p"])
        # prefer verdict from SNAP eval if present
        med_mark = pn.get("verdict") or mark_raw(sm, qm)
        b90_mark = pn.get("verdict_B90") or mark_raw(sb, qb)
        row[str(n)] = {
            "SNAP_Med": sm,
            "QAOA_Med": qm,
            "Med_mark": med_mark,
            "SNAP_B90": sb,
            "QAOA_B90": qb,
            "B90_mark": b90_mark,
        }
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    ROWS.write_text(json.dumps(data, indent=2) + "\n")
    # per-L extended md
    all_ns = [n for n in CORE_NS + [18, 21] if str(n) in row]
    md = [
        f"# RESULTS L={L}",
        "",
        "| n | SNAP Med | QAOA Med | Med mark | SNAP B90 | QAOA B90 | B90 mark |",
        "|---|----------|----------|----------|----------|----------|----------|",
    ]
    for n in all_ns:
        r = row[str(n)]
        md.append(
            f"| {n} | {r['SNAP_Med']:.4f} | {r['QAOA_Med']:.4f} | {r['Med_mark']} | "
            f"{r['SNAP_B90']:.4f} | {r['QAOA_B90']:.4f} | {r['B90_mark']} |"
        )
    md.append("")
    (RUN / f"RESULTS_L{L}.md").write_text("\n".join(md) + "\n")
    (RUN / f"RESULTS_L{L}.json").write_text(json.dumps({"L": L, "per_n": row}, indent=2) + "\n")
    return data


def run_pass(ns: list[int], label: str):
    for n in ns:
        if not ensure_pack(n):
            update_status(f"Phase2 {label}: STOP — cannot eval without pack n{n}")
            return False
    update_status(f"Phase2 {label}: start eval ns={ns} for all L")
    for L in L_LIST:
        update_status(f"Phase2 {label}: L={L} QAOA eval…")
        eval_qaoa_ns(L, ns)
        update_status(f"Phase2 {label}: L={L} SNAP power_p eval…")
        eval_snap_ns(L, ns)
        data = ingest_ns(L, ns)
        present = sorted({int(k) for row in data["rows"].values() for k in row})
        save_results_md(data, [n for n in CORE_NS + [18, 21] if n in present])
        update_status(f"Phase2 {label}: L={L} DONE")
    update_status(f"Phase2 {label}: COMPLETE for all L")
    (RUN / f"PHASE2_{label}_COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %z") + "\n")
    return True


def rewrite_status_header(phase2_note: str):
    text = STATUS.read_text() if STATUS.exists() else ""
    # append-only is fine; also drop "blocked" note via new lines
    update_status(phase2_note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=["n18", "n21", "all"], default="all")
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    rewrite_status_header("Phase2 IN PROGRESS — n18 then n21; no retrain.")
    ok18 = True
    if args.which in ("n18", "all"):
        ok18 = run_pass([18], "n18")
    ok21 = True
    if args.which in ("n21", "all") and ok18:
        ok21 = run_pass([21], "n21")
    if ok18 and (args.which == "n18" or ok21):
        if args.which == "all" and ok21:
            update_status("PHASE2 COMPLETE — n18+n21 Med/B90 for all L.")
            (RUN / "PHASE2_COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %z") + "\n")
        elif args.which == "n18":
            update_status("PHASE2 n18 COMPLETE — n21 not requested this run.")
    return 0 if ok18 and (args.which == "n18" or ok21) else 1


if __name__ == "__main__":
    raise SystemExit(main())
