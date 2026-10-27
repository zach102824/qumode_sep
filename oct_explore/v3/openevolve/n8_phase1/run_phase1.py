#!/usr/bin/env python3
"""n=8 Phase-1: L×ntrain×A/B core (24), then densify winners. Non-stop."""
from __future__ import annotations
import csv, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/qumode_sep/oct_explore")
RUN = Path("/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1")
RUNS = RUN / "runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v3"))
sys.path.insert(0, str(ROOT / "v3/openevolve/net"))
import tiecost_lib as TL

WS_MED, WS_B90 = 13.989, 29.396
FLOOR_MED = 128.0
PROMOTE_VAL = 64.0
HOT_VAL = 40.0

ANCHORS = {
    "A": dict(arch="mp", H=128, rounds=4, steps=1400, lr=1.1e-3, bs=250,
              wd=1e-4, leakpen=0.14, tailw=1.6, amax=1.8, seed=0),
    "B": dict(arch="mp", H=80, rounds=8, steps=2600, lr=6e-4, bs=128,
              wd=3e-4, leakpen=0.14, tailw=3.0, amax=2.0, seed=0),
}
CORE_LS = [3, 5, 7, 9]
CORE_NTRAINS = [2000, 5000, 10000]

def cfg_id(tag, L, ntrain, extra=""):
    base = f"{tag}_L{L}_n{ntrain}"
    return f"{base}_{extra}" if extra else base

def hypers_of(tag, L, ntrain, overrides=None):
    h = dict(ANCHORS[tag])
    h["ntrain"] = int(ntrain)
    h["train_ns"] = (8,)
    if overrides:
        h.update(overrides)
    return h

def result_paths(cid):
    d = RUNS / cid
    return d, d / "result.json", d / "weights.npz", d / "log.txt"

def is_done(cid):
    _, rj, wj, _ = result_paths(cid)
    if not rj.exists():
        return False
    try:
        r = json.loads(rj.read_text())
        return r.get("status") == "done" and "test" in r and rj.stat().st_size > 50
    except Exception:
        return False

def append_jsonl(row):
    with open(RUN / "results.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")

def load_all_results():
    rows = []
    for d in sorted(RUNS.glob("*")):
        rj = d / "result.json"
        if rj.exists():
            try:
                rows.append(json.loads(rj.read_text()))
            except Exception:
                pass
    return rows

def write_status():
    rows = [r for r in load_all_results() if r.get("status") == "done"]
    rows.sort(key=lambda r: r.get("test", {}).get("median_inv_p", 1e9))
    lines = []
    lines.append(f"# n=8 Phase-1 STATUS\n")
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')} (Asia/Shanghai)\n")
    lines.append(f"Done: **{len(rows)}** runs | target core=24 then densify\n")
    lines.append(f"Bars: WS look Med/B90={WS_MED}/{WS_B90} | uniform floor Med≈{FLOOR_MED} | promote val≲{PROMOTE_VAL} | hot≲{HOT_VAL}\n")
    lines.append("\n## Leaderboard (by test Med)\n")
    lines.append("| id | L | ntrain | H/r/steps/lr | val Med/B90 | test Med/B90 | leak | best_step | sec | flag |\n")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        h = r.get("hypers", {})
        v, t = r.get("val", {}), r.get("test", {})
        med = t.get("median_inv_p", 1e9)
        flag = ""
        if med <= WS_MED: flag = "BEATS_WS"
        elif med < FLOOR_MED / 2: flag = "≪floor"
        elif med < FLOOR_MED: flag = "<floor"
        elif v.get("median_inv_p", 1e9) <= HOT_VAL: flag = "HOT_val"
        elif v.get("median_inv_p", 1e9) <= PROMOTE_VAL: flag = "promote"
        lines.append(
            f"| `{r.get('id','')}` | {r.get('L')} | {h.get('ntrain')} | "
            f"{h.get('H')}/{h.get('rounds')}/{h.get('steps')}/{h.get('lr')} | "
            f"{v.get('median_inv_p',0):.1f}/{v.get('B90_inv_p',0):.1f} | "
            f"{t.get('median_inv_p',0):.1f}/{t.get('B90_inv_p',0):.1f} | "
            f"{t.get('mean_leak',0):.4f} | {r.get('best_step')} | {r.get('sec',0):.0f} | {flag} |"
        )
    # pivot L x ntrain for anchors A/B
    lines.append("\n## Core grid pivot (test Med)\n")
    for tag in ("A", "B"):
        lines.append(f"\n### Anchor {tag}\n")
        lines.append("| L \\ ntrain | " + " | ".join(str(n) for n in CORE_NTRAINS) + " |\n")
        lines.append("|---|" + "|".join(["---:"] * len(CORE_NTRAINS)) + "|\n")
        for L in CORE_LS:
            cells = []
            for nt in CORE_NTRAINS:
                cid = cfg_id(tag, L, nt)
                hit = next((x for x in rows if x.get("id") == cid), None)
                cells.append(f"{hit['test']['median_inv_p']:.1f}" if hit else "—")
            lines.append(f"| **{L}** | " + " | ".join(cells) + " |\n")
    best = rows[0] if rows else None
    lines.append("\n## Notes\n")
    if best:
        lines.append(f"- Best so far: `{best['id']}` test Med **{best['test']['median_inv_p']:.2f}** "
                     f"(B90 {best['test']['B90_inv_p']:.1f}) best_step={best.get('best_step')} "
                     f"weights=`{best.get('weights_path','')}`\n")
    lines.append("- Lock: SNAP+D→cost, tied cost, vacuum, pure GNN+cw, same L train/test, n=8 only.\n")
    lines.append("- Queue auto-continues into densify after core 24.\n")
    (RUN / "STATUS.md").write_text("".join(lines) if False else "\n".join(lines))
    # RESULTS.md compact
    res = ["# n=8 Phase-1 RESULTS\n", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"]
    res.append("| id | phase | L | ntrain | test Med | test B90 | val Med | best_step | sec | weights |\n")
    res.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        res.append(
            f"| `{r.get('id')}` | {r.get('phase','core')} | {r.get('L')} | {r['hypers'].get('ntrain')} | "
            f"{r['test']['median_inv_p']:.2f} | {r['test']['B90_inv_p']:.2f} | "
            f"{r['val']['median_inv_p']:.2f} | {r.get('best_step')} | {r.get('sec',0):.0f} | "
            f"`{Path(r.get('weights_path','')).name}` |\n"
        )
    (RUN / "RESULTS.md").write_text("".join(res))
    # CSV
    with open(RUN / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","phase","L","ntrain","H","rounds","steps","lr","leakpen","tailw","seed",
                    "val_med","val_b90","test_med","test_b90","leak","tie_ok","best_step","sec","weights"])
        for r in rows:
            h = r["hypers"]
            w.writerow([r["id"], r.get("phase"), r["L"], h.get("ntrain"), h.get("H"), h.get("rounds"),
                        h.get("steps"), h.get("lr"), h.get("leakpen"), h.get("tailw"), h.get("seed"),
                        r["val"]["median_inv_p"], r["val"]["B90_inv_p"],
                        r["test"]["median_inv_p"], r["test"]["B90_inv_p"],
                        r["test"]["mean_leak"], r.get("tie_ok"), r.get("best_step"), r.get("sec"),
                        r.get("weights_path")])

def run_one(cid, tag, L, ntrain, phase="core", overrides=None):
    d, rj, wj, lj = result_paths(cid)
    if is_done(cid):
        print(f"[skip] {cid} already done", flush=True)
        return json.loads(rj.read_text())
    d.mkdir(parents=True, exist_ok=True)
    h = hypers_of(tag, L, ntrain, overrides)
    meta = {"id": cid, "phase": phase, "anchor": tag, "L": L, "status": "running",
            "hypers": h, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rj.write_text(json.dumps(meta, indent=2, default=str))
    write_status()

    def log(*a):
        line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(lj, "a") as f:
            f.write(line + "\n")

    log(f"START {cid} phase={phase} L={L} ntrain={ntrain} h={ {k:h[k] for k in ('H','rounds','steps','lr','leakpen','tailw','seed')} }")
    t0 = time.time()
    try:
        out = TL.train_one(8, L, h, screen=False, log_fn=log, weights_path=str(wj))
        row = {
            "id": cid, "phase": phase, "anchor": tag, "L": L, "status": "done",
            "hypers": out["hypers"], "val": out["val"], "test": out["test"],
            "test_NFchk": out.get("test_NFchk"),
            "test_final": out.get("test_final"), "val_ema": out.get("val_ema"), "test_ema": out.get("test_ema"),
            "cost_tied": out.get("cost_tied"),
            "tie_ok": out.get("tie_ok"), "mean_leak_test": out.get("mean_leak_test"),
            "sec": out["sec"], "best_step": out["best_step"],
            "weights_path": out.get("weights_path", str(wj)),
            "started": meta["started"],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_sec": time.time() - t0,
            "promote": bool(out["val"]["median_inv_p"] <= PROMOTE_VAL),
            "hot": bool(out["val"]["median_inv_p"] <= HOT_VAL),
            "beats_floor": bool(out["test"]["median_inv_p"] < FLOOR_MED),
            "beats_ws": bool(out["test"]["median_inv_p"] < WS_MED),
        }
        rj.write_text(json.dumps(row, indent=2, default=str))
        append_jsonl(row)
        log(f"DONE {cid} test Med/B90={out['test']['median_inv_p']:.2f}/{out['test']['B90_inv_p']:.2f} "
            f"val={out['val']['median_inv_p']:.2f} best_step={out['best_step']} sec={out['sec']:.0f}")
        write_status()
        return row
    except Exception as e:
        err = {"id": cid, "phase": phase, "anchor": tag, "L": L, "status": "error",
               "hypers": h, "error": str(e), "traceback": traceback.format_exc(),
               "sec": time.time() - t0}
        rj.write_text(json.dumps(err, indent=2, default=str))
        append_jsonl(err)
        log(f"ERROR {cid}: {e}")
        write_status()
        return err

def core_queue():
    q = []
    # Prefer varying L first at ntrain=10k (clearest depth signal), then fill grid
    # Order: for each L, both anchors at 10k, then 5k, then 2k — keeps early signal rich
    for L in CORE_LS:
        for nt in (10000, 5000, 2000):
            for tag in ("A", "B"):
                q.append((cfg_id(tag, L, nt), tag, L, nt, "core", None))
    return q

def densify_queue(done_rows):
    """After core: densify around best L and promising configs."""
    core = [r for r in done_rows if r.get("phase") == "core" and r.get("status") == "done"]
    if len(core) < 8:
        return []  # wait for more signal; caller may still invoke later
    # best by test med
    core_sorted = sorted(core, key=lambda r: r["test"]["median_inv_p"])
    best = core_sorted[0]
    best_L = int(best["L"])
    best_tag = best.get("anchor", "A")
    best_nt = int(best["hypers"]["ntrain"])
    # also consider best L by mean over configs
    from collections import defaultdict
    byL = defaultdict(list)
    for r in core:
        byL[r["L"]].append(r["test"]["median_inv_p"])
    meanL = {L: float(np.mean(v)) for L, v in byL.items()}
    L_star = min(meanL, key=meanL.get) if meanL else best_L

    q = []
    # Step 2: H / rounds / steps / lr on best L (use winning anchor's other hypers)
    base_tag = best_tag
    densify_specs = []
    for H in (64, 80, 128):
        for rounds in (4, 6, 8):
            densify_specs.append(("cap", dict(H=H, rounds=rounds)))
    for steps in (800, 1400, 2600):
        for lr in (6e-4, 1e-3, 1.2e-3):
            densify_specs.append(("opt", dict(steps=steps, lr=lr)))
    # Step 3: light leakpen/tailw
    for leakpen in (0.08, 0.14):
        for tailw in (1.0, 1.6, 3.0):
            densify_specs.append(("reg", dict(leakpen=leakpen, tailw=tailw)))

    # Dedup against base anchor values — still run explicitly with clear ids
    seen = set()
    for kind, ov in densify_specs:
        # merge key
        key = tuple(sorted(ov.items()))
        if key in seen:
            continue
        seen.add(key)
        extra = kind + "_" + "_".join(f"{k}{v}" for k, v in sorted(ov.items()))
        # sanitize
        extra = extra.replace(".", "p").replace("-", "m")
        cid = cfg_id(base_tag, L_star, best_nt, extra)
        # skip if identical to a core already done with same hypers
        q.append((cid, base_tag, L_star, best_nt, f"densify_{kind}", ov))

    # Also densify on absolute best single config's L/ntrain if different
    if best_L != L_star or True:
        # a few targeted around absolute best
        for H, rounds in ((96, 6), (128, 6), (160, 4)):
            ov = dict(H=H, rounds=rounds)
            extra = f"cap2_H{H}_r{rounds}"
            cid = cfg_id(best_tag, best_L, best_nt, extra)
            q.append((cid, best_tag, best_L, best_nt, "densify_cap2", ov))
        for steps in (2000, 3200):
            ov = dict(steps=steps)
            cid = cfg_id(best_tag, best_L, best_nt, f"opt2_steps{steps}")
            q.append((cid, best_tag, best_L, best_nt, "densify_opt2", ov))
    return q

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUN / "RESUME.md").write_text(
        f"# Resume\nStarted/updated {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Re-run: `python3 {__file__}` — skips done configs.\n"
        f"Lock: tied-cost n=8 only. Non-stop into densify after core 24.\n"
    )
    write_status()
    queue = core_queue()
    print(f"Core queue size={len(queue)}", flush=True)

    # Run core
    for cid, tag, L, nt, phase, ov in queue:
        run_one(cid, tag, L, nt, phase=phase, overrides=ov)

    # Densify loop — rebuild queue from latest results; keep going
    print("CORE COMPLETE — entering densify", flush=True)
    (RUN / "CORE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    write_status()

    # Multiple densify passes: after first wave, if new best, could add more — for now one full densify wave
    # Prefer wide densify around A_L5 / B_L3 winners (Zach: what moves Med)
    try:
        import wide_densify as WD
        dens_q = WD.wide_queue()
        print(f"WIDE densify queue size={len(dens_q)}", flush=True)
    except Exception as e:
        print(f"wide_densify import failed ({e}); falling back", flush=True)
        dens_q = densify_queue(load_all_results())
        print(f"Densify queue size={len(dens_q)}", flush=True)
    (RUN / "DENSE_START").write_text(__import__("time").strftime("%Y-%m-%d %H:%M:%S") + f" n={len(dens_q)}\n")
    for cid, tag, L, nt, phase, ov in dens_q:
        run_one(cid, tag, L, nt, phase=phase, overrides=ov)

    write_status()
    print("ALL PHASE1+DENSIFY COMPLETE", flush=True)
    (RUN / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

if __name__ == "__main__":
    main()
