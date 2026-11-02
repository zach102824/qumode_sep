#!/usr/bin/env python3
"""Add n=6 scale1000 B=1000 column to existing progressive-shell L-ladder.

Reuses checkpoints from L_LADDER_scale1000.json. Does not re-eval n9/12/15/18.
No sealed bar for n6 — absolute Med/B90 only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

_NOV = Path(__file__).resolve().parents[1]
_ROOT = _NOV.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.scripts.gadget_Uc_explore import _jsonable  # noqa: E402
from Nov.scripts.angle_explore import L_LADDER, OUT, log, stopped  # noqa: E402
from Nov.scripts.nov1st_final_train import SEALED_REF  # noqa: E402
from Nov.scripts.run_L_ladder_scale1000 import (  # noqa: E402
    B,
    N_WORKERS,
    PROG,
    RESULT_JSON,
    RESULT_MD,
    load_state,
    scale1000_parallel,
)

NS_EXISTING = [9, 12, 15, 18]
NS_ALL = [6] + NS_EXISTING


def write_outputs(entries: list[dict]):
    payload = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "B": B,
        "ns": NS_ALL,
        "sealed": SEALED_REF,
        "L_list": list(L_LADDER),
        "n_workers": N_WORKERS,
        "train_ns_note": "progressive shell trained on n∈{6,9} (max_train=200 each)",
        "entries": entries,
    }
    RESULT_JSON.write_text(json.dumps(_jsonable(payload), indent=2))

    lines = [
        "# Progressive shell L-ladder — scale1000 B=1000",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai_",
        "",
        "Best progressive checkpoint per L (lowest train-val n9 Med). "
        "Motif Nov1st-final, SNAP head = shell (perL). "
        f"Eval B={B} on n∈{NS_ALL} (parallel workers={N_WORKERS}).",
        "",
        "**Train:** n∈{6,9} (max_train=200 each). **n=6** has no sealed bar — absolute Med/B90 only.",
        "",
        "## Sealed bar (L=32) — n9/12/15/18 only",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in NS_EXISTING:
        r = SEALED_REF[n]
        lines.append(f"| {n} | {r['Med']:.3f} | {r['B90']:.3f} |")

    lines += [
        "",
        "## Ladder (Med / B90)",
        "",
        "| L | seed | n6 Med/B90 | n9 Med/B90 | n12 Med/B90 | n15 Med/B90 | n18 Med/B90 | "
        "Med× / B90× (n9,12,15,18) |",
        "|---|------|------------|------------|-------------|-------------|-------------|"
        "---------------------------|",
    ]
    for e in entries:
        sc = e.get("scale") or {}
        abs_cells = []
        for n in NS_ALL:
            m = sc.get(str(n)) or sc.get(n) or {}
            if not m:
                abs_cells.append("—")
                continue
            med = m["median_inv_p"]
            b90 = m["B90_inv_p"]
            abs_cells.append(f"{med:.3f}/{b90:.3f}")
        x_parts = []
        for n in NS_EXISTING:
            m = sc.get(str(n)) or sc.get(n) or {}
            if not m:
                x_parts.append("—")
                continue
            med = m["median_inv_p"]
            b90 = m["B90_inv_p"]
            ref = SEALED_REF[n]
            x_parts.append(f"{med / ref['Med']:.3f}/{b90 / ref['B90']:.3f}")
        lines.append(
            f"| {e['L']} | {e['seed']} | "
            + " | ".join(abs_cells)
            + " | "
            + ", ".join(x_parts)
            + " |"
        )

    lines += [
        "",
        "## Checkpoint selection",
        "",
        "| L | seed | train-val n6 Med | train-val n9 Med | params |",
        "|---|------|------------------|------------------|--------|",
    ]
    for e in entries:
        n6v = e.get("n6_med_val")
        n6s = f"{n6v:.4f}" if isinstance(n6v, (int, float)) and n6v == n6v else "—"
        lines.append(
            f"| {e['L']} | {e['seed']} | {n6s} | {e['n9_med_val']:.4f} | "
            f"`{Path(e['params_path']).name}` |"
        )
    lines += ["", f"JSON: `{RESULT_JSON.name}`", ""]
    RESULT_MD.write_text("\n".join(lines))
    log(f"wrote {RESULT_MD} and {RESULT_JSON}")


def main():
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not RESULT_JSON.exists():
        raise SystemExit(f"missing {RESULT_JSON}; run run_L_ladder_scale1000.py first")

    payload = json.loads(RESULT_JSON.read_text())
    entries = payload["entries"]
    by_L = {int(e["L"]): e for e in entries}

    for L in L_LADDER:
        if stopped():
            log("STOP — writing partial")
            break
        e = by_L.get(L)
        if e is None:
            log(f"missing entry L={L}")
            continue
        sc = e.get("scale") or {}
        # normalize keys to int
        sc_norm = {int(k): v for k, v in sc.items()}
        if 6 in sc_norm and int(sc_norm[6].get("B", 0)) >= B:
            log(f"reuse n6 L={L} Med={sc_norm[6]['median_inv_p']:.3f}")
            e["scale"] = sc_norm
            continue

        pt = Path(e["params_path"])
        st = load_state(pt)
        log(f"=== scale1000 n6 B={B} L={L} seed{e['seed']} workers={N_WORKERS} ===")
        t0 = time.time()
        sc6 = scale1000_parallel(st, 8, [6], max_count=B, n_workers=N_WORKERS)
        log(f"done n6 L={L} in {time.time() - t0:.1f}s")
        if 6 not in sc6:
            log(f"FAILED n6 L={L}")
            e["scale"] = sc_norm
            continue
        sc_norm[6] = sc6[6]
        e["scale"] = sc_norm
        # also patch per-L scale json if present
        tag = e.get("tag") or f"prog_shell_L{L}_B1000"
        out_json = PROG / f"scale1000_{tag}.json"
        if out_json.exists():
            try:
                old = json.loads(out_json.read_text())
                old_sc = {int(k): v for k, v in (old.get("scale") or {}).items()}
                old_sc[6] = sc6[6]
                old["scale"] = old_sc
                rows = old.get("rows") or []
                rows = [r for r in rows if int(r.get("n", -1)) != 6]
                rows.insert(0, {
                    "tag": tag, "n": 6, "L": L,
                    "Med": sc6[6]["median_inv_p"], "B90": sc6[6]["B90_inv_p"],
                    "Med_x": None, "B90_x": None,
                    "snap_mode": st.get("snap_mode", "shell"),
                    "snap_shared": st.get("snap_shared", False),
                    "B": sc6[6]["B"],
                    "wall_s": sc6[6].get("wall_s"),
                    "n_workers": sc6[6].get("n_workers"),
                })
                old["rows"] = rows
                out_json.write_text(json.dumps(_jsonable(old), indent=2))
            except Exception as ex:
                log(f"patch {out_json.name} failed: {ex}")
        write_outputs([by_L[x] for x in L_LADDER if x in by_L])

    entries_out = [by_L[x] for x in L_LADDER if x in by_L]
    write_outputs(entries_out)
    log("n6 column complete")


if __name__ == "__main__":
    main()
