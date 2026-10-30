#!/usr/bin/env python3
"""Opus #6 — paper-ready depth ladder L16→32 for projector-Uc lineage.

Compiles sealed scale1000 Med/B90 from depth/soft_n15/tail RESULTS + backups.
Optionally re-evals named checkpoints (--reeval). Writes
runs/gadget_Uc/depth_ladder/{RESULTS.md,ladder.csv,ladder.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    _jsonable,
    build_W_BinH,
    scale1000_eval,
)

OUT = _PKG / "runs" / "gadget_Uc" / "depth_ladder"
ROOT = _PKG / "runs" / "gadget_Uc"

# Sealed historical scale1000 (Med, B90) — projector unsat BinH lineage
SEALED = [
    {
        "L": 16,
        "tag": "L16_train6912",
        "source": "train6912/RESULTS.md",
        "params": "unsat_params_L16_train6912_backup.pt",
        "Med": {9: 1.406, 12: 1.864, 15: 2.754, 18: 4.194},
        "B90": {9: 3.288, 12: 6.106, 15: 10.244, 18: 24.302},
    },
    {
        "L": 20,
        "tag": "L20_retrain",
        "source": "depth/RESULTS.md",
        "params": "unsat_params_L20_backup.pt",
        "Med": {9: 1.261, 12: 1.590, 15: 2.168, 18: 3.170},
        "B90": {9: 2.657, 12: 4.760, 15: 8.296, 18: 19.916},
    },
    {
        "L": 24,
        "tag": "L24_retrain",
        "source": "depth/RESULTS.md",
        "params": "unsat_params_L24_backup.pt",
        "Med": {9: 1.185, 12: 1.392, 15: 1.820, 18: 2.492},
        "B90": {9: 2.245, 12: 3.865, 15: 7.283, 18: 17.005},
    },
    {
        "L": 28,
        "tag": "L28_retrain",
        "source": "depth/RESULTS.md",
        "params": "unsat_params_L28_backup.pt",
        "Med": {9: 1.144, 12: 1.317, 15: 1.655, 18: 2.183},
        "B90": {9: 1.962, 12: 3.364, 15: 6.385, 18: 15.419},
    },
    {
        "L": 32,
        "tag": "L32_plain",
        "source": "soft_n15/RESULTS.md (L32 champ ref)",
        "params": "unsat_params_L32_plain_backup.pt",
        "Med": {9: 1.109, 12: 1.258, 15: 1.537, 18: 1.995},
        "B90": {9: 1.828, 12: 3.013, 15: 5.767, 18: 14.117},
    },
    {
        "L": 32,
        "tag": "L32_softn15_t91215",
        "source": "soft_n15/RESULTS.md",
        "params": "unsat_params_softn15_t91215_backup.pt",
        "Med": {9: 1.100, 12: 1.231, 15: 1.483, 18: 1.884},
        "B90": {9: 1.759, 12: 2.919, 15: 5.847, 18: 13.704},
    },
    {
        "L": 32,
        "tag": "L32_tail_plain",
        "source": "tail_retrain/RESULTS.md (current champ)",
        "params": "unsat_params.pt",
        "Med": {9: 1.086, 12: 1.192, 15: 1.411, 18: 1.752},
        "B90": {9: 1.799, 12: 2.829, 15: 5.520, 18: 11.813},
        "champ": True,
    },
]


def log(*a):
    print(time.strftime("%H:%M:%S"), "[ladder]", *a, flush=True)


def enrich(row):
    ns = [9, 12, 15, 18]
    meds = [row["Med"][n] for n in ns]
    b90s = [row["B90"][n] for n in ns]
    row["b_Med"] = fit_b(ns, meds)
    row["b_B90"] = fit_b(ns, b90s)
    return row


def write_outputs(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    # CSV
    with (OUT / "ladder.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "L", "tag", "champ", "source",
            "Med9", "B909", "Med12", "B9012", "Med15", "B9015", "Med18", "B9018",
            "b_Med", "b_B90", "C_Med", "C_B90",
        ])
        for r in rows:
            w.writerow([
                r["L"], r["tag"], int(bool(r.get("champ"))), r["source"],
                r["Med"][9], r["B90"][9], r["Med"][12], r["B90"][12],
                r["Med"][15], r["B90"][15], r["Med"][18], r["B90"][18],
                r["b_Med"]["b"], r["b_B90"]["b"],
                r["b_Med"]["C"], r["b_B90"]["C"],
            ])
    # JSON
    (OUT / "ladder.json").write_text(json.dumps(_jsonable({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S +0800"),
        "honesty": "Uc=exp(-iγ Hc) unsat projector BinH; sealed scale1000 lineage",
        "rows": rows,
    }), indent=2))
    # MD
    lines = [
        "# Depth ladder — projector-Uc BinH lineage (L16→32)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} +0800 (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. Sealed scale1000 Med/B90; b from log-linear fit on n∈{9,12,15,18}.",
        "",
        "## Ladder",
        "",
        "| L | tag | n9 | n12 | n15 | n18 | b_Med | b_B90 |",
        "|---|-----|----|-----|-----|-----|-------|-------|",
    ]
    for r in rows:
        star = " **" if r.get("champ") else ""
        end = "**" if r.get("champ") else ""
        lines.append(
            f"| {r['L']} | {star}{r['tag']}{end} | "
            f"{r['Med'][9]:.3f}/{r['B90'][9]:.3f} | "
            f"{r['Med'][12]:.3f}/{r['B90'][12]:.3f} | "
            f"{r['Med'][15]:.3f}/{r['B90'][15]:.3f} | "
            f"{r['Med'][18]:.3f}/{r['B90'][18]:.3f} | "
            f"{r['b_Med']['b']:.4f} | {r['b_B90']['b']:.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Primary depth path: L16→20→24→28→32_plain→soft-n15→**tail_plain** (champ).",
        "- Extra L32 rows show soft-n15 / plain checkpoints before tail retrain.",
        "- Sources: `depth/`, `train6912/`, `soft_n15/`, `tail_retrain/` RESULTS.",
        "- No blind-cost / stencil arms.",
        "",
        "Files: `ladder.csv`, `ladder.json`.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reeval", default="", help="comma tags to re-eval (e.g. L28_retrain,L32_tail_plain)")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = [enrich(dict(r)) for r in SEALED]
    tags = {x.strip() for x in args.reeval.split(",") if x.strip()}
    if tags:
        device = torch.device(args.device)
        W = build_W_BinH(nf=args.nf)
        for r in rows:
            if r["tag"] not in tags:
                continue
            pt = ROOT / r["params"]
            if not pt.exists():
                log("missing", pt); continue
            st = torch.load(pt, map_location="cpu", weights_only=False)
            L = int(st.get("L", st["gamma"].numel()))
            log(f"reeval {r['tag']} L={L}")
            sc = scale1000_eval(
                W, st["gamma"].float(), st["phi"].float(), L, args.nf, device, "unsat",
                ns=[9, 12, 15, 18], max_count=args.scale_B,
            )
            sc_i = {int(k): v for k, v in sc.items()}
            r["Med"] = {n: sc_i[n]["median_inv_p"] for n in (9, 12, 15, 18)}
            r["B90"] = {n: sc_i[n]["B90_inv_p"] for n in (9, 12, 15, 18)}
            r["source"] = f"reeval scale1000 B={args.scale_B} from {r['params']}"
            enrich(r)
    write_outputs(rows)


if __name__ == "__main__":
    main()
