#!/usr/bin/env python3
"""Forever-style evolve over qumode-native gate sequences (not GNN hypers).

Loop: seed/mutate Motif → short fixed-angle train on n6&9 → zero-shot n12
      → promote only if n12 S beats champion (and beats long-confirm baseline).
Longer confirm on promote. CSV: runs/seq_evolve_trials.csv
Stop: touch runs/STOP_SEQ_EVOLVE  or  --max_trials N

Direction (Zach 2026-10-29): QAOA-like sandwich is a dead end for transfer;
explore native-efficient motifs (SNAP/D/COST ordering, SNAP–D–SNAP, prep).
No free U_phase^q / dense nonlocal phase heads.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.device import resolve_device  # noqa: E402
from explore_OCT29.lib.seq_ansatz import (  # noqa: E402
    SEED_MOTIFS,
    Motif,
    mutate_motif,
)
from explore_OCT29.lib.seq_train import (  # noqa: E402
    save_seq_best,
    short_train_motif,
)
from explore_OCT29.lib.train_loop import RUNS  # noqa: E402

CSV_NAME = "seq_evolve_trials.csv"
STOP_NAME = "STOP_SEQ_EVOLVE"
BEST_DIR = "seq_best"
BASELINE_JSON = "long_confirm_1500.json"


def _load_n12_baseline(runs: Path) -> float:
    """Champion must beat long-confirm n12 S (if available); else start loose."""
    p = runs / BASELINE_JSON
    if not p.exists():
        # also check gnn_best_long_confirm
        p2 = runs / "gnn_best_long_confirm" / BASELINE_JSON
        p = p2 if p2.exists() else p
    if p.exists():
        d = json.loads(p.read_text())
        m = d.get("metrics", {}).get("12") or d.get("metrics", {}).get(12)
        if m and "S" in m:
            return float(m["S"])
        if m and "median_inv_p" in m:
            import math
            return 0.5 * (math.log(m["median_inv_p"]) + math.log(m["B90_inv_p"]))
    return 30.0  # loose if missing


def _load_champ(runs: Path):
    d = runs / BEST_DIR
    mj = d / "motif.json"
    if not mj.exists():
        return None, None, float("inf")
    payload = json.loads(mj.read_text())
    motif = Motif.from_dict(payload["motif"])
    S = payload.get("info", {}).get("S_n12")
    if S is None and "12" in payload.get("metrics", {}):
        S = payload["metrics"]["12"]["S"]
    return motif, payload, float(S if S is not None else "inf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_trials", type=int, default=0, help="0 = forever")
    ap.add_argument("--steps", type=int, default=80, help="short screen steps")
    ap.add_argument("--confirm_steps", type=int, default=400, help="on promote")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max_val", type=int, default=16)
    ap.add_argument("--max_n12", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--eps_mutate", type=float, default=0.65)
    ap.add_argument("--min_improve", type=float, default=0.02, help="relative S improve to promote")
    ap.add_argument("--nf", type=int, default=None)
    args = ap.parse_args(argv)

    device = resolve_device(args.device or None)
    runs = RUNS
    runs.mkdir(parents=True, exist_ok=True)
    csv_path = runs / CSV_NAME
    stop_path = runs / STOP_NAME

    baseline_S = _load_n12_baseline(runs)
    parent_motif, champ_payload, best_S = _load_champ(runs)
    # must also beat long-confirm baseline to be "useful"
    gate_S = min(best_S, baseline_S)
    print(
        f"evolve_seq device={device} short_steps={args.steps} confirm={args.confirm_steps} "
        f"baseline_n12_S={baseline_S:.4f} champ_S={best_S if best_S < 1e20 else float('inf'):.4f} "
        f"gate_S={gate_S:.4f}",
        flush=True,
    )
    if parent_motif is None:
        parent_motif = SEED_MOTIFS[0]
        print("no seq_best yet — parent=baseline_snap_d_cost", flush=True)

    fieldnames = [
        "trial_id", "wall_s", "keep", "mode", "S_n12", "S_train", "best_S_n12",
        "baseline_S", "n_params", "motif_json", "Med6", "B90_6", "Med9", "B90_9",
        "Med12", "B90_12",
    ]
    new_file = not csv_path.exists()
    rng = np.random.default_rng(args.seed)
    trial = 0
    seed_i = 0

    with open(csv_path, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
            fcsv.flush()

        while True:
            if stop_path.exists():
                print(f"{STOP_NAME} present — exit", flush=True)
                break
            if args.max_trials and trial >= args.max_trials:
                print("max_trials reached — exit", flush=True)
                break

            trial += 1
            t0 = time.time()
            # first pass: exhaust seed motifs, then mutate
            if seed_i < len(SEED_MOTIFS):
                cand = SEED_MOTIFS[seed_i]
                mode = "seed"
                seed_i += 1
            elif rng.random() < args.eps_mutate:
                cand = mutate_motif(parent_motif, rng)
                mode = "mutate"
            else:
                cand = SEED_MOTIFS[int(rng.integers(0, len(SEED_MOTIFS)))]
                # randomize L a bit
                L = int(np.clip(cand.L + int(rng.choice([-1, 0, 1])), 3, 8))
                cand = Motif(name=f"{cand.name}_L{L}", prep=cand.prep, layer=cand.layer, L=L)
                mode = "reseed"

            print(f"\n=== trial {trial} mode={mode} motif={cand.name} "
                  f"prep={cand.prep} layer={cand.layer} L={cand.L} n_params={cand.n_params()}",
                  flush=True)
            try:
                metrics, params, info = short_train_motif(
                    cand,
                    steps=args.steps,
                    lr=args.lr,
                    max_val=args.max_val,
                    max_n12=args.max_n12,
                    seed=args.seed + trial,
                    device=device,
                    nf=args.nf,
                )
            except Exception as e:
                print(f"trial {trial} FAILED: {e}", flush=True)
                w.writerow({
                    "trial_id": trial, "wall_s": f"{time.time()-t0:.1f}", "keep": "error",
                    "mode": mode, "S_n12": "", "S_train": "", "best_S_n12": f"{best_S:.6f}",
                    "baseline_S": f"{baseline_S:.6f}", "n_params": cand.n_params(),
                    "motif_json": json.dumps(cand.to_dict()),
                    "Med6": "", "B90_6": "", "Med9": "", "B90_9": "", "Med12": "", "B90_12": "",
                })
                fcsv.flush()
                continue

            m6 = metrics.get(6, {})
            m9 = metrics.get(9, {})
            m12 = metrics.get(12, {})
            S_n12 = float(m12["S"]) if m12 else float("inf")
            S_train = 0.35 * float(m6.get("S", 99)) + 0.65 * float(m9.get("S", 99))

            # promote: beat current champ AND show progress vs catastrophic baseline
            # relative improve vs gate_S (min of champ, baseline)
            improve = (gate_S - S_n12) / max(abs(gate_S), 1e-6)
            keep = S_n12 < gate_S * (1.0 - args.min_improve) if gate_S < 1e20 else S_n12 < 20.0
            # also require finite sensible n12 (not 1e8 class) — Med < 1e5 as sanity
            if m12 and m12.get("median_inv_p", 1e99) > 1e5:
                keep = False
                keep_tag = "reject_n12_explode"
            elif keep:
                keep_tag = "promote"
            else:
                keep_tag = "no"

            row = {
                "trial_id": trial,
                "wall_s": f"{time.time()-t0:.1f}",
                "keep": keep_tag,
                "mode": mode,
                "S_n12": f"{S_n12:.6f}",
                "S_train": f"{S_train:.6f}",
                "best_S_n12": f"{best_S if best_S < 1e20 else -1:.6f}",
                "baseline_S": f"{baseline_S:.6f}",
                "n_params": cand.n_params(),
                "motif_json": json.dumps(cand.to_dict()),
                "Med6": f"{m6.get('median_inv_p', float('nan')):.4f}",
                "B90_6": f"{m6.get('B90_inv_p', float('nan')):.4f}",
                "Med9": f"{m9.get('median_inv_p', float('nan')):.4f}",
                "B90_9": f"{m9.get('B90_inv_p', float('nan')):.4f}",
                "Med12": f"{m12.get('median_inv_p', float('nan')):.4f}",
                "B90_12": f"{m12.get('B90_inv_p', float('nan')):.4f}",
            }
            w.writerow(row)
            fcsv.flush()
            print(
                f"trial {trial} keep={keep_tag} S_n12={S_n12:.4f} S_train={S_train:.4f} "
                f"Med12={m12.get('median_inv_p', float('nan')):.3g} "
                f"improve_vs_gate={improve:.3%}",
                flush=True,
            )

            if keep_tag == "promote":
                # longer confirm before locking
                print(f"CONFIRM {args.confirm_steps} steps for {cand.name} ...", flush=True)
                metrics_c, params_c, info_c = short_train_motif(
                    cand,
                    steps=args.confirm_steps,
                    lr=args.lr,
                    max_val=max(args.max_val, 24),
                    max_n12=max(args.max_n12, 16),
                    seed=args.seed + trial + 10000,
                    device=device,
                    nf=args.nf,
                )
                m12c = metrics_c.get(12, {})
                S_c = float(m12c["S"]) if m12c else float("inf")
                med_c = m12c.get("median_inv_p", 1e99)
                if S_c < gate_S and med_c < 1e5:
                    save_seq_best(cand, params_c, metrics_c, info_c, runs_dir=runs, tag=BEST_DIR)
                    best_S = S_c
                    gate_S = min(best_S, baseline_S)
                    parent_motif = cand
                    print(f"PROMOTED {cand.name} S_n12={S_c:.4f} Med12={med_c:.3f} → runs/{BEST_DIR}/", flush=True)
                    # milestone stamp
                    (runs / "seq_milestone.json").write_text(json.dumps({
                        "trial": trial,
                        "motif": cand.to_dict(),
                        "S_n12": S_c,
                        "metrics": {str(k): v for k, v in metrics_c.items()},
                        "wall_s": time.time() - t0,
                    }, indent=2) + "\n")
                else:
                    print(
                        f"confirm FAILED to hold (S={S_c:.4f} Med12={med_c:.3g}) — not promoting",
                        flush=True,
                    )
                    # still adopt as parent if short screen was much better than parent train
                    if S_n12 < best_S:
                        parent_motif = cand
            else:
                # mild parent chase on train S if n12 not explode
                if m12 and m12.get("median_inv_p", 1e99) < 1e5 and S_n12 < best_S * 1.05:
                    parent_motif = cand


if __name__ == "__main__":
    main()
