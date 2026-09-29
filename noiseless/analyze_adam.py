#!/usr/bin/env python3
"""SPSA vs SPSA-Adam on jp (binary, λ=0 Gibbs, 20 four_sat × 25 trials, seed 20260917).

Arms:
  A       SPSA, random init L, 200 steps (existing baselines)
  A_adam  SPSA-Adam (lr 0.05), random init L, 200 steps
  C       SPSA, --grow --grow-steps-per-stage 200 (existing)
  C_adam  SPSA-Adam, --grow --grow-steps-per-stage 200 (fresh Adam state per stage)

Stats as in analyze_encoding_grow.py, plus failure-mode classification: the energy level
(rank among distinct energies of the instance, 0 = ground) of the argmax bitstring for
failed trials. Reads the full (gitignored) JSONs; writes adam_stats_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noiseless.analyze_encoding_grow import _latest, _paired, _stats
from noiseless.encoding import load_four_sat_npz
from noiseless.spsa_gibbs import ground_flat_from_bitstring

_REPO = Path(__file__).resolve().parents[1]
_RES = _REPO / "noiseless" / "results"
_HAM = _REPO / "Hamiltonians" / "four_sat"

SPSA_A = {2: "jp_L2_20260929T041311Z.json", 3: "jp_L3_20260929T041643Z.json",
          4: "jp_B0_20260928T042134Z.json"}

_levels_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _level_of(ham_file: str, bitstring: str) -> int:
    if ham_file not in _levels_cache:
        inst = load_four_sat_npz(_HAM / ham_file)
        e = np.asarray(inst["energies_flat"], dtype=float)
        _levels_cache[ham_file] = (e, np.unique(np.round(e, 9)))
    e, lv = _levels_cache[ham_file]
    en = round(float(e[ground_flat_from_bitstring(bitstring)]), 9)
    return int(np.searchsorted(lv, en))


def _failure_modes(recs: list[dict]) -> dict:
    fails = [r for r in recs if not r["success"]]
    if not fails:
        return {"n_fail": 0}
    lv = np.array([_level_of(r["ham_file"], r["most_likely_bitstring"]) for r in fails])
    pg = np.array([r["p_gs"] for r in fails])
    return {
        "n_fail": len(fails),
        "frac_first_excited": float(np.mean(lv == 1)),
        "frac_level_ge2": float(np.mean(lv >= 2)),
        "mean_level": float(lv.mean()),
        "mean_p_gs_fail": float(pg.mean()),
        "median_p_gs_fail": float(np.median(pg)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(_RES / "adam_stats_summary.json"))
    args = ap.parse_args()
    specs: list[tuple[str, int, str]] = [("A", L, f) for L, f in SPSA_A.items()]
    for label, prefix in (("A_adam", "jp_adam_A"), ("C", "jp_enc_C_bin_grow_ps200"),
                          ("C_adam", "jp_adam_C_grow_ps200")):
        f = _latest(prefix)
        if f:
            specs += [(label, L, f) for L in (2, 3, 4)]
    cache: dict[str, dict] = {}
    recs_by: dict[tuple, list[dict]] = {}
    table: dict = {"files": {}, "paired": {}}
    for label, L, f in specs:
        if f not in cache:
            cache[f] = json.loads((_RES / f).read_text())
        recs = [r for r in cache[f]["records"] if r.get("ok") and int(r["n_layers"]) == L]
        recs_by[(label, L)] = recs
        s = _stats(recs)
        s["nfev_mean"] = float(np.mean([r["nfev"] for r in recs]))
        s["nfev_set"] = sorted({int(r["nfev"]) for r in recs})
        s["optimizer"] = recs[0].get("optimizer", "spsa")
        s["adam_lr"] = recs[0].get("adam_lr")
        s["failure_modes"] = _failure_modes(recs)
        table[f"{label}_L{L}"] = s
        table["files"][f"{label}_L{L}"] = f
    for base, arm in (("A", "A_adam"), ("C", "C_adam"), ("A", "C_adam")):
        for L in (2, 3, 4):
            if (base, L) in recs_by and (arm, L) in recs_by:
                table["paired"][f"{arm}_vs_{base}_L{L}"] = _paired(recs_by[(base, L)], recs_by[(arm, L)])
    Path(args.out_json).write_text(json.dumps(table, indent=2))
    print("| arm | L | nfev/trial | success | mean p(GS) | median p(GS) | frac>0.5 | best-of-25 | wall/trial s "
          "| fails: frac 1st-exc / mean p(GS) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for label, L, _ in sorted(specs, key=lambda t: (t[1], t[0])):
        s = table[f"{label}_L{L}"]
        fm = s["failure_modes"]
        fstr = f"{fm['frac_first_excited']:.2f} / {fm['mean_p_gs_fail']:.3f}" if fm["n_fail"] else "-"
        print(f"| {label} | {L} | {s['nfev_mean']:.0f} | {s['success']:.3f} | {s['mean_p_gs']:.4f} "
              f"| {s['median_p_gs']:.4f} | {s['frac_p_gs_gt_0p5']:.3f} | {s['best_of_25_mean_p_gs']:.4f} "
              f"| {s['mean_wall_s']:.2f} | {fstr} |")
    print()
    for k, v in table["paired"].items():
        print(k, v)
    for k, v in table.items():
        if isinstance(v, dict) and v.get("stages"):
            print(k, [(st["n_layers"], round(st["success"], 3), round(st["mean_p_gs"], 3)) for st in v["stages"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
