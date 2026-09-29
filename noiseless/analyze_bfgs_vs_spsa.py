#!/usr/bin/env python3
"""Compare BFGS vs SPSA jp fleets (λ=0 and λ=2) -> markdown tables on stdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from noiseless.spsa_gibbs import betas_from_x  # noqa: E402

BETA_MAX = 2.1


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    recs = [r for r in d["records"] if r.get("ok")]
    for r in recs:
        ab = np.abs(betas_from_x(np.asarray(r["x"], dtype=float), int(r["n_layers"])))
        r["_mean_b"] = float(ab.mean())
        r["_max_b"] = float(ab.max())
        r["_frac_over"] = float(np.mean(ab > BETA_MAX))
    return {"args": d["args"], "records": recs, "n_fail": d["n_fail"]}


def fleet_stats(recs: list[dict], steps: int) -> dict:
    p = np.array([r["p_gs"] for r in recs])
    nit = np.array([r["nit"] for r in recs])
    nfev = np.array([r["nfev"] for r in recs])
    return {
        "n": len(recs),
        "success": float(np.mean([r["success"] for r in recs])),
        "mean_p": float(p.mean()),
        "median_p": float(np.median(p)),
        "mean_b": float(np.mean([r["_mean_b"] for r in recs])),
        "mean_max_b": float(np.mean([r["_max_b"] for r in recs])),
        "frac_over": float(np.mean([r["_frac_over"] for r in recs])),
        "trial_any_over": float(np.mean([r["_max_b"] > BETA_MAX for r in recs])),
        "mean_nfev": float(nfev.mean()),
        "median_nfev": float(np.median(nfev)),
        "mean_nit": float(nit.mean()),
        "frac_early": float(np.mean(nit < steps)),
        "mean_wall": float(np.mean([r["wall_s"] for r in recs])),
        "total_wall_min": float(np.sum([r["wall_s"] for r in recs]) / 60.0),
        "status": Counter(
            (r.get("opt_status"), r.get("opt_message")) for r in recs if r.get("opt_status") is not None
        ),
    }


def paired(spsa: list[dict], bfgs: list[dict], tol: float = 1e-6) -> dict:
    key = lambda r: (r["ham_file"], int(r["trial"]), int(r["seed"]))  # noqa: E731
    s = {key(r): r for r in spsa}
    b = {key(r): r for r in bfgs}
    common = sorted(set(s) & set(b))
    dp = np.array([b[k]["p_gs"] - s[k]["p_gs"] for k in common])
    sb = [(bool(s[k]["success"]), bool(b[k]["success"])) for k in common]
    return {
        "n": len(common),
        "unmatched": len(set(s) ^ set(b)),
        "win": float(np.mean(dp > tol)),
        "tie": float(np.mean(np.abs(dp) <= tol)),
        "lose": float(np.mean(dp < -tol)),
        "mean_dp": float(dp.mean()),
        "median_dp": float(np.median(dp)),
        "both": float(np.mean([x and y for x, y in sb])),
        "bfgs_only": float(np.mean([y and not x for x, y in sb])),
        "spsa_only": float(np.mean([x and not y for x, y in sb])),
        "neither": float(np.mean([not x and not y for x, y in sb])),
    }


def per_instance(recs: list[dict]) -> dict:
    out: dict[str, list[bool]] = {}
    for r in recs:
        out.setdefault(r["ham_file"], []).append(bool(r["success"]))
    return {k: (sum(v), len(v)) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spsa-b0", required=True)
    ap.add_argument("--bfgs-b0", required=True)
    ap.add_argument("--spsa-lam2", required=True)
    ap.add_argument("--bfgs-lam2", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    fleets = {
        "spsa_b0": load(a.spsa_b0),
        "bfgs_b0": load(a.bfgs_b0),
        "spsa_lam2": load(a.spsa_lam2),
        "bfgs_lam2": load(a.bfgs_lam2),
    }
    # Settings check: everything except optimizer must match per λ.
    skip = {"optimizer", "workers"}
    for lamk in ("b0", "lam2"):
        sa, ba = fleets[f"spsa_{lamk}"]["args"], fleets[f"bfgs_{lamk}"]["args"]
        diffs = {k: (sa.get(k), ba.get(k)) for k in set(sa) | set(ba) if k not in skip and sa.get(k) != ba.get(k)}
        print(f"<!-- args diff {lamk}: {diffs} -->")
    stats = {k: fleet_stats(v["records"], int(v["args"]["steps"])) for k, v in fleets.items()}
    pairs = {
        "b0": paired(fleets["spsa_b0"]["records"], fleets["bfgs_b0"]["records"]),
        "lam2": paired(fleets["spsa_lam2"]["records"], fleets["bfgs_lam2"]["records"]),
    }
    inst = {k: per_instance(v["records"]) for k, v in fleets.items()}
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps(
                {
                    "stats": {k: {**v, "status": [[list(s), c] for s, c in v["status"].items()]} for k, v in stats.items()},
                    "paired": pairs,
                    "per_instance": {k: {h: list(v) for h, v in d.items()} for k, d in inst.items()},
                },
                indent=2,
            )
        )
    cols = [("spsa_b0", "SPSA λ=0"), ("bfgs_b0", "BFGS λ=0"), ("spsa_lam2", "SPSA λ=2"), ("bfgs_lam2", "BFGS λ=2")]
    rows = [
        ("trials", "n", "{:d}"),
        ("success rate", "success", "{:.3f}"),
        ("mean p(GS)", "mean_p", "{:.4f}"),
        ("median p(GS)", "median_p", "{:.4f}"),
        ("mean abs(β)", "mean_b", "{:.3f}"),
        ("mean trial-max abs(β)", "mean_max_b", "{:.3f}"),
        ("frac of β with abs(β)>2.1", "frac_over", "{:.3f}"),
        ("frac of trials with any abs(β)>2.1", "trial_any_over", "{:.3f}"),
        ("mean nfev / trial", "mean_nfev", "{:.0f}"),
        ("median nfev / trial", "median_nfev", "{:.0f}"),
        ("mean nit", "mean_nit", "{:.1f}"),
        ("frac stopped before 200 it", "frac_early", "{:.3f}"),
        ("mean wall s / trial", "mean_wall", "{:.2f}"),
        ("total wall min (fleet)", "total_wall_min", "{:.1f}"),
    ]
    print("| metric | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, k, fmt in rows:
        print(f"| {label} | " + " | ".join(fmt.format(stats[c][k]) for c, _ in cols) + " |")
    print()
    for c, name in cols:
        if stats[c]["status"]:
            print(f"{name} status: " + "; ".join(f"{s[0]} '{s[1]}': {n}" for s, n in stats[c]["status"].most_common()))
    print()
    print("| paired (BFGS vs SPSA) | λ=0 | λ=2 |")
    print("|---|---|---|")
    for label, k, fmt in [
        ("matched trials", "n", "{:d}"),
        ("unmatched", "unmatched", "{:d}"),
        ("BFGS higher p(GS)", "win", "{:.3f}"),
        ("tie (abs Δ ≤ 1e-6)", "tie", "{:.3f}"),
        ("BFGS lower p(GS)", "lose", "{:.3f}"),
        ("mean Δp(GS) (BFGS−SPSA)", "mean_dp", "{:+.4f}"),
        ("median Δp(GS)", "median_dp", "{:+.4f}"),
        ("both succeed", "both", "{:.3f}"),
        ("only BFGS succeeds", "bfgs_only", "{:.3f}"),
        ("only SPSA succeeds", "spsa_only", "{:.3f}"),
        ("neither succeeds", "neither", "{:.3f}"),
    ]:
        print(f"| {label} | {fmt.format(pairs['b0'][k])} | {fmt.format(pairs['lam2'][k])} |")
    print()
    hams = sorted(inst["spsa_b0"])
    print("| instance | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for h in hams:
        print(f"| {h} | " + " | ".join("{}/{}".format(*inst[c].get(h, (0, 0))) for c, _ in cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
