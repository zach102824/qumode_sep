#!/usr/bin/env python3
"""Round 0: exact QAOA dual identity gate for Nov_3rd Hamming-shell.

Loads fair paper-n6 QAOA (β,γ), maps to warm dual:
  γ_warm = γ_QAOA
  φ_k = -β · k   (k=0..3)

Eval on scale1000 B=1000 CORE n∈{6,9,12,15}. PASS iff |R-1|≤0.005 for all CORE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, log, shell_phi_from_betas
from Nov_3rd.ensure_qaoa import ensure_qaoa_paper_n6
from Nov_3rd.eval_scale import eval_n

OUT = S.OUT / "beat_qaoa_12h" / "round_0"
TOL = 0.005


def verdict(R: float) -> str:
    if abs(R - 1.0) <= TOL:
        return "PASS"
    if R <= 0.98:
        return "BEAT_UNEXPECTED"
    if R <= 1.02:
        return "TIE_OUTSIDE_TOL"
    return "FAIL_UNDER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True, choices=[1, 2, 16])
    ap.add_argument("--ns", type=str, default="6,9,12,15")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ensure_qaoa", action="store_true",
                    help="train/eval fair paper-n6 QAOA if missing")
    ap.add_argument("--force_qaoa", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    L = args.L
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    if args.ensure_qaoa or args.force_qaoa or not S.qaoa_angles_path(L).exists():
        log(f"R0 L{L}: ensure fair paper-n6 QAOA")
        ensure_qaoa_paper_n6(L, device=device, force=args.force_qaoa, ns=ns)

    betas, gammas = load_qaoa(L)
    phi = shell_phi_from_betas(betas)
    gamma = gammas.reshape(L).contiguous()
    qref = load_qaoa_scale_ref(L)

    log(f"R0 dual L{L}: φ_k=-βk γ=γ_QAOA |β|={betas.abs().mean():.4f} |γ|={gammas.abs().mean():.4f}")
    rows = {}
    all_pass = True
    for n in ns:
        if not (S.SCALE1000_DIR / f"data_n{n}.npz").exists():
            log(f"skip n{n}: no pack")
            continue
        m = eval_n(n, L, gamma, phi, device, max_count=args.B)
        q = qref.get(n) or {}
        qmed = float(q.get("Med", float("nan")))
        qb90 = float(q.get("B90", float("nan")))
        med = float(m["median_inv_p"])
        b90 = float(m["B90_inv_p"])
        R = med / qmed if qmed == qmed and qmed > 0 else float("nan")
        v = verdict(R) if R == R else "NO_QAOA_REF"
        if v != "PASS":
            all_pass = False
        rows[n] = {
            "Med": med,
            "B90": b90,
            "QAOA_Med": qmed,
            "QAOA_B90": qb90,
            "R": R,
            "verdict": v,
            "mean_leak": float(m["mean_leak"]),
            "B": int(m["B"]),
            "wall_s": float(m["wall_s"]),
        }
        log(f"R0 L{L} n{n}: Med={med:.4f} QAOA={qmed:.4f} R={R:.6f} {v} leak={m['mean_leak']:.2e}")

    payload = {
        "L": L,
        "map": "phi_k=-beta*k, gamma=gamma_QAOA",
        "tol": TOL,
        "ns": ns,
        "B": args.B,
        "qaoa_angles": str(S.qaoa_angles_path(L)),
        "pass": all_pass and len(rows) == len(ns),
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "tz": "Asia/Shanghai",
    }
    out_json = OUT / f"dual_identity_L{L}.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# Round 0 dual identity L={L}",
        "",
        f"_Finished {payload['finished']} (Asia/Shanghai)_",
        "",
        f"**Map:** φ_k = −β·k , γ = γ_QAOA",
        f"**Tol:** |R−1| ≤ {TOL}",
        f"**Overall:** **{'PASS' if payload['pass'] else 'FAIL'}**",
        "",
        "| n | QAOA Med | warm Med | R | leak | verdict |",
        "|---|----------|----------|---|------|---------|",
    ]
    for n in ns:
        r = rows.get(n)
        if not r:
            continue
        lines.append(
            f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.6f} | "
            f"{r['mean_leak']:.2e} | {r['verdict']} |"
        )
    lines.append("")
    (OUT / f"dual_identity_L{L}.md").write_text("\n".join(lines) + "\n")
    log(f"R0 L{L} wrote {out_json} pass={payload['pass']}")
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
