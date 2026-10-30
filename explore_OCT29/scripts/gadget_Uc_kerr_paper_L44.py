#!/usr/bin/env python3
"""NON-DEPTH #2: Kerr budget refresh at L44 + paper mega-table finalize."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.packing import K_BITS, NU, n_cavities  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_grape_library import (  # noqa: E402
    U_from_C,
    process_fidelity,
    projector_Z,
    snap_from_phases,
    enumerate_library,
)
from explore_OCT29.scripts.joint_snap_pulse_budget import (  # noqa: E402
    K_AB,
    K_AB_HZ,
    clause_span_census,
    joint_snap_phases,
    pulse_account,
    support_fock,
)

OUT = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "kerr_paper_L44"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
ARMC_L44 = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "armC_L44_pad" / "results.json"
PAPER = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "PAPER_TABLE.md"

CHAMP_NUMS = {
    9: (1.051, 1.473),
    12: (1.113, 2.263),
    15: (1.233, 4.243),
    18: (1.409, 9.645),
}
ARMC_L16 = {
    9: (1.441, 3.314),
    12: (1.991, 6.252),
    15: (2.921, 10.959),
    18: (4.573, 27.810),
}
QAOA_L16 = {
    9: (2.242, 5.534),
    12: (3.263, 10.868),
    15: (5.099, 18.647),
    18: (7.894, 34.070),
}
# WalkSAT R=50 from PAPER_TABLE
WALK = {
    9: (12.5, 40.1, 51.0, 161.4),
    12: (26.5, 65.9, 107.0, 264.6),
    15: (43.8, 128.1, 176.0, 513.6),
    18: (71.5, 203.6, 287.0, 815.4),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[kerr_paper]", *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g = st["gamma"].detach().cpu().numpy().reshape(-1)
    L = int(st.get("L", g.shape[0]))
    assert L == 44, f"expected L44 champ, got L={L}"

    # fold mod 2π for Kerr time estimate
    g_abs = np.abs(g)
    g_fold = np.minimum(g_abs % (2 * math.pi), 2 * math.pi - (g_abs % (2 * math.pi)))
    # prefer complement if >π → fold to [0,π]
    g_pi = np.minimum(g_fold, 2 * math.pi - g_fold)
    g_pi = np.minimum(g_pi, math.pi)  # already ≤π after min with 2π-x when x in [0,2π)

    n_typ = 3.5  # mid Fock for 3-bit cavity (0..7) rough
    def kerr_us(gamma):
        return float(abs(gamma) / (K_AB * n_typ * n_typ) * 1e6)

    gamma_stats = {
        "L": L,
        "mean_abs": float(np.mean(g_abs)),
        "max_abs": float(np.max(g_abs)),
        "p90_abs": float(np.quantile(g_abs, 0.9)),
        "mean_fold_pi": float(np.mean(g_pi)),
        "max_fold_pi": float(np.max(g_pi)),
        "kerr_us_mean_raw": kerr_us(np.mean(g_abs)),
        "kerr_us_max_raw": kerr_us(np.max(g_abs)),
        "kerr_us_mean_fold": kerr_us(np.mean(g_pi)),
        "kerr_us_max_fold": kerr_us(np.max(g_pi)),
        "kerr_us_layer_sum_raw": float(np.sum([kerr_us(x) for x in g_abs])),
        "kerr_us_layer_sum_fold": float(np.sum([kerr_us(x) for x in g_pi])),
        "K_AB_kHz": K_AB_HZ / 1e3,
        "n_typ": n_typ,
    }

    # pulse accounts at γ=1 and at champ mean |γ|
    lib = enumerate_library()
    def C_of(e):
        return projector_Z(e["bits_signs"])
    Ca13 = C_of(lib["k1"][0])
    Cb13 = C_of(lib["k3"][0])
    Ca22 = C_of(lib["k2"][0])
    Cb22 = C_of(lib["k2"][1] if len(lib["k2"]) > 1 else lib["k2"][0])
    accounts = {
        "(1,3)_g1": pulse_account(Ca13, Cb13, 1.0),
        "(2,2)_g1": pulse_account(Ca22, Cb22, 1.0),
        f"(1,3)_gmean{gamma_stats['mean_abs']:.2f}": pulse_account(
            Ca13, Cb13, gamma_stats["mean_abs"]
        ),
        f"(1,3)_gfold{gamma_stats['mean_fold_pi']:.2f}": pulse_account(
            Ca13, Cb13, gamma_stats["mean_fold_pi"]
        ),
    }

    census = {}
    for n in (9, 12, 15, 18):
        log(f"census n{n}")
        c = clause_span_census(n)
        c["L44_total_abstract"] = float(c["pulses_per_layer_est"]["total_abstract"] * L)
        census[n] = c

    armC_L44 = None
    if ARMC_L44.exists():
        ad = json.loads(ARMC_L44.read_text())
        ladder = ad.get("ladder") or {}
        if len(ladder) >= 4:
            armC_L44 = {
                int(k): (float(v["median_inv_p"]), float(v["B90_inv_p"]))
                for k, v in ladder.items()
            }
            log("loaded ArmC L44", armC_L44)
        else:
            log(f"ArmC L44 partial ({len(ladder)} ns) — paper row pending")
    else:
        log("ArmC L44 results not ready — paper row pending")

    ns = [9, 12, 15, 18]
    b_champ = fit_b(ns, [CHAMP_NUMS[n][0] for n in ns])
    b_champ90 = fit_b(ns, [CHAMP_NUMS[n][1] for n in ns])
    b_arm16 = fit_b(ns, [ARMC_L16[n][0] for n in ns])
    b_qaoa = fit_b(ns, [QAOA_L16[n][0] for n in ns])
    b_walk = fit_b(ns, [WALK[n][0] for n in ns])
    b_arm44 = None
    if armC_L44 and all(n in armC_L44 for n in ns):
        b_arm44 = fit_b(ns, [armC_L44[n][0] for n in ns])

    results = {
        "gamma_stats": gamma_stats,
        "pulse_accounts": accounts,
        "census": census,
        "champ_L44": {str(k): list(v) for k, v in CHAMP_NUMS.items()},
        "armC_L16": {str(k): list(v) for k, v in ARMC_L16.items()},
        "armC_L44": {str(k): list(v) for k, v in armC_L44.items()} if armC_L44 else None,
        "b_fits": {
            "champ_Med": float(b_champ["b"]),
            "champ_B90": float(b_champ90["b"]),
            "armC_L16_Med": float(b_arm16["b"]),
            "qaoa_Med": float(b_qaoa["b"]),
            "walk_Med_flips": float(b_walk["b"]),
            "armC_L44_Med": float(b_arm44["b"]) if b_arm44 else None,
        },
        "honesty": "Uc=exp(-iγ Hc); Kerr refresh at L44; paper mega-table",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    def js(o):
        if isinstance(o, dict):
            return {str(k): js(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [js(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    (OUT / "results.json").write_text(json.dumps(js(results), indent=2))

    # RESULTS.md
    lines = [
        "# Kerr budget refresh @ L44 + paper mega-table",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. No L48+ pad. Blind banned.",
        "",
        "## L44 champ γ → Kerr",
        "",
        f"| stat | raw |γ| | fold→[0,π] |",
        f"|------|---------|-------------|",
        f"| mean | {gamma_stats['mean_abs']:.3f} | {gamma_stats['mean_fold_pi']:.3f} |",
        f"| max | {gamma_stats['max_abs']:.3f} | {gamma_stats['max_fold_pi']:.3f} |",
        f"| Kerr µs/layer (mean) | {gamma_stats['kerr_us_mean_raw']:.2f} | {gamma_stats['kerr_us_mean_fold']:.2f} |",
        f"| Kerr µs sum over L={L} | {gamma_stats['kerr_us_layer_sum_raw']:.1f} | {gamma_stats['kerr_us_layer_sum_fold']:.1f} |",
        "",
        f"K_AB≈{K_AB_HZ/1e3:.0f} kHz, n_typ={n_typ}. Fold assumes γ mod 2π + complement-mask ≤π "
        "(identity check already greenlit; not re-verified here).",
        "",
        "## Pulse accounts",
        "",
        "| split | joint Fid | local Fid | nonzero θ | Kerr T (µs) |",
        "|-------|-----------|-----------|-----------|-------------|",
    ]
    for name, blk in accounts.items():
        lines.append(
            f"| {name} | {blk['fid_joint_snap']:.6f} | {blk['fid_best_local_sep']:.4f} | "
            f"{blk['joint_snap_nonzero_phases']} | {blk['cross_Kerr_T_us']:.2f} |"
        )
    lines += [
        "",
        "## Clause-span census → L44 abstract pulses/inst",
        "",
        "| n | span2 | span3 | span4 | L44 abstract |",
        "|---|-------|-------|-------|--------------|",
    ]
    for n in ns:
        c = census[n]
        fs = c["frac_span"]
        lines.append(
            f"| {n} | {fs['2']:.3f} | {fs['3']:.3f} | {fs['4']:.3f} | "
            f"{c['L44_total_abstract']:.1f} |"
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")

    # Paper mega-table
    arm44_hdr = "ArmC L44 pad" if armC_L44 else "ArmC L44 *(pending)*"
    pl = [
        "# Paper mega-table — scale1000 B=1000 (L44 finalize)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = flips/evals (different unit).",
        "",
        "Honesty: champ = **L44 pad** projector-Uc ≡ `exp(-iγ Hc)`. L48 STOP. No blind-cost. "
        "ArmC L44 = interp of L16 shell (fair same-L densified baseline).",
        "",
        f"| n | **champ L44** | ArmC L16 | {arm44_hdr} | QAOA L16 | WalkSAT flips | WalkSAT evals |",
        "|---|---------------|----------|----------------|----------|---------------|---------------|",
    ]
    for n in ns:
        c = CHAMP_NUMS[n]
        a16 = ARMC_L16[n]
        q = QAOA_L16[n]
        w = WALK[n]
        if armC_L44 and n in armC_L44:
            a44s = f"{armC_L44[n][0]:.3f}/{armC_L44[n][1]:.3f}"
        else:
            a44s = "—"
        pl.append(
            f"| {n} | **{c[0]:.3f}/{c[1]:.3f}** | {a16[0]:.3f}/{a16[1]:.3f} | {a44s} | "
            f"{q[0]:.3f}/{q[1]:.3f} | {w[0]:.1f}/{w[1]:.1f} | {w[2]:.1f}/{w[3]:.1f} |"
        )
    pl += [
        "",
        "## b fits (9→18)",
        "",
        f"- champ L44 b_Med≈**{b_champ['b']:.4f}**, b_B90≈**{b_champ90['b']:.4f}**",
        f"- ArmC L16 b_Med≈**{b_arm16['b']:.4f}**",
    ]
    if b_arm44:
        pl.append(f"- ArmC L44 pad b_Med≈**{b_arm44['b']:.4f}**")
    else:
        pl.append("- ArmC L44 pad b_Med=*(pending eval)*")
    pl += [
        f"- QAOA L16 b_Med≈**{b_qaoa['b']:.4f}**",
        f"- WalkSAT flips b_Med≈**{b_walk['b']:.4f}**",
        "",
        "## Kerr one-liner",
        "",
        f"L44 mean |γ|={gamma_stats['mean_abs']:.2f} → ~{gamma_stats['kerr_us_mean_raw']:.2f} µs/layer raw; "
        f"fold→[0,π] mean={gamma_stats['mean_fold_pi']:.2f} → ~{gamma_stats['kerr_us_mean_fold']:.2f} µs/layer; "
        f"sum_L fold≈{gamma_stats['kerr_us_layer_sum_fold']:.0f} µs (cross-Kerr only, excl. SNAP/ECD).",
        "",
    ]
    paper_path = OUT / "PAPER_TABLE.md"
    paper_path.write_text("\n".join(pl))
    # also refresh renames_Hc PAPER_TABLE
    PAPER.write_text("\n".join(pl))
    log("wrote", OUT / "RESULTS.md", paper_path, PAPER)


if __name__ == "__main__":
    main()
