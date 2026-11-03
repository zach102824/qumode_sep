#!/usr/bin/env python3
"""Debug override: warm-only vs polished L=1 on scale1000 (does NOT touch settings locks).

Goal: distinguish n6 free-train polish damage vs motif/dual identity bug.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, log, shell_phi_from_betas
from Nov_3rd.eval_scale import eval_ladder

TIE_HI = 1.02
BEAT_HI = 0.98


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (str, bool)) or x is None:
        return x
    return str(x)


def verdict(R: float | None) -> str:
    if R is None:
        return "NO_QAOA"
    if R <= BEAT_HI:
        return "BEAT"
    if R <= TIE_HI:
        return "TIE"
    return "UNDER"


def compare_rows(rows: dict, qref: dict) -> dict:
    per = {}
    for n, m in sorted(rows.items()):
        qmed = None
        if n in qref and qref[n].get("Med") is not None:
            qmed = float(qref[n]["Med"])
        R = None if qmed is None or qmed <= 0 else float(m["median_inv_p"] / qmed)
        per[int(n)] = {
            "Med": float(m["median_inv_p"]),
            "B90": float(m["B90_inv_p"]),
            "QAOA_Med": qmed,
            "R": R,
            "verdict": verdict(R),
            "B": int(m["B"]),
            "wall_s": float(m.get("wall_s", 0.0)),
        }
    under_any = any(p["verdict"] == "UNDER" for p in per.values() if p["R"] is not None)
    all_tie_or_better = all(
        p["verdict"] in ("BEAT", "TIE") for p in per.values() if p["R"] is not None
    )
    return {
        "per_n": per,
        "under_any": under_any,
        "all_R_le_1p02": all_tie_or_better and bool(per),
    }


def write_md(path: Path, blob: dict):
    warm = blob["warm"]
    pol = blob["polished"]
    qref = blob["qaoa_ref"]
    lines = [
        "# Nov_3rd debug — L=1 warm-only vs polished",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        "Motif: Nov1st-final X-sandwich, QAOA-warm shell SNAP, BinH b=3.",
        "Warm = γ=γ_QAOA, φ=shell_phi_from_betas(β) (NO free polish).",
        "Polished = params_best_L1.pt (n6×1000 LBFGS free γ+φ).",
        "",
        "## Params",
        "",
        f"- QAOA β={blob['params']['betas']}",
        f"- warm γ={blob['params']['warm_gamma']}",
        f"- warm φ={blob['params']['warm_phi']}",
        f"- polished γ={blob['params']['polished_gamma']}",
        f"- polished φ={blob['params']['polished_phi']}",
        f"- ‖Δφ‖={blob['params']['dphi_norm']:.4f}",
        f"- Δγ={blob['params']['dgamma']:.4f}",
        "",
        "## scale1000 Med / R vs QAOA L=1",
        "",
        "| n | QAOA Med | warm Med | warm R | warm | polished Med | polished R | polished |",
        "|---|----------|----------|--------|------|--------------|------------|----------|",
    ]
    ns = sorted({int(x) for x in list(warm["per_n"]) + list(pol["per_n"]) + list(qref)})
    def _get(d, n):
        if n in d: return d[n]
        if str(n) in d: return d[str(n)]
        return None
    for n in ns:
        qr = _get(qref, n) or {}
        q = qr.get("Med") if isinstance(qr, dict) else None
        w = _get(warm["per_n"], n)
        p = _get(pol["per_n"], n)
        qm = f"{q:.3f}" if q is not None else "—"
        wm = f"{w['Med']:.3f}" if w else "—"
        wr = f"{w['R']:.3f}×" if w and w["R"] is not None else "—"
        wv = w["verdict"] if w else "—"
        pm = f"{p['Med']:.3f}" if p else "—"
        pr = f"{p['R']:.3f}×" if p and p["R"] is not None else "—"
        pv = p["verdict"] if p else "—"
        lines.append(f"| {n} | {qm} | {wm} | {wr} | {wv} | {pm} | {pr} | {pv} |")
    lines += ["", "## Verdict", "", blob["verdict_sentence"], "", "## Next fix", "", blob["next_fix"], ""]
    if blob.get("probe"):
        lines += ["## Probe", "", blob["probe"].get("summary_md", ""), ""]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 9, 12, 15])
    ap.add_argument("--skip-polished", action="store_true")
    ap.add_argument("--skip-warm", action="store_true")
    args = ap.parse_args()

    L = 1
    device = torch.device("cpu")
    S.OUT.mkdir(parents=True, exist_ok=True)
    ns = tuple(args.ns)

    betas, gammas = load_qaoa(L)
    warm_phi = shell_phi_from_betas(betas)
    qref = load_qaoa_scale_ref(L)
    # n6 may be missing from QAOA results — leave blank

    polished_path = S.OUT / "params_best_L1.pt"
    st = torch.load(polished_path, map_location="cpu", weights_only=False)
    pol_g = st["gamma"].detach().float().reshape(-1)
    pol_p = st["phi"].detach().float()

    params = {
        "betas": betas.tolist(),
        "warm_gamma": gammas.tolist(),
        "warm_phi": warm_phi.tolist(),
        "polished_gamma": pol_g.tolist(),
        "polished_phi": pol_p.tolist(),
        "dphi_norm": float((pol_p - warm_phi).norm().item()),
        "dgamma": float((pol_g - gammas).norm().item()),
        "polished_seed": st.get("seed"),
        "polished_path": str(polished_path),
    }
    log(f"debug L=1 ns={list(ns)} dphi={params['dphi_norm']:.4f} dgamma={params['dgamma']:.4f}")

    warm_cmp = {"per_n": {}, "under_any": False, "all_R_le_1p02": False}
    pol_cmp = {"per_n": {}, "under_any": False, "all_R_le_1p02": False}
    warm_rows = {}
    pol_rows = {}

    if not args.skip_warm:
        log("=== WARM-ONLY eval ===")
        warm_rows = eval_ladder(L, gammas, warm_phi, device, ns=ns)
        warm_cmp = compare_rows(warm_rows, qref)
        for n, p in warm_cmp["per_n"].items():
            r = "NA" if p["R"] is None else f"{p['R']:.3f}×"
            log(f"warm n{n}: Med={p['Med']:.3f} R={r} {p['verdict']}")

    if not args.skip_polished:
        log("=== POLISHED eval ===")
        pol_rows = eval_ladder(L, pol_g, pol_p, device, ns=ns)
        pol_cmp = compare_rows(pol_rows, qref)
        for n, p in pol_cmp["per_n"].items():
            r = "NA" if p["R"] is None else f"{p['R']:.3f}×"
            log(f"polished n{n}: Med={p['Med']:.3f} R={r} {p['verdict']}")

    # Root-cause on ns that have QAOA refs (typically 9+)
    warm_with_q = [p for p in warm_cmp["per_n"].values() if p["R"] is not None]
    if warm_with_q and all(p["R"] <= TIE_HI for p in warm_with_q):
        verdict_sentence = (
            "Root cause: n6×1000 free LBFGS polish (γ+φ) drifted off the QAOA shell "
            "and hurt transfer — warm-only Motif ≈ QAOA (R≤1.02 on all QAOA-ref n)."
        )
        next_fix = (
            "Next: freeze φ at warm shell and only train γ on n6 (probe A), "
            "or retrain with sealed train_ns=(9,12) like E1 (probe B)."
        )
        root = "n6_free_polish"
    elif warm_with_q and any(p["R"] > TIE_HI for p in warm_with_q):
        verdict_sentence = (
            "Root cause: warm-only already UNDER QAOA — motif/dual identity bug in "
            "Nov_3rd circuit (not just polish)."
        )
        next_fix = (
            "Next: audit Nov_3rd forward vs E0 Arm-C QAOA dual (W D W / SNAP shell / final W)."
        )
        root = "motif_or_dual_bug"
    else:
        verdict_sentence = "Incomplete: no QAOA refs for warm comparison."
        next_fix = "Re-run with n that have QAOA L=1 results."
        root = "incomplete"

    blob = {
        "meta": {
            "L": L,
            "ns": list(ns),
            "recipe": S.RECIPE_NAME,
            "motif": S.MOTIF,
            "asia_shanghai": time.strftime("%Y-%m-%d %H:%M %z"),
            "root_cause_tag": root,
        },
        "params": params,
        "qaoa_ref": {str(k): v for k, v in qref.items()},
        "warm": {"eval": warm_rows, **warm_cmp},
        "polished": {"eval": pol_rows, **pol_cmp},
        "verdict_sentence": verdict_sentence,
        "next_fix": next_fix,
        "probe": None,
    }

    out_json = S.OUT / "debug_L1_warm_vs_polished.json"
    out_md = S.OUT / "debug_L1_warm_vs_polished.md"
    out_json.write_text(json.dumps(_jsonable(blob), indent=2) + "\n")
    write_md(out_md, blob)
    log(f"wrote {out_md}")
    log(f"VERDICT: {verdict_sentence}")
    log(f"NEXT: {next_fix}")
    print(json.dumps({"root": root, "warm_all_le_1p02": warm_cmp.get("all_R_le_1p02"), "out_md": str(out_md)}))


if __name__ == "__main__":
    main()
