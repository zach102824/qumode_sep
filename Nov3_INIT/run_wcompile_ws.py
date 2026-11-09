"""SPSA W_BinH compile on workspace 96; frozen replay at 128. Compiler evals only."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import Ledger, iterations_equal4, spsa_block
from Nov3_INIT.run_v1 import _dump
from Nov3_INIT.wcompile import Q_COMPILE, sigmas, x0_spread
from Nov3_INIT.wcompile_ws import (
    NF_FIT,
    NF_REPLAY,
    column_overlap,
    compiler_rng,
    label_of,
    metrics,
    pulse_dict,
    workspace_map,
    x0_identity,
)

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WCOMP_WS.md"
RES = ROOT / "results" / "wcompile_ws"
CONF = RES / "CONFORMANCE.json"
SK = 2818048
HIST_COMPILER = 2048


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("wcompile_ws conformance does not match preregistration")
    if int(doc.get("sk_ledger", -1)) != SK:
        raise RuntimeError("SK ledger marker mismatch")
    return doc


def fit() -> None:
    _conf()
    sig = sigmas()
    t_iter = iterations_equal4(Q_COMPILE)
    ledger = Ledger()
    tails = []
    best = None
    ident = metrics(x0_identity(), NF_FIT)
    t0 = time.perf_counter()

    def propagate(x):
        met = metrics(x, NF_FIT)
        if not met["ok"]:
            raise RuntimeError("non-finite compile L")
        return float(met["L"]), met

    for start_id in range(4):
        rng_i = None if start_id == 0 else compiler_rng(start_id, 2)
        rng_s = compiler_rng(start_id, 0)
        x0 = x0_spread(start_id, rng_i)
        block = spsa_block(propagate, x0, sig, t_iter, rng_s, ledger)
        tails.append({
            "start_id": start_id,
            "L": float(block["E"]),
            "which": block["which"],
            "L_init": float(block["e_init"]),
        })
        if best is None or block["E"] < best["E"] or (block["E"] == best["E"] and start_id < best["start_id"]):
            best = dict(block)
            best["start_id"] = start_id
    if ledger.total() != Q_COMPILE:
        raise RuntimeError(f"compiler ledger {ledger.as_dict()} != {Q_COMPILE}")
    x = np.asarray(best["x"], dtype=np.float64)
    m96 = metrics(x, NF_FIT)
    m128 = metrics(x, NF_REPLAY)
    ov = column_overlap(workspace_map(x, NF_FIT), workspace_map(x, NF_REPLAY))
    lab = label_of(m96, m128, ov)
    payload = {
        "ok": True,
        "version": "N3INIT_wcompile_ws",
        "no_sk_fits": True,
        "sk_queries_added": 0,
        "historical_sk_queries": SK,
        "total_sk_queries": SK,
        "historical_compiler_evals_pr109": HIST_COMPILER,
        "compiler_queries_this_trial": ledger.as_dict(),
        "compiler_Q": Q_COMPILE,
        "label": lab,
        "nf_fit": NF_FIT,
        "nf_replay": NF_REPLAY,
        "identity_incumbent": ident,
        "selected_beats_identity_on_L": bool(m96["L"] < ident["L"] - 1e-15),
        "start_id": int(best["start_id"]),
        "which": best["which"],
        "n_iter": t_iter,
        "tails": tails,
        "metrics_nf96": m96,
        "metrics_nf128_frozen_replay": m128,
        "replay_overlap": ov,
        "pulse": pulse_dict(x),
        "seconds": time.perf_counter() - t0,
        "prereg_sha256": prereg_sha(),
        "selector": "coherent L; not F_avg",
        "not_a_hardware_claim": True,
        "pr109_is_history_projection_exploitable": True,
        "no_gr_patch": True,
    }
    _dump(RES / "summary.json", payload)
    _dump(RES / "pulse.json", payload["pulse"])
    p = m96["partition"]
    pr = m128["partition"]
    lines = [
        "# Corrected-workspace W_BinH compile K=3 Q=2048",
        "",
        "Common workspace 96, frozen replay 128. Real α, SNAP pad 0. Coherent L selector. Identity incumbent measured. SK ledger 2818048. Historical PR109 2048 compiler evals recorded separately. Not a hardware claim.",
        "",
        f"**{lab}.**",
        "",
        f"Selected start {best['start_id']} ({best['which']}) L={m96['L']:.4f} vs identity L={ident['L']:.4f} (beats identity {payload['selected_beats_identity_on_L']}). Compiler queries {ledger.total()}. Wall {payload['seconds']:.1f}s. SK queries 0.",
        "",
        "## NF=96 fit",
        "",
        f"L {m96['L']:.4f} F_HS {m96['F_hs']:.4f} F_avg {m96['F_avg']:.4f} F_worst {m96['F_worst']:.4f} (F_avg not selected).",
        f"p_code {p['mean_p_code']:.4f} p_pad {p['mean_p_pad_retained']:.4f} p_lost {p['mean_p_lost']:.3e} (sum {p['partition_sum_mean']:.6f}).",
        f"plus_fid {m96['plus_fid']:.4f} random_fid {m96['random_fid']:.4f}.",
        f"α = {pulse_dict(x)['alpha_real']}.",
        "",
        "## NF=128 frozen replay",
        "",
        f"L {m128['L']:.4f} F_HS {m128['F_hs']:.4f} ΔL {abs(m128['L']-m96['L']):.3e}.",
        f"p_code {pr['mean_p_code']:.4f} p_pad {pr['mean_p_pad_retained']:.4f} p_lost {pr['mean_p_lost']:.3e}.",
        f"full-column overlap {ov['mean_col_overlap_sq']:.6f} (worst {ov['worst_col_overlap_sq']:.6f}) code-block fro {ov['code_block_fro']:.3e}.",
        "",
        "PR109 remains history under repeated P_16 D P_16. This trial does not claim impossibility if FAIL_BUDGET. No extra K/α/SNAP after a miss.",
    ]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("fit",))
    args = parser.parse_args()
    _conf()
    if args.phase == "fit":
        fit()


if __name__ == "__main__":
    main()
