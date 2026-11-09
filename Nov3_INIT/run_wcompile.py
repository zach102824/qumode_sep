"""SPSA compile of W_BinH at NF=16, frozen replay at NF=32. Compiler evals only."""
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
from Nov3_INIT.wcompile import (
    COMPILER_SEED,
    K_PAIRS,
    NF_OPT,
    NF_REPLAY,
    Q_COMPILE,
    candidate_gates,
    metrics,
    pulse_dict,
    sigmas,
    x0_spread,
)

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WCOMP.md"
RES = ROOT / "results" / "wcompile"
CONF = RES / "CONFORMANCE.json"
SK_LEDGER = 2818048


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("wcompile conformance does not match the current preregistration")
    if int(doc.get("sk_ledger_untouched", -1)) != SK_LEDGER:
        raise RuntimeError("SK ledger marker mismatch")
    return doc


def _compiler_rng(start_id: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(COMPILER_SEED), int(NF_OPT), int(K_PAIRS), int(start_id), int(stream)])
    )


def fit() -> None:
    _conf()
    sig = sigmas()
    t_iter = iterations_equal4(Q_COMPILE)
    ledger = Ledger()
    tails = []
    best = None
    t0 = time.perf_counter()

    def propagate(x):
        met = metrics(x, NF_OPT)
        if not met["ok"]:
            raise RuntimeError("non-finite compile loss")
        return float(met["L"]), met

    for start_id in range(4):
        rng_i = None if start_id == 0 else _compiler_rng(start_id, 2)
        rng_s = _compiler_rng(start_id, 0)
        x0 = x0_spread(start_id, rng_i)
        block = spsa_block(propagate, x0, sig, t_iter, rng_s, ledger)
        tails.append({
            "start_id": start_id,
            "L": float(block["E"]),
            "which": block["which"],
            "e_init": float(block["e_init"]),
        })
        if best is None or block["E"] < best["E"] or (block["E"] == best["E"] and start_id < best["start_id"]):
            best = dict(block)
            best["start_id"] = start_id
    if ledger.total() != Q_COMPILE:
        raise RuntimeError(f"compiler ledger {ledger.as_dict()} != Q {Q_COMPILE}")
    x = np.asarray(best["x"], dtype=np.float64)
    m16 = metrics(x, NF_OPT)
    m32 = metrics(x, NF_REPLAY)
    gates = candidate_gates(m16, m32)
    payload = {
        "ok": True,
        "version": "N3INIT_wcompile",
        "no_sk_fits": True,
        "sk_queries_added": 0,
        "historical_sk_queries": SK_LEDGER,
        "total_sk_queries": SK_LEDGER,
        "compiler_queries": ledger.as_dict(),
        "compiler_Q": Q_COMPILE,
        "label": gates["label"],
        "gates": gates,
        "nf_opt": NF_OPT,
        "nf_replay": NF_REPLAY,
        "start_id": int(best["start_id"]),
        "which": best["which"],
        "n_iter": t_iter,
        "tails": tails,
        "metrics_nf16": m16,
        "metrics_nf32_frozen_replay": m32,
        "pulse": pulse_dict(x),
        "seconds": time.perf_counter() - t0,
        "prereg_sha256": prereg_sha(),
        "not_a_hardware_claim": True,
        "not_flat_prep": True,
        "not_ux_compile": True,
        "real_alpha_only": True,
        "complex_alpha_not_added": True,
        "pad_snap_not_added": True,
        "no_epsilon_sweep": True,
        "no_gr_patch": True,
    }
    _dump(RES / "summary.json", payload)
    _dump(RES / "pulse.json", payload["pulse"])
    lines = [
        "# W_BinH SNAP + true-D compile feasibility",
        "",
        "Real α, K=3 (D then SNAP), SNAP pad 0, true D from NF+32. SPSA Q=2048 compiler evals. SK ledger unchanged at 2818048. Not flat prep. Not U_X. Not a hardware claim.",
        "",
        f"**{gates['label']}.**",
        "",
        f"Selected start {best['start_id']} ({best['which']}). Compiler queries {ledger.total()}. Wall {payload['seconds']:.1f}s.",
        "",
        "## NF=16 (optimized)",
        "",
        f"F_avg {m16['F_avg']:.4f} F_worst {m16['F_worst']:.4f} F_HS {m16['F_hs']:.4f} L {m16['L']:.4f}.",
        f"p_leak mean {m16['p_leak_mean']:.3e} worst {m16['p_leak_worst']:.3e}. plus_fid {m16['plus_fid']:.4f} random_fid {m16['random_fid']:.4f}.",
        f"mean column norm {float(np.mean(m16['col_norm'])):.6f} (no renormalization).",
        "",
        "## NF=32 frozen replay (no reopt)",
        "",
        f"F_avg {m32['F_avg']:.4f} F_worst {m32['F_worst']:.4f} F_HS {m32['F_hs']:.4f}. ΔF_avg {gates['replay_dF_avg']:.3e}. replay_ok {gates['replay_ok']}.",
        f"p_leak mean {m32['p_leak_mean']:.3e} worst {m32['p_leak_worst']:.3e}.",
        "",
        "Global phase: one overall arg(tr(T† V)). Per-column phases are not free. Complex α and pad SNAP were not added.",
        "",
        "Fit workers 1, BLAS 1. Drop-in bars: F_avg≥0.99, F_worst≥0.95, mean p_leak≤0.01, F_HS≥0.99, replay |ΔF_avg|≤1e-3.",
        "",
        "## Remaining limitations",
        "",
        "Complex α (D7 not now; used by compile_flat_prep for one column only) was not added. SNAP pad phases stay 0. Depth was not swept. Cropped D8 was not used. A numerical compile is not a hardware claim. SK energy fits were not launched.",
    ]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("fit",))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("wcompile preregistration missing")
    _conf()
    if args.phase == "fit":
        fit()


if __name__ == "__main__":
    main()
