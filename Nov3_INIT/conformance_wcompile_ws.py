"""Workspace W_BinH compile freeze. No SPSA on the isometry. No SK."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import iterations_equal4
from Nov3_INIT.wcompile import COMPILER_SEED, K_PAIRS, N_PARAMS, Q_COMPILE, x0_identity
from Nov3_INIT.wcompile_ws import (
    NF_FIT,
    NF_REPLAY,
    SEED_NF_TAG,
    metrics,
    workspace_map,
)
from Nov9_SK_SPSA.experiments.FT.sim import displacement_matrix

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WCOMP_WS.md"
OUT = ROOT / "results" / "wcompile_ws" / "CONFORMANCE.json"
SK = 2818048


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def main() -> None:
    x0 = x0_identity()
    m = metrics(x0, NF_FIT)
    part = m["partition"]
    # aligned D(1.8)^3 on |7> at fit workspace
    D = displacement_matrix(1.8, NF_FIT, "restricted")
    v = np.zeros(NF_FIT)
    v[7] = 1.0
    for _ in range(3):
        v = D @ v
    p_lost_7 = float(1.0 - np.vdot(v, v).real)
    v16 = workspace_map(x0, 16)
    v96 = workspace_map(x0, NF_FIT)
    payload = {
        "ok": bool(
            abs(m["L"] - 2.0) < 1e-12
            and abs(m["F_avg"] - 0.125) < 1e-12
            and abs(m["F_hs"]) < 1e-18
            and part["mean_p_code"] > 1.0 - 1e-12
            and abs(part["mean_p_lost"]) < 1e-12
            and abs(part["partition_sum_mean"] - 1.0) < 1e-12
            and p_lost_7 < 1e-8
            and NF_FIT == 96
            and NF_REPLAY == 128
            and SEED_NF_TAG == 16
            and COMPILER_SEED == 33001
            and N_PARAMS == 27
            and K_PAIRS == 3
            and Q_COMPILE == 2048
            and iterations_equal4(Q_COMPILE) == 250
            and v16.shape[0] == 16
            and v96.shape[0] == 96
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "sk_ledger": SK,
        "compiler_Q": Q_COMPILE,
        "T": iterations_equal4(Q_COMPILE),
        "nf_fit": NF_FIT,
        "nf_replay": NF_REPLAY,
        "seed_nf_tag": SEED_NF_TAG,
        "identity_L": m["L"],
        "identity_F_avg": m["F_avg"],
        "identity_F_hs": m["F_hs"],
        "identity_partition": part,
        "preflight_aligned_D18_cubed_ket7_p_lost": p_lost_7,
        "no_per_gate_P16": True,
        "selector": "coherent L after one global phase; identity is start-0 init incumbent",
        "no_spsa_in_this_module": True,
        "no_sk_hamiltonian": True,
        "label_rule": "WCOMP_WS_CANDIDATE iff L<=0.02, F_HS>=0.99, mean p_code>=0.99, p_lost<=1e-8, replay overlap>=0.999 and |dL|<=1e-3; p_lost miss -> INCONCLUSIVE; else FAIL_BUDGET",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
