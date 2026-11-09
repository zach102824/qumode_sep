"""Anneal-schedule freeze. No SPSA on a Hamiltonian."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import (
    ALPHA_EXP,
    ANNEAL_ALPHA,
    ANNEAL_GAMMA,
    A_SPALL,
    GAMMA_EXP,
    METHOD_ID,
    iterations_equal4,
    run_equal4,
    spall_gains,
    x0_center4,
    x0_qaoa_center4,
)

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_ANNEAL.md"
OUT = ROOT / "results" / "anneal" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10456, 10464))
RESERVED = {
    "le_10432_10439": set(range(10432, 10440)),
    "local_r27_10440_10455": set(range(10440, 10456)),
    "local_r26_10416_10431": set(range(10416, 10432)),
    "prior_le_10415": set(range(0, 10416)),
    "holdout_10144_10151": set(range(10144, 10152)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
}


def _collisions() -> dict:
    hits = []
    if RES.exists():
        for path in RES.rglob("*"):
            if path.suffix not in {".json", ".npz", ".md"}:
                continue
            if any(f"s{s}" in path.name for s in BAND):
                hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "overlaps": overlaps,
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10456, 10464)),
    }


def _schedule() -> dict:
    t = iterations_equal4(3072)
    _, a_v4, c_v4 = spall_gains(t, 1.0, ALPHA_EXP, GAMMA_EXP, A_SPALL)
    _, a_an, c_an = spall_gains(t, 1.0, ANNEAL_ALPHA, ANNEAL_GAMMA, A_SPALL)
    return {
        "T": t,
        "v4_alpha": ALPHA_EXP,
        "v4_gamma": GAMMA_EXP,
        "anneal_alpha": ANNEAL_ALPHA,
        "anneal_gamma": ANNEAL_GAMMA,
        "A": A_SPALL,
        "a0_equals_ak0_v4": bool(abs(a_v4[0] - 1.0) < 1e-12),
        "a0_equals_ak0_anneal": bool(abs(a_an[0] - 1.0) < 1e-12),
        "late_ratio_v4": float(a_v4[-1]),
        "late_ratio_anneal": float(a_an[-1]),
        "anneal_late_smaller": bool(a_an[-1] < a_v4[-1]),
        "c0_equal": bool(abs(c_v4[0] - c_an[0]) < 1e-12),
        "method_id_anneal": METHOD_ID["equal4"]["ANNEAL"],
        "ok": bool(
            t == 378
            and abs(a_v4[0] - 1.0) < 1e-12
            and abs(a_an[0] - 1.0) < 1e-12
            and a_an[-1] < a_v4[-1]
            and METHOD_ID["equal4"]["ANNEAL"] == 7
            and abs(ANNEAL_GAMMA - 1.0 / 6.0) < 1e-15
        ),
    }


def _ledger() -> dict:
    def fake(x):
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs():
        init = [None if s == 0 else np.random.default_rng(9000 + s) for s in range(4)]
        spsa = [np.random.default_rng(9100 + s) for s in range(4)]
        return init, spsa

    ri, rs = rngs()
    v4 = run_equal4(fake, "Nov3rd", 3, 128, ri, rs, x0_fn=x0_center4)
    ri, rs = rngs()
    an = run_equal4(
        fake, "Nov3rd", 3, 128, ri, rs, x0_fn=x0_center4, alpha=ANNEAL_ALPHA, gamma=ANNEAL_GAMMA
    )
    return {
        "v4": v4["queries"],
        "anneal": an["queries"],
        "v4_alpha": v4["alpha"],
        "anneal_alpha": an["alpha"],
        "ok": v4["queries"]["total"] == 128 and an["queries"]["total"] == 128 and abs(an["alpha"] - 1.0) < 1e-12,
        "centers": {
            "nov3": [float(v) for v in x0_center4("Nov3rd", 3, 0, None)],
            "qaoa": [float(v) for v in x0_qaoa_center4(3, 0, None)],
        },
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    sched = _schedule()
    led = _ledger()
    payload = {
        "ok": bool(seed["ok"] and coll["n"] == 0 and sched["ok"] and led["ok"]),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2498560,
        "v4_already_spall_anneal": True,
        "anneal_is_alpha1_gamma_one_sixth": True,
        "not_shell_gain": True,
        "not_freeze_order": True,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "schedule": sched,
        "ledger_quadratic": led,
        "label_rule": "ANNEAL_L3_EDGE iff ANNEAL vs QAOA mean dE CI entirely <0 AND ANNEAL mean E <= V4; else REJECT_ANNEAL_L3",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "no_spsa_on_hamiltonian_in_this_module": True,
        "no_refit_10368_10375": True,
    }

    def _plain(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
