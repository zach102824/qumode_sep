"""Shell-gain freeze. No SPSA on a Hamiltonian."""
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
    METHOD_ID,
    iterations_equal4,
    run_equal4,
    shell_a_mult,
    sigmas_of,
    spsa_block,
    x0_center4,
    x0_qaoa_center4,
    Ledger,
)

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_SHELLGAIN.md"
OUT = ROOT / "results" / "shellgain" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10368, 10376))
RESERVED = {
    "le_two_stage_10360_10367": set(range(10360, 10368)),
    "local_r25_10344_10359": set(range(10344, 10360)),
    "ft_and_prior_le_10335": set(range(0, 10336)),
    "holdout_10144_10151": set(range(10144, 10152)),
    "pair_10328_10335": set(range(10328, 10336)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
}


def _collisions() -> dict:
    hits = []
    if RES.exists():
        for path in RES.rglob("*"):
            name = path.name
            if not any(f"s{s}" in name for s in BAND):
                continue
            if path.suffix in {".json", ".npz", ".md"}:
                hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "n": len(BAND),
        "overlaps": overlaps,
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10368, 10376)),
    }


def _gains() -> dict:
    s2 = shell_a_mult(3, 2.0)
    s4 = shell_a_mult(3, 4.0)
    joint = shell_a_mult(3, 1.0)
    sig = sigmas_of("Nov3rd", 3)
    expected_sig = np.array([0.10, 0.25, 0.25] + [0.15] * 9)
    return {
        "JOINT": [float(v) for v in joint],
        "SHELL2": [float(v) for v in s2],
        "SHELL4": [float(v) for v in s4],
        "QAOA": [1.0] * 6,
        "sigma_nov3_L3": [float(v) for v in sig],
        "T_Q3072": iterations_equal4(3072),
        "ok": bool(
            np.array_equal(joint, np.ones(12))
            and np.array_equal(s2[:3], np.ones(3))
            and np.array_equal(s2[3:], np.full(9, 2.0))
            and np.array_equal(s4[3:], np.full(9, 4.0))
            and np.array_equal(sig, expected_sig)
            and iterations_equal4(3072) == 378
            and METHOD_ID["equal4"]["Nov3rd"] == 1
            and METHOD_ID["equal4"]["SHELL2"] == 3
            and METHOD_ID["equal4"]["SHELL4"] == 4
            and METHOD_ID["equal4"]["QAOA"] == 0
        ),
    }


def _centers() -> dict:
    x_n = x0_center4("Nov3rd", 3, 0, None)
    x_q = x0_qaoa_center4(3, 0, None)
    return {
        "nov3": [float(v) for v in x_n],
        "qaoa": [float(v) for v in x_q],
        "ok": bool(
            float(x_n[0]) == 0.25
            and np.all(x_n[3::3] == -0.5)
            and np.array_equal(x_q, np.full(6, 0.25))
        ),
    }


def _ledger_and_emphasis() -> dict:
    def fake(x):
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs():
        init = [None if s == 0 else np.random.default_rng(5000 + s) for s in range(4)]
        spsa = [np.random.default_rng(6000 + s) for s in range(4)]
        return init, spsa

    ri, rs = rngs()
    joint = run_equal4(fake, "Nov3rd", 3, 128, ri, rs, x0_fn=x0_center4, a_mult=shell_a_mult(3, 1.0))
    ri, rs = rngs()
    shell4 = run_equal4(fake, "Nov3rd", 3, 128, ri, rs, x0_fn=x0_center4, a_mult=shell_a_mult(3, 4.0))
    # One iteration, identical RNG: Uc steps match; shells of SHELL4 move at least as far.
    x0 = x0_center4("Nov3rd", 3, 1, np.random.default_rng(7))
    sig = sigmas_of("Nov3rd", 3)

    def run_block(ratio):
        led = Ledger()
        return spsa_block(fake, x0, sig, 1, np.random.default_rng(11), led, a_mult=shell_a_mult(3, ratio))

    b1 = run_block(1.0)
    b4 = run_block(4.0)
    move1 = np.abs(np.asarray(b1["x"]) - x0)
    move4 = np.abs(np.asarray(b4["x"]) - x0)
    return {
        "joint_Q128": joint["queries"],
        "shell4_Q128": shell4["queries"],
        "ledger_ok": joint["queries"]["total"] == 128 and shell4["queries"]["total"] == 128,
        "shell_motion_shell4_ge_joint": bool(np.all(move4[3:] + 1e-15 >= move1[3:])),
        "uc_motion_equal": bool(np.allclose(move4[:3], move1[:3])),
        "a_mult_joint": b1["a_mult"],
        "a_mult_shell4": b4["a_mult"],
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    gains = _gains()
    centers = _centers()
    led = _ledger_and_emphasis()
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and gains["ok"]
            and centers["ok"]
            and led["ledger_ok"]
            and led["shell_motion_shell4_ge_joint"]
            and led["uc_motion_equal"]
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2301952,
        "not_le_two_stage": True,
        "not_paper_V_M": True,
        "not_percavity": True,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "gains": gains,
        "centers": centers,
        "ledger_quadratic": led,
        "label_rule": "SHELL_GAIN_L3_EDGE iff best(SHELL2,SHELL4) vs QAOA mean dE CI entirely <0 AND that winner mean E <= JOINT; else REJECT_SHELL_GAIN",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "no_spsa_on_hamiltonian_in_this_module": True,
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
