"""NF=32 Nov9 harness that accepts a per-instance B*.

Zach, 19:05 Asia/Shanghai on 2026-11-09: the next FT work focuses on the
max-entropy field. The b=1 and b=1.5 pilot remains the reference and this
module does not launch it. SPSA is not imported. Leakage is not renormalized.
Validation and final seeds are rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Nov9_SK_SPSA.experiments.FT.bstar import entropy_at, maximize_entropy, von_neumann_bits
from Nov9_SK_SPSA.experiments.FT.instance import make_J
from Nov9_SK_SPSA.experiments.FT.params import PILOT_SEEDS, TRAIN_SEEDS, assert_seed_allowed

CORRECTED_NF = 32
REFERENCE_FIELDS = (1.0, 1.5)
HERE = Path(__file__).resolve().parents[2] / "results" / "FT" / "harness_nf32"


def accept_bstar(records: list[dict]) -> dict:
    """Validate caller-supplied B* values and build idle task templates.

    Each record is {N, seed, Bstar, role}. role is pilot or train.
    The returned templates set nf=32 and optimization_authorized=False.
    """
    if not records:
        raise ValueError("no B* records")
    tasks = []
    for rec in records:
        seed = int(rec["seed"])
        role = str(rec["role"])
        assert_seed_allowed(seed, role)
        bstar = float(rec["Bstar"])
        N = int(rec["N"])
        checks = entropy_at(make_J(seed, N), bstar)
        if not checks["eligible"]:
            raise RuntimeError(f"B*={bstar} at N={N} seed={seed} failed positivity, parity, or gap checks")
        tasks.append({
            "N": N,
            "seed": seed,
            "role": role,
            "b": bstar,
            "field_source": "per_instance_Bstar",
            "fixed_field_sweep": False,
            "nf": CORRECTED_NF,
            "displacement": "true_oscillator_restricted_from_nf_plus_32",
            "renormalize_leakage": False,
            "optimizer": "SPSA_only_when_authorized",
            "optimization_authorized": False,
            "shared_ucz": True,
            "qaoa": "p_equals_L",
            "circuit": "[UcZ -> D(alpha_l) -> shell SNAP]^L",
            "checks": checks,
        })
    return {
        "corrected_nf": CORRECTED_NF,
        "reference_fields_not_relaunched": list(REFERENCE_FIELDS),
        "n_tasks": len(tasks),
        "optimization_authorized": False,
        "tasks": tasks,
    }


def _self_check() -> dict:
    bell = np.zeros(4, dtype=np.complex128)
    bell[0] = bell[3] = 1.0 / np.sqrt(2.0)
    product = np.zeros(4, dtype=np.complex128)
    product[0] = 1.0
    if abs(von_neumann_bits(bell, 1) - 1.0) > 1e-12:
        raise RuntimeError("Bell entropy is not 1 bit")
    if von_neumann_bits(product, 1) > 1e-12:
        raise RuntimeError("product entropy is not 0")
    J = make_J(10000, 6)
    found = maximize_entropy(J)
    accepted = accept_bstar([{"N": 6, "seed": 10000, "role": "pilot", "Bstar": found["Bstar"]}])
    task = accepted["tasks"][0]
    if task["nf"] != 32 or task["optimization_authorized"] or task["renormalize_leakage"]:
        raise RuntimeError("harness template is not the idle NF=32 B* form")
    try:
        accept_bstar([{"N": 6, "seed": 11000, "role": "pilot", "Bstar": 1.0}])
    except RuntimeError:
        blind_rejected = True
    else:
        raise RuntimeError("validation seed was accepted")
    return {
        "kind": "harness_self_check_not_a_campaign",
        "reference_b_1_and_1_5_retained": True,
        "optimization_run": False,
        "bell_entropy_bits": 1.0,
        "product_entropy_bits": 0.0,
        "one_dev_instance": {"N": 6, "seed": 10000, **found},
        "accepted_template": task,
        "validation_seed_rejected": blind_rejected,
        "pilot_seeds_not_scanned": list(PILOT_SEEDS[1:]),
        "train_seeds_not_scanned": list(TRAIN_SEEDS),
    }


def main():
    doc = _self_check()
    HERE.mkdir(parents=True, exist_ok=True)
    path = HERE / "SELF_CHECK.json"
    path.write_text(json.dumps(doc, indent=2))
    one = doc["one_dev_instance"]
    print(
        f"self-check B*={one['Bstar']:.6f} S={one['entropy_bits']:.4f} bits "
        f"gap={one['checks']['gap']:.4e} positive={one['checks']['positive']} even={one['checks']['even']}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
