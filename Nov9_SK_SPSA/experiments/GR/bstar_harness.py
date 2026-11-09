"""Accept a per-instance maximum-entropy field B*. Do not launch a batch.

Zach, 19:05 Asia/Shanghai on 2026-11-09: the next comparison is the max-entropy
case only. The finished b = 1 and b = 1.5 runs stay the reference. This module
builds the job list for a later assignment. It has no path into SPSA.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import assert_dev_seed, load_prereg
from hamiltonian import build_instance, parity_of_vector

# Protocol default circuits for the next assignment, not the GR ablation set.
CORE_ARMS = ("Nov3rd", "inter_id", "QAOA")
CUT = "low_bits_0_through_N//2-1_versus_the_rest"


def fixed_balanced_cut(n: int) -> dict:
    """One predetermined cut. The cut is not chosen to maximize entropy."""
    if int(n) < 2:
        raise ValueError(f"N={n} has no balanced cut")
    k = int(n) // 2
    side_a = list(range(k))
    side_b = list(range(k, int(n)))
    if abs(len(side_a) - len(side_b)) > 1:
        raise RuntimeError("cut is not balanced")
    return {"side_a": side_a, "side_b": side_b, "k": k, "name": CUT}


def cut_entropy(psi: np.ndarray, n: int) -> float:
    """Von Neumann entropy, natural log, of the fixed balanced cut.

    Site 0 is the least significant bit, so side A is the trailing axis.
    """
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    if psi.size != 1 << int(n):
        raise ValueError("state length does not match N")
    k = int(n) // 2
    mat = psi.reshape(1 << (int(n) - k), 1 << k)
    singular = np.linalg.svd(mat, compute_uv=False)
    weight = singular * singular
    weight = weight[weight > 0.0]
    return float(-np.sum(weight * np.log(weight)))


def spsa_b_key(b: float) -> int:
    """The integer already used in the SPSA SeedSequence: round(10 * b)."""
    return int(round(10.0 * float(b)))


def gs_contract(inst) -> dict:
    """Positive even ground state, gap, and parity symmetry. No new search."""
    parity = float(inst.gs_parity)
    report = {
        "seed": int(inst.seed),
        "N": int(inst.n),
        "b": float(inst.b),
        "gs_min": float(inst.gs_min),
        "gs_parity": parity,
        "residual": float(inst.residual),
        "gap": float(inst.gap),
        "delta_odd": float(inst.delta_odd),
        "E0": float(inst.E0),
        "E1": float(inst.E1),
        "positive": float(inst.gs_min) > -1e-8 and float(np.sum(inst.gs)) > 0.0,
        "even": parity > 1.0 - 1e-6,
        "gapped": float(inst.gap) > 0.0 and float(inst.delta_odd) > 0.0,
        "residual_ok": float(inst.residual) <= 1e-8,
        "cut": fixed_balanced_cut(inst.n),
        "cut_entropy": cut_entropy(inst.gs, inst.n),
    }
    report["pass"] = bool(report["positive"] and report["even"] and report["gapped"] and report["residual_ok"])
    return report


def cutoff_status(b: float, certificate: dict | None) -> dict:
    """The b = 1 and 1.5 cutoff does not transfer to a new B* until repeated.

    The displacement state does not use b, but the certified energy does.
    """
    key = spsa_b_key(b)
    certified = []
    if certificate:
        certified = [spsa_b_key(v) for v in certificate.get("b_values_certified", [])]
    ok = key in certified and certificate.get("passed") is True
    return {
        "b": float(b),
        "spsa_b_key": key,
        "certified_for_this_b": bool(ok),
        "reuse_of_b_1_and_1_5_cutoff": False,
    }


def load_manifest(path: Path) -> list[dict]:
    doc = json.loads(Path(path).read_text())
    rows = doc.get("instances")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest needs a non-empty instances list")
    seen = set()
    out = []
    for row in rows:
        seed = int(row["seed"])
        n = int(row["N"])
        b = float(row["B_star"])
        assert_dev_seed(seed, pilot=False)
        if n % 3 != 0 or n < 6:
            raise ValueError(f"N={n} is not a 3-bit cavity count")
        if not math.isfinite(b) or b <= 0.0:
            raise ValueError(f"B* must be finite and positive, got {b}")
        key = (seed, n)
        if key in seen:
            raise ValueError(f"duplicate instance {key}")
        seen.add(key)
        out.append(
            {
                "seed": seed,
                "N": n,
                "B_star": b,
                "spsa_b_key": spsa_b_key(b),
                "source": str(row.get("source", "manifest")),
            }
        )
    return out


def jobs_from_manifest(rows: list[dict], *, L_list: list[int], caps: list[int], nf: int | None, c: float | None) -> list[dict]:
    """Job records only. This function does not evaluate energy or call SPSA."""
    pre = load_prereg()
    arms = {a["name"]: a for a in pre["arms"]}
    missing = [name for name in CORE_ARMS if name not in arms]
    if missing:
        raise RuntimeError(f"core arms missing from prereg: {missing}")
    jobs = []
    for row in rows:
        for L in L_list:
            for cap in caps:
                for name in CORE_ARMS:
                    spec = arms[name]
                    if name == "inter_id" and not (spec["gate"] == "true" and spec["ucz"] == "Uz" and spec["encoding"] == "binary"):
                        raise RuntimeError("inter_id is no longer the true-D middle circuit")
                    jobs.append(
                        {
                            "seed": row["seed"],
                            "N": row["N"],
                            "b": row["B_star"],
                            "b_role": "per_instance_max_entropy_Bstar",
                            "spsa_b_key": row["spsa_b_key"],
                            "L": int(L),
                            "cap": int(cap),
                            "arm": name,
                            "arm_spec": spec,
                            "nf": nf,
                            "c": c,
                            "cutoff_certified_for_this_b": False,
                            "launch": False,
                        }
                    )
    return jobs


def refuse_launch() -> None:
    raise RuntimeError("B* batches stay unlaunched until the next three-idea assignment")


def self_check() -> dict:
    """One reference instance at the already reported b = 1, plus a Bell-cut identity."""
    n = 6
    k = n // 2
    psi = np.zeros(1 << n, dtype=np.float64)
    # Site 0 is in side A and site 3 is in side B. A single Bell pair has entropy log 2.
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[(1 << 0) | (1 << k)] = 1.0 / math.sqrt(2.0)
    s_bell = cut_entropy(psi, n)
    if abs(s_bell - math.log(2.0)) > 1e-12:
        raise RuntimeError(f"Bell cut entropy {s_bell} != log 2")
    inst = build_instance(10000, 6, 1.0, pilot=True)
    contract = gs_contract(inst)
    if not contract["pass"]:
        raise RuntimeError(f"reference contract failed: {contract}")
    rows = load_manifest_records([{"seed": 10000, "N": 6, "B_star": 1.0, "source": "reference_b_not_a_new_sweep"}])
    jobs = jobs_from_manifest(rows, L_list=[1], caps=[1024], nf=None, c=None)
    if len(jobs) != len(CORE_ARMS):
        raise RuntimeError("job builder did not emit exactly the three core arms")
    if any(job["launch"] for job in jobs):
        raise RuntimeError("job was marked launchable")
    blocked = False
    try:
        load_manifest_records([{"seed": 11000, "N": 6, "B_star": 1.0}])
    except RuntimeError:
        blocked = True
    if not blocked:
        raise RuntimeError("validation seed was accepted")
    try:
        refuse_launch()
        raise RuntimeError("launch was not refused")
    except RuntimeError as exc:
        if "unlaunched" not in str(exc):
            raise
    return {
        "bell_entropy": s_bell,
        "reference_contract_b1_seed10000_N6": contract,
        "core_arms": list(CORE_ARMS),
        "n_jobs_in_self_check": len(jobs),
        "spsa_called": False,
        "new_b_sweep": False,
    }


def load_manifest_records(rows: list[dict]) -> list[dict]:
    """Same checks as a file manifest, for the self-check."""
    tmp = Path("/tmp/gr_bstar_selfcheck.json")
    tmp.write_text(json.dumps({"instances": rows}))
    try:
        return load_manifest(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print("usage: python3 bstar_harness.py self-check | dry-jobs MANIFEST.json")
        return 0
    if argv[1] == "self-check":
        print(json.dumps(self_check(), indent=2))
        return 0
    if argv[1] == "dry-jobs":
        if len(argv) != 3:
            raise SystemExit("dry-jobs needs a manifest path")
        rows = load_manifest(Path(argv[2]))
        jobs = jobs_from_manifest(rows, L_list=[1, 2, 3], caps=[128, 1024], nf=None, c=None)
        print(json.dumps({"n_instances": len(rows), "n_jobs": len(jobs), "arms": list(CORE_ARMS), "launch": False}, indent=2))
        return 0
    if argv[1] == "launch":
        refuse_launch()
    raise SystemExit(f"unknown command {argv[1]}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
