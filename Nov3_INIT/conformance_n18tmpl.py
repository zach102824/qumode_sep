"""N=18 L=1 two-template freeze. No SPSA on panel Hamiltonians."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import METHOD_ID, iterations_equal4, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.circuit import Logical
from Nov3_INIT.n18tmpl import TEST, TRAIN, build_freeze
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_N18TMPL.md"
ALLOC = ROOT / "alloc.py"
RES = ROOT / "results"
OUT = RES / "n18tmpl" / "CONFORMANCE.json"
BAND = list(range(10808, 10824))
HOLD = list(range(10624, 10632))
Q = 1024
HIST = 3145904
INDEP = 16 * 2 * Q
XFER_IF_BOTH = 8 * 7
EXPECTED_IF_BOTH = INDEP + XFER_IF_BOTH
RESERVED = {
    "r34_10712_10727": set(range(10712, 10728)),
    "le_10728_10743": set(range(10728, 10744)),
    "held_unused_10624_10631": set(HOLD),
    "unused_hole_10560_10567": set(range(10560, 10568)),
    "pr124_10776_10791": set(range(10776, 10792)),
    "le_hour_10792_10807": set(range(10792, 10808)),
    "pr122_10744_10759": set(range(10744, 10760)),
    "le_pr125_10760_10775": set(range(10760, 10776)),
    "pr120_10696_10711": set(range(10696, 10712)),
    "pr118_10664_10679": set(range(10664, 10680)),
    "pr121_10680_10695": set(range(10680, 10696)),
    "pr119_10648_10663": set(range(10648, 10664)),
    "pr116_10600_10615": set(range(10600, 10616)),
    "pr117_10616_10623": set(range(10616, 10624)),
    "le_10432_10439": set(range(10432, 10440)),
    "le_pr115_10552_10559": set(range(10552, 10560)),
    "local_r31_10536_10551": set(range(10536, 10552)),
    "local_r32_10584_10599": set(range(10584, 10600)),
    "local_r30_10504_10519": set(range(10504, 10520)),
    "local_r33_10632_10647": set(range(10632, 10648)),
    "ft_10520_10535": set(range(10520, 10536)),
    "width_10568_10583": set(range(10568, 10584)),
    "ft_vm_10480_10487": set(range(10480, 10488)),
    "ft_anneal_10456_10463": set(range(10456, 10464)),
    "ft_spread8_10376_10383": set(range(10376, 10384)),
    "shellgain_10368_10375": set(range(10368, 10376)),
    "holdout_10144_10151": set(range(10144, 10152)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
}


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def _collisions() -> dict:
    hits = []
    if RES.exists():
        for path in RES.rglob("*"):
            if path.suffix not in {".json", ".npz", ".md"}:
                continue
            if any(f"s{s}" in path.name for s in BAND + HOLD):
                hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "train": list(TRAIN),
        "test": list(TEST),
        "hold_unused": HOLD,
        "n": len(BAND),
        "overlaps": overlaps,
        "ok": bool(
            all(len(v) == 0 for v in overlaps.values())
            and BAND == list(range(10808, 10824))
            and list(TRAIN) == list(range(10808, 10816))
            and list(TEST) == list(range(10816, 10824))
            and not (set(TRAIN) & set(TEST))
            and set(TRAIN) | set(TEST) == band
            and HOLD == list(range(10624, 10632))
            and not (band & set(HOLD))
        ),
    }


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _probe() -> dict:
    dummy = np.full(1 << 18, 2.0 ** (-0.5 * 18), dtype=np.float64)
    logical = Logical(18, 10018, 0.5, dummy)
    x = x0_center4("Nov3rd", 1, 0, None)
    rss0 = _rss_kib()
    t0 = time.perf_counter()
    e = logical.energy(logical.nov3(x, 1))
    dt = time.perf_counter() - t0
    rss1 = _rss_kib()
    proj = Q * dt
    allowed = bool(dt < 0.5 and proj < 1800.0 and rss1 / (1024.0 * 1024.0) < 4.0 and np.isfinite(e))
    return {
        "seed_not_fitted": 10018,
        "nov3_L1_seconds": dt,
        "nov3_L1_energy": float(e),
        "projected_Q1024_cell_seconds": proj,
        "rss_kib_start": rss0,
        "rss_kib_after": rss1,
        "allowed": allowed,
        "no_spsa": True,
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    init = [None if s == 0 else np.random.default_rng(11100 + s) for s in range(4)]
    spsa = [np.random.default_rng(11200 + s) for s in range(4)]
    out = run_equal4(fake, "Nov3rd", 1, Q, init, spsa, x0_fn=x0_center4)
    q = out["queries"]
    ok = bool(
        out["queries"]["total"] == Q
        and calls["n"] == Q
        and out["n_iter"] == 122
        and q["init"] == 4
        and q["calibration"] == 32
        and q["iteration"] == 976
        and q["candidate"] == 12
        and q["monitoring"] == 0
        and q["screen"] == 0
    )
    return {"ok": ok, "ledger": q, "n_iter": int(out["n_iter"])}


def _recipe() -> dict:
    xn = x0_center4("Nov3rd", 1, 0, None)
    xq = x0_qaoa_center4(1, 0, None)
    expected_n = np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0])
    expected_q = np.array([0.25, 0.25])
    return {
        "nov3_start0": [float(v) for v in xn],
        "qaoa_start0": [float(v) for v in xq],
        "n_params_nov3": n_params("Nov3rd", 1),
        "n_params_qaoa": n_params("QAOA", 1),
        "T": iterations_equal4(Q),
        "ok": bool(
            np.allclose(xn, expected_n)
            and np.allclose(xq, expected_q)
            and n_params("Nov3rd", 1) == 6
            and n_params("QAOA", 1) == 2
            and iterations_equal4(Q) == 122
            and METHOD_ID["equal4"]["Nov3rd"] == 1
            and METHOD_ID["equal4"]["QAOA"] == 0
        ),
    }


def _freeze_guard() -> dict:
    rng = np.random.default_rng(1)
    rows = []
    for i, s in enumerate(TRAIN):
        xn_s = rng.normal(size=6)
        xn_s[3] = 0.7 if i < 4 else -0.7
        xq_s = rng.normal(size=2)
        xq_s[1] = 0.3 if i % 2 == 0 else -0.3
        rows.append({"seed": int(s), "arm": "nov3_center4", "ok": True, "split": "train", "x": xn_s.tolist()})
        rows.append({"seed": int(s), "arm": "qaoa_center4", "ok": True, "split": "train", "x": xq_s.tolist()})
    fr = build_freeze(rows, "deadbeef")
    leaked = False
    try:
        bad = list(rows) + [{"seed": 10816, "arm": "nov3_center4", "ok": True, "split": "test", "x": [0.0] * 6}]
        build_freeze(bad, "deadbeef")
        leaked = True
    except ValueError:
        leaked = False
    return {
        "n_templates": fr["n_templates"],
        "planned_transfer_queries": fr["planned_transfer_queries"],
        "primary_gate": fr["primary_gate"],
        "nov3_two_template": fr["nov3_two_template"],
        "qaoa_two_template": fr["qaoa_two_template"],
        "no_test_peek": not leaked,
        "ok": bool(
            fr["n_templates"] == 7
            and fr["planned_transfer_queries"] == 56
            and fr["primary_gate"] == "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI"
            and fr["nov3_two_template"]["two_template"]
            and fr["qaoa_two_template"]["two_template"]
            and not leaked
            and fr["do_not_switch_primary_after_test"]
        ),
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    rec = _recipe()
    led = _ledger()
    probe = _probe()
    freeze = _freeze_guard()
    alloc_sha = hashlib.sha256(ALLOC.read_bytes()).hexdigest()
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and rec["ok"]
            and led["ok"]
            and probe["allowed"]
            and freeze["ok"]
            and packing(18) == [[a, a + 6, a + 12] for a in range(6)]
            and INDEP == 32768
            and EXPECTED_IF_BOTH == 32824
            and HIST + EXPECTED_IF_BOTH == 3178728
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "alloc_sha256": alloc_sha,
        "le_alloc_sha_prefix_same_v4_recipe": "de595de59",
        "historical_sk_queries": HIST,
        "expected_independent_queries": INDEP,
        "expected_transfer_queries_if_both_clusters": XFER_IF_BOTH,
        "expected_new_energy_queries_if_both_clusters": EXPECTED_IF_BOTH,
        "expected_total_if_both_clusters": HIST + EXPECTED_IF_BOTH,
        "Q": Q,
        "q": Q,
        "L": 1,
        "N": 18,
        "T": 122,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "recipe": rec,
        "ledger_quadratic": led,
        "probe": probe,
        "freeze_guard": freeze,
        "primary_gate": "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI",
        "do_not_switch_primary_after_test": True,
        "not_unsigned_mean_copy": True,
        "not_sign_aligned_only_copy": True,
        "not_architecture_change": True,
        "not_gr_retune": True,
        "not_l3_budget": True,
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "holdout_l2_fail_stands": True,
        "compiler_stream_closed": True,
        "pr111_closed": True,
        "pr117_l3_2xbudget_fail_stands": True,
        "pr118_n12_stands": True,
        "pr120_n9_unsigned_unstable_stands": True,
        "pr122_n9_two_template_stands": True,
        "pr124_n12_two_template_stands": True,
        "no_spsa_on_panel_hamiltonian_in_this_module": True,
        "label_rule": "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI iff n=8 better-of-two Nov3-QAOA dE CI entirely <0 on test; else UNSTABLE",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
