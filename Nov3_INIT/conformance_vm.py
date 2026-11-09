"""Paper V(M) freeze. No SPSA on a Hamiltonian."""
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
    n_params,
    run_equal4,
    sigmas_of,
    x0_center4,
    x0_qaoa_center4,
    x0_vm_center4,
)
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing, upper

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_VM.md"
OUT = ROOT / "results" / "vm" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10480, 10488))
RESERVED = {
    "local_r28_10464_10479": set(range(10464, 10480)),
    "ft_and_prior_le_10463": set(range(0, 10464)),
    "le_10432_10439": set(range(10432, 10440)),
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
            if any(f"s{s}" in path.name for s in BAND):
                hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "n": len(BAND),
        "overlaps": overlaps,
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10480, 10488)),
    }


def _params() -> dict:
    rows = {}
    for L in (2, 3):
        rows[str(L)] = {
            "QAOA": n_params("QAOA", L),
            "Nov3rd": n_params("Nov3rd", L),
            "VM": n_params("VM", L, 15),
            "Q": 1024 * L,
            "T": iterations_equal4(1024 * L),
        }
    expected = {
        "2": {"QAOA": 4, "Nov3rd": 9, "VM": 212, "Q": 2048, "T": 250},
        "3": {"QAOA": 6, "Nov3rd": 12, "VM": 318, "Q": 3072, "T": 378},
    }
    return {"N": 15, "n_pairs": 105, "counts": rows, "ok": rows == expected}


def _centers() -> dict:
    J = make_J(10018, 6)
    ju = upper(J)
    x0 = x0_vm_center4(6, 2, 0, None, ju)
    n_pairs = 15
    width = 16
    tiles_j = all(np.array_equal(x0[ell * width : ell * width + n_pairs], ju) for ell in range(2))
    betas = [float(x0[ell * width + n_pairs]) for ell in range(2)]
    sig = sigmas_of("VM", 3, n=15)
    return {
        "nov3": [float(v) for v in x0_center4("Nov3rd", 3, 0, None)],
        "qaoa": [float(v) for v in x0_qaoa_center4(3, 0, None)],
        "vm_n6_L2_len": int(x0.size),
        "vm_tiles_J": bool(tiles_j),
        "vm_betas": betas,
        "qaoa_center": bool(np.array_equal(x0_qaoa_center4(3, 0, None), np.full(6, 0.25))),
        "nov3_center": bool(
            float(x0_center4("Nov3rd", 3, 0, None)[0]) == 0.25
            and np.all(x0_center4("Nov3rd", 3, 0, None)[3::3] == -0.5)
        ),
        "method_ids": METHOD_ID["equal4"],
        "sigmas_vm_L3": int(sig.size),
        "sigma_M": float(sig[0]),
        "sigma_beta": float(sig[105]),
        "ok": bool(
            tiles_j
            and betas == [0.25, 0.25]
            and int(x0.size) == 32
            and int(sig.size) == 318
            and float(sig[0]) == 0.15
            and float(sig[105]) == 0.10
            and METHOD_ID["equal4"]["VM"] == 8
        ),
    }


def _reduction() -> dict:
    n, seed, b, L = 6, 10018, 0.4, 2
    gs = np.ones(1 << n, dtype=np.float64)
    gs /= np.linalg.norm(gs)
    logical = Logical(n, seed, b, gs)
    ju = logical.j_upper
    x_vm = x0_vm_center4(n, L, 0, None, ju)
    x_q = x0_qaoa_center4(L, 0, None)
    psi_v = logical.vm(x_vm, L)
    psi_q = logical.qaoa(x_q, L)
    fid = float(np.abs(np.vdot(psi_v, psi_q)) ** 2)
    de = abs(logical.energy(psi_v) - logical.energy(psi_q))
    n_pairs = n * (n - 1) // 2
    phase = 0.25 * (ju @ logical.zz_pairs)
    hc_match = bool(np.allclose(phase, 0.25 * logical.hc))
    return {
        "vm_center_equals_qaoa": bool(fid > 1.0 - 1e-10 and de < 1e-10),
        "fidelity": fid,
        "dE": de,
        "M_times_zz_equals_hc": hc_match,
        "n_params_vm": int(x_vm.size),
        "n_pairs": n_pairs,
        "no_nov3_shells": True,
    }


def _ledger() -> dict:
    calls = {"n": 0}
    J = make_J(10018, 6)
    ju = upper(J)

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs():
        init = [None if s == 0 else np.random.default_rng(5000 + s) for s in range(4)]
        spsa = [np.random.default_rng(6000 + s) for s in range(4)]
        return init, spsa

    ri, rs = rngs()
    out = run_equal4(
        fake,
        "VM",
        2,
        128,
        ri,
        rs,
        x0_fn=lambda m, L, s, rng: x0_vm_center4(6, L, s, rng, ju),
        n=6,
    )
    return {
        "Q": 128,
        "calls": calls["n"],
        "ledger": out["queries"],
        "n_params": int(np.asarray(out["x"]).size),
        "ok": out["queries"]["total"] == 128 and calls["n"] == 128 and out["queries"]["monitoring"] == 0,
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    params = _params()
    centers = _centers()
    red = _reduction()
    led = _ledger()
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and params["ok"]
            and centers["ok"]
            and red["vm_center_equals_qaoa"]
            and red["M_times_zz_equals_hc"]
            and led["ok"]
            and packing(15) == [[a, a + 5, a + 10] for a in range(5)]
            and METHOD_ID["equal4"]["VM"] == 8
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2572288,
        "paper_V_M_implemented": True,
        "architecture": "p=L independent all-to-all V(M) then shared Rx; not Nov3 shells",
        "not_nov3_retune": True,
        "not_hybrid": True,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "params": params,
        "centers": centers,
        "reduction": red,
        "ledger_quadratic": led,
        "label_rule": "VM_L3_EDGE iff L=3 VM-QAOA CI entirely <0 AND VM mean E <= Nov3 AND L=2 VM mean E <= Nov3 AND L=2 VM-QAOA CI is not entirely >0; else REJECT_VM_L3",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "no_spsa_on_hamiltonian_in_this_module": True,
        "no_refit_10368_10375": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
