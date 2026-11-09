"""Fock-cutoff freeze. No SPSA on a Hamiltonian."""
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

from Nov3_INIT.alloc import METHOD_ID, iterations_equal4, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.circuit import Logical
from Nov3_INIT.fock import D0, D0_PLUS_2, FockNov3, snap_level_phase, w_binh
from Nov9_SK_SPSA.experiments.FT.instance import packing
from Nov9_SK_SPSA.experiments.FT.sim import _code_amps, snap_level_phase as snap_ft

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_CUTOFF.md"
OUT = ROOT / "results" / "cutoff" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10520, 10528))
RESERVED = {
    "local_r30_10504_10519": set(range(10504, 10520)),
    "ft_vm_10480_10487": set(range(10480, 10488)),
    "le_n21_10432_10439": set(range(10432, 10440)),
    "ft_and_prior_le_10463": set(range(0, 10464)),
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
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10520, 10528)),
    }


def _params() -> dict:
    rows = {}
    for L, Q in ((1, 1024), (2, 2048)):
        rows[str(L)] = {
            "QAOA": n_params("QAOA", L),
            "Nov3rd": n_params("Nov3rd", L),
            "Q": Q,
            "T": iterations_equal4(Q),
        }
    expected = {
        "1": {"QAOA": 2, "Nov3rd": 6, "Q": 1024, "T": 122},
        "2": {"QAOA": 4, "Nov3rd": 9, "Q": 2048, "T": 250},
    }
    return {
        "N": 15,
        "D0": D0,
        "D0_plus_2": D0_PLUS_2,
        "counts": rows,
        "ok": rows == expected and D0 == 8 and D0_PLUS_2 == 10,
    }


def _w_pad() -> dict:
    w8 = w_binh(8)
    w10 = w_binh(10)
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
    h3 = h
    for _ in range(2):
        h3 = np.kron(h3, h)
    snap = snap_level_phase(10, 0.3, -0.2, 0.1)
    return {
        "w8_is_h3": bool(np.allclose(w8, h3)),
        "w10_code_is_h3": bool(np.allclose(w10[:8, :8], h3)),
        "w10_pad_identity": bool(np.allclose(w10[8:, 8:], np.eye(2)) and np.allclose(w10[:8, 8:], 0.0) and np.allclose(w10[8:, :8], 0.0)),
        "snap_f_ge_8_zero": bool(np.allclose(snap[8:], 0.0)),
        "snap_matches_ft": bool(np.allclose(snap, snap_ft(10, 0.3, -0.2, 0.1))),
        "ok": True,
    }


def _reduction() -> dict:
    n, seed, b, L = 6, 10018, 0.4, 2
    gs = np.ones(1 << n, dtype=np.float64)
    gs /= np.linalg.norm(gs)
    logical = Logical(n, seed, b, gs)
    x = x0_center4("Nov3rd", L, 0, None)
    psi_l = logical.nov3(x, L)
    e_l = logical.energy(psi_l)
    f8 = FockNov3(logical, 8)
    e8, p8 = f8.propagate(x, L)
    amp = _code_amps(p8, f8.tables)
    fid = float(np.abs(np.vdot(psi_l, amp)) ** 2)
    leak8 = f8.leakage(p8)
    f10 = FockNov3(logical, 10)
    e10, p10 = f10.propagate(x, L)
    leak10 = f10.leakage(p10)
    w = _w_pad()
    w["ok"] = bool(
        w["w8_is_h3"]
        and w["w10_code_is_h3"]
        and w["w10_pad_identity"]
        and w["snap_f_ge_8_zero"]
        and w["snap_matches_ft"]
    )
    return {
        "nf8_equals_logical": bool(fid > 1.0 - 1e-10 and abs(e8 - e_l) < 1e-10),
        "fidelity_nf8": fid,
        "dE_nf8": abs(e8 - e_l),
        "nf10_equals_nf8": bool(abs(e10 - e8) < 1e-10),
        "dE_nf10": abs(e10 - e8),
        "p_leak_nf8": leak8["p_leak"],
        "p_leak_nf10": leak10["p_leak"],
        "p_leak_both_numerical_zero": bool(abs(leak8["p_leak"]) < 1e-12 and abs(leak10["p_leak"]) < 1e-12),
        "ideal_W_L_pad_does_not_leak": True,
        "compiled_W_L_not_modelled": True,
        "w_pad": w,
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs():
        init = [None if s == 0 else np.random.default_rng(7000 + s) for s in range(4)]
        spsa = [np.random.default_rng(8000 + s) for s in range(4)]
        return init, spsa

    ri, rs = rngs()
    n3 = run_equal4(fake, "Nov3rd", 1, 128, ri, rs, x0_fn=x0_center4)
    calls["n"] = 0
    ri, rs = rngs()
    qa = run_equal4(fake, "QAOA", 1, 128, ri, rs, x0_fn=lambda m, L, s, rng: x0_qaoa_center4(L, s, rng))
    return {
        "nov3": n3["queries"],
        "qaoa": qa["queries"],
        "ok": n3["queries"]["total"] == 128 and qa["queries"]["total"] == 128 and calls["n"] == 128,
        "method_ids": {"Nov3rd": METHOD_ID["equal4"]["Nov3rd"], "QAOA": METHOD_ID["equal4"]["QAOA"]},
        "rng_omits_nf": True,
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    params = _params()
    red = _reduction()
    led = _ledger()
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and params["ok"]
            and red["nf8_equals_logical"]
            and red["nf10_equals_nf8"]
            and red["p_leak_both_numerical_zero"]
            and red["w_pad"]["ok"]
            and led["ok"]
            and packing(15) == [[a, a + 5, a + 10] for a in range(5)]
            and METHOD_ID["equal4"]["Nov3rd"] == 1
            and METHOD_ID["equal4"]["QAOA"] == 0
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2695168,
        "D0": D0,
        "D0_plus_2": D0_PLUS_2,
        "D0_plus_4_not_run": True,
        "ideal_W_L_only": True,
        "compiled_W_L_not_modelled": True,
        "not_nov9": True,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "params": params,
        "reduction": red,
        "ledger_quadratic": led,
        "label_rule": "CUTOFF_ROBUST_L1 iff L=1 Nov3-QAOA mean dE CI entirely <0 at both D0 and D0+2; else CUTOFF_SENSITIVE",
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
