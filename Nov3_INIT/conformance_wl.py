"""Non-ideal W_L freeze. No SPSA on a Hamiltonian."""
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
from Nov3_INIT.fock import EPS_WL, FockNov3, w_binh, w_nonideal
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WL.md"
OUT = ROOT / "results" / "wl" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10528, 10536))
RESERVED = {
    "local_r30_10504_10519": set(range(10504, 10520)),
    "ft_cutoff_10520_10527": set(range(10520, 10528)),
    "ft_vm_10480_10487": set(range(10480, 10488)),
    "le_10432_10439": set(range(10432, 10440)),
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
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10528, 10536)),
    }


def _channel() -> dict:
    w0 = w_nonideal(10, 0.0)
    w1 = w_nonideal(10, EPS_WL)
    g = w1 @ np.linalg.inv(w_binh(10))
    return {
        "eps": EPS_WL,
        "nf": 10,
        "eps0_equals_binh": bool(np.allclose(w0, w_binh(10))),
        "givens_real": bool(np.allclose(g.imag, 0.0) if np.iscomplexobj(g) else True),
        "g77": float(g[7, 7].real),
        "g88": float(g[8, 8].real),
        "g78": float(g[7, 8].real),
        "cos": float(np.cos(EPS_WL)),
        "sin": float(np.sin(EPS_WL)),
        "ok": bool(
            abs(EPS_WL - 0.10) < 1e-15
            and np.allclose(w0, w_binh(10))
            and abs(float(g[7, 7]) - float(np.cos(EPS_WL))) < 1e-12
            and abs(float(g[7, 8]) + float(np.sin(EPS_WL))) < 1e-12
            and METHOD_ID["equal4"]["WL"] == 9
        ),
    }


def _reduction() -> dict:
    n, seed, b, L = 6, 10018, 0.4, 1
    gs = np.ones(1 << n, dtype=np.float64)
    gs /= np.linalg.norm(gs)
    logical = Logical(n, seed, b, gs)
    xn = x0_center4("Nov3rd", L, 0, None)
    xq = x0_qaoa_center4(L, 0, None)
    e_l = logical.energy(logical.nov3(xn, L))
    e_ql = logical.energy(logical.qaoa(xq, L))
    ideal_rw = FockNov3(logical, 10, 0.0, "rewrite")
    ideal_hw = FockNov3(logical, 10, 0.0, "hardware")
    e_rw, _ = ideal_rw.propagate(xn, L)
    e_hw, _ = ideal_hw.propagate(xn, L)
    e_qrw, _ = ideal_rw.propagate_qaoa(xq, L)
    leaky = FockNov3(logical, 10, EPS_WL, "hardware")
    e_n, p_n = leaky.propagate(xn, L)
    e_q, p_q = leaky.propagate_qaoa(xq, L)
    leak_n = leaky.leakage(p_n)["p_leak"]
    leak_q = leaky.leakage(p_q)["p_leak"]
    return {
        "eps0_rewrite_equals_logical": bool(abs(e_rw - e_l) < 1e-10),
        "eps0_hardware_equals_rewrite": bool(abs(e_hw - e_rw) < 1e-10),
        "eps0_qaoa_equals_logical": bool(abs(e_qrw - e_ql) < 1e-10),
        "center_nov3_equals_qaoa_leaky": bool(abs(e_n - e_q) < 1e-10),
        "p_leak_leaky_nov3": leak_n,
        "p_leak_leaky_qaoa": leak_q,
        "p_leak_positive": bool(leak_n > 1e-6),
        "Q": 1024,
        "T": iterations_equal4(1024),
        "n_params": {"Nov3rd": n_params("Nov3rd", 1), "QAOA": n_params("QAOA", 1)},
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs():
        init = [None if s == 0 else np.random.default_rng(7100 + s) for s in range(4)]
        spsa = [np.random.default_rng(8100 + s) for s in range(4)]
        return init, spsa

    ri, rs = rngs()
    out = run_equal4(fake, "Nov3rd", 1, 128, ri, rs, x0_fn=x0_center4)
    return {"ok": out["queries"]["total"] == 128 and calls["n"] == 128, "ledger": out["queries"]}


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    ch = _channel()
    red = _reduction()
    led = _ledger()
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and ch["ok"]
            and red["eps0_rewrite_equals_logical"]
            and red["eps0_hardware_equals_rewrite"]
            and red["eps0_qaoa_equals_logical"]
            and red["center_nov3_equals_qaoa_leaky"]
            and red["p_leak_positive"]
            and red["T"] == 122
            and led["ok"]
            and packing(15) == [[a, a + 5, a + 10] for a in range(5)]
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2793472,
        "eps": EPS_WL,
        "nf": 10,
        "channel": "real Givens 7-8 after ideal W_BinH; SNAP phase 0 on f>=8",
        "not_d2_periodic_snap": True,
        "not_invented_pulse_compile": True,
        "compiled_snap_d_still_unmodelled": True,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "channel_check": ch,
        "reduction": red,
        "ledger_quadratic": led,
        "label_rule": "WL_ROBUST_L1 iff L=1 non-ideal Nov3-QAOA mean dE CI entirely <0; else WL_SENSITIVE",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "no_spsa_on_hamiltonian_in_this_module": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
