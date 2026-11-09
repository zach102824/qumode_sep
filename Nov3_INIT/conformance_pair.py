"""Pair-shell freeze. No SPSA on a Hamiltonian."""
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
    x0_pair_center4,
    x0_qaoa_center4,
)
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_PAIR.md"
OUT = ROOT / "results" / "pair" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10328, 10336))
RESERVED = {
    "le_10320_10327": set(range(10320, 10328)),
    "local_r23_10304_10319": set(range(10304, 10320)),
    "ft_and_prior_le_10247": set(range(0, 10248)),
    "holdout_10144_10151": set(range(10144, 10152)),
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
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10328, 10336)),
    }


def _params() -> dict:
    rows = {}
    for L in (2, 3):
        rows[str(L)] = {
            "QAOA": n_params("QAOA", L),
            "Nov3rd": n_params("Nov3rd", L),
            "Nov3pair": n_params("Nov3pair", L, 15),
            "Q": 1024 * L,
            "T": iterations_equal4(1024 * L),
        }
    expected = {"2": {"QAOA": 4, "Nov3rd": 9, "Nov3pair": 33, "Q": 2048, "T": 250}, "3": {"QAOA": 6, "Nov3rd": 12, "Nov3pair": 48, "Q": 3072, "T": 378}}
    return {"N": 15, "C": 5, "counts": rows, "ok": rows == expected}


def _centers() -> dict:
    C = 5
    x_s = x0_center4("Nov3rd", 3, 0, None)
    x_p = x0_pair_center4(C, 3, 0, None)
    x_q = x0_qaoa_center4(3, 0, None)
    shells = x_p[3:].reshape(3, C, 3)
    tiled = bool(np.allclose(shells, np.array([-0.5, 0.0, 0.0])[None, None, :]))
    return {
        "shared": [float(v) for v in x_s],
        "pair": [float(v) for v in x_p],
        "qaoa": [float(v) for v in x_q],
        "pair_tiles_shared_center": bool(tiled and float(x_p[0]) == 0.25 and np.all(x_p[1:3] == 0.0)),
        "shared_center": bool(float(x_s[0]) == 0.25 and np.all(x_s[3::3] == -0.5)),
        "qaoa_center": bool(np.array_equal(x_q, np.full(6, 0.25))),
        "method_ids": METHOD_ID["equal4"],
        "sigmas_pair_L3": int(sigmas_of("Nov3pair", 3, C=C).size),
    }


def _reduction() -> dict:
    n, seed, b, L = 6, 10018, 0.4, 2
    gs = np.ones(1 << n, dtype=np.float64)
    gs /= np.linalg.norm(gs)
    logical = Logical(n, seed, b, gs)
    C = len(packing(n))
    shared = np.array([0.2, -0.1, 0.15, 0.3, -0.2, 0.05, -0.4, 0.1, 0.0], dtype=np.float64)
    tiled = np.zeros(3 + 3 * C * L, dtype=np.float64)
    tiled[:3] = shared[:3]
    shells = tiled[3:].reshape(L, C, 3)
    shells[:] = shared[3:].reshape(L, 3)[:, None, :]
    psi_s = logical.nov3(shared, L)
    psi_p = logical.nov3_pair(tiled, L)
    fid = float(np.abs(np.vdot(psi_s, psi_p)) ** 2)
    de = abs(logical.energy(psi_s) - logical.energy(psi_p))
    xn = np.zeros(3 + 3 * C * L, dtype=np.float64)
    xn[0] = 0.25
    xn[3:].reshape(L, C, 3)[:, :, 0] = -0.5
    xq = np.full(2 * L, 0.25)
    a = logical.nov3_pair(xn, L)
    q = logical.qaoa(xq, L)
    fid_q = float(np.abs(np.vdot(a, q)) ** 2)
    de_q = abs(logical.energy(a) - logical.energy(q))
    return {
        "C": C,
        "shared_equals_tiled_pair": bool(fid > 1.0 - 1e-10 and de < 1e-10),
        "fidelity": fid,
        "dE": de,
        "pair_center_equals_qaoa": bool(fid_q > 1.0 - 1e-10 and de_q < 1e-10),
        "fidelity_qaoa": fid_q,
        "dE_qaoa": de_q,
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs():
        init = [None if s == 0 else np.random.default_rng(3000 + s) for s in range(4)]
        spsa = [np.random.default_rng(4000 + s) for s in range(4)]
        return init, spsa

    ri, rs = rngs()
    out = run_equal4(fake, "Nov3pair", 2, 128, ri, rs, x0_fn=lambda m, L, s, rng: x0_pair_center4(2, L, s, rng), C=2)
    return {
        "Q": 128,
        "calls": calls["n"],
        "ledger": out["queries"],
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
            and centers["pair_tiles_shared_center"]
            and centers["shared_center"]
            and centers["qaoa_center"]
            and centers["sigmas_pair_L3"] == 48
            and METHOD_ID["equal4"]["Nov3pair"] == 2
            and red["shared_equals_tiled_pair"]
            and red["pair_center_equals_qaoa"]
            and led["ok"]
            and packing(15) == [[a, a + 5, a + 10] for a in range(5)]
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2179072,
        "paper_V_M_not_implemented": True,
        "in_family": "per-cavity 3-shell SNAP; shared Uc",
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "params": params,
        "centers": centers,
        "reduction": red,
        "ledger_quadratic": led,
        "label_rule": "PAIR_SHELL_L3_EDGE iff L=3 pair-QAOA CI entirely <0 AND pair mean E <= shared mean E AND L=2 pair mean E <= shared AND L=2 pair-QAOA CI is not entirely >0; else REJECT_PAIR_SHELL",
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
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
