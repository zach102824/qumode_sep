"""Definition checks. No SPSA on a Hamiltonian, and no optimization seed is fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Nov3_INIT.alloc import iterations_equal4, iterations_halve, run_equal4, run_halve, x0_correlated, x0_equal4
from Nov3_INIT.bstar import published_check
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import build_instance
from Nov9_SK_SPSA.experiments.FT.sim import run_nov3, run_nov3_vacuum

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "CONFORMANCE.json"


def _check(name: str, ok: bool, detail: dict) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _per_site_h(psi: np.ndarray, q: int) -> np.ndarray:
    half = 1 << q
    rest = psi.size // (2 * half)
    x = np.array(psi, dtype=np.complex128, copy=True)
    a = x.reshape(rest, 2, half)
    a0 = a[:, 0, :].copy()
    a1 = a[:, 1, :].copy()
    s = 2.0 ** -0.5
    a[:, 0, :] = s * (a0 + a1)
    a[:, 1, :] = s * (a0 - a1)
    return x


def run() -> list[dict]:
    out = []
    pub = published_check()
    out.append(_check("Bstar_N9_seed10000", pub["abs_err"] <= 1e-5, pub))

    budgets = {128: (10, 24), 512: (58, 120), 1024: (122, 248), 384: (42, 88), 1536: (186, 376), 3072: (378, 760)}
    budget_ok = all(iterations_equal4(q) == pair[0] and iterations_halve(q) == pair[1] for q, pair in budgets.items())
    out.append(_check("iteration_counts", budget_ok, {str(k): list(v) for k, v in budgets.items()}))

    # Energy-only screen: fidelity is anti-aligned with energy and must not win.
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        e = float(np.sum(np.asarray(x) ** 2))
        return e, {"F": -e}

    def rngs(nstart):
        init, spsa = [], []
        for s in range(nstart):
            init.append(np.random.default_rng(1000 + s))
            spsa.append(np.random.default_rng(2000 + s))
        return init, spsa

    ri, rs = rngs(4)
    eq = run_equal4(fake, "QAOA", 1, 128, ri, rs)
    out.append(_check("equal4_ledger_128", eq["queries"]["total"] == 128 and calls["n"] == 128, eq["queries"]))
    n_after = calls["n"]
    ri, rs = rngs(8)
    hv = run_halve(fake, "Nov3rd", 1, 128, ri, rs)
    out.append(_check(
        "halve_ledger_128",
        hv["queries"]["total"] == 128 and hv["queries"]["screen"] == 8 and calls["n"] == n_after + 128,
        hv["queries"],
    ))
    # The quadratic is minimized near 0. Start 0 of halve Nov3 is not the zero vector
    # (s1=-0.5). Ranking must follow the recorded screen energies.
    order = list(np.argsort(hv["screen_energies"]))
    out.append(_check("screen_keeps_lowest_energy", hv["kept_start_ids"] == order[:2], {"kept": hv["kept_start_ids"], "order": order[:2]}))

    # Blind and foreign seeds are the runner's job. Here, reject the panels by the same predicate.
    def forbidden(seed: int) -> bool:
        return (11000 <= seed <= 11031) or (20000 <= seed <= 20999) or not (10018 <= seed <= 10025)

    out.append(_check(
        "seed_predicate",
        forbidden(11000) and forbidden(20000) and forbidden(10000) and (not forbidden(10018)) and (not forbidden(10025)),
        {},
    ))

    inst = build_instance(9, 10018, 0.4, role="n3init")
    logical = Logical(9, 10018, 0.4, inst.gs)
    # Global Walsh Nov3 matches the vacuum form on a non-trivial point.
    x = np.array([0.2, -0.1, 0.15, 0.3, -0.2, 0.4, -0.5, 0.1, 0.0], dtype=np.float64)
    unpacked = {"gamma": 0.2, "theta": {"K": -0.1, "M": 0.15}, "shells": x[3:].reshape(2, 3), "alphas": []}
    psi = logical.nov3(x, 2)
    vac = run_nov3_vacuum(inst, unpacked)
    plus = run_nov3(inst, unpacked)
    out.append(_check("vacuum_matches_plus", float(np.max(np.abs(psi - vac))) < 1e-8, {"max_abs_vac": float(np.max(np.abs(psi - vac))), "ft_E": plus["E"], "E": logical.energy(psi)}))

    # Algebraic reduction to QAOA: s1 = -2 beta, s2 = s3 = theta = 0, shared gamma.
    beta = np.array([0.2, -0.4])
    gamma = 0.3
    xn = np.zeros(3 + 6, dtype=np.float64)
    xn[0] = gamma
    xn[3] = -2 * beta[0]
    xn[6] = -2 * beta[1]
    xq = np.array([gamma, beta[0], gamma, beta[1]], dtype=np.float64)
    a = logical.nov3(xn, 2)
    b = logical.qaoa(xq, 2)
    fid = float(np.abs(np.vdot(a, b)) ** 2)
    de = abs(logical.energy(a) - logical.energy(b))
    out.append(_check("nov3_reduces_to_qaoa", fid > 1.0 - 1e-8 and de < 1e-8, {"fidelity": fid, "dE": de}))

    # Per-cavity Hadamards match the global Walsh on disjoint supports.
    psi0 = logical.plus.copy()
    for q in range(9):
        psi0 = _per_site_h(psi0, q)
    from Nov9_SK_SPSA.experiments.FT.sim import fwht
    d_h = float(np.max(np.abs(psi0 - fwht(logical.plus.copy()))))
    out.append(_check("fwht_is_all_qubit_hadamard", d_h < 1e-12, {"max_abs": d_h}))

    # Product energy.
    z = np.zeros(6, dtype=np.float64)
    z[0] = 0.0
    e_plus = logical.energy(logical.nov3(z, 1))
    out.append(_check("product_energy", abs(e_plus - (-0.4 * 9)) < 1e-8, {"E": e_plus, "E_plus": -3.6}))

    # Deterministic centers.
    c0 = x0_correlated("Nov3rd", 3, 0, None)
    e0 = x0_equal4("Nov3rd", 3, 0, None)
    out.append(_check(
        "centers",
        c0[0] == 0.25 and np.all(c0[3::3] == -0.5) and np.all(c0[4::3] == 0) and e0[0] == 0.25 and np.all(e0[1:] == 0),
        {},
    ))
    # Correlated layers share a base within the residual box.
    rng = np.random.default_rng(1)
    draw = x0_correlated("Nov3rd", 3, 1, rng)
    shells = draw[3:].reshape(3, 3)
    pair = np.max(np.abs(shells[:, None, :] - shells[None, :, :]))
    out.append(_check(
        "layer_residual_bound",
        bool(pair <= 0.30 + 1e-12 and np.all(np.abs(draw) <= 0.65 + 1e-12)),
        {"max_layer_gap": float(pair)},
    ))
    return out


def main() -> None:
    rows = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ok": all(r["ok"] for r in rows), "checks": rows}
    def _plain(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj))

    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    for row in rows:
        print(("PASS " if row["ok"] else "FAIL ") + row["name"])
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
