"""Classical PR106/107 interpretation audit. Zero optimization queries.

Does not touch SPSA, unread optimization seeds as fits, GR, validation, or final.
Re-propagating frozen selected x from existing cells is an oracle diagnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import x0_center4, x0_qaoa_center4
from Nov3_INIT.circuit import Logical
from Nov3_INIT.fock import (
    EPS_WL,
    FockNov3,
    _code_amps,
    givens_78,
    w_binh,
    w_nonideal,
)
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.sim import metrics_from_fock

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "wl" / "AUDIT.json"
CELLS = ROOT / "results" / "wl" / "cells"
INST = ROOT / "results" / "wl" / "instances"
HISTORICAL = 2818048
TOL = 1e-10


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def _logical(n=6, seed=10018, b=0.4):
    gs = np.ones(1 << n, dtype=np.float64)
    gs /= np.linalg.norm(gs)
    return Logical(n, seed, b, gs)


def _split(fock: FockNov3, psi: np.ndarray) -> dict:
    met = metrics_from_fock(psi, fock.tables)
    amp = _code_amps(psi, fock.tables)
    p_code = float(met["p_code"])
    e_uncond = float(met["E"])
    e_code_unnorm = float(fock.logical.energy(amp))
    if p_code > 1e-18:
        e_cond = float(fock.logical.energy(amp / np.sqrt(p_code)))
        f_ps = float(np.abs(np.vdot(fock.logical.gs, amp)) ** 2 / p_code)
    else:
        e_cond = None
        f_ps = None
    f_uncond = float(np.abs(np.vdot(fock.logical.gs, amp)) ** 2)
    return {
        "E_uncond": e_uncond,
        "E_cond_code": e_cond,
        "E_code_unnormalized": e_code_unnorm,
        "E_leak_block": float(e_uncond - e_code_unnorm),
        "p_code": p_code,
        "p_leak": float(met["p_leak"]),
        "F_uncond": f_uncond,
        "F_postselected_not_physical": f_ps,
        "norm": float(met["norm"]),
        "ok": bool(met["ok"]),
    }


def _nov3_vecs(L: int) -> list[tuple[str, np.ndarray]]:
    center = x0_center4("Nov3rd", L, 0, None)
    x = np.zeros(3 + 3 * L, dtype=np.float64)
    x[0], x[1], x[2] = 0.31, 0.11, -0.17
    for ell in range(L):
        x[3 + 3 * ell] = -0.42 + 0.08 * ell
        x[4 + 3 * ell] = 0.13 * (ell + 1)
        x[5 + 3 * ell] = -0.09 + 0.05 * ell
    return [("center0", center), ("offcenter", x)]


def _qaoa_vecs(L: int) -> list[tuple[str, np.ndarray]]:
    center = x0_qaoa_center4(L, 0, None)
    x = np.zeros(2 * L, dtype=np.float64)
    for ell in range(L):
        x[2 * ell] = 0.22 + 0.07 * ell
        x[2 * ell + 1] = 0.18 - 0.04 * ell
    return [("center0", center), ("offcenter", x)]


def identity_eps0() -> dict:
    logical = _logical()
    rows = []
    ok = True
    for L in (1, 2, 3):
        for order in ("rewrite", "hardware"):
            fock = FockNov3(logical, 10, 0.0, order)
            for name, x in _nov3_vecs(L):
                e_l = logical.energy(logical.nov3(x, L))
                psi_l = logical.nov3(x, L)
                e_f, psi_f = fock.propagate(x, L)
                amp = _code_amps(psi_f, fock.tables)
                fid = float(np.abs(np.vdot(psi_l, amp)) ** 2)
                dE = abs(e_f - e_l)
                leak = fock.leakage(psi_f)["p_leak"]
                row_ok = dE < TOL and abs(fid - 1.0) < 1e-12 and leak < 1e-12
                ok = ok and row_ok
                rows.append({
                    "kind": "Nov3", "L": L, "order": order, "vec": name,
                    "dE": dE, "state_fid": fid, "p_leak": leak, "ok": row_ok,
                })
            for name, x in _qaoa_vecs(L):
                e_l = logical.energy(logical.qaoa(x, L))
                psi_l = logical.qaoa(x, L)
                e_f, psi_f = fock.propagate_qaoa(x, L)
                amp = _code_amps(psi_f, fock.tables)
                fid = float(np.abs(np.vdot(psi_l, amp)) ** 2)
                dE = abs(e_f - e_l)
                leak = fock.leakage(psi_f)["p_leak"]
                row_ok = dE < TOL and abs(fid - 1.0) < 1e-12 and leak < 1e-12
                ok = ok and row_ok
                rows.append({
                    "kind": "QAOA", "L": L, "order": order, "vec": name,
                    "dE": dE, "state_fid": fid, "p_leak": leak, "ok": row_ok,
                })
    w = w_binh(10)
    herm = bool(np.allclose(w, w.T)) and bool(np.allclose(w @ w, np.eye(10)))
    wt_ok = herm
    return {
        "ok": bool(ok and wt_ok),
        "W_hermitian_involution": herm,
        "W_equals_Wdagger_at_eps0": herm,
        "parameter_order_nov3": "[gamma, theta_K, theta_M] + per layer (s1,s2,s3)",
        "parameter_order_qaoa": "per layer (gamma, beta); mixer SNAP s1=-2*beta",
        "initial_rewrite": "|+> then [UcZ W S W]^L",
        "initial_hardware": "|0> then [W UcZ W S]^L then final W",
        "SNAP_sign": "exp(-i * snap_level_phase); s1=-2 beta matches Rx(beta)=exp(-i beta X)",
        "rows": rows,
        "n_rows": len(rows),
        "n_fail": int(sum(not r["ok"] for r in rows)),
        "not_an_optimization_query": True,
    }


def hext() -> dict:
    logical = _logical()
    fock = FockNov3(logical, 10, EPS_WL, "hardware")
    x = _nov3_vecs(1)[1][1]
    _e, psi = fock.propagate(x, 1)
    split = _split(fock, psi)
    # Computational |8,0> on two cavities: leaked first cavity, vacuum second.
    basis = np.zeros(fock.tables.dim, dtype=np.complex128)
    tens = basis.reshape((10, 10))
    tens[8, 0] = 1.0
    leaked = _split(fock, basis)
    # Code computational |7,0> for contrast.
    code = np.zeros(fock.tables.dim, dtype=np.complex128)
    code.reshape((10, 10))[7, 0] = 1.0
    coded = _split(fock, code)
    # Selection uses unconditional E. Leakage cannot score 0 (H0 forbidden).
    penalty = bool(leaked["E_uncond"] > coded["E_uncond"] - 1e-9)
    return {
        "ok": bool(
            split["ok"]
            and leaked["p_code"] < 1e-15
            and coded["p_code"] > 1.0 - 1e-12
            and penalty
            and abs(split["E_uncond"] - split["E_code_unnormalized"] - split["E_leak_block"]) < 1e-12
        ),
        "objective": "unconditional <H^ext>, no postselection, no code renormalization",
        "fidelity": "unconditioned |<g|P^L psi>|^2; F_ps is diagnostic only",
        "D4": "H^ext leak blocks cannot go below E0; leaked pair |J|, leaked qubit +b",
        "on_offcenter_L1_eps010": split,
        "basis_leaked_8_0": leaked,
        "basis_code_7_0": coded,
        "leaked_basis_higher_energy_than_code_basis": penalty,
        "postselection_not_used_for_selection": True,
        "not_an_optimization_query": True,
    }


def frozen_cells() -> dict:
    rows = []
    if not CELLS.exists():
        return {"ok": False, "n": 0, "reason": "no cells"}
    for path in sorted(CELLS.glob("*.json")):
        rec = json.loads(path.read_text())
        if not rec.get("ok"):
            continue
        seed = int(rec["seed"])
        meta = json.loads((INST / f"N15_s{seed}.json").read_text())
        gs = np.load(INST / f"N15_s{seed}.npz")["gs"]
        logical = Logical(15, seed, float(meta["Bstar"]), gs)
        fock = FockNov3(logical, 10, float(rec["eps"]), rec["order"])
        x = np.asarray(rec["x"], dtype=np.float64)
        if rec["method"] == "QAOA":
            e, psi = fock.propagate_qaoa(x, 1)
        else:
            e, psi = fock.propagate(x, 1)
        split = _split(fock, psi)
        match = abs(e - float(rec["E"])) < 1e-10
        rows.append({
            "arm": rec["arm"],
            "seed": seed,
            "saved_E": float(rec["E"]),
            "replay_E_uncond": split["E_uncond"],
            "E_cond_code": split["E_cond_code"],
            "E_leak_block": split["E_leak_block"],
            "p_code": split["p_code"],
            "p_leak": split["p_leak"],
            "F_uncond": split["F_uncond"],
            "F_postselected_not_physical": split["F_postselected_not_physical"],
            "replay_matches_saved_E": match,
        })
    ok = bool(rows) and all(r["replay_matches_saved_E"] for r in rows)
    by_arm = {}
    for arm in ("nov3_ideal", "nov3_wl", "qaoa_wl"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        by_arm[arm] = {
            "n": len(sub),
            "mean_E_uncond": float(np.mean([r["replay_E_uncond"] for r in sub])),
            "mean_E_cond_code": float(np.mean([r["E_cond_code"] for r in sub])),
            "mean_p_code": float(np.mean([r["p_code"] for r in sub])),
            "mean_p_leak": float(np.mean([r["p_leak"] for r in sub])),
            "mean_E_leak_block": float(np.mean([r["E_leak_block"] for r in sub])),
        }
    return {
        "ok": ok,
        "n": len(rows),
        "by_arm": by_arm,
        "rows": rows,
        "oracle_diagnostic_not_selectable": True,
        "ledger_queries_added": 0,
        "not_an_optimization_query": True,
    }


def provenance() -> dict:
    g = givens_78(10, EPS_WL)
    protocol_has_givens = False
    protocol_has_eps = False
    protocol = (ROOT.parent / "Nov9_SK_SPSA" / "PROTOCOL.md").read_text()
    decisions = (ROOT.parent / "Nov9_SK_SPSA" / "DECISIONS.md").read_text()
    if "G_{7,8}" in protocol or "Givens" in protocol:
        protocol_has_givens = True
    if "0.10" in protocol and "Givens" in protocol:
        protocol_has_eps = True
    return {
        "edge_7_8_in_protocol_3_1": "The edge 7–8 leaves the code" in protocol,
        "protocol_specifies_givens_after_W": protocol_has_givens,
        "protocol_specifies_epsilon_0_10": protocol_has_eps,
        "d5_compiled_W_unmodelled": "compiled" in decisions.lower() and "W_L" in decisions,
        "eps": EPS_WL,
        "sin2": float(np.sin(EPS_WL) ** 2),
        "classification": "illustrative one-knob phenomenological sensitivity, not a specified physical error model, not hardware-calibrated",
        "ok_as_illustrative": True,
        "ok_as_compiled_W": False,
        "ok_as_hardware_calibrated": False,
        "g77": float(g[7, 7]),
        "g78": float(g[7, 8]),
        "g87": float(g[8, 7]),
        "w_nonideal_is_G_then_W": bool(np.allclose(w_nonideal(10, EPS_WL), givens_78(10, EPS_WL) @ w_binh(10))),
    }


def snapd() -> dict:
    grape = ROOT.parent / "explore_OCT29" / "scripts" / "gadget_Uc_grape_library.py"
    drivers = ROOT.parent / "explore_OCT29" / "lib" / "drivers.py"
    sim = ROOT.parent / "Nov9_SK_SPSA" / "experiments" / "FT" / "sim.py"
    return {
        "frozen_native_snap_D_compile_of_full_W_BinH_on_nf_ge_10": False,
        "missing": "no frozen SNAP-D (or D-SNAP-D) pulse sequence reproducing the 8x8 BinH W with a pad on NF>=10",
        "near_misses": [
            {
                "path": str(grape.relative_to(ROOT.parent)),
                "what": "fit_snap_D_to_target on cropped a, nf=8, used for U_X not W_BinH; D8 not a displacement (PROTOCOL 2.5)",
            },
            {
                "path": str(drivers.relative_to(ROOT.parent)),
                "what": "compile_flat_prep fits |0> to flat_8 only, not the full 8x8 Hadamard",
            },
            {
                "path": str(sim.relative_to(ROOT.parent)),
                "what": "module docstring: ideal W_L SNAP-displacement compilation and leakage are not modelled",
            },
        ],
        "pr106_identity_pad": "algebraic invariance of H^{⊗3} ⊕ I, not a physical pulse-compile cutoff test",
        "hardware_cutoff_convergence": "unvalidated",
        "ok": True,
    }


def main() -> None:
    ident = identity_eps0()
    hx = hext()
    cells = frozen_cells()
    prov = provenance()
    sd = snapd()
    payload = {
        "version": "N3INIT_wl_audit",
        "no_new_optimization_fits": True,
        "no_new_seeds": True,
        "no_invented_pulse_sequence": True,
        "no_gr_patch": True,
        "ledger_queries_added": 0,
        "historical_queries_preserved": HISTORICAL,
        "total_research_queries": HISTORICAL,
        "classical_conformance_not_in_ledger": True,
        "verdicts": {
            "1_provenance": "ILLUSTRATIVE_NOT_PHYSICAL_MODEL",
            "1_pass_as_compiled_W": False,
            "1_pass_as_hardware_calibrated": False,
            "2_eps0_native_vs_v4": "PASS" if ident["ok"] else "FAIL",
            "3_hext_unconditional_no_postselect": "PASS" if hx["ok"] else "FAIL",
            "4_secondary_equal_degradation": "FAIL_WORDING_THEN_CORRECTED",
            "5_frozen_snap_D_W_BinH": "ABSENT_UNVALIDATED",
        },
        "provenance": prov,
        "identity_eps0": ident,
        "hext": hx,
        "frozen_selected_replay": cells,
        "snapd": sd,
        "relabel": "PR107 numbers are a single phenomenological G_7,8(0.10) sensitivity test, not compiled-W or hardware validation. Do not sweep epsilon. Do not patch GR.",
    }
    payload["ok"] = bool(
        ident["ok"]
        and hx["ok"]
        and cells["ok"]
        and sd["ok"]
        and not prov["ok_as_compiled_W"]
        and not prov["ok_as_hardware_calibrated"]
        and payload["ledger_queries_added"] == 0
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    _dump(OUT, json.loads(json.dumps(payload, default=_plain)))
    print(json.dumps({k: payload[k] for k in ("ok", "verdicts", "ledger_queries_added", "historical_queries_preserved")}, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
