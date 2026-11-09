"""Bounded accounting and one-state audit. Does not refit and does not open blind seeds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULT_DIR
from encoding import feature_M, packing_from
from gates import displacement_matrix, flip_level, snap_diagonal
from hamiltonian import build_instance, logical_energy, plus_state
from model import build_fock_model, evaluate_nov3, evaluate_nov9

RUN_DIR = RESULT_DIR / "runs"
CODE = 8


def scan_runs() -> dict:
    caps = {128: 0, 1024: 0}
    init_present = 0
    init_start_ids = {0: 0, 1: 0, 2: 0, 3: 0}
    bad_counter = []
    missing_init = []
    for path in sorted(RUN_DIR.glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("status") != "ok":
            bad_counter.append(path.name)
            continue
        cap = int(doc["cap"])
        caps[cap] = caps.get(cap, 0) + 1
        q = doc["queries"]
        if int(q["total"]) != cap or int(q["monitoring"]) != 0 or int(q["calibration"]) + int(q["iteration"]) + int(q["candidate"]) != cap:
            bad_counter.append(path.name)
        if "init_start_E_oracle" not in doc or "init_best_energy_oracle" not in doc:
            missing_init.append(path.name)
        else:
            init_present += 1
            sid = int(doc["init_best_energy_oracle"]["start_id"])
            init_start_ids[sid] = init_start_ids.get(sid, 0) + 1
    n = sum(caps.values())
    return {
        "n_ok": n,
        "by_cap": caps,
        "init_oracle_records": init_present,
        "init_best_start_id_counts": init_start_ids,
        "counter_mismatches": bad_counter,
        "missing_init_record": missing_init,
        "spsa_cap_queries": 128 * caps.get(128, 0) + 1024 * caps.get(1024, 0),
        "initial_energy_oracle_calls": 4 * init_present,
        "full_energy_evaluations": 128 * caps.get(128, 0) + 1024 * caps.get(1024, 0) + 4 * init_present,
        "per_run_full_if_cap_1024": 1028,
        "per_run_full_if_cap_128": 132,
    }


def nov3_panel() -> dict:
    rows = []
    for seed in range(10000, 10008):
        path = RUN_DIR / f"n9_s{seed}_b10_L1_Nov3rd_c1024.json"
        doc = json.loads(path.read_text())
        inst = build_instance(seed, 9, 1.0, pilot=True)
        _raw, _hat, _scale, dM = feature_M(inst.J, doc["groups"], inst.rms_J, inst.z)
        ev = evaluate_nov3(inst, doc["groups"], dM, np.asarray(doc["x"], dtype=np.float64), 1)
        rows.append(
            {
                "seed": seed,
                "stored_F": doc["selected"]["F"],
                "stored_E": doc["selected"]["E"],
                "recomputed_F": ev["F"],
                "recomputed_E": ev["E"],
                "F_plus": doc["selected"]["F_plus"],
                "selected_start": doc["selected_start"],
                "selected_candidate": doc["selected_candidate"],
                "spsa_queries": doc["queries"]["total"],
                "j_upper_sha": doc["j_upper_sha"],
            }
        )
    fs = np.array([r["stored_F"] for r in rows])
    order = np.argsort(fs)
    return {
        "rows": rows,
        "sorted_F": fs[order].tolist(),
        "median_mean_of_two_middle": float(np.median(fs)),
        "two_middle_F": [float(fs[order][3]), float(fs[order][4])],
        "max_abs_recompute": float(max(abs(r["stored_F"] - r["recomputed_F"]) for r in rows)),
    }


def independent_observables(psi: np.ndarray, inst, groups) -> dict:
    """Energy and fidelity from the state, not from model._observe."""
    nf = psi.shape[0]
    C = psi.ndim
    logical = np.zeros(psi.shape, dtype=np.int64)
    code = np.ones(psi.shape, dtype=bool)
    for c, cav in enumerate(groups):
        f = np.arange(nf).reshape((nf,) + (1,) * (C - 1))
        f = np.moveaxis(f, 0, c)
        code &= f < CODE
        for r, site in enumerate(cav):
            bit = ((f >> r) & 1).astype(np.int64)
            logical |= bit << int(site)
    amp = np.zeros(1 << inst.n, dtype=np.complex128)
    mask = code
    np.add.at(amp, logical[mask], psi[mask])
    F = float(np.abs(np.vdot(inst.gs, amp)) ** 2)
    prob = (psi.real * psi.real + psi.imag * psi.imag)
    # Pair energy on the full Fock grid, written here.
    where = {}
    for c, cav in enumerate(groups):
        for r, site in enumerate(cav):
            where[int(site)] = (c, r)
    grids = np.meshgrid(*[np.arange(nf) for _ in range(C)], indexing="ij")
    leaked = [grids[c] >= CODE for c in range(C)]
    z = {}
    for c, cav in enumerate(groups):
        for r, site in enumerate(cav):
            z[int(site)] = 1.0 - 2.0 * ((grids[c] >> r) & 1)
    diag = np.zeros(psi.shape, dtype=np.float64)
    for i in range(inst.n):
        for j in range(i + 1, inst.n):
            ci, _ri = where[i]
            cj, _rj = where[j]
            both = ~(leaked[ci] | leaked[cj])
            zz = float(inst.J[i, j]) * z[i] * z[j]
            diag = diag + np.where(both, zz, abs(float(inst.J[i, j])))
    nleak = np.zeros(psi.shape, dtype=np.float64)
    for c in range(C):
        nleak = nleak + 3.0 * leaked[c]
    diag = diag + float(inst.b) * nleak
    e_diag = float(np.sum(diag * prob))
    xsum = 0.0
    for c, cav in enumerate(groups):
        p = np.moveaxis(psi, c, 0).reshape(nf, -1)
        for r, _site in enumerate(cav):
            acc = 0.0
            for f in range(CODE):
                if ((f >> r) & 1) == 0:
                    f2 = flip_level(f, r, "binary")
                    acc += 2.0 * float(np.real(np.dot(np.conj(p[f]), p[f2])))
            xsum += acc
    E = e_diag - float(inst.b) * xsum
    p_code = float(np.sum(np.abs(psi[(slice(0, CODE),) * C]) ** 2))
    return {"E": E, "F": F, "p_leak": 1.0 - p_code, "E_diag": e_diag, "norm": float(prob.sum())}


def propagate_inline(inst, groups, x, nf: int) -> np.ndarray:
    """Same gate order as Nov9th, observables not taken from _observe."""
    gamma, thK, thM = float(x[0]), float(x[1]), float(x[2])
    u = float(x[3])
    alpha = 0.3 * np.tanh(u)
    s1, s2, s3 = float(x[4]), float(x[5]), float(x[6])
    model = build_fock_model(inst, packing_from("audit", groups, "binary", {}), nf, "Uz", False)
    ang = gamma * (model.dJ_phase + thK * model.dK_phase + thM * model.dM_phase)
    C = len(groups)
    psi = np.zeros((nf,) * C, dtype=np.complex128)
    psi[(slice(0, CODE),) * C] = 8.0 ** (-0.5 * C)
    psi = psi * np.exp(-1j * ang)
    U = displacement_matrix(float(alpha), nf, strict8=False)
    for c in range(C):
        p = np.moveaxis(psi, c, 0)
        sh = p.shape
        p = U @ p.reshape(nf, -1)
        psi = np.moveaxis(p.reshape(sh), 0, c)
    sph = snap_diagonal(nf, s1, s2, s3, "binary")
    for c in range(C):
        shape = [1] * C
        shape[c] = nf
        psi = psi * sph.reshape(shape)
    return psi


def reconstruct_one() -> dict:
    path = RUN_DIR / "n9_s10000_b10_L1_inter_id_c1024.json"
    doc = json.loads(path.read_text())
    inst = build_instance(10000, 9, 1.0, pilot=True)
    x = np.asarray(doc["x"], dtype=np.float64)
    psi = propagate_inline(inst, doc["groups"], x, int(doc["nf"]))
    ind = independent_observables(psi, inst, doc["groups"])
    prod = evaluate_nov9(
        build_fock_model(inst, packing_from("audit", doc["groups"], "binary", {}), int(doc["nf"]), "Uz", False),
        x,
        1,
    )
    plus_E = logical_energy(inst, plus_state(inst.n))
    return {
        "file": path.name,
        "seed": 10000,
        "N": 9,
        "b": 1.0,
        "L": 1,
        "arm": "inter_id",
        "spsa_counter": doc["queries"],
        "stored": {"E": doc["selected"]["E"], "F": doc["selected"]["F"], "F_plus": doc["selected"]["F_plus"], "p_leak": doc["selected"]["p_leak"]},
        "production_reeval": {"E": prod["E"], "F": prod["F"], "p_leak": prod["p_leak"]},
        "independent": ind,
        "plus_energy": plus_E,
        "abs_E_minus_plus": abs(ind["E"] - plus_E),
        "abs_F_minus_Fplus": abs(ind["F"] - doc["selected"]["F_plus"]),
        "abs_independent_minus_stored_E": abs(ind["E"] - doc["selected"]["E"]),
        "abs_independent_minus_stored_F": abs(ind["F"] - doc["selected"]["F"]),
    }


def main():
    # feature_M is imported from model via the name used in run_pilot; confirm.
    out = {"scan": scan_runs(), "nov3_n9_b1_L1_cap1024": nov3_panel(), "reconstruction": reconstruct_one()}
    path = RESULT_DIR / "audit_correction.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] if k != "nov3_n9_b1_L1_cap1024" else {"median": out[k]["median_mean_of_two_middle"], "sorted_F": out[k]["sorted_F"], "recompute": out[k]["max_abs_recompute"]} for k in out}, indent=2))


if __name__ == "__main__":
    main()
