"""Bounded GR pilot: dev seeds 10000..10007, N=6 and 9, both fields, L=1,2,3.

Caps 128 and 1024 are separate SPSA runs. Fidelity is recorded after energy
selection and is not a stopping rule. Validation and final seeds are refused.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULT_DIR, assert_dev_seed, assert_pilot_N, load_prereg
from encoding import build_named_groups, feature_M, m_correlation, packing_from
from hamiltonian import build_instance, make_J
from model import (
    build_fock_model,
    evaluate_nov3,
    evaluate_nov9,
    evaluate_qaoa,
    metrics_from_eval,
    nov3_sigmas,
    nov3_x0,
    nov9_alpha_index,
    nov9_sigmas,
    nov9_x0,
    qaoa_sigmas,
    qaoa_x0,
)
from spsa import run_spsa

RUN_DIR = RESULT_DIR / "runs"


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _clean(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _clean(obj.tolist())
    return obj


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(doc), indent=2) + "\n")
    tmp.replace(path)


def _job_path(job: dict) -> Path:
    btag = int(round(10 * float(job["b"])))
    return RUN_DIR / f"n{job['N']}_s{job['seed']}_b{btag}_L{job['L']}_{job['arm']}_c{job['cap']}.json"


def _rng(seed, n, b, L, method_id, start_id, stream):
    import numpy as np

    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(n), int(round(10 * float(b))), int(L), int(method_id), int(start_id), int(stream)])
    )


def _starts(family: str, L: int, seed: int, n: int, b: float, method_id: int):
    x0s = []
    rngs = []
    for s in range(4):
        rngs.append(_rng(seed, n, b, L, method_id, s, 0))
        if s == 0:
            kind_rng = None
            kind = "E"
        else:
            kind_rng = _rng(seed, n, b, L, method_id, s, 2)
            kind = "R"
        if family == "nov9":
            x0s.append(nov9_x0(L, kind, kind_rng))
        elif family == "nov3":
            x0s.append(nov3_x0(L, kind, kind_rng))
        elif family == "qaoa":
            x0s.append(qaoa_x0(L, kind, kind_rng))
        else:
            raise ValueError(family)
    return x0s, rngs


def _evaluate_factory(job, inst, named, nf):
    arm = job["arm_spec"]
    family = arm["family"]
    L = int(job["L"])
    if family == "nov9":
        groups, ops = named[arm["rule"]]
        pk = packing_from(arm["name"], groups, arm["encoding"], ops)
        model = build_fock_model(inst, pk, int(nf), arm["ucz"], arm["gate"] == "strict8")
        dM = None

        def evaluate(x, _model=model, _L=L):
            return evaluate_nov9(_model, x, _L, keep_state=False)

        sigmas = nov9_sigmas(L)
        alpha_index = nov9_alpha_index(L)
        meta = {
            "encoding": arm["encoding"],
            "ucz": arm["ucz"],
            "gate": model.gate_name,
            "nf": model.nf,
            "groups": pk.groups,
            "classical_ops": pk.ops,
            "n_controls": 3 + 4 * L,
        }
        return evaluate, sigmas, alpha_index, meta
    if family == "nov3":
        groups, ops = named["inter_id"]
        _raw, _hat, _scale, dM = feature_M(inst.J, groups, inst.rms_J, inst.z)

        def evaluate(x, _inst=inst, _groups=groups, _dM=dM, _L=L):
            return evaluate_nov3(_inst, _groups, _dM, x, _L, keep_state=False)

        return evaluate, nov3_sigmas(L), None, {
            "encoding": "binary",
            "ucz": "code",
            "gate": "idealW",
            "nf": 8,
            "groups": groups,
            "classical_ops": ops,
            "n_controls": 3 + 3 * L,
        }
    if family == "qaoa":

        def evaluate(x, _inst=inst, _L=L):
            return evaluate_qaoa(_inst, x, _L, keep_state=False)

        return evaluate, qaoa_sigmas(L), None, {
            "encoding": "qubit",
            "ucz": "none",
            "gate": "none",
            "nf": None,
            "groups": None,
            "classical_ops": {"gs_reads": 0, "pair_reads": 0},
            "n_controls": 2 * L,
        }
    raise ValueError(family)


_INST = {}
_NAMED = {}


def _inst(seed: int, n: int, b: float):
    key = (int(seed), int(n), float(b))
    if key not in _INST:
        _INST[key] = build_instance(int(seed), int(n), float(b), pilot=True)
    return _INST[key]


def _named(seed: int, n: int, J):
    key = (int(seed), int(n))
    if key not in _NAMED:
        _NAMED[key] = build_named_groups(J, int(seed))
    return _NAMED[key]


def run_one(job: dict) -> dict:
    import numpy as np

    t0 = time.perf_counter()
    pre = load_prereg()
    assert_dev_seed(int(job["seed"]), pilot=True)
    assert_pilot_N(int(job["N"]))
    inst = _inst(int(job["seed"]), int(job["N"]), float(job["b"]))
    named = _named(int(job["seed"]), int(job["N"]), inst.J)
    evaluate, sigmas, alpha_index, meta = _evaluate_factory(job, inst, named, int(job["nf"]))
    x0s, rngs = _starts(job["arm_spec"]["family"], int(job["L"]), int(job["seed"]), int(job["N"]), float(job["b"]), int(job["arm_spec"]["method_id"]))
    init_rows = []
    for s, x0 in enumerate(x0s):
        ev = evaluate(x0)
        row = metrics_from_eval(inst, ev)
        row["start_id"] = s
        row["oracle"] = True
        row["counted_query"] = False
        init_rows.append(row)
    best_init = min(init_rows, key=lambda r: (r["E"], r["start_id"]))
    result = run_spsa(
        evaluate,
        x0s,
        np.asarray(sigmas, dtype=np.float64),
        int(job["cap"]),
        rngs,
        c=float(job["c"]),
        alpha_index=alpha_index,
    )
    selected = metrics_from_eval(inst, result.payload)
    doc = {
        "status": "fail" if result.failed else "ok",
        "failure": result.failure,
        "simulation_assumptions": pre["approval"],
        "hardware_claim": False,
        "shot_claim": False,
        "freeze": False,
        "validation_touched": False,
        "final_touched": False,
        "fidelity_role": "evaluation_only_unconditioned_no_postselection",
        "seed": int(job["seed"]),
        "N": int(job["N"]),
        "b": float(job["b"]),
        "L": int(job["L"]),
        "cap": int(job["cap"]),
        "arm": job["arm"],
        "method_id": int(job["arm_spec"]["method_id"]),
        "family": job["arm_spec"]["family"],
        **meta,
        "queries": result.counter.as_dict(),
        "n_iter_per_start": result.n_iter,
        "selected_start": result.start_id,
        "selected_candidate": result.candidate,
        "a_values": result.a_values,
        "selected": selected,
        "init_start_E_oracle": init_rows[0],
        "init_best_energy_oracle": best_init,
        "x": result.x.tolist(),
        "j_upper_sha": inst.j_upper_sha,
        "gs_sha": inst.gs_sha,
        "prereg_sha256": pre["_sha256"],
        "c_spsa": float(job["c"]),
        "seconds": time.perf_counter() - t0,
        "measurement": {
            "shots": 0,
            "settings_if_shots": 2,
            "W_L": "ideal BinH on 0..7, identity outside, compile and leakage NOT modelled",
        },
    }
    return doc


def _worker(job: dict) -> str:
    path = _job_path(job)
    pre_sha = job["prereg_sha256"]
    if path.exists():
        try:
            old = json.loads(path.read_text())
            if old.get("status") == "ok" and old.get("prereg_sha256") == pre_sha and int(old.get("queries", {}).get("total", -1)) == int(job["cap"]):
                return f"skip {path.name}"
        except Exception:
            pass
    try:
        doc = run_one(job)
        _write(path, doc)
        sel = doc["selected"]
        return f"{path.name} E={sel['E']:.4f} F={sel['F']:.4f} leak={sel['p_leak']:.4f} q={doc['queries']['total']} {doc['seconds']:.1f}s"
    except Exception as exc:
        _write(
            path,
            {
                "status": "error",
                "failure": traceback.format_exc(),
                "seed": job["seed"],
                "N": job["N"],
                "b": job["b"],
                "L": job["L"],
                "cap": job["cap"],
                "arm": job["arm"],
                "hardware_claim": False,
                "freeze": False,
            },
        )
        return f"ERROR {path.name} {exc}"


def build_jobs(nf: int, c: float, pre: dict) -> list:
    jobs = []
    arms = {a["name"]: a for a in pre["arms"]}
    for n in pre["pilot"]["N"]:
        for seed in range(int(pre["pilot"]["seeds"][0]), int(pre["pilot"]["seeds"][1]) + 1):
            for b in pre["fields"]:
                for L in pre["pilot"]["L"]:
                    for cap in pre["pilot"]["caps"]:
                        for name, spec in arms.items():
                            jobs.append(
                                {
                                    "seed": int(seed),
                                    "N": int(n),
                                    "b": float(b),
                                    "L": int(L),
                                    "cap": int(cap),
                                    "arm": name,
                                    "arm_spec": spec,
                                    "nf": int(nf),
                                    "c": float(c),
                                    "prereg_sha256": pre["_sha256"],
                                }
                            )
    return jobs


def write_instance_table(pre: dict) -> None:
    rows = []
    for n in pre["pilot"]["N"]:
        for seed in range(int(pre["pilot"]["seeds"][0]), int(pre["pilot"]["seeds"][1]) + 1):
            J = make_J(int(seed), int(n))
            named = build_named_groups(J, int(seed))
            per_b = {}
            sha = None
            for b in pre["fields"]:
                inst = build_instance(int(seed), int(n), float(b), pilot=True)
                sha = inst.j_upper_sha
                per_b[str(b)] = {
                    "E0": inst.E0,
                    "E1": inst.E1,
                    "gap": inst.gap,
                    "E0_odd": inst.E0_odd,
                    "delta_odd": inst.delta_odd,
                    "lambda_max": inst.lambda_max,
                    "F_plus": inst.F_plus,
                    "residual": inst.residual,
                    "gs_sha": inst.gs_sha,
                    "gs_min": inst.gs_min,
                    "gs_parity": inst.gs_parity,
                }
            corrs = {}
            ops = {}
            for rule, (groups, op) in named.items():
                inst = build_instance(int(seed), int(n), 1.0, pilot=True)
                corrs[rule] = m_correlation(inst.J, groups, inst.rms_J, inst.z, inst.hc)
                ops[rule] = op
            rows.append({"N": int(n), "seed": int(seed), "j_upper_sha": sha, "fields": per_b, "pearson_M_vs_Hzz": corrs, "classical_ops": ops})
    _write(RESULT_DIR / "instance_diagnostics.json", {"hardware_claim": False, "freeze": False, "rows": rows})


def postcheck() -> dict:
    """Recompute every selected true-D point at NF=48. Not an extra SPSA query."""
    import numpy as np

    pre = load_prereg()
    cutoff = json.loads((RESULT_DIR / "cutoff.json").read_text())
    tol = float(pre["cutoff"]["tol"])
    nf = int(cutoff["nf_sim"])
    rows = []
    worst = {"E": 0.0, "F": 0.0, "p_leak": 0.0, "outside": 0.0}
    n_fail = 0
    for path in sorted(RUN_DIR.glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("status") != "ok":
            continue
        if doc.get("gate") != "D_true_expm_truncated_generator":
            continue
        try:
            inst = _inst(doc["seed"], doc["N"], doc["b"])
            named = _named(doc["seed"], doc["N"], inst.J)
            spec = next(a for a in pre["arms"] if a["name"] == doc["arm"])
            job = {"arm": doc["arm"], "arm_spec": spec, "L": doc["L"], "N": doc["N"], "seed": doc["seed"], "b": doc["b"]}
            ev_nf, _s, _a, _m = _evaluate_factory(job, inst, named, nf)
            ev_48, _s, _a, _m = _evaluate_factory(job, inst, named, 48)
            x = np.asarray(doc["x"], dtype=np.float64)
            a = ev_nf(x)
            b = ev_48(x)
            dE = abs(a["E"] - b["E"])
            dF = abs(a["F"] - b["F"])
            dp = abs(a["p_leak"] - b["p_leak"])
            psi = evaluate_nov9(
                build_fock_model(
                    inst,
                    packing_from(spec["name"], named[spec["rule"]][0], spec["encoding"], named[spec["rule"]][1]),
                    48,
                    spec["ucz"],
                    False,
                ),
                x,
                int(doc["L"]),
                keep_state=True,
            )["psi"]
            outside = float(1.0 - np.sum(np.abs(psi[(slice(0, nf),) * psi.ndim]) ** 2))
            stored_gap = max(abs(a["E"] - doc["selected"]["E"]), abs(a["F"] - doc["selected"]["F"]))
            ok = max(dE, dF, dp, outside) <= tol and stored_gap <= 1e-8
            if not ok:
                n_fail += 1
            worst["E"] = max(worst["E"], dE)
            worst["F"] = max(worst["F"], dF)
            worst["p_leak"] = max(worst["p_leak"], dp)
            worst["outside"] = max(worst["outside"], outside)
            rows.append(
                {
                    "file": path.name,
                    "dE": dE,
                    "dF": dF,
                    "dp_leak": dp,
                    "outside": outside,
                    "stored_gap": stored_gap,
                    "pass": bool(ok),
                }
            )
        except Exception as exc:
            n_fail += 1
            rows.append({"file": path.name, "pass": False, "error": str(exc)})
    summary = {
        "tol": tol,
        "nf_sim": nf,
        "n": len(rows),
        "n_fail": n_fail,
        "worst": worst,
        "hardware_claim": False,
        "freeze": False,
    }
    _write(RESULT_DIR / "postcheck.json", summary)
    _write(RESULT_DIR / "postcheck_rows.json", {"rows": rows})
    print("postcheck", json.dumps(summary), flush=True)
    return summary


def main():
    import multiprocessing as mp

    pre = load_prereg()
    if not pre["d2_d5_user_confirmed"] or pre["hardware_claim"] or pre["freeze"]:
        raise SystemExit("prereg flags are inconsistent with the approved simulation benchmark")
    cutoff_path = RESULT_DIR / "cutoff.json"
    if not cutoff_path.exists():
        raise SystemExit("cutoff.json is missing; run cutoff.py before the pilot")
    cutoff = json.loads(cutoff_path.read_text())
    if cutoff["prereg_sha256"] != pre["_sha256"]:
        raise SystemExit("cutoff.json was built from a different prereg hash; rerun cutoff.py")
    nf = int(cutoff["nf_sim"])
    c = float(cutoff["c_spsa"])
    print(f"pilot nf={nf} c={c} approval={pre['approval']['when']}", flush=True)
    write_instance_table(pre)
    jobs = build_jobs(nf, c, pre)
    print(f"jobs {len(jobs)}", flush=True)
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(processes=4) as pool:
        for line in pool.imap_unordered(_worker, jobs, chunksize=1):
            done += 1
            print(f"[{done}/{len(jobs)}] {line}", flush=True)
    print("pilot dispatch complete", flush=True)
    postcheck()


if __name__ == "__main__":
    main()
