"""Freeze NF_sim and the SPSA step c before any dev optimization.

The ladder is the preregistered stress grid, not the optimized parameters.
Optimized points are checked again after the pilot.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ALPHA_MAX, RESULT_DIR, load_prereg
from encoding import build_named_groups, packing_from
from gates import alpha_from_u, u_from_alpha
from hamiltonian import build_instance, make_J, pair_K, hat, rms, upper, pair_diag, z_table, LogicalInstance
from model import build_fock_model, evaluate_nov9, nov9_x0


def _params(point: dict, L: int) -> np.ndarray:
    x = np.zeros(3 + 4 * L, dtype=np.float64)
    x[0] = float(point["gamma"])
    x[1] = float(point["theta"])
    x[2] = float(point["theta"])
    mag = float(point["alpha_mag"])
    for ell in range(L):
        if point["alpha_mode"] == "echo":
            alpha = ((-1.0) ** ell) * mag
        elif point["alpha_mode"] == "aligned":
            alpha = mag
        else:
            raise ValueError(point["alpha_mode"])
        x[3 + ell] = u_from_alpha(alpha)
    x[3 + L :] = float(point["s"])
    return x


def _packing(named: dict, key: str):
    spec = {
        "inter_id_Uz": ("inter_id", "binary", "Uz", False),
        "G_R_Uz": ("G_R", "binary", "Uz", False),
        "G_R_U0": ("G_R", "binary", "U0", False),
        "Gray_G_R_Uz": ("G_R", "gray", "Uz", False),
    }[key]
    rule, enc, ucz, strict = spec
    groups, ops = named[rule]
    return packing_from(key, groups, enc, ops), ucz, strict


def _outside(psi: np.ndarray, nf_cut: int) -> float:
    C = psi.ndim
    sl = (slice(0, nf_cut),) * C
    return float(1.0 - np.sum(np.abs(psi[sl]) ** 2))


def _dummy_instance(seed: int, n: int, b: float) -> LogicalInstance:
    """Couplings only, for a timing probe. The vector is not a ground state."""
    from hamiltonian import sha256_f64

    J = make_J(seed, n)
    z = z_table(n)
    hc = pair_diag(J, z)
    K_raw = pair_K(J)
    rms_J = rms(upper(J))
    K_hat, scale_K = hat(K_raw, rms_J)
    gs = np.zeros(1 << n, dtype=np.float64)
    gs[0] = 1.0
    return LogicalInstance(
        n=n,
        seed=seed,
        b=float(b),
        J=J,
        z=z,
        hc=hc,
        K_raw=K_raw,
        K_hat=K_hat,
        scale_K=float(scale_K),
        rms_J=float(rms_J),
        E0=float("nan"),
        E1=float("nan"),
        E0_odd=float("nan"),
        lambda_max=float("nan"),
        gap=float("nan"),
        delta_odd=float("nan"),
        gs=gs,
        F_plus=float("nan"),
        j_upper_sha=sha256_f64(upper(J)),
        gs_sha="timing-not-a-ground-state",
        gs_min=0.0,
        gs_parity=float("nan"),
        residual=float("nan"),
    )


def run_cutoff() -> dict:
    pre = load_prereg()
    grid = pre["cutoff"]
    ladder = list(grid["ladder"])
    ref = ladder[-1]
    tol = float(grid["tol"])
    rows = []
    worst = {nf: {"E": 0.0, "F": 0.0, "p_leak": 0.0, "outside": 0.0} for nf in ladder if nf != ref}
    t0 = time.perf_counter()
    cache_named = {}
    cache_inst = {}
    for seed in grid["seeds"]:
        for n in grid["N"]:
            named = build_named_groups(make_J(int(seed), int(n)), int(seed))
            cache_named[(seed, n)] = named
            for b in (1.0, 1.5):
                cache_inst[(seed, n, b)] = build_instance(int(seed), int(n), float(b), pilot=True)
    for seed in grid["seeds"]:
        for n in grid["N"]:
            named = cache_named[(seed, n)]
            for L in (1, 2, 3):
                for point in grid["points"]:
                    x = _params(point, L)
                    for enc in grid["encodings"]:
                        pk, ucz, strict = _packing(named, enc)
                        print(f"cutoff seed={seed} N={n} L={L} {point['name']} {enc}", flush=True)
                        ref_out = {}
                        ref_psi = None
                        for b in (1.0, 1.5):
                            inst = cache_inst[(seed, n, b)]
                            model = build_fock_model(inst, pk, ref, ucz, strict)
                            ev = evaluate_nov9(model, x, L, keep_state=True)
                            ref_out[b] = ev
                            ref_psi = ev["psi"]
                        for nf in ladder:
                            if nf == ref:
                                continue
                            metrics = {"E": 0.0, "F": 0.0, "p_leak": 0.0}
                            for b in (1.0, 1.5):
                                inst = cache_inst[(seed, n, b)]
                                model = build_fock_model(inst, pk, nf, ucz, strict)
                                ev = evaluate_nov9(model, x, L, keep_state=False)
                                metrics["E"] = max(metrics["E"], abs(ev["E"] - ref_out[b]["E"]))
                                metrics["F"] = max(metrics["F"], abs(ev["F"] - ref_out[b]["F"]))
                                metrics["p_leak"] = max(metrics["p_leak"], abs(ev["p_leak"] - ref_out[b]["p_leak"]))
                            outside = _outside(ref_psi, nf)
                            worst[nf]["E"] = max(worst[nf]["E"], metrics["E"])
                            worst[nf]["F"] = max(worst[nf]["F"], metrics["F"])
                            worst[nf]["p_leak"] = max(worst[nf]["p_leak"], metrics["p_leak"])
                            worst[nf]["outside"] = max(worst[nf]["outside"], outside)
                            rows.append(
                                {
                                    "seed": seed,
                                    "N": n,
                                    "L": L,
                                    "point": point["name"],
                                    "encoding": enc,
                                    "nf": nf,
                                    "dE": metrics["E"],
                                    "dF": metrics["F"],
                                    "dp_leak": metrics["p_leak"],
                                    "outside": outside,
                                }
                            )
                        del ref_psi
    chosen = None
    for nf in ladder:
        if nf == ref:
            chosen = nf
            break
        w = worst[nf]
        if max(w["E"], w["F"], w["p_leak"], w["outside"]) <= tol:
            chosen = nf
            break
    # leak-step probe at the chosen cutoff, start-E init, b=1 (state-independent)
    c = float(pre["spsa"]["c"])
    sigma_a = float(pre["spsa"]["sigma"]["alpha"])
    probe = []
    for _half in range(int(pre["spsa"]["alpha_leak_probe"]["halves_allowed_before_pilot"]) + 1):
        probe = []
        worst_dp = 0.0
        for seed in range(10000, 10008):
            for n in (6, 9):
                inst = build_instance(seed, n, 1.0, pilot=True)
                named = build_named_groups(inst.J, seed)
                for rule in ("inter_id", "G_R"):
                    pk = packing_from(rule, named[rule][0], "binary", named[rule][1])
                    model = build_fock_model(inst, pk, chosen, "Uz", False)
                    for L in (1, 2, 3):
                        x = nov9_x0(L, "E", None)
                        base = evaluate_nov9(model, x, L)["p_leak"]
                        alpha = alpha_from_u(float(x[3])) + c * sigma_a
                        x2 = x.copy()
                        x2[3] = u_from_alpha(alpha)
                        shifted = evaluate_nov9(model, x2, L)["p_leak"]
                        dp = abs(shifted - base)
                        worst_dp = max(worst_dp, dp)
                        probe.append({"seed": seed, "N": n, "rule": rule, "L": L, "dp_leak": dp, "base": base})
        if worst_dp <= float(pre["spsa"]["alpha_leak_probe"]["max_abs_dp_leak"]):
            break
        c *= 0.5
    # feasibility timings. N=15 has no stored ground state.
    timings = []
    for n, nf, use_gs in ((12, 16, True), (12, 24, True), (15, 16, False)):
        if use_gs:
            inst = build_instance(10000, n, 1.0, pilot=False)
        else:
            inst = _dummy_instance(10000, n, 1.0)
        named = build_named_groups(inst.J, 10000)
        pk = packing_from("inter", named["inter_id"][0], "binary", named["inter_id"][1])
        t_b = time.perf_counter()
        model = build_fock_model(inst, pk, nf, "Uz", False)
        build_s = time.perf_counter() - t_b
        x = nov9_x0(2, "E", None)
        t_e = time.perf_counter()
        reps = 3 if n <= 12 else 1
        for _ in range(reps):
            evaluate_nov9(model, x, 2, keep_state=False)
        eval_s = (time.perf_counter() - t_e) / reps
        timings.append(
            {
                "N": n,
                "nf": nf,
                "L": 2,
                "build_s": build_s,
                "eval_s": eval_s,
                "projected_1024_s": eval_s * 1024,
                "ground_state": use_gs,
                "note": "timing only; not an optimized result and not a fidelity claim",
            }
        )
        print(f"timing N={n} nf={nf} eval {eval_s:.4f}s proj1024 {eval_s*1024:.1f}s", flush=True)
    doc = {
        "prereg_sha256": pre["_sha256"],
        "tol": tol,
        "ladder": ladder,
        "worst": worst,
        "nf_sim": int(chosen),
        "c_spsa": c,
        "alpha_probe_worst_dp": max(p["dp_leak"] for p in probe),
        "alpha_probe_n": len(probe),
        "seconds": time.perf_counter() - t0,
        "timings": timings,
        "n_comparisons": len(rows),
        "simulation_assumptions": "D2-D5 approved 2026-11-09 18:41 Asia/Shanghai",
        "hardware_claim": False,
        "freeze": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "cutoff.json").write_text(json.dumps(doc, indent=2) + "\n")
    # rows can be large; store a compact npz of the maxima only. Full rows are useful and should be modest.
    (RESULT_DIR / "cutoff_rows.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps({k: doc[k] for k in ("nf_sim", "c_spsa", "alpha_probe_worst_dp", "worst", "seconds")}, indent=2))
    return doc


if __name__ == "__main__":
    run_cutoff()
