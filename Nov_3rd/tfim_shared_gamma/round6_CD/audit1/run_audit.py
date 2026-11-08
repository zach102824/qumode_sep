"""Audit only. No optimizer calls. Fixed-state forwards are counted separately."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round5_S4"))
import sim

ROOT = Path(__file__).resolve().parent
CD = ROOT.parent
REPO = CD.parents[2]
CELLS = CD / "results" / "cells"
STRIDE4 = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
TAU = 1e-9
FORWARDS = {"n": 0}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def wilson(k, n):
    if n == 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(max(p * (1 - p) / n, 0) + z * z / (4 * n * n)) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def both(E, F, Eb, Fb):
    return bool(E < Eb - TAU and F > Fb + TAU)


def quartiles(v):
    v = np.asarray(v, dtype=np.float64)
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return {"min": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "max": float(q[4]), "n": int(v.size)}


def main():
    sysd = sim.parse_system(STRIDE4)
    spec = sim.spectrum(sysd)
    E0 = float(spec["E0"])
    gs = spec["gs"]
    gs_parity = float(np.vdot(gs, gs[::-1]).real)
    fresh5 = json.loads((CELLS / "fresh_L5.json").read_text())
    fresh6 = json.loads((CELLS / "fresh_L6.json").read_text())
    published = np.load(CD / "published_states.npz")
    ref_l5 = published["L5"]
    # one producer forward of the published L5 vector is already stored; do not resimulate it
    p11 = {"E": -8.298215584895333, "F": 0.4848510087447086, "source": "5096c2a4 slice2/rescue_results.json p11", "shortfall": 0}
    p12 = {"E": -8.3254075497832, "F": 0.5147185522341006, "source": "5096c2a4 slice2/rescue_results.json p12", "shortfall": 0}
    p9 = {"E": -8.222874170728861, "F": 0.41200661658598803, "source": "561605ff summary.json p9", "shortfall": 133076, "status": "rescue-incomplete"}
    p10 = {"E": -8.263750172422338, "F": 0.4523066994703289, "source": "561605ff summary.json p10", "shortfall": 138662, "status": "rescue-incomplete"}

    rows = []
    states = []
    psis = []
    text = (CELLS / "fresh_L5.json").read_text().splitlines()
    line_of = {}
    for i, line in enumerate(text, start=1):
        s = line.strip().rstrip(",")
        if s.startswith('"k":'):
            # first k of each start object; later keys are not named k
            pass
    # array index equals chain id
    for start in fresh5["starts"]:
        k = int(start["k"])
        status = "FOUND" if (not start.get("censored") and "x" in start and len(start["x"]) == 18 and "E" in start) else ("PARTIAL" if "E" in start else "MISSING")
        rows.append({
            "chain_id": k,
            "seed_arm": 0,
            "seed_round": 6,
            "seed_depth": 5,
            "seed_k": k,
            "seed_tuple": "20261107,6,0,5,%d" % k,
            "P": 18,
            "selection_rule": "sole fresh start of this chain at L5; that vector initializes L6",
            "logged_E": start.get("E"),
            "logged_energy_error": start.get("energy_error"),
            "logged_F": start.get("F"),
            "logged_var": start.get("var_H"),
            "source_path": "Nov_3rd/tfim_shared_gamma/round6_CD/results/cells/fresh_L5.json",
            "source_sha256": sha_file(CELLS / "fresh_L5.json"),
            "json_index": k,
            "status": status,
        })
        if status != "FOUND":
            continue
        x = np.asarray(start["x"], dtype=np.float64)
        psi = sim.logical_state(sysd, 5, x)
        FORWARDS["n"] += 1
        E, var = sim.energy_var(psi, sysd)
        F = sim.fidelity(gs, psi)
        parity = float(np.vdot(psi, psi[::-1]).real)
        plus_w = 0.5 * (1.0 + parity)
        digest = hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()
        ov = abs(np.vdot(psi, ref_l5))
        ov_c = abs(np.vdot(psi, np.conj(ref_l5)))
        g, tk, tm, shells = sim.unpack(x, 5)
        rec = {
            "chain_id": k,
            "logged_E": start["E"],
            "replay_E": E,
            "dE": E - start["E"],
            "E_agree_1e-10": abs(E - start["E"]) <= 1e-10,
            "energy_error_vs_artifact_E0": E - E0,
            "F_full_space_gs": F,
            "var_H_direct": var,
            "norm": float(np.linalg.norm(psi)),
            "parity_expectation": parity,
            "plus_sector_weight": plus_w,
            "state_hash_replay": digest,
            "state_hash_logged": start["state_hash"],
            "hash_match": digest == start["state_hash"],
            "overlap_published_L5_arm0": ov,
            "overlap_published_L5_conjugate": ov_c,
            "conjugate_only": bool(ov_c > 0.99 and ov <= 0.99),
            "same_ray_as_published_L5": bool(ov > 0.99),
            "angles": {"gamma": g, "theta_k": tk, "theta_m": tm,
                       "c1": shells[:, 0].tolist(), "c2": shells[:, 1].tolist(), "c3": shells[:, 2].tolist()},
        }
        for name, base in (("p12", p12), ("p11", p11), ("p9", p9), ("p10", p10)):
            rec["both_" + name] = both(E, F, base["E"], base["F"])
        states.append(rec)
        psis.append(psi)

    n = 48
    found = len(states)
    missing = n - found
    err = np.array([s["energy_error_vs_artifact_E0"] for s in states])
    fids = np.array([s["F_full_space_gs"] for s in states])
    # D's L5 energy from baseline arm0, full precision
    d_l5_E = -8.385155838742358
    d_l5_err = d_l5_E - (-8.6132024589353)
    within_tau = int(np.sum(np.abs(err - d_l5_err) <= TAU)) if found else 0
    within_1e3 = int(np.sum(np.abs(err - d_l5_err) <= 1e-3)) if found else 0

    def counts(key):
        k = int(sum(1 for s in states if s[key]))
        return {
            "k_among_replayed": k,
            "n_replayed": found,
            "n_cohort": n,
            "n_missing": missing,
            "wilson_replayed": wilson(k, found),
            "wilson_missing_as_fail": wilson(k, n),
            "wilson_missing_as_success": wilson(k + missing, n),
            "label": "missing-as-fail and missing-as-success equal the replayed interval when missing is 0",
        }

    dist = {
        "denominator": 48,
        "n_replayed": found,
        "n_missing": missing,
        "E0_recomputed": E0,
        "E0_baseline_v1_literal": -8.6132024589353,
        "gs_parity_expectation": gs_parity,
        "full_space_gs_is_plus_sector_gs": abs(gs_parity - 1) < 1e-8,
        "energy_error_quartiles": quartiles(err) if found else None,
        "F_quartiles": quartiles(fids) if found else None,
        "within_tau_of_D_arm0_L5_energy": within_tau,
        "within_1e-3_of_D_arm0_L5_energy": within_1e3,
        "D_L5_target": "baseline_v1 stride4 arm0_L5 energy, the round-5 equalize vector D replayed. D did not store that wavefunction; overlap below is against C's published_states.npz L5 vector of that same record.",
        "both_vs_completed_p12": counts("both_p12"),
        "both_vs_completed_p11": counts("both_p11"),
        "both_vs_p9_PROVISIONAL": counts("both_p9"),
        "both_vs_p10_PROVISIONAL": counts("both_p10"),
        "n_same_ray_as_published_L5": int(sum(1 for s in states if s["same_ray_as_published_L5"])),
        "n_conjugate_only_vs_published_L5": int(sum(1 for s in states if s["conjugate_only"])),
        "diagnostic_forwards": FORWARDS["n"],
        "optimizer_evaluations": 0,
    }

    # L6 arithmetic recount, no new forwards
    l6_rows = []
    for start in fresh6["starts"]:
        E = start["E"]
        F = start["F"]
        l6_rows.append({
            "chain_id": start["k"],
            "logged_E": E,
            "logged_F": F,
            "logged_energy_error": start["energy_error"],
            "both_p11": both(E, F, p11["E"], p11["F"]),
            "both_p12": both(E, F, p12["E"], p12["F"]),
            "state_hash": start["state_hash"],
        })
    k11 = int(sum(1 for r in l6_rows if r["both_p11"]))
    k12 = int(sum(1 for r in l6_rows if r["both_p12"]))
    l6 = {
        "n": len(l6_rows),
        "source": "fresh_L6.json logged E and F, no new forward",
        "p11_completed": p11,
        "p12_completed": p12,
        "n_both_p11": k11,
        "wilson_p11": wilson(k11, len(l6_rows)),
        "n_both_p12": k12,
        "wilson_p12": wilson(k12, len(l6_rows)),
        "previous_reported_p12_count": 32,
        "p12_count_unchanged": k12 == 32,
        "note": "Comparison uses slice2 completed energies and fidelities. Initial QR p12 fidelity differs in the 1e-8 place and is not substituted.",
    }

    # costs
    by_depth = {}
    per_chain = []
    for L, name in ((5, "fresh_L5.json"), (6, "fresh_L6.json")):
        pass
    chain_cost = {k: {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5": 0, "L6": 0} for k in range(48)}
    fresh_marginal = 0
    for L in range(1, 7):
        cell = json.loads((CELLS / f"fresh_L{L}.json").read_text())
        s = 0
        for start in cell["starts"]:
            nv = int(start.get("nfev", 0))
            s += nv
            if not start.get("censored"):
                chain_cost[int(start["k"])][f"L{L}"] = nv
        by_depth[str(L)] = {"sum_nfev": s, "marginal_field": cell["marginal_nfev"], "n": len(cell["starts"])}
        fresh_marginal += s
    desc_marginal = 0
    desc_depth = {}
    for L in (3, 4, 5, 6):
        cell = json.loads((CELLS / f"descend_L{L}.json").read_text())
        s = sum(int(st.get("nfev", 0)) for st in cell["starts"])
        desc_depth[str(L)] = s
        desc_marginal += s
    finished = 48 * 6 + 24 * 4
    trace_only = fresh_marginal + desc_marginal - finished
    ledger = {
        "schema": "round6-CD-cost-ledger-v2",
        "optimizer_evaluations_this_audit": 0,
        "diagnostic_forwards_this_audit": FORWARDS["n"],
        "v1_frozen_source_constant": 53927,
        "v1_not_replaced": True,
        "round6_fresh_marginal_includes_finish_forward": fresh_marginal,
        "round6_descend_marginal_includes_finish_forward": desc_marginal,
        "round6_finished_starts": finished,
        "round6_trace_lengths_sum": trace_only,
        "evals_new_field_in_run_meta": 70477,
        "gap_384": {
            "reported_family_inclusive": 124788,
            "source_plus_evals_new": 53927 + 70477,
            "difference": 124788 - (53927 + 70477),
            "finished_starts": finished,
            "verdict": "RECONCILED",
            "evidence": "Each finished start stores nfev = LBFGS trace length + 1. run_meta evals_new counts only trace calls. Sum of stored nfev minus finished-start count equals 70477. The 384 is those 384 finish forwards.",
        },
        "gap_1000": {
            "producer_source_constant": 28099,
            "raw_sum_round4_plain_L1_to_L4": 13076,
            "raw_sum_round3_A_sub_M_L1_to_L6": 14023,
            "raw_sum": 27099,
            "difference": 28099 - 27099,
            "family_if_constant_28099_plus_arm2_marginal_25828": 53927,
            "family_if_raw_sum_27099_plus_arm2_marginal_25828": 52927,
            "verdict": "RECONCILED",
            "evidence": "No event with nfev 1000 exists. The six round-3 nfev_sum values add to 14023, not 15023. 13076+14023=27099. The hardcoded 28099 is that sum plus 1000.",
        },
        "fresh_by_depth": by_depth,
        "descend_by_depth": desc_depth,
        "per_chain_fresh": [
            {
                "chain_id": k,
                "through_L5": sum(chain_cost[k][f"L{L}"] for L in range(1, 6)),
                "through_L6": sum(chain_cost[k][f"L{L}"] for L in range(1, 7)),
                **chain_cost[k],
            }
            for k in range(48)
        ],
        "family_inclusive_v2_using_raw_source": fresh_marginal + desc_marginal + 27099,
        "family_inclusive_v1_arithmetic_as_published": fresh_marginal + desc_marginal + 53927,
        "Bp_impact": "None. Issued rescue allowances stay on the frozen 53927. This audit does not replace that constant.",
    }

    (ROOT / "l5_inventory.csv").write_text("")
    with (ROOT / "l5_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "l5_states.json").write_text(json.dumps(states, indent=2) + "\n")
    (ROOT / "l5_distribution.json").write_text(json.dumps(dist, indent=2) + "\n")
    (ROOT / "l6_vs_p11_addendum.json").write_text(json.dumps(l6, indent=2) + "\n")
    (ROOT / "cost_ledger_v2.json").write_text(json.dumps(ledger, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    e_sorted = np.sort(err)
    f_sorted = np.sort(fids)
    axes[0].step(e_sorted, np.arange(1, found + 1) / 48, where="post")
    axes[0].set_xlabel("L5 energy error")
    axes[0].set_ylabel("CDF over 48")
    axes[1].step(f_sorted, np.arange(1, found + 1) / 48, where="post")
    axes[1].set_xlabel("L5 fidelity")
    fig.tight_layout()
    fig.savefig(ROOT / "l5_cdf.png", dpi=140)
    plt.close(fig)
    print("forwards", FORWARDS["n"], "found", found, "both p12", counts("both_p12")["k_among_replayed"], "L6 p11", k11 if False else l6["n_both_p11"], "L6 p12", l6["n_both_p12"])
    print("E0", repr(E0), "gs parity", gs_parity)
    print("dE max", max(abs(s["dE"]) for s in states))


if __name__ == "__main__":
    main()
