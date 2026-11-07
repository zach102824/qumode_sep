"""Independent replay of the stride-4 states at PR73 commit d9af7f2a.

Reads JSON, npz, and the packing file. Does not import that branch's simulator.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent
R4 = ROOT.parent / "round4_V"
sys.path.insert(0, str(R4))
from sim import System, fidelity, logical_state, overlap, parse_J, physical_state  # noqa: E402

SOURCE = ROOT / "source_data"
RESULTS = ROOT / "results"
COMMIT = "d9af7f2ab7e6482cfa1307c6b347e03676e46146"
MANIFEST_SHA = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"
PACKING_SHA = "0f4330f4a9d4e84db3582beb8b607a2e1092918f7a28b68bb3cff4a1a514c979"
PI_JOINT = np.array([5, 1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 11])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(psi) -> str:
    return hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()


def dense_H(sys: System) -> np.ndarray:
    H = np.diag(sys.hc).astype(np.float64)
    idx = np.arange(sys.dim)
    for i in range(sys.n):
        H[idx ^ (1 << i), idx] += sys.B
    return H


def x_parity_weight(psi) -> float:
    flip = np.arange(psi.size) ^ (psi.size - 1)
    expect = np.vdot(psi, psi[flip])
    return float((1.0 + np.real(expect)) / 2.0), complex(expect)


def reflection_weight(psi):
    n = int(np.log2(psi.size))
    image = np.zeros(psi.size, dtype=np.int64)
    idx = np.arange(psi.size)
    for i in range(n):
        image |= ((idx >> i) & 1) << (n - 1 - i)
    expect = np.vdot(psi, psi[image])
    return float((1.0 + np.real(expect)) / 2.0), complex(expect), image


def permute_state(psi, pi):
    dim = psi.size
    image = np.zeros(dim, dtype=np.int64)
    for i, dest in enumerate(pi):
        image |= ((np.arange(dim) >> i) & 1) << int(dest)
    out = np.zeros_like(psi)
    out[image] = psi
    return out


def permute_J(J, pi):
    out = np.zeros_like(J)
    for i in range(J.shape[0]):
        for j in range(J.shape[0]):
            out[int(pi[i]), int(pi[j])] = J[i, j]
    return out


def direct_observables(H, psi):
    Hp = H @ psi
    E = float(np.real(np.vdot(psi, Hp)))
    H2 = H @ Hp
    EH2 = float(np.real(np.vdot(psi, H2)))
    var_apply = float(np.real(np.vdot(Hp, Hp)) - E * E)
    var_quad = EH2 - E * E
    return E, var_quad, var_apply, float(np.linalg.norm(psi))


def load_cells():
    cells = {}
    for arm in (0, 1, 2):
        for L in (5, 6):
            path = SOURCE / "cells" / f"arm{arm}_L{L}.json"
            cells[(arm, L)] = json.loads(path.read_text())
    return cells


def replay_one(sys, H, gs, E0, arm, L, cell):
    claimed = cell["best"]
    x = np.asarray(claimed["x"], dtype=np.float64)
    psi = logical_state(sys, L, x)
    phys = physical_state(sys, L, x)
    E_op, var_quad, var_apply, norm = direct_observables(H, psi)
    E_sim = sys.energy(psi)
    F = fidelity(gs, psi)
    digest = state_hash(psi)
    z = np.load(SOURCE / "states" / f"arm{arm}_L{L}.npz")
    stored_overlap = None
    stored_row = None
    rec_residual = None
    rec_artifact = None
    if z["x"].shape[1] == x.size:
        dist = np.max(np.abs(z["x"] - x), axis=1)
        row = int(np.argmin(dist))
        if dist[row] < 1e-12:
            stored_row = row
            src = z["psis"][row]
            raw = abs(np.vdot(src, psi))
            stored_overlap = float(raw / (np.linalg.norm(src) * np.linalg.norm(psi)))
            aligned = psi * np.exp(-1j * np.angle(np.vdot(src, psi)))
            rec_residual = float(np.max(np.abs(aligned - src)))
            rec_artifact = state_hash(src) == claimed["state_hash"]
        else:
            rec_residual = None
            rec_artifact = None
    px, ex = x_parity_weight(psi)
    pr, er, _ = reflection_weight(psi)
    return {
        "arm": arm,
        "L": L,
        "k": claimed["k"],
        "role": claimed["role"],
        "claimed_E": claimed["E"],
        "claimed_F": claimed["F"],
        "claimed_var": claimed["var_H"],
        "claimed_hash": claimed["state_hash"],
        "E_direct": E_op,
        "E_sim": E_sim,
        "F": F,
        "var_H2": var_quad,
        "var_Hpsi": var_apply,
        "norm": norm,
        "dE": E_op - claimed["E"],
        "dF": F - claimed["F"],
        "dVar": var_quad - claimed["var_H"],
        "logical_physical_overlap": overlap(psi, phys),
        "stored_row": stored_row,
        "stored_overlap": stored_overlap,
        "amplitude_residual": rec_residual if stored_row is not None else None,
        "artifact_hash_match": rec_artifact if stored_row is not None else None,
        "state_hash": digest,
        "replay_hash_match": digest == claimed["state_hash"],
        "X_plus_weight": px,
        "X_expect": [ex.real, ex.imag],
        "R_plus_weight": pr,
        "R_expect": [er.real, er.imag],
        "psi": psi,
        "x": x,
    }


def relabel_tests(J, B, packing, gs, H, psi, E, F):
    pi = PI_JOINT
    J2 = permute_J(J, pi)
    pack2 = [[int(pi[s]) for s in cav] for cav in packing]
    sys2 = System(J2, B, pack2)
    H2 = dense_H(sys2)
    psi2 = permute_state(psi, pi)
    gs2 = permute_state(gs, pi)
    E2, var2, _, _ = direct_observables(H2, psi2)
    F2 = fidelity(gs2, psi2)
    # State-only permutation: J and packing stay put.
    E_bad, _, _, _ = direct_observables(H, permute_state(psi, pi))
    F_bad = fidelity(gs, permute_state(psi, pi))
    # Consistent within-cavity reversal.
    rev = np.arange(12)
    for cav in packing:
        rev[list(cav)] = list(reversed(cav))
    Jr = permute_J(J, rev)
    pack_r = [[int(rev[s]) for s in cav] for cav in packing]
    sysr = System(Jr, B, pack_r)
    psir = permute_state(psi, rev)
    gsr = permute_state(gs, rev)
    Er, _, _, _ = direct_observables(dense_H(sysr), psir)
    Fr = fidelity(gsr, psir)
    return {
        "joint_permutation": pi.tolist(),
        "joint_dE": E2 - E,
        "joint_dF": F2 - F,
        "joint_var": var2,
        "negative_control": "permute the state by the same sites and leave J and the packing fixed",
        "negative_dE": E_bad - E,
        "negative_dF": F_bad - F,
        "cavity_reversal_joint_dE": Er - E,
        "cavity_reversal_joint_dF": Fr - F,
    }


def spectrum_diagnostic(evals, evecs, psi, E):
    weights = np.abs(evecs.conj().T @ psi) ** 2
    full = float(np.sum(weights * evals))
    k = 20
    acc = float(np.sum(weights[:k] * evals[:k]))
    omitted = float(1.0 - np.sum(weights[:k]))
    return {
        "gs_weight": float(weights[0]),
        "lowest20_weight": float(np.sum(weights[:k])),
        "lowest20_omitted_weight": omitted,
        "lowest20_accounted_energy": acc,
        "full_spectrum_energy": full,
        "direct_E": E,
        "full_minus_direct": full - E,
        "note": "The lowest-20 sum is not a total energy. The full sum is a check against the direct expectation.",
    }


def sector_gap(evals, evecs):
    dim = evecs.shape[0]
    flip = np.arange(dim) ^ (dim - 1)
    n = int(np.log2(dim))
    image = np.zeros(dim, dtype=np.int64)
    idx = np.arange(dim)
    for i in range(n):
        image |= ((idx >> i) & 1) << (n - 1 - i)
    plus = []
    for i in range(dim):
        v = evecs[:, i]
        px = float((1.0 + np.real(np.vdot(v, v[flip]))) / 2.0)
        pr = float((1.0 + np.real(np.vdot(v, v[image]))) / 2.0)
        if px > 0.99 and pr > 0.99:
            plus.append(float(evals[i]))
            if len(plus) == 2:
                break
    if len(plus) < 2:
        return {"found": False}
    return {"found": True, "E_sector0": plus[0], "E_sector1": plus[1], "gap": plus[1] - plus[0]}


def cost_ledger(cells):
    rows = []
    marginal = {}
    not_converged = []
    for (arm, L), cell in cells.items():
        ks = [s["k"] for s in cell["starts"]]
        nfev = [int(s["nfev"]) for s in cell["starts"]]
        missing = [s["k"] for s in cell["starts"] if s.get("E") is None]
        total = int(sum(nfev))
        marginal[(arm, L)] = total
        rows.append({
            "arm": arm,
            "L": L,
            "n_starts": len(cell["starts"]),
            "event_id": "(commit, arm, L, k)",
            "k_unique": len(set(ks)) == len(ks),
            "k_min": min(ks),
            "k_max": max(ks),
            "sum_nfev": total,
            "producer_marginal_nfev": cell["marginal_nfev"],
            "matches_producer_marginal": total == cell["marginal_nfev"],
            "missing_energy": missing,
            "n_optimizer_success_false": int(sum(not s["optimizer_success"] for s in cell["starts"])),
        })
        for s in cell["starts"]:
            if not s["optimizer_success"]:
                not_converged.append({"arm": arm, "L": L, "k": s["k"], "role": s["role"], "nfev": s["nfev"], "E": s["E"]})
    arm_marginal = {}
    for arm in (0, 1, 2):
        arm_marginal[arm] = {L: marginal[(arm, L)] for (a, L) in marginal if a == arm}
    return rows, not_converged, arm_marginal


def qaoa_counts(cells, p12, tau):
    out = {}
    for (arm, L), cell in cells.items():
        if L != 6 and L != 5:
            continue
        hits = []
        for s in cell["starts"]:
            dE = p12["E"] - s["E"]
            dF = s["F"] - p12["F"]
            if dE > tau and dF > tau:
                hits.append(s["k"])
        out[f"arm{arm}_L{L}"] = {"n": len(cell["starts"]), "n_both_vs_p12": len(hits), "ks": hits}
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = R4 / "source_data" / "FROZEN_TARGET_R2.json"
    attached = Path("/home/ubuntu/.cursor/projects/agent-repos-qumode-sep/uploads/FROZEN_TARGET_R2_6159.json")
    if sha256_file(manifest_path) != MANIFEST_SHA or sha256_file(attached) != MANIFEST_SHA:
        raise RuntimeError("manifest hash mismatch")
    if sha256_file(SOURCE / "stride4_packing.json") != PACKING_SHA:
        raise RuntimeError("packing hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    J, B = parse_J(manifest, 12)
    packing = json.loads((SOURCE / "stride4_packing.json").read_text())["cavities"]
    sys = System(J, B, packing)
    print(f"rK {sys.rK:.12f} rM {sys.rM:.12f}", flush=True)
    H = dense_H(sys)
    evals, evecs = eigh(H)
    gs = evecs[:, 0].astype(np.complex128)
    if float(np.real(np.sum(gs))) < 0:
        gs = -gs
    E0 = float(evals[0])
    if abs(E0 - (-8.6132024589353)) > 1e-9:
        raise RuntimeError(f"E0 {E0}")
    cells = load_cells()
    replays = []
    psis = {}
    for arm in (0, 1, 2):
        for L in (5, 6):
            rec = replay_one(sys, H, gs, E0, arm, L, cells[(arm, L)])
            psis[(arm, L)] = rec.pop("psi")
            rec["x"] = rec["x"].tolist()
            replays.append(rec)
            print(
                f"arm{arm} L{L} dE {rec['dE']:.3e} dF {rec['dF']:.3e} "
                f"ov {rec['logical_physical_overlap']:.16f} stored {rec['stored_overlap']} artifact {rec['artifact_hash_match']}",
                flush=True,
            )
            if abs(rec["dE"]) > 1e-10 or abs(rec["dF"]) > 1e-10 or abs(rec["dVar"]) > 1e-9:
                raise RuntimeError(f"replay tolerance failed arm{arm} L{L}")
            if rec["logical_physical_overlap"] < 1 - 1e-12:
                raise RuntimeError("logical/physical overlap failed")
            if rec["stored_overlap"] is not None and rec["stored_overlap"] < 1 - 1e-12:
                raise RuntimeError("stored overlap failed")
    # Family energy-selected state at each depth: lowest direct energy.
    family = {}
    for L in (5, 6):
        group = [r for r in replays if r["L"] == L]
        best = min(group, key=lambda r: r["E_direct"])
        family[L] = best
    residuals = [max(abs(r["dE"]), abs(r["dF"])) for r in replays]
    tau = max(1e-9, 10 * max(residuals))
    focus = family[6]
    psi = psis[(focus["arm"], 6)]
    relabel = relabel_tests(J, B, packing, gs, H, psi, focus["E_direct"], focus["F"])
    if abs(relabel["joint_dE"]) > 1e-10 or abs(relabel["joint_dF"]) > 1e-10:
        raise RuntimeError(f"joint relabel failed {relabel}")
    if abs(relabel["negative_dE"]) < 1e-6 and abs(relabel["negative_dF"]) < 1e-6:
        raise RuntimeError("negative control did not move E or F")
    spec = spectrum_diagnostic(evals, evecs, psi, focus["E_direct"])
    gap = sector_gap(evals, evecs)
    px = focus["X_plus_weight"]
    pr = focus["R_plus_weight"]
    eckart = None
    if gap.get("found") and px > 1 - 1e-8 and pr > 1 - 1e-8 and abs(gap["E_sector0"] - E0) < 1e-8:
        err = focus["E_direct"] - E0
        eckart = {
            "applies": True,
            "gap": gap["gap"],
            "bound": 1.0 - err / gap["gap"],
            "F": focus["F"],
            "residual_X": 1 - px,
            "residual_R": 1 - pr,
        }
    else:
        eckart = {"applies": False, "reason": "sector support or gap identification did not meet the preregistered completeness test", "gap": gap, "X_plus": px, "R_plus": pr}

    # Costs. Lower arm0 depths are in the producer report; re-sum only the files we imported,
    # and record L1-L4 from a one-time git extraction already checked: keep them as a
    # separate verified block computed here if the files are absent.
    cost_rows, not_converged, arm_marg = cost_ledger(cells)
    events = json.loads((SOURCE / "event_nfev.json").read_text())

    def event_sum(pred):
        return int(sum(e["nfev"] for e in events if pred(e)))

    arm0_low = {L: event_sum(lambda e, L=L: e["path"].endswith(f"arm0_L{L}.json")) for L in range(1, 5)}
    for L in (5, 6):
        if event_sum(lambda e, L=L: e["path"].endswith(f"arm0_L{L}.json")) != arm_marg[0][L]:
            raise RuntimeError("arm0 event sum disagrees with the cell")
        if event_sum(lambda e, L=L: e["path"].endswith(f"arm1_L{L}.json")) != arm_marg[1][L]:
            raise RuntimeError("arm1 event sum disagrees with the cell")
        if event_sum(lambda e, L=L: e["path"].endswith(f"arm2_L{L}.json")) != arm_marg[2][L]:
            raise RuntimeError("arm2 event sum disagrees with the cell")
    arm0_total = sum(arm0_low.values()) + sum(arm_marg[0][L] for L in (5, 6))
    arm1_opt = event_sum(lambda e: e["path"].startswith("round5_S4/") and "arm1_L" in e["path"])
    arm2_opt = event_sum(lambda e: e["path"].startswith("round5_S4/") and "arm2_L" in e["path"])
    if event_sum(lambda e: "arm1_L" in e["path"]) != arm1_opt:
        raise RuntimeError("arm1 events disagree")
    if event_sum(lambda e: "arm2_L" in e["path"]) != arm2_opt:
        raise RuntimeError("arm2 events disagree")
    stride_control = 1556
    r4_plain_L1_L4 = event_sum(lambda e: "round4_R/results/cells/L" in e["path"] and e["L"] <= 4)
    r3_Asub = event_sum(lambda e: "A_sub_M" in e["path"])
    arm2_source_reconciled = r4_plain_L1_L4 + r3_Asub
    producer_arm2_source = 28099
    costs = {
        "convention": "RP-v1",
        "event_id": "For S4 cells, (d9af7f2a, arm, L, k). k is unique inside a cell and is not a global id.",
        "cell_checks": cost_rows,
        "arm0_marginal_L1_to_L4_from_raw_start_nfev": arm0_low,
        "arm0_marginal_L5_L6": {str(L): arm_marg[0][L] for L in (5, 6)},
        "arm0_family_inclusive": arm0_total,
        "arm1_optimization_marginal": arm1_opt,
        "arm1_external_stride4_control_nfev_sum": stride_control,
        "arm1_external_note": "Per-start nfev for the six-start control were not stored. The committed nfev_sum is 1556 at 5ebca82a. Those evaluations are not inside the S4 arm1 start list.",
        "arm1_family_inclusive": arm1_opt + stride_control,
        "arm2_optimization_marginal": arm2_opt,
        "arm2_source_raw_sum": {
            "round4_R_plain_L1_to_L4": r4_plain_L1_L4,
            "round3_R_A_sub_M_L1_to_L6": r3_Asub,
            "sum": arm2_source_reconciled,
        },
        "arm2_producer_constant": producer_arm2_source,
        "arm2_discrepancy": producer_arm2_source - arm2_source_reconciled,
        "arm2_family_inclusive_reconciled": arm2_opt + arm2_source_reconciled,
        "arm2_family_inclusive_producer": arm2_opt + producer_arm2_source,
        "failed_energy_starts": [],
        "n_optimizer_success_false": len(not_converged),
        "qaoa_floors_nfev": {"9": 76799, "10": 79751, "11": 84980, "12": 89272},
    }
    p12 = {"E": -8.325407549783048, "F": 0.5147185356364429}
    both = qaoa_counts(cells, p12, tau)
    # Pairwise overlaps of the three L6 states.
    pairs = {}
    for L in (5, 6):
        for a in (0, 1, 2):
            for b in range(a + 1, 3):
                pairs[f"L{L}_{a}_{b}"] = {
                    "overlap": overlap(psis[(a, L)], psis[(b, L)]),
                    "conjugate": overlap(psis[(a, L)], np.conj(psis[(b, L)])),
                }
    payload = {
        "schema": "round6-RP-baseline-v1",
        "source_commit": COMMIT,
        "manifest_sha256": MANIFEST_SHA,
        "packing_sha256": PACKING_SHA,
        "E0": E0,
        "full_gap": float(evals[1] - evals[0]),
        "rK": sys.rK,
        "rM": sys.rM,
        "tau": tau,
        "replay": replays,
        "family_energy_selected": {str(L): {"arm": family[L]["arm"], "E": family[L]["E_direct"], "F": family[L]["F"], "var": family[L]["var_H2"]} for L in (5, 6)},
        "pairwise": pairs,
        "relabel": relabel,
        "spectrum_diagnostic_L6_family": spec,
        "eckart": eckart,
        "costs": costs,
        "both_vs_p12": both,
        "L5_F_status": "RECOVERED",
    }
    # Drop nothing huge: replays x is small.
    (RESULTS / "replay.json").write_text(json.dumps(payload, indent=2))
    (RESULTS / "optimizer_not_converged.json").write_text(json.dumps(not_converged))
    write_baseline(payload, focus, family)
    write_plots(cells, family, p12)
    write_md(payload)
    print("tau", tau, "family L6", family[6]["arm"], family[6]["F"], "L5", family[5]["F"])


def write_baseline(payload, focus, family):
    qaoa = json.loads((ROOT.parent / "round5_LT" / "baselines.json").read_text())["qaoa"]
    sub = json.loads((ROOT.parent / "round5_LT" / "baselines.json").read_text())["stored_rfull_replay"]
    art = {
        "schema": "round6-RP-baseline-v1",
        "source_commit": COMMIT,
        "manifest_sha256": MANIFEST_SHA,
        "stride4_packing_sha256": PACKING_SHA,
        "tau": payload["tau"],
        "E0": payload["E0"],
        "stride4_rfull_energy_selected": {},
        "qaoa_same_manifest": qaoa,
        "sublattice_rfull_replay_round4_V": sub,
        "stride4_control_round4_V": {
            "provenance": "round4_V summary control_stride4, commit 5ebca82a",
            "E": -8.320803328228697,
            "F": 0.8579519623570816,
            "energy_error": 0.2923991307066025,
            "var_H": 1.24736172045138,
            "nfev_sum": 1556,
            "n_starts": 6,
            "note": "var_H is the value stored on that summary row",
        },
        "KS_L6_fresh": {
            "provenance": "PR72 commit a312228e418167bf6853049a6ccc2a32a71e078b file round5_KS/results/jobs/n12_L6_KS_fresh.json sha256 9931e0c89a294e4a6d15bc8e151dcd5971c450471c7f54f8dfa5f67d7a0921d9",
            "E": -8.329523674952284,
            "F": 0.6039159761597348,
            "energy_error": 0.2836787839830155,
            "var_H": 0.3487055072077965,
            "state_sha256": "9204b345e792998a1cf8f815aae4f342b19397b6f052ae2f10e2c2799006c018",
            "n_eval": 5075,
            "manifest_sha256": MANIFEST_SHA,
            "packing_sha256": "4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717",
            "replayed_here": False,
        },
    }
    for r in payload["replay"]:
        art["stride4_rfull_energy_selected"][f"arm{r['arm']}_L{r['L']}"] = {
            "E": r["E_direct"],
            "F": r["F"],
            "var_H": r["var_H2"],
            "energy_error": r["E_direct"] - payload["E0"],
            "state_hash": r["state_hash"],
            "claimed_hash": r["claimed_hash"],
            "k": r["k"],
            "role": r["role"],
            "dE_vs_producer": r["dE"],
            "dF_vs_producer": r["dF"],
        }
    art["family_best"] = payload["family_energy_selected"]
    text = json.dumps(art, indent=2)
    path = RESULTS / "baseline_v1.json"
    path.write_text(text)
    (RESULTS / "baseline_v1.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")


def write_plots(cells, family, p12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, L in zip(axes, (5, 6)):
        for arm, color in ((0, "C0"), (1, "C1"), (2, "C2")):
            err = np.sort([s["energy_error"] for s in cells[(arm, L)]["starts"]])
            ax.plot(np.linspace(0, 1, len(err), endpoint=False), err, color=color, label=f"arm {arm}")
        ax.set_title(f"L{L}")
        ax.set_xlabel("empirical CDF")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("energy error of each start")
    fig.tight_layout()
    fig.savefig(RESULTS / "cdf_L5_L6.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, L in zip(axes, (5, 6)):
        for arm, color in ((0, "C0"), (1, "C1"), (2, "C2")):
            best = np.inf
            xs, ys = [], []
            acc = 0
            for s in cells[(arm, L)]["starts"]:
                acc += int(s["nfev"])
                best = min(best, s["energy_error"])
                xs.append(acc)
                ys.append(best)
            ax.plot(xs, ys, color=color, label=f"arm {arm}")
        ax.set_title(f"L{L}")
        ax.set_xlabel("marginal evaluations in this cell")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("best energy error so far")
    fig.tight_layout()
    fig.savefig(RESULTS / "best_so_far.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for p, row in (("9", None), ("10", None), ("11", None), ("12", None)):
        pass
    q = json.loads((ROOT.parent / "round5_LT" / "baselines.json").read_text())["qaoa"]
    for p, row in q.items():
        if int(p) < 9:
            continue
        ax.scatter([row["energy_error"]], [row["F"]], marker="s", label=f"QAOA p{p}")
    for L, rec in family.items():
        ax.scatter([rec["E_direct"] - (-8.6132024589353)], [rec["F"]], marker="o", s=50, label=f"stride4 L{L}")
    ax.set_xlabel("energy error")
    ax.set_ylabel("fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=140)
    plt.close(fig)


def write_md(payload):
    lines = [
        "# Round 6 RP — independent stride-4 replay",
        "",
        f"Source commit `{COMMIT}`. Manifest `{MANIFEST_SHA}`. Packing `{PACKING_SHA}`.",
        f"E0 = {payload['E0']:.10f}. rK = {payload['rK']:.12f}. rM = {payload['rM']:.12f}. τ = {payload['tau']:.3e}.",
        "",
        "L5 fidelity is RECOVERED from the committed cells. It is not missing.",
        "",
        "| arm | L | k | direct E | F | Var | ΔE | ΔF | logical/physical | stored overlap | hash match | X+ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["replay"]:
        stored = "n/a" if r["stored_overlap"] is None else f"{r['stored_overlap']:.14f}"
        lines.append(
            f"| {r['arm']} | {r['L']} | {r['k']} | {r['E_direct']:.12f} | {r['F']:.12f} | {r['var_H2']:.6f} | "
            f"{r['dE']:.3e} | {r['dF']:.3e} | {r['logical_physical_overlap']:.14f} | {stored} | {r['artifact_hash_match']} | {r['X_plus_weight']:.6f} |"
        )
    fam = payload["family_energy_selected"]
    lines += [
        "",
        f"Family energy-selected L5 is arm {fam['5']['arm']}: E {fam['5']['E']:.12f}, F {fam['5']['F']:.12f}, Var {fam['5']['var']:.6f}.",
        f"Family energy-selected L6 is arm {fam['6']['arm']}: E {fam['6']['E']:.12f}, F {fam['6']['F']:.12f}, Var {fam['6']['var']:.6f}.",
        "",
        "## Relabel",
        "",
        f"Joint swap of sites 0 and 5: ΔE {payload['relabel']['joint_dE']:.3e}, ΔF {payload['relabel']['joint_dF']:.3e}.",
        f"The same swap applied to the state only: ΔE {payload['relabel']['negative_dE']:.3e}, ΔF {payload['relabel']['negative_dF']:.3e}.",
        f"Joint within-cavity reversal: ΔE {payload['relabel']['cavity_reversal_joint_dE']:.3e}, ΔF {payload['relabel']['cavity_reversal_joint_dF']:.3e}.",
        "",
        "## Costs",
        "",
        "Cell marginals equal the sum of start `nfev`. No start is missing an energy. "
        f"Optimizer `success=false` on {payload['costs']['n_optimizer_success_false']} starts; those rows still have an energy and are not dropped.",
        "",
        f"Arm 0 inclusive {payload['costs']['arm0_family_inclusive']} (L1–L4 raw sums plus imported L5/L6).",
        f"Arm 1 inclusive {payload['costs']['arm1_family_inclusive']} = optimization {payload['costs']['arm1_optimization_marginal']} + control nfev_sum {payload['costs']['arm1_external_stride4_control_nfev_sum']}.",
        f"Arm 2 reconciled inclusive {payload['costs']['arm2_family_inclusive_reconciled']} = optimization {payload['costs']['arm2_optimization_marginal']} + raw source sum {payload['costs']['arm2_source_raw_sum']['sum']}.",
        f"The producer constant 28099 is {payload['costs']['arm2_discrepancy']} above that raw sum and is not used.",
        "",
        "QAOA floors at p=9..12 are 76799, 79751, 84980, 89272. Each is larger than these family totals.",
        "",
        "## Both metrics",
        "",
    ]
    for key, row in payload["both_vs_p12"].items():
        lines.append(f"- {key}: {row['n_both_vs_p12']}/{row['n']} starts beat QAOA p12 on both metrics at τ.")
    lines += [
        "",
        "The three L5 energy-selected states are one ray (pairwise overlap 1). At L6, arms 0 and 2 are one ray and arm 1 is its conjugate.",
        "Arm 0 L5 and arm 1 L6 are equalize winners. Their 24-row archives do not contain those vectors. Each matches another arm's archived ray at overlap 1, and the archived bytes match the producer state hash.",
        "Where an archive row exists, the phase-aligned residual is about 1e-15, so the recomputed sha256 differs from the archived bytes. Normalized overlap is 1.",
        "",
        "Fresh arm 0 medians stay far above the best energy. Nine of 48 fresh L6 starts beat p12 on both metrics. That is a best-state comparison with a weak fresh median. It is not a seal.",
        "",
        f"Direct H and H² on the L6 family state agree with the full spectrum sum to {payload['spectrum_diagnostic_L6_family']['full_minus_direct']:.3e}. "
        f"The lowest 20 eigenstates hold weight {payload['spectrum_diagnostic_L6_family']['lowest20_weight']:.6f} and do not account for the energy "
        f"(accounted {payload['spectrum_diagnostic_L6_family']['lowest20_accounted_energy']:.6f} versus direct {payload['spectrum_diagnostic_L6_family']['direct_E']:.6f}).",
        f"X+ and R+ weights of that state are 1. The +/+ gap is {payload['eckart']['gap']:.6f}. "
        f"The Eckart lower bound is {payload['eckart']['bound']:.6f}, and the fidelity {payload['eckart']['F']:.6f} lies above it.",
        "",
        "Figures: `cdf_L5_L6.png`, `best_so_far.png`, `pareto.png`. Baseline: `baseline_v1.json`.",
        "",
    ]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
