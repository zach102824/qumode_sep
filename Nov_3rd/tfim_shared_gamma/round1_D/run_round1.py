"""Run the round-1 family-D comparison.

Order: algebra checks, entropy-max fields, then L=1..4 on N=6 and N=12,
then L=5..6, then a subspace repair of D, then extra QAOA depths used only
as matched-parameter references. Fidelity never enters the optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import time
import traceback
from pathlib import Path

import thread_env  # noqa: F401
import numpy as np

import report
import test_algebra
from ansatz import Model, canonicalize_angles, energy_and_grad, eta_score, fidelity, final_state, n_params
from common import dump_json, jsonable, public_target, select_field, sha256_array
from optimize import Pool, cfg_for, optimize_higher, optimize_l1_family, optimize_l1_qaoa, polish_starts

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METHODS = ("qaoa", "shared_beta", "nov3_D", "drop_c2", "drop_c3")

SUPPORTS = {
    "qaoa": "independent gamma_l on all ZZ pairs of Hc; mixer beta_l * sum_i X_i. B is only in the energy.",
    "shared_beta": "one gamma on Hc; mixer beta_l * sum_i X_i. Identical to QAOA at L=1.",
    "nov3_D": "one gamma on Hc (coefficient fixed at 1, no theta); per layer c1,c2,c3 on intra-cavity X, XX, XXX, shared across cavities.",
    "drop_c2": "native D with c2 fixed at 0.",
    "drop_c3": "native D with c3 fixed at 0.",
}


def source_hash() -> str:
    h = hashlib.sha256()
    for name in ("common.py", "ansatz.py", "optimize.py", "test_algebra.py", "report.py", "run_round1.py"):
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def stable_rng(*parts) -> np.random.Generator:
    raw = "|".join(map(str, parts))
    seed = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)
    return np.random.Generator(np.random.PCG64(seed))


def phase_hash(psi: np.ndarray) -> str:
    v = np.asarray(psi, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) > 0:
        v *= abs(v[k]) / v[k]
    return sha256_array(v)


def embed_subspace_into_D(method: str, L: int, x) -> np.ndarray | None:
    """Map a shared-beta or ablation vector to a D vector that represents the same circuit."""
    x = np.asarray(x, dtype=np.float64)
    if method == "shared_beta":
        shell = np.zeros((L, 3))
        shell[:, 0] = x[1:]
    elif method == "drop_c2":
        pairs = x[1:].reshape(L, 2)
        shell = np.column_stack([pairs[:, 0], np.zeros(L), pairs[:, 1]])
    elif method == "drop_c3":
        pairs = x[1:].reshape(L, 2)
        shell = np.column_stack([pairs[:, 0], pairs[:, 1], np.zeros(L)])
    else:
        return None
    return np.concatenate([[float(x[0])], shell.reshape(-1)])


def publish(model, gs, target, method, L, opt, seconds) -> tuple[dict, np.ndarray | None]:
    if not opt.get("ok"):
        return {
            "n": model.n,
            "L": L,
            "method": method,
            "status": "failed",
            "error": opt.get("error", "unknown"),
            "seconds": seconds,
        }, None
    x = np.asarray(opt["x"], dtype=np.float64)
    E, g = energy_and_grad(model, method, L, x)
    psi = final_state(model, method, L, x)
    E_state = model.energy(psi)
    xc = canonicalize_angles(method, L, x)
    Ec, _gc = energy_and_grad(model, method, L, xc)
    E0 = float(target["E0"])
    Emax = float(target["Emax"])
    record = {
        "n": model.n,
        "L": L,
        "method": method,
        "status": "complete",
        "objective": "energy",
        "fidelity_used_for_selection": False,
        "n_params": n_params(method, L),
        "support": SUPPORTS[method],
        "E": float(E),
        "E_state": float(E_state),
        "E0": E0,
        "Emax": Emax,
        "E_init": float(target["B"] * model.n),
        "energy_error": float(E - E0),
        "eta": eta_score(float(E), E0, Emax),
        "F": fidelity(gs, psi),
        "variational_violation": float(min(0.0, E - E0)),
        "x_raw": x.tolist(),
        "x_canonical": xc.tolist(),
        "canonical_energy_delta": float(Ec - E),
        "grad_norm": float(np.linalg.norm(g)),
        "best_tag": opt.get("best_tag"),
        "E_median": opt.get("E_median"),
        "E_mean": opt.get("E_mean"),
        "E_std": opt.get("E_std"),
        "n_starts": opt.get("n_starts"),
        "n_finished": opt.get("n_finished"),
        "nfev_sum": opt.get("nfev_sum"),
        "seconds": float(seconds),
        "state_sha256": phase_hash(psi),
        "state_norm": float(np.linalg.norm(psi)),
        "grid": opt.get("grid"),
        "screen": opt.get("screen"),
        "starts": opt.get("starts"),
        "best_trace": opt.get("best_trace"),
        "beta_offset_conversion": opt.get("beta_offset_conversion"),
    }
    return record, psi


def adopt_if_better(pool, model, gs, target, current, challenger_opt, method, L):
    """Keep the lower energy. Both arguments are already energy minimizations of the same method."""
    if not challenger_opt.get("ok"):
        return current, None
    if current.get("status") == "complete" and float(challenger_opt["E_best"]) >= float(current["E"]) - 1e-12:
        # Still record that a repair was attempted.
        current.setdefault("repair_attempts", []).append(
            {"tag": challenger_opt.get("best_tag"), "E": challenger_opt.get("E_best"), "adopted": False}
        )
        return current, None
    t0 = time.perf_counter()
    # Merge start lists so the reproducibility gate sees both searches.
    record, psi = publish(model, gs, target, method, L, challenger_opt, current.get("seconds", 0.0) + time.perf_counter() - t0)
    if current.get("starts"):
        record["starts"] = list(current["starts"]) + list(challenger_opt.get("starts") or [])
        energies = [s["E"] for s in record["starts"] if s.get("E") is not None]
        record["n_starts"] = len(record["starts"])
        record["n_finished"] = len(energies)
        record["E_median"] = float(np.median(energies))
        record["nfev_sum"] = int(current.get("nfev_sum") or 0) + int(challenger_opt.get("nfev_sum") or 0)
    record["best_tag"] = "repair:" + str(challenger_opt.get("best_tag"))
    record.setdefault("repair_attempts", []).append({"tag": record["best_tag"], "E": record["E"], "adopted": True})
    return record, psi


def save_record(record, psi):
    out = RESULTS / "opt" / f"n{record['n']}_L{record['L']}_{record['method']}.json"
    dump_json(out, record)
    if psi is not None and record.get("status") == "complete":
        state_path = RESULTS / "states" / f"n{record['n']}_L{record['L']}_{record['method']}.npy"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(state_path, np.asarray(psi, dtype=np.complex128))
        angle_path = RESULTS / "angles" / f"n{record['n']}_L{record['L']}_{record['method']}.npz"
        angle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            angle_path,
            x_raw=np.asarray(record["x_raw"], dtype=np.float64),
            x_canonical=np.asarray(record["x_canonical"], dtype=np.float64),
        )


def run_one(pool, model, gs, target, method, L, prev, qaoa_x, shared_x, donor_x, log):
    conf = cfg_for(model.n)
    rng = stable_rng("round1D", model.n, L, method, "main")
    t0 = time.perf_counter()
    log(f"start N={model.n} L={L} {method}")
    if L == 1 and method == "qaoa":
        opt = optimize_l1_qaoa(pool, target["B"], rng, conf)
    elif L == 1:
        opt = optimize_l1_family(pool, method, rng, conf, qaoa_x=qaoa_x, donor_x=donor_x)
    else:
        opt = optimize_higher(pool, method, L, prev, qaoa_x, shared_x, donor_x, target["B"], rng, conf)
    record, psi = publish(model, gs, target, method, L, opt, time.perf_counter() - t0)
    if record.get("energy_error") is not None and record["energy_error"] < -1e-6:
        log(f"  WARNING energy below E0 by {record['energy_error']}")
    if record.get("canonical_energy_delta") is not None and abs(record["canonical_energy_delta"]) > 1e-6:
        log(f"  WARNING canonicalization moved the energy by {record['canonical_energy_delta']}")
    log(
        f"  done {method} E={record.get('E')} F={record.get('F')} "
        f"err={record.get('energy_error')} tag={record.get('best_tag')} ({record.get('seconds', 0):.1f}s)"
    )
    return record, psi


def repair_subspaces(pool, model, gs, target, rows_here, log):
    """D contains shared-beta and both ablations. Polish D from those points if they are better."""
    L = next(r["L"] for r in rows_here.values() if "L" in r)
    D = rows_here.get("nov3_D")
    if D is None or D.get("status") != "complete":
        return rows_here
    starts = []
    for method in ("shared_beta", "drop_c2", "drop_c3"):
        src = rows_here.get(method)
        if src is None or src.get("status") != "complete":
            continue
        if float(src["E"]) < float(D["E"]) - 1e-10:
            vec = embed_subspace_into_D(method, L, src["x_raw"])
            if vec is not None:
                starts.append((f"from_{method}", vec))
    if not starts:
        return rows_here
    log(f"  repair D from {[t for t, _ in starts]}")
    opt = polish_starts(pool, "nov3_D", L, starts, maxfun=cfg_for(model.n)["maxfun"])
    updated, psi = adopt_if_better(pool, model, gs, target, D, opt, "nov3_D", L)
    rows_here["nov3_D"] = updated
    if psi is not None:
        save_record(updated, psi)
        log(f"  repaired D E={updated['E']} F={updated['F']}")
    return rows_here


def zero_shot(model12, gs12, target12, x6, method, L, kappa_scale):
    x6 = np.asarray(x6, dtype=np.float64)
    rows = []
    variants = [("raw", x6.copy())]
    xk = x6.copy()
    if method == "qaoa":
        xk[:L] *= kappa_scale
    else:
        xk[0] *= kappa_scale
    variants.append(("kappa_cost", xk))
    E0 = float(target12["E0"])
    Emax = float(target12["Emax"])
    for kind, x in variants:
        psi = final_state(model12, method, L, x)
        E = model12.energy(psi)
        rows.append(
            {
                "L": L,
                "method": method,
                "kind": kind,
                "kappa_scale": float(kappa_scale) if kind == "kappa_cost" else 1.0,
                "E": float(E),
                "E0": E0,
                "energy_error": float(E - E0),
                "F": fidelity(gs12, psi),
                "eta": eta_score(float(E), E0, Emax),
                "trained_on": 6,
                "evaluated_on": 12,
                "n12_angles_trained": False,
            }
        )
    return rows


def write_bundle(rows, targets, transfer, notes, verdict=None):
    if verdict is None:
        verdict = report.build_verdict([r for r in rows if r.get("status") == "complete"])
    dump_json(RESULTS / "summary.json", {"rows": rows, "verdict": verdict, "notes": notes})
    dump_json(RESULTS / "verdict.json", verdict)
    dump_json(RESULTS / "transfer.json", transfer)
    report.write_summary_csv(RESULTS / "summary.csv", rows)
    figs = report.write_figures(RESULTS / "figures", rows, targets)
    report.write_results_md(RESULTS / "RESULTS.md", rows, targets, verdict, transfer, notes + [f"figures: {figs}"])
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-min", type=float, default=80.0)
    parser.add_argument("--reuse-target", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "run.log"
    t_run = time.perf_counter()
    deadline = t_run + args.budget_min * 60.0

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    def time_left():
        return deadline - time.perf_counter()

    log("algebra checks")
    algebra = test_algebra.main()
    dump_json(RESULTS / "algebra.json", algebra)
    log(f"algebra ok; physical/logical grad max abs {algebra.get('physical_vs_logical_grad_max_abs')}")

    targets = {}
    models = {}
    gs = {}
    for n in (6, 12):
        npz = RESULTS / f"target_n{n}_arrays.npz"
        js = RESULTS / f"target_n{n}.json"
        if args.reuse_target and npz.exists() and js.exists():
            log(f"reuse target N={n}")
            payload = json_load(js)
            arrays = np.load(npz)
            payload["J"] = arrays["J"]
            payload["hc_diag"] = arrays["hc"]
            payload["gs"] = arrays["gs"]
        else:
            payload = select_field(n, log=log)
            public = public_target(payload)
            dump_json(js, public)
            np.savez(npz, J=payload["J"], hc=payload["hc_diag"], gs=np.asarray(payload["gs"], dtype=np.complex128))
            log(
                f"froze N={n} B={payload['B']:.6f} S={payload['entropy_bits']:.6f} "
                f"E0={payload['E0']:.6f} gap={payload['gap']:.3e} "
                f"P+={payload['parity']['P_plus_weight']:.6f} R+={payload['reflection']['R_plus_weight']:.6f} "
                f"endpoint={payload['endpoint_unresolved']}"
            )
        targets[n] = payload if "gs" in payload else None
        if targets[n] is None:
            public = json_load(js)
            arrays = np.load(npz)
            public["J"] = arrays["J"]
            public["hc_diag"] = arrays["hc"]
            public["gs"] = arrays["gs"]
            targets[n] = public
        models[n] = Model(n, targets[n]["J"], float(targets[n]["B"]), targets[n]["hc_diag"])
        gs[n] = np.asarray(targets[n]["gs"], dtype=np.complex128)

    rows = []
    by_key = {}
    notes = [
        "Site 0 is MSB and the left end of the open chain. Z|0>=+|0>, Z|1>=-|1>.",
        "Shell phases use the contract formula. c2=c3=0 reproduces shared-gamma QAOA with c1=beta.",
        "QAOA L=2 primary seed uses beta2=beta1+0.2*|B| and gamma2=gamma1+0.2. "
        "The beta offset is the paper's -0.2 converted by beta_ours=beta_paper*B.",
        "L=3 schedules are linear interpolations of normalized layer time. Later depths use a natural cubic spline. "
        "The optimizer is not constrained to stay smooth.",
        "Zero shell does not insert an identity layer: two zero-shell layers at gamma equal one layer at 2 gamma.",
        "Kappa transfer scales cost angles only. It is not used to train N=12.",
        f"source_sha256={source_hash()}",
        f"budget_min={args.budget_min}",
    ]

    def remember(record, psi):
        save_record(record, psi)
        rows.append(record)
        by_key[(record["n"], record["L"], record["method"])] = record

    def run_depth_block(n_list, L_list, methods):
        for n in n_list:
            if time_left() < 30:
                notes.append(f"budget stopped before N={n} L-block {list(L_list)}")
                return False
            conf = cfg_for(n)
            pool = Pool(models[n], conf["workers"])
            try:
                smoke = polish_starts(
                    pool,
                    "qaoa",
                    1,
                    [("smoke", np.zeros(2)), ("smoke2", np.array([0.2, 0.15]))],
                    maxfun=3,
                )
                if not smoke.get("ok"):
                    raise RuntimeError(f"pool smoke failed: {smoke}")
                for L in L_list:
                    here = {}
                    for method in methods:
                        if time_left() < 20:
                            notes.append(f"budget stopped at N={n} L={L} before {method}")
                            return False
                        prev = by_key.get((n, L - 1, method), {}).get("x_raw") if L > 1 else None
                        qaoa_x = here.get("qaoa", {}).get("x_raw")
                        shared_x = here.get("shared_beta", {}).get("x_raw")
                        donor = here.get("nov3_D", {}).get("x_raw") if method in ("drop_c2", "drop_c3") else None
                        try:
                            record, psi = run_one(
                                pool, models[n], gs[n], targets[n], method, L, prev, qaoa_x, shared_x, donor, log
                            )
                        except Exception as exc:  # noqa: BLE001
                            log(f"  EXCEPTION {method}: {exc}")
                            log(traceback.format_exc())
                            record, psi = {
                                "n": n,
                                "L": L,
                                "method": method,
                                "status": "failed",
                                "error": repr(exc),
                            }, None
                        # L=1 shared-beta is the same ansatz as QAOA. Publish the better energy for both.
                        if L == 1 and method == "shared_beta" and record.get("status") == "complete":
                            qrec = here["qaoa"]
                            if float(record["E"]) + 1e-8 < float(qrec["E"]):
                                notes.append(
                                    f"N={n} L=1 shared-beta search beat the QAOA grid by "
                                    f"{float(qrec['E'])-float(record['E']):.3e}; both headlines use that point."
                                )
                                qrec = dict(qrec)
                                qrec["E"] = record["E"]
                                qrec["x_raw"] = record["x_raw"]
                                qrec["F"] = record["F"]
                                qrec["energy_error"] = record["energy_error"]
                                qrec["eta"] = record["eta"]
                                qrec["best_tag"] = "adopted_shared_search"
                                qrec["x_canonical"] = record["x_canonical"]
                                here["qaoa"] = qrec
                                by_key[(n, 1, "qaoa")] = qrec
                                for i, r in enumerate(rows):
                                    if r.get("n") == n and r.get("L") == 1 and r.get("method") == "qaoa":
                                        rows[i] = qrec
                                save_record(qrec, psi)
                            elif float(qrec["E"]) <= float(record["E"]):
                                record = dict(record)
                                record["headline_source"] = "qaoa_L1_grid"
                                record["E"] = qrec["E"]
                                record["x_raw"] = qrec["x_raw"]
                                record["x_canonical"] = qrec["x_canonical"]
                                record["F"] = qrec["F"]
                                record["energy_error"] = qrec["energy_error"]
                                record["eta"] = qrec["eta"]
                                record["state_sha256"] = qrec["state_sha256"]
                                record["best_tag"] = "identical_to_qaoa_L1"
                                psi = np.load(RESULTS / "states" / f"n{n}_L1_qaoa.npy")
                        remember(record, psi)
                        here[method] = record
                    if "nov3_D" in here:
                        repair_subspaces(pool, models[n], gs[n], targets[n], here, log)
                        by_key[(n, L, "nov3_D")] = here["nov3_D"]
                        # refresh the list entry
                        for i, r in enumerate(rows):
                            if r.get("n") == n and r.get("L") == L and r.get("method") == "nov3_D":
                                rows[i] = here["nov3_D"]
                    write_bundle(rows, targets, [], notes)
            finally:
                pool.close()
        return True

    log(f"optimization budget {args.budget_min:.1f} min")
    run_depth_block((6, 12), (1, 2, 3, 4), METHODS)
    if time_left() > 60:
        run_depth_block((6, 12), (5, 6), METHODS)
    else:
        notes.append("skipped L=5..6 because the budget was exhausted")
    # Matched-parameter QAOA references beyond the candidate ladder.
    if time_left() > 60:
        run_depth_block((6, 12), (7, 8), ("qaoa",))
    else:
        notes.append("skipped QAOA L=7..8 matched-parameter references")

    transfer = []
    if 6 in targets and 12 in targets:
        scale = float(targets[6]["kappa_mean_degree"]) / float(targets[12]["kappa_mean_degree"])
        notes.append(
            f"kappa6={targets[6]['kappa_mean_degree']:.6f} kappa12={targets[12]['kappa_mean_degree']:.6f} "
            f"cost-angle scale kappa6/kappa12={scale:.6f}"
        )
        for L in range(1, 9):
            for method in METHODS:
                src = by_key.get((6, L, method))
                if src is None or src.get("status") != "complete":
                    continue
                if n_params(method, L) != len(src["x_raw"]):
                    continue
                transfer.extend(zero_shot(models[12], gs[12], targets[12], src["x_raw"], method, L, scale))
    verdict = write_bundle(rows, targets, transfer, notes)
    log(f"verdict program_success={verdict['program_success']} in {(time.perf_counter()-t_run)/60:.1f} min")
    log(verdict["headline"])


def json_load(path: Path):
    import json

    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
