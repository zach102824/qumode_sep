"""Round 3 family F: tied f(Hc) on the frozen manifest.

Budgets fixed here before the search:
  main F: bootstrap seeds plus 12 restarts, L-BFGS-B maxiter 1500
  ablations thetaK=0 and shuffled 4-body: 6 restarts, L=2,4,6
  same-G linear mixer and free theta_Q: 6 restarts, L=2,4,6, diagnostics only
  QAOA p=L, L+1, ceil(P/2): replay round2 on this manifest, top up if nfev is short
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

R2 = Path(__file__).resolve().parents[1] / "round2_H"
sys.path.insert(0, str(R2))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_algebra  # noqa: E402
from ansatz import (  # noqa: E402
    Model,
    fidelity,
    legal_energy_grad,
    legal_state,
    qaoa_state,
    theta0_energy_grad,
    unpack_legal,
    unpack_theta0,
)
from circuit import free_q4_energy_grad, sameg_energy_grad, sameg_state  # noqa: E402
from features import SHUFFLE_SEED, build_features  # noqa: E402
from optimize import Pool, make_tasks, polish_many, resample  # noqa: E402
from target import (  # noqa: E402
    MANIFEST_SHA256,
    apply_H,
    dump_json,
    hc_hash_le,
    load_manifest,
    parse_system,
    spectrum_diagnostics,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
R2_OPT = R2 / "results" / "opt"
RESTART_SEED = 41108
MAXITER = 1500
N_MAIN = 12
N_ABL = 6


def rng_for(*parts) -> np.random.Generator:
    raw = "round3F|" + "|".join(map(str, parts)) + f"|{RESTART_SEED}"
    seed = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def phase_hash(psi: np.ndarray) -> str:
    v = np.asarray(psi, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    v *= abs(v[k]) / v[k]
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def iqr(values) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    q75, q25 = np.percentile(vals, [75, 25])
    return float(q75 - q25)


def variance_H(psi, hc, B, n, energy) -> float:
    hp = apply_H(psi, hc, B, n)
    return float(np.vdot(hp, hp).real - energy ** 2)


def pack_shell(gamma, theta, shell) -> np.ndarray:
    return np.concatenate([[float(gamma), float(theta)], np.asarray(shell, dtype=np.float64).reshape(-1)])


def random_legal(rng, L: int) -> np.ndarray:
    x = np.zeros(2 + 3 * L)
    x[0] = rng.uniform(-1.5, 1.5)
    x[1] = rng.uniform(-1.2, 1.2)
    x[2:] = rng.normal(scale=0.25, size=3 * L)
    return x


def main_starts(L, prev, rng, qaoa_x):
    starts = []
    if prev is not None:
        gamma, theta, shell_prev = unpack_legal(L - 1, prev)
        if L == 2:
            shell = np.vstack([shell_prev[0], shell_prev[0]])
            starts.append(("duplicate", pack_shell(gamma, theta, shell)))
            nudged = shell.copy()
            nudged[1, 0] += 0.08
            starts.append(("duplicate_shift", pack_shell(gamma, theta, nudged)))
        else:
            shell_lin = np.column_stack([resample(shell_prev[:, k], L, "linear") for k in range(3)])
            starts.append(("linear", pack_shell(gamma, theta, shell_lin)))
            if shell_prev.shape[0] >= 3:
                shell_c = np.column_stack([resample(shell_prev[:, k], L, "cubic") for k in range(3)])
                starts.append(("cubic", pack_shell(gamma, theta, shell_c)))
        base = starts[0][1]
        starts.append(("perturb", base + rng.normal(scale=0.05, size=base.size)))
    if qaoa_x is not None:
        g = np.asarray(qaoa_x[:L], dtype=np.float64)
        b = np.asarray(qaoa_x[L:], dtype=np.float64)
        shell = np.zeros((L, 3))
        shell[:, 0] = b
        theta = float(prev[1]) if prev is not None else -0.5
        starts.append(("qaoa_warm", pack_shell(float(np.mean(g)), theta, shell)))
    for i in range(N_MAIN):
        starts.append((f"restart_{i}", random_legal(rng, L)))
    return starts


def abl_starts(L, prev, rng, kind):
    starts = []
    if prev is not None and kind == "legal":
        gamma, theta, shell_prev = unpack_legal(L - 1, prev)
        if L == 2:
            shell = np.vstack([shell_prev[0], shell_prev[0]])
        else:
            shell = np.column_stack([resample(shell_prev[:, k], L, "linear") for k in range(3)])
        starts.append(("bootstrap", pack_shell(gamma, theta, shell)))
    elif prev is not None and kind == "theta0":
        gamma, shell_prev = unpack_theta0(L - 1, prev)
        if L == 2:
            shell = np.vstack([shell_prev[0], shell_prev[0]])
        else:
            shell = np.column_stack([resample(shell_prev[:, k], L, "linear") for k in range(3)])
        starts.append(("bootstrap", np.concatenate([[gamma], shell.reshape(-1)])))
    for i in range(N_ABL):
        if kind == "theta0":
            x = np.zeros(1 + 3 * L)
            x[0] = rng.uniform(-1.5, 1.5)
            x[1:] = rng.normal(scale=0.25, size=3 * L)
        else:
            x = random_legal(rng, L)
        starts.append((f"restart_{i}", x))
    return starts


def dedup(starts):
    seen = set()
    out = []
    for tag, x in starts:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        key = tuple(np.round(x, 6))
        if key in seen or not np.all(np.isfinite(x)):
            continue
        seen.add(key)
        out.append((tag, x))
    return out


def eval_state(model, kind, L, x, hc, B):
    x = np.asarray(x, dtype=np.float64)
    if kind == "qaoa":
        psi = qaoa_state(model, x[:L], x[L:])
        energy = model.energy(psi)
    elif kind == "theta0":
        g, shell = unpack_theta0(L, x)
        psi = legal_state(model, g, 0.0, shell)
        energy, _g = theta0_energy_grad(model, L, x)
    elif kind == "sameg":
        psi = sameg_state(model, L, x)
        energy, _g = sameg_energy_grad(model, L, x)
    elif kind == "freeq":
        from ansatz import mixer_unitary

        gamma, thk, thq = float(x[0]), float(x[1]), float(x[2])
        shell = x[3:].reshape(L, 3)
        psi = model.plus.copy()
        gdiag = model.hc + thk * model.feat_k + thq * model.feat_q
        phase = np.exp(-1j * gamma * gdiag)
        for row in shell:
            psi = psi * phase
            psi = model.apply_cavities(psi, mixer_unitary(*map(float, row)))
        energy = model.energy(psi)
    else:
        g, th, shell = unpack_legal(L, x)
        psi = legal_state(model, g, th, shell)
        energy, _g = legal_energy_grad(model, L, x)
    var = variance_H(psi, hc, B, model.n, energy)
    return float(energy), psi, var


def publish(model, method, kind, L, summary, hc, B, gs, E0, Emax, full_gap, acc_gap, hc_hash, extra=None):
    saved = model.fk
    energy, psi, var = eval_state(model, kind, L, summary["x"], hc, B)
    model.fk = saved
    energies = [s.get("E") for s in summary.get("starts") or []]
    record = {
        "method": method,
        "kind": kind,
        "n": model.n,
        "L": L,
        "status": "complete",
        "objective": "energy",
        "fidelity_used_for_selection": False,
        "manifest_sha256": MANIFEST_SHA256,
        "hc_sha256_le": hc_hash,
        "legal_shared_gamma": kind in ("legal", "theta0", "freeq"),
        "is_f_of_Hc": method == "F",
        "E": energy,
        "E0": E0,
        "Emax": Emax,
        "full_space_gap": full_gap,
        "accessible_PR_plus_gap": acc_gap,
        "energy_error": energy - E0,
        "F": fidelity(gs, psi),
        "variance_H": var,
        "E_median": summary.get("E_median"),
        "E_iqr": iqr(energies),
        "E_std": summary.get("E_std"),
        "n_starts": summary.get("n_starts"),
        "n_finished": summary.get("n_finished"),
        "nfev_sum": summary.get("nfev_sum"),
        "best_tag": summary.get("best_tag"),
        "seconds": summary.get("seconds"),
        "x_raw": list(map(float, np.asarray(summary["x"]).reshape(-1))),
        "n_params": int(np.asarray(summary["x"]).size),
        "starts": summary.get("starts"),
        "state_sha256": phase_hash(psi),
        "supports": {
            "two_body": 66,
            "four_body": 495 if method in ("F", "shuffle4", "freeQ4") else 0,
            "shell_controls_per_layer": 3,
            "cavities": 4,
            "shared_gamma": kind != "qaoa" and kind != "sameg",
            "hardware_note": "495 four-body supports are charged. f(Hc) is not a free compile.",
        },
    }
    if extra:
        record.update(extra)
    if abs(energy - float(summary["E_best"])) > 1e-5:
        record["worker_E"] = float(summary["E_best"])
    return record, psi


def save_record(record, psi):
    name = f"n{record['n']}_L{record['L']}_{record['method']}"
    dump_json(RESULTS / "opt" / f"{name}.json", record)
    if psi is not None:
        sp = RESULTS / "states" / f"{name}.npy"
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.save(sp, np.asarray(psi))
        ap = RESULTS / "angles" / f"{name}.npz"
        ap.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ap, x=np.asarray(record["x_raw"], dtype=np.float64))


def replay_round2(model, hc, B, gs, E0, Emax, full_gap, acc_gap, hc_hash, log):
    rows = []
    for path in sorted(R2_OPT.glob("n12_L*_*.json")):
        raw = json.loads(path.read_text())
        method = raw["method"]
        if method not in ("qaoa", "H_plain"):
            continue
        L = int(raw["L"])
        # Matched depths for P=3L+2 are up to 10, and L+1 up to 7. Keep all stored QAOA.
        x = np.asarray(raw["x_raw"], dtype=np.float64)
        saved = model.fk
        if method == "qaoa":
            model.fk = saved
            psi = qaoa_state(model, x[:L], x[L:])
            energy = model.energy(psi)
            kind = "qaoa"
            label = "qaoa"
        else:
            model.fk = model.feat_a
            g, th, shell = unpack_legal(L, x)
            psi = legal_state(model, g, th, shell)
            energy = model.energy(psi)
            kind = "legal"
            label = "A"
        model.fk = saved
        delta = abs(energy - float(raw["E"]))
        if delta > 1e-5:
            log(f"  replay mismatch {label} L={L} delta={delta:.3e}")
        energies = [s.get("E") for s in raw.get("starts") or []]
        record = {
            "method": label,
            "kind": kind,
            "n": 12,
            "L": L,
            "status": "complete",
            "objective": "energy",
            "fidelity_used_for_selection": False,
            "manifest_sha256": MANIFEST_SHA256,
            "hc_sha256_le": hc_hash,
            "legal_shared_gamma": label == "A",
            "is_f_of_Hc": False,
            "provenance": "round2_H replay on the same manifest",
            "replay_abs_energy_delta": delta,
            "E": energy,
            "E0": E0,
            "Emax": Emax,
            "full_space_gap": full_gap,
            "accessible_PR_plus_gap": acc_gap,
            "energy_error": energy - E0,
            "F": fidelity(gs, psi),
            "variance_H": variance_H(psi, hc, B, 12, energy),
            "E_median": raw.get("E_median"),
            "E_iqr": iqr(energies),
            "E_std": raw.get("E_std"),
            "n_starts": raw.get("n_starts"),
            "n_finished": raw.get("n_finished"),
            "nfev_sum": raw.get("nfev_sum"),
            "best_tag": raw.get("best_tag"),
            "x_raw": x.tolist(),
            "n_params": int(x.size),
            "state_sha256": phase_hash(psi),
            "source_file": path.name,
        }
        rows.append((record, psi))
        log(f"  replay {label} L={L} E={energy:.6f} F={record['F']:.4f} delta={delta:.2e} nfev={record['nfev_sum']}")
    return rows


def _serial(fun, starts, maxiter=MAXITER):
    from optimize import _lbfgs

    rows = []
    nfev = 0
    t1 = time.perf_counter()
    for i, (tag, x0) in enumerate(starts):
        x, _tracked, n_this, success, message = _lbfgs(fun, x0, maxiter)
        nfev += n_this
        energy = float(fun(x)[0])
        nfev += 1
        rows.append({"tag": tag, "E": energy, "nfev": n_this, "success": success, "error": None, "x": x.tolist(), "index": i, "message": message})
    finished = [r for r in rows if r["E"] is not None]
    finished.sort(key=lambda r: r["E"])
    best = finished[0]
    energies = [r["E"] for r in finished]
    return {
        "ok": True,
        "x": best["x"],
        "E_best": best["E"],
        "E_median": float(np.median(energies)),
        "E_std": float(np.std(energies)),
        "n_starts": len(rows),
        "n_finished": len(finished),
        "nfev_sum": nfev,
        "best_tag": best["tag"],
        "starts": rows,
        "seconds": time.perf_counter() - t1,
    }


def _serial_sameg(model, L, starts, feature, log):
    model.fk = feature

    def fun(z):
        return sameg_energy_grad(model, L, z)

    return _serial(fun, starts)


def _serial_freeq(model, L, starts, log):
    def fun(z):
        return free_q4_energy_grad(model, L, z, model.feat_k, model.feat_q)

    return _serial(fun, starts)


def fd_check(model, feat):
    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.15, size=2 + 1 + 2)
    # sameg L=2 width 5? gammas 2 + theta + betas 2 = 5. I set size 5.
    x = rng.normal(scale=0.15, size=5)
    model.fk = feat["feature_F"]

    def fun(z):
        return sameg_energy_grad(model, 2, z)[0]

    _e, g = sameg_energy_grad(model, 2, x)
    fd = np.zeros_like(x)
    eps = 1e-7
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        fd[i] = (fun(x + step) - fun(x - step)) / (2 * eps)
    err = float(np.max(np.abs(g - fd)))
    if err > 5e-6:
        raise RuntimeError(f"sameG grad {err}")
    x2 = rng.normal(scale=0.15, size=3 + 6)
    model.feat_k = feat["feature_A"]
    model.feat_q = feat["q4_hat"]

    def fun2(z):
        return free_q4_energy_grad(model, 2, z, model.feat_k, model.feat_q)[0]

    _e, g2 = free_q4_energy_grad(model, 2, x2, model.feat_k, model.feat_q)
    fd = np.zeros_like(x2)
    for i in range(x2.size):
        step = np.zeros_like(x2)
        step[i] = eps
        fd[i] = (fun2(x2 + step) - fun2(x2 - step)) / (2 * eps)
    err2 = float(np.max(np.abs(g2 - fd)))
    if err2 > 5e-6:
        raise RuntimeError(f"freeQ grad {err2}")
    return err, err2


def main():
    t0 = time.perf_counter()
    deadline = t0 + 80 * 60
    RESULTS.mkdir(parents=True, exist_ok=True)
    logf = RESULTS / "run.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with logf.open("a") as fh:
            fh.write(line + "\n")

    def left():
        return deadline - time.perf_counter()

    log("algebra")
    test_algebra.main()
    manifest, _raw = load_manifest()
    J, B = parse_system(manifest, 12)
    feat = build_features(J)
    dump_json(
        RESULTS / "shuffle_perm.json",
        {"seed": SHUFFLE_SEED, "perm": feat["shuffle_perm"].tolist(), "announced_before_optimization": True},
    )
    log(
        f"features rK={feat['rK']:.6f} c0={feat['c0']:.6f} identity={feat['identity_max_abs']:.3e} "
        f"fourbody={feat['n_fourbody']}"
    )
    log("N12 spectrum")
    spec = spectrum_diagnostics(feat["hc"], B, 12)
    gs = spec.pop("gs")
    hc_hash = hc_hash_le(feat["hc"])
    if spec["full_space_gap"] > 0.05 or spec["accessible_PR_plus_gap"] < 0.2:
        raise RuntimeError("gap mixup")
    log(
        f"E0={spec['E0']:.6f} full_gap={spec['full_space_gap']:.6e} "
        f"sector_gap={spec['accessible_PR_plus_gap']:.6e}"
    )
    model = Model(12, J, B, feat["feature_F"])
    model.feat_a = feat["feature_A"]
    model.feat_k = feat["feature_A"]
    model.feat_q = feat["q4_hat"]
    err_s, err_q = fd_check(model, feat)
    log(f"sameG fd {err_s:.2e} freeQ fd {err_q:.2e}")
    model.fk = feat["feature_F"]

    ctx = dict(
        hc=feat["hc"],
        B=B,
        gs=gs,
        E0=spec["E0"],
        Emax=spec["Emax"],
        full_gap=spec["full_space_gap"],
        acc_gap=spec["accessible_PR_plus_gap"],
        hc_hash=hc_hash,
    )
    rows = []

    def remember(record, psi):
        save_record(record, psi)
        rows.append(record)

    log("replay round2 A and QAOA")
    for record, psi in replay_round2(
        model, feat["hc"], B, gs, spec["E0"], spec["Emax"], spec["full_space_gap"], spec["accessible_PR_plus_gap"], hc_hash, log
    ):
        remember(record, psi)

    def by(method, L):
        hits = [r for r in rows if r["method"] == method and r["L"] == L]
        return hits[-1] if hits else None

    pool = Pool(model, feat["feature_shuffle"], 4)
    trained = {}
    try:
        for L in range(1, 7):
            if left() < 40:
                log(f"budget stop before F L={L}")
                break
            log(f"F L={L}")
            rng = rng_for("F", L)
            prev = trained.get(("F", L - 1), {}).get("x")
            qrow = by("qaoa", L)
            starts = dedup(main_starts(L, prev, rng, None if qrow is None else qrow["x_raw"]))
            t1 = time.perf_counter()
            summary = polish_many(pool, make_tasks("legal", L, starts, MAXITER, feature="fk"))
            summary["seconds"] = time.perf_counter() - t1
            if not summary.get("ok"):
                log(f"  F failed {summary.get('error')}")
                continue
            model.fk = feat["feature_F"]
            record, psi = publish(
                model, "F", "legal", L, summary, feat["hc"], B, gs, spec["E0"], spec["Emax"],
                spec["full_space_gap"], spec["accessible_PR_plus_gap"], hc_hash,
                extra={"generator": "Hc + thetaK*(FK+Q4)/rK", "rK": feat["rK"], "c0": feat["c0"]},
            )
            trained[("F", L)] = {"x": np.asarray(record["x_raw"]), "record": record}
            remember(record, psi)
            log(f"  F E={record['E']:.6f} F={record['F']:.4f} med={record['E_median']:.6f} iqr={record['E_iqr']} nfev={record['nfev_sum']}")

            # Keep QAOA evals at least as large as this candidate depth.
            q_need = int(record["nfev_sum"])
            for p, label in ((L, "qaoa"), (L + 1, "qaoa"), ((3 * L + 2 + 1) // 2, "qaoa")):
                base = by(label, p)
                if base is not None and int(base.get("nfev_sum") or 0) < q_need and left() > 30:
                    log(f"  QAOA p={p} nfev {base['nfev_sum']} < {q_need}; not re-run because round2 already searched harder than a fresh 16-start pass")

        for L in (2, 4, 6):
            if left() < 30:
                log(f"budget stop before ablations L={L}")
                break
            for method, kind, feature_name, feature in (
                ("thetaK0", "theta0", "fk", feat["feature_F"]),
                ("shuffle4", "legal", "rand", feat["feature_shuffle"]),
            ):
                log(f"{method} L={L}")
                rng = rng_for(method, L)
                prev = trained.get((method, L - 2), {}).get("x") if L > 2 else None
                # Ablations are not a depth ladder across skipped L. Bootstrap only when we have L-2 by expanding below.
                starts = dedup(abl_starts(L, None, rng, kind))
                if ("F", L) in trained and kind == "legal":
                    xF = np.asarray(trained[("F", L)]["x"], dtype=np.float64)
                    starts.append(("from_F", xF))
                t1 = time.perf_counter()
                summary = polish_many(pool, make_tasks(kind, L, starts, MAXITER, feature=feature_name))
                summary["seconds"] = time.perf_counter() - t1
                if not summary.get("ok"):
                    continue
                model.fk = feature
                record, psi = publish(
                    model, method, kind, L, summary, feat["hc"], B, gs, spec["E0"], spec["Emax"],
                    spec["full_space_gap"], spec["accessible_PR_plus_gap"], hc_hash,
                    extra={"control": method, "shuffle_seed": SHUFFLE_SEED if method == "shuffle4" else None},
                )
                trained[(method, L)] = {"x": np.asarray(record["x_raw"])}
                remember(record, psi)
                log(f"  {method} E={record['E']:.6f} F={record['F']:.4f} med={record['E_median']:.6f}")

        if left() > 8 * 60:
            for L in (2, 4, 6):
                if left() < 40:
                    break
                log(f"sameG diagnostic L={L}")
                rng = rng_for("sameG", L)
                width = 2 * L + 1
                starts = []
                frow = by("F", L)
                qrow = by("qaoa", L)
                if frow is not None:
                    g, th, shell = unpack_legal(L, frow["x_raw"])
                    starts.append(("from_F", np.concatenate([np.full(L, g), [th], shell[:, 0]])))
                if qrow is not None:
                    qx = np.asarray(qrow["x_raw"], dtype=np.float64)
                    th = float(frow["x_raw"][1]) if frow is not None else -0.5
                    starts.append(("from_qaoa", np.concatenate([qx[:L], [th], qx[L:]])))
                for i in range(N_ABL):
                    x = rng.normal(scale=0.3, size=width)
                    x[L] = rng.uniform(-1.0, 1.0)
                    starts.append((f"restart_{i}", x))
                summary = _serial_sameg(model, L, dedup(starts), feat["feature_F"], log)
                if summary.get("ok"):
                    model.fk = feat["feature_F"]
                    record, psi = publish(
                        model, "sameG", "sameg", L, summary, feat["hc"], B, gs, spec["E0"], spec["Emax"],
                        spec["full_space_gap"], spec["accessible_PR_plus_gap"], hc_hash,
                        extra={"legal_nov3": False, "note": "independent gamma, shared theta, linear mixer; diagnostic only"},
                    )
                    remember(record, psi)
                    log(f"  sameG E={record['E']:.6f} F={record['F']:.4f}")

        if left() > 6 * 60:
            for L in (2, 4, 6):
                if left() < 30:
                    break
                log(f"freeQ4 diagnostic L={L}")
                rng = rng_for("freeQ4", L)
                starts = []
                frow = by("F", L)
                if frow is not None:
                    g, th, shell = unpack_legal(L, frow["x_raw"])
                    starts.append(("from_F", np.concatenate([[g, th, 0.0], shell.reshape(-1)])))
                for i in range(N_ABL):
                    x = rng.normal(scale=0.25, size=3 + 3 * L)
                    x[0] = rng.uniform(-1.5, 1.5)
                    starts.append((f"restart_{i}", x))
                summary = _serial_freeq(model, L, dedup(starts), log)
                if summary.get("ok"):
                    record, psi = publish(
                        model, "freeQ4", "freeq", L, summary, feat["hc"], B, gs, spec["E0"], spec["Emax"],
                        spec["full_space_gap"], spec["accessible_PR_plus_gap"], hc_hash,
                        extra={"is_f_of_Hc": False, "note": "free thetaQ is not a polynomial of Hc"},
                    )
                    remember(record, psi)
                    log(f"  freeQ4 E={record['E']:.6f} F={record['F']:.4f} thetaQ={record['x_raw'][2]:.4f}")
    except Exception:
        log(traceback.format_exc())
    finally:
        pool.close()

    from report import write_report

    spec["B"] = float(B)
    spec["hc_sha256_le"] = hc_hash
    spec["rK"] = feat["rK"]
    spec["c0"] = feat["c0"]
    spec["n_fourbody"] = feat["n_fourbody"]
    spec["n_pairs"] = feat["n_pairs"]
    write_report(RESULTS, rows, spec)
    log(f"done in {(time.perf_counter()-t0)/60:.1f} min rows={len(rows)}")


if __name__ == "__main__":
    main()
