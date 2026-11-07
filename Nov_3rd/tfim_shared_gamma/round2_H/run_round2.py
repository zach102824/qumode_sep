"""N=12 family-H homotopy, with N=6 only as a smoke test.

The legal candidate always has one shared gamma. Relaxed gamma_l values are
stored separately and are not a certified ceiling.
"""
from __future__ import annotations

import hashlib
import time
import traceback
from pathlib import Path

import numpy as np

import test_smoke
import thread_env  # noqa: F401
from ansatz import (
    fidelity,
    legal_energy_grad,
    legal_state,
    qaoa_energy_grad,
    qaoa_pack,
    qaoa_state,
    theta0_energy_grad,
    unpack_legal,
    unpack_theta0,
)
from optimize import (
    BUDGET,
    Pool,
    fourier_qaoa,
    homotopy_seeds,
    legal_starts,
    make_tasks,
    polish_many,
    qaoa_starts,
)
from target import (
    MANIFEST_SHA256,
    dump_json,
    fk_matrix,
    hc_hash_le,
    load_manifest,
    normalize_feature,
    pair_diagonal,
    parse_system,
    random_feature,
    spectrum_diagnostics,
    upper_rms,
    z_table,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def matched_p(L: int) -> int:
    return int(np.ceil((3 * L + 2) / 2.0))


def phase_hash(psi: np.ndarray) -> str:
    v = np.asarray(psi, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    v *= abs(v[k]) / v[k]
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def build(n: int, manifest):
    from ansatz import Model

    J, B = parse_system(manifest, n)
    z = z_table(n)
    hc = pair_diagonal(J, z)
    K = fk_matrix(J)
    rms_J = upper_rms(J)
    Kn, rms_K, scale = normalize_feature(K, rms_J)
    fk = pair_diagonal(Kn, z)
    Kr = random_feature(n, rms_J, seed=20261107)
    fk_rand = pair_diagonal(Kr, z)
    model = Model(n, J, B, fk)
    meta = {
        "rms_J": rms_J,
        "rms_K": rms_K,
        "fk_scale": scale,
        "rms_K_normalized": upper_rms(Kn),
        "rms_random": upper_rms(Kr),
        "hc_sha256_le": hc_hash_le(hc),
        "B_hex": float(B).hex(),
        "manifest_sha256": MANIFEST_SHA256,
    }
    return model, fk_rand, meta, hc


def realize(model, kind, L, x, feature="fk"):
    saved = model.fk
    model.fk = model.fk if feature == "fk" else feature
    x = np.asarray(x, dtype=np.float64)
    if kind == "qaoa":
        psi = qaoa_state(model, x[:L], x[L:])
        E, gg, gb = qaoa_energy_grad(model, x[:L], x[L:])
        grad = float(np.linalg.norm(np.concatenate([gg, gb])))
    elif kind == "theta0":
        E, g = theta0_energy_grad(model, L, x)
        gma, shell = unpack_theta0(L, x)
        psi = legal_state(model, gma, 0.0, shell)
        grad = float(np.linalg.norm(g))
    else:
        E, g = legal_energy_grad(model, L, x)
        gma, theta, shell = unpack_legal(L, x)
        psi = legal_state(model, gma, theta, shell)
        grad = float(np.linalg.norm(g))
    model.fk = saved
    return float(E), psi, grad


def publish(model, spec, summary, seconds, extra=None):
    kind = spec["kind"]
    feature = spec.get("feature_diag")
    saved = model.fk
    if feature is not None:
        model.fk = feature
    E, psi, grad = realize(model, kind if kind != "homotopy" else "legal", spec["L"], summary["x"])
    model.fk = saved
    if abs(E - float(summary["E_best"])) > 1e-6:
        # Trust the parent recomputation.
        summary["E_worker"] = summary["E_best"]
        summary["E_best"] = E
    record = {
        "method": spec["method"],
        "kind": kind,
        "n": model.n,
        "L": spec["L"],
        "legal_shared_gamma": kind != "qaoa" and kind != "relaxed",
        "status": "complete",
        "objective": "energy",
        "fidelity_used_for_selection": False,
        "manifest_sha256": MANIFEST_SHA256,
        "hc_sha256_le": spec["hc_sha256_le"],
        "E": E,
        "E0": spec["E0"],
        "Emax": spec["Emax"],
        "full_space_gap": spec["full_space_gap"],
        "accessible_PR_plus_gap": spec["accessible_gap"],
        "energy_error": E - spec["E0"],
        "F": fidelity(spec["gs"], psi),
        "eta": (E - spec["Emax"]) / (spec["E0"] - spec["Emax"]),
        "E_median": summary["E_median"],
        "E_std": summary.get("E_std"),
        "n_starts": summary["n_starts"],
        "n_finished": summary["n_finished"],
        "nfev_sum": summary["nfev_sum"],
        "best_tag": summary.get("best_tag"),
        "grad_norm": grad,
        "seconds": seconds,
        "x_raw": list(map(float, np.asarray(summary["x"]).reshape(-1))),
        "starts": summary.get("starts"),
        "state_sha256": phase_hash(psi),
        "not_a_certified_ceiling": True,
    }
    if extra:
        record.update(extra)
    return record, psi


def save_record(record, psi):
    out = RESULTS / "opt" / f"n{record['n']}_L{record['L']}_{record['method']}.json"
    dump_json(out, record)
    if psi is not None:
        sp = RESULTS / "states" / f"n{record['n']}_L{record['L']}_{record['method']}.npy"
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.save(sp, np.asarray(psi))
        ap = RESULTS / "angles" / f"n{record['n']}_L{record['L']}_{record['method']}.npz"
        ap.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ap, x=np.asarray(record["x_raw"], dtype=np.float64))


def merge_summary(base, extra):
    """Keep the lower energy and add the new starts into the median and the eval count."""
    if not extra.get("ok"):
        base["nfev_sum"] = int(base.get("nfev_sum") or 0) + int(extra.get("nfev_sum") or 0)
        return base
    adopted = float(extra["E_best"]) < float(base["E_best"]) - 1e-12
    if adopted:
        base["x"] = extra["x"]
        base["E_best"] = extra["E_best"]
        base["best_tag"] = extra.get("best_tag")
        base["best"] = extra.get("best")
    starts = list(base.get("starts") or []) + list(extra.get("starts") or [])
    energies = [s["E"] for s in starts if s.get("E") is not None]
    base["starts"] = starts
    base["n_starts"] = len(starts)
    base["n_finished"] = len(energies)
    base["E_median"] = float(np.median(energies)) if energies else base.get("E_median")
    base["E_std"] = float(np.std(energies)) if energies else base.get("E_std")
    base["nfev_sum"] = int(base.get("nfev_sum") or 0) + int(extra.get("nfev_sum") or 0)
    base["adopted_extra"] = adopted
    return base


def main():
    t0 = time.perf_counter()
    deadline = t0 + 78 * 60
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "run.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    def left():
        return deadline - time.perf_counter()

    log("N6 smoke")
    smoke = test_smoke.main()
    dump_json(RESULTS / "smoke_n6.json", smoke)
    manifest, _raw = load_manifest()
    model, fk_rand, meta, hc = build(12, manifest)
    log("N12 spectrum")
    spec = spectrum_diagnostics(hc, model.B, 12)
    if not (spec["full_space_gap"] < 0.05 and spec["accessible_PR_plus_gap"] > 0.2):
        raise RuntimeError(f"gap resolution failed: full {spec['full_space_gap']} sector {spec['accessible_PR_plus_gap']}")
    if spec["lowest_states"][1]["P_plus_weight"] > 1e-6:
        raise RuntimeError("E1 is not the opposite-parity partner")
    gs = spec.pop("gs")
    public_spec = {k: v for k, v in spec.items()}
    public_spec.update(meta)
    public_spec["B"] = model.B
    dump_json(RESULTS / "diagnostics_n12.json", public_spec)
    np.savez(RESULTS / "diagnostics_n12.npz", hc=hc, fk=model.fk, fk_rand=fk_rand, gs=gs, J=model.J)
    log(
        f"N12 B={model.B} E0={spec['E0']:.6f} full_gap={spec['full_space_gap']:.6e} "
        f"sector_gap={spec['accessible_PR_plus_gap']:.6e} S={spec['entropy_bits']:.6f}"
    )
    ctx = {
        "E0": spec["E0"],
        "Emax": spec["Emax"],
        "full_space_gap": spec["full_space_gap"],
        "accessible_gap": spec["accessible_PR_plus_gap"],
        "gs": gs,
        "hc_sha256_le": meta["hc_sha256_le"],
    }
    model.E0 = spec["E0"]
    model.Emax = spec["Emax"]
    model.gs = gs

    rows = []
    qaoa = {}
    plain = {}
    homo = {}
    theta0 = {}
    randk = {}
    pool = Pool(model, fk_rand, BUDGET["workers"])
    notes = [
        "Manifest SHA256 " + MANIFEST_SHA256,
        "Hc diagonal SHA256 little-endian float64 " + meta["hc_sha256_le"],
        f"FK scale rms_J/rms_K = {meta['fk_scale']:.12g} (full upper triangle, zeros included).",
        "Homotopy independent-gamma stages are initializers. They are not a certified ceiling and not legal candidates.",
        "QAOA beta2 = beta1 + 0.2*|B| converts the paper offset because beta multiplies sum X and B is not in the mixer.",
        "N6 is a smoke test only. N12 is the comparison.",
    ]

    def remember(record, psi):
        save_record(record, psi)
        rows.append(record)

    try:
        def run_qaoa_depth(p):
            if p in qaoa or left() < 30:
                return
            log(f"QAOA p={p}")
            rng = np.random.default_rng(3000 + p)
            t1 = time.perf_counter()
            if p == 1:
                betas = np.linspace(0.0, 0.5 * np.pi, BUDGET["qaoa_grid_b"], endpoint=False)
                gammas = np.linspace(BUDGET["qaoa_g_lo"], BUDGET["qaoa_g_hi"], BUDGET["qaoa_grid_g"])
                xs = [qaoa_pack([g], [b]) for g in gammas for b in betas]
                tasks = [
                    {"kind": "qaoa", "L": 1, "index": i, "x0": x, "feature": "fk"}
                    for i, x in enumerate(xs)
                ]
                energies = pool.map_energy(tasks)
                order = np.argsort(energies)
                starts = []
                for i in order[:12]:
                    starts.append((f"grid_{int(i)}", xs[int(i)]))
                for i in range(6):
                    starts.append(("random", qaoa_pack([rng.uniform(-3, 3)], [rng.uniform(0, np.pi / 2)])))
                summary = polish_many(pool, make_tasks("qaoa", 1, starts, BUDGET["qaoa_maxfun_low"]))
                summary["nfev_sum"] += int(len(xs))
                summary["grid_min_E"] = float(np.min(energies))
                summary["grid_points"] = len(xs)
            else:
                cap = BUDGET["qaoa_starts_low"] if p <= 6 else BUDGET["qaoa_starts_high"]
                maxfun = BUDGET["qaoa_maxfun_low"] if p <= 6 else BUDGET["qaoa_maxfun_high"]
                prev = qaoa.get(p - 1, {}).get("x")
                starts = qaoa_starts(p, prev, model.B, rng, cap)
                summary = polish_many(pool, make_tasks("qaoa", p, starts, maxfun))
            if not summary.get("ok"):
                log(f"  QAOA p={p} failed {summary.get('error')}")
                return
            record, psi = publish(
                model,
                {"method": "qaoa", "kind": "qaoa", "L": p, "feature_diag": None, **ctx},
                summary,
                time.perf_counter() - t1,
            )
            record["grid_points"] = summary.get("grid_points")
            qaoa[p] = {**summary, "record": record, "psi": psi}
            remember(record, psi)
            log(f"  QAOA p={p} E={record['E']:.6f} F={record['F']:.4f} med={record['E_median']:.6f} nfev={record['nfev_sum']}")

        def ensure_qaoa(pmax):
            for p in range(1, pmax + 1):
                if left() < 25:
                    notes.append(f"budget stopped before QAOA p={p}")
                    return False
                run_qaoa_depth(p)
            return True

        def topup_qaoa(p, min_nfev, label):
            if p not in qaoa or left() < 20:
                return
            guard = 0
            while qaoa[p]["nfev_sum"] < min_nfev and guard < 6 and left() > 20:
                guard += 1
                need = min_nfev - qaoa[p]["nfev_sum"]
                n_starts = max(4, min(16, int(np.ceil(need / 80.0))))
                rng = np.random.default_rng(8000 + 100 * p + guard)
                base = np.asarray(qaoa[p]["x"], dtype=np.float64)
                starts = [(f"{label}_near_{i}", base + rng.normal(scale=0.12, size=base.size)) for i in range(max(2, n_starts // 2))]
                starts += [(f"{label}_fourier_{i}", fourier_qaoa(rng, p)) for i in range(max(2, n_starts // 4))]
                starts += [(f"{label}_wide_{i}", rng.normal(scale=0.5, size=2 * p)) for i in range(max(2, n_starts // 4))]
                extra = polish_many(pool, make_tasks("qaoa", p, starts, BUDGET["qaoa_maxfun_low"] if p <= 6 else BUDGET["qaoa_maxfun_high"]))
                merged = merge_summary(qaoa[p], extra)
                qaoa[p] = merged
                record, psi = publish(
                    model,
                    {"method": "qaoa", "kind": "qaoa", "L": p, "feature_diag": None, **ctx},
                    qaoa[p],
                    qaoa[p]["record"]["seconds"] if "record" in qaoa[p] else 0.0,
                )
                qaoa[p]["record"] = record
                qaoa[p]["psi"] = psi
                # replace stored row
                for i, r in enumerate(rows):
                    if r.get("method") == "qaoa" and r.get("L") == p:
                        rows[i] = record
                save_record(record, psi)
                log(f"  topup {label} QAOA p={p} E={record['E']:.6f} nfev={record['nfev_sum']} target={min_nfev}")

        for L in range(1, 7):
            if left() < 40:
                notes.append(f"budget stopped before candidate L={L}")
                break
            need = max(L + 1, matched_p(L))
            log(f"== L={L} need QAOA through p={need}")
            ensure_qaoa(need)

            # thetaK = 0 ablation, plain bootstrap.
            log(f"thetaK=0 L={L}")
            rng = np.random.default_rng(4000 + L)
            t1 = time.perf_counter()
            if L == 1:
                xs = []
                tags = []
                for i in range(400):
                    xs.append(np.array([rng.uniform(-2, 2), rng.uniform(-0.8, 0.8), rng.normal(scale=0.3), rng.normal(scale=0.3)]))
                    tags.append("screen")
                if 1 in qaoa:
                    b = float(np.asarray(qaoa[1]["x"])[1])
                    g = float(np.asarray(qaoa[1]["x"])[0])
                    for c2 in (-0.4, 0, 0.4):
                        for c3 in (-0.4, 0, 0.4):
                            xs.append(np.array([g, b, c2, c3]))
                            tags.append("qaoa_shell")
                tasks = [
                    {"kind": "theta0", "L": 1, "index": i, "x0": x, "feature": "fk"}
                    for i, x in enumerate(xs)
                ]
                energies = pool.map_energy(tasks)
                keep = []
                order = np.argsort(energies)[: BUDGET["l1_keep"]]
                for i in order:
                    keep.append((tags[int(i)], xs[int(i)]))
                for i, tag in enumerate(tags):
                    if tag != "screen":
                        keep.append((tag, xs[i]))
                summary = polish_many(pool, make_tasks("theta0", 1, keep, BUDGET["theta0_maxfun"]))
                summary["nfev_sum"] += len(xs)
            else:
                starts = legal_starts(L, theta0.get(L - 1, {}).get("x"), qaoa.get(L, {}).get("x"), None, model.B, rng, BUDGET["theta0_starts"], kind="theta0")
                summary = polish_many(pool, make_tasks("theta0", L, starts, BUDGET["theta0_maxfun"]))
            if summary.get("ok"):
                record, psi = publish(
                    model,
                    {"method": "thetaK0", "kind": "theta0", "L": L, "feature_diag": None, **ctx},
                    summary,
                    time.perf_counter() - t1,
                )
                theta0[L] = {**summary, "record": record}
                remember(record, psi)
                log(f"  theta0 E={record['E']:.6f} F={record['F']:.4f} med={record['E_median']:.6f}")

            log(f"plain full-K L={L}")
            rng = np.random.default_rng(5000 + L)
            t1 = time.perf_counter()
            if L == 1:
                xs = []
                for i in range(BUDGET["l1_screen"]):
                    xs.append(
                        np.array(
                            [
                                rng.uniform(-2, 2),
                                rng.uniform(-1.2, 1.2),
                                rng.uniform(-0.8, 0.8),
                                rng.normal(scale=0.3),
                                rng.normal(scale=0.3),
                            ]
                        )
                    )
                if 1 in qaoa:
                    qx = np.asarray(qaoa[1]["x"])
                    for th in (-0.8, -0.3, 0.0, 0.3, 0.8):
                        xs.append(np.array([float(qx[0]), th, float(qx[1]), 0.0, 0.0]))
                tasks = [{"kind": "legal", "L": 1, "index": i, "x0": x, "feature": "fk"} for i, x in enumerate(xs)]
                energies = pool.map_energy(tasks)
                starts = [(f"screen_{int(i)}", xs[int(i)]) for i in np.argsort(energies)[: BUDGET["l1_keep"]]]
                if 1 in qaoa:
                    qx = np.asarray(qaoa[1]["x"])
                    for th in (-0.5, 0.0, 0.5):
                        starts.append(("qaoa_th", np.array([float(qx[0]), th, float(qx[1]), 0.0, 0.0])))
                summary = polish_many(pool, make_tasks("legal", 1, starts, BUDGET["legal_maxfun"]))
                summary["nfev_sum"] += len(xs)
            else:
                starts = legal_starts(
                    L,
                    plain.get(L - 1, {}).get("x"),
                    qaoa.get(L, {}).get("x"),
                    theta0.get(L, {}).get("x"),
                    model.B,
                    rng,
                    BUDGET["legal_starts"],
                    kind="legal",
                )
                summary = polish_many(pool, make_tasks("legal", L, starts, BUDGET["legal_maxfun"]))
            if summary.get("ok"):
                record, psi = publish(
                    model,
                    {"method": "H_plain", "kind": "legal", "L": L, "feature_diag": None, **ctx},
                    summary,
                    time.perf_counter() - t1,
                    extra={"optimizer": "pagano_bootstrap_restarts", "theta_free": True},
                )
                plain[L] = {**summary, "record": record}
                remember(record, psi)
                log(f"  plain E={record['E']:.6f} F={record['F']:.4f} med={record['E_median']:.6f} nfev={record['nfev_sum']}")

            if L == 1 and L in plain:
                homo[L] = dict(plain[L])
                homo[L]["vacuous_L1"] = True
                log("  homotopy L=1 is vacuous (one gamma)")
            elif L in plain and left() > 30:
                log(f"homotopy L={L}")
                rng = np.random.default_rng(6000 + L)
                seeds = homotopy_seeds(L, plain.get(L - 1, {}).get("x"), plain[L]["x"], theta0.get(L, {}).get("x"), rng)
                seeds = [s for s in seeds if np.asarray(s[1]).size == (L + 1 + 3 * L)]
                tasks = []
                for i, (tag, x) in enumerate(seeds):
                    tasks.append(
                        {
                            "index": i,
                            "tag": tag,
                            "L": L,
                            "x0": np.asarray(x, dtype=np.float64),
                            "feature": "fk",
                            "lambdas": list(BUDGET["lambdas"]),
                            "lambda_maxfun": list(BUDGET["lambda_maxfun"]),
                            "polish_maxfun": BUDGET["polish_maxfun"],
                        }
                    )
                t1 = time.perf_counter()
                hrows = pool.map_homotopy(tasks)
                summary = {
                    "ok": any(r.get("E") is not None for r in hrows),
                    "starts": [
                        {
                            "tag": r.get("tag"),
                            "E": r.get("E"),
                            "nfev": r.get("nfev"),
                            "success": r.get("success"),
                            "error": r.get("error"),
                            "relaxed_E_after_lambda0": r.get("relaxed_E_after_lambda0"),
                            "gamma_spread_before_projection": r.get("gamma_spread_before_projection"),
                        }
                        for r in hrows
                    ],
                }
                finished = [r for r in hrows if r.get("E") is not None]
                if finished:
                    finished.sort(key=lambda r: r["E"])
                    best = finished[0]
                    energies = [r["E"] for r in finished]
                    relaxed_Es = [r["relaxed_E_after_lambda0"] for r in finished if r.get("relaxed_E_after_lambda0") is not None]
                    summary.update(
                        {
                            "ok": True,
                            "x": best["x"],
                            "E_best": best["E"],
                            "E_median": float(np.median(energies)),
                            "E_std": float(np.std(energies)),
                            "n_starts": len(hrows),
                            "n_finished": len(finished),
                            "nfev_sum": int(sum(r.get("nfev") or 0 for r in hrows)),
                            "best_tag": best.get("tag"),
                            "relaxed_best_E": float(min(relaxed_Es)) if relaxed_Es else None,
                            "best_stages": best.get("stages"),
                        }
                    )
                    record, psi = publish(
                        model,
                        {"method": "H_homotopy", "kind": "homotopy", "L": L, "feature_diag": None, **ctx},
                        summary,
                        time.perf_counter() - t1,
                        extra={
                            "optimizer": "lambda_homotopy_then_exact_shared_gamma_polish",
                            "relaxed_best_E": summary["relaxed_best_E"],
                            "relaxed_is_certified_ceiling": False,
                            "best_stages": summary["best_stages"],
                            "lambdas": list(BUDGET["lambdas"]),
                        },
                    )
                    homo[L] = {**summary, "record": record}
                    remember(record, psi)
                    log(
                        f"  homotopy E={record['E']:.6f} F={record['F']:.4f} med={record['E_median']:.6f} "
                        f"relaxedE={summary['relaxed_best_E']} nfev={record['nfev_sum']}"
                    )

            candidate_nfev = 0
            if L in homo:
                candidate_nfev = int(homo[L]["nfev_sum"])
            elif L in plain:
                candidate_nfev = int(plain[L]["nfev_sum"])
            for p in sorted({L, L + 1, matched_p(L)}):
                topup_qaoa(p, candidate_nfev, "fair")

            best_legal = None
            for bucket, name in ((homo, "homotopy"), (plain, "plain")):
                if L in bucket:
                    if best_legal is None or bucket[L]["E_best"] < best_legal[0]:
                        best_legal = (bucket[L]["E_best"], name, bucket[L])
            if best_legal and L in qaoa and best_legal[0] < qaoa[L]["E_best"] - 1e-4 and left() > 40:
                log(f"rescue QAOA because {best_legal[1]} beats equal-L QAOA")
                base_starts = max(qaoa[L]["n_starts"], 1)
                topup_qaoa(L, qaoa[L]["nfev_sum"] + 3 * base_starts * 60, "rescue4x")
                for p in (L + 1, matched_p(L)):
                    if p in qaoa:
                        topup_qaoa(p, qaoa[p]["nfev_sum"] + 2 * base_starts * 40, "rescue")

            if L in (2, 4, 6) and left() > 8 * 60:
                log(f"random-K plain L={L}")
                rng = np.random.default_rng(7000 + L)
                t1 = time.perf_counter()
                starts = legal_starts(L, None, qaoa.get(L, {}).get("x"), None, model.B, rng, BUDGET["random_starts"], kind="legal")
                summary = polish_many(pool, make_tasks("legal", L, starts, BUDGET["random_maxfun"], feature="rand"))
                if summary.get("ok"):
                    record, psi = publish(
                        model,
                        {"method": "randomK", "kind": "legal", "L": L, "feature_diag": fk_rand, **ctx},
                        summary,
                        time.perf_counter() - t1,
                        extra={"feature": "gaussian_K_same_rms", "seed": 20261107},
                    )
                    randk[L] = {**summary, "record": record}
                    remember(record, psi)
                    log(f"  randomK E={record['E']:.6f} F={record['F']:.4f}")

            dump_json(RESULTS / "summary_partial.json", {"notes": notes, "n_rows": len(rows)})
    except Exception:
        log(traceback.format_exc())
        notes.append("exception; partial results saved")
    finally:
        pool.close()

    from report import write_report

    spec["B"] = float(model.B)
    write_report(RESULTS, rows, notes, spec, meta)
    log(f"done in {(time.perf_counter()-t0)/60:.1f} min, rows={len(rows)}")


if __name__ == "__main__":
    main()
