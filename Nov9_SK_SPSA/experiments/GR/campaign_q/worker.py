"""One cold Nov3rd or QAOA cell. SPSA and the original circuits live in the GR package."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

GR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import minimize_scalar

from common import RESULT_DIR
from common_campaign import (
    CACHE,
    INSTANCES,
    RECIPE,
    allocation,
    assert_allowed_seed,
    job_id,
    load_prereg,
    prereg_sha,
)
from hamiltonian import apply_H_site, make_J, pair_diag, plus_state, sha256_f64, upper, z_table
from model import pop_phase_logical
from r2_b3 import (
    METHOD_ID,
    _rng,
    energy_of,
    entropy,
    features,
    fidelity,
    ground_state,
    n_controls,
    nov3_psi,
    packing,
    qaoa_psi,
    run_matched,
    sigmas_of,
    spectrum,
    x0_of,
)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _lock(path: Path, timeout: float = 7200.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                holder = int(path.read_text().strip() or "0")
            except (OSError, ValueError):
                holder = 0
            if holder and not _pid_alive(holder):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if time.time() > deadline:
                raise TimeoutError(f"lock timeout {path}")
            time.sleep(0.4)


def _unlock(path: Path) -> None:
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def search_bstar(J: np.ndarray, n: int) -> dict:
    """Primary scan matches the agreed 60-point interior refine. Edges extend."""
    hc = pair_diag(J, z_table(n))
    calls = {"n": 0}

    def ent(b: float) -> float:
        calls["n"] += 1
        return entropy(ground_state(hc, float(b), n), n)

    lo, hi, ngrid = 0.05, 3.0, 60
    windows = []
    for ext in range(7):
        if hi - lo < 1e-8:
            return {"ok": False, "reason": "window collapsed", "gs_calls": calls["n"], "windows": windows}
        grid = np.linspace(lo, hi, ngrid)
        values = [ent(float(b)) for b in grid]
        k = int(np.argmax(values))
        windows.append({"lo": float(lo), "hi": float(hi), "n": int(ngrid), "argmax": k, "extension": ext})
        if 0 < k < len(grid) - 1:
            bracket = (float(grid[k - 1]), float(grid[k + 1]))
            refined = minimize_scalar(lambda b: -ent(float(b)), bounds=bracket, method="bounded", options={"xatol": 1e-6})
            if not getattr(refined, "success", False):
                return {"ok": False, "reason": "refine failed", "gs_calls": calls["n"], "windows": windows}
            return {
                "ok": True,
                "B_star": float(refined.x),
                "gs_calls": calls["n"],
                "windows": windows,
                "interior": True,
            }
        if ext == 6:
            return {"ok": False, "reason": "boundary unresolved", "gs_calls": calls["n"], "windows": windows}
        if k == 0:
            new_hi = float(grid[1])
            new_lo = max(float(grid[0]) / 4.0, 1e-4)
            if new_lo >= float(grid[0]) - 1e-15:
                return {"ok": False, "reason": "cannot extend below", "gs_calls": calls["n"], "windows": windows}
            lo, hi = new_lo, new_hi
        else:
            new_lo = float(grid[-2])
            new_hi = min(float(grid[-1]) * 2.0, 48.0)
            if new_hi <= float(grid[-1]) + 1e-15:
                return {"ok": False, "reason": "cannot extend above", "gs_calls": calls["n"], "windows": windows}
            lo, hi = new_lo, new_hi
        ngrid = 40
    return {"ok": False, "reason": "unreachable", "gs_calls": calls["n"], "windows": windows}


def _reference_bstar() -> dict:
    path = RESULT_DIR / "r2_b3" / "bstar.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {(int(row["N"]), int(row["seed"])): row for row in data["instances"]}


def prepare_instance(seed: int, n: int, protocol_sha: str) -> dict:
    assert_allowed_seed(seed, "dev" if seed < 11000 else "final")
    if 10010 <= seed <= 10031 or 11000 <= seed <= 11031:
        raise RuntimeError("refusing a seed this cloud does not own")
    meta_path = INSTANCES / f"n{n}_s{seed}.json"
    npz_path = CACHE / f"n{n}_s{seed}.npz"
    if meta_path.exists() and npz_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == "ok" and meta.get("protocol_sha") == protocol_sha:
            return meta
    lock = CACHE / f"n{n}_s{seed}.lock"
    _lock(lock)
    try:
        if meta_path.exists() and npz_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("status") == "ok" and meta.get("protocol_sha") == protocol_sha:
                return meta
        t0 = time.perf_counter()
        J = make_J(seed, n)
        found = search_bstar(J, n)
        if not found["ok"]:
            found["status"] = "failed"
            found["seed"] = seed
            found["N"] = n
            found["seconds"] = time.perf_counter() - t0
            return found
        ref = _reference_bstar().get((n, seed))
        b_hat = float(found["B_star"])
        if ref is not None:
            b_ref = float(ref["B_star"])
            if abs(b_hat - b_ref) > 1e-5:
                return {
                    "status": "failed",
                    "reason": f"B* {b_hat} disagrees with stored R2 {b_ref}",
                    "seed": seed,
                    "N": n,
                    "gs_calls": found["gs_calls"],
                }
            b_used = b_ref
            source = "r2_b3_bstar_json_exact"
        else:
            b_used = b_hat
            source = "refined_scan"
        hc = pair_diag(J, z_table(n))
        try:
            spec = spectrum(hc, b_used, n)
        except RuntimeError as exc:
            return {"status": "failed", "reason": str(exc), "seed": seed, "N": n, "B_star": b_used}
        problems = []
        if spec["gs_min"] <= -1e-8:
            problems.append("gs not positive")
        if spec["gs_parity"] < 1.0 - 1e-6:
            problems.append("gs not even")
        if spec["residual"] > 1e-7:
            problems.append(f"residual {spec['residual']}")
        if spec["gap"] <= 0.0 or spec["delta_odd"] <= 0.0 or spec["delta_even"] <= 0.0:
            problems.append("gap")
        if problems:
            return {
                "status": "failed",
                "reason": "; ".join(problems),
                "seed": seed,
                "N": n,
                "B_star": b_used,
                "residual": spec["residual"],
                "gap": spec["gap"],
            }
        meta = {
            "status": "ok",
            "recipe": RECIPE,
            "protocol_sha": protocol_sha,
            "seed": seed,
            "N": n,
            "B_star": b_used,
            "B_star_recomputed": b_hat,
            "B_star_abs_diff": abs(b_hat - b_used),
            "B_source": source,
            "spsa_b_key": int(round(10.0 * b_used)),
            "cut": list(range(n // 2)),
            "S_A": spec["S_A"],
            "entropy_units": "nats",
            "E0": spec["E0"],
            "E1": spec["E1"],
            "gap": spec["gap"],
            "delta_odd": spec["delta_odd"],
            "delta_even": spec["delta_even"],
            "E1_is_odd": spec["E1_is_odd"],
            "near_degenerate_gap_lt_1e-4": bool(spec["gap"] < 1e-4),
            "F_plus": spec["F_plus"],
            "E_plus": -float(b_used) * n,
            "residual": spec["residual"],
            "gs_min": spec["gs_min"],
            "gs_parity": spec["gs_parity"],
            "J_sha256": sha256_f64(upper(J)),
            "J_upper": upper(J).tolist(),
            "gs_calls": found["gs_calls"],
            "windows": found["windows"],
            "prep_seconds": time.perf_counter() - t0,
            "field_note": "per-instance entropy maximizer, not a thermodynamic critical field",
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"n{n}_s{seed}.npz.tmp.npz"
        np.savez(tmp, J=J, gs=np.asarray(spec["gs"], dtype=np.float64), hc=hc)
        os.replace(tmp, npz_path)
        from common_campaign import atomic_write

        atomic_write(meta_path, json.dumps(_clean(meta), indent=2) + "\n")
        return meta
    finally:
        _unlock(lock)


def _propagate(method: str, L: int, packed):
    groups, d_k, d_m, hc, b, n = packed

    def propagate(x):
        if method == "Nov3rd":
            psi = nov3_psi(x, L, hc, d_k, d_m, groups, n)
        elif method == "QAOA":
            psi = qaoa_psi(x, L, hc, n)
        else:
            raise RuntimeError(method)
        return energy_of(psi, hc, b, n), psi

    return propagate


def execute(job: dict) -> dict:
    """Run one cell. Returns a JSON-safe record and does not write the run file."""
    if job.get("kind") == "noop":
        return {"status": "ok", "kind": "noop", "echo": job.get("echo")}
    try:
        return _execute(job)
    except Exception:
        return {
            "status": "failed",
            "retryable": True,
            "recipe": RECIPE,
            "job_id": job_id(job) if "N" in job else "",
            "seed": job.get("seed"),
            "N": job.get("N"),
            "L": job.get("L"),
            "q": job.get("q"),
            "method": job.get("method"),
            "stage": job.get("stage"),
            "attempt": job.get("attempt", 1),
            "traceback": traceback.format_exc(),
        }


def _execute(job: dict) -> dict:
    prereg = load_prereg()
    sha = prereg_sha()
    if job.get("protocol_sha") != sha:
        raise RuntimeError("job protocol hash does not match prereg.json")
    stage = str(job["stage"])
    seed = int(job["seed"])
    n = int(job["N"])
    L = int(job["L"])
    q = int(job["q"])
    method = str(job["method"])
    assert_allowed_seed(seed, stage)
    if n not in prereg["N"] or L not in prereg["L"] or method not in prereg["methods"]:
        raise RuntimeError("job is outside the frozen grid")
    if q not in prereg["budget"]["q_candidates"]:
        raise RuntimeError(f"q={q} is not in the frozen halving grid")
    alloc = allocation(q, L)
    if alloc is None:
        raise RuntimeError(f"q={q} L={L} is below the accounting floor")
    if stage == "final":
        freeze_path = CACHE.parent / f"FREEZE_n{n}.json"
        if not freeze_path.exists():
            raise RuntimeError(f"final seed opened before FREEZE_n{n}.json")
        freeze = json.loads(freeze_path.read_text())
        if freeze.get("protocol_sha") != sha or freeze.get("retuned") is not False or freeze.get("recipe") != RECIPE:
            raise RuntimeError("freeze file does not match v1")
    meta = prepare_instance(seed, n, sha)
    base = {
        "recipe": RECIPE,
        "protocol_sha": sha,
        "job_id": job_id(job),
        "stage": stage,
        "seed": seed,
        "N": n,
        "L": L,
        "q": q,
        "Q": alloc["Q"],
        "method": method,
        "attempt": int(job.get("attempt", 1)),
        "warm_start": False,
        "transfer_queries": 0,
        "optimizer": "SPSA",
        "fidelity_oracle_in_loop": False,
    }
    if meta.get("status") != "ok":
        base.update({"status": "failed", "retryable": True, "prep": meta, "reason": meta.get("reason", "prep failed")})
        return _clean(base)
    with np.load(CACHE / f"n{n}_s{seed}.npz") as data:
        J = np.asarray(data["J"], dtype=np.float64)
        gs = np.asarray(data["gs"], dtype=np.float64)
        hc = np.asarray(data["hc"], dtype=np.float64)
    b = float(meta["B_star"])
    groups, d_k, d_m, hc = features(J, n, hc)
    if groups != packing(n):
        raise RuntimeError("packing drifted")
    propagate = _propagate(method, L, (groups, d_k, d_m, hc, b, n))
    sig = sigmas_of(method, L)
    x0s = []
    rngs = []
    for s in range(4):
        rngs.append(_rng(seed, n, b, L, METHOD_ID[method], s, 0))
        init_rng = None if s == 0 else _rng(seed, n, b, L, METHOD_ID[method], s, 2)
        x0s.append(x0_of(method, L, "E" if s == 0 else "R", init_rng))
    t0 = time.perf_counter()
    out = run_matched(propagate, x0s, sig, alloc["Q"], rngs)
    seconds = time.perf_counter() - t0
    if out["queries"]["total"] != alloc["Q"] or out["queries"]["monitoring"] != 0:
        raise RuntimeError(f"ledger {out['queries']} != {alloc['Q']}")
    e_audit = energy_of(out["psi"], hc, b, n)
    if abs(e_audit - out["E"]) > 1e-8:
        raise RuntimeError(f"audit energy {e_audit} != {out['E']}")
    F = fidelity(gs, out["psi"])
    denom = 1.0 - float(meta["F_plus"])
    eta = (F - float(meta["F_plus"])) / denom if denom > 1e-15 else None
    span = float(meta["E_plus"]) - float(meta["E0"])
    eta_e = (float(meta["E_plus"]) - float(out["E"])) / span if abs(span) > 1e-15 else None
    base.update(
        {
            "status": "ok",
            "retryable": False,
            "B_star": b,
            "spsa_b_key": meta["spsa_b_key"],
            "J_sha256": meta["J_sha256"],
            "S_A": meta["S_A"],
            "F_plus": meta["F_plus"],
            "E_plus": meta["E_plus"],
            "E0": meta["E0"],
            "gap": meta["gap"],
            "delta_odd": meta["delta_odd"],
            "delta_even": meta["delta_even"],
            "E1_is_odd": meta["E1_is_odd"],
            "E": float(out["E"]),
            "energy_error": float(out["E"]) - float(meta["E0"]),
            "F": F,
            "F_role": "posthoc_on_charged_selected_state",
            "eta": eta,
            "eta_E": eta_e,
            "queries": out["queries"],
            "posthoc_audit_energy": 1,
            "n_iter_per_start": out["n_iter"],
            "selected_start": out["start_id"],
            "selected_point": out["which"],
            "a_gains": out["a_gains"],
            "x": np.asarray(out["x"], dtype=np.float64).tolist(),
            "n_controls": int(n_controls(method, L)),
            "seconds": seconds,
            "chart": "nov3_polynomial_shell_shared_Uc" if method == "Nov3rd" else "qaoa_independent_2p",
            "hardware_claim": False,
        }
    )
    return _clean(base)


def conformance_static() -> dict:
    """No SPSA. Circuit identity, cut, packing, budget predicate, one B* check, seed wall."""
    prereg = load_prereg()
    report = {"recipe": RECIPE, "protocol_sha": prereg["_sha256"], "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise RuntimeError(f"conformance failed: {name} {detail}")

    dev = seed_span_check = list(range(10000, 10010))
    check("dev_len", dev == list(range(prereg["seeds"]["dev"]["start"], prereg["seeds"]["dev"]["stop"] + 1)), str(len(dev)))
    final = list(range(prereg["seeds"]["final"]["start"], prereg["seeds"]["final"]["stop"] + 1))
    check("final_len_1000", len(final) == 1000)
    blocked = []
    for seed, stage in ((10010, "dev"), (10018, "dev"), (10026, "dev"), (11000, "dev"), (11031, "final"), (19999, "final"), (21000, "final")):
        try:
            assert_allowed_seed(seed, stage)
        except RuntimeError:
            blocked.append(seed)
    check("foreign_seeds_blocked", len(blocked) == 7, str(blocked))

    from common_campaign import seed_span as _span

    check("span", _span(prereg["seeds"]["dev"]) == dev)

    for n in (6, 9, 12, 15, 18):
        groups = packing(n)
        sites = [s for g in groups for s in g]
        m = n // 3
        expect = [[a, a + m, a + 2 * m] for a in range(m)]
        check(f"packing_{n}", groups == expect and sorted(sites) == list(range(n)))

    n = 5
    psi = np.zeros(1 << n, dtype=np.float64)
    psi[0] = psi[1 | (1 << (n // 2))] = 2.0 ** -0.5
    s = entropy(psi, n)
    check("odd_n_cut_entropy", abs(s - np.log(2.0)) < 1e-12, str(s))

    J = make_J(10000, 6)
    hc = pair_diag(J, z_table(6))
    groups, dk, dm, hc = features(J, 6, hc)
    rng = np.random.default_rng(1)
    x = x0_of("Nov3rd", 2, "R", rng)
    psi_a = nov3_psi(x, 2, hc, dk, dm, groups, 6)
    gamma, thk, thm = float(x[0]), float(x[1]), float(x[2])
    shells = np.asarray(x[3:], dtype=np.float64).reshape(2, 3)
    gdiag = hc + thk * dk + thm * dm
    psi_b = plus_state(6)
    for ell in range(2):
        psi_b = psi_b * np.exp(-1j * gamma * gdiag)
        for cav in groups:
            for site in cav:
                psi_b = apply_H_site(psi_b, int(site))
            psi_b = psi_b * np.exp(-1j * pop_phase_logical(6, cav, float(shells[ell, 0]), float(shells[ell, 1]), float(shells[ell, 2])))
            for site in cav:
                psi_b = apply_H_site(psi_b, int(site))
    check("nov3_spec_overlap", abs(abs(np.vdot(psi_a, psi_b)) ** 2 - 1.0) < 1e-12)
    check("nov3_one_gamma_slot", n_controls("Nov3rd", 6) == 3 + 18 and n_controls("QAOA", 6) == 12)

    vars_ = []
    for seed in range(10000, 10020):
        u = upper(make_J(seed, 12))
        vars_.append(float(np.var(u, ddof=1)))
    rel = abs(float(np.mean(vars_)) - (1.0 / 12.0)) / (1.0 / 12.0)
    check("J_variance_1_over_N", rel < 0.2, f"rel {rel:.4f}")

    slots = 0
    for L in prereg["L"]:
        qs = runnable_qs_local(L, prereg["budget"]["q_candidates"])
        slots += len(qs)
        if L == 1:
            check("floor_L1", qs == [1024, 512, 256, 128, 64], str(qs))
        if L == 6:
            check("floor_L6", 16 in qs and 32 in qs and 8 not in prereg["budget"]["q_candidates"])
    check("q_slots", slots == 38, str(slots))
    check("q32_L1_floor", allocation(32, 1) is None and allocation(64, 1)["T"] == 2)
    check("q16_L4", allocation(16, 4)["T"] == 2 and allocation(16, 3) is None)

    calls = {"n": 0}

    def prop(x):
        calls["n"] += 1
        return float(np.asarray(x)[0] ** 2), np.zeros(2)

    for q, L in ((64, 1), (1024, 1), (16, 6)):
        alloc = allocation(q, L)
        calls["n"] = 0
        rngs = [np.random.default_rng(i) for i in range(4)]
        out = run_matched(prop, [np.zeros(2) for _ in range(4)], np.full(2, 0.1), alloc["Q"], rngs)
        check(f"budget_q{q}_L{L}", calls["n"] == alloc["Q"] and out["queries"]["monitoring"] == 0 and out["queries"]["init"] == 4, str(out["queries"]))

    found = search_bstar(make_J(10000, 6), 6)
    ref = _reference_bstar()[(6, 10000)]
    diff = abs(found["B_star"] - float(ref["B_star"]))
    check("bstar_matches_r2_n6_s10000", found["ok"] and diff <= 1e-5, f"diff {diff}")
    report["passed"] = all(row["ok"] for row in report["checks"])
    return report


def runnable_qs_local(L: int, candidates: list[int]) -> list[int]:
    return [int(q) for q in candidates if allocation(int(q), int(L)) is not None]
