"""Energy-cost LBFGS grid: free QAOA vs shared γ+β_ℓ.

Protocol (locked):
  H = Hc + Hx, Γ=0.5, Nov_3rd QAOA-line φ_k=-β k
  cost = ⟨H⟩ per instance; one LBFGS; init β=γ=+0.01
  N=32 instances (seeds 0..31); report median F + median i95
  Parallel: ProcessPool ≤8 workers (throttled on large n)
"""
from __future__ import annotations

import os as _os
for _k, _v in {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}.items():
    _os.environ.setdefault(_k, _v)

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import make_instance, hc_torch, gs_torch  # noqa: E402
from circuit_snap import make_ansatz, fidelity  # noqa: E402
from circuit_fourier import shell_phi_from_betas  # noqa: E402
from energy_util import energy  # noqa: E402

GAMMA_X = 0.5
N_INST = 32
NS = (6, 9, 12, 15, 18)
LS = (2, 4, 6, 8, 10, 12, 14, 16)
METHODS = ("free_qaoa", "shared_gb")

# LBFGS caps (documented in RESULTS.md)
OUTER_STEPS = 40          # outer opt.step calls
LBFGS_MAX_ITER = 5        # max_iter per outer step  → ≤200 LBFGS iters
LBFGS_LR = 0.25
INIT_VAL = 0.01           # β = γ = +0.01 exactly (no jitter)


def log(msg: str, log_path: Path | None = None):
    line = f"{time.strftime('%H:%M:%S')} [efid] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(line + "\n")


def workers_for_n(n: int, max_workers: int) -> int:
    """Throttle workers on large Hilbert spaces to avoid OOM."""
    if n >= 18:
        return min(2, max_workers)
    if n >= 15:
        return min(3, max_workers)
    if n >= 12:
        return min(4, max_workers)
    return max_workers



def cache_one_instance(args):
    """Top-level for ProcessPool: (n, seed, Gamma, cache_dir_str) -> (seed, E0)."""
    n, seed, Gamma, cache_dir_str = args
    torch.set_num_threads(1)
    from pspin import make_instance
    inst = make_instance(n, Gamma, seed)
    path = Path(cache_dir_str) / f"n{n}_seed{seed}.npz"
    np.savez_compressed(
        path,
        Hc_diag=inst.Hc_diag.astype(np.float64),
        gs=inst.gs.astype(np.complex128),
        E0=np.array(inst.E0),
        J=inst.J,
    )
    return seed, float(inst.E0)



def init_params(method: str, L: int, device):
    """β=γ=+0.01 exactly."""
    if method == "free_qaoa":
        gamma = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
        beta = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
        params = [gamma, beta]

        def angles():
            return gamma, shell_phi_from_betas(beta)

        def angles_det():
            return gamma.detach(), shell_phi_from_betas(beta.detach())

        n_params = 2 * L
    elif method == "shared_gb":
        beta = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
        g_scalar = torch.nn.Parameter(torch.tensor(INIT_VAL, dtype=torch.float32, device=device))
        params = [beta, g_scalar]

        def angles():
            return g_scalar.expand(L).contiguous(), shell_phi_from_betas(beta)

        def angles_det():
            return g_scalar.detach().expand(L).contiguous(), shell_phi_from_betas(beta.detach())

        n_params = L + 1
    else:
        raise ValueError(method)
    return params, angles, angles_det, n_params


def optimize_instance(sim, Hc, gs, n: int, method: str, L: int, device):
    """One LBFGS run minimizing ⟨H⟩. Returns final F, i95, E, F_curve."""
    params, angles, angles_det, n_params = init_params(method, L, device)
    F_curve = []

    with torch.no_grad():
        gvec, phi = angles_det()
        F0 = float(fidelity(sim, Hc, gs, gvec, phi).cpu())
        E0 = float(energy(sim, Hc, GAMMA_X, n, gvec, phi).cpu())
    F_curve.append(F0)

    for step in range(OUTER_STEPS):
        opt = torch.optim.LBFGS(
            params,
            lr=LBFGS_LR,
            max_iter=LBFGS_MAX_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            gvec, phi = angles()
            E = energy(sim, Hc, GAMMA_X, n, gvec, phi)
            E.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return E

        try:
            opt.step(closure)
        except Exception:
            break

        with torch.no_grad():
            gvec, phi = angles_det()
            F = float(fidelity(sim, Hc, gs, gvec, phi).cpu())
            E = float(energy(sim, Hc, GAMMA_X, n, gvec, phi).cpu())
        F_curve.append(F)

        # early stop if energy stalled
        if step >= 5:
            recent = F_curve[-4:]
            if max(recent) - min(recent) < 1e-5 and abs(F - F_curve[-2]) < 1e-6:
                # pad remaining? no — i95 uses this curve's final
                break

    final_F = F_curve[-1]
    target = 0.95 * final_F
    i95 = len(F_curve) - 1
    for i, f in enumerate(F_curve):
        if f >= target - 1e-12:
            i95 = i
            break
    with torch.no_grad():
        gvec, phi = angles_det()
        final_E = float(energy(sim, Hc, GAMMA_X, n, gvec, phi).cpu())
    return {
        "final_F": final_F,
        "final_E": final_E,
        "i95": int(i95),
        "n_params": n_params,
        "n_steps": len(F_curve) - 1,
        "F_curve": F_curve,
        "init_F": F_curve[0],
        "init_E": E0,
    }


def run_cell(payload: dict) -> dict:
    """Worker: run all N_INST instances for one (n, L, method)."""
    n = int(payload["n"])
    L = int(payload["L"])
    method = str(payload["method"])
    cache_dir = Path(payload["cache_dir"])
    n_inst = int(payload.get("N_INST", N_INST))
    try:
        torch.set_num_threads(1)  # avoid oversubscription inside workers
        device = torch.device("cpu")
        sim = make_ansatz(n, L, device)
        Fs, i95s, Es = [], [], []
        per = []
        for seed in range(n_inst):
            cache_path = cache_dir / f"n{n}_seed{seed}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                Hc_np = data["Hc_diag"]
                gs_np = data["gs"]
                E0 = float(data["E0"])
            else:
                inst = make_instance(n, GAMMA_X, seed)
                Hc_np = inst.Hc_diag
                gs_np = inst.gs
                E0 = inst.E0
            Hc = torch.as_tensor(Hc_np, dtype=torch.float32, device=device)
            gs = torch.as_tensor(gs_np, dtype=torch.complex64, device=device)
            out = optimize_instance(sim, Hc, gs, n, method, L, device)
            Fs.append(out["final_F"])
            i95s.append(out["i95"])
            Es.append(out["final_E"])
            per.append(
                {
                    "n": n,
                    "L": L,
                    "method": method,
                    "seed": seed,
                    "E0": E0,
                    "final_F": out["final_F"],
                    "final_E": out["final_E"],
                    "i95": out["i95"],
                    "n_steps": out["n_steps"],
                    "init_F": out["init_F"],
                    "n_params": out["n_params"],
                }
            )
        return {
            "ok": True,
            "n": n,
            "L": L,
            "method": method,
            "median_F": float(np.median(Fs)),
            "median_i95": float(np.median(i95s)),
            "mean_F": float(np.mean(Fs)),
            "mean_i95": float(np.mean(i95s)),
            "median_E": float(np.median(Es)),
            "n_params": per[0]["n_params"],
            "per_instance": per,
        }
    except Exception as e:
        return {
            "ok": False,
            "n": n,
            "L": L,
            "method": method,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def build_instance_cache(ns, cache_dir: Path, log_path: Path, max_workers: int, n_inst: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for n in ns:
        missing = [s for s in range(n_inst) if not (cache_dir / f"n{n}_seed{s}.npz").exists()]
        if not missing:
            log(f"cache n={n} complete ({n_inst})", log_path)
            continue
        w = workers_for_n(n, max_workers)
        log(f"building cache n={n} missing={len(missing)} workers={w}", log_path)
        t0 = time.time()
        payloads = [(n, s, GAMMA_X, str(cache_dir)) for s in missing]
        if w == 1:
            for pay in payloads:
                seed, E0 = cache_one_instance(pay)
                log(f"  cached n={n} seed={seed} E0={E0:.4f}", log_path)
        else:
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = {ex.submit(cache_one_instance, pay): pay[1] for pay in payloads}
                for fut in as_completed(futs):
                    seed, E0 = fut.result()
                    log(f"  cached n={n} seed={seed} E0={E0:.4f}", log_path)
        log(f"cache n={n} done in {time.time()-t0:.1f}s", log_path)



def write_summary(out_dir: Path, cells: list[dict], log_path: Path):
    """Write per_instance.csv then regenerate summary/RESULTS with energy columns."""
    # per-instance CSV
    inst_path = out_dir / "per_instance.csv"
    fields = [
        "n", "L", "method", "seed", "E0", "final_F", "final_E",
        "i95", "n_steps", "init_F", "n_params",
    ]
    rows = []
    for c in cells:
        if c.get("ok") and "per_instance" in c:
            rows.extend(c["per_instance"])
    rows.sort(key=lambda r: (r["n"], r["L"], r["method"], r["seed"]))
    with inst_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # cells json (strip heavy)
    slim = []
    for c in cells:
        d = {k: v for k, v in c.items() if k != "per_instance"}
        if "per_instance" in c:
            d["n_inst"] = len(c["per_instance"])
            # also stash median energies for cells.json
            if c.get("ok"):
                Es = [p["final_E"] for p in c["per_instance"]]
                E0s = [p["E0"] for p in c["per_instance"]]
                d["median_E"] = float(np.median(Es))
                d["median_E0"] = float(np.median(E0s))
                d["median_dE"] = float(np.median([e - e0 for e, e0 in zip(Es, E0s)]))
        slim.append(d)
    (out_dir / "cells.json").write_text(json.dumps(slim, indent=2))

    # regenerate pivoted summary + RESULTS.md with energy cols
    try:
        from regen_energy_fid_results import load_per_instance, write_outputs
        bucket = load_per_instance(inst_path)
        write_outputs(out_dir, bucket)
        log(f"wrote {out_dir / 'summary.csv'} + {out_dir / 'RESULTS.md'} (with E/E0/ΔE)", log_path)
    except Exception as e:
        log(f"regen summary failed: {e}", log_path)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(PKG / "runs/compare/energy_fid_grid"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--Ls", type=int, nargs="+", default=list(LS))
    ap.add_argument("--methods", type=str, nargs="+", default=list(METHODS))
    ap.add_argument("--smoke", action="store_true", help="n=6 L=2 only, N still 32 unless --N")
    ap.add_argument("--N", type=int, default=N_INST, help="override instance count (tests/smoke)")
    ap.add_argument("--skip-cache-build", action="store_true")
    args = ap.parse_args()

    n_inst = int(args.N)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "instance_cache"
    log_path = out_dir / "grid.log"
    max_workers = min(8, int(args.max_workers))

    ns = tuple(args.ns)
    Ls = tuple(args.Ls)
    methods = tuple(args.methods)
    if args.smoke:
        ns, Ls = (6,), (2,)

    log(f"START out={out_dir} workers≤{max_workers} ns={ns} Ls={Ls} methods={methods} N={n_inst}", log_path)
    log(
        f"LBFGS OUTER_STEPS={OUTER_STEPS} max_iter={LBFGS_MAX_ITER} init=+{INIT_VAL} Γ={GAMMA_X}",
        log_path,
    )

    if not args.skip_cache_build:
        build_instance_cache(ns, cache_dir, log_path, max_workers, n_inst)

    # resume: load existing cells.json
    cells = []
    cells_path = out_dir / "cells.json"
    done = set()
    if cells_path.exists():
        try:
            prev = json.loads(cells_path.read_text())
            for c in prev:
                if c.get("ok"):
                    done.add((c["n"], c["L"], c["method"]))
                    # re-load per_instance from csv if needed — for resume we re-run missing only
                    cells.append(c)
            log(f"resume: {len(done)} cells already done", log_path)
        except Exception as e:
            log(f"resume skip: {e}", log_path)
            cells = []
            done = set()

    # For resume with per_instance, prefer re-reading per_instance.csv into cells
    inst_csv = out_dir / "per_instance.csv"
    if inst_csv.exists() and done:
        # rebuild cells with per_instance from csv
        from collections import defaultdict
        bucket = defaultdict(list)
        with inst_csv.open() as f:
            for row in csv.DictReader(f):
                key = (int(row["n"]), int(row["L"]), row["method"])
                bucket[key].append(
                    {
                        "n": int(row["n"]),
                        "L": int(row["L"]),
                        "method": row["method"],
                        "seed": int(row["seed"]),
                        "E0": float(row["E0"]),
                        "final_F": float(row["final_F"]),
                        "final_E": float(row["final_E"]),
                        "i95": int(float(row["i95"])),
                        "n_steps": int(float(row["n_steps"])),
                        "init_F": float(row["init_F"]),
                        "n_params": int(float(row["n_params"])),
                    }
                )
        cells = []
        for key, per in bucket.items():
            Fs = [p["final_F"] for p in per]
            i95s = [p["i95"] for p in per]
            Es = [p["final_E"] for p in per]
            cells.append(
                {
                    "ok": True,
                    "n": key[0],
                    "L": key[1],
                    "method": key[2],
                    "median_F": float(np.median(Fs)),
                    "median_i95": float(np.median(i95s)),
                    "mean_F": float(np.mean(Fs)),
                    "mean_i95": float(np.mean(i95s)),
                    "median_E": float(np.median(Es)),
                    "n_params": per[0]["n_params"],
                    "per_instance": per,
                }
            )
        done = {(c["n"], c["L"], c["method"]) for c in cells}
        log(f"resume rebuilt {len(done)} cells from per_instance.csv", log_path)

    # run per-n to control memory
    for n in ns:
        jobs = []
        for L in Ls:
            for method in methods:
                if (n, L, method) in done:
                    continue
                jobs.append(
                    {
                        "n": n,
                        "L": L,
                        "method": method,
                        "cache_dir": str(cache_dir),
                        "N_INST": n_inst,
                    }
                )
        if not jobs:
            log(f"n={n}: all cells done", log_path)
            continue
        w = workers_for_n(n, max_workers)
        log(f"n={n}: dispatch {len(jobs)} cells with {w} workers", log_path)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(run_cell, job): job for job in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "ok": False,
                        "n": job["n"],
                        "L": job["L"],
                        "method": job["method"],
                        "error": f"{e}\n{traceback.format_exc()}",
                    }
                cells.append(result)
                if result.get("ok"):
                    log(
                        f"DONE n={result['n']} L={result['L']} {result['method']} "
                        f"medF={result['median_F']:.4f} medi95={result['median_i95']:.1f}",
                        log_path,
                    )
                else:
                    log(
                        f"FAIL n={result['n']} L={result['L']} {result['method']}: "
                        f"{result.get('error','')[:200]}",
                        log_path,
                    )
                write_summary(out_dir, [c for c in cells if c.get("ok") or not c.get("ok")], log_path)
        log(f"n={n} block done in {time.time()-t0:.1f}s", log_path)

    write_summary(out_dir, cells, log_path)
    ok = sum(1 for c in cells if c.get("ok"))
    bad = sum(1 for c in cells if not c.get("ok"))
    log(f"ALL DONE ok={ok} fail={bad} → {out_dir / 'RESULTS.md'}", log_path)


if __name__ == "__main__":
    # spawn is safer with torch; set if not already
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
