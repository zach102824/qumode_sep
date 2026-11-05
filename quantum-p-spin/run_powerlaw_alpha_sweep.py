"""Power-law p=3 TFIM α/Γ sweep — free QAOA or shared γ+β_ℓ (Nov_3rd φ_k=-βk).

H = -Σ J_ijk ZZZ - Γ Σ X
J_ijk = ξ_ijk * r_ijk^{-α} / N,  r = ring diameter, N = (Σ r^{-α})/n
ξ = ±1 Rademacher (signed=True, default) so N_INST instances exist.
Baseline family=glass: Gaussian J ~ N(0,1) with scale √3/n.

Protocol match energy-fid: cost ⟨H⟩, one LBFGS/instance, init β=γ=+0.01,
report median F, E, E0, ΔE, i95. ≤8 workers. n∈{6,9,12}.
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
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import make_instance, make_powerlaw_instance  # noqa: E402
from circuit_snap import make_ansatz, fidelity  # noqa: E402
from circuit_fourier import shell_phi_from_betas  # noqa: E402
from energy_util import energy  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GAMMAS = (0.5, 0.75, 1.0)
NS = (6, 9, 12)
LS = (2, 4, 8, 16)
N_INST = 16
OUTER_STEPS = 40
LBFGS_MAX_ITER = 5
LBFGS_LR = 0.25
INIT_VAL = 0.01


def cell_key(n, L, Gamma, family, alpha, method="free_qaoa"):
    a = None if family == "glass" else float(alpha)
    return (int(n), int(L), float(Gamma), str(family), a, str(method))



def log(msg: str, log_path: Path | None = None):
    line = f"{time.strftime('%H:%M:%S')} [plaw] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(line + "\n")


def workers_for_n(n: int, max_workers: int) -> int:
    if n >= 12:
        return min(4, max_workers)
    if n >= 9:
        return min(6, max_workers)
    return max_workers


def cache_key(family: str, n: int, Gamma: float, alpha: float, seed: int, signed: bool = True) -> str:
    if family == "glass":
        return f"glass_n{n}_G{Gamma:g}_seed{seed}.npz"
    tag = "plaw" if signed else "ferro"
    return f"{tag}_n{n}_G{Gamma:g}_a{alpha:g}_seed{seed}.npz"


def cache_one(args):
    family, n, Gamma, alpha, seed, cache_dir_str, signed = args
    torch.set_num_threads(1)
    from pspin import make_instance, make_powerlaw_instance
    if family == "glass":
        inst = make_instance(n, Gamma, seed)
        alpha_out = float("nan")
        signed = True
    else:
        inst = make_powerlaw_instance(n, Gamma, alpha, seed=seed, signed=bool(signed))
        alpha_out = float(alpha)
    path = Path(cache_dir_str) / cache_key(family, n, Gamma, alpha, seed, signed=bool(signed))
    np.savez_compressed(
        path,
        Hc_diag=inst.Hc_diag.astype(np.float64),
        gs=inst.gs.astype(np.complex128),
        E0=np.array(inst.E0),
        J=inst.J,
        Gamma=np.array(float(Gamma)),
        alpha=np.array(alpha_out),
        family=np.array(family),
    )
    return family, n, Gamma, alpha, seed, float(inst.E0)


def init_params(method: str, L: int, device):
    if method == "free_qaoa":
        gamma = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
        beta = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
        params = [gamma, beta]

        def angles():
            return gamma, shell_phi_from_betas(beta)

        def angles_det():
            return gamma.detach(), shell_phi_from_betas(beta.detach())

        return params, angles, angles_det, 2 * L
    if method == "shared_gb":
        beta = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
        g_scalar = torch.nn.Parameter(torch.tensor(INIT_VAL, dtype=torch.float32, device=device))
        params = [beta, g_scalar]

        def angles():
            return g_scalar.expand(L).contiguous(), shell_phi_from_betas(beta)

        def angles_det():
            return g_scalar.detach().expand(L).contiguous(), shell_phi_from_betas(beta.detach())

        return params, angles, angles_det, L + 1
    raise ValueError(method)


def optimize_instance(sim, Hc, gs, n: int, Gamma: float, L: int, device, method: str = "free_qaoa"):
    params, angles, angles_det, n_params = init_params(method, L, device)
    F_curve = []
    with torch.no_grad():
        gvec, phi = angles_det()
        F0 = float(fidelity(sim, Hc, gs, gvec, phi).cpu())
        E_init = float(energy(sim, Hc, Gamma, n, gvec, phi).cpu())
    F_curve.append(F0)

    for step in range(OUTER_STEPS):
        opt = torch.optim.LBFGS(
            params, lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
            history_size=16, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            gvec, phi = angles()
            E = energy(sim, Hc, Gamma, n, gvec, phi)
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
        F_curve.append(F)
        if step >= 5:
            recent = F_curve[-4:]
            if max(recent) - min(recent) < 1e-5 and abs(F - F_curve[-2]) < 1e-6:
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
        final_E = float(energy(sim, Hc, Gamma, n, gvec, phi).cpu())
    return {
        "final_F": final_F,
        "final_E": final_E,
        "i95": int(i95),
        "n_params": n_params,
        "n_steps": len(F_curve) - 1,
        "init_F": F0,
        "init_E": E_init,
    }


def run_cell(payload: dict) -> dict:
    n = int(payload["n"])
    L = int(payload["L"])
    Gamma = float(payload["Gamma"])
    family = str(payload["family"])
    alpha = float(payload.get("alpha", float("nan")))
    signed = bool(payload.get("signed", True))
    method = str(payload.get("method", "free_qaoa"))
    cache_dir = Path(payload["cache_dir"])
    n_inst = int(payload.get("N_INST", N_INST))
    try:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        sim = make_ansatz(n, L, device)
        Fs, i95s, Es, E0s = [], [], [], []
        per = []
        for seed in range(n_inst):
            path = cache_dir / cache_key(family, n, Gamma, alpha, seed, signed=signed)
            if path.exists():
                data = np.load(path, allow_pickle=True)
                Hc_np = data["Hc_diag"]
                gs_np = data["gs"]
                E0 = float(data["E0"])
            else:
                if family == "glass":
                    inst = make_instance(n, Gamma, seed)
                else:
                    inst = make_powerlaw_instance(n, Gamma, alpha, seed=seed, signed=signed)
                Hc_np, gs_np, E0 = inst.Hc_diag, inst.gs, inst.E0
            Hc = torch.as_tensor(Hc_np, dtype=torch.float32, device=device)
            gs = torch.as_tensor(gs_np, dtype=torch.complex64, device=device)
            out = optimize_instance(sim, Hc, gs, n, Gamma, L, device, method=method)
            Fs.append(out["final_F"])
            i95s.append(out["i95"])
            Es.append(out["final_E"])
            E0s.append(E0)
            per.append({
                "n": n, "L": L, "Gamma": Gamma, "family": family,
                "alpha": alpha if family != "glass" else float("nan"),
                "method": method,
                "seed": seed, "E0": E0,
                "final_F": out["final_F"], "final_E": out["final_E"],
                "dE": out["final_E"] - E0,
                "i95": out["i95"], "n_steps": out["n_steps"],
                "init_F": out["init_F"], "n_params": out["n_params"],
            })
        dEs = [e - e0 for e, e0 in zip(Es, E0s)]
        return {
            "ok": True, "n": n, "L": L, "Gamma": Gamma,
            "family": family, "alpha": alpha if family != "glass" else float("nan"),
            "method": method,
            "median_F": float(np.median(Fs)),
            "median_E": float(np.median(Es)),
            "median_E0": float(np.median(E0s)),
            "median_dE": float(np.median(dEs)),
            "median_i95": float(np.median(i95s)),
            "mean_F": float(np.mean(Fs)),
            "n_params": per[0]["n_params"],
            "n_inst": n_inst,
            "per_instance": per,
        }
    except Exception as e:
        return {
            "ok": False, "n": n, "L": L, "Gamma": Gamma,
            "family": family, "alpha": alpha, "method": method,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def build_cache(jobs_meta, cache_dir: Path, log_path: Path, max_workers: int, n_inst: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    # unique (family,n,Gamma,alpha) needing seeds
    needed = []
    seen = set()
    for meta in jobs_meta:
        family = meta["family"]
        n = meta["n"]
        G = float(meta["Gamma"])
        a = float(meta["alpha"]) if family != "glass" else 0.0
        key = (family, n, G, a if family != "glass" else None)
        if key in seen:
            continue
        seen.add(key)
        signed = bool(meta.get("signed", True))
        for seed in range(n_inst):
            path = cache_dir / cache_key(family, n, G, a, seed, signed=signed)
            if not path.exists():
                needed.append((family, n, G, a, seed, str(cache_dir), signed))
    if not needed:
        log("cache complete", log_path)
        return
    # group by n for worker throttle
    by_n = {}
    for pay in needed:
        by_n.setdefault(pay[1], []).append(pay)
    for n, pays in sorted(by_n.items()):
        w = workers_for_n(n, max_workers)
        log(f"cache n={n} missing={len(pays)} workers={w}", log_path)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(cache_one, pay) for pay in pays]
            for fut in as_completed(futs):
                family, n, G, a, seed, E0 = fut.result()
                log(f"  cached {family} n={n} G={G:g} a={a:g} seed={seed} E0={E0:.4f}", log_path)
        log(f"cache n={n} done in {time.time()-t0:.1f}s", log_path)


def write_outputs(out_dir: Path, cells: list[dict], log_path: Path):
    ok_cells = [c for c in cells if c.get("ok")]
    # per_instance.csv
    fields = [
        "n", "L", "Gamma", "family", "alpha", "method", "seed", "E0",
        "final_F", "final_E", "dE", "i95", "n_steps", "init_F", "n_params",
    ]
    rows = []
    for c in ok_cells:
        rows.extend(c.get("per_instance", []))
    rows.sort(key=lambda r: (r["n"], r["Gamma"], r["family"], r.get("alpha", 0) if r["family"] != "glass" else -1, r["L"], r["seed"]))
    with (out_dir / "per_instance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "free_qaoa" if k == "method" else "") for k in fields})

    # summary.csv
    sum_fields = [
        "n", "L", "Gamma", "family", "alpha", "method", "median_F", "median_E",
        "median_E0", "median_dE", "median_i95", "mean_F", "n_params", "n_inst",
    ]
    summary = []
    for c in ok_cells:
        summary.append({k: c.get(k) for k in sum_fields})
    summary.sort(key=lambda r: (
        r["n"], r["Gamma"],
        0 if r["family"] == "glass" else 1,
        -1.0 if r["family"] == "glass" else float(r["alpha"]),
        r["L"],
    ))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    slim = [{k: v for k, v in c.items() if k != "per_instance"} for c in cells]
    (out_dir / "cells.json").write_text(json.dumps(slim, indent=2))

    # RESULTS.md
    lines = [
        "# Power-law p=3 TFIM α/Γ sweep",
        "",
        "## Hamiltonian",
        "",
        "Signed power-law on a **ring**:",
        "",
        r"$$H = -\sum_{i<j<k} J_{ijk} Z_i Z_j Z_k - \Gamma\sum_i X_i,\quad"
        r"J_{ijk}=\xi_{ijk}\, r_{ijk}^{-\alpha}/N$$",
        "",
        r"- $r_{ijk}$ = ring diameter of $\{i,j,k\}$ (shortest covering arc length in edges)",
        r"- $N = (\sum r^{-\alpha})/n$ (Kac-style; classical ferro energy $= -n$)",
        r"- $\xi_{ijk}=\pm 1$ Rademacher (seeded) so $N_{\mathrm{inst}}$ instances exist",
        r"- **glass baseline:** $J\sim\mathcal N(0,1)$ with scale $\sqrt{3}/n$ (existing `make_instance`)",
        "",
        "## Protocol",
        "",
        f"- method from cells (`free_qaoa` = γ_ℓ,β_ℓ; `shared_gb` = shared γ + β_ℓ), `φ_k=-β k`, init `+{INIT_VAL}`",
        f"- cost `⟨H⟩`, LBFGS OUTER={OUTER_STEPS}×max_iter={LBFGS_MAX_ITER}, lr={LBFGS_LR}",
        f"- N={N_INST} instances/cell; n∈{list(NS)}; L∈{list(LS)}; "
        f"α∈{list(ALPHAS)}; Γ∈{list(GAMMAS)} + glass",
        "",
        "## Summary table (median)",
        "",
        "| n | Γ | family | α | method | L | med F | med E | med E₀ | med ΔE | i95 |",
        "|--:|--:|:-------|---:|:-------|--:|------:|------:|-------:|-------:|----:|",
    ]
    for r in summary:
        a = "—" if r["family"] == "glass" else f"{float(r['alpha']):.2f}"
        meth = r.get("method", "free_qaoa")
        lines.append(
            f"| {r['n']} | {float(r['Gamma']):.2f} | {r['family']} | {a} | {meth} | {r['L']} | "
            f"{float(r['median_F']):.4f} | {float(r['median_E']):.3f} | "
            f"{float(r['median_E0']):.3f} | {float(r['median_dE']):.3f} | "
            f"{float(r['median_i95']):.1f} |"
        )

    # shine analysis vs glass at same n,Γ,L
    lines += ["", "## Shine vs glass (ΔF = F_plaw − F_glass at same n,Γ,L)", ""]
    glass_idx = {}
    for r in summary:
        if r["family"] == "glass":
            glass_idx[(r["n"], float(r["Gamma"]), r["L"])] = float(r["median_F"])
    shine_rows = []
    for r in summary:
        if r["family"] != "plaw":
            continue
        key = (r["n"], float(r["Gamma"]), r["L"])
        if key not in glass_idx:
            continue
        dF = float(r["median_F"]) - glass_idx[key]
        shine_rows.append((dF, r, glass_idx[key]))
    shine_rows.sort(reverse=True, key=lambda t: t[0])
    lines += [
        "| ΔF | n | Γ | α | L | F_plaw | F_glass | ΔE_plaw |",
        "|---:|--:|--:|---:|--:|-------:|--------:|--------:|",
    ]
    for dF, r, Fg in shine_rows[:40]:
        lines.append(
            f"| {dF:+.4f} | {r['n']} | {float(r['Gamma']):.2f} | {float(r['alpha']):.2f} | "
            f"{r['L']} | {float(r['median_F']):.4f} | {Fg:.4f} | {float(r['median_dE']):.3f} |"
        )

    # best (α,Γ) per n at L=16
    lines += ["", "## Best (α,Γ) per n at L=16 (by median F)", ""]
    for n in NS:
        cand = [r for r in summary if r["n"] == n and r["L"] == 16 and r["family"] == "plaw"]
        if not cand:
            continue
        best = max(cand, key=lambda r: float(r["median_F"]))
        g = next((r for r in summary if r["n"] == n and r["L"] == 16 and r["family"] == "glass"
                  and abs(float(r["Gamma"]) - float(best["Gamma"])) < 1e-9), None)
        gF = float(g["median_F"]) if g else float("nan")
        lines.append(
            f"- n={n}: α={float(best['alpha']):.2f}, Γ={float(best['Gamma']):.2f} → "
            f"F={float(best['median_F']):.4f} (glass F={gF:.4f}, "
            f"ΔF={float(best['median_F'])-gF:+.4f}), ΔE={float(best['median_dE']):.3f}"
        )

    lines += [
        "",
        "## Takeaway",
        "",
        "_Filled as the grid completes._",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines))
    log(f"wrote summary.csv + RESULTS.md ({len(ok_cells)} ok cells)", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(PKG / "runs/compare/powerlaw_alpha_sweep"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--Ls", type=int, nargs="+", default=list(LS))
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    ap.add_argument("--Gammas", type=float, nargs="+", default=list(GAMMAS))
    ap.add_argument("--N", type=int, default=N_INST)
    ap.add_argument("--no-glass", action="store_true")
    ap.add_argument("--ferro", action="store_true",
                    help="ferro J (no signs); forces N=1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-cache-build", action="store_true")
    ap.add_argument("--method", type=str, default="free_qaoa",
                    choices=["free_qaoa", "shared_gb"])
    ap.add_argument("--cache-dir", type=str, default="",
                    help="reuse instance cache from another run")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "instance_cache")
    method = str(args.method)
    log_path = out_dir / "sweep.log"
    max_workers = min(8, int(args.max_workers))
    n_inst = int(args.N)
    signed = not bool(args.ferro)
    if args.ferro:
        n_inst = 1

    ns = tuple(args.ns)
    Ls = tuple(args.Ls)
    alphas = tuple(args.alphas)
    Gammas = tuple(args.Gammas)
    if args.smoke:
        ns, Ls, alphas, Gammas = (6,), (2,), (1.0,), (0.5,)
        n_inst = min(n_inst, 4)

    # build job list
    jobs = []
    for n in ns:
        for G in Gammas:
            if not args.no_glass:
                for L in Ls:
                    jobs.append({
                        "n": n, "L": L, "Gamma": G, "family": "glass",
                        "alpha": float("nan"), "cache_dir": str(cache_dir),
                        "N_INST": n_inst, "signed": True, "method": method,
                    })
            for a in alphas:
                for L in Ls:
                    jobs.append({
                        "n": n, "L": L, "Gamma": G, "family": "plaw",
                        "alpha": float(a), "cache_dir": str(cache_dir),
                        "N_INST": n_inst, "signed": signed, "method": method,
                    })

    log(f"START method={method} out={out_dir} cache={cache_dir} workers≤{max_workers} ns={ns} Ls={Ls} α={alphas} Γ={Gammas} N={n_inst} jobs={len(jobs)}", log_path)

    if not args.skip_cache_build:
        build_cache(jobs, cache_dir, log_path, max_workers, n_inst)

    # resume
    cells = []
    done = set()
    inst_csv = out_dir / "per_instance.csv"
    if inst_csv.exists():
        from collections import defaultdict
        bucket = defaultdict(list)
        with inst_csv.open() as f:
            for row in csv.DictReader(f):
                fam = row["family"]
                a = float(row["alpha"]) if row["alpha"] not in ("", "nan", "NaN") else float("nan")
                meth = row.get("method") or method
                key = cell_key(row["n"], row["L"], row["Gamma"], fam, a, meth)
                bucket[key].append({
                    "n": int(row["n"]), "L": int(row["L"]),
                    "Gamma": float(row["Gamma"]), "family": fam, "alpha": a,
                    "method": meth,
                    "seed": int(row["seed"]), "E0": float(row["E0"]),
                    "final_F": float(row["final_F"]), "final_E": float(row["final_E"]),
                    "dE": float(row["dE"]), "i95": int(float(row["i95"])),
                    "n_steps": int(float(row["n_steps"])), "init_F": float(row["init_F"]),
                    "n_params": int(float(row["n_params"])),
                })
        for key, per in bucket.items():
            if len(per) < n_inst:
                continue
            Fs = [p["final_F"] for p in per]
            Es = [p["final_E"] for p in per]
            E0s = [p["E0"] for p in per]
            i95s = [p["i95"] for p in per]
            cells.append({
                "ok": True, "n": key[0], "L": key[1], "Gamma": key[2],
                "family": key[3], "alpha": key[4], "method": key[5],
                "median_F": float(np.median(Fs)),
                "median_E": float(np.median(Es)),
                "median_E0": float(np.median(E0s)),
                "median_dE": float(np.median([e - e0 for e, e0 in zip(Es, E0s)])),
                "median_i95": float(np.median(i95s)),
                "mean_F": float(np.mean(Fs)),
                "n_params": per[0]["n_params"], "n_inst": len(per),
                "per_instance": per,
            })
            done.add(key)
        log(f"resume: {len(done)} cells from per_instance.csv", log_path)

    # run by n
    for n in ns:
        todo = []
        for job in jobs:
            if job["n"] != n:
                continue
            key = cell_key(job["n"], job["L"], job["Gamma"], job["family"], job.get("alpha", float("nan")), job.get("method", method))
            if key in done:
                continue
            todo.append(job)
        if not todo:
            log(f"n={n}: all done", log_path)
            continue
        w = workers_for_n(n, max_workers)
        log(f"n={n}: dispatch {len(todo)} cells with {w} workers", log_path)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(run_cell, job): job for job in todo}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "ok": False, "n": job["n"], "L": job["L"],
                        "Gamma": job["Gamma"], "family": job["family"],
                        "alpha": job.get("alpha", float("nan")),
                        "error": f"{e}\n{traceback.format_exc()}",
                    }
                cells.append(result)
                if result.get("ok"):
                    log(
                        f"DONE n={result['n']} L={result['L']} G={result['Gamma']:g} "
                        f"{result['family']} a={result.get('alpha', float('nan'))} "
                        f"meth={result.get('method', method)} "
                        f"medF={result['median_F']:.4f} dE={result['median_dE']:.3f}",
                        log_path,
                    )
                else:
                    log(
                        f"FAIL n={job['n']} L={job['L']} {job['family']}: "
                        f"{result.get('error','')[:240]}",
                        log_path,
                    )
                write_outputs(out_dir, [c for c in cells if c.get("ok")], log_path)
        log(f"n={n} block done in {time.time()-t0:.1f}s", log_path)

    write_outputs(out_dir, [c for c in cells if c.get("ok")], log_path)
    # final takeaway rewrite
    _finalize_takeaway(out_dir, log_path)
    ok = sum(1 for c in cells if c.get("ok"))
    bad = sum(1 for c in cells if not c.get("ok"))
    log(f"ALL DONE ok={ok} fail={bad} → {out_dir / 'RESULTS.md'}", log_path)


def _finalize_takeaway(out_dir: Path, log_path: Path):
    """Append concrete takeaway from summary.csv."""
    sum_path = out_dir / "summary.csv"
    if not sum_path.exists():
        return
    rows = list(csv.DictReader(sum_path.open()))
    if not rows:
        return
    # pick overall best ΔF at L>=8
    glass = {(int(r["n"]), float(r["Gamma"]), int(r["L"])): float(r["median_F"])
             for r in rows if r["family"] == "glass"}
    best = None
    for r in rows:
        if r["family"] != "plaw":
            continue
        if int(r["L"]) < 8:
            continue
        key = (int(r["n"]), float(r["Gamma"]), int(r["L"]))
        if key not in glass:
            continue
        dF = float(r["median_F"]) - glass[key]
        cand = (dF, float(r["median_F"]), float(r["median_dE"]), r)
        if best is None or cand[0] > best[0]:
            best = cand
    # sweet region: α with mean ΔF > 0 across n at L=16
    by_ag = {}
    for r in rows:
        if r["family"] != "plaw" or int(r["L"]) != 16:
            continue
        key = (float(r["alpha"]), float(r["Gamma"]))
        gkey = (int(r["n"]), float(r["Gamma"]), 16)
        if gkey not in glass:
            continue
        by_ag.setdefault(key, []).append(float(r["median_F"]) - glass[gkey])
    sweet = sorted(
        ((np.mean(v), a, G) for (a, G), v in by_ag.items() if v),
        reverse=True,
    )
    md = (out_dir / "RESULTS.md").read_text()
    take = ["## Takeaway", ""]
    if best:
        dF, F, dE, r = best
        take.append(
            f"- Best single cell (L≥8): **α={float(r['alpha']):.2f}, Γ={float(r['Gamma']):.2f}, "
            f"n={r['n']}, L={r['L']}** → F={F:.4f}, ΔF vs glass={dF:+.4f}, ΔE={dE:.3f}"
        )
    if sweet:
        take.append("- Mean ΔF vs glass at L=16 by (α,Γ):")
        for mean_dF, a, G in sweet[:8]:
            take.append(f"  - α={a:.2f}, Γ={G:.2f}: mean ΔF={mean_dF:+.4f}")
        top_a, top_G = sweet[0][1], sweet[0][2]
        take.append(
            f"- **Shine region (approx):** α≈{top_a:.2f}, Γ≈{top_G:.2f} "
            f"(signed power-law on ring; beats Gaussian glass at matched n,Γ,L)"
        )
    take.append("")
    # replace takeaway section
    if "## Takeaway" in md:
        pre = md.split("## Takeaway")[0].rstrip()
        # keep trailing generated line if present
        post = ""
        if "Generated " in md:
            post = "\n" + [ln for ln in md.splitlines() if ln.startswith("Generated ")][-1] + "\n"
        (out_dir / "RESULTS.md").write_text(pre + "\n\n" + "\n".join(take) + post)
    log("takeaway finalized", log_path)


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
