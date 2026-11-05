"""2-body power-law TFIM shared_gb QAOA fidelity ladder.

H = -Σ_{i<j} J_ij Z_i Z_j - Γ Σ_i X_i
J_ij = ξ_ij r_ij^{-α}/N, r=ring dist, N=(Σ r^{-α})/n, ξ=±1.

Protocol matches 3-body powerlaw shared_gb sweep:
  shared_gb (1 γ + β_ℓ), φ_k=-βk, cost ⟨H⟩,
  LBFGS OUTER=40×max_iter=5, init +0.01, N=16/cell.
Also reports median half-cut GS entanglement S(n,α).
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import make_tfim2_plaw_instance, half_cut_entropy  # noqa: E402
from circuit_snap import make_ansatz, fidelity  # noqa: E402
from circuit_fourier import shell_phi_from_betas  # noqa: E402
from energy_util import energy  # noqa: E402

ALPHAS = (0.25, 1.0, 1.5, 2.0)
NS = (6, 9, 12)
LS = (2, 4, 8, 16)
GAMMA = 0.5
N_INST = 16
OUTER_STEPS = 40
LBFGS_MAX_ITER = 5
LBFGS_LR = 0.25
INIT_VAL = 0.01
METHOD = "shared_gb"


def cell_key(n, L, alpha):
    return (int(n), int(L), float(alpha))


def log(msg: str, log_path: Path | None = None):
    line = f"{time.strftime('%H:%M:%S')} [tfim2] {msg}"
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


def cache_key(n: int, Gamma: float, alpha: float, seed: int) -> str:
    return f"tfim2_1d_n{n}_G{Gamma:g}_a{alpha:g}_seed{seed}.npz"


def cache_one(args):
    n, Gamma, alpha, seed, cache_dir_str = args
    torch.set_num_threads(1)
    from pspin import make_tfim2_plaw_instance, half_cut_entropy
    inst = make_tfim2_plaw_instance(n, Gamma, alpha, seed=seed, signed=True)
    S = half_cut_entropy(inst.gs, n)
    path = Path(cache_dir_str) / cache_key(n, Gamma, alpha, seed)
    np.savez_compressed(
        path,
        Hc_diag=inst.Hc_diag.astype(np.float64),
        gs=inst.gs.astype(np.complex128),
        E0=np.array(inst.E0),
        J=inst.J,
        Gamma=np.array(float(Gamma)),
        alpha=np.array(float(alpha)),
        S_half=np.array(float(S)),
    )
    return n, Gamma, alpha, seed, float(inst.E0), float(S)


def init_params(L: int, device):
    beta = torch.nn.Parameter(torch.full((L,), INIT_VAL, dtype=torch.float32, device=device))
    g_scalar = torch.nn.Parameter(torch.tensor(INIT_VAL, dtype=torch.float32, device=device))
    params = [beta, g_scalar]

    def angles():
        return g_scalar.expand(L).contiguous(), shell_phi_from_betas(beta)

    def angles_det():
        return g_scalar.detach().expand(L).contiguous(), shell_phi_from_betas(beta.detach())

    return params, angles, angles_det, L + 1


def optimize_instance(sim, Hc, gs, n: int, Gamma: float, L: int, device):
    params, angles, angles_det, n_params = init_params(L, device)
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
    alpha = float(payload["alpha"])
    cache_dir = Path(payload["cache_dir"])
    n_inst = int(payload.get("N_INST", N_INST))
    try:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        sim = make_ansatz(n, L, device)
        Fs, i95s, Es, E0s, Ss = [], [], [], [], []
        per = []
        for seed in range(n_inst):
            path = cache_dir / cache_key(n, Gamma, alpha, seed)
            if path.exists():
                data = np.load(path, allow_pickle=True)
                Hc_np = data["Hc_diag"]
                gs_np = data["gs"]
                E0 = float(data["E0"])
                if "S_half" in data.files:
                    S = float(data["S_half"])
                else:
                    S = half_cut_entropy(gs_np, n)
            else:
                inst = make_tfim2_plaw_instance(n, Gamma, alpha, seed=seed, signed=True)
                Hc_np, gs_np, E0 = inst.Hc_diag, inst.gs, inst.E0
                S = half_cut_entropy(inst.gs, n)
            Hc = torch.as_tensor(Hc_np, dtype=torch.float32, device=device)
            gs = torch.as_tensor(gs_np, dtype=torch.complex64, device=device)
            out = optimize_instance(sim, Hc, gs, n, Gamma, L, device)
            Fs.append(out["final_F"])
            i95s.append(out["i95"])
            Es.append(out["final_E"])
            E0s.append(E0)
            Ss.append(S)
            per.append({
                "n": n, "L": L, "Gamma": Gamma, "alpha": alpha,
                "method": METHOD, "seed": seed, "E0": E0, "S_half": S,
                "final_F": out["final_F"], "final_E": out["final_E"],
                "dE": out["final_E"] - E0,
                "i95": out["i95"], "n_steps": out["n_steps"],
                "init_F": out["init_F"], "n_params": out["n_params"],
            })
        dEs = [e - e0 for e, e0 in zip(Es, E0s)]
        return {
            "ok": True, "n": n, "L": L, "Gamma": Gamma, "alpha": alpha,
            "method": METHOD,
            "median_F": float(np.median(Fs)),
            "median_E": float(np.median(Es)),
            "median_E0": float(np.median(E0s)),
            "median_dE": float(np.median(dEs)),
            "median_i95": float(np.median(i95s)),
            "median_S": float(np.median(Ss)),
            "mean_F": float(np.mean(Fs)),
            "n_params": per[0]["n_params"],
            "n_inst": n_inst,
            "per_instance": per,
        }
    except Exception as e:
        return {
            "ok": False, "n": n, "L": L, "Gamma": Gamma, "alpha": alpha,
            "method": METHOD,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def build_cache(jobs_meta, cache_dir: Path, log_path: Path, max_workers: int, n_inst: int, Gamma: float):
    cache_dir.mkdir(parents=True, exist_ok=True)
    needed = []
    seen = set()
    for meta in jobs_meta:
        n = meta["n"]
        a = float(meta["alpha"])
        key = (n, a)
        if key in seen:
            continue
        seen.add(key)
        for seed in range(n_inst):
            path = cache_dir / cache_key(n, Gamma, a, seed)
            if not path.exists():
                needed.append((n, Gamma, a, seed, str(cache_dir)))
    if not needed:
        log("cache complete", log_path)
        return
    by_n = {}
    for pay in needed:
        by_n.setdefault(pay[0], []).append(pay)
    for n, pays in sorted(by_n.items()):
        w = workers_for_n(n, max_workers)
        log(f"cache n={n} missing={len(pays)} workers={w}", log_path)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(cache_one, pay) for pay in pays]
            for fut in as_completed(futs):
                n, G, a, seed, E0, S = fut.result()
                log(f"  cached n={n} G={G:g} a={a:g} seed={seed} E0={E0:.4f} S={S:.4f}", log_path)
        log(f"cache n={n} done in {time.time()-t0:.1f}s", log_path)


def write_outputs(out_dir: Path, cells: list[dict], log_path: Path):
    ok_cells = [c for c in cells if c.get("ok")]
    fields = [
        "n", "L", "Gamma", "alpha", "method", "seed", "E0", "S_half",
        "final_F", "final_E", "dE", "i95", "n_steps", "init_F", "n_params",
    ]
    rows = []
    for c in ok_cells:
        rows.extend(c.get("per_instance", []))
    rows.sort(key=lambda r: (r["n"], float(r["alpha"]), r["L"], r["seed"]))
    with (out_dir / "per_instance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    sum_fields = [
        "n", "L", "Gamma", "alpha", "method", "median_F", "median_E",
        "median_E0", "median_dE", "median_i95", "median_S", "mean_F",
        "n_params", "n_inst",
    ]
    summary = []
    for c in ok_cells:
        summary.append({k: c.get(k) for k in sum_fields})
    summary.sort(key=lambda r: (r["n"], float(r["alpha"]), r["L"]))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    slim = [{k: v for k, v in c.items() if k != "per_instance"} for c in cells]
    (out_dir / "cells.json").write_text(json.dumps(slim, indent=2))

    # median S by (n,α) — independent of L (same GS)
    S_by = {}
    for r in summary:
        key = (int(r["n"]), float(r["alpha"]))
        S_by[key] = float(r["median_S"])

    lines = [
        "# 2-body power-law TFIM shared_gb ladder (open 1D chain)",
        "",
        "## Hamiltonian",
        "",
        r"$$H = -\sum_{i<j} J_{ij} Z_i Z_j - \Gamma\sum_i X_i,\quad"
        r"J_{ij}=\xi_{ij}\, r_{ij}^{-\alpha}/N$$",
        "",
        r"- $r_{ij}=|i-j|$ (open 1D chain; no wrap-around)",
        r"- $N=(\sum_{i<j} r^{-\alpha})/n$ (Kac; classical ferro $E=-n$)",
        r"- $\xi_{ij}=\pm 1$ Rademacher (signed)",
        r"- No glass baseline in this ladder",
        "",
        "## Protocol",
        "",
        f"- `shared_gb`: one shared γ + β_ℓ (L+1 params), `φ_k=-βk` Hamming-shell SNAP",
        f"- cost `⟨H⟩`, LBFGS OUTER={OUTER_STEPS}×max_iter={LBFGS_MAX_ITER}, lr={LBFGS_LR}, init `+{INIT_VAL}`",
        f"- N={N_INST} instances/cell; n∈{list(NS)}; L∈{list(LS)}; α∈{list(ALPHAS)}; Γ={GAMMA}",
        "",
        "## Median fidelity ladders (per α)",
        "",
    ]
    for a in sorted({float(r["alpha"]) for r in summary}):
        lines.append(f"### α={a:g}, Γ={GAMMA}")
        lines.append("")
        lines.append("| n \\ L | " + " | ".join(str(L) for L in LS) + " |")
        lines.append("|--:|" + "|".join(["---:"] * len(LS)) + "|")
        for n in NS:
            cells_n = {int(r["L"]): float(r["median_F"])
                       for r in summary if int(r["n"]) == n and abs(float(r["alpha"]) - a) < 1e-12}
            if not cells_n:
                continue
            row = f"| **{n}** | " + " | ".join(
                f"{cells_n[L]:.4f}" if L in cells_n else "—" for L in LS
            ) + " |"
            lines.append(row)
        lines.append("")

    lines += [
        "## Median half-cut entanglement S of GS",
        "",
        "| n \\ α | " + " | ".join(f"{a:g}" for a in ALPHAS) + " |",
        "|--:|" + "|".join(["---:"] * len(ALPHAS)) + "|",
    ]
    for n in NS:
        vals = []
        for a in ALPHAS:
            key = (n, float(a))
            vals.append(f"{S_by[key]:.4f}" if key in S_by else "—")
        lines.append(f"| **{n}** | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Notes",
        "",
        "_Filled after the grid completes._",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines))
    log(f"wrote summary.csv + RESULTS.md ({len(ok_cells)} ok cells)", log_path)


def finalize_notes(out_dir: Path, log_path: Path):
    sum_path = out_dir / "summary.csv"
    if not sum_path.exists():
        return
    rows = list(csv.DictReader(sum_path.open()))
    if not rows:
        return
    notes = ["## Notes", ""]
    # Does L help?
    for a in ALPHAS:
        sub = [r for r in rows if abs(float(r["alpha"]) - a) < 1e-12]
        if not sub:
            continue
        deltas = []
        for n in NS:
            byL = {int(r["L"]): float(r["median_F"]) for r in sub if int(r["n"]) == n}
            if 2 in byL and 16 in byL:
                deltas.append((n, byL[16] - byL[2], byL[2], byL[16]))
        if deltas:
            notes.append(f"- **α={a:g} L-gain (F@16−F@2):** " + ", ".join(
                f"n{n}: {d:+.4f} ({f2:.3f}→{f16:.3f})" for n, d, f2, f16 in deltas
            ))
    # n6 vs n9/12 at L=2
    notes.append("- **n=6 vs larger at L=2 (median F):**")
    for a in ALPHAS:
        byn = {int(r["n"]): float(r["median_F"])
               for r in rows if abs(float(r["alpha"]) - a) < 1e-12 and int(r["L"]) == 2}
        if byn:
            notes.append(
                f"  - α={a:g}: " + ", ".join(f"n{n}={byn[n]:.4f}" for n in NS if n in byn)
            )
    # S trend
    S_by = {}
    for r in rows:
        S_by[(int(r["n"]), float(r["alpha"]))] = float(r["median_S"])
    notes.append("- **Half-cut S (median):** larger S ⇒ more entangled GS (harder for shallow QAOA).")
    for a in ALPHAS:
        notes.append(
            "  - α={:g}: ".format(a) + ", ".join(
                f"n{n} S={S_by[(n,float(a))]:.3f}" for n in NS if (n, float(a)) in S_by
            )
        )
    notes.append("")
    md = (out_dir / "RESULTS.md").read_text()
    if "## Notes" in md:
        pre = md.split("## Notes")[0].rstrip()
        post = ""
        if "Generated " in md:
            post = "\n" + [ln for ln in md.splitlines() if ln.startswith("Generated ")][-1] + "\n"
        (out_dir / "RESULTS.md").write_text(pre + "\n\n" + "\n".join(notes) + post)
    log("notes finalized", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(PKG / "runs/compare/tfim2_plaw_ladder_1d"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--Ls", type=int, nargs="+", default=list(LS))
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    ap.add_argument("--Gamma", type=float, default=GAMMA)
    ap.add_argument("--N", type=int, default=N_INST)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-cache-build", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "instance_cache"
    log_path = out_dir / "sweep.log"
    max_workers = min(8, int(args.max_workers))
    n_inst = int(args.N)
    Gamma = float(args.Gamma)

    ns = tuple(args.ns)
    Ls = tuple(args.Ls)
    alphas = tuple(args.alphas)
    if args.smoke:
        ns, Ls, alphas = (6,), (2,), (1.0,)
        n_inst = min(n_inst, 4)

    jobs = []
    for n in ns:
        for a in alphas:
            for L in Ls:
                jobs.append({
                    "n": n, "L": L, "Gamma": Gamma, "alpha": float(a),
                    "cache_dir": str(cache_dir), "N_INST": n_inst,
                })

    log(
        f"START out={out_dir} cache={cache_dir} workers≤{max_workers} "
        f"ns={ns} Ls={Ls} α={alphas} Γ={Gamma} N={n_inst} jobs={len(jobs)}",
        log_path,
    )

    if not args.skip_cache_build:
        build_cache(jobs, cache_dir, log_path, max_workers, n_inst, Gamma)

    cells = []
    done = set()
    inst_csv = out_dir / "per_instance.csv"
    if inst_csv.exists():
        bucket = defaultdict(list)
        with inst_csv.open() as f:
            for row in csv.DictReader(f):
                key = cell_key(row["n"], row["L"], row["alpha"])
                bucket[key].append({
                    "n": int(row["n"]), "L": int(row["L"]),
                    "Gamma": float(row["Gamma"]), "alpha": float(row["alpha"]),
                    "method": row.get("method") or METHOD,
                    "seed": int(row["seed"]), "E0": float(row["E0"]),
                    "S_half": float(row.get("S_half") or 0.0),
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
            Ss = [p["S_half"] for p in per]
            cells.append({
                "ok": True, "n": key[0], "L": key[1], "Gamma": Gamma,
                "alpha": key[2], "method": METHOD,
                "median_F": float(np.median(Fs)),
                "median_E": float(np.median(Es)),
                "median_E0": float(np.median(E0s)),
                "median_dE": float(np.median([e - e0 for e, e0 in zip(Es, E0s)])),
                "median_i95": float(np.median(i95s)),
                "median_S": float(np.median(Ss)),
                "mean_F": float(np.mean(Fs)),
                "n_params": per[0]["n_params"], "n_inst": len(per),
                "per_instance": per,
            })
            done.add(key)
        log(f"resume: {len(done)} cells from per_instance.csv", log_path)

    for n in ns:
        todo = []
        for job in jobs:
            if job["n"] != n:
                continue
            key = cell_key(job["n"], job["L"], job["alpha"])
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
                        "Gamma": job["Gamma"], "alpha": job["alpha"],
                        "error": f"{e}\n{traceback.format_exc()}",
                    }
                cells.append(result)
                if result.get("ok"):
                    log(
                        f"DONE n={result['n']} L={result['L']} a={result['alpha']:g} "
                        f"medF={result['median_F']:.4f} dE={result['median_dE']:.3f} "
                        f"S={result['median_S']:.4f}",
                        log_path,
                    )
                else:
                    log(
                        f"FAIL n={job['n']} L={job['L']} a={job['alpha']}: "
                        f"{result.get('error','')[:240]}",
                        log_path,
                    )
                write_outputs(out_dir, [c for c in cells if c.get("ok")], log_path)
        log(f"n={n} block done in {time.time()-t0:.1f}s", log_path)

    write_outputs(out_dir, [c for c in cells if c.get("ok")], log_path)
    finalize_notes(out_dir, log_path)
    ok = sum(1 for c in cells if c.get("ok"))
    bad = sum(1 for c in cells if not c.get("ok"))
    log(f"ALL DONE ok={ok} fail={bad} → {out_dir / 'RESULTS.md'}", log_path)


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
