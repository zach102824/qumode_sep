#!/usr/bin/env python3
"""Nov_3rd ladder: L small→large; compare vs fair paper-n6 QAOA; STOP if UNDER.

Eval order (Zach t2495u):
  1) ALL L on n∈{6,9,12,15}  (CORE) — gate / STOP
  2) then n=18 across surviving L
  3) then n=21 across surviving L (skip if no scale1000 pack)

Train: n=6×1000 L-BFGS only; warm-fallback recovers exact QAOA dual.
QAOA refs: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.data_io import (
    load_qaoa,
    load_qaoa_scale_ref,
    load_split_E,
    load_train_E,
    log,
    shell_phi_from_betas,
)
from Nov_3rd.ensure_qaoa import ensure_qaoa_paper_n6
from Nov_3rd.eval_scale import eval_ladder
from Nov_3rd.train_bfgs import train_seed
from explore_OCT29.lib.qaoa import qaoa_psucc


TIE_HI = 1.02
BEAT_HI = 0.98


def _set_threads(n: int = S.NUM_THREADS):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    log(f"threads={n} (OMP/MKL/torch)")


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


@torch.no_grad()
def qaoa_med_b90_on_scale(L, n, betas, gammas, max_count, device):
    pack = S.SCALE1000_DIR / f"data_n{n}.npz"
    if not pack.exists():
        return None, None
    E, _ = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    invs = []
    chunk = 32 if n >= 15 else 64
    for i0 in range(0, len(E), chunk):
        Eb = E[i0 : i0 + chunk]
        ps = qaoa_psucc(betas, gammas, Eb, n)
        invs.extend((1.0 / ps.clamp(min=1e-12)).cpu().numpy().tolist())
    arr = np.asarray(invs, dtype=np.float64)
    return float(np.median(arr)), float(np.quantile(arr, 0.90))


def compare_vs_qaoa(L, rows, qref, betas, gammas, device) -> dict:
    per = {}
    under = False
    for n, m in sorted(rows.items()):
        qmed = qref[n]["Med"] if n in qref and qref[n].get("Med") is not None else None
        qb90 = qref[n].get("B90") if n in qref else None
        if qmed is None:
            log(f"L{L} n{n}: QAOA Med missing — computing ideal QAOA…")
            qmed, qb90 = qaoa_med_b90_on_scale(L, n, betas, gammas, S.SCALE_B, device)
        if qmed is None or not np.isfinite(qmed) or qmed <= 0:
            per[n] = {
                "Med": m["median_inv_p"], "B90": m["B90_inv_p"],
                "R": None, "verdict": "NO_QAOA",
            }
            continue
        R = float(m["median_inv_p"] / qmed)
        if R <= BEAT_HI:
            v = "BEAT"
        elif R <= TIE_HI:
            v = "TIE"
        else:
            v = "UNDER"
            under = True
        per[n] = {
            "Med": m["median_inv_p"], "B90": m["B90_inv_p"],
            "QAOA_Med": qmed, "QAOA_B90": qb90, "R": R, "verdict": v,
        }
        log(f"L{L} n{n}: Med={m['median_inv_p']:.3f} Q={qmed:.3f} R={R:.3f}× → {v}")
    overall = "UNDER" if under else (
        "BEAT" if any(p.get("verdict") == "BEAT" for p in per.values()) else "TIE"
    )
    return {"per_n": per, "overall": overall, "under": under}


def write_L_results(L, rows, warm_rows, best_seed, meta, cmp, warm_cmp, phase="core"):
    lines = [
        f"# Nov_3rd — RESULTS L={L} ({phase})",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"Phase: **{phase}**  ns={sorted(rows.keys())}",
        f"Overall vs QAOA (accepted): **{cmp['overall']}**",
        f"Warm-only vs QAOA: **{warm_cmp['overall']}**",
        f"Best seed = {best_seed}",
        (
            "Warm recovery: **yes** (polish rejected; exact QAOA dual shipped)"
            if meta.get("best_used_warm")
            else "Warm recovery: no (polish train Med ≤ warm)"
        ),
        "",
        "QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.",
        "",
        "| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | "
        "acc Med | acc B90 | R_acc | acc |",
        "|---|----------|----------|----------|----------|--------|------|"
        "---------|---------|-------|-----|",
    ]
    ns = sorted(set(list(cmp["per_n"]) + list(warm_cmp["per_n"])))

    def f3(x):
        return "" if x is None else f"{float(x):.3f}"

    def fr(x):
        return "" if x is None else f"{float(x):.3f}×"

    for n in ns:
        w = warm_cmp["per_n"].get(n, {})
        a = cmp["per_n"].get(n, {})
        qm = w.get("QAOA_Med", a.get("QAOA_Med"))
        qb = w.get("QAOA_B90", a.get("QAOA_B90"))
        lines.append(
            f"| {n} | {f3(qm)} | {f3(qb)} | {f3(w.get('Med'))} | {f3(w.get('B90'))} | "
            f"{fr(w.get('R'))} | {w.get('verdict', '')} | "
            f"{f3(a.get('Med'))} | {f3(a.get('B90'))} | {fr(a.get('R'))} | "
            f"{a.get('verdict', '')} |"
        )
    lines.append("")
    if cmp["under"]:
        lines += [
            "## STOP",
            "At least one n UNDER QAOA (R>1.02) on accepted ckpt — do not climb L.",
            "",
        ]
    suffix = "" if phase == "core" else f"_{phase}"
    (S.OUT / f"RESULTS_L{L}{suffix}.md").write_text("\n".join(lines) + "\n")
    (S.OUT / f"results_L{L}{suffix}.json").write_text(
        json.dumps(
            _jsonable({
                "meta": meta, "eval": rows, "warm_eval": warm_rows,
                "compare": cmp, "warm_compare": warm_cmp,
                "best_seed": best_seed, "phase": phase,
            }),
            indent=2,
        )
    )


def write_summary(all_L, phase="core"):
    ns = sorted({n for b in all_L.values() for n in b["eval"]})
    lines = [
        f"# Nov_3rd — SUMMARY ({phase})",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"L set={list(S.L_LIST)}; phase={phase}; ns={ns}; gate R≤{TIE_HI}",
        "QAOA: paper-n6 fair same-L. Accepted uses warm-fallback.",
        "",
        "## Accepted Med",
        "| L | " + " | ".join(f"n{n}" for n in ns) + " | vs QAOA |",
        "|---|" + "|".join(["-----"] * len(ns)) + "|---------|",
    ]
    for L in S.L_LIST:
        if L not in all_L:
            continue
        cells = []
        for n in ns:
            m = all_L[L]["eval"].get(n)
            cells.append(f"{m['median_inv_p']:.3f}" if m else "—")
        lines.append(
            f"| {L} | " + " | ".join(cells) + f" | **{all_L[L]['compare']['overall']}** |"
        )
    lines.append("")
    name = "RESULTS.md" if phase == "core" else f"RESULTS_{phase}.md"
    (S.OUT / name).write_text("\n".join(lines) + "\n")
    jname = "results_all.json" if phase == "core" else f"results_all_{phase}.json"
    (S.OUT / jname).write_text(json.dumps(_jsonable(all_L), indent=2))


def _git_push_L(L, cmp, warm_cmp, phase="core"):
    try:
        suffix = "" if phase == "core" else f"_{phase}"
        files = [
            str(S.OUT / f"RESULTS_L{L}{suffix}.md"),
            str(S.OUT / f"results_L{L}{suffix}.json"),
            str(S.OUT / f"train_summary_L{L}.json"),
            str(S.OUT / ("RESULTS.md" if phase == "core" else f"RESULTS_{phase}.md")),
            str(S.OUT / ("results_all.json" if phase == "core" else f"results_all_{phase}.json")),
            str(S.qaoa_angles_path(L)),
            str(S.qaoa_results_path(L)),
            str(S.OUT / f"params_best_L{L}.pt"),
        ]
        for seed in S.SEEDS:
            files.append(str(S.OUT / f"params_seed{seed}_L{L}.pt"))
        existing = [f for f in files if Path(f).exists()]
        subprocess.run(["git", "add", "--"] + existing, cwd=str(_ROOT), check=False)
        subprocess.run(
            [
                "git", "commit", "-m",
                f"Nov_3rd RESULTS L={L} {phase} {cmp['overall']} "
                f"(warm {warm_cmp['overall']}; paper-n6)",
            ],
            cwd=str(_ROOT),
            check=False,
        )
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(_ROOT), check=False)
        log(f"L={L} {phase}: pushed")
    except Exception as e:
        log(f"L={L} {phase}: git push skipped ({e})")


def _available(ns):
    out = []
    for n in ns:
        if (S.SCALE1000_DIR / f"data_n{n}.npz").exists():
            out.append(int(n))
        else:
            log(f"skip n{n}: no scale1000 pack")
    return out


def run_core_ladder(device, E_train, E_val):
    """Phase 1: train + eval CORE ns for every L; STOP on UNDER."""
    ns = _available(S.EVAL_NS_CORE)
    log(f"PHASE1 CORE start L={list(S.L_LIST)} ns={ns}")
    all_L = {}
    surviving = []
    for L in S.L_LIST:
        t0 = time.time()
        log(f"=== L={L} ensure fair QAOA (ns={ns}) ===")
        ensure_qaoa_paper_n6(L, device=device, ns=ns)
        betas0, gammas0 = load_qaoa(L)
        warm_phi = shell_phi_from_betas(betas0)

        log(f"=== L={L} warm-only CORE eval ===")
        warm_rows = eval_ladder(L, gammas0, warm_phi, device, ns=ns)
        qref = load_qaoa_scale_ref(L)
        warm_cmp = compare_vs_qaoa(L, warm_rows, qref, betas0, gammas0, device)
        log(f"L={L} warm-only overall {warm_cmp['overall']}")

        log(f"=== L={L} train (LBFGS + warm-fallback) ===")
        best_all = None
        seed_rows = []
        for seed in S.SEEDS:
            best = train_seed(
                L=L, device=device, E_train=E_train,
                betas0=betas0, gammas0=gammas0, seed=seed, E_val=E_val,
            )
            used_warm = bool(best.get("used_warm") or best.get("rejected_polish"))
            if used_warm:
                log(
                    f"L{L} seed{seed}: warm recovered; "
                    f"trainMed=warmMed≈{best['warm_med']:.4f}"
                )
            seed_rows.append({
                "seed": seed, "score": best["score"], "round": best["round"],
                "train_med": best["train_med"], "warm_med": best["warm_med"],
                "used_warm": used_warm,
                "rejected_polish": bool(best.get("rejected_polish", False)),
            })
            torch.save(best["state"], S.OUT / f"params_seed{seed}_L{L}.pt")
            if best_all is None or best["score"] < best_all["score"]:
                best_all = {**best, "seed": seed}
        torch.save(best_all["state"], S.OUT / f"params_best_L{L}.pt")
        (S.OUT / f"train_summary_L{L}.json").write_text(
            json.dumps(
                _jsonable({
                    "seeds": seed_rows,
                    "best_seed": best_all["seed"],
                    "L": L,
                    "n_seeds_used_warm": sum(1 for r in seed_rows if r.get("used_warm")),
                    "best_used_warm": bool(
                        best_all.get("used_warm") or best_all.get("rejected_polish")
                    ),
                    "qaoa_source": "paper_n6",
                    "eval_ns_core": list(ns),
                }),
                indent=2,
            )
        )

        used_warm_best = bool(best_all.get("used_warm") or best_all.get("rejected_polish"))
        if used_warm_best:
            log(f"=== L={L} accepted == warm (skip re-eval) ===")
            rows, cmp = warm_rows, warm_cmp
        else:
            log(f"=== L={L} accepted CORE eval ===")
            st = best_all["state"]
            rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=ns)
            cmp = compare_vs_qaoa(L, rows, qref, betas0, gammas0, device)

        meta = {
            "recipe": S.RECIPE_NAME, "L": L, "phase": "core",
            "eval_ns": list(ns), "b": S.B_BITS,
            "optimizer": S.OPTIMIZER, "train_ns": list(S.TRAIN_NS),
            "max_train": S.MAX_TRAIN, "scale_B": S.SCALE_B,
            "best_seed": best_all["seed"], "qaoa_ref": qref,
            "qaoa_source": "paper_n6",
            "train_summary": seed_rows,
            "best_used_warm": used_warm_best,
            "best_warm_med": best_all.get("warm_med"),
            "best_train_med": best_all.get("train_med"),
            "wall_s": time.time() - t0,
            "state": best_all["state"],
            "betas": betas0.detach().cpu(),
            "gammas": gammas0.detach().cpu(),
        }
        write_L_results(L, rows, warm_rows, best_all["seed"], meta, cmp, warm_cmp, "core")
        all_L[L] = {
            "meta": meta, "eval": rows, "warm_eval": warm_rows,
            "compare": cmp, "warm_compare": warm_cmp,
            "best_seed": best_all["seed"],
        }
        write_summary(all_L, "core")
        log(
            f"L={L} CORE overall {cmp['overall']} (warm {warm_cmp['overall']}) "
            f"wall={meta['wall_s']/60:.1f}min"
        )
        _git_push_L(L, cmp, warm_cmp, "core")
        if cmp["under"]:
            log(f"STOP CORE ladder at L={L}: UNDER — no further L / no n18")
            write_summary(all_L, "core")
            return all_L, surviving  # surviving = Ls before UNDER
        surviving.append(L)
    log("PHASE1 CORE DONE — all L TIE/BEAT on n∈{6,9,12,15}")
    return all_L, surviving


def run_extend_pass(all_L, surviving, ns_pass, phase, device):
    """Phase 2/3: eval extra n across surviving L (reuse trained ckpts)."""
    ns = _available(ns_pass)
    if not ns:
        log(f"PHASE {phase}: no packs for {list(ns_pass)} — skip")
        return all_L
    log(f"PHASE {phase} start surviving L={surviving} ns={ns}")
    ext = {}
    for L in surviving:
        t0 = time.time()
        ensure_qaoa_paper_n6(L, device=device, ns=ns)
        betas0, gammas0 = load_qaoa(L)
        warm_phi = shell_phi_from_betas(betas0)
        st = torch.load(S.OUT / f"params_best_L{L}.pt", map_location=device, weights_only=False)
        log(f"=== L={L} {phase} warm eval ===")
        warm_rows = eval_ladder(L, gammas0, warm_phi, device, ns=ns)
        qref = load_qaoa_scale_ref(L)
        warm_cmp = compare_vs_qaoa(L, warm_rows, qref, betas0, gammas0, device)
        used_warm = bool(st.get("used_warm") or st.get("rejected_polish"))
        if used_warm:
            rows, cmp = warm_rows, warm_cmp
            log(f"L={L} {phase}: accepted==warm")
        else:
            log(f"=== L={L} {phase} accepted eval ===")
            rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=ns)
            cmp = compare_vs_qaoa(L, rows, qref, betas0, gammas0, device)
        # merge into core rows for summary convenience
        merged_eval = dict(all_L[L]["eval"])
        merged_eval.update(rows)
        merged_warm = dict(all_L[L]["warm_eval"])
        merged_warm.update(warm_rows)
        meta = dict(all_L[L]["meta"])
        meta.update({"phase": phase, "eval_ns_extra": list(ns), "wall_s_extra": time.time() - t0})
        write_L_results(L, rows, warm_rows, all_L[L]["best_seed"], meta, cmp, warm_cmp, phase)
        ext[L] = {
            "meta": meta, "eval": merged_eval, "warm_eval": merged_warm,
            "compare": cmp, "warm_compare": warm_cmp,
            "best_seed": all_L[L]["best_seed"],
            "extra_compare": cmp,
        }
        # also refresh core blob with merged eval
        all_L[L]["eval"] = merged_eval
        all_L[L]["warm_eval"] = merged_warm
        log(f"L={L} {phase} overall {cmp['overall']} wall={(time.time()-t0)/60:.1f}min")
        _git_push_L(L, cmp, warm_cmp, phase)
        if cmp["under"]:
            log(f"NOTE: L={L} {phase} UNDER on ns={ns} (core already passed)")
    write_summary(ext, phase)
    write_summary(all_L, "core")  # refresh with merged
    log(f"PHASE {phase} DONE")
    return all_L


def main():
    _set_threads(S.NUM_THREADS)
    S.assert_locked(
        b=S.B_BITS, W_frame=S.W_FRAME, snap_head=S.SNAP_HEAD,
        optimizer=S.OPTIMIZER, train_ns=S.TRAIN_NS, max_train=S.MAX_TRAIN,
        qaoa_warm=S.QAOA_WARM, snap_shared_layers=S.SNAP_SHARED_LAYERS,
        L_list=S.L_LIST, eval_ns=S.EVAL_NS, eval_ns_core=S.EVAL_NS_CORE,
    )
    tj = S.OUT / "tests.json"
    if not tj.exists():
        log("REFUSE: run python -m Nov_3rd.tests.run_tests first")
        sys.exit(2)
    blob = json.loads(tj.read_text())
    if blob.get("failed"):
        log("REFUSE: tests failed —", blob["failed"])
        sys.exit(2)
    log(f"tests OK ({len(blob.get('passed', []))} passed)")
    log(
        f"eval plan: CORE={list(S.EVAL_NS_CORE)} → PASS2={list(S.EVAL_NS_PASS2)} "
        f"→ PASS3={list(S.EVAL_NS_PASS3)}; train={S.TRAIN_NS}×{S.MAX_TRAIN}"
    )

    S.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    E_train = load_train_E(device)
    E_val, _ = load_split_E(6, "val", S.MAX_VAL, device, data_dir=S.SEALED_DATA)

    all_L, surviving = run_core_ladder(device, E_train, E_val)
    if not surviving:
        log("No surviving L — stop before n18/n21")
        sys.exit(4)

    all_L = run_extend_pass(all_L, surviving, S.EVAL_NS_PASS2, "n18", device)
    all_L = run_extend_pass(all_L, surviving, S.EVAL_NS_PASS3, "n21", device)
    log("DONE all phases")


if __name__ == "__main__":
    main()
