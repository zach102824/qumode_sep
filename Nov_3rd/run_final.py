#!/usr/bin/env python3
"""Nov_3rd ladder: L small→large; Hamming-shell SNAP vs fair paper-n6 QAOA.

Eval order (Zach t2495u + t2527u):
  1) ALL L on n∈{6,9,12,15}  (CORE)
  2) then n=18 across surviving L
  3) then n=21 across surviving L (skip if no scale1000 pack)

Selection (Zach t2527u):
  - train Med warm-fallback stays (polish worse on n6 → ship warm)
  - after CORE eval of polished ckpt: if any n R>1.02 vs QAOA,
    discard polish and ship warm (TIE dual); KEEP climbing L
    (do NOT STOP the ladder on polish transfer UNDER)

Train: n=6×1000 L-BFGS only. QAOA refs: paper β=0.01 γ=-0.01 nstart=1.
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
            f"Warm recovery: **yes** ({meta.get('warm_ship_reason', 'warm shipped')}; exact QAOA dual)"
            if meta.get("best_used_warm")
            else "Warm recovery: no (polish kept; CORE R≤1.02)"
        ),
        "",
        "Motif: Nov1st-final X-sandwich + **Hamming-shell SNAP** (QAOA-warm).",
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
    if meta.get("transfer_under_fallback"):
        lines += [
            "## Transfer-UNDER → ship warm (keep climbing)",
            "Polished ckpt had R>1.02 on at least one CORE n; discarded polish,",
            "shipped warm Hamming-shell SNAP dual (expect TIE). Ladder continues.",
            "",
        ]
    elif cmp["under"]:
        lines += [
            "## NOTE",
            "Accepted (warm) still UNDER vs QAOA on some n — logged; ladder still climbs (t2527u).",
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


def _warm_state(L, betas0, gammas0, seed=0):
    """Exact QAOA dual ckpt (Hamming-shell SNAP warm)."""
    from Nov_3rd.train_bfgs import _ckpt_state
    warm_phi = shell_phi_from_betas(betas0)
    st = _ckpt_state(gamma=gammas0, phi=warm_phi, L=L, seed=seed)
    st["used_warm"] = True
    st["rejected_polish"] = True
    return st, warm_phi


def run_core_ladder(device, E_train, E_val, start_L=None, skip_train_L=None):
    """Phase 1: train + eval CORE ns for every L.

    Zach t2527u: polish transfer UNDER → ship warm, KEEP climbing (no STOP).
    """
    ns = _available(S.EVAL_NS_CORE)
    L_run = [L for L in S.L_LIST if start_L is None or L >= int(start_L)]
    skip_train_L = set(skip_train_L or [])
    log(f"PHASE1 CORE start L={L_run} ns={ns} skip_train={sorted(skip_train_L)}")
    all_L = {}
    surviving = []
    for L in L_run:
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

        transfer_under_fallback = False
        warm_ship_reason = None
        polish_cmp_saved = None

        if L in skip_train_L and (S.OUT / f"results_L{L}.json").exists():
            log(f"=== L={L} REUSE prior train; apply t2527u ship-warm if polish UNDER ===")
            prior = json.loads((S.OUT / f"results_L{L}.json").read_text())
            seed_rows = prior.get("meta", {}).get("train_summary", [])
            best_seed = prior.get("best_seed", 0)
            best_all = {
                "seed": best_seed,
                "score": prior.get("meta", {}).get("best_train_med"),
                "train_med": prior.get("meta", {}).get("best_train_med"),
                "warm_med": prior.get("meta", {}).get("best_warm_med"),
                "used_warm": bool(prior.get("meta", {}).get("best_used_warm")),
                "rejected_polish": bool(prior.get("meta", {}).get("best_used_warm")),
                "state": None,
            }
            # Prefer live warm_rows/warm_cmp just computed; prior polish compare if present
            polish_cmp_saved = prior.get("compare")
            used_warm_best = bool(best_all.get("used_warm") or best_all.get("rejected_polish"))
            if used_warm_best:
                rows, cmp = warm_rows, warm_cmp
                warm_ship_reason = "train_med_fallback_reused"
            else:
                # Re-check prior polish vs QAOA; if UNDER → ship warm
                prior_under = bool((polish_cmp_saved or {}).get("under"))
                if not prior_under and polish_cmp_saved and polish_cmp_saved.get("per_n"):
                    # recompute under from R if needed
                    prior_under = any(
                        (p.get("verdict") == "UNDER") or (p.get("R") is not None and p["R"] > TIE_HI)
                        for p in polish_cmp_saved["per_n"].values()
                    )
                if prior_under or (polish_cmp_saved or {}).get("overall") == "UNDER":
                    log(
                        f"L={L}: prior polish UNDER on CORE — discard polish, "
                        f"ship warm Hamming-shell dual, KEEP climbing (t2527u)"
                    )
                    st_warm, _ = _warm_state(L, betas0, gammas0, seed=best_seed)
                    st_warm["reject_reason"] = "transfer_under_fallback"
                    best_all["state"] = st_warm
                    best_all["used_warm"] = True
                    best_all["rejected_polish"] = True
                    torch.save(st_warm, S.OUT / f"params_best_L{L}.pt")
                    rows, cmp = warm_rows, warm_cmp
                    used_warm_best = True
                    transfer_under_fallback = True
                    warm_ship_reason = "transfer_under_fallback"
                else:
                    # polish was OK — keep prior accepted eval if present
                    rows = prior.get("eval") or warm_rows
                    cmp = polish_cmp_saved or warm_cmp
                    ck = S.OUT / f"params_best_L{L}.pt"
                    if ck.exists():
                        best_all["state"] = torch.load(ck, map_location=device, weights_only=False)
        else:
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
                        f"L{L} seed{seed}: warm recovered (train Med); "
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

            used_warm_best = bool(best_all.get("used_warm") or best_all.get("rejected_polish"))
            if used_warm_best:
                log(f"=== L={L} accepted == warm (train-Med fallback; skip re-eval) ===")
                rows, cmp = warm_rows, warm_cmp
                warm_ship_reason = "train_med_fallback"
            else:
                log(f"=== L={L} polish CORE eval ===")
                st = best_all["state"]
                rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=ns)
                cmp = compare_vs_qaoa(L, rows, qref, betas0, gammas0, device)
                polish_cmp_saved = cmp
                if cmp["under"]:
                    log(
                        f"L={L}: polish UNDER on CORE (R>1.02) — discard polish, "
                        f"ship warm Hamming-shell dual, KEEP climbing (t2527u)"
                    )
                    st_warm, _ = _warm_state(L, betas0, gammas0, seed=best_all["seed"])
                    st_warm["reject_reason"] = "transfer_under_fallback"
                    st_warm["polish_overall"] = cmp["overall"]
                    best_all["state"] = st_warm
                    best_all["used_warm"] = True
                    best_all["rejected_polish"] = True
                    torch.save(st_warm, S.OUT / f"params_best_L{L}.pt")
                    rows, cmp = warm_rows, warm_cmp
                    used_warm_best = True
                    transfer_under_fallback = True
                    warm_ship_reason = "transfer_under_fallback"
                else:
                    log(f"L={L}: polish CORE OK ({cmp['overall']}) — ship polish")

        (S.OUT / f"train_summary_L{L}.json").write_text(
            json.dumps(
                _jsonable({
                    "seeds": seed_rows,
                    "best_seed": best_all["seed"],
                    "L": L,
                    "n_seeds_used_warm": sum(1 for r in seed_rows if r.get("used_warm")),
                    "best_used_warm": used_warm_best,
                    "transfer_under_fallback": transfer_under_fallback,
                    "warm_ship_reason": warm_ship_reason,
                    "polish_compare": polish_cmp_saved,
                    "qaoa_source": "paper_n6",
                    "eval_ns_core": list(ns),
                }),
                indent=2,
            )
        )

        if best_all.get("state") is None:
            st_warm, _ = _warm_state(L, betas0, gammas0, seed=best_all["seed"])
            best_all["state"] = st_warm
            torch.save(st_warm, S.OUT / f"params_best_L{L}.pt")

        meta = {
            "recipe": S.RECIPE_NAME, "L": L, "phase": "core",
            "eval_ns": list(ns), "b": S.B_BITS,
            "optimizer": S.OPTIMIZER, "train_ns": list(S.TRAIN_NS),
            "max_train": S.MAX_TRAIN, "scale_B": S.SCALE_B,
            "best_seed": best_all["seed"], "qaoa_ref": qref,
            "qaoa_source": "paper_n6",
            "train_summary": seed_rows,
            "best_used_warm": used_warm_best,
            "transfer_under_fallback": transfer_under_fallback,
            "warm_ship_reason": warm_ship_reason,
            "polish_compare": polish_cmp_saved,
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
            f"L={L} CORE accepted={cmp['overall']} (warm {warm_cmp['overall']}; "
            f"ship={warm_ship_reason or 'polish'}) "
            f"wall={meta['wall_s']/60:.1f}min — keep climbing"
        )
        _git_push_L(L, cmp, warm_cmp, "core")
        surviving.append(L)
    log("PHASE1 CORE DONE — climbed all requested L (t2527u: no STOP on polish UNDER)")
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
        # Also honor CORE all_L meta if ckpt predates reject_reason flag
        meta0 = (all_L.get(L) or {}).get("meta") or {}
        if meta0.get("best_used_warm") or meta0.get("transfer_under_fallback"):
            used_warm = True
        if used_warm:
            rows, cmp = warm_rows, warm_cmp
            log(f"L={L} {phase}: accepted==warm (ship={meta0.get('warm_ship_reason', 'warm')})")
        else:
            log(f"=== L={L} {phase} accepted eval ===")
            rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=ns)
            cmp = compare_vs_qaoa(L, rows, qref, betas0, gammas0, device)
            if cmp["under"]:
                log(
                    f"L={L} {phase}: polish UNDER on extra ns — ship warm for this pass "
                    f"(CORE already decided; t2527u)"
                )
                rows, cmp = warm_rows, warm_cmp
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


def _load_prior_L(L, device):
    """Load a previously written CORE results_L{L}.json into all_L entry."""
    p = S.OUT / f"results_L{L}.json"
    if not p.exists():
        return None
    blob = json.loads(p.read_text())
    return {
        "meta": blob.get("meta", {}),
        "eval": {int(k): v for k, v in (blob.get("eval") or {}).items()},
        "warm_eval": {int(k): v for k, v in (blob.get("warm_eval") or {}).items()},
        "compare": blob.get("compare", {}),
        "warm_compare": blob.get("warm_compare", {}),
        "best_seed": blob.get("best_seed"),
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Nov_3rd Hamming-shell SNAP ladder")
    ap.add_argument("--start-L", type=int, default=None,
                    help="First L in L_LIST to (re)run; prior smaller L loaded if present")
    ap.add_argument("--skip-train-L", type=int, nargs="*", default=[],
                    help="Reuse prior train/RESULTS for these L; apply t2527u ship-warm rule")
    args = ap.parse_args(argv)

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
    log(
        f"t2527u rule: polish CORE UNDER → ship warm (TIE dual), KEEP climbing; "
        f"start_L={args.start_L} skip_train_L={args.skip_train_L}"
    )

    S.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    E_train = load_train_E(device)
    E_val, _ = load_split_E(6, "val", S.MAX_VAL, device, data_dir=S.SEALED_DATA)

    # Prefill all_L / surviving with prior L < start_L (e.g. L=1 when restarting at 4)
    prefill = {}
    pre_surv = []
    if args.start_L is not None:
        for L in S.L_LIST:
            if L >= int(args.start_L):
                break
            prior = _load_prior_L(L, device)
            if prior is None:
                log(f"WARN: missing prior results_L{L}.json before start_L={args.start_L}")
                continue
            prefill[L] = prior
            pre_surv.append(L)
            log(f"prefill L={L} overall={prior.get('compare', {}).get('overall')}")

    all_L, surviving = run_core_ladder(
        device, E_train, E_val,
        start_L=args.start_L,
        skip_train_L=args.skip_train_L,
    )
    # merge prefill
    merged = dict(prefill)
    merged.update(all_L)
    all_L = merged
    surviving = pre_surv + [L for L in surviving if L not in pre_surv]
    write_summary(all_L, "core")

    if not surviving:
        log("No surviving L — stop before n18/n21")
        sys.exit(4)

    all_L = run_extend_pass(all_L, surviving, S.EVAL_NS_PASS2, "n18", device)
    all_L = run_extend_pass(all_L, surviving, S.EVAL_NS_PASS3, "n21", device)
    log("DONE all phases")


if __name__ == "__main__":
    main()
