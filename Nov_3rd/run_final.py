#!/usr/bin/env python3
"""Nov_3rd ladder: L small→large; after each L compare vs QAOA; STOP if UNDER.

Fair same-L QAOA: paper init β=0.01 γ=-0.01, nstart=1, train n6×1000
(see ensure_qaoa / qaoa_init). Warm dual φ_j=-β·j; polish rejected if
train Med > warm (exact QAOA dual recovered).
"""
from __future__ import annotations

import json
import os
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
def qaoa_med_on_scale(L: int, n: int, betas, gammas, max_count: int, device) -> float | None:
    """Fallback: compute ideal QAOA Med on local scale1000 if results.json missing n."""
    pack = S.SCALE1000_DIR / f"data_n{n}.npz"
    if not pack.exists():
        return None
    E, _ = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    invs = []
    B = len(E)
    chunk = 32 if n >= 15 else 64
    for i0 in range(0, B, chunk):
        Eb = E[i0 : i0 + chunk]
        ps = qaoa_psucc(betas, gammas, Eb, n)
        invs.extend((1.0 / ps.clamp(min=1e-12)).cpu().numpy().tolist())
    return float(np.median(np.asarray(invs, dtype=np.float64)))


def compare_vs_qaoa(L: int, rows: dict, qref: dict, betas, gammas, device) -> dict:
    """Return per-n R and verdict; overall UNDER if any n UNDER."""
    per = {}
    under = False
    for n, m in sorted(rows.items()):
        qmed = None
        qb90 = None
        if n in qref and qref[n].get("Med") is not None:
            qmed = float(qref[n]["Med"])
            qb90 = qref[n].get("B90")
        else:
            log(f"L{L} n{n}: QAOA Med missing in results — computing ideal QAOA…")
            qmed = qaoa_med_on_scale(L, n, betas, gammas, S.SCALE_B, device)
        if qmed is None or not np.isfinite(qmed) or qmed <= 0:
            per[n] = {
                "Med": m["median_inv_p"],
                "B90": m["B90_inv_p"],
                "R": None,
                "verdict": "NO_QAOA",
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
            "Med": m["median_inv_p"],
            "B90": m["B90_inv_p"],
            "QAOA_Med": qmed,
            "QAOA_B90": qb90,
            "R": R,
            "verdict": v,
        }
        log(f"L{L} n{n}: Med={m['median_inv_p']:.3f} Q={qmed:.3f} R={R:.3f}× → {v}")
    overall = "UNDER" if under else (
        "BEAT" if any(p.get("verdict") == "BEAT" for p in per.values()) else "TIE"
    )
    return {"per_n": per, "overall": overall, "under": under}


def write_L_results(
    L: int,
    rows: dict,
    warm_rows: dict,
    best_seed: int,
    meta: dict,
    cmp: dict,
    warm_cmp: dict,
):
    lines = [
        f"# Nov_3rd — RESULTS L={L}",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"Overall vs QAOA (accepted ckpt): **{cmp['overall']}** (UNDER stops ladder)",
        f"Warm-only vs QAOA: **{warm_cmp['overall']}**",
        f"Best seed = {best_seed}",
        (
            f"Warm recovery: **yes** (polish rejected; exact QAOA dual shipped)"
            if meta.get("best_used_warm")
            else "Warm recovery: no (polish train Med ≤ warm)"
        ),
        "",
        "QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000 (fair same-L).",
        "",
        "| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | "
        "acc Med | acc B90 | R_acc | acc |",
        "|---|----------|----------|----------|----------|--------|------|"
        "---------|---------|-------|-----|",
    ]
    ns = sorted(set(list(cmp["per_n"]) + list(warm_cmp["per_n"])))
    for n in ns:
        w = warm_cmp["per_n"].get(n, {})
        a = cmp["per_n"].get(n, {})
        qm = w.get("QAOA_Med", a.get("QAOA_Med"))
        qb = w.get("QAOA_B90", a.get("QAOA_B90"))
        def f3(x):
            return "" if x is None else f"{float(x):.3f}"
        def fr(x):
            return "" if x is None else f"{float(x):.3f}×"
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
            "At least one n is UNDER QAOA (R>1.02) on the **accepted** ckpt. "
            "Per plan: do not continue larger L.",
            "",
        ]
    (S.OUT / f"RESULTS_L{L}.md").write_text("\n".join(lines) + "\n")
    (S.OUT / f"results_L{L}.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "meta": meta,
                    "eval": rows,
                    "warm_eval": warm_rows,
                    "compare": cmp,
                    "warm_compare": warm_cmp,
                    "best_seed": best_seed,
                }
            ),
            indent=2,
        )
    )


def write_summary(all_L: dict):
    ns = sorted({n for b in all_L.values() for n in b["eval"]})
    lines = [
        "# Nov_3rd — SUMMARY",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"L set={list(S.L_LIST)}; eval sim ns={list(S.EVAL_NS_SIM)}; gate R≤{TIE_HI} vs QAOA",
        "QAOA: paper init n6×1000 fair same-L. Accepted ckpt uses warm-fallback.",
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
    (S.OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (S.OUT / "results_all.json").write_text(json.dumps(_jsonable(all_L), indent=2))


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    torch.set_num_threads(8)

    S.assert_locked(
        b=S.B_BITS, W_frame=S.W_FRAME, snap_head=S.SNAP_HEAD,
        optimizer=S.OPTIMIZER, train_ns=S.TRAIN_NS, max_train=S.MAX_TRAIN,
        qaoa_warm=S.QAOA_WARM, snap_shared_layers=S.SNAP_SHARED_LAYERS,
        L_list=S.L_LIST, eval_ns=S.EVAL_NS,
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

    S.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    E_train = load_train_E(device)
    E_val, _ = load_split_E(6, "val", S.MAX_VAL, device, data_dir=S.SEALED_DATA)

    log(f"START ladder L={list(S.L_LIST)} eval_sim={list(S.EVAL_NS_SIM)}")
    log("QAOA refs: paper init n6×1000 (ensure_qaoa_paper_n6)")
    all_L = {}
    for L in S.L_LIST:
        t_L0 = time.time()
        log(f"=== L={L} ensure fair QAOA ===")
        ensure_qaoa_paper_n6(L, device=device)
        betas0, gammas0 = load_qaoa(L)
        warm_phi = shell_phi_from_betas(betas0)
        log(f"=== L={L} warm-only baseline eval ===")
        warm_rows = eval_ladder(L, gammas0, warm_phi, device, ns=S.EVAL_NS)
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
                    f"L{L} seed{seed}: warm recovered (rejected polish); "
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
        n_warm = sum(1 for r in seed_rows if r.get("used_warm"))
        (S.OUT / f"train_summary_L{L}.json").write_text(
            json.dumps(
                _jsonable({
                    "seeds": seed_rows,
                    "best_seed": best_all["seed"],
                    "L": L,
                    "n_seeds_used_warm": n_warm,
                    "best_used_warm": bool(
                        best_all.get("used_warm") or best_all.get("rejected_polish")
                    ),
                    "qaoa_source": "paper_n6",
                }),
                indent=2,
            )
        )
        if best_all.get("used_warm") or best_all.get("rejected_polish"):
            log(
                f"L{L} best seed{best_all['seed']}: shipping QAOA-warm angles "
                f"(polish rejected on train Med)"
            )
        st = best_all["state"]
        # If warm recovered, accepted == warm; still re-eval for a clean accepted table
        # (cheap when identical; keep protocol uniform).
        log(f"=== L={L} accepted-ckpt eval ===")
        rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=S.EVAL_NS)
        cmp = compare_vs_qaoa(L, rows, qref, betas0, gammas0, device)
        meta = {
            "recipe": S.RECIPE_NAME, "L": L, "b": S.B_BITS,
            "optimizer": S.OPTIMIZER, "train_ns": list(S.TRAIN_NS),
            "max_train": S.MAX_TRAIN, "scale_B": S.SCALE_B,
            "best_seed": best_all["seed"], "qaoa_ref": qref,
            "qaoa_source": "paper_n6",
            "train_summary": seed_rows, "compare": cmp,
            "warm_compare": warm_cmp,
            "best_used_warm": bool(
                best_all.get("used_warm") or best_all.get("rejected_polish")
            ),
            "best_warm_med": best_all.get("warm_med"),
            "best_train_med": best_all.get("train_med"),
            "wall_s": time.time() - t_L0,
        }
        write_L_results(L, rows, warm_rows, best_all["seed"], meta, cmp, warm_cmp)
        all_L[L] = {
            "meta": meta,
            "eval": rows,
            "warm_eval": warm_rows,
            "compare": cmp,
            "warm_compare": warm_cmp,
            "best_seed": best_all["seed"],
        }
        write_summary(all_L)
        log(
            f"L={L} overall {cmp['overall']} (warm {warm_cmp['overall']}) "
            f"wall={meta['wall_s']/60:.1f}min → RESULTS_L{L}.md"
        )
        # commit per L so progress is on main
        try:
            import subprocess
            files = [
                str(S.OUT / f"RESULTS_L{L}.md"),
                str(S.OUT / f"results_L{L}.json"),
                str(S.OUT / f"train_summary_L{L}.json"),
                str(S.OUT / "RESULTS.md"),
                str(S.OUT / "results_all.json"),
                str(S.qaoa_angles_path(L)),
                str(S.qaoa_results_path(L)),
            ]
            # also ckpts
            for seed in S.SEEDS:
                files.append(str(S.OUT / f"params_seed{seed}_L{L}.pt"))
            files.append(str(S.OUT / f"params_best_L{L}.pt"))
            existing = [f for f in files if Path(f).exists()]
            subprocess.run(["git", "add", "--"] + existing, cwd=str(_ROOT), check=False)
            subprocess.run(
                [
                    "git", "commit", "-m",
                    f"Nov_3rd RESULTS L={L} {cmp['overall']} (warm {warm_cmp['overall']}; paper-n6 QAOA)",
                ],
                cwd=str(_ROOT),
                check=False,
            )
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(_ROOT), check=False)
            log(f"L={L}: pushed RESULTS / ckpts")
        except Exception as e:
            log(f"L={L}: git push skipped ({e})")
        if cmp["under"]:
            log(f"STOP ladder at L={L}: UNDER QAOA — investigate before larger L")
            sys.exit(4)
    log("DONE all L — all TIE/BEAT")


if __name__ == "__main__":
    main()
