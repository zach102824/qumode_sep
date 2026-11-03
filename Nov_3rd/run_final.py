#!/usr/bin/env python3
"""Nov_3rd ladder: L small→large; after each L compare vs QAOA; STOP if UNDER."""
from __future__ import annotations

import json
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
)
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
    # chunk for memory
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
        if n in qref and qref[n].get("Med") is not None:
            qmed = float(qref[n]["Med"])
        else:
            log(f"L{L} n{n}: QAOA Med missing in results — computing ideal QAOA…")
            qmed = qaoa_med_on_scale(L, n, betas, gammas, S.SCALE_B, device)
        if qmed is None or not np.isfinite(qmed) or qmed <= 0:
            per[n] = {"Med": m["median_inv_p"], "B90": m["B90_inv_p"], "R": None, "verdict": "NO_QAOA"}
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
            "R": R,
            "verdict": v,
        }
        log(f"L{L} n{n}: Med={m['median_inv_p']:.3f} Q={qmed:.3f} R={R:.3f}× → {v}")
    overall = "UNDER" if under else (
        "BEAT" if any(p.get("verdict") == "BEAT" for p in per.values()) else "TIE"
    )
    return {"per_n": per, "overall": overall, "under": under}


def write_L_results(L: int, rows: dict, best_seed: int, meta: dict, cmp: dict):
    lines = [
        f"# Nov_3rd — RESULTS L={L}",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"Overall vs QAOA: **{cmp['overall']}** (UNDER stops ladder)",
        f"Best seed = {best_seed}",
        (
            f"Warm recovery: **yes** (polish rejected; exact QAOA dual shipped)"
            if meta.get("best_used_warm")
            else "Warm recovery: no (polish train Med ≤ warm)"
        ),
        "",
        "| n | Med | B90 | QAOA Med | R | verdict |",
        "|---|-----|-----|----------|---|---------|",
    ]
    for n, p in sorted(cmp["per_n"].items()):
        r = "" if p["R"] is None else f"{p['R']:.3f}×"
        qm = "" if p.get("QAOA_Med") is None else f"{p['QAOA_Med']:.3f}"
        lines.append(
            f"| {n} | {p['Med']:.3f} | {p['B90']:.3f} | {qm} | {r} | {p['verdict']} |"
        )
    lines.append("")
    if cmp["under"]:
        lines += [
            "## STOP",
            "At least one n is UNDER QAOA (R>1.02). "
            "Per plan: something missing/wrong — do not continue larger L.",
            "",
        ]
    (S.OUT / f"RESULTS_L{L}.md").write_text("\n".join(lines) + "\n")
    (S.OUT / f"results_L{L}.json").write_text(
        json.dumps(_jsonable({"meta": meta, "eval": rows, "compare": cmp, "best_seed": best_seed}), indent=2)
    )


def write_summary(all_L: dict):
    ns = sorted({n for b in all_L.values() for n in b["eval"]})
    lines = [
        "# Nov_3rd — SUMMARY",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"L set={list(S.L_LIST)}; eval sim ns={list(S.EVAL_NS_SIM)}; gate R≤{TIE_HI} vs QAOA",
        "",
        "## Med",
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
    all_L = {}
    for L in S.L_LIST:
        if not S.qaoa_angles_path(L).exists():
            log(f"STOP: missing QAOA angles for L={L} at {S.qaoa_angles_path(L)}")
            log("Train/copy angles into Nov_3rd/qaoa_angles/ then re-run")
            write_summary(all_L)
            sys.exit(3)
        betas0, gammas0 = load_qaoa(L)
        log(f"=== L={L} train ===")
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
        rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=S.EVAL_NS)
        qref = load_qaoa_scale_ref(L)
        cmp = compare_vs_qaoa(L, rows, qref, betas0, gammas0, device)
        meta = {
            "recipe": S.RECIPE_NAME, "L": L, "b": S.B_BITS,
            "optimizer": S.OPTIMIZER, "train_ns": list(S.TRAIN_NS),
            "max_train": S.MAX_TRAIN, "scale_B": S.SCALE_B,
            "best_seed": best_all["seed"], "qaoa_ref": qref,
            "train_summary": seed_rows, "compare": cmp,
            "best_used_warm": bool(
                best_all.get("used_warm") or best_all.get("rejected_polish")
            ),
            "best_warm_med": best_all.get("warm_med"),
            "best_train_med": best_all.get("train_med"),
        }
        write_L_results(L, rows, best_all["seed"], meta, cmp)
        all_L[L] = {"meta": meta, "eval": rows, "compare": cmp, "best_seed": best_all["seed"]}
        write_summary(all_L)
        log(f"L={L} overall {cmp['overall']}")
        if cmp["under"]:
            log(f"STOP ladder at L={L}: UNDER QAOA — investigate before larger L")
            sys.exit(4)
    log("DONE all L — all TIE/BEAT")


if __name__ == "__main__":
    main()
