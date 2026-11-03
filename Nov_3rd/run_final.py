#!/usr/bin/env python3
"""Entry: for each L in L_LIST, L-BFGS train (n6×1000) + scale1000 eval n=9..18."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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


def write_L_results(L: int, rows: dict, best_seed: int, meta: dict):
    qref = meta.get("qaoa_ref") or {}
    lines = [
        f"# Nov_3rd — RESULTS L={L}",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"b={S.B_BITS} bits/cavity; BinH; shell; QAOA-warm; L-BFGS; train n6×{S.MAX_TRAIN}",
        f"Best seed = {best_seed}",
        "",
        "| n | Med | B90 | vs QAOA Med |",
        "|---|-----|-----|-------------|",
    ]
    for n, m in sorted(rows.items()):
        rQ = ""
        if n in qref:
            rQ = f"{m['median_inv_p']/qref[n]['Med']:.3f}×"
        lines.append(
            f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | {rQ} |"
        )
    (S.OUT / f"RESULTS_L{L}.md").write_text("\n".join(lines) + "\n")
    (S.OUT / f"results_L{L}.json").write_text(
        json.dumps(_jsonable({"meta": meta, "eval": rows, "best_seed": best_seed}), indent=2)
    )


def write_summary(all_L: dict):
    ns = sorted({n for block in all_L.values() for n in block["eval"]})
    lines = [
        "# Nov_3rd — SUMMARY (Med)",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"L set = {list(S.L_LIST)}; eval n = {S.EVAL_N_LO}…{S.EVAL_N_HI}; "
        f"b={S.B_BITS} bits/cavity (sim n%3==0 → {list(S.EVAL_NS_SIM)})",
        "",
        "| L | " + " | ".join(f"n{n}" for n in ns) + " |",
        "|---|" + "|".join(["-----"] * len(ns)) + "|",
    ]
    for L in S.L_LIST:
        if L not in all_L:
            continue
        cells = []
        for n in ns:
            m = all_L[L]["eval"].get(n)
            cells.append(f"{m['median_inv_p']:.3f}" if m else "—")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    lines.append("")
    (S.OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (S.OUT / "results_all.json").write_text(json.dumps(_jsonable(all_L), indent=2))
    log("wrote", S.OUT / "RESULTS.md")


def main():
    S.assert_locked(
        b=S.B_BITS,
        W_frame=S.W_FRAME,
        snap_head=S.SNAP_HEAD,
        optimizer=S.OPTIMIZER,
        train_ns=S.TRAIN_NS,
        max_train=S.MAX_TRAIN,
        qaoa_warm=S.QAOA_WARM,
        snap_shared_layers=S.SNAP_SHARED_LAYERS,
        L_list=S.L_LIST,
        eval_ns=S.EVAL_NS,
    )
    S.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    E_train = load_train_E(device)
    E_val, _ = load_split_E(6, "val", S.MAX_VAL, device, data_dir=S.SEALED_DATA)

    log(
        f"START {S.RECIPE_NAME} L_LIST={list(S.L_LIST)} "
        f"eval_n={S.EVAL_N_LO}..{S.EVAL_N_HI} b={S.B_BITS}"
    )

    all_L = {}
    missing_qaoa = []
    for L in S.L_LIST:
        if not S.qaoa_angles_path(L).exists():
            log(f"SKIP L={L}: missing QAOA angles {S.qaoa_angles_path(L)}")
            missing_qaoa.append(L)
            continue
        betas0, gammas0 = load_qaoa(L)
        log(f"=== L={L} ===")
        best_all = None
        seed_rows = []
        for seed in S.SEEDS:
            best = train_seed(
                L=L,
                device=device,
                E_train=E_train,
                betas0=betas0,
                gammas0=gammas0,
                seed=seed,
                E_val=E_val,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "score": best["score"],
                    "round": best["round"],
                    "train_med": best["train_med"],
                    "warm_med": best["warm_med"],
                }
            )
            torch.save(best["state"], S.OUT / f"params_seed{seed}_L{L}.pt")
            if best_all is None or best["score"] < best_all["score"]:
                best_all = {**best, "seed": seed}

        torch.save(best_all["state"], S.OUT / f"params_best_L{L}.pt")
        (S.OUT / f"train_summary_L{L}.json").write_text(
            json.dumps(
                _jsonable({"seeds": seed_rows, "best_seed": best_all["seed"], "L": L}),
                indent=2,
            )
        )
        st = best_all["state"]
        rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=S.EVAL_NS)
        meta = {
            "recipe": S.RECIPE_NAME,
            "L": L,
            "b": S.B_BITS,
            "bits_per_cavity": S.BITS_PER_CAVITY,
            "W": S.W_FRAME,
            "snap_mode": S.SNAP_HEAD,
            "optimizer": S.OPTIMIZER,
            "train_ns": list(S.TRAIN_NS),
            "max_train": S.MAX_TRAIN,
            "scale_B": S.SCALE_B,
            "eval_ns": list(S.EVAL_NS),
            "eval_ns_sim": list(S.EVAL_NS_SIM),
            "L_list": list(S.L_LIST),
            "best_seed": best_all["seed"],
            "qaoa_ref": load_qaoa_scale_ref(L),
            "train_summary": seed_rows,
        }
        write_L_results(L, rows, best_all["seed"], meta)
        all_L[str(L)] = {"meta": meta, "eval": rows, "best_seed": best_all["seed"]}
        # also key by int for summary
        all_L[L] = all_L[str(L)]

    # summary uses int keys only
    summary = {L: all_L[L] for L in S.L_LIST if L in all_L}
    write_summary(summary)
    if missing_qaoa:
        log(f"missing QAOA angles for L={missing_qaoa} — train those then re-run")
    log("DONE")


if __name__ == "__main__":
    main()
