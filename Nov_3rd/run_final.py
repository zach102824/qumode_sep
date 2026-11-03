#!/usr/bin/env python3
"""Entry: train Nov_3rd (L-BFGS, n6×1000) + scale1000 eval."""
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


def write_results(rows: dict, best_seed: int, meta: dict):
    S.OUT.mkdir(parents=True, exist_ok=True)
    qref = meta.get("qaoa_ref") or {}
    lines = [
        "# Nov_3rd — RESULTS",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Recipe (locked)",
        f"- Motif: `|0⟩→[Uc^X→SNAP]^{{L}}→W→Fock`, W=H^{{\\otimes {S.B_BITS}}}` (BinH)",
        "- QAOA-warm shared **shell** SNAP",
        f"- Train: n={list(S.TRAIN_NS)} × {S.MAX_TRAIN}; **{S.OPTIMIZER}**",
        f"- L={meta['L']}; eval scale1000 B={S.SCALE_B}",
        "",
        f"## Best seed = {best_seed}",
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
    lines.append("")
    (S.OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (S.OUT / "results.json").write_text(
        json.dumps(_jsonable({"meta": meta, "eval": rows, "best_seed": best_seed}), indent=2)
    )
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
    )
    S.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    L = S.L_PRIMARY

    betas0, gammas0 = load_qaoa(L)
    log(
        f"START recipe={S.RECIPE_NAME} L={L} b={S.B_BITS} W={S.W_FRAME} "
        f"head={S.SNAP_HEAD} opt={S.OPTIMIZER} train={S.TRAIN_NS}×{S.MAX_TRAIN}"
    )
    E_train = load_train_E(device)
    E_val, _ = load_split_E(6, "val", S.MAX_VAL, device, data_dir=S.SEALED_DATA)

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
        pt = S.OUT / f"params_seed{seed}_L{L}.pt"
        torch.save(best["state"], pt)
        log(f"saved {pt}")
        if best_all is None or best["score"] < best_all["score"]:
            best_all = {**best, "seed": seed}

    torch.save(best_all["state"], S.OUT / f"params_best_L{L}.pt")
    (S.OUT / "train_summary.json").write_text(
        json.dumps(_jsonable({"seeds": seed_rows, "best_seed": best_all["seed"]}), indent=2)
    )

    st = best_all["state"]
    rows = eval_ladder(L, st["gamma"], st["phi"], device)
    meta = {
        "recipe": S.RECIPE_NAME,
        "L": L,
        "b": S.B_BITS,
        "W": S.W_FRAME,
        "snap_mode": S.SNAP_HEAD,
        "optimizer": S.OPTIMIZER,
        "train_ns": list(S.TRAIN_NS),
        "max_train": S.MAX_TRAIN,
        "scale_B": S.SCALE_B,
        "best_seed": best_all["seed"],
        "qaoa_ref": load_qaoa_scale_ref(L),
        "train_summary": seed_rows,
        "forbidden_note": list(S.FORBIDDEN),
    }
    write_results(rows, best_all["seed"], meta)
    log("DONE")


if __name__ == "__main__":
    main()
