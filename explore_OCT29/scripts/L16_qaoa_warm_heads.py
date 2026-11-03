#!/usr/bin/env python3
"""Items 2–3 — L=16 QAOA-warm head ladder / E4-gap polish on a fixed packing b.

Train n9=200 + n12=128; eval scale1000 n9/12/15/(18 if n%b==0).
Compare to QAOA L=16 and (optional) Arm C champ numbers from RESULTS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.blockmix import (  # noqa: E402
    blockmix_psucc,
    expand_head_theta,
    qaoa_theta_from_beta,
)
from explore_OCT29.lib.qaoa import inv_p_from_psucc, summarize_inv  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _ROOT / "Nov" / "runs" / "snap_qaoa_warm" / "L16"
QAOA_PATH = _PKG / "runs" / "qaoa_L16_scale1000" / "angles.json"

# Arm C champ Med (scale1000) from prior ladder — for comparison only
ARM_C = {9: 1.44, 12: 1.99, 15: 2.92, 18: 4.57}
QAOA_REF = {9: 2.24, 12: 3.26, 15: 5.10, 18: 7.89}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[L16]", *a, flush=True)


def load_qaoa():
    blob = json.loads(QAOA_PATH.read_text())
    best = blob["best"]
    betas = torch.tensor(best["betas"], dtype=torch.float32).reshape(-1)
    gammas = torch.tensor(best["gammas"], dtype=torch.float32).reshape(-1)
    assert betas.numel() == 16
    return betas, gammas


def load_E(n, data_dir, split, max_count):
    pack = load_pack(n, data_dir=Path(data_dir))
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"no E n={n} {split}")
    return torch.as_tensor(E, dtype=torch.float32)


def init_head_from_qaoa(mode: str, betas: torch.Tensor, b: int) -> torch.Tensor:
    L = int(betas.numel())
    if mode == "x_pop":
        return (-betas).reshape(L, 1).contiguous()
    if mode == "shell":
        pcs = torch.arange(b + 1, dtype=torch.float32)
        return (-betas).reshape(L, 1) * pcs.reshape(1, -1)
    if mode == "gauss":
        out = torch.zeros(L, 3)
        out[:, 1] = -betas
        return out
    if mode == "free":
        return qaoa_theta_from_beta(betas, b)
    raise ValueError(mode)


def thetas_from_params(mode, params, b):
    return torch.stack([expand_head_theta(mode, params[ell], b) for ell in range(params.shape[0])], 0)


def batch_loss(mode, gammas, params, E_batch, n, b):
    thetas = thetas_from_params(mode, params, b)
    ps = blockmix_psucc(gammas, thetas, E_batch, n, b)
    return torch.log((1.0 / ps.clamp(min=1e-12))).mean()


def train_multi_n(mode, b, betas0, gammas0, packs, *, steps, lr, seed, freeze_gamma, chunk, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    L = 16
    params = torch.nn.Parameter(init_head_from_qaoa(mode, betas0, b).to(device))
    if freeze_gamma:
        gammas = gammas0.detach().to(device).reshape(L)
        opt = torch.optim.Adam([params], lr=lr)
    else:
        gammas = torch.nn.Parameter(gammas0.detach().to(device).reshape(L).clone())
        opt = torch.optim.Adam([params, gammas], lr=lr)
    packs = [(n, E.to(device)) for n, E in packs]
    rng = np.random.default_rng(seed)
    best = {"loss": float("inf"), "params": None, "gammas": None, "step": -1}
    t0 = time.time()
    with torch.no_grad():
        n0, E0 = packs[0]
        loss0 = float(batch_loss(mode, gammas.detach() if freeze_gamma else gammas, params, E0[: min(chunk, E0.shape[0])], n0, b))
    log(f"  init loss≈{loss0:.4f} mode={mode} b={b} seed={seed}")
    for step in range(steps):
        opt.zero_grad()
        n, E = packs[step % len(packs)]
        idx = rng.choice(E.shape[0], size=min(chunk, E.shape[0]), replace=False)
        loss = batch_loss(mode, gammas, params, E[idx], n, b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params] + ([] if freeze_gamma else [gammas]), 5.0)
        opt.step()
        lv = float(loss.detach())
        if lv < best["loss"]:
            best = {
                "loss": lv,
                "params": params.detach().cpu().clone(),
                "gammas": (gammas.detach().cpu().clone() if not freeze_gamma else gammas0.detach().cpu().clone()),
                "step": step,
            }
        if step % 50 == 0 or step == steps - 1:
            log(f"  step {step:4d}/{steps} loss={lv:.4f} best={best['loss']:.4f}")
    best["wall_s"] = time.time() - t0
    best["init_loss"] = loss0
    return best


@torch.no_grad()
def eval_n(mode, b, gammas, params, E, n, chunk=4):
    thetas = thetas_from_params(mode, params, b)
    invs = []
    for s in range(0, E.shape[0], chunk):
        ps = blockmix_psucc(gammas, thetas, E[s : s + chunk], n, b).cpu().numpy()
        invs.extend(inv_p_from_psucc(ps).tolist())
    return summarize_inv(invs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=3)
    ap.add_argument("--heads", type=str, default="gauss,shell,free")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--B_eval", type=int, default=1000)
    ap.add_argument("--eval_ns", type=str, default="9,12,15,18")
    ap.add_argument("--freeze_gamma", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    b = int(args.b)
    heads = [x.strip() for x in args.heads.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    eval_ns = [n for n in eval_ns if n % b == 0]
    train_ns = [n for n in (9, 12) if n % b == 0]
    if not train_ns:
        raise SystemExit(f"no train n divisible by b={b}")

    tag = args.tag or f"b{b}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    betas0, gammas0 = load_qaoa()
    log(f"QAOA L=16 warm; packing b={b}; heads={heads}; eval_ns={eval_ns}")

    packs = []
    counts = {9: 200, 12: 128}
    for n in train_ns:
        E = load_E(n, _PKG / "data", "train", counts[n])
        packs.append((n, E))
        log(f"train n={n} B={len(E)}")

    E_eval = {n: load_E(n, _PKG / "data" / "scale1000", "scale1000", args.B_eval) for n in eval_ns}

    qaoa_base = {}
    with torch.no_grad():
        for n, E in E_eval.items():
            s = eval_n("x_pop", 1, gammas0, init_head_from_qaoa("x_pop", betas0, 1), E, n, chunk=4)
            qaoa_base[n] = s
            log(f"QAOA n{n} Med={s['median_inv_p']:.4f}")

    rows = []
    for mode in heads:
        for seed in seeds:
            run_dir = out / f"{mode}_s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            marker = run_dir / "done.json"
            if marker.exists():
                log(f"skip {mode}_s{seed}")
                rows.append(json.loads(marker.read_text()))
                continue
            log(f"=== {mode} s{seed} ===")
            best = train_multi_n(
                mode, b, betas0, gammas0, packs,
                steps=args.steps, lr=args.lr, seed=seed,
                freeze_gamma=args.freeze_gamma, chunk=args.chunk, device=device,
            )
            evals = {}
            for n, E in E_eval.items():
                summ = eval_n(mode, b, best["gammas"], best["params"], E, n, chunk=4)
                R = summ["median_inv_p"] / qaoa_base[n]["median_inv_p"]
                arm = ARM_C.get(n)
                evals[str(n)] = {
                    **summ,
                    "R": R,
                    "QAOA_med": qaoa_base[n]["median_inv_p"],
                    "ArmC_med": arm,
                    "R_vs_ArmC": (summ["median_inv_p"] / arm) if arm else None,
                }
                extra = f" ArmC×={evals[str(n)]['R_vs_ArmC']:.3f}" if arm else ""
                log(f"  n{n}: Med={summ['median_inv_p']:.4f} R={R:.4f}×{extra}")
            rec = {
                "mode": mode, "seed": seed, "b": b, "L": 16,
                "train_best_loss": best["loss"], "train_wall_s": best["wall_s"],
                "evals": evals, "when": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            }
            torch.save({"params": best["params"], "gammas": best["gammas"], "mode": mode, "b": b}, run_dir / "params.pt")
            marker.write_text(json.dumps(rec, indent=2))
            rows.append(rec)

    (out / "all_rows.json").write_text(json.dumps(rows, indent=2))
    lines = [
        f"# L=16 QAOA-warm heads b={b}",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M:%S %z')}_",
        "",
        "| mode | seed | " + " | ".join(f"n{n} Med/R/ArmC×" for n in eval_ns) + " |",
        "|------|------|" + "|".join(["---"] * len(eval_ns)) + "|",
    ]
    for r in rows:
        cells = []
        for n in eval_ns:
            e = r["evals"][str(n)]
            ac = e.get("R_vs_ArmC")
            acs = f"{ac:.3f}" if ac is not None else "-"
            cells.append(f"{e['median_inv_p']:.3f}/{e['R']:.3f}/{acs}")
        lines.append(f"| {r['mode']} | {r['seed']} | " + " | ".join(cells) + " |")
    (out / "RESULTS.md").write_text("\n".join(lines))
    log(f"wrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
