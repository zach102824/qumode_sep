#!/usr/bin/env python3
"""Item 1 — PACK-12 screen: QAOA-warm SNAP heads across bits-per-cavity b.

n=12 train (128) + n=12 eval B=1000 (or --B). Heads: x_pop, gauss, shell, free.
Always init θ from QAOA: θ_B = -β · pc(B), γ = γ_QAOA, then Adam polish.

Writes Nov/runs/snap_qaoa_warm/pack12/
"""
from __future__ import annotations

import argparse
import json
import math
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
    head_dim,
    qaoa_theta_from_beta,
)
from explore_OCT29.lib.qaoa import inv_p_from_psucc, summarize_inv  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _ROOT / "Nov" / "runs" / "snap_qaoa_warm" / "pack12"
QAOA_DIRS = {
    1: _PKG / "runs" / "qaoa_L1_scale1000",
    2: _PKG / "runs" / "qaoa_L2_scale1000",
    4: _PKG / "runs" / "qaoa_L4_scale1000",
    8: _PKG / "runs" / "qaoa_L8_scale1000",
    16: _PKG / "runs" / "qaoa_L16_scale1000",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[pack12]", *a, flush=True)


def load_qaoa(L: int):
    path = QAOA_DIRS[L] / "angles.json"
    if not path.exists():
        raise FileNotFoundError(path)
    blob = json.loads(path.read_text())
    best = blob["best"]
    betas = torch.tensor(best["betas"], dtype=torch.float32).reshape(-1)
    gammas = torch.tensor(best["gammas"], dtype=torch.float32).reshape(-1)
    assert betas.numel() == L
    return betas, gammas, path


def load_E(n, data_dir, split, max_count):
    pack = load_pack(n, data_dir=Path(data_dir))
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"no E n={n} {split}")
    return torch.as_tensor(E, dtype=torch.float32), list(map(int, seeds))


def init_head_from_qaoa(mode: str, betas: torch.Tensor, b: int) -> torch.Tensor:
    """Lift QAOA β into head params so expand_head_theta ≈ -β·pc (up to global phase)."""
    L = int(betas.numel())
    d = head_dim(mode, b)
    if mode == "x_pop":
        # expand: θ = β_head * pc ; want -β_QAOA * pc ⇒ β_head = -β_QAOA
        return (-betas).reshape(L, 1).contiguous()
    if mode == "shell":
        # φ_pc = -β * pc for pc=0..b
        pcs = torch.arange(b + 1, dtype=torch.float32)
        return (-betas).reshape(L, 1) * pcs.reshape(1, -1)
    if mode == "gauss":
        # α + β_g pc + δ pc² ; set α=0, β_g=-β_QAOA, δ=0
        out = torch.zeros(L, 3)
        out[:, 1] = -betas
        return out
    if mode == "free":
        return qaoa_theta_from_beta(betas, b)  # already -β·pc
    if mode == "bits":
        # θ = Σ α_j bit_j ; pc = Σ bit_j ⇒ α_j = -β all j
        out = torch.zeros(L, b)
        out[:] = (-betas).reshape(L, 1)
        return out
    raise ValueError(mode)


def thetas_from_params(mode: str, params: torch.Tensor, b: int) -> torch.Tensor:
    L = params.shape[0]
    rows = []
    for ell in range(L):
        rows.append(expand_head_theta(mode, params[ell], b))
    return torch.stack(rows, dim=0)


def batch_loss(mode, gammas, params, E_batch, n, b):
    thetas = thetas_from_params(mode, params, b)
    ps = blockmix_psucc(gammas, thetas, E_batch, n, b)
    inv = 1.0 / ps.clamp(min=1e-12)
    return torch.log(inv).mean()


def train_one(
    mode, b, L, betas0, gammas0, E_train, *,
    steps, lr, seed, freeze_gamma, chunk, device,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = 12
    params = torch.nn.Parameter(init_head_from_qaoa(mode, betas0, b).to(device))
    if freeze_gamma:
        gammas = gammas0.detach().to(device).reshape(L)
        opt = torch.optim.Adam([params], lr=lr)
    else:
        gammas = torch.nn.Parameter(gammas0.detach().to(device).reshape(L).clone())
        opt = torch.optim.Adam([params, gammas], lr=lr)

    E_train = E_train.to(device)
    Btr = E_train.shape[0]
    rng = np.random.default_rng(seed)
    best = {"loss": float("inf"), "params": None, "gammas": None, "step": -1}
    t0 = time.time()
    with torch.no_grad():
        # init loss on a chunk
        idx = list(range(min(chunk, Btr)))
        loss0 = float(batch_loss(mode, gammas.detach() if freeze_gamma else gammas, params, E_train[idx], n, b))
    log(f"  init loss≈{loss0:.4f} mode={mode} b={b} L={L} freezeγ={freeze_gamma} seed={seed}")

    for step in range(steps):
        opt.zero_grad()
        # mini-batch
        idx = rng.choice(Btr, size=min(chunk, Btr), replace=False)
        loss = batch_loss(mode, gammas, params, E_train[idx], n, b)
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
def eval_n12(mode, b, gammas, params, E, chunk=16):
    n = 12
    thetas = thetas_from_params(mode, params, b)
    invs = []
    B = E.shape[0]
    for s in range(0, B, chunk):
        ps = blockmix_psucc(gammas, thetas, E[s : s + chunk], n, b).cpu().numpy()
        invs.extend(inv_p_from_psucc(ps).tolist())
    return summarize_inv(invs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--bs", type=str, default="1,2,3,4,6,12")
    ap.add_argument("--heads", type=str, default="x_pop,gauss,shell,free")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--B_eval", type=int, default=1000)
    ap.add_argument("--freeze_gamma", action="store_true")
    ap.add_argument("--train_data", type=str, default=str(_PKG / "data"))
    ap.add_argument("--eval_data", type=str, default=str(_PKG / "data" / "scale1000"))
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="scale1000")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--only", type=str, default="", help="optional filter tag substring")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    L = int(args.L)
    bs = [int(x) for x in args.bs.split(",") if x.strip()]
    heads = [x.strip() for x in args.heads.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    out_L = OUT / f"L{L}" / ("freezeG" if args.freeze_gamma else "jointG")
    out_L.mkdir(parents=True, exist_ok=True)

    betas0, gammas0, apath = load_qaoa(L)
    log(f"QAOA warm L={L} from {apath}")

    E_train, _ = load_E(12, args.train_data, args.train_split, 128)
    E_eval, _ = load_E(12, args.eval_data, args.eval_split, args.B_eval)
    log(f"train B={len(E_train)} eval B={len(E_eval)}")

    # baseline QAOA med at n12 from dual identity (b=1 x_pop no train)
    with torch.no_grad():
        q_sum = eval_n12("x_pop", 1, gammas0, init_head_from_qaoa("x_pop", betas0, 1), E_eval, chunk=16)
    log(f"QAOA n12 Med={q_sum['median_inv_p']:.4f} B90={q_sum['B90_inv_p']:.4f}")

    rows = []
    for b in bs:
        if 12 % b != 0:
            log(f"skip b={b} (not dividing 12)")
            continue
        for mode in heads:
            # degeneracies
            if b == 1 and mode not in ("x_pop", "bits", "free", "shell"):
                # gauss on b=1: pc∈{0,1} → α+β pc + δ pc² spans affine on 2 levels = free; allow
                pass
            if b == 1 and mode == "gauss":
                pass
            if b == 2 and mode == "gauss":
                # Opus: gauss equals shell at b=2 in expressivity for pc; still run both
                pass
            for seed in seeds:
                tag = f"b{b}_{mode}_s{seed}"
                if args.only and args.only not in tag:
                    continue
                run_dir = out_L / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                marker = run_dir / "done.json"
                if marker.exists():
                    log(f"skip existing {tag}")
                    rows.append(json.loads(marker.read_text()))
                    continue
                log(f"=== train {tag} ===")
                try:
                    best = train_one(
                        mode, b, L, betas0, gammas0, E_train,
                        steps=args.steps, lr=args.lr, seed=seed,
                        freeze_gamma=args.freeze_gamma, chunk=args.chunk, device=device,
                    )
                    summ = eval_n12(mode, b, best["gammas"], best["params"], E_eval, chunk=16)
                    R = summ["median_inv_p"] / q_sum["median_inv_p"]
                    rec = {
                        "tag": tag,
                        "b": b,
                        "mode": mode,
                        "seed": seed,
                        "L": L,
                        "freeze_gamma": args.freeze_gamma,
                        "train_best_loss": best["loss"],
                        "train_init_loss": best["init_loss"],
                        "train_best_step": best["step"],
                        "train_wall_s": best["wall_s"],
                        "eval": summ,
                        "QAOA_med": q_sum["median_inv_p"],
                        "R": R,
                        "when": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    }
                    torch.save(
                        {"params": best["params"], "gammas": best["gammas"], "mode": mode, "b": b, "L": L},
                        run_dir / "params.pt",
                    )
                    marker.write_text(json.dumps(rec, indent=2))
                    rows.append(rec)
                    log(f"DONE {tag}: Med={summ['median_inv_p']:.4f} R={R:.4f}× QAOA wall={best['wall_s']:.0f}s")
                except Exception as e:
                    log(f"FAIL {tag}: {e}")
                    rows.append({"tag": tag, "error": str(e), "pass": False})

    # summarize
    (out_L / "all_rows.json").write_text(json.dumps(rows, indent=2))
    # best per (b,mode)
    lines = [
        f"# PACK-12 L={L} freezeγ={args.freeze_gamma}",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M:%S %z')}_",
        "",
        f"QAOA n12 Med={q_sum['median_inv_p']:.4f} (warm baseline)",
        "",
        "| b | mode | seed | Med | B90 | R vs QAOA | train_loss | wall_s |",
        "|---|------|------|-----|-----|-----------|------------|--------|",
    ]
    for r in sorted(rows, key=lambda x: (x.get("b", 99), x.get("mode", ""), x.get("seed", 0))):
        if "eval" not in r:
            continue
        e = r["eval"]
        lines.append(
            f"| {r['b']} | {r['mode']} | {r['seed']} | {e['median_inv_p']:.4f} | "
            f"{e['B90_inv_p']:.4f} | {r['R']:.4f}× | {r['train_best_loss']:.4f} | {r['train_wall_s']:.0f} |"
        )
    # best R per (b,mode)
    lines += ["", "## Best seed per (b, mode)", "", "| b | mode | best R | Med |", "|---|------|--------|-----|"]
    best_map = {}
    for r in rows:
        if "R" not in r:
            continue
        k = (r["b"], r["mode"])
        if k not in best_map or r["R"] < best_map[k]["R"]:
            best_map[k] = r
    for (b, mode), r in sorted(best_map.items()):
        lines.append(f"| {b} | {mode} | {r['R']:.4f}× | {r['eval']['median_inv_p']:.4f} |")

    # pass/kill vs b=3
    lines += ["", "## vs b=3 (same head, best seed)", ""]
    for mode in heads:
        ref = best_map.get((3, mode))
        if not ref:
            continue
        lines.append(f"### mode={mode} (b=3 R={ref['R']:.4f}×)")
        for b in bs:
            o = best_map.get((b, mode))
            if not o:
                continue
            ratio = o["R"] / ref["R"]
            flag = "WIN" if ratio <= 0.95 else ("TIE" if ratio <= 1.02 else "LOSE")
            lines.append(f"- b={b}: R={o['R']:.4f}× → {ratio:.3f}× vs b=3  [{flag}]")
    (out_L / "RESULTS.md").write_text("\n".join(lines))
    log(f"wrote {out_L / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
