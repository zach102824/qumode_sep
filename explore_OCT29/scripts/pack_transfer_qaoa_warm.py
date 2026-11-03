#!/usr/bin/env python3
"""Item 4 — transfer winning packing to other n at matched L (QAOA-warm always).

Train on n9(+n12 if available) with Adam; eval scale1000 at requested ns.
Requires n % b == 0 for every train/eval n.
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
    head_dim,
    qaoa_theta_from_beta,
)
from explore_OCT29.lib.qaoa import inv_p_from_psucc, summarize_inv  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _ROOT / "Nov" / "runs" / "snap_qaoa_warm" / "transfer"
QAOA_DIRS = {
    1: _PKG / "runs" / "qaoa_L1_scale1000",
    2: _PKG / "runs" / "qaoa_L2_scale1000",
    4: _PKG / "runs" / "qaoa_L4_scale1000",
    8: _PKG / "runs" / "qaoa_L8_scale1000",
    16: _PKG / "runs" / "qaoa_L16_scale1000",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[xfer]", *a, flush=True)


def load_qaoa(L: int):
    path = QAOA_DIRS[L] / "angles.json"
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
    if mode == "bits":
        out = torch.zeros(L, b)
        out[:] = (-betas).reshape(L, 1)
        return out
    raise ValueError(mode)


def thetas_from_params(mode: str, params: torch.Tensor, b: int) -> torch.Tensor:
    rows = [expand_head_theta(mode, params[ell], b) for ell in range(params.shape[0])]
    return torch.stack(rows, dim=0)


def batch_loss(mode, gammas, params, E_batch, n, b):
    thetas = thetas_from_params(mode, params, b)
    ps = blockmix_psucc(gammas, thetas, E_batch, n, b)
    inv = 1.0 / ps.clamp(min=1e-12)
    return torch.log(inv).mean()


def train_multi_n(
    mode, b, L, betas0, gammas0, packs, *,
    steps, lr, seed, freeze_gamma, chunk, device,
):
    """packs: list of (n, E_train). Alternates mini-batches across n."""
    torch.manual_seed(seed)
    np.random.seed(seed)
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
        idx = list(range(min(chunk, E0.shape[0])))
        loss0 = float(batch_loss(mode, gammas.detach() if freeze_gamma else gammas, params, E0[idx], n0, b))
    log(f"  init loss≈{loss0:.4f} mode={mode} b={b} L={L} seed={seed} ns={[n for n,_ in packs]}")

    for step in range(steps):
        opt.zero_grad()
        n, E = packs[step % len(packs)]
        Btr = E.shape[0]
        idx = rng.choice(Btr, size=min(chunk, Btr), replace=False)
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
def eval_n(mode, b, gammas, params, E, n, chunk=8):
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
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--mode", type=str, default="shell")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--B_eval", type=int, default=1000)
    ap.add_argument("--train_ns", type=str, default="9,12")
    ap.add_argument("--eval_ns", type=str, default="9,12,15,18")
    ap.add_argument("--train_counts", type=str, default="200,128")
    ap.add_argument("--freeze_gamma", action="store_true")
    ap.add_argument("--train_data", type=str, default=str(_PKG / "data"))
    ap.add_argument("--eval_data", type=str, default=str(_PKG / "data" / "scale1000"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    L = int(args.L)
    b = int(args.b)
    mode = args.mode
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    train_ns = [int(x) for x in args.train_ns.split(",") if x.strip()]
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    train_counts = [int(x) for x in args.train_counts.split(",") if x.strip()]
    while len(train_counts) < len(train_ns):
        train_counts.append(train_counts[-1])

    for n in train_ns + eval_ns:
        if n % b != 0:
            raise SystemExit(f"n={n} not divisible by b={b} — skip/adjust packing")

    tag = args.tag or f"L{L}_b{b}_{mode}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    betas0, gammas0, apath = load_qaoa(L)
    log(f"QAOA warm L={L} from {apath}")

    packs = []
    for n, cnt in zip(train_ns, train_counts):
        E, _ = load_E(n, args.train_data, "train", cnt)
        packs.append((n, E))
        log(f"train n={n} B={len(E)}")

    E_eval = {}
    for n in eval_ns:
        E, _ = load_E(n, args.eval_data, "scale1000", args.B_eval)
        E_eval[n] = E
        log(f"eval n={n} B={len(E)}")

    # QAOA baselines via dual identity
    qaoa_base = {}
    with torch.no_grad():
        for n, E in E_eval.items():
            # use x_pop b=1 identity OR same-b dual — both = QAOA
            s = eval_n("x_pop", 1 if n % 1 == 0 else b, gammas0, init_head_from_qaoa("x_pop", betas0, 1), E, n, chunk=8)
            # always b=1 works
            qaoa_base[n] = s
            log(f"QAOA n{n} Med={s['median_inv_p']:.4f} B90={s['B90_inv_p']:.4f}")

    rows = []
    for seed in seeds:
        run_dir = out / f"s{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / "done.json"
        if marker.exists():
            log(f"skip existing s{seed}")
            rows.append(json.loads(marker.read_text()))
            continue
        log(f"=== train {tag} s{seed} ===")
        best = train_multi_n(
            mode, b, L, betas0, gammas0, packs,
            steps=args.steps, lr=args.lr, seed=seed,
            freeze_gamma=args.freeze_gamma, chunk=args.chunk, device=device,
        )
        evals = {}
        for n, E in E_eval.items():
            summ = eval_n(mode, b, best["gammas"], best["params"], E, n, chunk=8)
            R = summ["median_inv_p"] / qaoa_base[n]["median_inv_p"]
            evals[str(n)] = {**summ, "R": R, "QAOA_med": qaoa_base[n]["median_inv_p"]}
            log(f"  eval n{n}: Med={summ['median_inv_p']:.4f} R={R:.4f}×")
        rec = {
            "tag": tag,
            "b": b,
            "mode": mode,
            "seed": seed,
            "L": L,
            "freeze_gamma": args.freeze_gamma,
            "train_best_loss": best["loss"],
            "train_wall_s": best["wall_s"],
            "evals": evals,
            "when": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        torch.save(
            {"params": best["params"], "gammas": best["gammas"], "mode": mode, "b": b, "L": L},
            run_dir / "params.pt",
        )
        marker.write_text(json.dumps(rec, indent=2))
        rows.append(rec)

    (out / "all_rows.json").write_text(json.dumps(rows, indent=2))
    # markdown
    lines = [
        f"# Transfer {tag}",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M:%S %z')}_",
        "",
        f"b={b} mode={mode} L={L} train_ns={train_ns}",
        "",
        "| seed | " + " | ".join(f"n{n} Med/R" for n in eval_ns) + " | wall_s |",
        "|------|" + "|".join(["------"] * len(eval_ns)) + "|--------|",
    ]
    for r in rows:
        cells = []
        for n in eval_ns:
            e = r["evals"][str(n)]
            cells.append(f"{e['median_inv_p']:.3f}/{e['R']:.3f}×")
        lines.append(f"| {r['seed']} | " + " | ".join(cells) + f" | {r['train_wall_s']:.0f} |")
    # best by mean R over eval ns
    def mean_R(r):
        return float(np.mean([r["evals"][str(n)]["R"] for n in eval_ns]))
    best = min(rows, key=mean_R)
    lines += ["", f"**Best seed {best['seed']}** mean R={mean_R(best):.4f}", ""]
    for n in eval_ns:
        e = best["evals"][str(n)]
        q = qaoa_base[n]
        lines.append(
            f"- n{n}: Med={e['median_inv_p']:.4f} / B90={e['B90_inv_p']:.4f}  "
            f"(QAOA {q['median_inv_p']:.4f}/{q['B90_inv_p']:.4f})  R={e['R']:.4f}×"
        )
    (out / "RESULTS.md").write_text("\n".join(lines))
    log(f"wrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
