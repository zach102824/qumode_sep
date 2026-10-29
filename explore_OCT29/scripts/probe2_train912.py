#!/usr/bin/env python3
"""Opus probe 2 — train projector on {n9,n12}, zero-shot n15/n18.

Compare to bake-off Phase B (n9-only). Fixed W from driver_bakeoff/W.pt.
No GNN. L=4, shared γ/β.
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

from explore_OCT29.lib.ansatz import AnsatzConfig  # noqa: E402
from explore_OCT29.lib.drivers import (  # noqa: E402
    CompiledW,
    ProjectorAnsatz,
    eval_projector,
    fit_b,
    linear_ramp,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT_DEFAULT = _PKG / "runs" / "probe2_train912"
W_DEFAULT = _PKG / "runs" / "driver_bakeoff" / "W.pt"
BAKE_JSON = _PKG / "runs" / "driver_bakeoff" / "results.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe2]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _loss(out, leakpen=0.1):
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


def eval_ladder(W, gamma, beta, L, nf, ns, max_counts, device):
    out = {}
    for n in ns:
        path = _PKG / "data" / f"data_n{n}.npz"
        if not path.exists():
            log(f"MISSING pack n{n} — skip")
            out[n] = {"error": "missing_pack", "n": n}
            continue
        mc = max_counts.get(n, 8)
        t0 = time.time()
        m = eval_projector(
            n, W, gamma, beta, L=L, split="val", max_count=mc, nf=nf, device=device
        )
        out[n] = m
        log(
            f"eval n{n}: Med={m['median_inv_p']:.4g} B90={m['B90_inv_p']:.4g} "
            f"S={m['S']:.3f} leak={m['mean_leak']:.3e} ({time.time()-t0:.1f}s B={m['B']})"
        )
    return out


def fit_ladder(ladder):
    ns, meds, b90s = [], [], []
    for n in sorted(int(k) for k in ladder if str(k).isdigit() and "median_inv_p" in ladder[k]):
        m = ladder[n]
        ns.append(n)
        meds.append(m["median_inv_p"])
        b90s.append(m["B90_inv_p"])
    fits = {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}
    # ratios
    def med(n):
        m = ladder.get(n) or {}
        return m.get("median_inv_p")
    fits["ratios"] = {
        "Med15_over_Med12": (None if not med(15) or not med(12) else med(15) / med(12)),
        "Med18_over_Med15": (None if not med(18) or not med(15) else med(18) / med(15)),
        "Med12_over_Med9": (None if not med(12) or not med(9) else med(12) / med(9)),
    }
    return fits


def train_projector_912(
    W: CompiledW,
    L: int = 4,
    steps: int = 120,
    max_train: int = 32,
    max_val: int = 16,
    nf: int = 16,
    seed: int = 0,
    Delta0: float = 1.0,
    leakpen: float = 0.1,
    device=None,
    warm_gamma=None,
    warm_beta=None,
):
    """Adam then short LBFGS; train loss = mean over n9 and n12 batches."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if warm_gamma is not None:
        gamma = torch.nn.Parameter(torch.as_tensor(warm_gamma, dtype=torch.float32, device=device))
        beta = torch.nn.Parameter(torch.as_tensor(warm_beta, dtype=torch.float32, device=device))
    else:
        g0, b0 = linear_ramp(L, Delta0)
        gamma = torch.nn.Parameter(torch.as_tensor(g0, dtype=torch.float32, device=device))
        beta = torch.nn.Parameter(torch.as_tensor(b0, dtype=torch.float32, device=device))

    sims = {}
    packs = {}
    Es = {}
    for n in (9, 12):
        pack = load_pack(n)
        packs[n] = pack
        seeds, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(seeds) == 0:
            # n12 pack may have empty train — fall back to val as train pool
            log(f"n{n} train empty — using val as train pool")
            seeds, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        if E is None or len(seeds) == 0:
            raise RuntimeError(f"no instances for n={n}")
        Es[n] = E
        cfg = AnsatzConfig(n=n, L=L, nf=nf)
        sims[n] = ProjectorAnsatz(cfg, W.W.to(device), device=device).to(device)
        log(f"train pool n{n}: B={len(E)}")

    opt = torch.optim.Adam([gamma, beta], lr=5e-3)
    best = {"score": float("inf"), "gamma": None, "beta": None, "val": None, "step": -1}
    adam_steps = int(steps * 0.75)

    for step in range(steps):
        if step == adam_steps:
            opt = torch.optim.LBFGS(
                [gamma, beta], lr=0.2, max_iter=4, history_size=20, line_search_fn="strong_wolfe"
            )
            log(f"switch LBFGS at step {step}")

        def closure():
            opt.zero_grad(set_to_none=True)
            losses = []
            for n, sim in sims.items():
                E = Es[n]
                bs = min(6, len(E))
                idx = rng.choice(len(E), size=bs, replace=False)
                for b in idx:
                    Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
                    out = sim.forward_proj(Eb, gamma, beta)
                    losses.append(_loss(out, leakpen=leakpen))
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        if isinstance(opt, torch.optim.LBFGS):
            loss = opt.step(closure)
        else:
            loss = closure()
            torch.nn.utils.clip_grad_norm_([gamma, beta], 5.0)
            opt.step()

        do_val = (step % 5 == 0) or (step + 1 == steps) or step < 3
        if not do_val:
            continue
        # promote on combined S: 0.4*n9 + 0.6*n12 (n12 matters more for transfer)
        vals = {}
        for n in (9, 12):
            vals[n] = eval_projector(
                n, W, gamma.detach(), beta.detach(), L=L, split="val",
                max_count=max_val, nf=nf, device=device,
            )
        sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        if sc < best["score"]:
            best = {
                "score": sc,
                "gamma": gamma.detach().cpu().clone(),
                "beta": beta.detach().cpu().clone(),
                "val": vals,
                "step": step,
                "train_loss": float(loss.detach()) if torch.is_tensor(loss) else float(loss),
            }
            log(
                f"step {step}: NEW best Scomb={sc:.4f} "
                + " ".join(
                    f"n{n}:Med={vals[n]['median_inv_p']:.2f}/B90={vals[n]['B90_inv_p']:.2f}"
                    for n in (9, 12)
                )
            )
        elif step % 10 == 0:
            log(f"step {step}: Scomb={sc:.4f} stagnant={step - best['step']}")

    return best


def write_report(results, out_dir: Path):
    lines = []
    lines.append("# Opus probe 2 — projector train {n9,n12}, zero-shot n15/n18\n\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_\n\n")
    lines.append("Fixed W from bake-off. No GNN. L=4.\n\n")
    lines.append("## Table (val Med / B90)\n\n")
    lines.append("| model | n9 | n12 | n15 | n18 | b_Med | b_B90 | Med15/12 | Med18/15 |\n")
    lines.append("|-------|----|-----|-----|-----|-------|-------|----------|----------|\n")

    def row(name, block):
        if not block:
            return
        lad = block.get("ladder", {})
        fits = block.get("fits", {})

        def cell(n):
            m = lad.get(n) or lad.get(str(n)) or {}
            if "median_inv_p" not in m:
                return "—"
            return f"{m['median_inv_p']:.3g}/{m['B90_inv_p']:.3g}"

        bM = fits.get("Med", {}).get("b", float("nan"))
        bB = fits.get("B90", {}).get("b", float("nan"))
        r = fits.get("ratios", {})
        r1512 = r.get("Med15_over_Med12")
        r1815 = r.get("Med18_over_Med15")
        s1512 = "—" if r1512 is None else f"{r1512:.2f}"
        s1815 = "—" if r1815 is None else f"{r1815:.2f}"
        lines.append(
            f"| {name} | {cell(9)} | {cell(12)} | {cell(15)} | {cell(18)} | "
            f"{bM:.3f} | {bB:.3f} | {s1512} | {s1815} |\n"
        )

    row("n9-only (bake-off)", results.get("n9_only"))
    row("train {9,12} (probe2)", results.get("train_912"))
    lines.append("\n## Verdict\n\n")
    lines.append(results.get("verdict", "—") + "\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    log(f"wrote {out_dir/'RESULTS.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--W", default=str(W_DEFAULT))
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=16)
    ap.add_argument("--max_val15", type=int, default=8)
    ap.add_argument("--max_val18", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t00 = time.time()

    W = CompiledW.load(args.W, device=device)
    log(f"W F={W.fidelity:.6f} leak={W.leak:.3e} from {args.W}")

    # bake-off n9-only baseline (reload angles if present, else from results.json)
    n9_only = None
    bake_pt = _PKG / "runs" / "driver_bakeoff" / "projector_params.pt"
    if bake_pt.exists():
        blob = torch.load(bake_pt, map_location="cpu", weights_only=False)
        g9 = blob["gamma"]
        b9 = blob["beta"]
        log(f"loaded bake-off projector angles from {bake_pt}")
    elif BAKE_JSON.exists():
        js = json.loads(BAKE_JSON.read_text())
        g9 = torch.as_tensor(js["projector_trained"]["gamma"])
        b9 = torch.as_tensor(js["projector_trained"]["beta"])
        log("loaded bake-off angles from results.json")
    else:
        raise SystemExit("no bake-off projector params")

    max_counts = {9: args.max_val, 12: args.max_val, 15: args.max_val15, 18: args.max_val18}
    ns_eval = (9, 12, 15, 18)

    log("=== re-eval n9-only (bake-off) on full ladder ===")
    lad9 = eval_ladder(W, g9, b9, args.L, args.nf, ns_eval, max_counts, device)
    n9_only = {"ladder": lad9, "fits": fit_ladder(lad9), "gamma": g9.tolist(), "beta": b9.tolist(),
               "source": "driver_bakeoff Phase B"}
    results = {
        "args": vars(args),
        "W": W.to_dict(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n9_only": n9_only,
    }
    write_report(results, out_dir)

    log("=== train projector on {n9,n12} ===")
    # warm-start from bake-off angles
    best = train_projector_912(
        W, L=args.L, steps=args.steps, max_train=args.max_train, max_val=args.max_val,
        nf=args.nf, seed=args.seed, device=device,
        warm_gamma=g9.tolist(), warm_beta=b9.tolist(),
    )
    torch.save(
        {"gamma": best["gamma"], "beta": best["beta"], "info": {
            "step": best["step"], "score": best["score"], "train_loss": best.get("train_loss"),
            "val_912": best.get("val"),
        }},
        out_dir / "projector_912_params.pt",
    )
    log("=== eval train{9,12} ladder ===")
    lad912 = eval_ladder(W, best["gamma"], best["beta"], args.L, args.nf, ns_eval, max_counts, device)
    results["train_912"] = {
        "ladder": lad912,
        "fits": fit_ladder(lad912),
        "gamma": best["gamma"].tolist(),
        "beta": best["beta"].tolist(),
        "best_step": best["step"],
        "best_Scomb": best["score"],
    }

    # verdict
    b9m = n9_only["fits"]["Med"]["b"]
    b12m = results["train_912"]["fits"]["Med"]["b"]
    r9 = n9_only["fits"]["ratios"]
    r12 = results["train_912"]["fits"]["ratios"]
    m15_9 = (lad9.get(15) or {}).get("median_inv_p")
    m15_12 = (lad912.get(15) or {}).get("median_inv_p")
    verdict = (
        f"n9-only b_Med={b9m:.3f}; train{{9,12}} b_Med={b12m:.3f}. "
        f"Med15: n9-only={m15_9} → train912={m15_12}. "
        f"Ratios train912 Med15/12={r12.get('Med15_over_Med12')}, Med18/15={r12.get('Med18_over_Med15')}. "
    )
    if b12m < 1.7 and (r12.get("Med15_over_Med12") or 99) < 5:
        verdict += (
            "Transfer looks healthy (b≪2, no Med15 cliff) → next: probe 3 Fourier schedule "
            "(cheap depth-at-deploy) OR probe 5 seq vocab with W atoms."
        )
        next_rec = "probe3_or_probe5"
    elif b12m < b9m and (m15_12 or 1e99) < (m15_9 or 1e99):
        verdict += "Train{9,12} helps vs n9-only but still soft → prefer probe 3 (Fourier L growth) before probe 5."
        next_rec = "probe3"
    else:
        verdict += "Little gain from n12 training → try probe 5 (compiled W atoms in seq-evolve) or Fourier."
        next_rec = "probe5"
    results["verdict"] = verdict
    results["next"] = next_rec
    results["wall_s"] = time.time() - t00
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_report(results, out_dir)
    log(verdict)
    log(f"DONE in {results['wall_s']:.0f}s → next={next_rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
