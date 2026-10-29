#!/usr/bin/env python3
"""Opus probe 7 — multi-depth Fourier + α distill of fourier_proj_L16 champ.

Opus 2026-10-29 20:25 pick:
  1) LS-distill champ γ/β → q sine modes; eval no-retrain.
  2) Train q=6 + depth scale α: θ_ℓ(L)=(L_ref/L)^α · Fourier(c,L);
     sample L∈{8,16} on train{9,12}; deploy L∈{16,24,32}.
  3) Gate on val2 = FULL val packs (n15 B=40, n18 B=16 max available);
     seal test except milestone.

Orthogonal to probe6 (mixer) / stopped probe5 (SNAP rearrangements).
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
    fit_b,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.probe3_fourier import (  # noqa: E402
    fourier_schedule,
    init_modes_from_angles,
)

OUT = _PKG / "runs" / "probe7_fourier_alpha"
W_PATH = _PKG / "runs" / "driver_bakeoff" / "W.pt"
CHAMP_PARAMS = _PKG / "runs" / "probe5_seq_vocab" / "champ_params.pt"
STOP = _PKG / "runs" / "STOP_PROBE7"
L_REF = 16


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe7]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


def angles_scaled(u_g, u_b, L: int, alpha: torch.Tensor | float, L_ref: int = L_REF):
    """θ_ℓ(L) = (L_ref/L)^α · Fourier(u, L)."""
    scale = (float(L_ref) / float(L)) ** (
        alpha if not torch.is_tensor(alpha) else alpha.to(dtype=torch.float32)
    )
    if torch.is_tensor(scale):
        g = fourier_schedule(u_g, L) * scale
        b = fourier_schedule(u_b, L) * scale
    else:
        g = fourier_schedule(u_g, L) * float(scale)
        b = fourier_schedule(u_b, L) * float(scale)
    return g, b


def _loss_batch(inv_ps: torch.Tensor, leaks: torch.Tensor, leakpen=0.1, cvar_lam=0.3, cvar_q=0.9):
    """Mean log-inv_p + λ·CVaR_q(log-inv_p) + leak. Targets B90 tail."""
    z = torch.log(inv_ps.clamp(min=1.0))
    mean = z.mean()
    # CVaR ≈ mean of top (1-q) fraction
    k = max(1, int(math.ceil((1.0 - cvar_q) * z.numel())))
    topk = torch.topk(z, k).values.mean()
    leak = leaks.mean()
    return mean + cvar_lam * topk + leakpen * leak


@torch.no_grad()
def eval_full(W, u_g, u_b, alpha, n, L, nf, split, max_count, device, L_ref=L_REF):
    g, b = angles_scaled(u_g, u_b, L, alpha, L_ref=L_ref)
    pack = load_pack(n)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = ProjectorAnsatz(cfg, W.W.to(device), device=device).to(device)
    invs, leaks = [], []
    t0 = time.time()
    for i in range(len(seeds)):
        Eb = torch.as_tensor(E[i], dtype=torch.float32, device=device)
        out = sim.forward_proj(Eb, g, b)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n,
        "L": L,
        "split": split,
        "B": int(len(invs)),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12))),
        "mean_leak": float(np.mean(leaks)),
        "L_times_Med": float(L * med),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "frac_inv_le_4": float(np.mean(invs_a <= 4.0)),
        "wall_s": time.time() - t0,
        "alpha": float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha),
    }


def distill_qs(gamma, beta, qs):
    """LS fit q modes; return reconstruction MSE and coeffs."""
    out = {}
    g_t = torch.as_tensor(gamma, dtype=torch.float32).reshape(-1)
    b_t = torch.as_tensor(beta, dtype=torch.float32).reshape(-1)
    L = int(g_t.numel())
    for q in qs:
        u_g, u_b = init_modes_from_angles(g_t, b_t, q)
        g_hat = fourier_schedule(u_g.detach(), L)
        b_hat = fourier_schedule(u_b.detach(), L)
        mse_g = float(torch.mean((g_hat - g_t) ** 2))
        mse_b = float(torch.mean((b_hat - b_t) ** 2))
        out[q] = {
            "u_gamma": u_g.detach().cpu(),
            "u_beta": u_b.detach().cpu(),
            "mse_gamma": mse_g,
            "mse_beta": mse_b,
            "mse": mse_g + mse_b,
        }
        log(f"distill q={q}: mse_g={mse_g:.4g} mse_b={mse_b:.4g}")
    return out


def train_multi_L(
    W,
    q,
    steps,
    max_train,
    max_val,
    nf,
    seed,
    device,
    warm_g=None,
    warm_b=None,
    L_train_set=(8, 16),
    L_ref=L_REF,
    cvar_lam=0.3,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if warm_g is not None:
        u_g, u_b = init_modes_from_angles(
            torch.as_tensor(warm_g, dtype=torch.float32),
            torch.as_tensor(warm_b, dtype=torch.float32),
            q,
        )
        u_g = torch.nn.Parameter(u_g.to(device))
        u_b = torch.nn.Parameter(u_b.to(device))
    else:
        u_g = torch.nn.Parameter(torch.randn(q, device=device) * 0.5)
        u_b = torch.nn.Parameter(torch.randn(q, device=device) * 0.5)
    # α=0 → classic Fourier; learn whether angles should shrink with L
    alpha = torch.nn.Parameter(torch.zeros((), device=device))

    Es = {}
    sims = {}  # (n, L) -> sim
    for n in (9, 12):
        pack = load_pack(n)
        seeds, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(seeds) == 0:
            seeds, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        Es[n] = E
        for L in L_train_set:
            sims[(n, L)] = ProjectorAnsatz(
                AnsatzConfig(n=n, L=L, nf=nf), W.W.to(device), device=device
            ).to(device)
        log(f"train pool n{n}: B={len(E)}")

    params = [u_g, u_b, alpha]
    opt = torch.optim.Adam(params, lr=5e-3)
    best = {"score": float("inf"), "u_g": None, "u_b": None, "alpha": 0.0, "step": -1}
    adam_cut = int(steps * 0.7)

    for step in range(steps):
        if STOP.exists():
            log("STOP_PROBE7 — exit train early")
            break
        if step == adam_cut:
            opt = torch.optim.LBFGS(
                params, lr=0.15, max_iter=4, history_size=20, line_search_fn="strong_wolfe"
            )
            log(f"LBFGS at {step}")

        def closure():
            opt.zero_grad(set_to_none=True)
            L = int(rng.choice(list(L_train_set)))
            g, b = angles_scaled(u_g, u_b, L, alpha, L_ref=L_ref)
            invs, leaks = [], []
            for n in (9, 12):
                sim = sims[(n, L)]
                idx = rng.choice(len(Es[n]), size=min(6, len(Es[n])), replace=False)
                for i in idx:
                    Eb = torch.as_tensor(Es[n][i], dtype=torch.float32, device=device)
                    out = sim.forward_proj(Eb, g, b)
                    invs.append(out["inv_p"])
                    leaks.append(out["p_leak"])
            loss = _loss_batch(torch.stack(invs), torch.stack(leaks), cvar_lam=cvar_lam)
            loss.backward()
            return loss

        if isinstance(opt, torch.optim.LBFGS):
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()

        if not ((step % 5 == 0) or step < 3 or step + 1 == steps):
            continue
        # val monitor at L_ref on n9/12 (small B for speed)
        vals = {}
        for n in (9, 12):
            vals[n] = eval_full(
                W, u_g.detach(), u_b.detach(), alpha.detach(),
                n, L_ref, nf, "val", max_val, device, L_ref=L_ref,
            )
        sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        if sc < best["score"]:
            best = {
                "score": sc,
                "u_g": u_g.detach().cpu().clone(),
                "u_b": u_b.detach().cpu().clone(),
                "alpha": float(alpha.detach().cpu()),
                "val": vals,
                "step": step,
            }
            log(
                f"step {step}: NEW Scomb={sc:.4f} α={best['alpha']:.4f} "
                + " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in (9, 12)
                )
            )
        else:
            log(
                f"step {step}: Scomb={sc:.4f} α={float(alpha.detach().cpu()):.4f} "
                f"(best={best['score']:.4f}@{best['step']})"
            )
    return best


def ladder(W, u_g, u_b, alpha, Ls, ns, counts, nf, device, split="val"):
    out = {}
    for n in ns:
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        out[n] = {}
        for L in Ls:
            m = eval_full(
                W, u_g, u_b, alpha, n, L, nf, split, counts.get(n, 16), device
            )
            if m is None:
                continue
            out[n][L] = m
            log(
                f"{split} n{n} L{L}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} "
                f"L×Med={m['L_times_Med']:.1f} frac≤2={m['frac_inv_le_2']:.2f} "
                f"leak={m['mean_leak']:.4f} B={m['B']}"
            )
    return out


def fits_by_L(sweep_d, Ls, ns=(12, 15, 18)):
    fb = {}
    for L in Ls:
        n_list, meds, b90s = [], [], []
        for n in ns:
            m = (sweep_d.get(n) or {}).get(L)
            if m and "median_inv_p" in m:
                n_list.append(n)
                meds.append(m["median_inv_p"])
                b90s.append(m["B90_inv_p"])
        fb[L] = {"Med": fit_b(n_list, meds), "B90": fit_b(n_list, b90s), "ns": n_list}
    return fb


def write_md(results, out_dir):
    lines = [
        "# Opus probe 7 — Fourier + α multi-depth distill",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Gate = **val2** (full val packs: n15 B=40, n18 B=16). Test sealed except milestone.",
        "Retired yardstick: probe5 val b≈1.08 (selection bias).",
        "",
        "## Phase 1 — LS distill (no retrain) at L=16",
        "",
        "| q | mse | val2 n15 Med/B90 | val2 n18 Med/B90 | L×Med15 |",
        "|---|-----|------------------|------------------|---------|",
    ]
    d = results.get("distill") or {}
    for q, block in sorted(d.items(), key=lambda kv: int(kv[0])):
        v15 = ((block.get("val2") or {}).get("15") or {}).get(16) or {}
        v18 = ((block.get("val2") or {}).get("18") or {}).get(16) or {}
        lines.append(
            f"| {q} | {block.get('mse', float('nan')):.4g} | "
            f"{v15.get('median_inv_p', float('nan')):.3g}/"
            f"{v15.get('B90_inv_p', float('nan')):.3g} | "
            f"{v18.get('median_inv_p', float('nan')):.3g}/"
            f"{v18.get('B90_inv_p', float('nan')):.3g} | "
            f"{v15.get('L_times_Med', float('nan')):.1f} |"
        )
    lines += ["", "## Phase 2 — multi-L train q + α", ""]
    tr = results.get("train") or {}
    lines.append(
        f"- q={tr.get('q')} α*={tr.get('alpha')} best_step={tr.get('step')} "
        f"Scomb={tr.get('score')}"
    )
    lines += ["", "### val2 deploy sweep", ""]
    sw = results.get("val2_sweep") or {}
    Ls = results.get("deploy_Ls") or []
    lines.append("| n \\ L | " + " | ".join(str(L) for L in Ls) + " |")
    lines.append("|-------|" + "|".join(["------"] * len(Ls)) + "|")
    for n in (9, 12, 15, 18):
        cells = []
        for L in Ls:
            m = (sw.get(str(n)) or sw.get(n) or {}).get(str(L)) or (sw.get(n) or {}).get(L) or {}
            if "median_inv_p" in m:
                cells.append(f"{m['median_inv_p']:.3g}/{m['B90_inv_p']:.3g}")
            else:
                cells.append("—")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "### b_Med over n∈{12,15,18} by L (val2)", ""]
    lines.append("| L | b_Med | b_B90 |")
    lines.append("|---|-------|-------|")
    for L, f in (results.get("val2_fits") or {}).items():
        bm = (f.get("Med") or {}).get("b", float("nan"))
        bb = (f.get("B90") or {}).get("b", float("nan"))
        lines.append(f"| {L} | {bm:.3f} | {bb:.3f} |")
    if results.get("test_milestone"):
        lines += ["", "## Test milestone (sealed; look once)", ""]
        tm = results["test_milestone"]
        for n, m in tm.items():
            if isinstance(m, dict) and "median_inv_p" in m:
                lines.append(
                    f"- test n{n} L{m.get('L')}: Med={m['median_inv_p']:.3f} "
                    f"B90={m['B90_inv_p']:.3f} L×Med={m['L_times_Med']:.1f} "
                    f"frac≤2={m['frac_inv_le_2']:.2f}"
                )
    lines.append("")
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--W", default=str(W_PATH))
    ap.add_argument("--qs", default="3,4,6,8")
    ap.add_argument("--q_train", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cvar_lam", type=float, default=0.3)
    ap.add_argument("--L_ref", type=int, default=L_REF)
    ap.add_argument("--deploy_Ls", default="16,24,32")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_test", action="store_true",
                    help="Do not touch sealed test (default: one milestone at best L)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    qs = [int(x) for x in args.qs.split(",") if x.strip()]
    deploy_Ls = [int(x) for x in args.deploy_Ls.split(",") if x.strip()]
    # val2 = full available val
    val2_counts = {9: 40, 12: 50, 15: 40, 18: 16}
    test_counts = {9: 40, 12: 50, 15: 40, 18: 16}

    W = CompiledW.load(args.W, device=device)
    log(f"W F={W.fidelity:.6f}")

    champ = torch.load(CHAMP_PARAMS, map_location="cpu", weights_only=False)
    gamma_c = champ["gamma"].reshape(-1).to(dtype=torch.float32)
    beta_c = champ["beta_proj"].reshape(-1).to(dtype=torch.float32)
    log(f"champ L={gamma_c.numel()} γ/β loaded")

    results = {
        "args": vars(args),
        "gate": "val2_full_val",
        "val2_counts": val2_counts,
        "note": (
            "val2 = full val packs (n15 B=40, n18 B=16 max in pack). "
            "Cannot hit B≥40 at n18 without regenerating. Test sealed."
        ),
        "deploy_Ls": deploy_Ls,
        "champ_ArmA_test_ref": {
            "n15": {"Med": 6.325, "B90": 24.024, "L_times_Med": 101.2, "B": 40},
            "n18": {"Med": 6.410, "B90": 27.512, "L_times_Med": 102.6, "B": 16},
        },
    }

    # ---- Phase 1: distill ----
    log("=== Phase 1: LS distill champ → q modes ===")
    dist = distill_qs(gamma_c, beta_c, qs)
    dist_out = {}
    for q, block in dist.items():
        ug = block["u_gamma"].to(device)
        ub = block["u_beta"].to(device)
        # α=0 for pure distill deploy
        val2 = ladder(
            W, ug, ub, 0.0, [args.L_ref], (9, 12, 15, 18),
            val2_counts, args.nf, device, split="val",
        )
        torch.save(
            {"u_gamma": block["u_gamma"], "u_beta": block["u_beta"], "q": q, "alpha": 0.0,
             "mse": block["mse"]},
            out_dir / f"distill_q{q}.pt",
        )
        dist_out[q] = {
            "mse": block["mse"],
            "mse_gamma": block["mse_gamma"],
            "mse_beta": block["mse_beta"],
            "u_gamma": block["u_gamma"].tolist(),
            "u_beta": block["u_beta"].tolist(),
            "val2": val2,
        }
    results["distill"] = dist_out
    # pick best distill by val2 n15 Med at L_ref
    def _med15(q):
        m = ((dist_out[q].get("val2") or {}).get(15) or {}).get(args.L_ref) or {}
        return m.get("median_inv_p", 1e99)

    best_q_distill = min(qs, key=_med15)
    log(f"best distill by val2 n15 Med: q={best_q_distill} Med={_med15(best_q_distill):.3f}")
    results["best_distill_q"] = best_q_distill

    if args.skip_train:
        write_md(results, out_dir)
        (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
        log("skip_train — DONE")
        return 0

    # ---- Phase 2: multi-L + α ----
    q_tr = args.q_train
    warm = dist[q_tr]
    log(f"=== Phase 2: multi-L train q={q_tr} + α (warm distill) ===")
    best = train_multi_L(
        W, q_tr, args.steps, args.max_train, args.max_val, args.nf, args.seed, device,
        warm_g=warm["u_gamma"], warm_b=warm["u_beta"],
        L_train_set=(8, 16), L_ref=args.L_ref, cvar_lam=args.cvar_lam,
    )
    torch.save(
        {
            "u_gamma": best["u_g"], "u_beta": best["u_b"], "alpha": best["alpha"],
            "q": q_tr, "L_ref": args.L_ref, "step": best["step"], "score": best["score"],
        },
        out_dir / f"modes_q{q_tr}_alpha.pt",
    )
    results["train"] = {
        "q": q_tr,
        "alpha": best["alpha"],
        "step": best["step"],
        "score": best["score"],
        "u_gamma": best["u_g"].tolist() if best["u_g"] is not None else None,
        "u_beta": best["u_b"].tolist() if best["u_b"] is not None else None,
    }

    ug = best["u_g"].to(device)
    ub = best["u_b"].to(device)
    alpha = best["alpha"]
    log(f"=== val2 deploy sweep L={deploy_Ls} α={alpha:.4f} ===")
    sw = ladder(
        W, ug, ub, alpha, deploy_Ls, (9, 12, 15, 18),
        val2_counts, args.nf, device, split="val",
    )
    results["val2_sweep"] = sw
    results["val2_fits"] = fits_by_L(sw, deploy_Ls)

    # pick best deploy L by val2 n15 Med
    best_L = None
    best_med = 1e99
    for L in deploy_Ls:
        m = (sw.get(15) or {}).get(L)
        if m and m["median_inv_p"] < best_med:
            best_med = m["median_inv_p"]
            best_L = L
    results["best_deploy_L_val2"] = best_L
    log(f"best deploy L on val2 n15: L={best_L} Med={best_med:.3f}")

    if not args.skip_test and best_L is not None:
        log(f"=== TEST milestone at L={best_L} (sealed look-once) ===")
        tm = {}
        for n in (15, 18):
            m = eval_full(
                W, ug, ub, alpha, n, best_L, args.nf, "test",
                test_counts[n], device, L_ref=args.L_ref,
            )
            tm[n] = m
            if m:
                log(
                    f"TEST n{n} L{best_L}: Med={m['median_inv_p']:.3f} "
                    f"B90={m['B90_inv_p']:.3f} L×Med={m['L_times_Med']:.1f}"
                )
        results["test_milestone"] = tm

    write_md(results, out_dir)
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    log("DONE →", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
