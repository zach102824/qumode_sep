#!/usr/bin/env python3
"""One fixed γ (shared or per-instance) + deeper projector circuit.

Circuit: |0⟩→W→[U_Φ(γ) → W·SNAP₀(β_ℓ)·W†]^L
  - γ is a scalar reused every layer (NOT γ_ℓ schedule)
  - β_ℓ from Fourier q=3 modes (resampled at deploy L)

Variants
  A) shared_gamma: one γ for all instances (fixed-angle); train γ + β-modes on {9,12}
  B) per_instance_gamma: freeze β schedule; L-BFGS/Adam on scalar γ per val instance

Leave probe5 running. Results → runs/fixed_gamma_deep/
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
from explore_OCT29.lib.drivers import CompiledW, ProjectorAnsatz, eval_projector, fit_b  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.probe3_fourier import angles_from_modes, fourier_schedule  # noqa: E402

OUT = _PKG / "runs" / "fixed_gamma_deep"
W_PATH = _PKG / "runs" / "driver_bakeoff" / "W.pt"
Q3_MODES = _PKG / "runs" / "probe3_fourier_deep" / "modes_q3_Ltrain6.pt"
CHAMP = _PKG / "runs" / "probe5_seq_vocab" / "champ.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fixγ]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return o


def tile_gamma(g: torch.Tensor, L: int) -> torch.Tensor:
    g = g.reshape(()).to(dtype=torch.float32)
    return g.expand(L).clone()


def betas_from_modes(u_b: torch.Tensor, L: int) -> torch.Tensor:
    return fourier_schedule(u_b, L)


@torch.no_grad()
def eval_fixed_gamma(n, W, gamma_scalar, u_beta, L, nf, max_count, device, split="val"):
    """Shared γ tiled across layers; β from Fourier modes at L."""
    g = tile_gamma(torch.as_tensor(gamma_scalar, dtype=torch.float32, device=device), L)
    b = betas_from_modes(u_beta.to(device), L)
    return eval_projector(n, W, g, b, L=L, split=split, max_count=max_count, nf=nf, device=device)


def train_shared_gamma(
    W, u_beta_init, L_train, steps, max_train, max_val, nf, seed, device, gamma0=1.0
):
    """Adam on scalar γ + q β-modes; loss on {9,12}. Circuit uses tiled γ."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(torch.tensor(float(gamma0), dtype=torch.float32, device=device))
    u_b = torch.nn.Parameter(u_beta_init.detach().clone().to(device=device, dtype=torch.float32))
    sims, Es = {}, {}
    for n in (9, 12):
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        Es[n] = E
        sims[n] = ProjectorAnsatz(
            AnsatzConfig(n=n, L=L_train, nf=nf), W.W.to(device), device=device
        ).to(device)
        log(f"shared train pool n{n}: B={len(E)}")

    opt = torch.optim.Adam([gamma, u_b], lr=5e-3)
    best = {"score": float("inf"), "gamma": None, "u_b": None, "step": -1}
    cut = int(steps * 0.7)

    def loss_fn():
        g = tile_gamma(gamma, L_train)
        b = betas_from_modes(u_b, L_train)
        losses = []
        for n, sim in sims.items():
            idx = rng.choice(len(Es[n]), size=min(6, len(Es[n])), replace=False)
            for i in idx:
                Eb = torch.as_tensor(Es[n][i], dtype=torch.float32, device=device)
                out = sim.forward_proj(Eb, g, b)
                losses.append(torch.log(out["inv_p"].clamp(min=1.0)) + 0.1 * out["p_leak"])
        return torch.stack(losses).mean()

    for step in range(steps):
        if step == cut:
            opt = torch.optim.LBFGS(
                [gamma, u_b], lr=0.2, max_iter=4, history_size=20, line_search_fn="strong_wolfe"
            )
            log(f"shared LBFGS at {step}")

        if isinstance(opt, torch.optim.LBFGS):
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()

        if not ((step % 5 == 0) or step < 2 or step + 1 == steps):
            continue
        vals = {}
        for n in (9, 12):
            vals[n] = eval_fixed_gamma(
                n, W, float(gamma.detach()), u_b.detach(), L_train, nf, max_val, device
            )
        sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        if sc < best["score"]:
            best = {
                "score": sc,
                "gamma": float(gamma.detach().cpu()),
                "u_b": u_b.detach().cpu().clone(),
                "val": vals,
                "step": step,
            }
            log(
                f"shared step {step}: Scomb={sc:.3f} γ={best['gamma']:.3f} "
                + " ".join(f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}" for n in (9, 12))
            )
    return best


def opt_gamma_one_instance(sim, E, beta, gamma0, steps=25, lr=0.15):
    """Classical outer: optimize scalar γ for one instance; β frozen."""
    gamma = torch.nn.Parameter(torch.tensor(float(gamma0), dtype=torch.float32, device=E.device))
    opt = torch.optim.LBFGS([gamma], lr=lr, max_iter=4, history_size=10, line_search_fn="strong_wolfe")
    L = beta.shape[0]
    Eb = E

    def closure():
        opt.zero_grad(set_to_none=True)
        g = tile_gamma(gamma, L)
        out = sim.forward_proj(Eb, g, beta)
        loss = torch.log(out["inv_p"].clamp(min=1.0)) + 0.05 * out["p_leak"]
        loss.backward()
        return loss

    try:
        for _ in range(max(1, steps // 4)):
            opt.step(closure)
    except Exception:
        # Adam fallback
        opt_a = torch.optim.Adam([gamma], lr=3e-2)
        for _ in range(steps):
            opt_a.zero_grad(set_to_none=True)
            g = tile_gamma(gamma, L)
            out = sim.forward_proj(Eb, g, beta.detach())
            loss = torch.log(out["inv_p"].clamp(min=1.0)) + 0.05 * out["p_leak"]
            loss.backward()
            opt_a.step()
    with torch.no_grad():
        g = tile_gamma(gamma, L)
        out = sim.forward_proj(Eb, g, beta)
    return float(gamma.detach().cpu()), float(out["inv_p"].cpu()), float(out["p_leak"].cpu())


def eval_per_instance_gamma(n, W, u_beta, L, nf, max_count, device, gamma0=1.0, opt_steps=20):
    pack = load_pack(n)
    seeds, _, _, _, E = _split_arrays(pack, "val", max_count=max_count)
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = ProjectorAnsatz(cfg, W.W.to(device), device=device).to(device)
    beta = betas_from_modes(u_beta.to(device), L).detach()
    invs, leaks, gammas = [], [], []
    t0 = time.time()
    for i in range(len(seeds)):
        Eb = torch.as_tensor(E[i], dtype=torch.float32, device=device)
        g_star, inv, leak = opt_gamma_one_instance(sim, Eb, beta, gamma0, steps=opt_steps)
        invs.append(inv)
        leaks.append(leak)
        gammas.append(g_star)
    invs_a = np.asarray(invs)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n, "L": L, "B": len(invs),
        "median_inv_p": med, "B90_inv_p": b90, "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "gamma_median": float(np.median(gammas)),
        "gamma_mean": float(np.mean(gammas)),
        "gamma_std": float(np.std(gammas)),
        "wall_s": time.time() - t0,
        "variant": "per_instance_gamma",
    }


def sweep_shared(W, gamma, u_b, Ls, max_counts, nf, device):
    out = {}
    for n in (9, 12, 15, 18):
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        out[n] = {}
        for L in Ls:
            m = eval_fixed_gamma(n, W, gamma, u_b, L, nf, max_counts.get(n, 4), device)
            out[n][L] = m
            log(f"shared n{n} L{L}: Med={m['median_inv_p']:.4g} B90={m['B90_inv_p']:.4g} S={m['S']:.3f}")
    return out


def sweep_per_inst(W, u_b, Ls, max_counts, nf, device, gamma0, opt_steps):
    out = {}
    for n in (9, 12, 15, 18):
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        out[n] = {}
        for L in Ls:
            # skip very expensive combos if needed
            m = eval_per_instance_gamma(
                n, W, u_b, L, nf, max_counts.get(n, 4), device, gamma0=gamma0, opt_steps=opt_steps
            )
            out[n][L] = m
            log(
                f"per-inst n{n} L{L}: Med={m['median_inv_p']:.4g} B90={m['B90_inv_p']:.4g} "
                f"γ~{m['gamma_median']:.3f}±{m['gamma_std']:.3f} ({m['wall_s']:.1f}s)"
            )
    return out


def fits(sw, Ls):
    fb = {}
    for L in Ls:
        ns, meds, b90s = [], [], []
        for n in (9, 12, 15, 18):
            m = (sw.get(n) or {}).get(L)
            if m and "median_inv_p" in m:
                ns.append(n); meds.append(m["median_inv_p"]); b90s.append(m["B90_inv_p"])
        fb[L] = {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}
    return fb


def write_report(results, out_dir):
    lines = [
        "# Fixed γ (one per instance / shared) + deeper projector\n\n",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_\n\n",
        "Circuit: `|0⟩→W→[U_Φ(γ) → W·SNAP₀(β_ℓ)·W†]^L` with **same γ every layer**.\n\n",
        "## Clarification\n\n",
        "- **shared_gamma**: one γ for all instances (fixed-angle); β_ℓ from trained Fourier modes.\n",
        "- **per_instance_gamma**: β schedule frozen; scalar γ optimized per val instance (classical outer). "
        "Stronger than transferable fixed-angle — not a size-independent deployed policy unless γ is later predicted.\n\n",
    ]
    champ = results.get("champ_ref", {})
    lines.append(
        f"## Refs\n\n- Probe5 champ fourier_proj_L16: n15≈{champ.get('n15_med', '?')}, b≈{champ.get('b', '?')}\n"
        f"- Fourier q3 schedule L20: n15≈5.99, b≈1.24\n\n"
    )
    for key, title in (
        ("shared", "Shared γ (fixed-angle)"),
        ("per_instance", "Per-instance γ (opt at eval)"),
    ):
        block = results.get(key)
        if not block:
            continue
        lines.append(f"## {title}\n\n")
        if "gamma" in block:
            lines.append(f"Trained/init γ={block['gamma']:.4g}; L_train={block.get('L_train')}\n\n")
        Ls = block["Ls"]
        sw = block["sweep"]
        lines.append("| n \\ L | " + " | ".join(str(L) for L in Ls) + " |\n")
        lines.append("|-------|" + "|".join(["------"] * len(Ls)) + "|\n")
        for n in (9, 12, 15, 18):
            cells = []
            for L in Ls:
                m = (sw.get(n) or {}).get(L) or {}
                if "median_inv_p" in m:
                    cells.append(f"{m['median_inv_p']:.3g}/{m['B90_inv_p']:.3g}")
                else:
                    cells.append("—")
            lines.append(f"| {n} | " + " | ".join(cells) + " |\n")
        lines.append("\n| L | b_Med | b_B90 |\n|---|-------|-------|\n")
        for L, f in block.get("fits_by_L", {}).items():
            lines.append(f"| {L} | {f['Med']['b']:.3f} | {f['B90']['b']:.3f} |\n")
        lines.append("\n")
    lines.append("## Verdict\n\n")
    lines.append(results.get("verdict", "—") + "\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    log(f"wrote {out_dir/'RESULTS.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="4,8,12,16,20,24")
    ap.add_argument("--L_train", type=int, default=6)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--max_val15", type=int, default=6)
    ap.add_argument("--max_val18", type=int, default=4)
    ap.add_argument("--opt_steps", type=int, default=20, help="per-instance γ opt steps")
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_per_instance", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    t00 = time.time()

    W = CompiledW.load(W_PATH, device=device)
    blob = torch.load(Q3_MODES, map_location="cpu", weights_only=False)
    u_b0 = blob["u_beta"].float()
    # init γ from mean of Fourier γ schedule at L_train
    u_g0 = blob["u_gamma"].float()
    g_sched = fourier_schedule(u_g0, args.L_train)
    gamma0 = float(g_sched.abs().mean())  # positive-ish scale
    log(f"init γ0={gamma0:.3f} from |Fourier γ| mean; u_β={u_b0.tolist()}")

    champ_ref = {"n15_med": 2.51, "b": 1.08}
    if CHAMP.exists():
        c = json.loads(CHAMP.read_text())
        champ_ref = {
            "n15_med": c["ladder"]["15"]["median_inv_p"],
            "b": c["b_med"],
            "motif": c["motif"]["name"],
        }

    max_counts = {9: args.max_val, 12: args.max_val, 15: args.max_val15, 18: args.max_val18}

    # --- A) shared γ ---
    log(f"=== A) train shared γ + β-modes at L_train={args.L_train} ===")
    best = train_shared_gamma(
        W, u_b0, args.L_train, args.steps, args.max_train, args.max_val,
        args.nf, args.seed, device, gamma0=gamma0,
    )
    torch.save({"gamma": best["gamma"], "u_beta": best["u_b"], "L_train": args.L_train},
               OUT / "shared_gamma_params.pt")
    sw_s = sweep_shared(W, best["gamma"], best["u_b"], Ls, max_counts, args.nf, device)
    shared_block = {
        "gamma": best["gamma"], "L_train": args.L_train, "Ls": Ls,
        "u_beta": best["u_b"].tolist(),
        "sweep": sw_s, "fits_by_L": fits(sw_s, Ls),
        "best_Scomb": best["score"], "best_step": best["step"],
    }

    results = {"champ_ref": champ_ref, "shared": shared_block, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_report(results, OUT)

    # --- B) per-instance γ ---
    if not args.skip_per_instance:
        log("=== B) per-instance γ opt at eval (β from shared train) ===")
        # use fewer L for expensive per-inst on n18
        Ls_pi = [L for L in Ls if L <= 20]
        sw_p = sweep_per_inst(
            W, best["u_b"], Ls_pi, max_counts, args.nf, device,
            gamma0=best["gamma"], opt_steps=args.opt_steps,
        )
        results["per_instance"] = {
            "gamma0": best["gamma"], "Ls": Ls_pi,
            "u_beta": best["u_b"].tolist(),
            "sweep": sw_p, "fits_by_L": fits(sw_p, Ls_pi),
            "note": "γ optimized per val instance; β frozen from shared train — not a transferable deployed policy",
        }
        write_report(results, OUT)

    # verdict
    def med15(block, L):
        if not block:
            return None
        m = (block["sweep"].get(15) or {}).get(L) or {}
        return m.get("median_inv_p")

    m15_s12 = med15(shared_block, 12)
    m15_s20 = med15(shared_block, 20) or med15(shared_block, 16)
    m15_p12 = med15(results.get("per_instance"), 12)
    m15_p20 = med15(results.get("per_instance"), 20) or med15(results.get("per_instance"), 16)
    b_s = (shared_block["fits_by_L"].get(20) or shared_block["fits_by_L"].get(16) or {}).get("Med", {}).get("b")
    b_p = None
    if results.get("per_instance"):
        b_p = (results["per_instance"]["fits_by_L"].get(20) or results["per_instance"]["fits_by_L"].get(16) or {}).get("Med", {}).get("b")

    champ_m = champ_ref["n15_med"]
    verdict = (
        f"Shared γ: n15@L12={m15_s12}, n15@deep={m15_s20}, b_deep={b_s}. "
        f"Per-instance γ: n15@L12={m15_p12}, n15@deep={m15_p20}, b_deep={b_p}. "
        f"Champ free L16 n15={champ_m:.3g}. "
    )
    # Does depth help with single γ?
    series_s = [med15(shared_block, L) for L in Ls if med15(shared_block, L) is not None]
    mono_s = all(series_s[i + 1] <= series_s[i] * 1.08 for i in range(len(series_s) - 1)) if len(series_s) > 1 else False
    if m15_s20 and m15_s20 < champ_m * 1.3 and mono_s:
        verdict += (
            "Shared one-γ + depth scales reasonably (near champ) — γ_ℓ schedule not strictly required for OK scaling, "
            "but free γ_ℓ / Fourier still better at matched L."
        )
        nxt = "one_gamma_viable"
    elif m15_p20 and m15_p20 < champ_m * 1.15:
        verdict += (
            "Per-instance γ + depth can match/beat champ, but that γ is fit per formula at eval "
            "(not transferable). Prefer predicting γ from cheap features or keep γ_ℓ/Fourier for deployment."
        )
        nxt = "per_inst_strong_need_predictor"
    elif mono_s and m15_s20 and m15_s20 < 15:
        verdict += (
            "One shared γ + depth improves with L but lags free γ_ℓ / Fourier champ — "
            "γ_ℓ (or Fourier) schedule still worth keeping for best scaling."
        )
        nxt = "prefer_gamma_schedule"
    else:
        verdict += (
            "One fixed γ does not scale like γ_ℓ schedule / free-angle champ — keep layerwise or Fourier γ."
        )
        nxt = "need_gamma_schedule"

    results["verdict"] = verdict
    results["next"] = nxt
    results["wall_s"] = time.time() - t00
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_report(results, OUT)
    (OUT / "NEXT").write_text(nxt + "\n")
    log(verdict)
    log(f"DONE in {results['wall_s']:.0f}s → {nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
