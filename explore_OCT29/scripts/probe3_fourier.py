#!/usr/bin/env python3
"""Opus probe 3 — FOURIER schedule on projector; train L=4, deploy L∈{4,6,8,12}.

γ_ℓ, β_ℓ from q Fourier modes (Zhou et al.). Train modes at L_train on {n9,n12};
evaluate same modes sampled at larger L with no retraining.
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

OUT_DEFAULT = _PKG / "runs" / "probe3_fourier"
W_DEFAULT = _PKG / "runs" / "driver_bakeoff" / "W.pt"
P2_PARAMS = _PKG / "runs" / "probe2_train912" / "projector_912_params.pt"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe3]", *a, flush=True)


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
    return obj


def fourier_schedule(coeffs: torch.Tensor, L: int) -> torch.Tensor:
    """Zhou-style: θ_ℓ = Σ_{k=0}^{q-1} c_k sin( (k+0.5)(ℓ+0.5) π / L ), ℓ=0..L-1.
    coeffs: (q,)
    """
    q = coeffs.shape[0]
    device = coeffs.device
    dtype = coeffs.dtype
    ell = torch.arange(L, device=device, dtype=dtype)
    out = torch.zeros(L, device=device, dtype=dtype)
    for k in range(q):
        out = out + coeffs[k] * torch.sin((k + 0.5) * (ell + 0.5) * math.pi / L)
    return out


def angles_from_modes(u_gamma: torch.Tensor, u_beta: torch.Tensor, L: int):
    return fourier_schedule(u_gamma, L), fourier_schedule(u_beta, L)


def _loss(out, leakpen=0.1):
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


def init_modes_from_angles(gamma: torch.Tensor, beta: torch.Tensor, q: int):
    """Least-squares fit of Fourier modes to existing L angles (warm start)."""
    L = gamma.numel()
    # build design matrix sin((k+0.5)(ell+0.5)pi/L)
    ell = np.arange(L, dtype=np.float64)
    A = np.stack([np.sin((k + 0.5) * (ell + 0.5) * np.pi / L) for k in range(q)], axis=1)
    ug, _, _, _ = np.linalg.lstsq(A, gamma.detach().cpu().numpy().astype(np.float64), rcond=None)
    ub, _, _, _ = np.linalg.lstsq(A, beta.detach().cpu().numpy().astype(np.float64), rcond=None)
    return (
        torch.nn.Parameter(torch.as_tensor(ug, dtype=torch.float32)),
        torch.nn.Parameter(torch.as_tensor(ub, dtype=torch.float32)),
    )


@torch.no_grad()
def eval_at_L(W, u_g, u_b, n, L, nf, max_count, device):
    g, b = angles_from_modes(u_g, u_b, L)
    return eval_projector(n, W, g, b, L=L, split="val", max_count=max_count, nf=nf, device=device)


def train_modes(
    W, q, L_train, steps, max_train, max_val, nf, seed, device, warm_g=None, warm_b=None
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

    sims, Es = {}, {}
    for n in (9, 12):
        pack = load_pack(n)
        seeds, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(seeds) == 0:
            seeds, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        Es[n] = E
        sims[n] = ProjectorAnsatz(
            AnsatzConfig(n=n, L=L_train, nf=nf), W.W.to(device), device=device
        ).to(device)
        log(f"train pool n{n}: B={len(E)}")

    opt = torch.optim.Adam([u_g, u_b], lr=5e-3)
    best = {"score": float("inf"), "u_g": None, "u_b": None, "step": -1}
    adam_cut = int(steps * 0.7)

    for step in range(steps):
        if step == adam_cut:
            opt = torch.optim.LBFGS(
                [u_g, u_b], lr=0.2, max_iter=4, history_size=20, line_search_fn="strong_wolfe"
            )
            log(f"LBFGS at {step}")

        def closure():
            opt.zero_grad(set_to_none=True)
            g, b = angles_from_modes(u_g, u_b, L_train)
            losses = []
            for n, sim in sims.items():
                # temporarily need sim.L == L_train (already)
                idx = rng.choice(len(Es[n]), size=min(6, len(Es[n])), replace=False)
                for i in idx:
                    Eb = torch.as_tensor(Es[n][i], dtype=torch.float32, device=device)
                    out = sim.forward_proj(Eb, g, b)
                    losses.append(_loss(out))
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        if isinstance(opt, torch.optim.LBFGS):
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()

        if not ((step % 5 == 0) or step < 3 or step + 1 == steps):
            continue
        vals = {}
        for n in (9, 12):
            vals[n] = eval_at_L(W, u_g.detach(), u_b.detach(), n, L_train, nf, max_val, device)
        sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        if sc < best["score"]:
            best = {
                "score": sc,
                "u_g": u_g.detach().cpu().clone(),
                "u_b": u_b.detach().cpu().clone(),
                "val": vals,
                "step": step,
            }
            log(
                f"step {step}: NEW Scomb={sc:.4f} "
                + " ".join(f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}" for n in (9, 12))
            )
    return best


def write_report(results, out_dir):
    lines = ["# Opus probe 3 — Fourier schedule on projector\n\n",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_\n\n",
             "Train Fourier modes at L=4 on {n9,n12}; deploy same modes at larger L, no retrain.\n\n",
             "## Med / B90 vs deploy L\n\n"]
    # table: rows = n, cols = L
    Ls = results.get("Ls", [])
    lines.append("| n \\ L | " + " | ".join(str(L) for L in Ls) + " |\n")
    lines.append("|-------|" + "|".join(["------"] * len(Ls)) + "|\n")
    sweep = results.get("sweep", {})
    for n in (9, 12, 15, 18):
        cells = []
        for L in Ls:
            m = (sweep.get(str(n)) or sweep.get(n) or {}).get(str(L)) or (sweep.get(str(n)) or {}).get(L) or {}
            # nested: sweep[n][L]
            if not m:
                # try sweep[n][L] after json int keys
                block = sweep.get(str(n), sweep.get(n, {}))
                m = block.get(str(L), block.get(L, {})) if isinstance(block, dict) else {}
            if "median_inv_p" in m:
                cells.append(f"{m['median_inv_p']:.3g}/{m['B90_inv_p']:.3g}")
            else:
                cells.append("—")
        lines.append(f"| {n} | " + " | ".join(cells) + " |\n")
    lines.append("\n## Fits b_Med over n at each L\n\n")
    lines.append("| L | b_Med | b_B90 |\n|---|-------|-------|\n")
    for L, f in (results.get("fits_by_L") or {}).items():
        lines.append(f"| {L} | {f['Med']['b']:.3f} | {f['B90']['b']:.3f} |\n")
    lines.append("\n## Verdict\n\n")
    lines.append(results.get("verdict", "—") + "\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    log(f"wrote {out_dir/'RESULTS.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--W", default=str(W_DEFAULT))
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--L_train", type=int, default=4)
    ap.add_argument("--Ls", type=str, default="4,6,8,12")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--max_val15", type=int, default=6)
    ap.add_argument("--max_val18", type=int, default=4)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    t00 = time.time()

    W = CompiledW.load(args.W, device=device)
    log(f"W F={W.fidelity:.6f}")

    warm_g = warm_b = None
    if P2_PARAMS.exists():
        blob = torch.load(P2_PARAMS, map_location="cpu", weights_only=False)
        warm_g, warm_b = blob["gamma"], blob["beta"]
        log(f"warm-start modes from probe2 angles")

    log(f"=== train q={args.q} modes at L={args.L_train} on {{9,12}} ===")
    best = train_modes(
        W, args.q, args.L_train, args.steps, args.max_train, args.max_val,
        args.nf, args.seed, device, warm_g=warm_g, warm_b=warm_b,
    )
    torch.save({"u_gamma": best["u_g"], "u_beta": best["u_b"], "q": args.q, "L_train": args.L_train,
                "step": best["step"], "score": best["score"]}, out_dir / "fourier_modes.pt")

    max_counts = {9: args.max_val, 12: args.max_val, 15: args.max_val15, 18: args.max_val18}
    sweep = {}
    fits_by_L = {}
    log(f"=== deploy sweep L={Ls} ===")
    for n in (9, 12, 15, 18):
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        sweep[n] = {}
        for L in Ls:
            # need ProjectorAnsatz with matching L — eval_projector builds from L
            m = eval_at_L(W, best["u_g"].to(device), best["u_b"].to(device), n, L, args.nf, max_counts[n], device)
            sweep[n][L] = m
            log(f"n{n} L{L}: Med={m['median_inv_p']:.4g} B90={m['B90_inv_p']:.4g} S={m['S']:.3f}")

    for L in Ls:
        ns, meds, b90s = [], [], []
        for n in (9, 12, 15, 18):
            m = (sweep.get(n) or {}).get(L)
            if m and "median_inv_p" in m:
                ns.append(n); meds.append(m["median_inv_p"]); b90s.append(m["B90_inv_p"])
        fits_by_L[L] = {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}

    # monotonicity check at n15
    meds15 = [sweep.get(15, {}).get(L, {}).get("median_inv_p") for L in Ls]
    mono = all(
        meds15[i] is not None and meds15[i + 1] is not None and meds15[i + 1] <= meds15[i] * 1.05
        for i in range(len(meds15) - 1)
    ) if all(x is not None for x in meds15) else False
    b4 = fits_by_L.get(4, {}).get("Med", {}).get("b", float("nan"))
    b12 = fits_by_L.get(12, {}).get("Med", {}).get("b", float("nan"))
    verdict = (
        f"Fourier q={args.q}: b_Med@L4={b4:.3f}, b_Med@L12={b12:.3f}. "
        f"n15 Med vs L: {meds15}. monotonic_nonincreasing≈{mono}. "
    )
    if mono and (b12 < b4 or (meds15[-1] or 1e9) < (meds15[0] or 0) * 0.85):
        verdict += "Depth-at-deploy works → keep Fourier; optional probe 5 for native novelty."
        nxt = "probe5_optional"
    elif mono:
        verdict += "Mild L benefit; try q=3 or INTERP warm-start; probe 5 still useful for mixer search."
        nxt = "probe5"
    else:
        verdict += "Not monotonic — Fourier alone insufficient; go probe 5 (W atoms in seq-evolve)."
        nxt = "probe5"

    results = {
        "args": vars(args), "Ls": Ls, "sweep": sweep, "fits_by_L": fits_by_L,
        "u_gamma": best["u_g"].tolist(), "u_beta": best["u_b"].tolist(),
        "best_step": best["step"], "best_Scomb": best["score"],
        "verdict": verdict, "next": nxt, "wall_s": time.time() - t00,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_report(results, out_dir)
    log(verdict)
    log(f"DONE in {results['wall_s']:.0f}s → next={nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
