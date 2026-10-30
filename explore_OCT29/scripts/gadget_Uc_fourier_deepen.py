#!/usr/bin/env python3
"""Fourier-γ deepen on projector-gadget (unsat) BinH Arm-C shell mixer.

Soft-promoted family only (no parity/stencil).
- Fit q Fourier modes to unsat γ; free shell φ_(pc) (L,4) or Fourier per shell.
- Train modes+phi at L_train on {9,12}; deploy L∈ deploy_Ls.
- Keep only if scale/val n15/n18 beats projector baseline & Arm C.

Writes runs/gadget_Uc/fourier_deepen/.
"""
from __future__ import annotations

import argparse
import json
import math
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
from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    GadgetAnsatz,
    _S,
    _jsonable,
    _loss,
    build_W_BinH,
    eval_arm,
    make_sim,
    scale1000_eval,
)
from explore_OCT29.scripts.probe3_fourier import fourier_schedule  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "fourier_deepen"
UNSAT_PARAMS = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
BASELINE_SCALE = {
    9: {"Med": 1.445, "B90": 3.222},
    12: {"Med": 1.926, "B90": 6.148},
    15: {"Med": 2.820, "B90": 10.063},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fourier_proj]", *a, flush=True)


def init_modes_from_gamma(gamma: torch.Tensor, q: int) -> torch.Tensor:
    L = int(gamma.numel())
    ell = np.arange(L, dtype=np.float64)
    A = np.stack([np.sin((k + 0.5) * (ell + 0.5) * np.pi / L) for k in range(q)], axis=1)
    ug, _, _, _ = np.linalg.lstsq(A, gamma.detach().cpu().numpy().astype(np.float64), rcond=None)
    return torch.as_tensor(ug, dtype=torch.float32)


def init_modes_from_phi_col(phi_col: torch.Tensor, q: int) -> torch.Tensor:
    return init_modes_from_gamma(phi_col.reshape(-1), q)


def phi_from_modes(u_phi: torch.Tensor, L: int) -> torch.Tensor:
    """u_phi: (4, q) → phi (L, 4)."""
    cols = [fourier_schedule(u_phi[s], L) for s in range(4)]
    return torch.stack(cols, dim=1)  # (L,4)


def train_fourier(
    W, q, L_train, nf, device, steps=80, max_train=32, max_val=12, seed=0,
    gamma0=None, phi0=None, lr=3e-3, fourier_phi=True,
):
    rng = np.random.default_rng(seed)
    if gamma0 is not None:
        u_g = torch.nn.Parameter(init_modes_from_gamma(gamma0.reshape(-1), q).to(device))
    else:
        u_g = torch.nn.Parameter(torch.randn(q, device=device) * 0.3)

    if fourier_phi:
        if phi0 is not None:
            cols = []
            for s in range(4):
                cols.append(init_modes_from_phi_col(phi0[:, s], q))
            u_phi = torch.nn.Parameter(torch.stack(cols, dim=0).to(device))  # (4,q)
        else:
            u_phi = torch.nn.Parameter(torch.zeros(4, q, device=device))
        phi_free = None
        params = [u_g, u_phi]
    else:
        u_phi = None
        if phi0 is not None and phi0.shape[0] == L_train:
            phi_free = torch.nn.Parameter(phi0.detach().clone().to(device))
        else:
            # resample/pad
            phi_free = torch.nn.Parameter(torch.zeros(L_train, 4, device=device))
            if phi0 is not None:
                L0 = phi0.shape[0]
                for s in range(4):
                    # linear interp warm
                    x = np.linspace(0, 1, L0)
                    xp = np.linspace(0, 1, L_train)
                    phi_free.data[:, s] = torch.as_tensor(
                        np.interp(xp, x, phi0[:, s].cpu().numpy()), dtype=torch.float32
                    )
        params = [u_g, phi_free]

    packs = {}
    for n in (9, 12):
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = torch.as_tensor(E, dtype=torch.float32, device=device)

    sims = {n: make_sim(n, W, L_train, nf, device) for n in (9, 12)}
    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}

    def angles():
        g = fourier_schedule(u_g, L_train)
        if fourier_phi:
            p = phi_from_modes(u_phi, L_train)
        else:
            p = phi_free
        return g, p

    for step in range(steps):
        opt.zero_grad()
        g, p = angles()
        losses = []
        for n, Eall in packs.items():
            idx = int(rng.integers(0, len(Eall)))
            out = sims[n].forward_gadget(Eall[idx], g, p, "unsat")
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(params, lr=0.15, max_iter=5, history_size=8)

                def closure():
                    opt_l.zero_grad()
                    gg, pp = angles()
                    ls = []
                    for n, Eall in packs.items():
                        for j in range(min(8, len(Eall))):
                            o = sims[n].forward_gadget(Eall[j], gg, pp, "unsat")
                            ls.append(_loss(o))
                    lt = torch.stack(ls).mean()
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)

            g, p = angles()
            vals = {}
            for n in (9, 12):
                vals[n] = eval_arm(
                    n, W, g.detach(), p.detach(), L_train, nf, device, "unsat",
                    split="val", max_count=max_val,
                )
            sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
            if sc < best["score"]:
                st = {
                    "u_gamma": u_g.detach().cpu().clone(),
                    "u_phi": None if u_phi is None else u_phi.detach().cpu().clone(),
                    "phi_free": None if phi_free is None else phi_free.detach().cpu().clone(),
                    "fourier_phi": fourier_phi,
                    "q": q,
                    "L_train": L_train,
                    "cost_mode": "unsat",
                    "W": "BinH",
                    "gamma_Ltrain": g.detach().cpu().clone(),
                    "phi_Ltrain": p.detach().cpu().clone(),
                }
                best = {"score": sc, "step": step, "val": vals, "state": st}
            if step % 10 == 0 or step + 1 == steps:
                log(
                    f"step {step}: S={sc:.3f} "
                    f"n9={vals[9]['median_inv_p']:.3f}/{vals[9]['B90_inv_p']:.3f} "
                    f"n12={vals[12]['median_inv_p']:.3f}/{vals[12]['B90_inv_p']:.3f}"
                )
    return best


@torch.no_grad()
def deploy_eval(W, state, L, nf, device, split_counts, scale_ns=None, scale_B=0):
    q = state["q"]
    u_g = state["u_gamma"].to(device)
    if state["fourier_phi"]:
        phi = phi_from_modes(state["u_phi"].to(device), L)
    else:
        # tile/interp free phi to L
        p0 = state["phi_free"]
        if p0.shape[0] == L:
            phi = p0.to(device)
        else:
            x = np.linspace(0, 1, p0.shape[0])
            xp = np.linspace(0, 1, L)
            arr = np.stack(
                [np.interp(xp, x, p0[:, s].cpu().numpy()) for s in range(4)], axis=1
            )
            phi = torch.as_tensor(arr, dtype=torch.float32, device=device)
    gamma = fourier_schedule(u_g, L)
    out = {"L": L, "ladder": {}}
    for split, counts in split_counts.items():
        out["ladder"][split] = {}
        for n, mc in counts.items():
            m = eval_arm(
                n, W, gamma, phi, L, nf, device, "unsat",
                split=split, max_count=mc,
            )
            if m:
                out["ladder"][split][n] = m
                log(
                    f"deploy L{L} {split} n{n}: "
                    f"Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} B={m['B']}"
                )
    if scale_ns and scale_B:
        out["scale1000"] = scale1000_eval(
            W, gamma.cpu(), phi.cpu(), L, nf, device, "unsat",
            ns=list(scale_ns), max_count=scale_B,
        )
    out["gamma"] = gamma.cpu()
    out["phi"] = phi.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--L_train", type=int, default=16)
    ap.add_argument("--deploy_Ls", type=str, default="16,20")
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_fourier_phi", action="store_true")
    ap.add_argument("--scale_ns", type=str, default="9,12,15")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--scale_only_best", action="store_true", default=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(UNSAT_PARAMS, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    log("warm from unsat projector params", UNSAT_PARAMS)

    fourier_phi = not args.no_fourier_phi
    best = train_fourier(
        W, args.q, args.L_train, args.nf, device,
        steps=args.steps, seed=args.seed,
        gamma0=g0, phi0=p0, fourier_phi=fourier_phi,
    )
    torch.save(best["state"], OUT / "modes.pt")
    results = {
        "q": args.q,
        "L_train": args.L_train,
        "fourier_phi": fourier_phi,
        "train_best_step": best["step"],
        "train_score": best["score"],
        "baseline_projector_scale": BASELINE_SCALE,
        "armC_B1000_ref": ARM_C_B1000,
        "deploys": {},
    }

    deploy_Ls = [int(x) for x in args.deploy_Ls.split(",") if x.strip()]
    split_counts = {
        "val": {9: 24, 12: 24, 15: 16, 18: 16},
        "test": {9: 40, 12: 40, 15: 40, 18: 16},
    }
    # first pass: ladder only
    for L in deploy_Ls:
        log(f"=== deploy L={L} ladder ===")
        results["deploys"][L] = deploy_eval(
            W, best["state"], L, args.nf, device, split_counts,
            scale_ns=None, scale_B=0,
        )

    # pick best deploy by val n15 Med (lower better), require <= baseline*1.02
    def val15(dep):
        m = dep.get("ladder", {}).get("val", {}).get(15) or dep.get("ladder", {}).get("val", {}).get("15")
        return m["median_inv_p"] if m else 1e9

    best_L = min(deploy_Ls, key=lambda L: val15(results["deploys"][L]))
    log(f"best deploy L by val n15 Med: {best_L}")

    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]
    # scale best L; if L!=16 also scale L=16 for fair compare
    for L in sorted(set([best_L, 16]) & set(deploy_Ls) | {best_L}):
        log(f"=== scale1000 L={L} ns={scale_ns} ===")
        sc = scale1000_eval(
            W,
            results["deploys"][L]["gamma"],
            results["deploys"][L]["phi"],
            L, args.nf, device, "unsat",
            ns=scale_ns, max_count=args.scale_B,
        )
        results["deploys"][L]["scale1000"] = sc

    # keep gate
    keep = False
    reason = []
    sc_best = results["deploys"][best_L].get("scale1000") or {}
    m15 = sc_best.get(15) or sc_best.get("15")
    if m15:
        base = BASELINE_SCALE[15]["Med"]
        arm = ARM_C_B1000[15]["Med"]
        if m15["median_inv_p"] < base * 0.98 and m15["median_inv_p"] <= arm * 1.02:
            keep = True
            reason.append(
                f"n15 Med {m15['median_inv_p']:.3f} < baseline {base:.3f}*0.98 and ≤ ArmC*1.02"
            )
        else:
            reason.append(
                f"n15 Med {m15['median_inv_p']:.3f} vs baseline {base:.3f} ArmC {arm:.3f} — no keep"
            )
    else:
        # fall back to val
        v = results["deploys"][best_L]["ladder"]["val"].get(15)
        if v and v["median_inv_p"] < 2.5:
            reason.append(f"no scale15; val15={v['median_inv_p']:.3f}")
        else:
            reason.append("no scale n15")

    results["keep"] = keep
    results["keep_reason"] = reason
    results["best_L"] = best_L
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # RESULTS.md
    lines = [
        "# Fourier-γ deepen — projector-gadget (unsat) BinH",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"q={args.q} L_train={args.L_train} fourier_phi={fourier_phi} best_L={best_L} **keep={keep}**",
        "",
        f"Reason: {'; '.join(reason)}",
        "",
        "## Baseline projector scale1000",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n, m in BASELINE_SCALE.items():
        lines.append(f"| {n} | {m['Med']:.3f} | {m['B90']:.3f} |")
    lines += ["", "## Deploys", ""]
    for L in deploy_Ls:
        dep = results["deploys"][L]
        lines += [f"### L={L}", "", "#### test", "",
                  "| n | Med | B90 | B |", "|---|-----|-----|---|"]
        for n in (9, 12, 15, 18):
            m = dep["ladder"]["test"].get(n) or dep["ladder"]["test"].get(str(n))
            if m:
                lines.append(
                    f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | {m['B']} |"
                )
        sc = dep.get("scale1000") or {}
        if sc:
            lines += ["", "#### scale1000", "",
                      "| n | Med | B90 | vs base | vs ArmC |",
                      "|---|-----|-----|---------|---------|"]
            for n in scale_ns:
                m = sc.get(n) or sc.get(str(n))
                if not m:
                    continue
                b = BASELINE_SCALE.get(n, {}).get("Med")
                a = ARM_C_B1000.get(n, {}).get("Med")
                rb = (b / m["median_inv_p"]) if b else float("nan")
                ra = (a / m["median_inv_p"]) if a else float("nan")
                lines.append(
                    f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                    f"×{rb:.3f} | ×{ra:.3f} |"
                )
        lines.append("")
    lines += [
        "## Verdict",
        "",
        ("KEEP Fourier deepen — replaces L16 free-γ projector schedule."
         if keep else
         "DISCARD Fourier deepen — stick with free-γ/φ projector unsat_params.pt."),
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md", "keep=", keep)
    # save best deploy angles
    torch.save(
        {
            "gamma": results["deploys"][best_L]["gamma"],
            "phi": results["deploys"][best_L]["phi"],
            "L": best_L,
            "cost_mode": "unsat",
            "W": "BinH",
            "keep": keep,
        },
        OUT / f"deploy_L{best_L}_params.pt",
    )


if __name__ == "__main__":
    main()
