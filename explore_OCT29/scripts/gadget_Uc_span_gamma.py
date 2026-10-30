#!/usr/bin/env python3
"""Span-split γ_ℓ^(s) on projector-Uc L28 BinH backbone (probe8-style).

COST: arg = Σ_{s∈{2,3,4}} γ_ℓ^(s) · E^(s) · (n/m); mixer = W·SNAP(φ_pc)·W†.
Warm γ_ℓ^(s) ← champ γ_ℓ. Train {6,9,12}. Promote vs L28 gate:
  n15 Med < 1.655*0.98, n18 Med < 2.183*0.98.

Optional --also_L32: retrain plain unsat at L=32 from L28 interp.

Writes runs/gadget_Uc/span_gamma/
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
from explore_OCT29.lib.packing import NU  # noqa: E402
from explore_OCT29.lib.sat_gen import clause_count  # noqa: E402
from explore_OCT29.lib.spectral_arm import stack_span_E, warm_gamma_span  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_depth import interp_angles  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    GadgetAnsatz,
    _S,
    _jsonable,
    _loss,
    make_sim,
    scale1000_eval,
    train_arm,
)
from explore_OCT29.scripts.gadget_Uc_freedom import FreedomAnsatz, make_free  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "span_gamma"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
SCALE_DIR = _PKG / "data" / "scale1000"
BASE28 = {
    9: {"Med": 1.144, "B90": 1.962},
    12: {"Med": 1.317, "B90": 3.364},
    15: {"Med": 1.655, "B90": 6.385},
    18: {"Med": 2.183, "B90": 15.419},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[spanγ]", *a, flush=True)


class SpanGammaAnsatz(FreedomAnsatz):
    """Per-layer γ_ℓ^(s) for s=2,3,4 on binary BinH grid."""

    def forward_span_gamma(
        self,
        E: torch.Tensor,
        E_span: torch.Tensor,
        gamma_span: torch.Tensor,
        phi: torch.Tensor,
    ) -> dict:
        L = self.L
        gs = gamma_span.to(dtype=self.rdtype, device=self.lam.device).reshape(L, 3)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        n = self.cfg.n
        scale = float(n) / float(clause_count(n))
        Es = E_span.to(device=self.lam.device, dtype=self.rdtype)  # (3, 2^n)
        grids = torch.stack([Es[s][self.grid_to_x] * scale for s in range(3)], dim=0)

        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)
        sl = tuple(slice(0, NU) for _ in range(self.C))

        for ell in range(L):
            arg = (gs[ell].reshape(3, *([1] * self.C)) * grids).sum(dim=0)
            arg = arg - arg.amax()
            phase = torch.exp(-1j * arg.to(self.cdtype))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new
            ph = phi[ell].to(self.rdtype).reshape(4)
            th = ph[self.popcount]
            mixer = self.Wmat @ self._snap_theta(th) @ self.Wdag
            psi = self._apply_all(psi, mixer)

        probs = psi.real.square() + psi.imag.square()
        p_comp = probs[sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        E_grid = E.to(device=self.lam.device)[self.grid_to_x]
        p_succ = p_comp[E_grid == 0].sum().clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        return {"p_succ": p_succ, "p_leak": p_leak, "inv_p": inv_p, "norm": probs.sum()}


def make_span(n, W, L, nf, device):
    return SpanGammaAnsatz(AnsatzConfig(n=n, L=L, nf=nf), W, device=device).to(device)


def preload(n, max_train, device):
    pack = load_pack(n)
    _, cls_v, cls_s, _, E = _split_arrays(pack, "train", max_count=max_train)
    if E is None or len(E) == 0:
        _, cls_v, cls_s, _, E = _split_arrays(pack, "val", max_count=max_train)
    spans = [stack_span_E(n, cls_v[b], cls_s[b]) for b in range(len(E))]
    return {
        "E": torch.as_tensor(E, dtype=torch.float32, device=device),
        "E_span": torch.as_tensor(np.stack(spans), dtype=torch.float32, device=device),
        "cls_v": cls_v,
        "cls_s": cls_s,
    }


@torch.no_grad()
def eval_span(n, W, gspan, phi, L, nf, device, split="val", max_count=24,
              data_dir=None, progress_every=0):
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        key = "scale1000" if "scale1000_seeds" in pack else "test"
    seeds, cls_v, cls_s, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    sim = make_span(n, W, L, nf, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        Esp = torch.as_tensor(
            stack_span_E(n, cls_v[b], cls_s[b]), dtype=torch.float32, device=device
        )
        out = sim.forward_span_gamma(Eb, Esp, gspan, phi)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if progress_every and ((b + 1) % progress_every == 0 or b + 1 == B):
            log(f"eval n{n} {b+1}/{B} Med≈{np.median(invs):.3f}")
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n, "split": key, "B": B,
        "median_inv_p": med, "B90_inv_p": b90, "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "wall_s": time.time() - t0, "L": L,
    }


def train_span(W, L, nf, device, steps=100, max_train=32, max_val=12, seed=0,
               gamma0=None, phi0=None, train_ns=(6, 9, 12), lr=2e-3):
    rng = np.random.default_rng(seed)
    if gamma0 is not None:
        gspan = torch.nn.Parameter(warm_gamma_span(gamma0.reshape(-1), 3))
    else:
        gspan = torch.nn.Parameter(torch.linspace(0.3, 1.0, L).unsqueeze(1).expand(L, 3).clone())
    phi = torch.nn.Parameter(
        phi0.detach().clone() if phi0 is not None else torch.zeros(L, 4)
    )
    params = [gspan, phi]
    packs = {n: preload(n, max_train, device) for n in train_ns}
    sims = {n: make_span(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}
    n_lo, n_hi = min(train_ns), max(train_ns)

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n, P in packs.items():
            idx = int(rng.integers(0, len(P["E"])))
            out = sims[n].forward_span_gamma(P["E"][idx], P["E_span"][idx], gspan, phi)
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(params, lr=0.15, max_iter=4, history_size=8)

                def closure():
                    opt_l.zero_grad()
                    ls = []
                    for n, P in packs.items():
                        for j in range(min(6, len(P["E"]))):
                            o = sims[n].forward_span_gamma(
                                P["E"][j], P["E_span"][j], gspan, phi
                            )
                            ls.append(_loss(o))
                    lt = torch.stack(ls).mean()
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)

            vals = {}
            for n in train_ns:
                vals[n] = eval_span(
                    n, W, gspan.detach(), phi.detach(), L, nf, device,
                    split="val", max_count=max_val,
                )
            if 9 in vals:
                sc = 0.25 * vals[n_lo]["S"] + 0.35 * vals[9]["S"] + 0.4 * vals[n_hi]["S"]
            else:
                sc = 0.4 * vals[n_lo]["S"] + 0.6 * vals[n_hi]["S"]
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma_span": gspan.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "L": L,
                        "mode": "span_gamma",
                        "W": "BinH",
                        "cost_mode": "projector_span",
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns
                )
                gm = gspan.detach().mean(dim=0).cpu().numpy()
                log(f"step {step}: S={sc:.3f} {bits} γmean_s={np.array2string(gm, precision=3)}")
    return best


def scale_span(W, st, L, nf, device, ns, max_count=1000):
    out = {}
    for n in ns:
        m = eval_span(
            n, W, st["gamma_span"], st["phi"], L, nf, device,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=50 if n >= 15 else 100,
        )
        if m:
            out[n] = m
            log(f"scale n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}")
    return out


def gate(sc, base=BASE28):
    keep = True
    reasons = []
    for n in (15, 18):
        m = sc.get(n) or sc.get(str(n))
        thr = base[n]["Med"] * 0.98
        if not m:
            keep = False
            reasons.append(f"missing n{n}")
            continue
        ok = m["median_inv_p"] < thr
        reasons.append(f"n{n} {m['median_inv_p']:.3f} {'<' if ok else '>='} {thr:.3f}")
        keep = keep and ok
    return keep, reasons


def run_L32(W, device, nf, steps, scale_B):
    """Plain unsat retrain at L=32 from L28 champ."""
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    L = 32
    g_w, p_w = interp_angles(g0, p0, L)
    log(f"=== L=32 plain retrain from L28 ===")
    best = train_arm(
        "unsat", W, L, nf, device, steps=steps, max_train=32, max_val=12, seed=32,
        gamma0=g_w, phi0=p_w, train_ns=(6, 9, 12),
    )
    st = best["state"]
    st["L"] = L
    torch.save(st, OUT / "retrain_L32_params.pt")
    sc = scale1000_eval(
        W, st["gamma"], st["phi"], L, nf, device, "unsat",
        ns=[9, 12, 15, 18], max_count=scale_B,
    )
    keep, reasons = gate(sc, BASE28)
    # also vs span if we have better base later — for now vs L28
    ns = [9, 12, 15, 18]
    bm = fit_b(ns, [sc[n]["median_inv_p"] for n in ns])
    bb = fit_b(ns, [sc[n]["B90_inv_p"] for n in ns])
    return {
        "scale1000": sc, "keep": keep, "reasons": reasons,
        "b_Med": bm, "b_B90": bb, "train_step": best["step"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=28)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--also_L32", action="store_true")
    ap.add_argument("--skip_span", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    L = args.L
    assert st0["gamma"].numel() == L or True
    # if champ L != args.L, interp
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    if g0.numel() != L:
        from explore_OCT29.scripts.gadget_Uc_depth import interp_angles
        g0, p0 = interp_angles(g0, p0, L)
        log(f"interp champ L{st0['gamma'].numel()} → L{L}")

    results = {"baseline_L28": BASE28, "armC": ARM_C_B1000, "L": L}

    if not args.skip_span:
        log("=== train span-split γ_ℓ^(s) ===")
        best = train_span(
            W, L, args.nf, device, steps=args.steps, seed=7,
            gamma0=g0, phi0=p0, train_ns=(6, 9, 12),
        )
        st = best["state"]
        torch.save(st, OUT / "span_gamma_params.pt")
        log("scale1000 span")
        sc = scale_span(W, st, L, args.nf, device, [9, 12, 15, 18], args.scale_B)
        keep, reasons = gate(sc)
        gm = st["gamma_span"].mean(dim=0).tolist()
        results["span_gamma"] = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "scale1000": sc,
            "keep": keep,
            "reasons": reasons,
            "gamma_span_mean": gm,
        }
        log(f"span keep={keep}", reasons, "γmean", gm)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    if args.also_L32:
        results["L32"] = run_L32(W, device, args.nf, args.steps, args.scale_B)
        log("L32", results["L32"]["keep"], results["L32"]["reasons"])
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # RESULTS.md
    lines = [
        "# Span-split γ_ℓ^(s) on projector-Uc L28",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"Gate vs L28: n15 < {BASE28[15]['Med']*0.98:.3f}, n18 < {BASE28[18]['Med']*0.98:.3f}.",
        "",
        "## Span-split",
        "",
    ]
    sg = results.get("span_gamma")
    if sg:
        lines += [
            f"keep=**{sg['keep']}** — {'; '.join(sg['reasons'])}",
            "",
            f"γ_ℓ mean by span s=2,3,4: {sg.get('gamma_span_mean')}",
            "",
            "| n | spanγ | L28 champ | Arm C |",
            "|---|-------|-----------|-------|",
        ]
        for n in (9, 12, 15, 18):
            m = sg["scale1000"].get(n) or sg["scale1000"].get(str(n))
            b = BASE28[n]
            a = ARM_C_B1000[n]
            ms = f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}" if m else "—"
            lines.append(f"| {n} | {ms} | {b['Med']:.3f}/{b['B90']:.3f} | {a['Med']:.3f}/{a['B90']:.3f} |")
        lines.append("")
        if sg["keep"]:
            lines.append("**PROMOTE** span-split over L28 plain.")
        else:
            lines.append("**DISCARD** span-split — keep L28 plain champ.")
        lines.append("")

    if "L32" in results:
        L32 = results["L32"]
        lines += [
            "## Optional L=32 plain",
            "",
            f"keep=**{L32['keep']}** — {'; '.join(L32['reasons'])}",
            "",
            "| n | L32 | L28 |",
            "|---|-----|-----|",
        ]
        for n in (9, 12, 15, 18):
            m = L32["scale1000"].get(n) or L32["scale1000"].get(str(n))
            b = BASE28[n]
            ms = f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}" if m else "—"
            lines.append(f"| {n} | {ms} | {b['Med']:.3f}/{b['B90']:.3f} |")
        lines += [
            "",
            f"b_Med≈{L32['b_Med']['b']:.4f}, b_B90≈{L32['b_B90']['b']:.4f}.",
            "",
            ("PROMOTE L32." if L32["keep"] else "DISCARD L32 — stop depth ladder."),
            "",
        ]

    lines += [
        "## Next",
        "",
        "If both discard: train-set tweak (add n15 soft) or freeze depth/span.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
