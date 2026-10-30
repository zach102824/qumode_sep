#!/usr/bin/env python3
"""Opus probe 8 — span-split γ_ℓ^(s) on Arm C spectral mixer.

Opus #3 (19:08): split COST by clause span s∈{2,3,4} → three shared γ_ℓ^(s)/layer.
Warm from Arm C (probe6 promoted): γ_ℓ^(s)←γ_C, mixer = W_H·SNAP(φ_pc)·W_H†.
Train {9,12} Adam~120 + LBFGS; gate on val2 (full val); sealed test only if val2 beats Arm C.

Stop: touch explore_OCT29/runs/STOP_PROBE8
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
from explore_OCT29.lib.spectral_arm import (  # noqa: E402
    build_W_H,
    load_arm_C,
    stack_span_E,
    warm_gamma_span,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.probe6_spectral_mixer import SpectralAnsatz  # noqa: E402

OUT = _PKG / "runs" / "probe8_span_split"
STOP = _PKG / "runs" / "STOP_PROBE8"
SPANS = (2, 3, 4)  # index 0,1,2


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe8]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


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


def _loss(out, leakpen=0.1):
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


class SpanSplitAnsatz(SpectralAnsatz):
    """Arm C mixer + COST = ∏_s exp(-i γ_ℓ^(s) E^(s) n/m), joint amax-center."""

    def forward_span(
        self,
        E_span: torch.Tensor,
        gamma_span: torch.Tensor,
        phi: torch.Tensor,
    ) -> dict:
        """E_span: (3, 2^n) float; gamma_span: (L, 3); phi: (L, 4)."""
        L = self.L
        gs = gamma_span.to(dtype=self.rdtype, device=self.lam.device).reshape(L, 3)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device).reshape(L, 4)
        n = self.cfg.n
        m = float(clause_count(n))
        scale = float(n) / m

        Es = E_span.to(device=self.lam.device, dtype=self.rdtype)
        # map each span energy onto Gray product grid → (3, NU,...,NU)
        grids = []
        for s in range(3):
            g = Es[s][self.grid_to_x] * scale
            grids.append(g)
        # stack (3, *grid)
        E_grids = torch.stack(grids, dim=0)

        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)
        sl = tuple(slice(0, NU) for _ in range(self.C))

        for ell in range(L):
            # weighted cost arg = Σ_s γ^(s) E_grid^(s); center by amax
            arg = (gs[ell].reshape(3, *([1] * self.C)) * E_grids).sum(dim=0)
            arg = arg - arg.amax()
            phase = torch.exp(-1j * arg.to(self.cdtype))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new

            # Arm C shell mixer
            ph = phi[ell].reshape(4)
            th = ph[self.popcount]
            mixer = self.Wmat @ self._snap_theta(th) @ self.Wdag
            psi = self._apply_all(psi, mixer)

        probs = psi.real.square() + psi.imag.square()
        p_comp = probs[sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        # success: total E = sum spans == 0
        E_tot = Es.sum(dim=0)
        E_grid = E_tot[self.grid_to_x]
        succ = p_comp[E_grid == 0].sum()
        p_succ = succ.clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        return {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": inv_p,
            "norm": probs.sum(),
        }


def preload_span_pack(n: int, split: str, max_count: int | None, device):
    pack = load_pack(n)
    seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, split, max_count=max_count)
    if len(seeds) == 0:
        return None
    Esp = []
    for b in range(len(seeds)):
        Esp.append(stack_span_E(n, cls_v[b], cls_s[b]))
    Esp_t = torch.as_tensor(np.stack(Esp, axis=0), dtype=torch.float32, device=device)
    # sanity vs stored E when present
    if E is not None:
        recon = Esp_t.sum(dim=1).cpu().numpy()
        diff = float(np.max(np.abs(recon - E.astype(np.float32))))
        if diff > 0:
            log(f"WARN span sum != E at n{n} {split}: maxdiff={diff}")
    return {
        "n": n,
        "split": split,
        "seeds": seeds,
        "nsol": nsol,
        "E_span": Esp_t,  # (B, 3, 2^n)
        "B": len(seeds),
    }


@torch.no_grad()
def eval_span(sim_cache, bundle, gamma_span, phi, L, nf, device):
    n = bundle["n"]
    if n not in sim_cache:
        cfg = AnsatzConfig(n=n, L=L, nf=nf)
        W = build_W_H(nf=nf)
        sim_cache[n] = SpanSplitAnsatz(cfg, W, device=device).to(device)
    sim = sim_cache[n]
    invs, leaks = [], []
    t0 = time.time()
    Esp = bundle["E_span"]
    for b in range(bundle["B"]):
        out = sim.forward_span(Esp[b], gamma_span, phi)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n,
        "split": bundle["split"],
        "B": bundle["B"],
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "L": L,
        "L_times_Med": float(L * med),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "frac_inv_le_4": float(np.mean(invs_a <= 4.0)),
        "wall_s": time.time() - t0,
    }


@torch.no_grad()
def eval_arm_C_baseline(bundle, gamma, phi, L, nf, device, sim_cache):
    """Equal-γ span forward == Arm C COST; same mixer."""
    gspan = warm_gamma_span(gamma, 3).to(device)
    return eval_span(sim_cache, bundle, gspan, phi.to(device), L, nf, device)


def train(
    gamma0, phi0, L, nf, device, steps=120, max_train=32, max_val=12, seed=0, lr=2e-3,
    freeze_phi=False,
):
    rng = np.random.default_rng(seed)
    gspan = torch.nn.Parameter(warm_gamma_span(gamma0, 3))
    phi = torch.nn.Parameter(phi0.detach().clone().to(dtype=torch.float32))
    if freeze_phi:
        phi.requires_grad_(False)
        params = [gspan]
    else:
        params = [gspan, phi]

    packs = {}
    for n in (9, 12):
        b = preload_span_pack(n, "train", max_train, device)
        if b is None or b["B"] == 0:
            b = preload_span_pack(n, "val", max_train, device)
        packs[n] = b
        log(f"train pack n{n}: B={b['B']} E_span={tuple(b['E_span'].shape)}")

    val_bundles = {
        n: preload_span_pack(n, "val", max_val, device) for n in (9, 12)
    }
    sims = {}
    W = build_W_H(nf=nf)
    for n in (9, 12):
        sims[n] = SpanSplitAnsatz(AnsatzConfig(n=n, L=L, nf=nf), W, device=device).to(device)

    # warm-start sanity: equal-γ should match Arm C path (loss finite, inv reasonable)
    with torch.no_grad():
        for n, b in packs.items():
            out0 = sims[n].forward_span(b["E_span"][0], gspan.detach(), phi.detach())
            log(f"warm n{n} inv_p={float(out0['inv_p']):.3f} leak={float(out0['p_leak']):.4f}")

    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}

    for step in range(steps):
        if stopped():
            log("STOP_PROBE8 seen — break train")
            break
        opt.zero_grad()
        losses = []
        for n, b in packs.items():
            idx = int(rng.integers(0, b["B"]))
            out = sims[n].forward_span(b["E_span"][idx], gspan, phi)
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(
                    [p for p in params if p.requires_grad],
                    lr=0.2, max_iter=4, history_size=8,
                )

                def closure():
                    opt_l.zero_grad()
                    ls = []
                    for n, b in packs.items():
                        for j in range(min(8, b["B"])):
                            o = sims[n].forward_span(b["E_span"][j], gspan, phi)
                            ls.append(_loss(o))
                    lt = torch.stack(ls).mean()
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)

            vals = {}
            scache = {}
            for n, vb in val_bundles.items():
                vals[n] = eval_span(scache, vb, gspan.detach(), phi.detach(), L, nf, device)
            sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma_span": gspan.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "mode": "span_shell",
                        "freeze_phi": freeze_phi,
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                # γ ratios mean over layers
                g = gspan.detach()
                ratios = (g / g.mean(dim=1, keepdim=True).clamp(min=1e-6)).mean(dim=0)
                log(
                    f"train step {step}: Scomb={sc:.3f}",
                    f"n9:{vals[9]['median_inv_p']:.2f}/{vals[9]['B90_inv_p']:.2f}",
                    f"n12:{vals[12]['median_inv_p']:.2f}/{vals[12]['B90_inv_p']:.2f}",
                    f"γratio_mean≈[{ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f}]",
                )
    return best


def fit_b_ns(ladder, ns=(12, 15, 18)):
    ns_u, meds, b90s = [], [], []
    for n in ns:
        m = ladder.get(str(n)) or ladder.get(n)
        if m:
            ns_u.append(int(n))
            meds.append(m["median_inv_p"])
            b90s.append(m["B90_inv_p"])
    if len(ns_u) < 2:
        return None
    return {"Med": fit_b(ns_u, meds), "B90": fit_b(ns_u, b90s), "ns": ns_u}


def write_results(results):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    lines = [
        "# Opus probe 8 — span-split γ on Arm C spectral mixer",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "COST: γ_ℓ^(s) for clause spans s∈{2,3,4}; mixer = Arm C W_H·SNAP(φ_pc)·W_H†.",
        "Gate = **val2** (full val). Test sealed unless val2 beats Arm C.",
        "",
    ]
    if "arm_C_val2" in results:
        lines.append("## Arm C baseline (val2, equal-γ span path)")
        lines.append("| n | Med | B90 | leak | L×Med | frac≤2 | B |")
        lines.append("|---|-----|-----|------|-------|--------|---|")
        for n in (9, 12, 15, 18):
            m = results["arm_C_val2"].get(str(n)) or results["arm_C_val2"].get(n)
            if not m:
                continue
            lines.append(
                f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                f"{m['mean_leak']:.4f} | {m['L_times_Med']:.1f} | "
                f"{m['frac_inv_le_2']:.2f} | {m['B']} |"
            )
        lines.append("")
    if "span_val2" in results:
        lines.append("## Span-split (val2)")
        lines.append("| n | Med | B90 | leak | L×Med | frac≤2 | B |")
        lines.append("|---|-----|-----|------|-------|--------|---|")
        for n in (9, 12, 15, 18):
            m = results["span_val2"].get(str(n)) or results["span_val2"].get(n)
            if not m:
                continue
            lines.append(
                f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                f"{m['mean_leak']:.4f} | {m['L_times_Med']:.1f} | "
                f"{m['frac_inv_le_2']:.2f} | {m['B']} |"
            )
        lines.append("")
    if "gamma_ratios" in results:
        lines.append(f"## Learned γ ratios (mean over L)\n\n`{results['gamma_ratios']}`\n")
    if "promote" in results:
        lines.append(f"## Promote vs Arm C (val2)\n\n`{results['promote']}`\n")
    if "test_milestone" in results:
        lines.append(f"## Test milestone\n\n`{results['test_milestone']}`\n")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "results.json", "and RESULTS.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--freeze_phi", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--force_test", action="store_true",
                    help="Look at sealed test even if val2 does not beat Arm C")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L, nf = args.L, args.nf

    arm = load_arm_C()
    if arm["L"] != L:
        raise SystemExit(f"Arm C L={arm['L']} != --L {L}")
    gamma_c, phi_c = arm["gamma"], arm["phi"]
    log("loaded Arm C", arm["path"], f"L={L}", "phi", tuple(phi_c.shape))

    results = {
        "L": L, "nf": nf, "seed": args.seed, "steps": args.steps,
        "freeze_phi": args.freeze_phi, "arm_C": arm["path"],
    }

    # val2 counts: full packs
    val2_counts = {9: 40, 12: 50, 15: 40, 18: 16}
    test_counts = {9: 40, 12: 50, 15: 40, 18: 16}

    log("=== preload val2 span packs ===")
    val2 = {}
    for n, mc in val2_counts.items():
        if stopped():
            break
        val2[n] = preload_span_pack(n, "val", mc, device)
        log(f"val2 n{n}: B={val2[n]['B']}")

    sim_cache = {}
    log("=== Arm C baseline on val2 (equal-γ span path) ===")
    results["arm_C_val2"] = {}
    for n, b in val2.items():
        m = eval_arm_C_baseline(b, gamma_c, phi_c, L, nf, device, sim_cache)
        results["arm_C_val2"][n] = m
        log(
            f"ArmC val2 n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}",
            f"leak={m['mean_leak']:.4f} L*Med={m['L_times_Med']:.1f}",
            f"frac≤2={m['frac_inv_le_2']:.2f} B={m['B']}",
        )

    if args.skip_train:
        write_results(results)
        return

    if stopped():
        log("stopped before train")
        write_results(results)
        return

    log("=== train span-split γ (warm Arm C) ===")
    best = train(
        gamma_c, phi_c, L, nf, device,
        steps=args.steps, max_train=args.max_train, max_val=args.max_val,
        seed=args.seed, freeze_phi=args.freeze_phi,
    )
    st = best["state"]
    torch.save(st, OUT / "span_params.pt")
    results["train_best_step"] = best["step"]
    results["train_score"] = best["score"]
    results["train_val_small"] = best["val"]

    gspan = st["gamma_span"]
    phi = st["phi"]
    # ratios
    g = gspan
    ratios = (g / g.mean(dim=1, keepdim=True).clamp(min=1e-6)).mean(dim=0)
    results["gamma_ratios"] = {
        "span2": float(ratios[0]),
        "span3": float(ratios[1]),
        "span4": float(ratios[2]),
        "gamma_span_mean": g.mean(dim=0).tolist(),
    }
    log("γ ratios", results["gamma_ratios"])

    log("=== span-split val2 ladder ===")
    results["span_val2"] = {}
    sim_cache = {}
    for n, b in val2.items():
        if stopped():
            break
        m = eval_span(sim_cache, b, gspan, phi, L, nf, device)
        results["span_val2"][n] = m
        log(
            f"span val2 n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}",
            f"leak={m['mean_leak']:.4f} L*Med={m['L_times_Med']:.1f}",
            f"frac≤2={m['frac_inv_le_2']:.2f} B={m['B']}",
        )
    results["b_val2_12_15_18"] = fit_b_ns(results["span_val2"])

    # promote on val2 vs Arm C: need better Med at n15 AND n18
    try:
        a15 = results["arm_C_val2"][15]["median_inv_p"]
        a18 = results["arm_C_val2"][18]["median_inv_p"]
        s15 = results["span_val2"][15]["median_inv_p"]
        s18 = results["span_val2"][18]["median_inv_p"]
        r15 = a15 / max(s15, 1e-12)
        r18 = a18 / max(s18, 1e-12)
        ok = (r15 >= 1.05) and (r18 >= 1.05)  # soft 5% — not as hard as 1.3× mixer gate
        results["promote"] = {
            "ratio_n15": r15, "ratio_n18": r18, "ok": ok,
            "armC_n15": a15, "span_n15": s15,
            "armC_n18": a18, "span_n18": s18,
        }
        log("promote(val2)", results["promote"])
    except Exception as e:
        results["promote"] = {"error": str(e), "ok": False}
        ok = False

    if results["promote"].get("ok") or args.force_test:
        log("=== sealed test milestone ===")
        results["test_milestone"] = {}
        for n, mc in test_counts.items():
            if n < 15:
                continue  # only n15/n18 sealed look
            b = preload_span_pack(n, "test", mc, device)
            m = eval_span({}, b, gspan, phi, L, nf, device)
            results["test_milestone"][n] = m
            log(
                f"TEST n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}",
                f"L*Med={m['L_times_Med']:.1f} frac≤2={m['frac_inv_le_2']:.2f}",
            )
    else:
        log("val2 did not beat Arm C — test stays sealed")

    write_results(results)
    log("DONE →", OUT)


if __name__ == "__main__":
    main()
