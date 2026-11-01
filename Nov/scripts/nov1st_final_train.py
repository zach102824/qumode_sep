#!/usr/bin/env python3
"""Nov1st-final skeleton — sandwich-cost BinH ≡ sealed BinH-proj.

LOCKED motif (Zach 2026-11-01):
  |0⟩ → [Uc^X → bare SNAP]^L → final W → Fock
  Uc^X = W D W,  W=W†=H^{⊗3},  D=exp(-iγ E)=Uc^Z

Sealed equivalent (same unitary):
  prep W → [D → W SNAP W†]^L → Fock  (no final W)
  = GadgetAnsatz.forward_gadget(..., cost_mode="unsat")

Default CLI: --smoke only (overlap check). Train/scale1000 are gated.
Writes Nov/runs/Nov1st_final/.

Do NOT use the killed arm: prep W → [W D W† → bare SNAP] → final W.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_NOV = Path(__file__).resolve().parents[1]
_ROOT = _NOV.parent
_PKG = _ROOT / "explore_OCT29"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.ansatz import AnsatzConfig  # noqa: E402
from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.packing import NU  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    GadgetAnsatz,
    _jsonable,
)
from explore_OCT29.scripts.probe6_spectral_mixer import _loss, _S  # noqa: E402

OUT = _NOV / "runs" / "Nov1st_final"
SCALE_DIR = _PKG / "data" / "scale1000"

SEALED_REF = {
    9: {"Med": 1.092, "B90": 1.776},
    12: {"Med": 1.197, "B90": 2.843},
    15: {"Med": 1.418, "B90": 5.505},
    18: {"Med": 1.752, "B90": 11.353},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[Nov1st-final]", *a, flush=True)


class Nov1stFinalAnsatz(GadgetAnsatz):
    """|0⟩ → [Uc^X=W D W → bare SNAP]^L → final W → Fock."""

    def forward_nov1st_final(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
    ) -> dict:
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()  # no prep W
        Cn = self.cost_Cn(E)
        sl = tuple(slice(0, NU) for _ in range(self.C))

        for ell in range(L):
            # Uc^X = W D W  (W=W†)
            psi = self._apply_all(psi, self.Wmat)
            phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new
            psi = self._apply_all(psi, self.Wmat)
            # bare SNAP
            ph = phi[ell].to(self.rdtype).reshape(4)
            th = ph[self.popcount]
            psi = self._apply_all(psi, self._snap_theta(th))

        # final W
        psi = self._apply_all(psi, self.Wmat)

        probs = psi.real.square() + psi.imag.square()
        p_comp = probs[sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        E_grid = E.to(device=self.lam.device)[self.grid_to_x]
        succ = p_comp[E_grid == 0].sum()
        p_succ = succ.clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        return {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": inv_p,
            "norm": probs.sum(),
            "arm": "nov1st_final",
            "probs": probs,
        }

    def forward_dispatch(self, E, gamma, phi, *, use_sealed: bool):
        if use_sealed:
            out = self.forward_gadget(E, gamma, phi, "unsat")
            out = dict(out)
            out["arm"] = "sealed_unsat"
            return out
        return self.forward_nov1st_final(E, gamma, phi)


def make_sim(n, W, L, nf, device):
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    return Nov1stFinalAnsatz(cfg, W, device=device).to(device)


@torch.no_grad()
def smoke_overlap(n=6, L=2, nf=8, device="cpu", seed=0) -> dict:
    """Prove Nov1st-final ≈ sealed forward_gadget(unsat) on one random instance."""
    device = torch.device(device)
    W = build_W_BinH(nf=nf)
    rng = np.random.default_rng(seed)
    pack = load_pack(n)
    _, _, _, _, E = _split_arrays(pack, "train", max_count=1)
    if E is None or len(E) == 0:
        _, _, _, _, E = _split_arrays(pack, "val", max_count=1)
    Eb = torch.as_tensor(E[0], dtype=torch.float32, device=device)
    gamma = torch.as_tensor(rng.uniform(0.2, 1.2, size=L), dtype=torch.float32)
    phi = torch.as_tensor(rng.normal(0.0, 0.4, size=(L, 4)), dtype=torch.float32)
    sim = make_sim(n, W, L, nf, device)
    out_n = sim.forward_nov1st_final(Eb, gamma, phi)
    out_s = sim.forward_gadget(Eb, gamma, phi, "unsat")
    # rebuild state probs for sealed via a second forward that returns probs —
    # sealed return has no probs; recompute overlap via p_succ / inv_p + full state.
    # Re-run sealed internals: use nov vs sealed success + leak as coarse; full overlap
    # from cloned forward that exposes psi. For exact overlap, recompute sealed probs.
    psi0 = sim.vacuum()
    psi_s = sim._apply_all(psi0, sim.Wmat)
    Cn = sim.cost_Cn(Eb)
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    for ell in range(L):
        phase = torch.exp(-1j * gamma[ell].to(sim.cdtype) * Cn.to(sim.cdtype))
        psi_new = psi_s.clone()
        psi_new[sl] = psi_s[sl] * phase
        psi_s = psi_new
        ph = phi[ell].to(sim.rdtype).reshape(4)
        th = ph[sim.popcount]
        mixer = sim.Wmat @ sim._snap_theta(th) @ sim.Wdag
        psi_s = sim._apply_all(psi_s, mixer)
    # Nov1st state: invert from probs is lossy; re-run nov collecting psi
    psi_n = sim.vacuum()
    for ell in range(L):
        psi_n = sim._apply_all(psi_n, sim.Wmat)
        phase = torch.exp(-1j * gamma[ell].to(sim.cdtype) * Cn.to(sim.cdtype))
        psi_new = psi_n.clone()
        psi_new[sl] = psi_n[sl] * phase
        psi_n = psi_new
        psi_n = sim._apply_all(psi_n, sim.Wmat)
        ph = phi[ell].to(sim.rdtype).reshape(4)
        th = ph[sim.popcount]
        psi_n = sim._apply_all(psi_n, sim._snap_theta(th))
    psi_n = sim._apply_all(psi_n, sim.Wmat)

    # state overlap |⟨ψ_s|ψ_n⟩|
    ov = torch.vdot(psi_s.reshape(-1), psi_n.reshape(-1))
    ov_abs = float(ov.abs().cpu())
    pn = (psi_n.real.square() + psi_n.imag.square()).cpu().numpy()
    ps = (psi_s.real.square() + psi_s.imag.square()).cpu().numpy()
    max_dp = float(np.max(np.abs(pn - ps)))
    out = {
        "n": n,
        "L": L,
        "overlap_abs": ov_abs,
        "max_abs_delta_p": max_dp,
        "nov_inv_p": float(out_n["inv_p"].cpu()),
        "sealed_inv_p": float(out_s["inv_p"].cpu()),
        "pass": ov_abs > 1.0 - 1e-5 and max_dp < 1e-6,
    }
    log(
        f"smoke n{n} L{L}: |⟨s|n⟩|={ov_abs:.8f} max|Δp|={max_dp:.3e} "
        f"inv {out['nov_inv_p']:.4f} vs {out['sealed_inv_p']:.4f} "
        f"{'PASS' if out['pass'] else 'FAIL'}"
    )
    return out


@torch.no_grad()
def eval_split(
    n, W, gamma, phi, L, nf, device, *, use_sealed, split="val",
    max_count=24, data_dir=None, progress_every=0,
):
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        if "scale1000_seeds" in pack:
            key = "scale1000"
        elif "test_seeds" in pack:
            key = "test"
        else:
            log(f"skip n{n}: no {split}_seeds")
            return None
    seeds, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(E) == 0:
        log(f"skip n{n} {key}: empty")
        return None
    sim = make_sim(n, W, L, nf, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(E)
    for b in range(B):
        out = sim.forward_dispatch(
            torch.as_tensor(E[b], dtype=torch.float32, device=device),
            gamma, phi, use_sealed=use_sealed,
        )
        invs.append(float(out["inv_p"].detach().cpu()))
        leaks.append(float(out["p_leak"].detach().cpu()))
        if progress_every and (b + 1) % progress_every == 0:
            log(
                f"eval n{n} {key} {b+1}/{B} Med≈{np.median(invs):.3f} "
                f"({(b+1)/(time.time()-t0+1e-9):.2f}/s)"
            )
    invs = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs))
    b90 = float(np.quantile(invs, 0.90))
    return {
        "n": n,
        "split": key,
        "B": B,
        "median_inv_p": med,
        "B90_inv_p": b90,
        "mean_leak": float(np.mean(leaks)),
        "S": _S(med, b90),
        "L": L,
        "motif": "sealed_unsat" if use_sealed else "nov1st_final",
        "wall_s": time.time() - t0,
    }


def scale1000_eval(W, gamma, phi, L, nf, device, ns, *, use_sealed, max_count=1000):
    out = {}
    if not SCALE_DIR.exists():
        log("scale1000 dir missing; skip")
        return out
    for n in ns:
        m = eval_split(
            n, W, gamma, phi, L, nf, device, use_sealed=use_sealed,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=50 if n >= 15 else 100,
        )
        if m:
            out[n] = m
            ref = SEALED_REF.get(n)
            bit = ""
            if ref:
                bit = (
                    f"  vs sealed Med {m['median_inv_p']/ref['Med']:.3f}× "
                    f"B90 {m['B90_inv_p']/ref['B90']:.3f}×"
                )
            log(
                f"scale1000 n{n} L{L}: Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} B={m['B']}{bit}"
            )
    return out


def train_ns(
    W, L, nf, device, *, train_ns=(6, 9), steps=150, max_train=48, max_val=12,
    seed=0, lr=2e-3, use_sealed=False, gamma0=None, phi0=None,
):
    train_ns = tuple(int(x) for x in train_ns)
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(
        gamma0.detach().clone().float()
        if gamma0 is not None
        else torch.linspace(0.3, 1.0, L)
    )
    phi = torch.nn.Parameter(
        phi0.detach().clone().float() if phi0 is not None else torch.zeros(L, 4)
    )
    packs = {}
    for n in train_ns:
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = {"E": torch.as_tensor(E, dtype=torch.float32, device=device)}
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam([gamma, phi], lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}
    n_lo, n_hi = min(train_ns), max(train_ns)

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n, P in packs.items():
            idx = int(rng.integers(0, len(P["E"])))
            out = sims[n].forward_dispatch(
                P["E"][idx], gamma, phi, use_sealed=use_sealed
            )
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gamma, phi], 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(
                    [gamma, phi], lr=0.2, max_iter=4, history_size=8
                )

                def closure():
                    opt_l.zero_grad()
                    ls = []
                    for n, P in packs.items():
                        for j in range(min(8, len(P["E"]))):
                            o = sims[n].forward_dispatch(
                                P["E"][j], gamma, phi, use_sealed=use_sealed
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
                vals[n] = eval_split(
                    n, W, gamma.detach(), phi.detach(), L, nf, device,
                    use_sealed=use_sealed, split="val", max_count=max_val,
                )
            sc = 0.4 * vals[n_lo]["S"] + 0.6 * vals[n_hi]["S"]
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "L": L,
                        "motif": "sealed_unsat" if use_sealed else "nov1st_final",
                        "W": "BinH",
                        "train_ns": list(train_ns),
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns if vals.get(n)
                )
                log(f"train L={L} step {step}: Scomb={sc:.3f} {bits}")
    return best


def write_results_md(results: dict, path: Path):
    lines = [
        "# Nov1st-final — sandwich-cost BinH ≡ sealed BinH-proj",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        r"Motif: $|0\rangle \to [U_c^{X}\to\mathrm{SNAP}]^L \to W \to$ Fock "
        r"($U_c^{X}=W D W$).",
        "",
        "## Sealed reference",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        r = SEALED_REF[n]
        lines.append(f"| {n} | {r['Med']:.3f} | {r['B90']:.3f} |")
    lines += ["", "## Smoke", "", "```json", json.dumps(results.get("smoke"), indent=2), "```", ""]
    if results.get("scale1000"):
        lines += ["## scale1000", "", "| n | Med | B90 | Med×sealed | B90×sealed |",
                  "|---|-----|-----|------------|------------|"]
        sc = results["scale1000"]
        for n in (9, 12, 15, 18):
            m = sc.get(str(n)) or sc.get(n)
            if not m:
                lines.append(f"| {n} | — | — | — | — |")
                continue
            ref = SEALED_REF[n]
            lines.append(
                f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                f"{m['median_inv_p']/ref['Med']:.3f}× | {m['B90_inv_p']/ref['B90']:.3f}× |"
            )
    lines += ["", f"Verdict: {results.get('verdict', '(pending)')}", "",
              "Plan: `Nov/PLAN_Nov1st_final.md`.", ""]
    path.write_text("\n".join(lines))


def verdict_scale(sc: dict) -> str:
    if not sc:
        return "No scale1000 yet."
    bits = []
    close = True
    for n in (9, 12, 15, 18):
        m = sc.get(str(n)) or sc.get(n)
        if not m:
            close = False
            bits.append(f"n{n}: missing")
            continue
        ref = SEALED_REF[n]
        rmed = m["median_inv_p"] / ref["Med"]
        rb90 = m["B90_inv_p"] / ref["B90"]
        ok = (0.85 <= rmed <= 1.15) and (0.85 <= rb90 <= 1.15)
        close = close and ok
        bits.append(
            f"n{n}: Med {m['median_inv_p']:.3f} ({rmed:.3f}×) "
            f"B90 {m['B90_inv_p']:.3f} ({rb90:.3f}×)" + (" OK" if ok else " FAR")
        )
    head = "**CLOSE** to sealed (±15%)." if close else "**NOT close** to sealed reference."
    return head + "\n\n- " + "\n- ".join(bits)


def main():
    ap = argparse.ArgumentParser(description="Nov1st-final skeleton (smoke default)")
    ap.add_argument("--smoke", action="store_true", help="overlap check (default if no --train/--scale1000)")
    ap.add_argument("--train", action="store_true", help="train L on n∈{6,9}; no scale1000 unless also flagged")
    ap.add_argument("--scale1000", action="store_true", help="eval scale1000 B (needs --params or prior train)")
    ap.add_argument("--use-sealed", action="store_true", help="call sealed forward_gadget(unsat) instead of rewrite")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--max_train", type=int, default=48)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--train_ns", default="6,9")
    ap.add_argument("--scale_ns", default="9,12,15,18")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="", help="path to params_L*.pt for --scale1000")
    ap.add_argument("--smoke_L", type=int, default=2)
    args = ap.parse_args()

    # default = smoke only
    if not args.train and not args.scale1000:
        args.smoke = True

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    use_sealed = bool(args.use_sealed)
    train_ns_t = tuple(int(x) for x in args.train_ns.split(",") if x.strip())
    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]

    results = {
        "motif": "sealed_unsat" if use_sealed else "nov1st_final",
        "train_ns": list(train_ns_t),
        "L": args.L,
        "sealed_ref": SEALED_REF,
    }

    if args.smoke:
        results["smoke"] = smoke_overlap(
            n=6, L=args.smoke_L, nf=args.nf, device=args.device, seed=args.seed
        )
        (OUT / "smoke.json").write_text(json.dumps(_jsonable(results["smoke"]), indent=2))
        if not results["smoke"]["pass"]:
            log("SMOKE FAIL — refuse to claim equivalence; fix forward before train")
            write_results_md(results, OUT / "RESULTS.md")
            if args.train or args.scale1000:
                raise SystemExit(2)

    gamma = phi = None
    if args.train:
        log(
            f"=== train L={args.L} ns={train_ns_t} max_train={args.max_train} "
            f"steps={args.steps} sealed={use_sealed} ==="
        )
        best = train_ns(
            W, args.L, args.nf, device,
            train_ns=train_ns_t, steps=args.steps, max_train=args.max_train,
            max_val=args.max_val, seed=args.seed, use_sealed=use_sealed,
        )
        state = best["state"]
        pt = OUT / f"params_L{args.L}.pt"
        torch.save(state, pt)
        gamma, phi = state["gamma"], state["phi"]
        results["train"] = {
            "score": best["score"],
            "step": best["step"],
            "val": best.get("val"),
            "state_path": str(pt),
        }
        log(f"wrote {pt}")
    elif args.params:
        blob = torch.load(args.params, map_location="cpu", weights_only=False)
        gamma, phi = blob["gamma"].float(), blob["phi"].float()
        results["params"] = args.params

    if args.scale1000:
        if gamma is None or phi is None:
            raise SystemExit("--scale1000 needs --train or --params")
        log(f"=== scale1000 L={args.L} B={args.scale_B} ===")
        sc = scale1000_eval(
            W, gamma, phi, args.L, args.nf, device, scale_ns,
            use_sealed=use_sealed, max_count=args.scale_B,
        )
        results["scale1000"] = sc
        results["verdict"] = verdict_scale(sc)

    results_path = OUT / "results.json"
    # merge prior if smoke-only then later train
    if results_path.exists() and not args.train and not args.scale1000:
        pass
    elif results_path.exists() and (args.train or args.scale1000):
        try:
            prev = json.loads(results_path.read_text())
            if "smoke" in prev and "smoke" not in results:
                results["smoke"] = prev["smoke"]
            if "train" in prev and "train" not in results:
                results["train"] = prev["train"]
        except Exception:
            pass
    results_path.write_text(json.dumps(_jsonable(results), indent=2))
    write_results_md(results, OUT / "RESULTS.md")
    log("DONE (skeleton). Plan: Nov/PLAN_Nov1st_final.md")


if __name__ == "__main__":
    main()
