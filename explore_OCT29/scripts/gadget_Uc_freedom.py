#!/usr/bin/env python3
"""Freedom ladder on projector-Uc (BinH): span clause-weights + local εΔ.

Train {6,9,12} L=16; promote vs champ scale1000 n15 Med 2.754 / n18 4.194.
Also writes two-cavity factor census (no GRAPE).

Outputs: runs/gadget_Uc/freedom/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.ansatz import AnsatzConfig  # noqa: E402
from explore_OCT29.lib.packing import NU  # noqa: E402
from explore_OCT29.lib.sat_gen import clause_count, fast_energies  # noqa: E402
from explore_OCT29.lib.spectral_arm import clause_span, stack_span_E  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_grid_to_x_binary  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    GadgetAnsatz,
    _S,
    _jsonable,
    _loss,
    build_W_BinH,
    make_sim,
)

OUT = _PKG / "runs" / "gadget_Uc" / "freedom"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
BASE = {
    9: {"Med": 1.406, "B90": 3.288},
    12: {"Med": 1.864, "B90": 6.106},
    15: {"Med": 2.754, "B90": 10.244},
    18: {"Med": 4.194, "B90": 24.302},
}
SCALE_DIR = _PKG / "data" / "scale1000"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[free]", *a, flush=True)


# ---------------------------------------------------------------------------
# Two-cavity census
# ---------------------------------------------------------------------------

def local_ks(vars_):
    by = {}
    for x in vars_:
        by.setdefault(int(x) // 3, []).append(int(x) % 3)
    return tuple(sorted(len(v) for v in by.values()))


def two_cavity_census(ns=(9, 12, 15, 18), max_inst=200) -> dict:
    out = {}
    for n in ns:
        pack = load_pack(n)
        # prefer train else val
        split = "train" if f"train_seeds" in pack else "val"
        if f"{split}_cls_v" not in pack:
            if n >= 15:
                # small packs may have only val/test
                split = "val" if f"val_cls_v" in pack else "test"
            else:
                continue
        _, cv, cs, _, _ = _split_arrays(pack, split, max_count=max_inst)
        if len(cv) == 0:
            # try scale1000
            if SCALE_DIR.exists():
                try:
                    pack = load_pack(n, data_dir=SCALE_DIR)
                    split = "scale1000" if "scale1000_cls_v" in pack else "test"
                    _, cv, cs, _, _ = _split_arrays(pack, split, max_count=max_inst)
                except Exception:
                    out[n] = {"error": "no pack"}
                    continue
        spans = Counter()
        ks = Counter()
        n_clauses = 0
        for b in range(len(cv)):
            for vrow in cv[b]:
                vs = [int(x) for x in vrow]
                sp = len({x // 3 for x in vs})
                spans[sp] += 1
                ks[local_ks(vs)] += 1
                n_clauses += 1
        two = spans.get(2, 0)
        out[n] = {
            "split": split,
            "B_inst": len(cv),
            "n_clauses": n_clauses,
            "span_frac": {str(k): v / max(n_clauses, 1) for k, v in sorted(spans.items())},
            "span_counts": {str(k): int(v) for k, v in sorted(spans.items())},
            "local_k_frac": {str(k): v / max(n_clauses, 1) for k, v in ks.most_common()},
            "two_cavity_frac": two / max(n_clauses, 1),
            "two_cavity_splits": {
                "(1,3)": ks.get((1, 3), 0) / max(n_clauses, 1),
                "(2,2)": ks.get((2, 2), 0) / max(n_clauses, 1),
            },
            "note": "C = C_a ⊗ C_b for span-2; GRAPE library = tensor of local projectors.",
        }
        log(
            f"census n{n}: two_cav={out[n]['two_cavity_frac']:.3f} "
            f"(1,3)={out[n]['two_cavity_splits']['(1,3)']:.3f} "
            f"(2,2)={out[n]['two_cavity_splits']['(2,2)']:.3f} spans={out[n]['span_frac']}"
        )
    # verify tensor structure on one (2,2) example
    # C = |pat_a><pat_a| ⊗ |pat_b><pat_b| on 8⊗8
    Pa = np.zeros((8, 8)); Pa[0, 0] = 1  # rank-1 toy
    Pb = np.zeros((8, 8)); Pb[3, 3] = 1
    C = np.kron(Pa, Pb)
    err = float(np.linalg.norm(C @ C - C))
    out["tensor_check"] = {"kron_proj_err": err, "pass": err < 1e-12}
    return out


# ---------------------------------------------------------------------------
# Freedom ansatz
# ---------------------------------------------------------------------------

class FreedomAnsatz(GadgetAnsatz):
    """Projector backbone + span weights and/or local εΔ."""

    def _delta_grid(self, delta8: torch.Tensor) -> torch.Tensor:
        """Δ[f0,...,fC-1] = sum_c δ[f_c] on NU^C computational grid."""
        d = delta8.to(dtype=self.rdtype, device=self.lam.device).reshape(NU)
        # build via successive broadcast add
        # start scalar 0 shaped (1,)*C then add
        shape = [NU] * self.C
        Delta = torch.zeros(shape, dtype=self.rdtype, device=self.lam.device)
        for c in range(self.C):
            view = [1] * self.C
            view[c] = NU
            Delta = Delta + d.reshape(view)
        return Delta

    def forward_freedom(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        mode: str,
        E_span: torch.Tensor | None = None,
        w_span: torch.Tensor | None = None,
        delta8: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> dict:
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        n = self.cfg.n
        m = float(clause_count(n))
        scale = float(n) / m
        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)
        sl = tuple(slice(0, NU) for _ in range(self.C))

        if mode == "span_w":
            assert E_span is not None and w_span is not None
            Es = E_span.to(device=self.lam.device, dtype=self.rdtype)  # (3, 2^n)
            w = w_span.to(dtype=self.rdtype, device=self.lam.device).reshape(3)
            # softplus to keep positive weights
            w = torch.nn.functional.softplus(w)
            grids = []
            for s in range(3):
                grids.append(Es[s][self.grid_to_x] * scale)
            E_grids = torch.stack(grids, dim=0)  # (3, *grid)
            # Φ = sum_s w_s E_grid^s  (γ applied per layer)
            Phi = (w.reshape(3, *([1] * self.C)) * E_grids).sum(dim=0)
            Phi = Phi - Phi.amax()
        elif mode == "eps_delta":
            Cn = self.cost_Cn(E)
            assert delta8 is not None and eps is not None
            # ε = 0.15 * tanh(raw) so bounded
            eps_v = 0.15 * torch.tanh(eps.to(dtype=self.rdtype, device=self.lam.device))
            d8 = delta8.to(dtype=self.rdtype, device=self.lam.device).reshape(NU)
            Delta = self._delta_grid(d8)
            Phi = Cn + eps_v * Delta
            Phi = Phi - Phi.amax()
        elif mode == "span_w_eps":
            assert E_span is not None and w_span is not None
            assert delta8 is not None and eps is not None
            Es = E_span.to(device=self.lam.device, dtype=self.rdtype)
            w = torch.nn.functional.softplus(
                w_span.to(dtype=self.rdtype, device=self.lam.device).reshape(3)
            )
            grids = [Es[s][self.grid_to_x] * scale for s in range(3)]
            E_grids = torch.stack(grids, dim=0)
            Phi = (w.reshape(3, *([1] * self.C)) * E_grids).sum(dim=0)
            eps_v = 0.15 * torch.tanh(eps.to(dtype=self.rdtype, device=self.lam.device))
            Phi = Phi + eps_v * self._delta_grid(delta8)
            Phi = Phi - Phi.amax()
        else:
            raise ValueError(mode)

        for ell in range(L):
            phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Phi.to(self.cdtype))
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
        return {"p_succ": p_succ, "p_leak": p_leak, "inv_p": inv_p, "norm": probs.sum(), "arm": mode}


def make_free(n, W, L, nf, device):
    return FreedomAnsatz(AnsatzConfig(n=n, L=L, nf=nf), W, device=device).to(device)


def preload(n, max_train, device, need_span: bool):
    pack = load_pack(n)
    seeds, cls_v, cls_s, _, E = _split_arrays(pack, "train", max_count=max_train)
    if E is None or len(E) == 0:
        seeds, cls_v, cls_s, _, E = _split_arrays(pack, "val", max_count=max_train)
    entry = {
        "E": torch.as_tensor(E, dtype=torch.float32, device=device),
        "cls_v": cls_v,
        "cls_s": cls_s,
    }
    if need_span:
        spans = []
        for b in range(len(E)):
            spans.append(stack_span_E(n, cls_v[b], cls_s[b]))
        entry["E_span"] = torch.as_tensor(np.stack(spans), dtype=torch.float32, device=device)
    return entry


@torch.no_grad()
def eval_free(n, W, gamma, phi, L, nf, device, mode, w_span=None, delta8=None, eps=None,
              split="val", max_count=24, data_dir=None, progress_every=0):
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        key = "scale1000" if f"scale1000_seeds" in pack else "test"
    seeds, cls_v, cls_s, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    sim = make_free(n, W, L, nf, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(seeds)
    need_span = mode in ("span_w", "span_w_eps")
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        Esp = None
        if need_span:
            Esp = torch.as_tensor(
                stack_span_E(n, cls_v[b], cls_s[b]), dtype=torch.float32, device=device
            )
        out = sim.forward_freedom(
            Eb, gamma, phi, mode, E_span=Esp, w_span=w_span, delta8=delta8, eps=eps
        )
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if progress_every and ((b + 1) % progress_every == 0 or b + 1 == B):
            log(f"eval {mode} n{n} {b+1}/{B} Med≈{np.median(invs):.3f}")
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n, "split": key, "B": B,
        "median_inv_p": med, "B90_inv_p": b90, "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "wall_s": time.time() - t0, "L": L, "mode": mode,
    }


def train_free(
    mode, W, L, nf, device, steps=80, max_train=32, max_val=12, seed=0,
    gamma0=None, phi0=None, train_ns=(6, 9, 12), lr=2e-3,
):
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(
        gamma0.detach().clone() if gamma0 is not None else torch.linspace(0.3, 1.0, L)
    )
    phi = torch.nn.Parameter(
        phi0.detach().clone() if phi0 is not None else torch.zeros(L, 4)
    )
    params = [gamma, phi]
    w_span = delta8 = eps = None
    if mode in ("span_w", "span_w_eps"):
        # softplus(0)≈0.69; init near 1 after softplus → raw≈0.541
        w_span = torch.nn.Parameter(torch.full((3,), 0.541))
        params.append(w_span)
    if mode in ("eps_delta", "span_w_eps"):
        delta8 = torch.nn.Parameter(torch.zeros(NU))
        eps = torch.nn.Parameter(torch.tensor(0.0))  # tanh→0
        params.extend([delta8, eps])

    need_span = mode in ("span_w", "span_w_eps")
    packs = {n: preload(n, max_train, device, need_span) for n in train_ns}
    sims = {n: make_free(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}
    n_lo, n_hi = min(train_ns), max(train_ns)

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n, P in packs.items():
            idx = int(rng.integers(0, len(P["E"])))
            Esp = P["E_span"][idx] if need_span else None
            out = sims[n].forward_freedom(
                P["E"][idx], gamma, phi, mode,
                E_span=Esp, w_span=w_span, delta8=delta8, eps=eps,
            )
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
                            Esp = P["E_span"][j] if need_span else None
                            o = sims[n].forward_freedom(
                                P["E"][j], gamma, phi, mode,
                                E_span=Esp, w_span=w_span, delta8=delta8, eps=eps,
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
                vals[n] = eval_free(
                    n, W, gamma.detach(), phi.detach(), L, nf, device, mode,
                    w_span=None if w_span is None else w_span.detach(),
                    delta8=None if delta8 is None else delta8.detach(),
                    eps=None if eps is None else eps.detach(),
                    split="val", max_count=max_val,
                )
            sc = 0.3 * vals[n_lo]["S"] + 0.3 * vals.get(9, vals[n_lo])["S"] + 0.4 * vals[n_hi]["S"]
            if 9 not in vals:
                sc = 0.4 * vals[n_lo]["S"] + 0.6 * vals[n_hi]["S"]
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "w_span": None if w_span is None else w_span.detach().cpu().clone(),
                        "delta8": None if delta8 is None else delta8.detach().cpu().clone(),
                        "eps": None if eps is None else eps.detach().cpu().clone(),
                        "mode": mode,
                        "W": "BinH",
                        "cost_mode": "projector_freedom",
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns
                )
                extra = ""
                if w_span is not None:
                    ww = torch.nn.functional.softplus(w_span.detach()).cpu().numpy()
                    extra += f" w={np.array2string(ww, precision=3)}"
                if eps is not None:
                    extra += f" eps={float(0.15*torch.tanh(eps.detach())):.4f}"
                log(f"train {mode} step {step}: S={sc:.3f} {bits}{extra}")
    return best


def scale_free(W, st, L, nf, device, mode, ns, max_count=1000):
    out = {}
    for n in ns:
        m = eval_free(
            n, W, st["gamma"], st["phi"], L, nf, device, mode,
            w_span=st.get("w_span"), delta8=st.get("delta8"), eps=st.get("eps"),
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=100 if n < 15 else 50,
        )
        if m:
            out[n] = m
            log(f"scale {mode} n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", default="span_w,eps_delta")
    ap.add_argument("--scale_ns", default="9,12,15,18")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--skip_scale", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)

    log("=== two-cavity census ===")
    census = two_cavity_census()
    (OUT / "two_cavity_census.json").write_text(json.dumps(census, indent=2))

    results = {
        "baseline_champ": BASE,
        "armC": ARM_C_B1000,
        "census": census,
        "arms": {},
    }
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]

    for mode in arms:
        log(f"=== train {mode} ===")
        best = train_free(
            mode, W, args.L, args.nf, device, steps=args.steps, seed=3 + hash(mode) % 50,
            gamma0=st0["gamma"].float(), phi0=st0["phi"].float(),
            train_ns=(6, 9, 12),
        )
        st = best["state"]
        torch.save(st, OUT / f"{mode}_params.pt")
        sc = {}
        if not args.skip_scale:
            sc = scale_free(W, st, args.L, args.nf, device, mode, scale_ns, args.scale_B)
        keep = False
        reasons = []
        m15 = sc.get(15)
        m18 = sc.get(18)
        if m15:
            ok15 = m15["median_inv_p"] < BASE[15]["Med"] * 0.98
            reasons.append(
                f"n15 {m15['median_inv_p']:.3f} {'<' if ok15 else '>='} {BASE[15]['Med']*0.98:.3f}"
            )
        if m18:
            ok18 = m18["median_inv_p"] < BASE[18]["Med"] * 0.98
            reasons.append(
                f"n18 {m18['median_inv_p']:.3f} {'<' if ok18 else '>='} {BASE[18]['Med']*0.98:.3f}"
            )
        if m15 and m18:
            keep = (
                m15["median_inv_p"] < BASE[15]["Med"] * 0.98
                and m18["median_inv_p"] < BASE[18]["Med"] * 0.98
            )
        results["arms"][mode] = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "scale1000": sc,
            "keep": keep,
            "reasons": reasons,
            "w_span_raw": None if st.get("w_span") is None else st["w_span"].tolist(),
            "eps_raw": None if st.get("eps") is None else float(st["eps"]),
        }
        log(f"{mode} keep={keep} {reasons}")
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # RESULTS.md
    lines = [
        "# Freedom ladder on projector-Uc (BinH)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Champ baseline train{6,9,12}: n15 **2.754**/10.244, n18 **4.194**/24.302.",
        "",
        "## Two-cavity projector factors",
        "",
        "Span-2 clauses factor as \(C=C_a\\otimes C_b\) with local \((k_a,k_b)\\in\\{(1,3),(2,2)\\}\). "
        "Still in the finite GRAPE library (tensor of local projectors).",
        "",
        "| n | two-cav frac | (1,3) | (2,2) | span3 | span4 |",
        "|---|--------------|-------|-------|-------|-------|",
    ]
    for n in (9, 12, 15, 18):
        c = census.get(n) or census.get(str(n))
        if not c or "error" in c:
            lines.append(f"| {n} | — | — | — | — | — |")
            continue
        sf = c["span_frac"]
        lines.append(
            f"| {n} | {c['two_cavity_frac']:.3f} | "
            f"{c['two_cavity_splits']['(1,3)']:.3f} | {c['two_cavity_splits']['(2,2)']:.3f} | "
            f"{float(sf.get('3', 0)):.3f} | {float(sf.get('4', 0)):.3f} |"
        )
    tc = census.get("tensor_check", {})
    lines += [
        "",
        f"Tensor check kron projector: err={tc.get('kron_proj_err')} pass={tc.get('pass')}",
        "",
        "## Freedom arms scale1000",
        "",
        "| arm | n9 | n12 | n15 | n18 | keep |",
        "|-----|----|-----|-----|-----|------|",
    ]

    def fmt(arm, n):
        m = (results["arms"].get(arm, {}).get("scale1000") or {}).get(n)
        if not m:
            m = (results["arms"].get(arm, {}).get("scale1000") or {}).get(str(n))
        return f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}" if m else "—"

    for mode in arms:
        block = results["arms"].get(mode, {})
        lines.append(
            f"| {mode} | {fmt(mode,9)} | {fmt(mode,12)} | {fmt(mode,15)} | {fmt(mode,18)} | "
            f"{block.get('keep')} |"
        )
    lines += ["", "## vs champ", ""]
    for mode in arms:
        block = results["arms"].get(mode, {})
        lines.append(f"- **{mode}**: keep={block.get('keep')} — {'; '.join(block.get('reasons') or [])}")
    lines += [
        "",
        "## Verdict",
        "",
    ]
    any_keep = any(results["arms"].get(m, {}).get("keep") for m in arms)
    if any_keep:
        winner = [m for m in arms if results["arms"][m]["keep"]][0]
        lines.append(f"PROMOTE `{winner}` over champ unsat_params.")
    else:
        lines.append("DISCARD freedom arms — keep train{6,9,12} projector champ.")
    lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
