#!/usr/bin/env python3
"""Opus probe 6 — spectral mixer W_H · SNAP(θ) · W_H† (ideal W_H).

Arms at L=16:
  A) champion fourier_proj_L16 re-eval (existing flat W + PROJ) on val+test
  B) X-only: θ_k = β · popcount(k)  (depth-matched Gray-X control)
  C) free shells: θ_k = φ_{popcount(k)} for k=0..3, warm-started at champion PROJ

Train B/C on train{9,12}; promote metric: test Med/B90 at n15 & n18 + L×Med.
Orthogonal to probe5 (needs full W_H frame, not flat-only W).
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

from explore_OCT29.lib.ansatz import AnsatzConfig, MultiCavityAnsatz  # noqa: E402
from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.gray_grid import gray, ungray  # noqa: E402
from explore_OCT29.lib.packing import NU  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _PKG / "runs" / "probe6_spectral_mixer"
CHAMP_PARAMS = _PKG / "runs" / "probe5_seq_vocab" / "champ_params.pt"
CHAMP_JSON = _PKG / "runs" / "probe5_seq_vocab" / "champ.json"
W_PATH = _PKG / "runs" / "driver_bakeoff" / "W.pt"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe6]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def popcount(x: int) -> int:
    return int(bin(int(x)).count("1"))


def build_W_H(nf: int = 16, dtype=torch.complex64) -> torch.Tensor:
    """Ideal W_H = Π_gray · H^{⊗3} on levels 0..7; identity pad on 8..nf-1.

    Bit basis |B⟩ maps to Fock |ungray(B)⟩ (matches gray_grid packing).
    """
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
    H3 = H1
    for _ in range(2):
        H3 = torch.kron(H3, H1)
    # H3 is 8×8 in binary order B=0..7
    Pi = torch.zeros(8, 8, dtype=torch.float64)
    for B in range(8):
        f = ungray(B)
        Pi[f, B] = 1.0
    WH8 = Pi @ H3  # Fock ← bits via Gray after H
    W = torch.eye(nf, dtype=torch.float64)
    W[:8, :8] = WH8
    return W.to(dtype=dtype)


def build_flat_W(nf: int = 16, dtype=torch.complex64) -> torch.Tensor:
    blob = torch.load(W_PATH, map_location="cpu", weights_only=False)
    W = blob["W"] if isinstance(blob, dict) else blob
    W = W.to(dtype=dtype)
    if W.shape[0] != nf:
        raise ValueError(f"W nf={W.shape[0]} != {nf}")
    return W


class SpectralAnsatz(MultiCavityAnsatz):
    """|0⟩→W→[U_Φ(γ)→W·diag(e^{-iθ})·W†]^L with shell or X phases."""

    def __init__(self, cfg: AnsatzConfig, W: torch.Tensor, device=None, dtype=torch.complex64):
        super().__init__(cfg, device=device, dtype=dtype)
        Wc = W.detach().to(device=self.lam.device, dtype=self.cdtype)
        if Wc.shape[0] != self.nf:
            raise ValueError(f"W nf={Wc.shape[0]} != sim nf={self.nf}")
        self.register_buffer("Wmat", Wc)
        self.register_buffer("Wdag", Wc.conj().T.contiguous())
        pc = torch.tensor([popcount(k) for k in range(NU)], dtype=torch.long)
        self.register_buffer("popcount", pc)

    def _apply_all(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            psi = self._apply_local(psi, M, c)
        return psi

    def _snap_theta(self, theta8: torch.Tensor) -> torch.Tensor:
        """theta8: length-8 phases on Fock/bit-frame levels 0..7; pad 0."""
        ph = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
        ph = ph.clone()
        ph[:NU] = theta8.to(self.rdtype)
        return torch.diag(torch.exp(-1j * ph.to(self.cdtype)))

    def forward_spectral(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        mode: str,
        beta: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> dict:
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)
        Cn = self.cost_Cn(E)
        for ell in range(L):
            phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
            sl = tuple(slice(0, NU) for _ in range(self.C))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new
            if mode == "proj":
                # θ_0 = β; others 0 — classic PROJ in this W frame
                th = torch.zeros(NU, dtype=self.rdtype, device=self.lam.device)
                th[0] = beta[ell].to(self.rdtype)
            elif mode == "x":
                b = beta[ell].to(self.rdtype)
                th = b * self.popcount.to(self.rdtype)
            elif mode == "shell":
                # phi: (L, 4) → θ_k = φ_{pc(k)}
                ph = phi[ell].to(self.rdtype).reshape(4)
                th = ph[self.popcount]
            else:
                raise ValueError(mode)
            mixer = self.Wmat @ self._snap_theta(th) @ self.Wdag
            psi = self._apply_all(psi, mixer)

        probs = psi.real.square() + psi.imag.square()
        comp_sl = tuple(slice(0, NU) for _ in range(self.C))
        p_comp = probs[comp_sl]
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
            "arm": mode,
        }


def _loss(out, leakpen=0.1):
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


@torch.no_grad()
def eval_arm(n, W, gamma, mode, L, nf, device, split="val", max_count=24,
             beta=None, phi=None):
    pack = load_pack(n)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = SpectralAnsatz(cfg, W, device=device).to(device)
    invs, leaks = [], []
    t0 = time.time()
    for b in range(len(seeds)):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_spectral(Eb, gamma, mode, beta=beta, phi=phi)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n, "split": split, "B": len(invs),
        "median_inv_p": med, "B90_inv_p": b90, "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)), "wall_s": time.time() - t0,
        "L": L, "L_times_Med": L * med, "mode": mode,
    }


def train_arm(
    mode, W, L, nf, device, steps=120, max_train=32, max_val=12, seed=0,
    gamma0=None, beta0=None, lr=2e-3,
):
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(
        (gamma0.detach().clone() if gamma0 is not None
         else torch.linspace(0.3, 1.0, L))
    )
    if mode == "x":
        beta = torch.nn.Parameter(
            (beta0.detach().clone().reshape(L) if beta0 is not None
             else torch.linspace(0.8, 0.2, L))
        )
        phi = None
        params = [gamma, beta]
    elif mode == "shell":
        # warm at champion PROJ: φ0=β_champ, φ1=φ2=φ3=0
        phi = torch.zeros(L, 4)
        if beta0 is not None:
            phi[:, 0] = beta0.detach().reshape(L).to(phi.dtype)
        phi = torch.nn.Parameter(phi)
        beta = None
        params = [gamma, phi]
    else:
        raise ValueError(mode)

    packs = {}
    for n in (9, 12):
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = torch.as_tensor(E, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}
    sims = {
        n: SpectralAnsatz(AnsatzConfig(n=n, L=L, nf=nf), W, device=device).to(device)
        for n in (9, 12)
    }

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n, Eall in packs.items():
            idx = int(rng.integers(0, len(Eall)))
            out = sims[n].forward_spectral(
                Eall[idx], gamma, mode, beta=beta, phi=phi
            )
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            # short LBFGS polish mid/late
            if step in (steps // 2, steps - 1) and sum(p.numel() for p in params) <= 80:
                opt_l = torch.optim.LBFGS(params, lr=0.2, max_iter=4, history_size=8)

                def closure():
                    opt_l.zero_grad()
                    ls = []
                    for n, Eall in packs.items():
                        for j in range(min(8, len(Eall))):
                            o = sims[n].forward_spectral(
                                Eall[j], gamma, mode, beta=beta, phi=phi
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
            for n in (9, 12):
                vals[n] = eval_arm(
                    n, W, gamma.detach(), mode, L, nf, device,
                    split="val", max_count=max_val, beta=None if beta is None else beta.detach(),
                    phi=None if phi is None else phi.detach(),
                )
            sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "beta": None if beta is None else beta.detach().cpu().clone(),
                        "phi": None if phi is None else phi.detach().cpu().clone(),
                        "mode": mode,
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                log(
                    f"train {mode} step {step}: Scomb={sc:.3f}",
                    f"n9:{vals[9]['median_inv_p']:.2f}/{vals[9]['B90_inv_p']:.2f}",
                    f"n12:{vals[12]['median_inv_p']:.2f}/{vals[12]['B90_inv_p']:.2f}",
                )
    return best


def ladder_eval(W, gamma, mode, L, nf, device, beta=None, phi=None, splits=None, counts=None):
    splits = splits or ("val", "test")
    counts = counts or {
        ("val", 9): 24, ("val", 12): 24, ("val", 15): 16, ("val", 18): 16,
        ("test", 9): 40, ("test", 12): 40, ("test", 15): 40, ("test", 18): 16,
    }
    out = {}
    for split in splits:
        out[split] = {}
        for n in (9, 12, 15, 18):
            mc = counts.get((split, n), 16)
            m = eval_arm(
                n, W, gamma, mode, L, nf, device, split=split, max_count=mc,
                beta=beta, phi=phi,
            )
            if m:
                out[split][n] = m
                log(
                    f"eval {mode} {split} n{n}: Med={m['median_inv_p']:.3f}",
                    f"B90={m['B90_inv_p']:.3f} leak={m['mean_leak']:.4f}",
                    f"L*Med={m['L_times_Med']:.1f} B={m['B']}",
                )
    return out


def fit_b_ns(ladder_split, ns=(12, 15, 18)):
    ns_u, meds, b90s = [], [], []
    for n in ns:
        m = ladder_split.get(n)
        if m:
            ns_u.append(n); meds.append(m["median_inv_p"]); b90s.append(m["B90_inv_p"])
    if len(ns_u) < 2:
        return None
    return {"Med": fit_b(ns_u, meds), "B90": fit_b(ns_u, b90s), "ns": ns_u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L, nf = args.L, args.nf

    champ = torch.load(CHAMP_PARAMS, map_location="cpu", weights_only=False)
    gamma_c = champ["gamma"].reshape(L).to(dtype=torch.float32)
    beta_c = champ["beta_proj"].reshape(L).to(dtype=torch.float32)
    log("loaded champ L=", L, "gamma", tuple(gamma_c.shape), "beta", tuple(beta_c.shape))

    W_flat = build_flat_W(nf=nf)
    W_H = build_W_H(nf=nf)
    # sanity: W_H|0⟩ ≈ flat
    flat = torch.zeros(nf, dtype=W_H.dtype); flat[:NU] = 1.0 / math.sqrt(NU)
    col0 = W_H[:, 0]
    F = float(torch.abs(torch.vdot(flat, col0)).real ** 2)
    log(f"W_H|0> fidelity to flat={F:.6f}")

    results = {"L": L, "nf": nf, "seed": args.seed, "steps": args.steps}

    # ---- Arm A: champion re-eval (flat W, PROJ) ----
    log("=== Arm A: champion re-eval (flat W + PROJ) ===")
    results["A_champ"] = ladder_eval(
        W_flat, gamma_c, "proj", L, nf, device, beta=beta_c,
    )
    results["A_champ"]["b_test_12_15_18"] = fit_b_ns(results["A_champ"]["test"])
    results["A_champ"]["b_val_9_12_15"] = fit_b_ns(results["A_champ"]["val"], ns=(9, 12, 15))

    if args.skip_train:
        _write(results)
        return

    # ---- Arm B: X-only on W_H ----
    log("=== Arm B: X-only spectral (W_H, θ=β·pc) ===")
    best_b = train_arm(
        "x", W_H, L, nf, device, steps=args.steps,
        max_train=args.max_train, max_val=args.max_val, seed=args.seed,
        gamma0=gamma_c, beta0=beta_c,
    )
    st = best_b["state"]
    torch.save(st, OUT / "arm_B_x_params.pt")
    results["B_x"] = {
        "train_best_step": best_b["step"],
        "train_score": best_b["score"],
        "train_val": best_b["val"],
        "ladder": ladder_eval(
            W_H, st["gamma"], "x", L, nf, device, beta=st["beta"],
        ),
    }
    results["B_x"]["b_test_12_15_18"] = fit_b_ns(results["B_x"]["ladder"]["test"])

    # ---- Arm C: free shells on W_H, warm at PROJ ----
    log("=== Arm C: free shell phases (W_H, warm PROJ) ===")
    best_c = train_arm(
        "shell", W_H, L, nf, device, steps=args.steps,
        max_train=args.max_train, max_val=args.max_val, seed=args.seed + 1,
        gamma0=gamma_c, beta0=beta_c,
    )
    stc = best_c["state"]
    torch.save(stc, OUT / "arm_C_shell_params.pt")
    results["C_shell"] = {
        "train_best_step": best_c["step"],
        "train_score": best_c["score"],
        "train_val": best_c["val"],
        "ladder": ladder_eval(
            W_H, stc["gamma"], "shell", L, nf, device, phi=stc["phi"],
        ),
    }
    results["C_shell"]["b_test_12_15_18"] = fit_b_ns(results["C_shell"]["ladder"]["test"])

    # promote check: C beats A by ≥1.3× Med on test n15 and n18
    try:
        a15 = results["A_champ"]["test"][15]["median_inv_p"]
        a18 = results["A_champ"]["test"][18]["median_inv_p"]
        c15 = results["C_shell"]["ladder"]["test"][15]["median_inv_p"]
        c18 = results["C_shell"]["ladder"]["test"][18]["median_inv_p"]
        results["promote_C"] = {
            "ratio_n15": a15 / max(c15, 1e-12),
            "ratio_n18": a18 / max(c18, 1e-12),
            "ok": (a15 / max(c15, 1e-12) >= 1.3) and (a18 / max(c18, 1e-12) >= 1.3),
        }
        log("promote_C", results["promote_C"])
    except Exception as e:
        results["promote_C"] = {"error": str(e)}

    _write(results)
    log("DONE →", OUT)


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


def _write(results):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    # markdown summary
    lines = [
        "# Opus probe 6 — spectral mixer W_H·SNAP(θ)·W_H†",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Arms: A=champ PROJ (flat W), B=X-only (W_H), C=shell φ_pc (W_H).",
        "",
    ]

    def dump_arm(name, block, ladder_key=None):
        lines.append(f"## {name}")
        lad = block.get(ladder_key) if ladder_key else block
        if ladder_key is None and "test" in block:
            lad = block
        if not isinstance(lad, dict):
            return
        for split in ("val", "test"):
            if split not in lad:
                continue
            lines.append(f"### {split}")
            lines.append("| n | Med | B90 | leak | L×Med | B |")
            lines.append("|---|-----|-----|------|-------|---|")
            for n in (9, 12, 15, 18):
                m = lad[split].get(n) if isinstance(lad[split], dict) else None
                if not m:
                    continue
                lines.append(
                    f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                    f"{m['mean_leak']:.4f} | {m['L_times_Med']:.1f} | {m['B']} |"
                )
            lines.append("")
        for bk in ("b_test_12_15_18", "b_val_9_12_15"):
            b = block.get(bk)
            if b and isinstance(b, dict) and "Med" in b:
                med = b["Med"]
                lines.append(
                    f"- {bk}: b_Med≈{med.get('b', med.get('slope', '?'))} "
                    f"ns={b.get('ns')}"
                )
        lines.append("")

    if "A_champ" in results:
        dump_arm("A — champion PROJ re-eval", results["A_champ"])
    if "B_x" in results:
        dump_arm("B — X-only on W_H", results["B_x"], ladder_key="ladder")
        if "b_test_12_15_18" in results["B_x"]:
            results["B_x"]  # already
    if "C_shell" in results:
        dump_arm("C — free shells on W_H", results["C_shell"], ladder_key="ladder")
    if "promote_C" in results:
        lines.append(f"## Promote C vs A\n\n`{results['promote_C']}`\n")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", path, "and RESULTS.md")


if __name__ == "__main__":
    main()
