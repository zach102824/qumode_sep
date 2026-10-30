#!/usr/bin/env python3
"""Robust GRAPE probes: cost targets that do NOT equal exp(-iγ Hc).

Arms (L=16, BinH W, Arm C shell mixer):
  mixer_only  — skip cost; only W·SNAP(φ)·W† layers (fully instance-blind)
  random_diag — shared fake E_rand[n] for cost phase; score with true E
  shuffle_E   — E_cost[s]=E[perm[s]] fixed perm per n (scrambled Hc)
  universal_cover — all 26 local projectors/cavity/layer (no clause map; = global phase)
  trunc_Hc    — cost from first m/2 clauses only; score with full E

Also re-cites failed parity / stencil from gadget_Uc. Writes runs/robust_grape/.
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
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import (  # noqa: E402
    build_W_BinH,
    build_grid_to_x_binary,
)
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    GadgetAnsatz,
    ARM_C_B1000,
    _jsonable,
    fit_b_ns,
    load_warm,
)
from explore_OCT29.scripts.probe6_spectral_mixer import _S, _loss  # noqa: E402

OUT = _PKG / "runs" / "robust_grape"
SCALE_DIR = _PKG / "data" / "scale1000"
GADGET_JSON = _PKG / "runs" / "gadget_Uc" / "results.json"
PROJ_L16 = _PKG / "runs" / "gadget_Uc" / "unsat_params_L16_train6912_backup.pt"

# Projector L16 train{6,9,12} scale1000 (from conversation / PLAN)
PROJ_L16_REF = {
    9: {"Med": 1.406, "B90": 3.288},
    12: {"Med": 1.864, "B90": 6.106},
    15: {"Med": 2.754, "B90": 10.244},
    18: {"Med": 4.194, "B90": 24.302},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[robust]", *a, flush=True)


def _perm_for_n(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000 * n)
    return rng.permutation(1 << n).astype(np.int64)



def _universal_cover_mult() -> np.ndarray:
    """Sum of diag of all 26 local Z-projectors on Fock 0..7.

    Analytic: each Fock state sits in exactly 7 patterns → mult = 7 * ones(8).
    Product ∏ U(C) = e^{-i γ 7} I  (global phase only) ≡ mixer_only.
    """
    import itertools
    from explore_OCT29.scripts.gadget_Uc_grape_library import projector_Z
    mult = np.zeros(8, dtype=np.float64)
    for k in (1, 2, 3):
        for bits in itertools.combinations(range(3), k):
            for signs in itertools.product([-1, 1], repeat=k):
                C = projector_Z(list(zip(bits, signs)))
                mult += np.real(np.diag(C))
    return mult.astype(np.float32)


def _trunc_E_from_clauses(n: int, cls_v, cls_s, frac: float = 0.5) -> np.ndarray:
    """Unsat energy from first floor(m*frac) clauses only (not full Hc)."""
    from explore_OCT29.lib.sat_gen import fast_energies
    m = len(cls_v)
    keep = max(1, int(m * frac))
    cl = []
    for i in range(keep):
        v = [int(x) for x in cls_v[i]]
        s = [int(x) for x in cls_s[i]]
        cl.append((v, s))
    return fast_energies(n, cl).astype(np.float32)


def _rand_E_for_n(n: int, seed: int = 7) -> np.ndarray:
    """Shared fake unsat-like energies: integers 0..m clipped, not from clauses."""
    rng = np.random.default_rng(seed + 17 * n)
    dim = 1 << n
    m = int(round(9.93 * n))
    # random Poisson-ish counts; force a few zeros so loss can move
    E = rng.integers(0, max(m // 4, 2), size=dim).astype(np.float32)
    zeros = rng.choice(dim, size=max(dim // 64, 1), replace=False)
    E[zeros] = 0.0
    return E


class RobustAnsatz(GadgetAnsatz):
    """GadgetAnsatz + mixer_only / random_diag / shuffle_E / universal / trunc cost modes."""

    def _apply_universal_cover(self, psi: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Product of all 26 local U(C) per cavity = diag(e^{-iγ·7}) = global phase."""
        if not hasattr(self, "_univ_mult") or self._univ_mult is None:
            m = _universal_cover_mult()
            self.register_buffer(
                "_univ_mult",
                torch.as_tensor(m, dtype=self.rdtype, device=self.lam.device),
            )
        g = gamma.to(dtype=self.rdtype)
        th = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
        th[:NU] = g * self._univ_mult[:NU]
        Ucav = torch.diag(torch.exp(-1j * th.to(self.cdtype)))
        for cav in range(self.C):
            psi = self._apply_local(psi, Ucav, cav)
        return psi

    def forward_robust(
        self,
        E_true: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        cost_mode: str,
        E_cost: torch.Tensor | None = None,
    ) -> dict:
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)

        if cost_mode == "mixer_only":
            Cn = None
        elif cost_mode == "universal_cover":
            # per-cavity phase γ*mult[f]; mult≡7 → global phase (handled in apply)
            Cn = "universal_cover"
        elif cost_mode in ("random_diag", "shuffle_E", "unsat", "trunc_Hc"):
            if E_cost is None:
                raise ValueError(f"{cost_mode} needs E_cost")
            Cn = self.cost_Cn(E_cost)
        elif cost_mode == "stencil_x":
            Cn = None
        else:
            raise ValueError(cost_mode)

        for ell in range(L):
            if cost_mode == "stencil_x":
                psi = self._apply_stencil_x(psi, gamma[ell])
            elif cost_mode == "universal_cover":
                psi = self._apply_universal_cover(psi, gamma[ell])
            elif cost_mode != "mixer_only":
                phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
                sl = tuple(slice(0, NU) for _ in range(self.C))
                psi_new = psi.clone()
                psi_new[sl] = psi[sl] * phase
                psi = psi_new
            # mixer always
            ph = phi[ell].to(self.rdtype).reshape(4)
            th = ph[self.popcount]
            mixer = self.Wmat @ self._snap_theta(th) @ self.Wdag
            psi = self._apply_all(psi, mixer)

        probs = psi.real.square() + psi.imag.square()
        comp_sl = tuple(slice(0, NU) for _ in range(self.C))
        p_comp = probs[comp_sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        E_grid = E_true.to(device=self.lam.device)[self.grid_to_x]
        succ = p_comp[E_grid == 0].sum()
        p_succ = succ.clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        return {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": inv_p,
            "norm": probs.sum(),
            "arm": cost_mode,
        }


def make_sim(n, W, L, nf, device):
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    return RobustAnsatz(cfg, W, device=device).to(device)


def _E_cost_for(mode: str, n: int, E_true: np.ndarray, cache: dict,
                cls_v=None, cls_s=None) -> torch.Tensor:
    if mode == "unsat":
        return torch.as_tensor(E_true, dtype=torch.float32)
    if mode == "random_diag":
        if ("rand", n) not in cache:
            cache[("rand", n)] = _rand_E_for_n(n)
        return torch.as_tensor(cache[("rand", n)], dtype=torch.float32)
    if mode == "shuffle_E":
        if ("perm", n) not in cache:
            cache[("perm", n)] = _perm_for_n(n)
        perm = cache[("perm", n)]
        return torch.as_tensor(E_true[perm], dtype=torch.float32)
    if mode == "trunc_Hc":
        if cls_v is None or cls_s is None:
            raise ValueError("trunc_Hc needs clauses")
        return torch.as_tensor(
            _trunc_E_from_clauses(n, cls_v, cls_s, frac=0.5), dtype=torch.float32
        )
    if mode in ("mixer_only", "stencil_x", "universal_cover"):
        return None
    raise ValueError(mode)


@torch.no_grad()
def eval_arm(
    n, W, gamma, phi, L, nf, device, cost_mode,
    split="val", max_count=24, data_dir=None, progress_every=0, cache=None,
):
    if cache is None:
        cache = {}
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        if "scale1000_seeds" in pack:
            key = "scale1000"
        elif "test_seeds" in pack:
            key = "test"
        else:
            raise RuntimeError(f"n={n}: no split {split}")
    seeds, cls_v, cls_s, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    sim = make_sim(n, W, L, nf, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        Ec = _E_cost_for(
            cost_mode, n, E[b], cache,
            cls_v=cls_v[b] if cost_mode == "trunc_Hc" else None,
            cls_s=cls_s[b] if cost_mode == "trunc_Hc" else None,
        )
        if Ec is not None:
            Ec = Ec.to(device)
        out = sim.forward_robust(Eb, gamma, phi, cost_mode, E_cost=Ec)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if progress_every and ((b + 1) % progress_every == 0 or b + 1 == B):
            rate = (b + 1) / max(time.time() - t0, 1e-9)
            log(
                f"eval {cost_mode} n{n} {b+1}/{B} Med≈{np.median(invs):.3f} "
                f"rate={rate:.2f}/s"
            )
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n,
        "split": key,
        "B": len(invs),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "wall_s": time.time() - t0,
        "L": L,
        "L_times_Med": L * med,
        "cost_mode": cost_mode,
    }


def train_arm(
    cost_mode, W, L, nf, device, steps=80, max_train=32, max_val=12, seed=0,
    gamma0=None, phi0=None, lr=2e-3, train_ns=(9, 12),
):
    train_ns = tuple(int(x) for x in train_ns)
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(
        gamma0.detach().clone() if gamma0 is not None else torch.linspace(0.3, 1.0, L)
    )
    if phi0 is not None:
        phi = torch.nn.Parameter(phi0.detach().clone())
    else:
        phi = torch.nn.Parameter(torch.zeros(L, 4))
    # mixer_only: gamma unused — still keep param for API; optimize phi only
    if cost_mode == "mixer_only":
        params = [phi]
    else:
        params = [gamma, phi]
    opt = torch.optim.Adam(params, lr=lr)
    cache: dict = {}

    packs = {}
    for n in train_ns:
        pack = load_pack(n)
        seeds, cls_v, cls_s, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            seeds, cls_v, cls_s, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = {"E": E, "cls_v": cls_v, "cls_s": cls_s, "N": len(E)}
        # warm cache
        if cost_mode == "random_diag":
            cache[("rand", n)] = _rand_E_for_n(n)
        if cost_mode == "shuffle_E":
            cache[("perm", n)] = _perm_for_n(n)

    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    best = {"score": float("inf"), "state": None, "step": -1, "val": {}}

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n in train_ns:
            P = packs[n]
            idx = int(rng.integers(0, P["N"]))
            Eb = torch.as_tensor(P["E"][idx], dtype=torch.float32, device=device)
            Ec = _E_cost_for(cost_mode, n, P["E"][idx], cache, cls_v=P["cls_v"][idx] if cost_mode=="trunc_Hc" else None, cls_s=P["cls_s"][idx] if cost_mode=="trunc_Hc" else None)
            if Ec is not None:
                Ec = Ec.to(device)
            out = sims[n].forward_robust(Eb, gamma, phi, cost_mode, E_cost=Ec)
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            val = {}
            with torch.no_grad():
                scs = []
                for n in train_ns:
                    invs = []
                    P = packs[n]
                    for j in range(min(max_val, P["N"])):
                        Eb = torch.as_tensor(P["E"][j], dtype=torch.float32, device=device)
                        Ec = _E_cost_for(cost_mode, n, P["E"][j], cache, cls_v=P["cls_v"][j] if cost_mode=="trunc_Hc" else None, cls_s=P["cls_s"][j] if cost_mode=="trunc_Hc" else None)
                        if Ec is not None:
                            Ec = Ec.to(device)
                        o = sims[n].forward_robust(
                            Eb, gamma.detach(), phi.detach(), cost_mode, E_cost=Ec
                        )
                        invs.append(float(o["inv_p"].cpu()))
                    med = float(np.median(invs))
                    b90 = float(np.quantile(invs, 0.9))
                    val[n] = {"Med": med, "B90": b90}
                    scs.append(_S(med, b90))
                sc = float(np.mean(scs))
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": val,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "L": L,
                        "cost_mode": cost_mode,
                        "train_ns": list(train_ns),
                    },
                }
            bits = " ".join(f"n{n}:{val[n]['Med']:.2f}/{val[n]['B90']:.2f}" for n in train_ns)
            log(f"train {cost_mode} step {step}: Scomb={sc:.3f}", bits)

    return best


def ladder_eval(W, gamma, phi, L, nf, device, cost_mode, counts=None, cache=None):
    if cache is None:
        cache = {}
    if counts is None:
        counts = {"val": 24, "test": 24}
    out = {}
    for split, mc in counts.items():
        out[split] = {}
        for n in (9, 12, 15, 18):
            try:
                m = eval_arm(
                    n, W, gamma, phi, L, nf, device, cost_mode,
                    split=split, max_count=mc, cache=cache,
                )
            except FileNotFoundError:
                m = None
            if m:
                out[split][n] = m
                log(
                    f"ladder {cost_mode} {split} n{n}: "
                    f"Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}"
                )
    return out


def scale1000_eval(W, gamma, phi, L, nf, device, cost_mode, ns, max_count=1000, cache=None):
    if cache is None:
        cache = {}
    out = {}
    for n in ns:
        m = eval_arm(
            n, W, gamma, phi, L, nf, device, cost_mode,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=100 if n >= 12 else 200, cache=cache,
        )
        if m:
            out[n] = m
            log(
                f"scale1000 {cost_mode} n{n}: "
                f"Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} B={m['B']}"
            )
    return out


def cite_failed_from_gadget() -> dict:
    if not GADGET_JSON.exists():
        return {}
    d = json.loads(GADGET_JSON.read_text())
    out = {}
    for arm in ("parity", "stencil_x"):
        b = d.get(arm) or {}
        sc = b.get("scale1000") or {}
        row = {}
        for n in (9, 12, 15, 18):
            m = sc.get(str(n)) or sc.get(n)
            if m:
                row[n] = {
                    "median_inv_p": m["median_inv_p"],
                    "B90_inv_p": m["B90_inv_p"],
                    "B": m.get("B"),
                    "source": "gadget_Uc/results.json",
                }
        out[arm] = {"scale1000": row, "equals_exp_Hc": False, "failed": True}
    return out


def write_verdict(results: dict, path: Path):
    lines = [
        "# Robust GRAPE probes — verdict",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Doc: `docs/ROBUST_GRAPE.md`. **Equals exp(-iγ Hc)?** is the key column.",
        "",
        "## Verdict table (L=16)",
        "",
        "| arm | =exp(Hc)? | instance-blind? | n9 Med/B90 | n12 | n15 | vs ArmC |",
        "|-----|-----------|-----------------|------------|-----|-----|---------|",
    ]
    ref = ARM_C_B1000

    def fmt_row(name, equals, blind, sc_dict):
        cells = []
        ratio15 = None
        for n in (9, 12, 15):
            m = None
            if sc_dict:
                m = sc_dict.get(n) or sc_dict.get(str(n))
            if m and "median_inv_p" in m:
                cells.append(f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}")
                if n == 15:
                    ratio15 = ref[15]["Med"] / max(m["median_inv_p"], 1e-12)
            else:
                cells.append("—")
        vs = f"×{ratio15:.3f}" if ratio15 is not None else "—"
        lines.append(
            f"| {name} | {equals} | {blind} | {cells[0]} | {cells[1]} | {cells[2]} | {vs} |"
        )

    # refs
    fmt_row(
        "Arm C (Hc densify)",
        "YES",
        "no",
        {n: {"median_inv_p": ref[n]["Med"], "B90_inv_p": ref[n]["B90"]} for n in ref},
    )
    fmt_row(
        "projector (=Hc)",
        "YES",
        "no*",
        {n: {"median_inv_p": PROJ_L16_REF[n]["Med"], "B90_inv_p": PROJ_L16_REF[n]["B90"]} for n in PROJ_L16_REF},
    )

    failed = results.get("failed_cited") or {}
    for arm in ("parity", "stencil_x"):
        sc = (failed.get(arm) or {}).get("scale1000") or {}
        fmt_row(f"{arm} (FAILED)", "NO", "stencil=yes / parity=no", sc)

    for arm in results.get("arms_run", []):
        block = results.get(arm) or {}
        sc = block.get("scale1000") or {}
        # fallback test ladder
        if not sc:
            sc = (block.get("ladder") or {}).get("test") or {}
        equals = "NO"
        blind = {
            "mixer_only": "YES",
            "random_diag": "YES",
            "shuffle_E": "scrambled",
            "universal_cover": "YES(=global phase)",
            "trunc_Hc": "partial Hc",
            "stencil_x": "YES",
            "unsat": "no",
        }.get(arm, "?")
        fmt_row(arm, equals, blind, sc)

    lines += [
        "",
        "\\*projector: classical clause→C schedule still instance-aware; GRAPE shapes frozen.",
        "",
        "## Interpretation",
        "",
    ]
    # auto verdict text
    robust_ok = []
    robust_fail = []
    for arm in results.get("arms_run", []):
        block = results.get(arm) or {}
        sc = block.get("scale1000") or (block.get("ladder") or {}).get("test") or {}
        m15 = sc.get(15) or sc.get("15")
        if not m15:
            continue
        med = m15["median_inv_p"] if isinstance(m15, dict) else None
        if med is None:
            continue
        if med <= 1.15 * ref[15]["Med"]:
            robust_ok.append(arm)
        else:
            robust_fail.append((arm, med))
    if not robust_ok and robust_fail:
        lines.append(
            "- **No truly Hc-free arm matches Arm C.** Mixer-only / random / shuffle "
            "stay large-Med → GRAPE is not inventing SAT structure without cost."
        )
        lines.append(
            "- **Hardware-robust story that works:** frozen clause-projector library "
            "(= Uc) + shared gamma/phi. That **renames** Hc, does not avoid it."
        )
    elif robust_ok:
        lines.append(f"- Unexpected: competitive robust arms: {robust_ok}")
    for arm, med in robust_fail:
        lines.append(f"- `{arm}` n15 Med={med:.1f} ≫ Arm C {ref[15]['Med']:.3f} → discard for SAT.")
    lines += [
        "",
        "## Params / artifacts",
        "",
        f"- Out: `{OUT}`",
        f"- Plan doc: `docs/ROBUST_GRAPE.md`",
        "",
    ]
    for arm in results.get("arms_run", []):
        p = OUT / f"{arm}_params.pt"
        if p.exists():
            lines.append(f"- `{arm}`: `{p.relative_to(_PKG)}`")
    path.write_text("\n".join(lines) + "\n")
    log("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--scale_ns", type=str, default="9,12,15")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--skip_scale", action="store_true")
    ap.add_argument(
        "--arms",
        type=str,
        default="mixer_only,random_diag,shuffle_E",
    )
    ap.add_argument("--ladder_B", type=int, default=24)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L, nf = args.L, args.nf
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    failed = cite_failed_from_gadget()
    results = {
        "L": L,
        "nf": nf,
        "seed": args.seed,
        "steps": args.steps,
        "train_ns": [9, 12],
        "armC_B1000_ref": ARM_C_B1000,
        "projector_L16_ref": PROJ_L16_REF,
        "failed_cited": failed,
        "note": "probes must NOT equal exp(-iγ Hc); see docs/ROBUST_GRAPE.md",
        "arms_run": [],
    }

    W = build_W_BinH(nf=nf)
    g0, p0 = load_warm(L)

    for arm in arms:
        log(f"=== train arm {arm} ===")
        cache: dict = {}
        best = train_arm(
            arm, W, L, nf, device, steps=args.steps,
            max_train=args.max_train, max_val=args.max_val,
            seed=args.seed + (abs(hash(arm)) % 997),
            gamma0=g0, phi0=p0, train_ns=(9, 12),
        )
        st = best["state"]
        torch.save(st, OUT / f"{arm}_params.pt")
        log(f"ladder {arm}")
        lad = ladder_eval(
            W, st["gamma"], st["phi"], L, nf, device, arm,
            counts={"val": args.ladder_B, "test": args.ladder_B},
            cache=cache,
        )
        block = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "train_val": best["val"],
            "ladder": lad,
            "b_test_12_15_18": fit_b_ns(lad.get("test", {})),
            "equals_exp_Hc": False,
        }
        if not args.skip_scale:
            scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]
            log(f"scale1000 {arm} ns={scale_ns} B={args.scale_B}")
            block["scale1000"] = scale1000_eval(
                W, st["gamma"], st["phi"], L, nf, device, arm,
                ns=scale_ns, max_count=args.scale_B, cache=cache,
            )
            block["b_scale"] = fit_b_ns(block["scale1000"], ns=tuple(scale_ns))
        results[arm] = block
        results["arms_run"].append(arm)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_verdict(results, OUT / "RESULTS.md")

    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_verdict(results, OUT / "RESULTS.md")
    # also refresh docs pointer
    log("DONE", OUT)


if __name__ == "__main__":
    main()
