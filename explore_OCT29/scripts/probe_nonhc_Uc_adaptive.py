#!/usr/bin/env python3
"""Adaptive / sample-feedback non-Hc Uc probes (Zach: not secretly encode Hc).

Arms (L=16, BinH W, Arm-C SNAP+D shell):
  1) fb_marginal — mid-circuit soft bit-marginals → classical 1-body exploratory
     diagonal for the *next* cost layer. No clauses in the cost. Not a fixed
     unitary; ≠ exp(-iγ Hc) / clause-projector product / densified E.
  2) fb_break — mid-circuit soft samples → WalkSAT-style per-var break/make
     1-body field (uses instance only to score break on the *sample*, cost is
     still 1-local ≠ densified unsat Hc). Classical detach update.

Score always with true E (Med/B90). Kill if n15 Med ≱ trunc_Hc (~192) or ≫ ArmC.

Writes explore_OCT29/runs/nonhc_Uc_adaptive/{RESULTS.md,results.json,*_params.pt}.
Leaves runs/nonhc_Uc RESULTS intact. ArmC L44 not needed.
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
from explore_OCT29.lib.packing import K_BITS, NU  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    GadgetAnsatz,
    _jsonable,
    fit_b_ns,
    load_warm,
)
from explore_OCT29.scripts.probe6_spectral_mixer import _S, _loss  # noqa: E402
from explore_OCT29.scripts.probe_nonhc_Uc import (  # noqa: E402
    TRUNC_HC_REF,
    PROJ_L16_REF,
    honesty_vs_Hc,
)

OUT = _PKG / "runs" / "nonhc_Uc_adaptive"
SCALE_DIR = _PKG / "data" / "scale1000"

EQUALS_EXP_HC = {
    "fb_marginal": False,
    "fb_break": False,
}

FEATURE_DOC = {
    "fb_marginal": (
        "After each mixer, read computational bit-marginals m_i=P(bit_i=1) from the "
        "state (soft 'shots'). Classical detach → exploratory 1-body field "
        "h_i=−(2m_i−1) (exploratory); next cost diagonal E[x]=∑_i h_i z_i(x) with z=1−2x. "
        "No clauses enter the cost. Sequence is measurement-feedback adaptive → "
        "not a fixed unitary exp(−iγ Hc) / projector product / densified E."
    ),
    "fb_break": (
        "After each mixer, take soft expected assignment (or mode) from bit "
        "marginals; score WalkSAT-style break/make per variable on that assignment "
        "using the instance clauses; classical detach → 1-body field "
        "h_i ∝ (make_i − break_i). Cost diagonal E[x]=∑ h_i z_i — 1-local, "
        "≠ densified unsat count Hc=∑_c C_c. Adaptive across layers from samples."
    ),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[nonhc-adapt]", *a, flush=True)


# ---------------------------------------------------------------------------
# Bit helpers / fields
# ---------------------------------------------------------------------------

def _bit_matrix(n: int) -> np.ndarray:
    """bits[i, x] ∈ {0,1} for var i (MSB=var0), x=0..2^n-1."""
    xs = np.arange(1 << n, dtype=np.int64)
    bits = np.zeros((n, 1 << n), dtype=np.float64)
    for i in range(n):
        bits[i] = ((xs >> (n - 1 - i)) & 1).astype(np.float64)
    return bits


def field_to_E(h: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """E[x] = ∑_i h_i * z_i(x), z=1−2bit; shift min→0."""
    z = 1.0 - 2.0 * bits
    E = (h.astype(np.float64)[:, None] * z).sum(axis=0).astype(np.float32)
    E -= E.min()
    return E


def soft_bit_marginals_from_psi(
    p_comp: torch.Tensor, grid_to_x: torch.Tensor, n: int, bits_t: torch.Tensor
) -> torch.Tensor:
    """m_i = P(bit_i=1) from computational probs. bits_t: (n, 2^n) float."""
    p = p_comp.reshape(-1).to(dtype=torch.float64)
    # renormalize on computational (leak already outside)
    s = p.sum().clamp(min=1e-30)
    p = p / s
    x = grid_to_x.reshape(-1).long()
    # gather bits for occupied x
    # m_i = sum_x p(x) * bit_i(x)
    bx = bits_t[:, x]  # (n, dim_grid)
    m = (bx * p[None, :]).sum(dim=1)  # (n,)
    return m.to(dtype=torch.float32)


def break_make_field(n: int, cls_v, cls_s, x_bits: np.ndarray) -> np.ndarray:
    """WalkSAT-style break/make on a single assignment x_bits[i]∈{0,1}.

    make[v] = # currently-unsat clauses that contain v (flipping may satisfy).
    break[v] = # currently-sat clauses that become unsat if v flips
               (approx: clauses with exactly one true lit which is v).
    Returns h = (make - break) / m, O(1).
    """
    m_cls = max(len(cls_v), 1)
    make = np.zeros(n, dtype=np.float64)
    brk = np.zeros(n, dtype=np.float64)
    for ci in range(len(cls_v)):
        vs = [int(v) for v in cls_v[ci]]
        ss = [int(s) for s in cls_s[ci]]
        # lit true if (sign>0 and bit=1) or (sign<0 and bit=0)
        trues = []
        for v, s in zip(vs, ss):
            bit = int(x_bits[v])
            lit = (bit == 1) if s > 0 else (bit == 0)
            trues.append(lit)
        ntrue = sum(1 for t in trues if t)
        if ntrue == 0:
            for v in vs:
                make[v] += 1.0
        elif ntrue == 1:
            # the unique true lit's var would break this clause if flipped
            for v, s, t in zip(vs, ss, trues):
                if t:
                    brk[v] += 1.0
                    break
    h = (make - brk) / float(m_cls)
    return h.astype(np.float32)


def algebra_adaptive_note() -> dict:
    return {
        "fixed_unitary": False,
        "equals_densified_Uc": False,
        "equals_exp_Hc": False,
        "note": (
            "Cost diagonal at layer ℓ+1 is a classical function of layer-ℓ "
            "computational marginals/samples (detached). Channel is adaptive "
            "measurement-feedback, not a fixed U=exp(-iγ Hc)."
        ),
    }


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------

class AdaptiveNonHcAnsatz(GadgetAnsatz):
    """GadgetAnsatz + mid-circuit classical feedback cost diagonals."""

    def __init__(self, cfg: AnsatzConfig, W: torch.Tensor, device=None, dtype=torch.complex64):
        super().__init__(cfg, W, device=device, dtype=dtype)
        n = cfg.n
        bits = _bit_matrix(n)
        self.register_buffer(
            "bits_all", torch.as_tensor(bits, dtype=torch.float32, device=self.lam.device)
        )
        # optional learnable feature→γ scale (round-adaptive mask)
        # features: [mean|2m-1|, entropy_proxy]; affine → sigmoid gate in (0.25, 1.75)
        self.feat_w = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.feat_b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _p_comp(self, psi: torch.Tensor) -> torch.Tensor:
        probs = psi.real.square() + psi.imag.square()
        sl = tuple(slice(0, NU) for _ in range(self.C))
        return probs[sl]

    def _apply_diag_cost(self, psi: torch.Tensor, gamma: torch.Tensor, E_cost: torch.Tensor) -> torch.Tensor:
        Cn = self.cost_Cn(E_cost)
        phase = torch.exp(-1j * gamma.to(self.cdtype) * Cn.to(self.cdtype))
        sl = tuple(slice(0, NU) for _ in range(self.C))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        return psi_new

    def _mixer(self, psi: torch.Tensor, phi_ell: torch.Tensor) -> torch.Tensor:
        ph = phi_ell.to(self.rdtype).reshape(4)
        th = ph[self.popcount]
        mixer = self.Wmat @ self._snap_theta(th) @ self.Wdag
        return self._apply_all(psi, mixer)

    def _gamma_mask_from_m(self, m: torch.Tensor) -> torch.Tensor:
        """Scalar ∈ ~[0.25, 1.75] from marginal features (not clause indicators)."""
        mag = (2.0 * m - 1.0).abs().mean()
        # binary entropy proxy: mean of h(m)=−m log m −(1−m)log(1−m) / log2
        eps = 1e-6
        mm = m.clamp(eps, 1 - eps)
        ent = (-(mm * mm.log() + (1 - mm) * (1 - mm).log()) / math.log(2.0)).mean()
        feat = torch.stack([mag, ent])
        raw = (self.feat_w * feat).sum() + self.feat_b
        return 0.25 + 1.5 * torch.sigmoid(raw)

    def forward_adaptive(
        self,
        E_true: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        cost_mode: str,
        cls_v=None,
        cls_s=None,
        collect_honesty: bool = False,
    ) -> dict:
        L = self.L
        n = self.cfg.n
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)

        # initial field: zero → flat cost (identity phases) until first feedback
        h = torch.zeros(n, dtype=torch.float32, device=self.lam.device)
        honesty_layers = []
        E_true_np = None
        if collect_honesty:
            E_true_np = E_true.detach().cpu().numpy()

        bits_t = self.bits_all  # (n, 2^n)
        bits_np = bits_t.detach().cpu().numpy()

        for ell in range(L):
            # build diagonal from current classical field
            z = 1.0 - 2.0 * bits_t  # (n, 2^n)
            E_vec = (h[:, None] * z).sum(dim=0).to(torch.float32)
            E_vec = E_vec - E_vec.min()
            # γ mask from previous marginal features (for ell=0, m unknown → mask=1 via h=0 feat)
            if ell == 0:
                g_eff = gamma[ell]
            else:
                # recompute m from last stored — use h magnitude as stand-in already
                # better: store last_m
                g_eff = gamma[ell] * self._gamma_mask_from_m(last_m)

            psi = self._apply_diag_cost(psi, g_eff, E_vec)
            psi = self._mixer(psi, phi[ell])

            # classical feedback from computational marginals
            p_comp = self._p_comp(psi)
            m = soft_bit_marginals_from_psi(p_comp, self.grid_to_x, n, bits_t)
            last_m = m

            if cost_mode == "fb_marginal":
                # exploratory 1-body: HIGH cost when agreeing with current majority
                # E=∑ h z, z=1-2bit → h=-(2m-1) makes majority-aligned x expensive
                h_new = -(2.0 * m - 1.0)
            elif cost_mode == "fb_break":
                # mode assignment from soft marginals; WalkSAT make-break field
                x_bits = (m.detach().cpu().numpy() >= 0.5).astype(np.float64)
                if cls_v is None or cls_s is None:
                    raise ValueError("fb_break needs clauses")
                h_np = break_make_field(n, cls_v, cls_s, x_bits)
                # flip vars with make>break; target = flipped mode on those coords
                flip = (h_np > 0).astype(np.float64)
                target_bit = np.where(flip > 0, 1.0 - x_bits, x_bits)
                # E=∑ h z, z=1-2bit: want low E at target → h = -w * z_target
                w = np.abs(h_np) + 0.05
                h_new = torch.as_tensor(
                    -w * (1.0 - 2.0 * target_bit),
                    dtype=torch.float32,
                    device=self.lam.device,
                )
            else:
                raise ValueError(cost_mode)

            h = h_new.detach()  # classical update — no grad through feedback

            if collect_honesty and ell in (0, L // 2, L - 1):
                E_np = E_vec.detach().cpu().numpy()
                honesty_layers.append(
                    {"layer": ell, **honesty_vs_Hc(E_np, E_true_np)}
                )

        probs = psi.real.square() + psi.imag.square()
        comp_sl = tuple(slice(0, NU) for _ in range(self.C))
        p_comp = probs[comp_sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        E_grid = E_true.to(device=self.lam.device)[self.grid_to_x]
        succ = p_comp[E_grid == 0].sum()
        p_succ = succ.clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        out = {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": inv_p,
            "norm": probs.sum(),
            "arm": cost_mode,
            "equals_exp_Hc": False,
        }
        if collect_honesty:
            out["honesty_layers"] = honesty_layers
        return out


def make_sim(n, W, L, nf, device):
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    return AdaptiveNonHcAnsatz(cfg, W, device=device).to(device)


@torch.no_grad()
def eval_arm(
    n, W, gamma, phi, L, nf, device, cost_mode,
    feat_w=None, feat_b=None,
    split="val", max_count=24, data_dir=None, progress_every=0,
):
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
    if feat_w is not None:
        sim.feat_w.copy_(feat_w.to(sim.feat_w.device))
        sim.feat_b.copy_(feat_b.to(sim.feat_b.device))
    invs, leaks = [], []
    honesty_samples = []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_adaptive(
            Eb, gamma, phi, cost_mode,
            cls_v=cls_v[b], cls_s=cls_s[b],
            collect_honesty=(b < 2),
        )
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if b < 2 and "honesty_layers" in out:
            honesty_samples.append(out["honesty_layers"])
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
        "equals_exp_Hc": False,
        "honesty_samples": honesty_samples,
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
    # share feat params via a reference sim; also optimize them
    ref = make_sim(train_ns[0], W, L, nf, device)
    opt = torch.optim.Adam(
        [gamma, phi, ref.feat_w, ref.feat_b], lr=lr
    )

    packs = {}
    for n in train_ns:
        pack = load_pack(n)
        seeds, cls_v, cls_s, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            seeds, cls_v, cls_s, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = {"E": E, "cls_v": cls_v, "cls_s": cls_s, "N": len(E)}

    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    best = {"score": float("inf"), "state": None, "step": -1, "val": {}}

    for step in range(steps):
        # sync feat params into all sims
        for n in train_ns:
            sims[n].feat_w.data.copy_(ref.feat_w.data)
            sims[n].feat_b.data.copy_(ref.feat_b.data)
            # also need grads — share by using ref for n0 and copying... simpler: optimize each
        opt.zero_grad()
        # wire feat grads: replace sims' feat with ref's for autograd
        for n in train_ns:
            sims[n].feat_w = ref.feat_w
            sims[n].feat_b = ref.feat_b

        losses = []
        for n in train_ns:
            P = packs[n]
            idx = int(rng.integers(0, P["N"]))
            Eb = torch.as_tensor(P["E"][idx], dtype=torch.float32, device=device)
            out = sims[n].forward_adaptive(
                Eb, gamma, phi, cost_mode,
                cls_v=P["cls_v"][idx], cls_s=P["cls_s"][idx],
            )
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
                        o = sims[n].forward_adaptive(
                            Eb, gamma.detach(), phi.detach(), cost_mode,
                            cls_v=P["cls_v"][j], cls_s=P["cls_s"][j],
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
                        "feat_w": ref.feat_w.detach().cpu().clone(),
                        "feat_b": ref.feat_b.detach().cpu().clone(),
                        "L": L,
                        "cost_mode": cost_mode,
                        "train_ns": list(train_ns),
                        "equals_exp_Hc": False,
                    },
                }
            bits = " ".join(
                f"n{n}:{val[n]['Med']:.2f}/{val[n]['B90']:.2f}" for n in train_ns
            )
            log(f"train {cost_mode} step {step}: Scomb={sc:.3f}", bits)

    return best


def scale_eval(W, gamma, phi, L, nf, device, cost_mode, ns, max_count, feat_w=None, feat_b=None):
    out = {}
    for n in ns:
        pe = 50 if n >= 15 else (100 if n >= 12 else 200)
        m = eval_arm(
            n, W, gamma, phi, L, nf, device, cost_mode,
            feat_w=feat_w, feat_b=feat_b,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=pe,
        )
        if m:
            out[n] = m
            log(
                f"scale {cost_mode} n{n}: Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} B={m['B']}"
            )
    return out


def verdict_for(arm: str, sc: dict) -> str:
    m15 = sc.get(15) or sc.get("15")
    if not m15:
        return "incomplete"
    med = m15["median_inv_p"]
    armc = ARM_C_B1000[15]["Med"]
    trunc = TRUNC_HC_REF[15]["Med"]
    if med <= 1.2 * armc:
        return "PROMOTE (≈ArmC — investigate leakage to Hc)"
    if med < trunc:
        return f"WEAK interest (beats trunc_Hc {trunc:.1f} but ≫ ArmC {armc:.3f})"
    return f"KILL (Med={med:.1f} ≱ trunc_Hc {trunc:.1f}; ≫ ArmC {armc:.3f})"


def write_results(results: dict, path: Path):
    lines = [
        "# Adaptive non-Hc Uc probes — verdict",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} (Asia/Shanghai)_",
        "",
        "Ask (Zach): different Uc that does **not** secretly encode Hc.",
        "This track: **measurement / sample-feedback adaptive** cost (not fixed exp(−iγ Hc)).",
        "Prior fixed non-Hc (surrogate_pair / noncomm_xz / wrong_graph) all KILL — left intact in `runs/nonhc_Uc/`.",
        "",
        "## Feature / honesty docs",
        "",
    ]
    for arm, doc in FEATURE_DOC.items():
        lines += [f"### `{arm}`", "", doc, ""]
    if results.get("algebra"):
        lines += [
            "### Algebra / channel note",
            "",
            "```json",
            json.dumps(results["algebra"], indent=2),
            "```",
            "",
        ]
    lines += [
        "## Verdict table (L=16, train{9,12}, W=H^{⊗3} BinH, screen B=200)",
        "",
        "| arm | =exp(Hc)? | fixed Uc? | n9 Med/B90 | n12 | n15 | vs ArmC n15 | vs trunc_Hc n15 | verdict |",
        "|-----|-----------|-----------|------------|-----|-----|-------------|-----------------|---------|",
    ]

    def row(name, equals, fixed, sc):
        cells = []
        for nn in (9, 12, 15):
            m = (sc or {}).get(nn) or (sc or {}).get(str(nn))
            if m and "median_inv_p" in m:
                cells.append(f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}")
            else:
                cells.append("—")
        m15 = (sc or {}).get(15) or (sc or {}).get("15")
        if m15 and "median_inv_p" in m15:
            vs_a = ARM_C_B1000[15]["Med"] / max(m15["median_inv_p"], 1e-12)
            vs_t = TRUNC_HC_REF[15]["Med"] / max(m15["median_inv_p"], 1e-12)
            vsA = f"×{vs_a:.4f}"
            vsT = f"×{vs_t:.4f}"
            verd = verdict_for(name, sc)
        else:
            vsA = vsT = verd = "—"
        lines.append(
            f"| {name} | {equals} | {fixed} | {cells[0]} | {cells[1]} | {cells[2]} | {vsA} | {vsT} | {verd} |"
        )

    row(
        "Arm C densify",
        "YES",
        "YES",
        {n: {"median_inv_p": ARM_C_B1000[n]["Med"], "B90_inv_p": ARM_C_B1000[n]["B90"]} for n in (9, 12, 15)},
    )
    row(
        "projector (=Hc)",
        "YES",
        "YES",
        {n: {"median_inv_p": PROJ_L16_REF[n]["Med"], "B90_inv_p": PROJ_L16_REF[n]["B90"]} for n in (9, 12, 15)},
    )
    row(
        "trunc_Hc (prior)",
        "NO (partial)",
        "YES",
        {n: {"median_inv_p": TRUNC_HC_REF[n]["Med"], "B90_inv_p": TRUNC_HC_REF[n]["B90"]} for n in (9, 12, 15)},
    )
    for arm in results.get("arms_run", []):
        block = results.get(arm) or {}
        sc = block.get("scale1000") or block.get("screen") or {}
        row(arm, "NO", "NO (adaptive)", sc)

    lines += ["", "## Interpretation", ""]
    for arm in results.get("arms_run", []):
        block = results.get(arm) or {}
        sc = block.get("scale1000") or block.get("screen") or {}
        lines.append(
            f"- **{arm}**: {verdict_for(arm, sc)} — =exp(Hc)? **NO**; fixed Uc? **NO**. "
            f"{FEATURE_DOC[arm][:140]}…"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Out: `{OUT}`",
        f"- Script: `scripts/probe_nonhc_Uc_adaptive.py`",
        f"- JSON: `runs/nonhc_Uc_adaptive/results.json`",
        f"- Prior fixed non-Hc: `runs/nonhc_Uc/` (untouched)",
        "",
        "ArmC L44 fair-bar stopped / not needed.",
        "",
    ]
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
    ap.add_argument("--arms", type=str, default="fb_marginal,fb_break")
    ap.add_argument("--screen_B", type=int, default=200)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--scale_ns", type=str, default="9,12,15")
    ap.add_argument("--skip_promote", action="store_true")
    ap.add_argument("--promote_all", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L, nf = args.L, args.nf
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]

    alg = algebra_adaptive_note()
    log("algebra", alg)

    results = {
        "L": L,
        "nf": nf,
        "seed": args.seed,
        "steps": args.steps,
        "train_ns": [9, 12],
        "W": "BinH H^{⊗3}",
        "armC_B1000_ref": ARM_C_B1000,
        "trunc_Hc_ref": TRUNC_HC_REF,
        "projector_L16_ref": PROJ_L16_REF,
        "feature_docs": FEATURE_DOC,
        "algebra": alg,
        "note": (
            "Adaptive sample-feedback Uc; must NOT secretly encode Hc "
            "(no fixed clause projectors / densified E_unsat as the cost unitary)"
        ),
        "arms_run": [],
        "screen_B": args.screen_B,
        "scale_B": args.scale_B,
    }

    W = build_W_BinH(nf=nf)
    g0, p0 = load_warm(L)

    for arm in arms:
        log(f"=== train arm {arm} ===")
        assert EQUALS_EXP_HC[arm] is False
        best = train_arm(
            arm, W, L, nf, device, steps=args.steps,
            max_train=args.max_train, max_val=args.max_val,
            seed=args.seed + (abs(hash(arm)) % 997),
            gamma0=g0, phi0=p0, train_ns=(9, 12),
        )
        st = best["state"]
        torch.save(st, OUT / f"{arm}_params.pt")

        log(f"=== screen B={args.screen_B} {arm} ===")
        screen = scale_eval(
            W, st["gamma"], st["phi"], L, nf, device, arm,
            ns=scale_ns, max_count=args.screen_B,
            feat_w=st.get("feat_w"), feat_b=st.get("feat_b"),
        )
        verd = verdict_for(arm, screen)
        block = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "train_val": {str(k): v for k, v in best["val"].items()},
            "screen": screen,
            "screen_verdict": verd,
            "equals_exp_Hc": False,
            "fixed_unitary": False,
            "feature_doc": FEATURE_DOC[arm],
            "b_screen": fit_b_ns(screen, ns=tuple(scale_ns)),
        }
        do_promote = args.promote_all or (not args.skip_promote and not verd.startswith("KILL"))
        if do_promote and args.scale_B > args.screen_B:
            log(f"=== promote B={args.scale_B} {arm} ===")
            block["scale1000"] = scale_eval(
                W, st["gamma"], st["phi"], L, nf, device, arm,
                ns=scale_ns, max_count=args.scale_B,
                feat_w=st.get("feat_w"), feat_b=st.get("feat_b"),
            )
            block["final_verdict"] = verdict_for(arm, block["scale1000"])
            block["b_scale"] = fit_b_ns(block["scale1000"], ns=tuple(scale_ns))
        else:
            block["scale1000"] = screen
            block["final_verdict"] = verd
            if verd.startswith("KILL"):
                log(f"not promoting {arm}: {verd}")

        results[arm] = block
        results["arms_run"].append(arm)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_results(results, OUT / "RESULTS.md")

    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_results(results, OUT / "RESULTS.md")
    log("DONE", OUT)


if __name__ == "__main__":
    main()
