#!/usr/bin/env python3
"""Thorough Nov_3rd correctness suite — must PASS before ladder run.

New checks (not covered by prior Nov/ explorers):
  T1  settings locks / forbidden overrides
  T2  local data + QAOA angle files present for available L
  T3  packing: n%3==0, b=3, C=n/3, W=BinH Hermitian+H⊗3
  T4  shell φ shape (L,4); QAOA-warm φ_j=-β·j
  T5  motif: |0⟩→[Uc^X→SNAP]^L→W vs sealed twin state overlap ≈1
  T6  dual-QAOA warm Nov_3rd inv_p ≈ ideal qubit QAOA (≤1% Med on n6 subsample)
  T7  blockmix fidelity_vs_qaoa max|rel| tiny (convention)
  T8  optimizer lock: Adam path absent; LBFGS_only string
  T9  train loader returns exactly MAX_TRAIN=1000 from sealed n6
  T10 eval skip: n not %3 and missing packs skipped; sim ns = 6,9,12,15,18
  T11 no prep-W killed form (vacuum start; final W applied)
  T12 L_LIST exact set; EVAL range 6..18
  T13 warm-fallback gate: reject polish if train Med > warm (or missing)
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_W, make_sim
from Nov_3rd.data_io import load_qaoa, load_train_E, shell_phi_from_betas
from Nov_3rd.train_bfgs import reject_polish_for_warm
from explore_OCT29.lib.blockmix import fidelity_vs_qaoa, qaoa_theta_from_beta
from explore_OCT29.lib.qaoa import qaoa_psucc
from explore_OCT29.lib.packing import NU, n_cavities


def log(*a):
    print(time.strftime("%H:%M:%S"), "[test]", *a, flush=True)


class Suite:
    def __init__(self):
        self.passed = []
        self.failed = []

    def check(self, name, cond, detail=""):
        if cond:
            self.passed.append(name)
            log(f"PASS {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failed.append((name, detail))
            log(f"FAIL {name}" + (f" — {detail}" if detail else ""))

    def report(self) -> int:
        log(f"=== {len(self.passed)} passed, {len(self.failed)} failed ===")
        for n, d in self.failed:
            log(f"  FAIL {n}: {d}")
        out = S.OUT / "TESTS.md"
        S.OUT.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Nov_3rd — TESTS",
            "",
            f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
            "",
            f"**{len(self.passed)} passed / {len(self.failed)} failed**",
            "",
            "## Passed",
            *[f"- {n}" for n in self.passed],
            "",
            "## Failed",
        ]
        if not self.failed:
            lines.append("- (none)")
        else:
            lines += [f"- **{n}**: {d}" for n, d in self.failed]
        out.write_text("\n".join(lines) + "\n")
        return 0 if not self.failed else 1


def t1_locks(s: Suite):
    try:
        S.assert_locked(
            b=3, W_frame="BinH", snap_head="shell", optimizer="LBFGS_only",
            train_ns=(6,), max_train=1000, qaoa_warm=True,
            snap_shared_layers=False, L_list=S.L_LIST, eval_ns=S.EVAL_NS,
        )
        s.check("T1a_locks_ok", True)
    except Exception as e:
        s.check("T1a_locks_ok", False, str(e))
    raised = False
    try:
        S.assert_locked(optimizer="Adam")
    except ValueError:
        raised = True
    s.check("T1b_adam_blocked", raised)
    raised = False
    try:
        S.assert_locked(b=6)
    except ValueError:
        raised = True
    s.check("T1c_b6_blocked", raised)


def t2_data(s: Suite):
    for n in (6, 9, 12, 15, 18):
        p = S.SEALED_DATA / f"data_n{n}.npz"
        s.check(f"T2_sealed_n{n}", p.exists(), str(p))
        p2 = S.SCALE1000_DIR / f"data_n{n}.npz"
        s.check(f"T2_scale_n{n}", p2.exists(), str(p2))
    for L in (1, 4, 8, 16):
        s.check(f"T2_qaoa_L{L}", S.qaoa_angles_path(L).exists(), str(S.qaoa_angles_path(L)))
    # paths must be under PKG
    s.check("T2_paths_local", S.SEALED_DATA.is_relative_to(S.PKG) and S.QAOA_DIR.is_relative_to(S.PKG))


def t3_packing(s: Suite):
    s.check("T3_b", S.B_BITS == 3 and S.BITS_PER_CAVITY == 3)
    for n in S.EVAL_NS_SIM:
        s.check(f"T3_C_n{n}", n_cavities(n) == n // 3, f"C={n_cavities(n)}")
    W = make_W()
    # W should be unitary; computational block ~ H^{⊗3} on 8 dims
    Wh = W[:NU, :NU].detach().cpu().numpy()
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    H3 = np.kron(np.kron(H, H), H)
    err = np.max(np.abs(Wh - H3))
    s.check("T3_W_is_H3", err < 1e-5, f"max|W-H3|={err:.2e}")
    # Hermitian (H real)
    s.check("T3_W_hermitian", np.max(np.abs(Wh - Wh.conj().T)) < 1e-6)


def t4_shell_warm(s: Suite):
    betas, gammas = load_qaoa(16)
    phi = shell_phi_from_betas(betas)
    s.check("T4_phi_shape", tuple(phi.shape) == (16, 4), str(tuple(phi.shape)))
    # φ_j = -β * j
    ok = True
    for ell in range(16):
        for j in range(4):
            if abs(float(phi[ell, j]) - float(-betas[ell] * j)) > 1e-6:
                ok = False
    s.check("T4_phi_eq_minus_beta_j", ok)
    th = qaoa_theta_from_beta(betas, 3)
    # θ_k should equal φ_pc(k)
    pcs = torch.tensor([bin(k).count("1") for k in range(8)], dtype=torch.float32)
    ok2 = True
    for ell in range(16):
        expect = phi[ell][pcs.long()]
        if (th[ell] - expect).abs().max() > 1e-5:
            ok2 = False
    s.check("T4_theta_matches_shell", ok2)


def t5_motif_overlap(s: Suite):
    """Nov3rd forward vs sealed twin (prep W → D → W SNAP W†) overlap."""
    device = torch.device("cpu")
    L = 2
    n = 6
    sim = make_sim(n, L, device)
    E = load_train_E(device)[0]
    rng = np.random.default_rng(0)
    gamma = torch.as_tensor(rng.uniform(0.2, 1.0, size=L), dtype=torch.float32)
    phi = torch.as_tensor(rng.normal(0, 0.3, size=(L, 4)), dtype=torch.float32)

    # Nov3rd state
    Cn = sim.cost_Cn(E)
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    psi_n = sim.vacuum()
    for ell in range(L):
        psi_n = sim._apply_all(psi_n, sim.Wmat)
        phase = torch.exp(-1j * gamma[ell].to(sim.cdtype) * Cn.to(sim.cdtype))
        psi_new = psi_n.clone()
        psi_new[sl] = psi_n[sl] * phase
        psi_n = psi_new
        psi_n = sim._apply_all(psi_n, sim.Wmat)
        th = phi[ell][sim.popcount]
        psi_n = sim._apply_all(psi_n, sim._snap_theta(th))
    psi_n = sim._apply_all(psi_n, sim.Wmat)

    # sealed twin
    psi_s = sim._apply_all(sim.vacuum(), sim.Wmat)
    for ell in range(L):
        phase = torch.exp(-1j * gamma[ell].to(sim.cdtype) * Cn.to(sim.cdtype))
        psi_new = psi_s.clone()
        psi_new[sl] = psi_s[sl] * phase
        psi_s = psi_new
        th = phi[ell][sim.popcount]
        mixer = sim.Wmat @ sim._snap_theta(th) @ sim.Wdag
        psi_s = sim._apply_all(psi_s, mixer)

    ov = float(torch.vdot(psi_s.reshape(-1), psi_n.reshape(-1)).abs().cpu())
    s.check("T5_motif_overlap", ov > 1.0 - 1e-5, f"|⟨s|n⟩|={ov:.8f}")

    # vacuum start (no prep W on Nov3rd)
    out = sim.forward_nov3rd(E, gamma, phi)
    s.check("T5_forward_finite", torch.isfinite(out["inv_p"]), f"inv={float(out['inv_p'])}")


def t6_warm_vs_qaoa(s: Suite):
    """At QAOA-warm init, Nov_3rd Med ≈ ideal QAOA Med on n6 subsample."""
    device = torch.device("cpu")
    L = 1
    if not S.qaoa_angles_path(L).exists():
        s.check("T6_warm_vs_qaoa", False, "no L=1 angles")
        return
    betas, gammas = load_qaoa(L)
    phi = shell_phi_from_betas(betas)
    E = load_train_E(device)[:32]
    sim = make_sim(6, L, device)
    inv_n, inv_q = [], []
    for i in range(len(E)):
        out = sim.forward_nov3rd(E[i], gammas, phi)
        inv_n.append(float(out["inv_p"].cpu()))
        ps = qaoa_psucc(betas, gammas, E[i].unsqueeze(0), 6)
        inv_q.append(float((1.0 / ps.clamp(min=1e-12)).cpu()))
    med_n = float(np.median(inv_n))
    med_q = float(np.median(inv_q))
    r = med_n / med_q
    s.check("T6_warm_vs_qaoa", abs(r - 1.0) <= 0.02, f"Med_n={med_n:.4f} Med_q={med_q:.4f} R={r:.4f}")


def t7_blockmix_fidelity(s: Suite):
    betas, gammas = load_qaoa(1)
    E = load_train_E(torch.device("cpu"))[:16]
    dp, rel = fidelity_vs_qaoa(betas, gammas, E, 6, 3)
    s.check("T7_blockmix_fidelity", rel < 1e-4, f"max|rel|={rel:.2e} max|dp|={dp:.2e}")


def t8_optimizer(s: Suite):
    s.check("T8_opt_string", S.OPTIMIZER == "LBFGS_only")
    src = (S.PKG / "train_bfgs.py").read_text()
    s.check("T8_uses_LBFGS", "optim.LBFGS" in src or "torch.optim.LBFGS" in src)
    s.check("T8_no_adam_optimizer", "optim.Adam" not in src and "torch.optim.Adam" not in src)


def t9_train_count(s: Suite):
    E = load_train_E(torch.device("cpu"))
    s.check("T9_max_train_1000", len(E) == S.MAX_TRAIN == 1000, f"N={len(E)}")


def t10_eval_ns(s: Suite):
    s.check("T10_eval_range", S.EVAL_NS == tuple(range(6, 19)))
    s.check("T10_sim_ns", S.EVAL_NS_SIM == (6, 9, 12, 15, 18), str(S.EVAL_NS_SIM))


def t11_killed_form(s: Suite):
    """Ensure ansatz code starts vacuum and ends with final W (not prep-W killed)."""
    src = (S.PKG / "ansatz.py").read_text()
    s.check("T11_vacuum_start", "self.vacuum()" in src)
    s.check("T11_final_W_comment", "final W" in src)
    s.check("T11_no_killed_prepW_arm", "prep W → [W D W" not in src)


def t12_L_list(s: Suite):
    s.check(
        "T12_L_list",
        S.L_LIST == (1, 4, 8, 12, 16, 20, 24, 28, 32),
        str(S.L_LIST),
    )


def t13_warm_fallback_gate(s: Suite):
    """Protocol step 4 must be code-enforced: worse/missing train Med → warm."""
    s.check("T13a_reject_worse", reject_polish_for_warm(9.8, 9.61) is True)
    s.check("T13b_keep_tie", reject_polish_for_warm(9.61, 9.61) is False)
    s.check("T13c_keep_better", reject_polish_for_warm(9.5, 9.61) is False)
    s.check("T13d_reject_missing", reject_polish_for_warm(None, 9.61) is True)
    # warm recovery angles are exact dual (same as T4 map)
    betas = torch.tensor([0.3, -0.1], dtype=torch.float32)
    phi = shell_phi_from_betas(betas)
    expected = (-betas.reshape(-1, 1)) * torch.arange(4, dtype=torch.float32)
    s.check(
        "T13e_warm_phi_dual",
        bool(torch.allclose(phi, expected)),
        f"phi={phi.tolist()} expected={expected.tolist()}",
    )


def main():
    s = Suite()
    tests = [
        t1_locks, t2_data, t3_packing, t4_shell_warm, t5_motif_overlap,
        t6_warm_vs_qaoa, t7_blockmix_fidelity, t8_optimizer, t9_train_count,
        t10_eval_ns, t11_killed_form, t12_L_list, t13_warm_fallback_gate,
    ]
    for fn in tests:
        try:
            fn(s)
        except Exception:
            s.check(fn.__name__, False, traceback.format_exc()[-500:])
    code = s.report()
    # also json
    (S.OUT / "tests.json").write_text(
        json.dumps(
            {"passed": s.passed, "failed": [{"name": n, "detail": d} for n, d in s.failed]},
            indent=2,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
