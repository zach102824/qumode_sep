#!/usr/bin/env python3
"""QN-RS math gates: Fourier warm start, a(n), residual norm, bootstrap, verdict."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.qn_rs import (  # noqa: E402
    a_of_n,
    combine_verdicts,
    load_fourier_modes,
    mode_matrix,
    paired_median_ci,
    param_count,
    qnrs_verdict,
    residual_ratio,
    schedules,
    seed_passes,
    synthesize,
)

REF16 = _ROOT / "Nov_3rd/runs/beat_qaoa_12h/qn_rs/fourier_qaoa_ref/Q6/L16"


class T:
    def __init__(self):
        self.failed = []

    def check(self, name: str, cond: bool, detail: str = ""):
        if cond:
            print(f"PASS {name}" + (f" — {detail}" if detail else ""), flush=True)
        else:
            self.failed.append(name)
            print(f"FAIL {name}" + (f" — {detail}" if detail else ""), flush=True)


def test_basis(t: T):
    L, Q = 16, 6
    U = mode_matrix(L, Q, "dst", dtype=torch.float64)
    V = mode_matrix(L, Q, "dct", dtype=torch.float64)
    ell, q = 7, 3
    ang = (q - 0.5) * (ell - 0.5) * math.pi / L
    t.check("dst_entry", abs(float(U[ell - 1, q - 1]) - math.sin(ang)) < 1e-12)
    t.check("dct_entry", abs(float(V[ell - 1, q - 1]) - math.cos(ang)) < 1e-12)
    # t_ℓ = (ℓ−½)/L, so the argument is (q−½) π t_ℓ.
    t_ell = (ell - 0.5) / L
    t.check("matches_t_ell", abs(ang - (q - 0.5) * math.pi * t_ell) < 1e-15)


def test_warm_is_fourier_qaoa(t: T):
    u, v = load_fourier_modes(REF16 / "params.pt", 6)
    L = 16
    sch = schedules(u, v, torch.zeros(6), torch.zeros(6), 0.0, L, 15)
    k = torch.arange(4, dtype=torch.float32)
    phi_lin = -sch["beta"].reshape(-1, 1) * k.reshape(1, -1)
    err = float((sch["phi"] - phi_lin).abs().max())
    t.check("sw0_phi_is_minus_beta_k", err < 1e-6, f"max {err:.2e}")
    t.check("gamma_matches_dst", torch.allclose(sch["gamma"], synthesize(u, L, "dst")))
    t.check("beta_matches_dct", torch.allclose(sch["beta"], synthesize(v, L, "dct")))
    ang = json.loads((REF16 / "angles.json").read_text())["best"]
    g_ref = torch.tensor(ang["gammas"], dtype=torch.float32)
    b_ref = torch.tensor(ang["betas"], dtype=torch.float32)
    t.check("gamma_matches_saved_angles", torch.allclose(sch["gamma"], g_ref, atol=1e-5, rtol=1e-5))
    t.check("beta_matches_saved_angles", torch.allclose(sch["beta"], b_ref, atol=1e-5, rtol=1e-5))
    # a(n) does not move the circuit while r2 = r3 = 0.
    sch_p = schedules(u, v, torch.zeros(6), torch.zeros(6), 1.7, L, 15)
    t.check("p_inert_at_zero_residual", torch.allclose(sch["phi"], sch_p["phi"], atol=1e-6))
    t.check("param_count_25", param_count(6, 6) == 25)


def test_a_and_residual(t: T):
    for p in (-0.4, 0.0, 1.25):
        a9 = float(a_of_n(9, torch.tensor(p)))
        t.check(f"a9_p{p}", abs(a9 - 1.0) < 1e-6, f"{a9}")
    p = torch.tensor(0.5, dtype=torch.float32)
    a15 = a_of_n(15, p)
    want = (8.0 / 14.0) ** 0.5
    t.check("a15", abs(float(a15) - want) < 1e-6, f"{float(a15)} vs {want}")
    beta = torch.tensor([3.0, 0.0, 0.0, 0.0])
    r2 = torch.tensor([0.3, 0.0, 0.0, 0.0])
    r3 = torch.zeros(4)
    # ‖a(r2,r3)‖ = 0.5 * 0.3, ‖β‖ = 3 → ratio 0.05
    ratio = residual_ratio(beta, r2, r3, 0.5)
    t.check("residual_ratio", abs(ratio - 0.05) < 1e-6, f"{ratio}")
    t.check("zero_residual", residual_ratio(beta, torch.zeros(4), torch.zeros(4), 2.0) == 0.0)


def test_bootstrap_and_verdict(t: T):
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    same = paired_median_ci(base, base.copy(), n_boot=400, seed=1)
    t.check("identical_ci_includes_0", (not same["excludes_0"]) and abs(same["delta_med"]) < 1e-12)
    better = base - 0.5
    beat = paired_median_ci(better, base, n_boot=400, seed=2)
    t.check("shift_ci_excludes_0", beat["ours_better"] and beat["ci_hi"] < 0.0, str(beat))

    def row(better_flag: bool, ratio: float) -> dict:
        if better_flag:
            boot = {"ours_better": True, "excludes_0": True, "delta_med": -0.05, "ci_lo": -0.08, "ci_hi": -0.02}
            med, fq = 2.30, 2.385
        else:
            boot = {"ours_better": False, "excludes_0": False, "delta_med": 0.01, "ci_lo": -0.02, "ci_hi": 0.04}
            med, fq = 2.40, 2.385
        return {
            "med_n12": med,
            "med_n15": med,
            "fq_n12": fq,
            "fq_n15": fq,
            "boot_n12": boot,
            "boot_n15": boot,
            "residual_ratio": ratio,
        }

    t.check("seed_pass_needs_residual", seed_passes(row(True, 0.2)["boot_n12"], row(True, 0.2)["boot_n15"], 0.2))
    t.check("seed_fail_small_residual", not seed_passes(row(True, 0.05)["boot_n12"], row(True, 0.05)["boot_n15"], 0.05))
    v_pass = qnrs_verdict([row(True, 0.2), row(True, 0.15), row(False, 0.0)], depth_tag="L16")
    t.check("two_of_three_pass", v_pass["verdict"] == "pass" and v_pass["n_pass_seeds"] == 2)
    v_kill = qnrs_verdict([row(False, 0.3), row(False, 0.3), row(False, 0.3)], depth_tag="L16")
    t.check("no_n15_gain_kills", v_kill["verdict"] == "kill")
    dead = row(True, 0.001)
    v_dead = qnrs_verdict([dead, dead, dead], depth_tag="L16")
    t.check("residual_collapse_kills", v_dead["verdict"] == "kill")
    v_one = qnrs_verdict([row(True, 0.2), row(False, 0.2), row(False, 0.2)], depth_tag="L16")
    t.check("one_seed_needs_edit", v_one["verdict"] == "needs_edit")
    t.check("both_depths_ship", combine_verdicts(v_pass, v_pass) == "ship")
    t.check("l16_pass_l32_kill_needs_edit", combine_verdicts(v_pass, v_kill) == "needs_edit")
    t.check("l16_kill_stays_kill", combine_verdicts(v_kill, None) == "kill")


def test_batch_and_grad(t: T):
    from Nov_3rd.ansatz import make_sim
    from explore_OCT29.lib.qaoa import qaoa_psucc

    torch.manual_seed(0)
    n, L = 6, 4
    sim = make_sim(n, L, torch.device("cpu"))
    # One satisfiable-looking cost: a few zero-energy strings, rest positive.
    E = torch.randint(0, 6, (3, 1 << n), dtype=torch.float32)
    E[:, 0] = 0
    E[:, 5] = 0
    u = torch.tensor([0.2, -0.1, 0.05, 0.01])
    v = torch.tensor([0.4, 0.1, -0.05, 0.02])
    s = torch.nn.Parameter(torch.tensor([0.05, -0.02, 0.0, 0.01]))
    w = torch.nn.Parameter(torch.zeros(4))
    p = torch.nn.Parameter(torch.tensor(0.3))
    sch = schedules(u, v, s, w, p, L, n)
    single = []
    for i in range(3):
        single.append(sim.forward_nov3rd(E[i], sch["gamma"], sch["phi"])["inv_p"])
    single = torch.stack(single)
    batched = sim.forward_nov3rd_batch(E, sch["gamma"], sch["phi"])["inv_p"]
    err = float((single - batched).detach().abs().max())
    t.check("batch_matches_single", err < 1e-5, f"max {err:.2e}")

    # s = w = 0 reproduces qubit QAOA on the same angles.
    sch0 = schedules(u, v, torch.zeros(4), torch.zeros(4), 0.0, L, n)
    snap = sim.forward_nov3rd_batch(E, sch0["gamma"], sch0["phi"])["inv_p"]
    q = qaoa_psucc(sch0["beta"].detach(), sch0["gamma"].detach(), E, n)
    qinv = 1.0 / q.clamp(min=1e-12)
    rel = float(((snap - qinv).abs() / qinv.clamp(min=1e-6)).max())
    t.check("sw0_matches_qaoa", rel < 1e-4, f"max rel {rel:.2e}")

    loss = batched.mean()
    loss.backward()
    t.check("grad_reaches_s", s.grad is not None and float(s.grad.norm()) > 0.0, f"grad {None if s.grad is None else float(s.grad.norm()):.3e}")
    t.check("grad_reaches_p", p.grad is not None and abs(float(p.grad)) > 0.0)


def main():
    t = T()
    test_basis(t)
    test_warm_is_fourier_qaoa(t)
    test_a_and_residual(t)
    test_bootstrap_and_verdict(t)
    test_batch_and_grad(t)
    print(f"=== {len(t.failed)} failed ===", flush=True)
    return 1 if t.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
