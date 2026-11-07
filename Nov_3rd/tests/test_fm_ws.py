#!/usr/bin/env python3
"""FM-WS math gates: DST/DCT inverse, champ warm-start, shared exponents, verdict."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = _ROOT / "Nov_3rd" / "scripts"
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

from Nov_3rd.fm_ws import (  # noqa: E402
    analyze,
    degree_mad,
    fmws_verdict,
    gamma_phi_from_modes,
    mode_matrix,
    param_count,
    qaoa_angles_from_modes,
    qaoa_modes_from_angles,
    s_of_n,
    schedules_from_modes,
    synthesize,
    truncation_rel_l2,
    warm_modes_from_champ,
)
from beat_params import phi_from_c as phi_ref  # noqa: E402

CHAMP = (
    _ROOT
    / "Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/power_p/params_L16_seed0.pt"
)
PACK6 = (
    _ROOT
    / "Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/train_packs/train_n6.npz"
)


class T:
    def __init__(self):
        self.failed = []

    def check(self, name: str, cond: bool, detail: str = ""):
        if cond:
            print(f"PASS {name}" + (f" — {detail}" if detail else ""), flush=True)
        else:
            self.failed.append(name)
            print(f"FAIL {name}" + (f" — {detail}" if detail else ""), flush=True)


def _rand_schedule(L: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, generator=g, dtype=torch.float64)


def test_basis_and_inverse(t: T):
    # Closed form, 1-based, matches Opus u_q / v_q.
    L, Q = 8, 3
    U = mode_matrix(L, Q, "dst")
    V = mode_matrix(L, Q, "dct")
    ell, q = 5, 2  # 1-based
    ang = torch.tensor((q - 0.5) * (ell - 0.5) * torch.pi / L, dtype=torch.float64)
    t.check("dst_entry", torch.allclose(U[ell - 1, q - 1], torch.sin(ang)))
    t.check("dct_entry", torch.allclose(V[ell - 1, q - 1], torch.cos(ang)))

    for L in (1, 2, 16, 32):
        M = mode_matrix(L, L, "dct")
        eye = torch.eye(L, dtype=torch.float64)
        gram = M.transpose(0, 1) @ M
        t.check(
            f"dct_orth_L{L}",
            torch.allclose(gram, 0.5 * L * eye, atol=1e-8, rtol=1e-8),
            f"max {float((gram - 0.5 * L * eye).abs().max()):.2e}",
        )
        S = mode_matrix(L, L, "dst")
        gram_s = S.transpose(0, 1) @ S
        t.check(f"dst_orth_L{L}", torch.allclose(gram_s, 0.5 * L * eye, atol=1e-8, rtol=1e-8))
        for kind, seed in (("dct", 1), ("dst", 2)):
            x = _rand_schedule(L, seed + L)
            recon = synthesize(analyze(x, kind), L, kind)
            err = float((recon - x).abs().max())
            t.check(f"roundtrip_{kind}_L{L}", err < 1e-8, f"max {err:.2e}")


def test_pad_and_truncation(t: T):
    coeff = torch.tensor([0.4, -0.2, 0.05], dtype=torch.float64)
    padded = torch.zeros(6, dtype=torch.float64)
    padded[:3] = coeff
    a = synthesize(coeff, 16, "dst")
    b = synthesize(padded, 16, "dst")
    t.check("pad_matches_short_Q", torch.allclose(a, b, atol=1e-12))
    # INTERP changes the basis with L; it is not a repeat of the short schedule.
    short = synthesize(coeff[:1], 4, "dst")
    stretched = synthesize(coeff[:1], 8, "dst")
    t.check("interp_changes_length", int(stretched.numel()) == 8 and int(short.numel()) == 4)
    t.check("interp_is_not_a_repeat", not torch.allclose(stretched[:4], short, atol=1e-6))
    x = _rand_schedule(16, 7)
    rel = truncation_rel_l2(x, 16, "dct")
    t.check("full_Q_truncation_zero", rel < 1e-8, f"rel {rel:.2e}")
    rel_q = truncation_rel_l2(x, 4, "dct")
    t.check("partial_Q_drops_energy", rel_q > 1e-3, f"rel {rel_q:.3f}")


def test_exponents_and_count(t: T):
    t.check("s9_is_one", abs(s_of_n(9) - 1.0) < 1e-15)
    t.check("s_formula", abs(s_of_n(6) - 8.0 / 5.0) < 1e-15)
    t.check("count_q6", param_count(6) == 6 * 6 + 3)
    t.check("count_q4", param_count(4) == 27)

    Q, L = 3, 5
    A = torch.randn(Q, 3)
    B = torch.randn(Q, 3)
    p2 = torch.tensor(0.4)
    p3 = torch.tensor(-0.25)
    lam = torch.tensor(0.7)
    g9, c19, c29, c39 = schedules_from_modes(A, B, p2, p3, lam, L, 9)
    g_raw = synthesize(A, L, "dst")
    c_raw = synthesize(B, L, "dct")
    t.check("n9_kills_scale", torch.allclose(g9[:, :3], g_raw, atol=1e-6))
    t.check("n9_c_unscaled", torch.allclose(torch.stack([c19, c29, c39], 1), c_raw, atol=1e-6))

    g6, c16, c26, c36 = schedules_from_modes(A, B, p2, p3, lam, L, 6)
    s = s_of_n(6)
    t.check("c1_locked_p1", torch.allclose(c16, c19, atol=1e-6))
    t.check("c2_power", torch.allclose(c26, c29 * (s ** 0.4), atol=1e-5))
    t.check("c3_power", torch.allclose(c36, c39 * (s ** -0.25), atol=1e-5))
    # Degree 1: λ p_1 = 0. Degree 2: λ p_2.
    t.check("gamma_d1_no_n", torch.allclose(g6[:, 0], g9[:, 0], atol=1e-6))
    t.check("gamma_d4_tied_to_d1", torch.allclose(g6[:, 3], g6[:, 0], atol=1e-6))
    t.check(
        "gamma_d2_lambda_p",
        torch.allclose(g6[:, 1], g9[:, 1] * (s ** (0.7 * 0.4)), atol=1e-5),
    )
    A_tied = A[:, :1].expand(Q, 3).contiguous()
    g_tied, _, _, _ = schedules_from_modes(A_tied, B, p2, torch.tensor(0.0), torch.tensor(0.0), L, 6)
    t.check("degree_mad_zero_if_tied_and_lambda0", degree_mad(g_tied) < 1e-6)

    # φ matches the shared shell polynomial.
    _, phi = gamma_phi_from_modes(A, B, p2, p3, lam, L, 12)
    _, c1, c2, c3 = schedules_from_modes(A, B, p2, p3, lam, L, 12)
    t.check("phi_matches_beat_params", torch.allclose(phi, phi_ref(c1, c2, c3), atol=1e-6))


def test_qaoa_fourier_roundtrip(t: T):
    L = 16
    g = torch.Generator().manual_seed(3)
    betas = torch.randn(L, generator=g)
    gammas = torch.randn(L, generator=g)
    modes = qaoa_modes_from_angles(betas, gammas, Q=L)
    b2, g2 = qaoa_angles_from_modes(modes["A"].double(), modes["B"].double(), L)
    t.check("qaoa_gamma_dst", torch.allclose(g2.float(), gammas, atol=1e-5))
    t.check("qaoa_beta_dct", torch.allclose(b2.float(), betas, atol=1e-5))
    # Pad: Q<L synthesis equals the zero-padded coefficient vector.
    modes_q = qaoa_modes_from_angles(betas, gammas, Q=4)
    b4, g4 = qaoa_angles_from_modes(modes_q["A"], modes_q["B"], L)
    A_pad = torch.zeros(L)
    B_pad = torch.zeros(L)
    A_pad[:4] = modes_q["A"]
    B_pad[:4] = modes_q["B"]
    bfull, gfull = qaoa_angles_from_modes(A_pad, B_pad, L)
    t.check("qaoa_pad", torch.allclose(b4, bfull) and torch.allclose(g4, gfull))


def test_verdict_rules(t: T):
    champ16 = {6: 1.0, 9: 1.0, 12: 1.0, 15: 2.0}
    q6 = {6: 1.0, 9: 1.0, 12: 1.0, 15: 2.0}
    ship = fmws_verdict(
        q6_l16=q6,
        champ_l16=champ16,
        q6_l32_n15=1.40,  # drop vs 1.50 = 6.67%
        q6_l48_n15=1.40,
        champ_l32_n15=1.50,
        fq_l32_n15=1.55,
        fq_l48_n15=1.55,
    )
    t.check("verdict_ship", ship["verdict"] == "ship", ship["verdict"])

    worse = dict(q6)
    worse[15] = 2.0 * 1.006  # 0.6% worse
    kill_a = fmws_verdict(
        q6_l16=worse,
        champ_l16=champ16,
        q6_l32_n15=1.40,
        q6_l48_n15=1.40,
        champ_l32_n15=1.50,
        fq_l32_n15=1.55,
        fq_l48_n15=1.55,
    )
    t.check("verdict_kill_gateA", kill_a["verdict"] == "kill")

    kill_b = fmws_verdict(
        q6_l16=q6,
        champ_l16=champ16,
        q6_l32_n15=1.49,  # <1% drop
        q6_l48_n15=1.49,
        champ_l32_n15=1.50,
        fq_l32_n15=1.60,
        fq_l48_n15=1.60,
    )
    t.check("verdict_kill_neither_depth", kill_b["verdict"] == "kill")

    one = fmws_verdict(
        q6_l16=q6,
        champ_l16=champ16,
        q6_l32_n15=1.40,
        q6_l48_n15=1.49,
        champ_l32_n15=1.50,
        fq_l32_n15=1.55,
        fq_l48_n15=1.60,
    )
    t.check("verdict_needs_edit_one_depth", one["verdict"] == "needs_edit")

    tied = fmws_verdict(
        q6_l16=q6,
        champ_l16=champ16,
        q6_l32_n15=1.40,
        q6_l48_n15=1.40,
        champ_l32_n15=1.50,
        fq_l32_n15=1.39,  # Fourier QAOA better than FM
        fq_l48_n15=1.39,
    )
    t.check("verdict_needs_edit_qaoa_also_gains", tied["verdict"] == "needs_edit")

    core = dict(q6)
    core[12] = 1.0 * 1.01  # 1% worse at n12, n15 fine
    edit = fmws_verdict(
        q6_l16=core,
        champ_l16=champ16,
        q6_l32_n15=1.40,
        q6_l48_n15=1.40,
        champ_l32_n15=1.50,
        fq_l32_n15=1.55,
        fq_l48_n15=1.55,
    )
    t.check("verdict_needs_edit_core_regression", edit["verdict"] == "needs_edit")
    edge = dict(q6)
    edge[15] = 2.0 * 1.005  # exactly 0.5% worse → still within
    edge_v = fmws_verdict(
        q6_l16=edge,
        champ_l16=champ16,
        q6_l32_n15=1.40,
        q6_l48_n15=1.40,
        champ_l32_n15=1.50,
        fq_l32_n15=1.55,
        fq_l48_n15=1.55,
    )
    t.check("verdict_boundary_within", edge_v["gate_A_q6_l16_n15_within_0.5pct"] and edge_v["verdict"] == "ship")


def test_champ_warm_start(t: T):
    if not CHAMP.exists() or not PACK6.exists():
        t.check("champ_files_present", False, "missing champ or train pack")
        return
    import numpy as np

    from Nov_3rd.ansatz import make_sim

    blob = torch.load(CHAMP, map_location="cpu", weights_only=False)
    gamma = blob["gamma"].float().reshape(-1)
    c1, c2, c3 = blob["c1"].float(), blob["c2"].float(), blob["c3"].float()
    p = float(blob["p"])
    L = int(gamma.numel())
    warm = warm_modes_from_champ(gamma, c1, c2, c3, p, Q=L)
    A, B = warm["A"], warm["B"]
    t.check("warm_A_tied", torch.allclose(A[:, 0], A[:, 1]) and torch.allclose(A[:, 0], A[:, 2]))
    t.check("warm_lambda0", float(warm["lam"]) == 0.0)
    t.check("warm_p_shared", abs(float(warm["p2"]) - p) < 1e-8 and abs(float(warm["p3"]) - p) < 1e-8)
    for n in (6, 9, 15):
        g, cc1, cc2, cc3 = schedules_from_modes(
            A, B, torch.tensor(p), torch.tensor(p), torch.tensor(0.0), L, n
        )
        scale = s_of_n(n) ** p
        t.check(f"warm_gamma_n{n}", torch.allclose(g[:, 0], gamma, atol=1e-5), f"max {(g[:, 0] - gamma).abs().max():.2e}")
        t.check(f"warm_degrees_equal_n{n}", degree_mad(g) < 1e-5)
        t.check(f"warm_c1_n{n}", torch.allclose(cc1, c1, atol=1e-5))
        t.check(f"warm_c2_n{n}", torch.allclose(cc2, c2 * scale, atol=1e-5))
        t.check(f"warm_c3_n{n}", torch.allclose(cc3, c3 * scale, atol=1e-5))

    # Circuit: equal-degree Walsh γ reproduces scalar-γ champ probabilities.
    E = torch.as_tensor(np.load(PACK6)["train_E"][0], dtype=torch.float32)
    sim = make_sim(6, L, torch.device("cpu"))
    with torch.no_grad():
        g_s, phi_s = gamma, phi_ref(c1, c2 * (s_of_n(6) ** p), c3 * (s_of_n(6) ** p))
        g_w, phi_w = gamma_phi_from_modes(
            A, B, torch.tensor(p), torch.tensor(p), torch.tensor(0.0), L, 6
        )
        out_s = sim.forward_nov3rd(E, g_s, phi_s)
        out_w = sim.forward_nov3rd(E, g_w, phi_w)
    ds = abs(float(out_s["inv_p"]) - float(out_w["inv_p"]))
    t.check("warm_forward_matches_scalar_champ", ds < 1e-4, f"|Δinv|={ds:.3e}")

    # 4-SAT: degree ≥5 is absent, degree 4 is not. Zeroing γ_4 must move inv_p,
    # which is why the tie γ_4=γ_1 is load-bearing for champ recovery.
    comps = sim.walsh_degree_components_grid(E)
    e4 = float(comps[3].abs().sum())
    t.check("walsh_degree4_present", e4 > 1e-3, f"l1={e4:.3e}")
    g_drop = g_w.clone()
    g_drop[:, 3] = 0
    with torch.no_grad():
        out_drop = sim.forward_nov3rd(E, g_drop, phi_w)
    dd = abs(float(out_drop["inv_p"]) - float(out_w["inv_p"]))
    t.check("zeroing_degree4_changes_inv", dd > 1e-3, f"|Δinv|={dd:.3e}")

    # Truncation diagnostic (not a gate): how smooth is the champ?
    for Q in (4, 6):
        rel_g = truncation_rel_l2(gamma, Q, "dst")
        rel_c = truncation_rel_l2(c1, Q, "dct")
        print(f"INFO champ L16 Q={Q} relL2 γ={rel_g:.4f} c1={rel_c:.4f}", flush=True)
        t.check(f"trunc_finite_Q{Q}", rel_g == rel_g and rel_c == rel_c)


def test_grad_flows(t: T):
    if not PACK6.exists():
        t.check("grad_pack", False, "missing pack")
        return
    import numpy as np

    from Nov_3rd.ansatz import make_sim

    E = torch.as_tensor(np.load(PACK6)["train_E"][1], dtype=torch.float32)
    Q, L = 4, 8
    A = torch.nn.Parameter(torch.randn(Q, 3) * 0.05)
    B = torch.nn.Parameter(torch.randn(Q, 3) * 0.02)
    p2 = torch.nn.Parameter(torch.tensor(0.3))
    p3 = torch.nn.Parameter(torch.tensor(0.2))
    lam = torch.nn.Parameter(torch.tensor(0.5))
    sim = make_sim(6, L, torch.device("cpu"))
    gamma, phi = gamma_phi_from_modes(A, B, p2, p3, lam, L, 6)
    out = sim.forward_nov3rd(E, gamma, phi)
    out["inv_p"].backward()
    t.check("grad_A", A.grad is not None and torch.isfinite(A.grad).all() and float(A.grad.abs().sum()) > 0)
    t.check("grad_B", B.grad is not None and float(B.grad.abs().sum()) > 0)
    t.check("grad_p2", p2.grad is not None and abs(float(p2.grad)) > 0)
    t.check("grad_lam", lam.grad is not None and abs(float(lam.grad)) > 0)
    # p_1 lock: degree-1 column of A still trains, but a pure n-scale of degree 1 is absent.
    # λ multiplies p_1=0, so changing only the degree-1 n-path is impossible — checked above
    # by gamma_d1_no_n. Here just require the graph did not explode.
    t.check("grad_finite_loss", math_isfinite(float(out["inv_p"].detach())))


def test_batch_matches_single(t: T):
    if not PACK6.exists():
        t.check("batch_pack", False, "missing pack")
        return
    import numpy as np

    from Nov_3rd.ansatz import make_sim

    E = torch.as_tensor(np.load(PACK6)["train_E"][:5], dtype=torch.float32)
    sim = make_sim(6, 8, torch.device("cpu"))
    g = torch.Generator().manual_seed(4)
    gamma_s = torch.randn(8, generator=g)
    phi = torch.randn(8, 4, generator=g) * 0.1
    gamma_w = gamma_s.reshape(8, 1).expand(8, 4).contiguous()
    with torch.no_grad():
        for name, gamma in (("scalar", gamma_s), ("graded", gamma_w)):
            batched = sim.forward_nov3rd_batch(E, gamma, phi)["inv_p"]
            singles = torch.stack([sim.forward_nov3rd(E[i], gamma, phi)["inv_p"] for i in range(len(E))])
            err = float((batched - singles).abs().max())
            t.check(f"batch_matches_{name}", err < 1e-4, f"max|Δinv|={err:.3e}")
    # Gradients through the batch agree with the sum of single backwards.
    A = torch.nn.Parameter(torch.randn(3, 3) * 0.05)
    B = torch.nn.Parameter(torch.randn(3, 3) * 0.02)
    p2 = torch.nn.Parameter(torch.tensor(0.2))
    p3 = torch.nn.Parameter(torch.tensor(-0.1))
    lam = torch.nn.Parameter(torch.tensor(0.3))
    gamma, phi = gamma_phi_from_modes(A, B, p2, p3, lam, 8, 6)
    inv = sim.forward_nov3rd_batch(E, gamma, phi)["inv_p"].sum()
    inv.backward()
    t.check("batch_grad_A", A.grad is not None and float(A.grad.abs().sum()) > 0 and torch.isfinite(A.grad).all())


def math_isfinite(x: float) -> bool:
    return x == x and abs(x) < 1e30


def main() -> int:
    torch.set_num_threads(4)
    t = T()
    test_basis_and_inverse(t)
    test_pad_and_truncation(t)
    test_exponents_and_count(t)
    test_qaoa_fourier_roundtrip(t)
    test_verdict_rules(t)
    test_champ_warm_start(t)
    test_grad_flows(t)
    test_batch_matches_single(t)
    print(f"=== {len(t.failed)} failed ===", flush=True)
    return 1 if t.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
