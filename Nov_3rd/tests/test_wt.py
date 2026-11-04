#!/usr/bin/env python3
"""WT correctness: τ=1 recovers scalar cost, graded QAOA matches, verdict gates."""
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

from Nov_3rd.ansatz import make_sim
from Nov_3rd.wt import (
    analyze,
    fwht_unnormalized,
    graded_gamma,
    graded_spectrum,
    normalized_cost,
    phi_from_c,
    qaoa_angles_from_modes,
    qaoa_modes_from_angles,
    qaoa_psucc_graded,
    snap_angles,
    synthesize,
    tau_full,
    tilt_collapsed,
    walsh_degree_components,
    wt_verdict,
)
from beat_params import phi_from_c as phi_ref
from explore_OCT29.lib.qaoa import qaoa_psucc


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"FAIL {name}" + (f" — {detail}" if detail else ""))
    print(f"PASS {name}" + (f" — {detail}" if detail else ""), flush=True)


def _load_one(n: int) -> torch.Tensor:
    import numpy as np

    from Nov_3rd import settings as S

    path = S.OUT / "beat_qaoa_12h" / "train_packs" / f"train_n{n}.npz"
    E = np.load(path, allow_pickle=True)["train_E"]
    return torch.as_tensor(E[0], dtype=torch.float32)


def test_phi_matches_beat_params():
    c1 = torch.tensor([0.2, -0.4])
    c2 = torch.tensor([0.05, 0.01])
    c3 = torch.tensor([-0.02, 0.03])
    err = (phi_from_c(c1, c2, c3) - phi_ref(c1, c2, c3)).abs().max().item()
    check("phi_from_c matches beat_params", err < 1e-6, f"max abs {err:.2e}")


def test_tau1_locked_and_gamma():
    free = torch.tensor([1.5, 0.25, -0.5], requires_grad=True)
    full = tau_full(free)
    check("tau_1 is 1", abs(float(full[0].detach()) - 1.0) < 1e-8)
    g = graded_gamma(torch.tensor([0.4, -0.2]), free)
    check("graded shape", tuple(g.shape) == (2, 4))
    check("degree 1 column is scalar gamma", torch.allclose(g[:, 0], torch.tensor([0.4, -0.2])))
    check("degree 2 is gamma * tau2", torch.allclose(g[:, 1], torch.tensor([0.4, -0.2]) * 1.5))
    check("collapse at ones", tilt_collapsed(torch.ones(3)))
    check("no collapse at 1.1", not tilt_collapsed(torch.tensor([1.1, 1.0, 1.0])))
    check("collapse inside 0.05", tilt_collapsed(torch.tensor([1.04, 0.97, 1.01])))


def test_walsh_degree_span_and_constant():
    E = _load_one(6)
    f = normalized_cost(E, 6)
    comps = walsh_degree_components(f)
    # degree 0
    N = f.numel()
    F = fwht_unnormalized(f)
    dc = fwht_unnormalized(torch.where(torch.arange(N) == 0, F, torch.zeros_like(F))) / float(N)
    recon = dc + comps.sum(dim=0)
    rel = (recon - f).abs().max().item() / f.abs().max().item()
    check("degrees 0..4 reconstruct 4-SAT cost", rel < 1e-4, f"rel {rel:.2e}")
    shifted = walsh_degree_components(f - f.amax())
    rel_s = (shifted - comps).abs().max().item() / comps.abs().max().item()
    check("degree ≥ 1 invariant to subtracting max", rel_s < 1e-4, f"rel {rel_s:.2e}")


def test_tau1_matches_ansatz_scalar():
    E = _load_one(6)
    L = 2
    sim = make_sim(6, L, torch.device("cpu"))
    gamma = torch.tensor([0.31, -0.17])
    c1 = torch.tensor([-0.4, -0.2])
    c2 = torch.tensor([0.02, -0.01])
    c3 = torch.tensor([0.005, 0.004])
    p = torch.tensor(0.15)
    free = torch.ones(3)
    gw, phi = snap_angles(gamma, c1, c2, c3, p, free, 6)
    # scalar path uses the same φ
    out_s = sim.forward_nov3rd(E, gamma, phi)
    out_g = sim.forward_nov3rd(E, gw, phi)
    rel = abs(float(out_s["inv_p"]) - float(out_g["inv_p"])) / float(out_s["inv_p"])
    check("τ=1 SNAP inv_p matches scalar γ", rel < 1e-4, f"rel {rel:.2e}")
    free2 = torch.tensor([1.7, 0.4, 1.3])
    gw2, phi2 = snap_angles(gamma, c1, c2, c3, p, free2, 6)
    out_t = sim.forward_nov3rd(E, gw2, phi2)
    moved = abs(float(out_t["inv_p"]) - float(out_s["inv_p"])) / float(out_s["inv_p"])
    check("τ≠1 changes inv_p", moved > 1e-3, f"rel {moved:.3f}")
    # grid diag matches bitstring spectrum placed on the product grid
    f = sim.cost_bitstring(E)
    spec = graded_spectrum(f, tau_full(free2))
    diag = sim.graded_cost_diag(E, tau_full(free2))
    err = (spec[sim.grid_to_x] - diag).abs().max().item()
    check("graded spectrum matches ansatz diag", err < 1e-5, f"max abs {err:.2e}")


def test_champ_tau1_recovery():
    """Published L=16 power_p champ, one n=6 instance: τ=1 equals the scalar circuit."""
    ckpt = (
        _ROOT
        / "Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/power_p/params_L16_seed0.pt"
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    E = _load_one(6)
    L = int(state["L"])
    sim = make_sim(6, L, torch.device("cpu"))
    gamma = state["gamma"]
    c1, c2, c3 = state["c1"], state["c2"], state["c3"]
    p = torch.tensor(float(state["p"]))
    free = torch.ones(3)
    gw, phi = snap_angles(gamma, c1, c2, c3, p, free, 6)
    out_s = sim.forward_nov3rd(E, gamma, phi)
    out_g = sim.forward_nov3rd(E, gw, phi)
    rel = abs(float(out_s["inv_p"]) - float(out_g["inv_p"])) / max(float(out_s["inv_p"]), 1e-12)
    check("champ τ=1 recovers scalar inv_p", rel < 1e-4, f"rel {rel:.2e} inv={float(out_s['inv_p']):.4f}")


def test_graded_qaoa_tau1():
    E = _load_one(9).reshape(1, -1)
    # a second instance
    import numpy as np

    from Nov_3rd import settings as S

    raw = np.load(S.OUT / "beat_qaoa_12h/train_packs/train_n9.npz", allow_pickle=True)["train_E"]
    E = torch.as_tensor(raw[:4], dtype=torch.float32)
    betas = torch.tensor([0.4, 0.25])
    gammas = torch.tensor([-0.8, -1.1])
    ps = qaoa_psucc(betas, gammas, E, 9)
    pg = qaoa_psucc_graded(betas, gammas, E, 9, torch.ones(4))
    rel = (ps - pg).abs().max().item() / ps.abs().max().item()
    check("graded QAOA τ=1 matches ungraded", rel < 1e-4, f"rel {rel:.2e}")
    pg2 = qaoa_psucc_graded(betas, gammas, E, 9, torch.tensor([1.0, 1.8, 0.3, 1.4]))
    moved = (ps - pg2).abs().max().item() / ps.abs().max().item()
    check("graded QAOA τ≠1 changes p_succ", moved > 1e-3, f"rel {moved:.3f}")


def test_fourier_roundtrip_and_published_endpoint():
    x = torch.tensor([0.2, -0.5, 0.1, 0.4, -0.3, 0.05, 0.2, -0.1], dtype=torch.float64)
    coeff = analyze(x, "dst")
    recon = synthesize(coeff, int(x.numel()), "dst")
    err = (recon - x).abs().max().item()
    check("DST-IV roundtrip", err < 1e-8, f"max abs {err:.2e}")
    # Published Q=6 L=16 Fourier QAOA endpoint (FM-WS params.pt).
    A = torch.tensor(
        [-7.339097023010254, -0.4961847960948944, -0.8876267671585083,
         0.0010843287454918027, -0.21717193722724915, -0.017589250579476357]
    )
    B = torch.tensor(
        [0.9387958645820618, 0.04797294735908508, 0.13868790864944458,
         0.035230573266744614, 0.040093641728162766, -0.00445396127179265]
    )
    betas, gammas = qaoa_angles_from_modes(A, B, 16)
    check(
        "Fourier γ_0 matches published",
        abs(float(gammas[0]) - (-0.7501239776611328)) < 1e-5,
        f"got {float(gammas[0]):.6f}",
    )
    check(
        "Fourier β_last matches published",
        abs(float(betas[-1]) - 0.08028702437877655) < 1e-5,
        f"got {float(betas[-1]):.6f}",
    )
    modes = qaoa_modes_from_angles(betas, gammas, 6)
    # Truncation of an already Q=6 schedule is the schedule itself.
    b2, g2 = qaoa_angles_from_modes(modes["A"], modes["B"], 16)
    check("modes of synthesized angles reproduce them", torch.allclose(g2, gammas, atol=1e-5))
    check("beta modes reproduce", torch.allclose(b2, betas, atol=1e-5))


def test_verdict_gates():
    walsh = 72.8
    champ = 2.579
    fourier = 2.385
    ship = wt_verdict(
        l2_n15=[80.0, 70.0, 90.0],
        walsh_n15=walsh,
        l16=[
            {"snap_n15": 2.10, "graded_n15": 2.30, "tau_dev": 0.4},
            {"snap_n15": 2.50, "graded_n15": 2.55, "tau_dev": 0.01},
        ],
        fourier_n15=fourier,
        champ_n15=champ,
    )
    check("ship when one seed clears", ship["verdict"] == "ship", ship["verdict"])
    killed = wt_verdict(
        l2_n15=[73.0, 74.0],
        walsh_n15=walsh,
        l16=[{"snap_n15": 2.0, "graded_n15": 2.2, "tau_dev": 0.4}],
        fourier_n15=fourier,
        champ_n15=champ,
    )
    check("kill when every L=2 seed is worse", killed["verdict"] == "kill")
    collapsed = wt_verdict(
        l2_n15=[70.0],
        walsh_n15=walsh,
        l16=[{"snap_n15": 2.0, "graded_n15": 2.2, "tau_dev": 0.02}],
        fourier_n15=fourier,
        champ_n15=champ,
    )
    check("kill when every seed collapses", collapsed["verdict"] == "kill")
    # gap without = 2.385-2.579 = -0.194. gap with = 2.40-2.50 = -0.10, which grew,
    # but SNAP does not beat graded. needs-edit (not a gap-kill).
    edit = wt_verdict(
        l2_n15=[70.0],
        walsh_n15=walsh,
        l16=[{"snap_n15": 2.50, "graded_n15": 2.40, "tau_dev": 0.3}],
        fourier_n15=fourier,
        champ_n15=champ,
    )
    check("needs-edit when gap grows but graded QAOA still wins", edit["verdict"] == "needs-edit", edit["verdict"])
    # gap with smaller than gap without → kill
    gap_dead = wt_verdict(
        l2_n15=[70.0],
        walsh_n15=walsh,
        l16=[{"snap_n15": 2.70, "graded_n15": 2.40, "tau_dev": 0.3}],
        fourier_n15=fourier,
        champ_n15=champ,
    )
    check("kill when the gap does not grow", gap_dead["verdict"] == "kill" and gap_dead["gap_kill"])


def main() -> int:
    test_phi_matches_beat_params()
    test_tau1_locked_and_gamma()
    test_walsh_degree_span_and_constant()
    test_tau1_matches_ansatz_scalar()
    test_champ_tau1_recovery()
    test_graded_qaoa_tau1()
    test_fourier_roundtrip_and_published_endpoint()
    test_verdict_gates()
    print("ALL WT TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
