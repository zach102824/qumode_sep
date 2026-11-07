#!/usr/bin/env python3
"""LN-LA unit checks: load feature, Walsh weights, warm-start = champ."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = _ROOT / "Nov_3rd" / "scripts"
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

from Nov_3rd.ansatz import fwht_unnormalized as fwht_torch
from Nov_3rd.ansatz import make_sim
from Nov_3rd.lnla import (
    a_of_n,
    angles_from_champ,
    cavity_loads,
    cavity_touch,
    cost_diag_grid,
    forward_lnla,
    fwht_unnormalized,
    graded_cost_bitstring,
    load_multipliers,
    normalized_cost_bitstring,
    shell_phi,
    subset_multipliers,
)
from beat_params import phi_from_c
from explore_OCT29.lib.sat_gen import gen_instance


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS" if ok else "FAIL"), name, detail)
    if not ok:
        raise SystemExit(1)


def test_loads_and_multipliers() -> None:
    # Two clauses on n=6. Incidences: cav0 gets 3+1, cav1 gets 1+3.
    clause_vars = np.array([[0, 1, 2, 3], [0, 3, 4, 5]], dtype=np.int64)
    rho, r = cavity_loads(6, clause_vars)
    check("rho", np.allclose(rho, [4.0, 4.0]), str(rho))
    check("r_mean_one", abs(float(r.mean()) - 1.0) < 1e-12, str(r))

    rho_u, r_u = cavity_loads(6, np.array([[0, 1, 2, 3]], dtype=np.int64))
    check("uneven_rho", np.allclose(rho_u, [3.0, 1.0]), str(rho_u))
    check("uneven_r", np.allclose(r_u, [1.5, 0.5]), str(r_u))

    # α=0 → every subset weight is 1, including the empty product.
    m0 = subset_multipliers(r_u, 0.0)
    check("alpha0_weights", np.allclose(m0, 1.0), str(m0))

    # Single cavity 0: r0^{-α}. Both cavities: (r0 r1)^{-α/2}.
    m = subset_multipliers(np.array([1.5, 0.5]), 1.0)
    check("single_cav", abs(m[0b01] - (1.5 ** -1.0)) < 1e-12, str(m[0b01]))
    both = (1.5 ** -0.5) * (0.5 ** -0.5)
    check("two_cav", abs(m[0b11] - both) < 1e-12, f"{m[0b11]} vs {both}")
    check("heavy_downweighted", m[0b01] < 1.0 < m[0b10])

    touch = cavity_touch(6)
    # var0 = MSB = bit 5; var3 = bit 2. Mode touches cavities 0 and 1.
    idx = (1 << 5) | (1 << (5 - 3))
    check("touch_both", int(touch[idx]) == 0b11, str(int(touch[idx])))
    check("touch_empty", int(touch[0]) == 0)
    var0 = 1 << 5
    check("touch_var0", int(touch[var0]) == 0b01, str(int(touch[var0])))
    w = load_multipliers(np.array([1.5, 0.5]), 1.0, touch)
    check("mode_var0", abs(w[var0] - (1.5 ** -1.0)) < 1e-12)
    check("mode_both", abs(w[idx] - both) < 1e-12)
    check("mode_empty", abs(w[0] - 1.0) < 1e-12)


def test_fwht_and_alpha0() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    y = fwht_unnormalized(fwht_unnormalized(x)) / 64.0
    check("fwht_roundtrip", np.allclose(y, x, atol=1e-10))
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = fwht_torch(xt).numpy()
    check("fwht_matches_torch", np.allclose(yt, fwht_unnormalized(x), atol=1e-10))

    _m, cl, E = gen_instance(6, 51060000008, compute_E=True)
    f = normalized_cost_bitstring(E, 6)
    g = graded_cost_bitstring(E, 6, np.array([1.2, 0.8]), 0.0)
    check("alpha0_bitstring", np.allclose(f, g, atol=1e-10))
    # r = 1 leaves every Walsh weight at 1 for any α.
    g1 = graded_cost_bitstring(E, 6, np.ones(2), 0.8)
    check("r_ones_is_cost", np.allclose(f, g1, atol=1e-10))


def test_warm_circuit() -> None:
    device = torch.device("cpu")
    n, L = 6, 2
    sim = make_sim(n, L, device)
    _m, cl, E_np = gen_instance(6, 51060000008, compute_E=True)
    E = torch.as_tensor(E_np, dtype=torch.float32)
    gamma = torch.tensor([-0.4, -0.7], dtype=torch.float32)
    phi = shell_phi(
        torch.tensor([-0.3, -0.2]),
        torch.tensor([0.05, -0.04]),
        torch.tensor([0.01, 0.02]),
    )
    rho, r = cavity_loads(6, [v for v, _s in cl])
    check("real_r_mean", abs(float(r.mean()) - 1.0) < 1e-12, str(r))

    champ = sim.forward_nov3rd(E, gamma, phi)
    warm = forward_lnla(sim, E, gamma, phi, r, 0.0, 0.0)
    check(
        "warm_inv_exact",
        torch.equal(champ["inv_p"], warm["inv_p"]),
        f"champ {float(champ['inv_p'])} warm {float(warm['inv_p'])}",
    )
    # Same operator without the shortcut.
    manual = forward_lnla(sim, E, gamma, phi, r, 0.0, 0.0, exact_warm=False)
    check(
        "warm_path_close",
        abs(float(manual["inv_p"]) - float(champ["inv_p"])) < 1e-5,
        f"manual {float(manual['inv_p'])}",
    )
    d0 = sim.cost_Cn(E)
    d1 = cost_diag_grid(sim, E, np.ones(2), 0.6)
    err = float((d0 - d1).abs().max())
    check("diag_r_ones", err < 1e-5, f"max|Δ|={err:.3e}")

    ones = forward_lnla(sim, E, gamma, phi, np.ones(2), 0.6, 0.0, exact_warm=False)
    rel = abs(float(ones["inv_p"]) - float(champ["inv_p"])) / float(champ["inv_p"])
    check("r_ones_circuit", rel < 1e-4, f"rel={rel:.3e}")

    # β ≠ 0 but r = 1 is still the champ shell (per-cavity scale is 1).
    b1 = forward_lnla(sim, E, gamma, phi, np.ones(2), 0.0, 0.4, exact_warm=False)
    rel_b = abs(float(b1["inv_p"]) - float(champ["inv_p"])) / float(champ["inv_p"])
    check("beta_on_r_ones", rel_b < 1e-5, f"rel={rel_b:.3e}")

    # Real n=6 loads are narrow. An extreme ratio checks that the wires move the state.
    r_ext = np.array([1.8, 0.2])
    moved = forward_lnla(sim, E, gamma, phi, r_ext, 0.0, 1.0, exact_warm=False)
    check(
        "beta_moves_inv",
        abs(float(moved["inv_p"]) - float(champ["inv_p"])) > 1e-3,
        f"moved {float(moved['inv_p'])} champ {float(champ['inv_p'])}",
    )
    graded = forward_lnla(sim, E, gamma, phi, r_ext, 1.0, 0.0, exact_warm=False)
    d_ext = cost_diag_grid(sim, E, r_ext, 1.0)
    check("diag_changes", float((d0 - d_ext).abs().max()) > 1e-3)
    check(
        "alpha_moves_inv",
        abs(float(graded["inv_p"]) - float(champ["inv_p"])) > 1e-3,
        f"graded {float(graded['inv_p'])}",
    )
    check("norm", abs(float(graded["norm"]) - 1.0) < 1e-4, str(float(graded["norm"])))
    check("leak_small", float(graded["p_leak"]) < 1e-4, str(float(graded["p_leak"])))


def test_phi_matches_beat_params() -> None:
    c1 = torch.tensor([0.2, -0.1])
    c2 = torch.tensor([0.03, 0.04])
    c3 = torch.tensor([-0.01, 0.02])
    a = torch.allclose(shell_phi(c1, c2, c3), phi_from_c(c1, c2, c3))
    check("shell_phi", bool(a))
    check("a9", abs(a_of_n(9, 0.37) - 1.0) < 1e-12)
    # Frozen L=16 champ: a_9 = 1, so φ is the raw polynomial.
    pt = (
        _ROOT
        / "Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/power_p/params_L16_seed0.pt"
    )
    state = torch.load(pt, map_location="cpu", weights_only=False)
    g, phi, a_n = angles_from_champ(state, 9)
    check("champ_L", int(g.numel()) == 16 and tuple(phi.shape) == (16, 4))
    check("champ_a9", abs(a_n - 1.0) < 1e-12, str(a_n))
    raw = shell_phi(state["c1"], state["c2"], state["c3"])
    check("a9_phi", torch.allclose(phi, raw))


def main() -> None:
    test_loads_and_multipliers()
    test_fwht_and_alpha0()
    test_warm_circuit()
    test_phi_matches_beat_params()
    print("ALL LN-LA TESTS PASSED")


if __name__ == "__main__":
    main()
