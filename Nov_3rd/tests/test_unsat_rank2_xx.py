"""Histogram score and the μ=−0.25 verdict. No training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.quantum_h_transfer import Circuit, center_cn, decode, walsh_pieces
from Nov_3rd.quantum_h_tf import FockSim
from Nov_3rd.unsat_rank2_xx import (
    LAM,
    MU,
    PUBLISHED_N12_L16,
    ZXScore,
    best_product_unsat,
    clearly_below,
    decide_unsat_xx,
    dense_gap,
    plus_energy,
    product_energy_unsat,
    statevector_energy,
)


def _quantum(peak=0.04, gap=0.05):
    return {n: {"peak_med": peak, "prod_gap_med": gap} for n in (6, 9, 12, 15)}


def _cells(r12, r6=0.90, r9=0.90, r15=0.90, seed_Rs=None):
    if seed_Rs is None:
        seed_Rs = [r12, r12, r12]

    def mark(R):
        if R <= 0.98:
            return "BEAT"
        if R <= 1.02:
            return "TIE"
        return "UNDER"

    cells = {}
    for L, n, R in (
        (16, 6, r6),
        (16, 9, r9),
        (16, 12, r12),
        (16, 15, r15),
        (2, 6, 0.70),
        (2, 9, 0.70),
        (2, 12, 0.70),
        (2, 15, 0.70),
    ):
        srs = list(seed_Rs) if (L, n) == (16, 12) else [R, R, R]
        cells[("walsh", L, n)] = {"R": float(R), "mark": mark(R), "seed_Rs": srs, "med": 0.01}
        cells[("power_p", L, n)] = {"R": 0.99, "mark": "TIE", "seed_Rs": [0.99, 0.99, 0.99], "med": 0.02}
    return cells


def _batch(n=6, B=2, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.integers(0, 5, size=(B, 1 << n)).astype(np.float64)
    pulse = center_cn(score, n, max(n, 1))
    sim = FockSim(n, dtype=torch.float64)
    pieces = walsh_pieces(torch.as_tensor(pulse, dtype=torch.float64))
    circ = Circuit(
        sim,
        torch.as_tensor(pulse, dtype=torch.float64),
        torch.as_tensor(score, dtype=torch.float64),
        torch.zeros(B, dtype=torch.float64),
        pieces=pieces,
    )
    return circ, score


def test_mu0_histogram_matches_pauli_score():
    circ, _score = _batch()
    g = torch.tensor([0.2, -0.1], dtype=torch.float64)
    phi = torch.tensor([[0.0, 0.1, -0.05, 0.02], [0.0, -0.08, 0.03, 0.01]], dtype=torch.float64)
    pauli = circ.energy(g, phi, False, LAM)
    hist = ZXScore(circ, 0.0).energy(g, phi, False, LAM)
    assert torch.allclose(pauli, hist, atol=1e-8), (pauli - hist).abs().max()
    # graded weights that are not all equal still have to match at μ=0,
    # because XX is off and the field term is the same operator
    gw = torch.tensor([[0.2, 0.05, -0.1, 0.0], [-0.1, 0.02, 0.07, -0.03]], dtype=torch.float64)
    pauli_g = circ.energy(gw, phi, True, LAM)
    hist_g = ZXScore(circ, 0.0).energy(gw, phi, True, LAM)
    assert torch.allclose(pauli_g, hist_g, atol=1e-8)


def test_histogram_matches_statevector_and_plus():
    circ, score = _batch(seed=1)
    g = torch.tensor([0.15, -0.07, 0.02], dtype=torch.float64)
    phi = torch.tensor(
        [[0.0, -0.12, 0.04, 0.01], [0.0, 0.05, -0.02, 0.03], [0.0, 0.02, 0.01, -0.04]],
        dtype=torch.float64,
    )
    zx = ZXScore(circ, MU)
    got = zx.energy(g, phi, False, LAM)
    psi = circ.final_amps(g, phi, False).detach().cpu().numpy()
    pZ, _pX = circ.bit_histograms(g, phi, False)
    amps = circ.final_amps(g, phi, False)
    probs = amps.real.square() + amps.imag.square()
    assert torch.allclose(pZ, probs, atol=1e-10)
    for i in range(score.shape[0]):
        ref = statevector_energy(psi[i], score[i], 6, LAM, MU)
        assert abs(float(got[i]) - ref) < 1e-8, (float(got[i]), ref)
    # L=0 is |+⟩
    z = torch.zeros(0, dtype=torch.float64)
    e0 = ZXScore(circ, MU).energy(z, z.new_zeros(0, 4), False, LAM)
    for i in range(score.shape[0]):
        assert abs(float(e0[i]) - plus_energy(score[i], 6, LAM, MU)) < 1e-8


def test_lanczos_matches_dense_and_product_limits():
    circ, score = _batch(B=1, seed=2)
    err = dense_gap(score[0], 6, LAM, MU)
    assert err < 1e-8, err
    err0 = dense_gap(score[0], 6, LAM, 0.0)
    assert err0 < 1e-8, err0
    n = 6
    diag = score[0]
    e_plus = product_energy_unsat(np.full(n, 0.5 * np.pi), diag, n, LAM, MU)
    assert abs(e_plus - plus_energy(diag, n, LAM, MU)) < 1e-8
    e_bit = product_energy_unsat(np.zeros(n), diag, n, LAM, MU)
    assert abs(e_bit - float(diag[0])) < 1e-8
    e_prod = best_product_unsat(diag, n, LAM, MU, seed=3)
    assert e_prod <= e_plus + 1e-6


def test_xx_term_has_a_gradient():
    circ, _score = _batch(B=1, seed=4)
    theta = torch.nn.Parameter(torch.tensor(
        [0.2, -0.1, 0.05, 0.0, 0.02, -0.01, 0.0, 0.01, 0.0, -0.02, 0.03, 0.0, 0.01, -0.04, 0.02],
        dtype=torch.float64,
    ))

    def loss_of(th):
        g, phi, graded = decode("walsh", th, 2, 6)
        return ZXScore(circ, MU).energy(g, phi, graded, LAM).sum()

    loss_of(theta).backward()
    g_auto = float(theta.grad[0])
    eps = 1e-6
    with torch.no_grad():
        theta[0] += eps
        lp = float(loss_of(theta))
        theta[0] -= 2 * eps
        lm = float(loss_of(theta))
        theta[0] += eps
    g_fd = (lp - lm) / (2 * eps)
    assert abs(g_auto - g_fd) < 1e-4, (g_auto, g_fd)


def test_clear_drop_needs_both_medians():
    published = PUBLISHED_N12_L16
    assert clearly_below(0.900, published, [0.89, 0.90, 0.91])["clear"]
    # headline drops, one seed does not, median seed does not
    assert not clearly_below(0.910, published, [0.70, 0.97, 0.99])["clear"]
    # inside the 0.02 bar
    assert not clearly_below(0.930, published, [0.92, 0.93, 0.94])["clear"]
    # exactly the bar
    assert clearly_below(published - 0.02, published, [0.91, 0.92, 0.92])["clear"]


def test_pass_when_n12_walsh_drops_clearly():
    out = decide_unsat_xx(_cells(0.880, seed_Rs=[0.86, 0.88, 0.90]), _quantum())
    assert out["verdict"] == "PASS", out
    assert out["null"] is False


def test_null_when_drop_is_inside_the_bar():
    out = decide_unsat_xx(_cells(0.935, seed_Rs=[0.92, 0.93, 0.95]), _quantum())
    assert out["verdict"] == "needs_edit", out
    assert out["null"] is True
    assert "null" in out["reason"]


def test_null_when_xx_is_worse_but_still_a_tie():
    out = decide_unsat_xx(_cells(1.000, seed_Rs=[0.99, 1.00, 1.01]), _quantum())
    assert out["verdict"] == "needs_edit", out
    assert out["null"] is True


def test_kill_classical_and_reversal():
    cells = _cells(0.880, seed_Rs=[0.86, 0.88, 0.90])
    q = _quantum()
    q[12] = {"peak_med": 0.7, "prod_gap_med": 0.1}
    out = decide_unsat_xx(cells, q)
    assert out["verdict"] == "KILL", out
    rev = decide_unsat_xx(_cells(1.08, r6=0.90, r9=1.10, r15=1.00, seed_Rs=[1.05, 1.08, 1.12]), _quantum())
    assert rev["verdict"] == "KILL", rev


def test_clear_ratio_with_product_ground_state_is_not_a_pass():
    out = decide_unsat_xx(
        _cells(0.880, seed_Rs=[0.86, 0.88, 0.90]),
        _quantum(peak=0.04, gap=0.005),
    )
    assert out["verdict"] == "needs_edit", out
    assert out["null"] is False
    assert "product gap" in out["reason"]
