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
    bootstrap_ratio,
    clearly_below,
    GAP_COVERAGE,
    decide_deepen,
    decide_unsat_xx,
    dense_gap,
    product_gap_tail,
    paired_instance_counts,
    paired_seed_table,
    plus_energy,
    product_energy_unsat,
    statevector_energy,
    wilson_interval,
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


def _five(r12, r15, r6=0.40, r9=0.40):
    def mark(R):
        if R <= 0.98:
            return "BEAT"
        if R <= 1.02:
            return "TIE"
        return "UNDER"

    cells = {}
    for n, R, seeds in (
        (6, r6, [r6] * 5),
        (9, r9, [r9] * 5),
        (12, r12, [r12] * 5),
        (15, r15, [r15] * 5),
    ):
        cells[("walsh", 16, n)] = {"R": float(R), "mark": mark(R), "seed_Rs": list(seeds)}
    return cells


def _paired_all(n_seeds=5, n_inst=32):
    per = []
    for _ in range(n_seeds):
        per.append(paired_instance_counts(np.full(n_inst, 0.1), np.full(n_inst, 0.4)))
    table = {
        "per_seed": per,
        "n_majority": n_seeds,
        "n_seeds": n_seeds,
        "win": n_seeds * n_inst,
        "tie": 0,
        "loss": 0,
        "n": n_seeds * n_inst,
    }
    return {12: table, 15: table}


def _ci_for(cells, hi12, hi15):
    out = {}
    for n, hi in ((12, hi12), (15, hi15)):
        R = float(cells[("walsh", 16, n)]["R"])
        out[("walsh", 16, n)] = {"R": R, "lo": R - 0.02, "hi": float(hi)}
    return out


def test_paired_counts_and_wilson():
    row = paired_instance_counts([0.1, 0.2, 0.4], [0.3, 0.2, 0.3])
    assert (row["win"], row["tie"], row["loss"], row["n"]) == (1, 1, 1, 3)
    table = paired_seed_table(
        [np.array([0.1, 0.1, 0.1, 0.9]), np.array([0.2, 0.2, 0.8, 0.9])],
        [np.array([0.3, 0.3, 0.3, 0.3]), np.array([0.3, 0.3, 0.3, 0.3])],
    )
    assert table["n_majority"] == 1
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi
    assert hi - lo < 0.2


def test_instance_bootstrap_is_exact_on_a_constant_batch():
    eps_m = [np.full(24, 0.30), np.full(24, 0.30)]
    eps_b = [np.full(24, 1.00), np.full(24, 1.00)]
    out = bootstrap_ratio(eps_m, eps_b, n_boot=200, seed=1, mode="instance")
    assert abs(out["R"] - 0.30) < 1e-12
    assert abs(out["lo"] - 0.30) < 1e-12
    assert abs(out["hi"] - 0.30) < 1e-12
    spread = bootstrap_ratio(
        [np.full(12, 0.2), np.full(12, 0.4), np.full(12, 0.5)],
        [np.full(12, 1.0), np.full(12, 1.0), np.full(12, 1.0)],
        n_boot=300,
        seed=2,
        mode="both",
    )
    assert spread["lo"] <= spread["R"] <= spread["hi"]


def test_deepen_pass_requires_both_sizes_inside_the_interval():
    cells = _five(0.33, 0.36)
    paired_k = {("walsh", 16, n): _paired_all()[n] for n in (12, 15)}
    out = decide_deepen(
        cells,
        _quantum(),
        paired_k,
        _ci_for(cells, 0.50, 0.55),
        n18={"status": "infeasible", "detail": "skipped"},
    )
    assert out["verdict"] == "PASS", out
    assert "n=18 probe was not scored" in out["reason"]
    soft = decide_deepen(cells, _quantum(), paired_k, _ci_for(cells, 1.05, 0.55))
    assert soft["verdict"] == "needs_edit", soft
    assert soft["null"] is False
    assert "CI" in soft["reason"]


def test_deepen_n15_tie_and_short_seed_list():
    cells = _five(0.33, 1.00)
    paired_k = {("walsh", 16, n): _paired_all()[n] for n in (12, 15)}
    out = decide_deepen(cells, _quantum(), paired_k, _ci_for(cells, 0.40, 1.00))
    assert out["verdict"] == "needs_edit", out
    assert "n=15" in out["reason"]
    short = decide_deepen(_cells(0.33, seed_Rs=[0.33, 0.33, 0.33]), _quantum(), paired_k, _ci_for(cells, 0.40, 0.40))
    assert short["verdict"] == "needs_edit", short
    assert "5" in short["reason"]


def test_deepen_kill_still_ignores_a_passing_n18_note():
    cells = _five(0.33, 0.36)
    paired_k = {("walsh", 16, n): _paired_all()[n] for n in (12, 15)}
    q = _quantum()
    q[12] = {"peak_med": 0.8, "prod_gap_med": 0.2}
    out = decide_deepen(
        cells,
        q,
        paired_k,
        _ci_for(cells, 0.40, 0.40),
        n18={"status": "probe", "R": 0.40, "med": 0.001, "B": 16, "peak_med": 0.01, "prod_gap_med": 0.08},
    )
    assert out["verdict"] == "KILL", out
    assert "n=18 probe" in out["reason"]


def _with_coverage(frac12, frac15, n12=128, n15=64):
    q = _quantum()

    def fill(n, frac, n_inst):
        n_lt = int(round((1.0 - frac) * n_inst))
        # Keep the stored fraction exact so the gate does not depend on rounding.
        q[n]["n_inst"] = n_inst
        q[n]["n_gap_lt_0p02"] = n_lt
        q[n]["n_gap_lt_0p05"] = n_inst // 2
        q[n]["frac_gap_lt_0p02"] = 1.0 - frac
        q[n]["frac_gap_lt_0p05"] = 0.5
        q[n]["frac_gap_ge_0p02"] = frac
        q[n]["n_gap_ge_0p02"] = n_inst - n_lt
        q[n]["gap_min"] = 0.019 if frac < GAP_COVERAGE else 0.021

    fill(12, frac12, n12)
    fill(15, frac15, n15)
    return q


def test_product_gap_tail_counts_open_floors():
    gap = np.array([0.019, 0.02, 0.049, 0.05, 0.08])
    tail = product_gap_tail(gap, peak=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert tail["n_gap_lt_0p02"] == 1
    assert tail["n_gap_lt_0p05"] == 3
    assert tail["n_gap_ge_0p02"] == 4
    assert abs(tail["frac_gap_ge_0p02"] - 0.8) < 1e-12
    assert abs(tail["gap_min"] - 0.019) < 1e-15
    assert abs(tail["peak_max"] - 0.5) < 1e-15


def test_round7_coverage_is_opt_in_and_names_a_quantumness_fail():
    cells = _five(0.33, 0.36)
    paired_k = {("walsh", 16, n): _paired_all()[n] for n in (12, 15)}
    ci = _ci_for(cells, 0.50, 0.55)
    plain = decide_deepen(cells, _quantum(), paired_k, ci)
    assert plain["verdict"] == "PASS", plain
    missing = decide_deepen(cells, _quantum(), paired_k, ci, gap_coverage=GAP_COVERAGE)
    assert missing["verdict"] == "needs_edit", missing
    assert missing["reason"].startswith("quantumness fail")
    ok = decide_deepen(cells, _with_coverage(0.984375, 0.96875), paired_k, ci, gap_coverage=GAP_COVERAGE)
    assert ok["verdict"] == "PASS", ok
    assert "frac gap<0.02" in ok["reason"]
    assert "frac gap<0.05" in ok["reason"]
    # 1/64 is still inside 95%. 8/128 is not.
    edge = decide_deepen(cells, _with_coverage(63 / 64, 1.0, n12=64, n15=32), paired_k, ci, gap_coverage=GAP_COVERAGE)
    assert edge["verdict"] == "PASS", edge
    bad = decide_deepen(cells, _with_coverage(120 / 128, 1.0), paired_k, ci, gap_coverage=GAP_COVERAGE)
    assert bad["verdict"] == "needs_edit", bad
    assert bad["null"] is False
    assert bad["reason"].startswith("quantumness fail")
    assert "n=12" in bad["reason"]
    assert "frac < 0.05" in bad["reason"]
    assert bad["quantum_ok"] is False
