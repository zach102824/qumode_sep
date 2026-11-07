"""Identity checks for the L=16 SNAP rescue. No pack I/O, no n=21."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.hva_safe import cn_from_unsat, fourier_state, hva_state, snap_state, state_fidelity
from Nov_3rd.snap_rescue import (
    a_of_n,
    pad_identity,
    phi_from_hva_mixer,
    phi_power,
    tile_layers,
    verdict_pair,
    xx_popcount_eigenvalues,
)


def test_xx_popcount_spectrum():
    ev = xx_popcount_eigenvalues(torch.float64)
    expect = torch.tensor([3.0, -1.0, -1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(ev, expect), ev


def test_hva_embeds_in_shell():
    rng = torch.Generator().manual_seed(2)
    n, L, B = 6, 3, 2
    Cn = torch.randn(B, 1 << n, dtype=torch.float64, generator=rng)
    beta_x = torch.randn(L, dtype=torch.float64, generator=rng)
    beta_xx = torch.randn(L, dtype=torch.float64, generator=rng)
    gammas = torch.randn(L, dtype=torch.float64, generator=rng)
    sh = hva_state(Cn, beta_x, beta_xx, gammas)
    # Scalar γ isolates the mixer map. Equal Walsh weights are a separate 4-local check.
    ss = snap_state(Cn, gammas, phi_from_hva_mixer(beta_x, beta_xx))
    fid = float(state_fidelity(sh, ss).min())
    assert fid > 1 - 1e-8, fid
    # The XXX direction is the part of φ orthogonal to the HVA image.
    # A pure popcount-1 vs popcount-2 split is not an XX phase.
    phi = phi_from_hva_mixer(beta_x, beta_xx)
    # ev_XX is equal on k=1 and k=2, and the X piece is linear, so
    # φ_1 - 2 φ_0? just check φ_1 - φ_2 equals the linear gap -β (k=1 minus k=2).
    gap = phi[:, 1] - phi[:, 2]
    assert torch.allclose(gap, beta_x), gap


def test_power_p_paper_init_is_fourier():
    rng = torch.Generator().manual_seed(3)
    n, L, B = 6, 4, 2
    Cn = torch.randn(B, 1 << n, dtype=torch.float64, generator=rng)
    gamma = torch.randn(L, dtype=torch.float64, generator=rng)
    beta = torch.rand(L, dtype=torch.float64, generator=rng) + 0.05
    z = torch.zeros(L, dtype=torch.float64)
    p = torch.zeros((), dtype=torch.float64)
    sf = fourier_state(Cn, beta, gamma)
    for n_eval in (6, 9, 12):
        phi = phi_power(-beta, z, z, p, n_eval)
        ss = snap_state(Cn, gamma, phi)
        fid = float(state_fidelity(sf, ss).min())
        assert fid > 1 - 1e-8, (n_eval, fid)


def test_equal_walsh_matches_scalar_on_4local():
    n = 6
    E = torch.zeros(1 << n, dtype=torch.float64)
    for x in range(1 << n):
        bits = [((x >> (n - 1 - q)) & 1) for q in range(4)]
        if bits == [1, 1, 1, 1]:
            E[x] = 1.0
    Cn = cn_from_unsat(E.reshape(1, -1), n)
    g = torch.tensor([0.2, -0.15, 0.05], dtype=torch.float64)
    phi = torch.tensor(
        [[0.0, -0.2, 0.1, 0.4], [0.2, -0.1, 0.0, 0.3], [-0.2, 0.4, -0.3, 0.1]],
        dtype=torch.float64,
    )
    gw = g.reshape(-1, 1).expand(-1, 4).contiguous()
    a = snap_state(Cn, g, phi)
    b = snap_state(Cn, gw, phi)
    fid = float(state_fidelity(a, b).min())
    assert fid > 1 - 1e-8, fid


def test_a_of_n_fixes_n9():
    for p_val in (-1.5, 0.0, 0.7, 3.0):
        p = torch.tensor(p_val, dtype=torch.float64)
        a9 = float(a_of_n(9, p))
        assert abs(a9 - 1.0) < 1e-12, a9
    p = torch.tensor(1.0, dtype=torch.float64)
    assert abs(float(a_of_n(6, p)) - 8.0 / 5.0) < 1e-12
    assert abs(float(a_of_n(12, p)) - 8.0 / 11.0) < 1e-12
    assert abs(float(a_of_n(15, p)) - 8.0 / 14.0) < 1e-12


def test_pad_and_tile():
    block = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    tiled = tile_layers(block, 2)
    assert tiled.shape == (4, 4)
    assert tiled[2, 0] == 1.0 and tiled[3, 3] == 8.0
    padded = pad_identity(block, 5)
    assert padded.shape == (5, 4)
    assert padded[0, 0] == 1.0 and padded[1, 3] == 8.0
    assert padded[2:].sum() == 0.0


def test_verdict_thresholds():
    win = verdict_pair(0.010, 0.014)
    assert win["verdict"] == "PASS" and win["reached_098"]
    tie = verdict_pair(0.014, 0.014)
    assert tie["verdict"] == "PASS"
    assert abs(tie["R"] - 1.0) < 1e-12
    assert tie["reached_098"] is False
    near = verdict_pair(0.0090, 0.010)
    assert near["verdict"] == "PASS" and near["R"] <= 0.98
    lose = verdict_pair(0.020, 0.014)
    assert lose["verdict"] == "needs_edit" and lose["reached_098"] is False


def main():
    test_xx_popcount_spectrum()
    test_hva_embeds_in_shell()
    test_equal_walsh_matches_scalar_on_4local()
    test_power_p_paper_init_is_fourier()
    test_a_of_n_fixes_n9()
    test_pad_and_tile()
    test_verdict_thresholds()
    print("snap rescue tests ok")


if __name__ == "__main__":
    main()
