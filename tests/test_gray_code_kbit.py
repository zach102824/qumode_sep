import numpy as np
from noiseless.ecd_kbit import KbitLayout, gray_encode
from noiseless.relayout_kbit import XorKbitSim, cavity_masks_for


def test_gray_bijection_and_adjacent():
    for k in (3, 4, 5):
        g = [gray_encode(f) for f in range(1 << k)]
        assert sorted(g) == list(range(1 << k))
        assert all(bin(g[f] ^ g[f + 1]).count("1") == 1 for f in range(len(g) - 1))


def test_layout_roundtrip_gray():
    lay = KbitLayout(4, 20, encoding="gray")
    flat = lay.flat_of_logical()
    assert len(set(flat.tolist())) == flat.size
    for j in range(flat.size):
        assert lay.logical_of_flat(int(flat[j])) == j


def test_relabeled_guess_at_vacuum_gray():
    k, nf = 3, 24
    n = 2 + 2 * k
    rng = np.random.default_rng(0)
    E = rng.normal(size=1 << n)
    gl = int(np.argmin(E))
    lay = KbitLayout(k, nf, encoding="gray")
    for guess in (5, 77, 200, gl):
        xa, xb = cavity_masks_for(guess, k)
        sim = XorKbitSim(lay, E, 1, format(gl, f"0{n}b"), xa=xa, xb=xb)
        d, e = guess >> (2 * k + 1) & 1, guess >> (2 * k) & 1
        fl = int(np.ravel_multi_index((d, e, 0, 0), lay.dims))
        assert sim.energies_flat[fl] == E[guess]
