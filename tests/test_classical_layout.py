import numpy as np

from noiseless.classical_layout import layout_for_run, marginals, sa_sample, slot_order
from noiseless.relayout_kbit import bit_perm_map, cavity_masks_for


def test_sa_budget_and_layout():
    n, k = 10, 4
    rng = np.random.default_rng(0)
    E = rng.random(2 ** n)
    seen = sa_sample(E, n, 100, rng)
    assert len(seen) <= 100 and all(np.isclose(E[v], e) for v, e in seen.items())
    centers, m = marginals(seen, n)
    assert centers[0] == min(seen, key=seen.get)
    assert sorted(slot_order(k)) == list(range(n))
    for mode in ("uncertain_low", "fixed_low"):
        for j in range(3):
            perm, c = layout_for_run(j, centers, m, n, k, rng, 0.3, mode)
            assert sorted(perm) == list(range(n))
            Pm = bit_perm_map(perm, n)
            pc = int(np.flatnonzero(Pm == c)[0])
            xa, xb = cavity_masks_for(pc, k)
            assert ((xa << k) | xb) ^ (pc & ((1 << 2 * k) - 1)) == 0
