"""CAP packing, relabeling, and QAOA-invariance checks. No scale1000 required."""
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

from beat_params import phi_from_c
from Nov_3rd.cavity_packing import (
    cooccurrence,
    identity_perm,
    pack_permutation,
    packing_score,
    permute_energy,
    random_permutations,
)
from explore_OCT29.lib.qaoa import qaoa_psucc
from explore_OCT29.lib.sat_gen import fast_energies


def _clauses_from_array(cls_v, cls_s):
    cl = []
    for v, s in zip(cls_v, cls_s):
        cl.append((list(map(int, v)), list(map(int, s))))
    return cl


def test_deterministic_and_bijective():
    rng = np.random.default_rng(0)
    cls = rng.integers(0, 9, size=(40, 4), dtype=np.int64)
    # force distinct vars inside each clause
    for i in range(len(cls)):
        cls[i] = rng.choice(9, size=4, replace=False)
    w = cooccurrence(cls, 9)
    a = pack_permutation(w, True)
    b = pack_permutation(w, True)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(9))
    lo = pack_permutation(w, False)
    assert packing_score(w, a) >= packing_score(w, lo)


def test_heavy_pair_is_packed_and_antipacked():
    # n=6. Many clauses on {0,1,2,*} so the triple 0,1,2 co-occurs; 3,4,5 never do.
    rows = []
    for extra in (3, 4, 5, 3, 4, 5):
        rows.append([0, 1, 2, extra])
    # a few disjoint clauses so the complement is not empty of structure
    rows.append([3, 4, 5, 0])
    cls = np.asarray(rows, dtype=np.int64)
    w = cooccurrence(cls, 6)
    pi_max = pack_permutation(w, True)
    pi_min = pack_permutation(w, False)
    # variables 0,1,2 should share a cavity under π_max
    cav_of = {}
    for c in range(2):
        for v in pi_max[3 * c : 3 * c + 3]:
            cav_of[int(v)] = c
    assert cav_of[0] == cav_of[1] == cav_of[2], pi_max
    cav_min = {}
    for c in range(2):
        for v in pi_min[3 * c : 3 * c + 3]:
            cav_min[int(v)] = c
    # anti-pack should not keep all three together
    assert not (cav_min[0] == cav_min[1] == cav_min[2]), pi_min
    assert packing_score(w, pi_max) > packing_score(w, pi_min)


def test_uniform_weights_are_identity():
    w = np.zeros((6, 6), dtype=np.int32)
    pi = pack_permutation(w, True)
    assert np.array_equal(pi, identity_perm(6))
    pi_min = pack_permutation(w, False)
    assert np.array_equal(pi_min, identity_perm(6))


def test_energy_matches_relabeled_clauses():
    n = 6
    rng = np.random.default_rng(1)
    cls_v = np.stack([rng.choice(n, size=4, replace=False) for _ in range(12)]).astype(np.int64)
    cls_s = rng.choice([-1, 1], size=cls_v.shape).astype(np.int64)
    E = fast_energies(n, _clauses_from_array(cls_v, cls_s))
    w = cooccurrence(cls_v, n)
    for pi in [identity_perm(n), pack_permutation(w, True), random_permutations(n, k=2, seed=7)[0]]:
        inv = np.empty(n, dtype=np.int64)
        inv[pi] = np.arange(n)
        cls_v2 = inv[cls_v]
        E2 = fast_energies(n, _clauses_from_array(cls_v2, cls_s))
        Ep = permute_energy(E, pi)
        assert np.array_equal(Ep, E2), (pi, int((Ep != E2).sum()))
        assert int((Ep == 0).sum()) == int((E == 0).sum())


def test_random_perms_deterministic_and_distinct_from_stream():
    a = random_permutations(15, k=8, seed=20261104)
    b = random_permutations(15, k=8, seed=20261104)
    assert len(a) == 8
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
        assert sorted(x.tolist()) == list(range(15))
    # different n → different stream, not a prefix accident on length
    c = random_permutations(12, k=8, seed=20261104)
    assert not np.array_equal(a[0][:12], c[0])


def test_qaoa_perm_invariance_and_snap_identity():
    """QAOA Med is unchanged by π. Identity π leaves the SNAP state untouched."""
    from Nov_3rd.ansatz import make_sim

    n = 6
    rng = np.random.default_rng(2)
    cls_v = np.stack([rng.choice(n, size=4, replace=False) for _ in range(20)]).astype(np.int64)
    cls_s = rng.choice([-1, 1], size=cls_v.shape).astype(np.int64)
    Es = []
    for _ in range(4):
        # independent formulas
        cv = np.stack([rng.choice(n, size=4, replace=False) for _ in range(20)])
        cs = rng.choice([-1, 1], size=cv.shape)
        Es.append(fast_energies(n, _clauses_from_array(cv, cs)))
    E = np.stack(Es).astype(np.float32)
    betas = torch.tensor([0.3, 0.2], dtype=torch.float32)
    gammas = torch.tensor([-0.4, -0.5], dtype=torch.float32)
    pi = random_permutations(n, k=1, seed=11)[0]
    Ep = permute_energy(E, pi).astype(np.float32)
    p0 = qaoa_psucc(betas, gammas, torch.tensor(E), n)
    p1 = qaoa_psucc(betas, gammas, torch.tensor(Ep), n)
    # per-instance success, not just the median
    assert torch.max(torch.abs(p0 - p1)).item() < 1e-5

    sim = make_sim(n, 2, torch.device("cpu"))
    phi = phi_from_c(
        torch.tensor([-0.3, -0.2]),
        torch.tensor([0.05, -0.04]),
        torch.tensor([0.01, 0.02]),
    )
    gamma = torch.tensor([-0.4, -0.5])
    with torch.no_grad():
        a = sim.forward_nov3rd(torch.tensor(E[0]), gamma, phi)
        b = sim.forward_nov3rd(torch.tensor(permute_energy(E[0], identity_perm(n)).astype(np.float32)), gamma, phi)
    assert abs(float(a["inv_p"]) - float(b["inv_p"])) < 1e-6


def main():
    tests = [
        test_deterministic_and_bijective,
        test_heavy_pair_is_packed_and_antipacked,
        test_uniform_weights_are_identity,
        test_energy_matches_relabeled_clauses,
        test_random_perms_deterministic_and_distinct_from_stream,
        test_qaoa_perm_invariance_and_snap_identity,
    ]
    for fn in tests:
        fn()
        print("PASS", fn.__name__, flush=True)
    print(f"=== {len(tests)} passed ===")


if __name__ == "__main__":
    main()
