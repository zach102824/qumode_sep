"""Classical low-energy statistics -> explore layout + XOR (IMPROVE_N12_N14 ideas 1/2).

A budgeted single-flip simulated annealing over the logical energy table (every distinct table entry
read counts as one classical lookup). From the lowest-energy distinct states seen we take
Boltzmann-weighted bit marginals. Explore run j then uses:
  * center c_j = j-th lowest-energy distinct state seen (cycled), XOR'd to sit at Fock vacuum;
  * a permutation that puts the most uncertain bits on the transmons, then on the cavity Gray digits
    from least to most significant (A/B alternating), so likely-fixed bits sit on the high digits.
Runs j > 0 add rank noise so different runs try different layouts.
Physical slot order (MSB first): 0, 1 = transmons; 2 .. k+1 = cavity A digits MSB -> LSB; then B.
"""
from __future__ import annotations

import numpy as np


def sa_sample(E: np.ndarray, n: int, budget: int, rng: np.random.Generator, sweeps_T=(2.0, 0.05)):
    seen: dict[int, float] = {}

    def look(v):
        if v not in seen:
            seen[v] = float(E[v])
        return seen[v]

    scale = float(np.std(E[rng.integers(0, len(E), 64)]))  # 64 random probes, counted
    for v in rng.integers(0, len(E), 64):
        look(int(v))
    scale = max(scale, 1e-9)
    T0, T1 = sweeps_T
    while len(seen) < budget:
        v = int(rng.integers(len(E)))
        e = look(v)
        nsteps = 8 * n
        for t in range(nsteps):
            if len(seen) >= budget:
                break
            T = scale * T0 * (T1 / T0) ** (t / max(1, nsteps - 1))
            w = v ^ (1 << int(rng.integers(n)))
            ew = look(w)
            if ew <= e or rng.random() < np.exp(-(ew - e) / T):
                v, e = w, ew
    return seen


def marginals(seen: dict, n: int, top: int = 24):
    items = sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))[:top]
    vs = np.array([v for v, _ in items])
    es = np.array([e for _, e in items])
    sd = max(float(es.std()), 1e-9)
    w = np.exp(-(es - es.min()) / sd)
    w /= w.sum()
    bits = (vs[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    m = (w[:, None] * bits).sum(0)
    return [int(v) for v in vs], m


def slot_order(k: int) -> list[int]:
    """Physical slots from 'most freely rotated / least significant' to most significant."""
    out = [0, 1]
    for d in range(k - 1, -1, -1):  # LSB digit first
        out += [2 + d, 2 + k + d]
    return out


def layout_for_run(j: int, centers, m, n: int, k: int, rng: np.random.Generator, noise: float = 0.15,
                   unc_mode: str = "uncertain_low"):
    u = np.minimum(m, 1 - m)  # 0 = fixed, 0.5 = uncertain
    if j > 0:
        u = u + noise * rng.random(n)
    order_bits = list(np.argsort(-u, kind="stable"))  # most uncertain first
    if unc_mode == "fixed_low":
        order_bits = order_bits[::-1]
    perm = np.empty(n, dtype=int)
    for slot, b in zip(slot_order(k), order_bits):
        perm[slot] = int(b)
    return perm, int(centers[j % len(centers)])
