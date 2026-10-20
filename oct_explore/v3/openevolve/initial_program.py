"""Evolvable formula -> native-gate-parameter mapping for the cavity chain (3 cavities x 5 bits at n=15).
Only numpy/math allowed. The design sees ONLY the formula (clauses); it must not solve the formula.
Interface (called once per formula):
    design(n, ks, clauses) -> dict with
      perm   : list[int] len n, chain position q holds variable perm[q] (cavity c = positions sum(ks[:c])..)
      pv     : array (n,) in (0,1), per-VARIABLE projection measure P(x_v=1) (used only to weight the least-squares projection)
      gamma  : array (L,) cost-phase strengths
      beta   : array (L,) or (L, C) per-cavity one-bit-flip mixer angles
      w      : array (m, L) clause weights per layer (default 1)
      snap   : optional list over L of list over C of arrays (2^k_c,) extra per-cavity SNAP phases (logical order, radians)
clauses: list of (vars[4], signs[4]); a literal (v, s) is violated when x_v == (0 if s > 0 else 1). Clause violated iff all 4 literals are.
"""
import numpy as np

# EVOLVE-BLOCK-START
def place(n, ks, clauses, iters=3000, seed=0):
    """simulated annealing: minimise # clauses not inside one cavity or two neighbouring cavities."""
    rng = np.random.default_rng(seed); off = np.cumsum([0] + list(ks))
    cav_of_pos = np.repeat(np.arange(len(ks)), ks); V = np.array([c[0] for c in clauses])
    def cost(pos):
        cc = cav_of_pos[pos[V]]; lo, hi = cc.min(1), cc.max(1)
        return np.sum(hi - lo > 1)  # 3 contiguous cavities -> spans >1 => not representable (approx)
    perm = np.arange(n); pos = np.argsort(perm); cur = cost(pos); T = 2.0
    for t in range(iters):
        i, j = rng.integers(0, n, 2)
        if cav_of_pos[pos[i]] == cav_of_pos[pos[j]]: continue
        pos[i], pos[j] = pos[j], pos[i]; c = cost(pos)
        if c <= cur or rng.random() < np.exp((cur - c) / T): cur = c
        else: pos[i], pos[j] = pos[j], pos[i]
        T = max(0.05, T * 0.999)
    return list(np.argsort(pos))

def design(n, ks, clauses):
    L = 4; m = len(clauses)
    x = np.array([1.1304587, -3.1274996, 0.3672308, -6.4900058, -0.5844118, 9.6225362, -0.3927512, 8.4856458])
    beta, gamma = x[0::2], x[1::2]
    return dict(perm=place(n, ks, clauses), pv=np.full(n, 0.5), gamma=gamma, beta=beta, w=np.ones((m, L)), snap=None)
# EVOLVE-BLOCK-END
