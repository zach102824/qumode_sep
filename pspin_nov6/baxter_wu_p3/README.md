# Baxter–Wu up/down triangles, round 2

Sparse uniform-sign hypergraphs from round 1 are a different model. This directory is the triangular-lattice torus

```
H = - sum_{up ▲} ZZZ + lambda * sum_{down ▼} ZZZ - Gamma * sum_i X_i
```

with `lambda = 1` in the QAOA runs. Sites are `(x, y)`, index `x + Lx * y`, periodic. Faces:

- up `{(x,y), (x+1,y), (x+1,y+1)}`
- down `{(x,y), (x,y+1), (x+1,y+1)}`

`M = 2N` with no repeated face and no up/down coincidence on the tori that were used.

The target is the lowest eigenvector of the full `2^N` Hamiltonian. A 3-coloring, and the `Z2×Z2` of flipping two sublattices, exists only when both periods are divisible by 3. That was checked per torus. It holds on 3×3 and 3×6 and fails on 3×4, 3×5, and 3×7.

The staggered sign pattern is not gauge-equivalent to the uniform ferromagnet on any of those tori (GF(2) rank, confirmed by enumerating gauges). `lambda = 1` was kept. The cost spectrum lies on multiples of 8, so `gamma` has period `π/4`. `beta` has period `π`. Neither half-period is a symmetry.

```
cd pspin_nov6/baxter_wu_p3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_round2.py
```

Round-2 numbers are in `runs/round2/RESULTS.md`.
