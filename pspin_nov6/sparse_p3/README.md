# Sparse uniform-sign 3-body QAOA (round 1)

This directory is Family B only. It is a **sparse distinct-index** model, not the collective p-spin and not a random-sign spin glass.

## Hamiltonian

On N qubits, M = N triples of a 3-regular simple hypergraph (each vertex in three triples, triples of three distinct indices, no repeated triple):

```
H = - sum_a Z_{i_a} Z_{j_a} Z_{k_a} - Gamma * sum_i X_i
```

Couplings are uniformly +1. Qubit 0 is the least-significant bit of the computational index.

The collective ferromagnet `-(sum_i Z_i)^3 / N^2` is a different operator. The bookkeeping identity

```
(sum Z)^3 = 6 * sum_{i<j<k} Z_i Z_j Z_k + (3N-2) * sum Z_i
```

is checked numerically and is **not** the Hamiltonian that is optimized. Expanding the collective model produces every triple plus a longitudinal field. This model keeps N selected triples and no longitudinal field.

At Gamma = 0 a full-rank GF(2) incidence matrix makes the classical ground state unique: the all-+ product state (computational basis index 0), energy -N. That product limit is recorded and is not treated as a successful preparation.

## Graphs

Configuration model, NumPy PCG64, seeds 0, 1, 2, ... in order. A draw is kept when it is simple, exactly 3-regular, and the N x N incidence matrix has GF(2) rank N. The first four distinct graphs at each N in {6, 8, 10, 12} are used. No energy or fidelity filter.

A linear hypergraph (edges intersect in at most one vertex) is impossible at N = 6: 3N pairs would be required and C(6, 2) = 15 < 18. Larger N can be linear, but the seed-order rule does not prefer linear graphs. Each saved graph is tagged `linear: true/false`.

## QAOA

Cost and mixer, applied with layer 1 first:

```
H_C = - sum_a ZZZ_a
H_X = sum_i X_i
|psi> = prod_{l = L ... 1} exp(-i beta_l H_X) exp(-i gamma_l H_C) |+>
```

The objective is the full target energy `<H_C> - Gamma <H_X>`. Fidelity is measured afterward against the full-space ground state. No symmetry sector is projected out.

Gamma is a coefficient in H. It is not used to rescale the angles.

### Periods

Every cost eigenvalue equals M - 2 n_satisfied, so the spectrum lies in one parity class and steps by 2. Therefore `gamma -> gamma + pi` multiplies the cost unitary by a global phase, and `pi/2` is not a period. The same counting applies to `H_X`, so `beta` has period `pi` and not `pi/2`. Both statements are checked on a state that is not an eigenstate of the generator.

`(gamma, beta) -> (-gamma, -beta)` sends the QAOA state to its complex conjugate. Energy and fidelity to a real ground state are unchanged.

Reported durations use the principal branch in `(-pi/2, pi/2]`. Raw optimizer angles, the integer branch index (how many periods were subtracted), and `sum |gamma|` on both the raw and the principal branches are stored. A consecutive jump of the principal angles is recorded. The principal angles are replayed and required to reproduce the optimized state.

Odd p also gives a spectrum symmetry: conjugation by `prod_i Y_i` sends H to -H, so Emax = -E0. `eta = (Emax - E) / (Emax - E0)`.

### Search

Independent angles are the primary ansatz (2L parameters).

- L = 1: grid on `[0, pi) x [0, pi)`, then L-BFGS-B from the best energy nodes and from random seeds. A separate maximization of fidelity, from the best fidelity nodes, checks that the energy minimizer did not miss a lower-energy point.
- L = 2: the Pagano seed `(beta, beta - 0.2)`, `(gamma, gamma + 0.2)`, sign variants, zero-layer pads, and random starts.
- L >= 3: resample the previous schedule on `s in [0, 1]` (linear at L = 3, natural cubic spline after that), plus the same interpolation of the principal branch, small perturbations, zero-layer pads, and random starts.

Zero-layer pads copy the previous energy, so the reported energy is nested in L whenever a pad is a candidate. Nested energy does not force the fidelity to increase, and the fidelity is not edited when it falls.

Shared-gamma (one gamma, independent betas) is a secondary diagnostic at the same depths.

### Annealing baselines

The interpolating Hamiltonian is

```
H(s) = (1-s) (-H_X) + s (H_C - Gamma H_X)
     = s H_C - [(1-s) + s Gamma] H_X
```

with slice points `s_l = (l - 1/2) / L`. The Trotter factor `exp(-i beta H_X) exp(-i gamma H_C)` has `gamma = dt * s` and `beta = -dt * [(1-s) + s Gamma]`. The minus sign is required by that decomposition.

- Matched baseline: one scale factor makes `sum |gamma| + sum |beta|` equal the principal-branch QAOA duration. The gamma/beta ratio and the sign of beta are kept.
- Fixed baseline: `dt = 0.5` per slice, independent of the QAOA angles.

## Ground-state window

Prescribed fields: 0.3, 0.5, 0.7, 0.9, 1.1, 1.3.

A ground-state survey (gap, half-cut entropy, product overlap, overlap with |+>) is written to `gamma_survey.json` before any QAOA fidelity is computed. Two adaptive fields are added from that survey, not from the QAOA scores:

- 0.95: half-cut entropy becomes appreciable while the overlap with |+> is still modest.
- 1.0: minimum median gap on the surveyed grid.

## Observables

- Gap E1 - E0 of the full Hamiltonian, and Emax.
- `eta = (Emax - E) / (Emax - E0)`.
- F = |<ground state|psi>|^2 with the full-space ground state.
- Energy.
- Half-cut entropy in bits. The cut is qubits `0 .. N/2-1` (low bits) versus the complementary high bits. It is not a path cut; the hypergraph is not a chain. The same bit partition is used for every graph of a given N.
- Order: `m_z = (1/N) sum <Z_i>`, `m_x`, and the mean clause expectation `<ZZZ>`.
- Overlap with |+>, with the classical all-+ bitstring, with the best computational-basis bitstring, and with the best real product state.
- Instance min / median / max over the four seeds. Medians are not a license to drop graphs.

## Reproduce

```
cd pspin_nov6/sparse_p3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_round1.py
```

Dependencies: `requirements.txt` (numpy, scipy, matplotlib). The run writes `runs/round1/`. The round-1 writeup is `runs/round1/RESULTS.md`.

Validation (`validate.py`) covers the Pauli identity, an independent Kronecker diagonal, Lanczos versus dense eigh, the periods, conjugation, expm replay of the circuit, the unique classical ground state, a partial spin-flip gauge, the odd-p global sign flip, the half-cut entropy of a product state and of a Bell pair, and the difference from the collective diagonal.
