# Round 1 results — sparse uniform-sign 3-body QAOA

This run is Family B only: a sparse, distinct-index, uniform-sign 3-body model. It is a sum of N selected triples on a 3-regular simple hypergraph. It is a different operator from the collective p-spin `-(sum Z)^3/N^2`, which uses every triple and generates a longitudinal field. Couplings are all +1. There is no random-sign glass.

```
H = - sum_a Z_i Z_j Z_k - Gamma * sum_i X_i
```

QAOA minimizes the expectation of this full Hamiltonian. The target state is the lowest eigenvector of the full 2^N matrix. F is the squared overlap with that vector. `eta = (Emax - E) / (Emax - E0)`. The half-cut entropy is the von Neumann entropy, in bits, of qubits `0 .. N/2-1` (the low bits of the computational index).

## Sample

Four hypergraphs per N in {6, 8, 10, 12}, taken as the first distinct full-rank simple 3-regular configuration-model draws in seed order. No energy or fidelity filter. Every graph is reported.

| N | graph ids | linear |
|---|---|---|
| 6 | seed 11, 261, 285, 595 | all false; a linear 3-uniform hypergraph is impossible because 3N = 18 > C(6,2) = 15 |
| 8 | seed 5, 18, 22, 28 | all false |
| 10 | seed 27, 30, 121, 166 | all false |
| 12 | seed 8, 12, 17, 29 | all false |

At Gamma = 0 each of these graphs has a unique classical ground state, the all-+ product, energy -N. That product limit is the Gamma → 0 endpoint of the curves below.

Prescribed fields: 0.3, 0.5, 0.7, 0.9, 1.1, 1.3. Two further fields were added from the ground-state survey, before any QAOA fidelity was computed: 0.95 (entropy rising, overlap with |+> still modest) and 1.0 (minimum median gap on the survey grid).

## Checks that passed

26 construction checks passed, including an independent Kronecker build of each ZZZ, Lanczos versus dense eigh, circuit replay against `expm`, period π for both γ and β, failure of period π/2, the conjugation (γ, β) → (−γ, −β), a two-qubit gauge flip, and the odd-p global sign flip (same angles, fidelities agree). A separate N=12 dense audit at Gamma = 1 on `N12_seed8` matched Lanczos with overlap 1 and an energy difference of 2e-14.

Conjugation by `prod_i Y_i` sends H to −H, so Emax = −E0 on every surveyed point (`emax_plus_e0` stayed below 1e-6).

All 128 QAOA cells finished. The saved angles replay to the reported energy. Folding every angle by its π branch reproduces the optimized state. The L=1 energy minimizer beat the L=1 grid on every cell. An independent L=1 fidelity maximization did not find a lower energy; the fidelity it gained over the energy minimizer is about 0.003–0.01, while its energy is worse. The low fidelities are properties of the landscape.

Energy is nested in depth on every ladder, because a zero-layer pad of the previous schedule is always a seed. Fidelity is not forced to be nested. Nine of 128 ladders have a downward fidelity step. The clearest is `N6_seed261` at Gamma = 0.3: F = 0.100, 0.118, 0.547, 0.528, 0.957 at L = 1..5, while the energy kept falling. Most of the other eight are flat to three digits.

Angle branch indices were nonzero on 1507 layer-counts across the run. Durations below use the principal branch in (−π/2, π/2]. The largest jump between consecutive principal angles is 3.05.

## Ground state

The median gap has a minimum at Gamma = 1.0: 0.73 (N=6), 0.49 (N=8), 0.42 (N=10), 0.31 (N=12). The half-cut entropy peaks in the same window, about 1.6 bits at N=12 (a 6-qubit cut, whose maximum is 6 bits). The best product-state overlap is lowest there as well, about 0.62 at N=12, Gamma = 1.0. On the ordered side of this window the ground state is a dressed product: at Gamma ≤ 0.7, F_product ≥ 0.99 and the half-cut entropy is below 0.08 bits for N ≥ 8.

Overlap with |+> and the ordered product cross near Gamma = 1. Median |<+>|GS>|^2 at Gamma = 1.0 is 0.51, 0.44, 0.38, 0.34 for N = 6, 8, 10, 12. By Gamma = 1.1 that overlap is already 0.65–0.68, and by Gamma = 1.3 it is about 0.8.

These four sizes show a shrinking but still O(1) minimum gap. They do not resolve an exponentially small gap. The exponential-gap statement in the fully connected p-spin literature was not re-derived here and is not a result for this sparse uniform-sign ensemble.

## Fidelity

Medians over the four graphs. Full per-graph ladders are in `per_graph_fidelity.json` and `qaoa_independent.csv`.

### Adaptive Gamma = 0.95 (closest Pagano-shaped cut)

| N | F(L=0) | F(L=1) | F(L=2) | F(L=3) | F(L=4) | F(L=5) | F5−F1 | S_gs | F_product |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.415 | 0.534 | 0.561 | 0.584 | 0.604 | 0.730 | 0.197 | 0.72 | 0.84 |
| 8 | 0.303 | 0.414 | 0.450 | 0.486 | 0.512 | 0.531 | 0.117 | 0.77 | 0.83 |
| 10 | 0.215 | 0.320 | 0.354 | 0.371 | 0.376 | 0.389 | 0.070 | 0.80 | 0.83 |
| 12 | 0.139 | 0.222 | 0.252 | 0.270 | 0.281 | 0.293 | 0.073 | 0.77 | 0.86 |

F rises with depth and falls with N. At N=12, F stays near 0.3 at L=5. Instance spread at N=12 is F1 ∈ [0.218, 0.249] and F5 ∈ [0.278, 0.340]. Residual energy E−E0 at L=5 is 0.32–0.40, comparable to the gap (median 0.39). L=1 sits inside 0.02–0.25 only at N=12. At N=8, L=1 is 0.41.

### Ordered side, where L=1 is low and the ground state is almost a product

Median F5−F1 at N=12 is 0.026 (Gamma 0.3), 0.032 (0.5), 0.029 (0.7). Median F1 at N=12 is 0.009, 0.012, 0.022. F_product is 0.994–1.000 and the half-cut entropy is ≤ 0.05 bits. This is the dressed all-+ product, not an intermediate entangled target.

At N=6 the same fields are a different story. Median F5 is 0.978, 0.994, 0.914. Two of the four graphs at Gamma = 0.3 reach F = 0.999 and 1.000 with E−E0 < 0.003. One graph at Gamma = 0.7 stays at F5 = 0.377. The N=6 jump does not survive: at Gamma = 0.3, median F5 is 0.129 (N=8), 0.056 (N=10), 0.036 (N=12).

### Easy paramagnet

At Gamma = 1.1, median F1 is already 0.81–0.89 and F5 is 0.90–0.95. At Gamma = 1.3, F1 is 0.94–0.96 and F5 is 0.96–1.00. These cells are the easy band the search was asked to exclude. Depth still adds a few points of fidelity because L=0 is not yet 1.

### Gap-minimum field Gamma = 1.0

Median F1 is 0.64, 0.58, 0.53, 0.51 and F5 is 0.80, 0.69, 0.63, 0.60 for N = 6, 8, 10, 12. The ground state is the most entangled point in the scan (S_gs = 1.59 bits and F_product = 0.62 at N=12) and the overlap with |+> is already 0.34 before any layer. L=1 is above 0.5 for N ≤ 10.

## Baselines and shared gamma

At every N ≥ 8 and every Gamma, L=5 QAOA fidelity beat both annealing baselines on all four graphs.

The matched baseline is a first-order Trotter of H(s) = (1−s)(−∑X) + s H, with one scale chosen so sum|γ| + sum|β| equals the principal-branch QAOA duration. β is negative in that decomposition. The fixed baseline uses the same shape with dt = 0.5 per slice and does not inherit the QAOA angles. Neither baseline was itself optimized. On the ordered side the matched anneal stays near F = 0 at L=5 (median 0.00–0.01 for Gamma ≤ 0.7 at N=12), so the comparison there is against a short linear schedule. Beating it is a statement about these two schedules.

Shared gamma (one γ, independent β_ℓ) tracks the independent ladder and is a bit lower at L=5. At Gamma = 0.95, N=12, the median shared-gamma F5 is 0.252 against independent 0.293. At Gamma = 1.3 the two are close because both are already near 1.

## Scaling

Per-layer gain (F5−F1)/4, medians:

| Gamma | N=6 | N=8 | N=10 | N=12 | r² vs 1/N | r² vs 2^(−N/2) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.043 | 0.028 | 0.017 | 0.011 | 0.996 | 0.981 |
| 0.95 | 0.049 | 0.029 | 0.017 | 0.018 | 0.950 | 0.976 |
| 1.00 | 0.040 | 0.028 | 0.024 | 0.024 | 0.929 | 0.966 |

Four sizes cannot establish an asymptotic law. At Gamma = 0.95 and 1.0 the N=10 and N=12 gains are essentially equal, which a pure 2^(−N/2) decay would not do (that factor drops by two between those sizes). A high r² on four decreasing points is not a scaling collapse. Plots of F against 1/N and against L/N are in this directory; the curves for different N do not fall on one line.

## What this round does and does not show

The quantitative depth separation worth quoting is the Gamma = 0.95 median ladder: from N=8 to N=12, F1 goes 0.41 → 0.32 → 0.22 and F5 goes 0.53 → 0.39 → 0.29, with F5−F1 about 0.07–0.12. Fidelity grows with depth, shrinks with N, and stays far from 0.99 for N ≥ 8.

The ordered-side N=6 result F5 ≈ 0.99 is real on those graphs and is the wrong success condition. It is a small-N preparation of a product ground state. At N=12 the same fields leave F5 ≈ 0.04.

No field in the scanned window puts all of the following together at N=8, 10, and 12: L=1 fidelity inside 0.02–0.25, a depth gain of order 0.15 or more, and a ground state that is both entangled (half-cut entropy ≳ 0.5 and product overlap ≲ 0.7) and far from |+>. The low-F1 side is a product state. The entangled side already has a large overlap with |+>.

## Hypotheses that were tested on these sizes

- Uniform +1 on a full-rank 3-regular hypergraph has a unique classical all-+ ground state, and a gauge flip or a global sign flip leaves QAOA fidelity unchanged. Both held in the validation.
- A first-order transition with an exponentially small gap, as discussed for fully connected p-spin models, is not measured here. The sparse-model minimum gap from N=6 to N=12 is 0.73 → 0.31.
- An L/N fidelity collapse is not established. The Gamma = 0.9 gain happens to track 1/N across these four sizes; neighboring fields do not separate 1/N from other decreasing functions, and N=10 matches N=12 at Gamma ≥ 0.95.
- Half-cut entropies of 1–2 bits at N=12 occur at Gamma ≈ 1.0–1.2. They occur together with |<+>|GS>|^2 ≈ 0.3–0.7, so L=1 is no longer low.
- Source identifiers cited from memory in the planning note were not re-read for this run. The numbers above come from this directory's spectra and QAOA output.

## Next

- Keep this ensemble as the uniform-sign control. The N≥8 ordered side shows what a product ground state looks like under L ≤ 5: low fidelity, small depth slope, entropy near 0.
- A depth extension L ~ N/2 on one N=12 graph at Gamma = 0.7 would show whether the N=6 climb to F ≈ 1 is just L comparable to N. That is a different question from the L/N fan at fixed small L.
- The finite-size crossing moves only slowly. A larger-N gap estimate (the minimum near Gamma = 1) is the direct test of whether the gap later collapses. These exact diagonalizations stop at N=12.
- If the goal is low L=1 fidelity and a non-product ground state at the same field, this uniform-sign ferromagnet does not supply that window for N=8–12 at L ≤ 5. A later round can change the graph family. This round did not add random signs.
