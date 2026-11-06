# Round 2: two-window circulant 3-body ring

**Verdict: not useful as a low-L1 entangled target.** The Hamiltonian does not collapse and it is not gauge-equivalent to a ferromagnet, at any N from 6 to 14. The ground state that QAOA actually sees is still the translation orbit of one product state (orbit overlap 0.86–0.98). L=1 already reaches squared overlap 0.84–0.94 at N=8, 9, 10, and 11. The only low L=1 values are at N=12, on that orbit.

## Hamiltonian, checked before any optimizer

```text
H = -sum_i Z_i Z_{i+1} Z_{i+3} + kappa * sum_i Z_i Z_{i+2} Z_{i+3} - Gamma * sum_i X_i
```

Periodic boundaries. Z|0⟩=+1, Z|1⟩=−1. Kappa was 1, with one side row at 0.5. Normalization is in Gamma; the ZZZ coefficients are not divided by N.

Two independent diagonals agree. One loops over clause parities. The other builds each ZZZ by a Kronecker product of Pauli matrices, with qubit 0 the fastest index. Max absolute difference is 0 at N=6, 7, and 8 for both kappas. Windows do not share a support for N=6..14, so no coefficients cancel. Each N has 2N distinct triples.

Reflection i → −i exchanges the patterns (0,1,3) and (0,2,3). Their coefficients are −1 and +kappa, so the reflected diagonal is a different operator. The max |E(x)−E(Rx)| is 16 at N=8 and 32 at N=12 for kappa=1. Reflection does not commute with H. The brief's k=0 reflection-even target was not used.

Translation does commute (diagonal mismatch 0). The X-graph is the hypercube, which is connected, and every off-diagonal entry of H is 0 or −Gamma. For Gamma>0 the ground state is unique and can be chosen strictly positive. A positive ground state of a translation-invariant Hamiltonian is momentum 0. Dense diagonalizations at N=6, 7, and 8 match that: gap > 0.2 in the checked cases, minimum amplitude positive, ||T|GS⟩−|GS⟩|| < 1e-14. QAOA and the reported overlaps use this full-space ground state. Momentum 0 was a check, not a hand-picked sector that might have dropped the ground state.

Global Y = ⊗ Y_i sends every Z and every X to minus itself, so H → −H. At N=8, kappa=1, Gamma=1, E_max = −E_0 = 11.29969338. Eta uses those edges: (E−E_max)/(E_0−E_max).

Global X flips every triple and fixes |+⟩. The same angles give the same fidelity on the sign-flipped couplings (absolute difference 8e-16 on a random two-layer test at N=8).

## Frustration is real, including at multiples of 7

Satisfiability is the GF(2) system

```text
x_i + x_{i+1} + x_{i+3} = 0
x_i + x_{i+2} + x_{i+3} = 1
```

Row reduction, not a divisibility argument: for every N=6..14 the coefficient rank is N and the system is inconsistent. There are zero satisfying assignments, including at N=7 and N=14. The brief's suggestion that those two sizes might become gauge-solvable is false for these signs. Enumeration agrees: the classical minimum sits above −N(1+kappa).

| N | rank | satisfiable | degeneracy | E_min | fully satisfied | excess |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 6 | 6 | no | 6 | −8 | −12 | 4 |
| 7 | 7 | no | 14 | −8 | −14 | 6 |
| 8 | 8 | no | 32 | −8 | −16 | 8 |
| 9 | 9 | no | 72 | −8 | −18 | 10 |
| 10 | 10 | no | 160 | −8 | −20 | 12 |
| 11 | 11 | no | 352 | −8 | −22 | 14 |
| 12 | 12 | no | 6 | −16 | −24 | 8 |
| 13 | 13 | no | 26 | −16 | −26 | 10 |
| 14 | 14 | no | 84 | −16 | −28 | 12 |

Kappa=0.5 has the same signs and is frustrated as well (N=12 excess 6, degeneracy 6).

## The ground state is an orbit of a product

Half-chain entropy is the von Neumann entropy, in bits, of the contiguous arc `[0, N//2)`. It is several bits. That entropy is mostly the translation superposition. The maximum overlap with the normalized orbit of one real product state is:

| N | Gamma | F_orbit | F with plus | entropy (bits) | gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.0 | 0.877 | 0.506 | 2.00 | 0.444 |
| 8 | 1.3 | 0.912 | 0.652 | 1.57 | 0.922 |
| 10 | 1.0 | 0.867 | 0.543 | 1.79 | 0.750 |
| 11 | 1.0 | 0.867 | 0.563 | 1.66 | 0.852 |
| 12 | 1.0 | 0.980 | 0.071 | 2.59 | 0.023 |
| 12 | 1.2 | 0.912 | 0.231 | 2.47 | 0.146 |

At N=12, Gamma=1, the classical ground space is a single 6-cycle and the positive ground state is that orbit (overlap 0.98) with a gap of 0.023. At N=8 the classical space is four orbits of length 8; one optimized product orbit still reaches 0.85–0.91, so the transverse-field ground state has not left the orbit-product regime. This is the kill the brief called K1. It is a weaker product than round-1 collective p=3 (coherent overlap ≥0.95 there), but it is the same kind of failure: the target is a symmetrized product, and the entropy is not evidence of a hard entangled phase.

## L=1 is high except on that one orbit

F1max is the best squared overlap at one layer, from a grid over the gamma period and a local fidelity maximization. F_E is the overlap of the energy minimum. They agree to about 0.02, so this is not the round-1 pure-p4 failure mode (energy-selected F looked hard while the fidelity maximum was 0.9).

| N | Gamma=1.0 F1max | F_E | Gamma=1.3 F1max | F_E |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.837 | 0.814 | 0.903 | 0.889 |
| 9 | 0.839 | 0.825 | 0.901 | 0.892 |
| 10 | 0.873 | 0.863 | 0.921 | 0.915 |
| 11 | 0.901 | 0.895 | 0.939 | 0.935 |
| 12 | 0.179 | 0.165 | 0.595 | 0.583 |

At N=12, Gamma=1.2, F1max=0.411 and F_E=0.397. Kappa=0.5 at N=12, Gamma=1 gives F1max=0.74 on a coarser grid of the longer 2π period; a finer grid could only raise it.

Gamma period for kappa=1 is π/4, not π. The gcd of the cost eigenvalues, in half-integer units, is 16, and a direct phase check gives a global phase at T=π/4 and not at T/2. Beta has period π and not π/2. Folds into the minimal-absolute cell are flagged in `results/ladder.json` (`gamma_period_fold`, `beta_period_pi_fold`). Folded angles were re-evaluated; a mismatch would have been tagged and was not.

## Depth at the only low-L1 size

N=8–11 were not given a deep search: L=1 is already above 0.83. At N=12 the ladder minimizes energy with the Pagano seeds (L=2 shifts 0.05, 0.1, 0.2; higher depth linear then cubic interpolation), zero-padding, and 8 random restarts, BFGS capped at 40 iterations. That is smaller than a 20-restart search. The overlaps below are therefore a lower envelope of this budget, not a certified maximum. A 20-restart run at full iteration count did not finish inside the batch.

| Gamma | L=1 | 2 | 3 | 4 | 5 | 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2, F at best energy | 0.397 | 0.485 | 0.535 | 0.584 | 0.584 | 0.655 |
| 1.0, F at best energy | 0.165 | 0.240 | 0.290 | 0.352 |  |  |

At Gamma=1.2 the gain from the fidelity-maximizing L=1 value 0.411 to the L=6 energy minimum 0.655 is 0.24. That is larger than the round-1 sparse near-miss (about 0.07 at N=12), and L=5 added nothing because the zero-pad of L=4 remained the best energy. The endpoint fidelities at each depth stay within a few hundredths of that best-energy fidelity, so the ladder is not hiding a much better overlap at slightly worse energy. Shared-gamma at L=4, as a side check only, reaches F=0.482 against the independent-angle F=0.584.

Eta at Gamma=1.2, L=6 is already about 0.983 while the overlap is 0.655. The spectrum is wide (E_max=−E_0), so eta moves before the state is the ground state. The same thing happened for collective p=3.

## What this batch does not say

There is no three-size window where L=1 overlap is low and later energy-selected depths are much higher on a target that is not an orbit-product. There is no scaling fit and no speedup claim. The depth numbers are one size, a reduced restart budget, and a ground state whose overlap with one product orbit is 0.91 (Gamma=1.2) or 0.98 (Gamma=1.0).

## Files

`hamiltonian.py`, `verify_algebra.py`, `observables.py`, `qaoa.py`, `run_round2.py`. Tables: `results/l1_survey.csv`, `results/depth_ladder.csv`, `results/ladder.json`. Plots: `results/F1max_vs_Gamma.png`, `results/F_vs_L_N12.png`. Term lists: `results/hamiltonian_terms_N{8,10,12}_k1.csv` and `results/hamiltonian_terms_N12_k0.5.csv`.
