# Family A, round 1: collective ferromagnetic p=3

**Verdict: FAIL.**

The pre-set gate was: at N=8, 10, and 12, for one common Γ, L=1 squared ground-state overlap F lies in [0.02, 0.25], and F then rises materially through L=3..5 (F5−F1 ≥ 0.10, with F3>F1 and F5>F1). Zero of 21 (N, Γ) points pass the low-L1 half. All 21 have F1>0.50. The reference shape — low L=1, steady growth, absolute F often below 0.4 — is not what this Hamiltonian does at these sizes.

The search itself is intact. Energy is non-increasing in L by the zero-pad construction. Pauli, full-matrix, and total-spin checks agree. Odd-p sign flip and statevector replay agree to 1e-14. High overlap here is a property of a weakly entangled coherent ground state, not a claim of entangled many-body control.

## Model

```text
H = -N m_z^3 - Γ Σ_i X_i,    m_z = (Σ_i Z_i)/N
  = -(Σ Z)^3 / N^2 - Γ Σ X
```

Z|0⟩=+1, Z|1⟩=−1. This is Wauters–Mbeng–Santoro Eq. (1) at p=3 with transverse field h=Γ (arXiv:2003.07419, HTML opened). The infinite-N mean-field energy density −cos³θ − Γ sinθ has a first-order point at Γ_c=3√3/4≈1.299038 and a spinodal at Γ=3/2. Those are not finite-N eigenvalues. The dashed line on `results/gs_phase_diagram.png` is Γ_c.

### Collective cubic versus distinct-index triples

Exact integer identity, failed on 0 of 2^N basis states at N=6, 8, 10, 12 (`pauli_enum.py`, independent of the Dicke solver):

```text
(Σ_i Z_i)^3 = 6 Σ_{i<j<k} Z_i Z_j Z_k + (3N−2) Σ_i Z_i
```

So the diagonal cost is a three-body sum plus a one-body field:

```text
-N m_z^3 = c3 Σ_{i<j<k} ZZZ + c1 Σ_i Z
c3 = -6/N^2,   c1 = -(3N−2)/N^2
```

The field energy is c1⟨Σ Z⟩=−(3−2/N)⟨m_z⟩, which is O(1), while the collective energy is O(N). It is not negligible at N≤12. On the symmetric-subspace ground state at Γ=0.6 the field expectation is −2.64 (N=8), −2.75 (N=12), and −2.87 (N=32). Dropping it shifts that ground energy by −2.61, −2.73, and −2.86 respectively (`gs_scan.csv`). Term lists with exact fractions are `results/hamiltonian_terms_N{6,8,10,12}.csv`.

## Symmetries that were checked, not assumed

- H commutes with total spin. Every irrep was diagonalized. At N=6 and N=8 the irrep spectra, with the standard multiplicities, reproduce the full 2^N spectrum to 1e-14. The ground state is in S=N/2 for all 189 points of the fine Γ grid at N=6..32. Full-matrix ground states at N=6, 8, 10 match the embedded symmetric vector with overlap 1, and ⟨S²⟩=S(S+1). N=12 was checked by residual of the embedded vector (1e-15), not by a dense 4096 diagonalization of every Γ. N=16, 20, 32 use the irrep scan only.
- ∏_i Y_i sends both ΣZ and ΣX to minus themselves, so H→−H. Therefore E_max=−E_0. The full spectra at N=6, Γ=0.7 and N=8, Γ=1.2 match their negatives to 1e-14, and |E0+Emax| on the whole scan is at most 3e-14. Pagano η uses these full-spectrum edges.
- ∏_i X_i flips the sign of the cubic term, commutes with ΣX, and fixes |+⟩. The same angles therefore have the same fidelity on +N m_z^3 and −N m_z^3. Over every stored schedule, |ΔF|≤2.7e-14 and the two ground energies agree.
- Mixer period is π, not π/2: exp(−iπ Σ X)=(−1)^N. Cost period is the minimal T=2π N²/gcd(Δm³). For every even N here the gcd is 8, so T=π N²/4. In the Wauters generator that is a γ_W period of π/4, smaller than the π cell they quote. Half of T is not a period. Angles were folded only into the minimal-absolute cell of these two periods. Conjugation (γ,β)→(−γ,−β) was checked (energy error 0) and was not used to rewrite schedules. Folds that actually moved an angle are flagged `beta_period_pi_fold` in `angles.json`. Gamma folds did not occur; the optima already sat inside (−T/2, T/2).

## QAOA convention

Layer 1 acts first, cost then mixer:

```text
|ψ⟩ = Π_l exp(−i β_l Σ_i X_i) exp(−i γ_l H_cost) |+⟩^N
H_cost = −(Σ Z)^3 / N^2
```

β multiplies Σ X, so one qubit sees exp(−i β X), which is RX(2β) in the Qiskit convention. The Wauters pair is γ_W=γ/N² (generator −(Σ Z)^3) and β_W=−β (generator −Σ X). The objective is ⟨H⟩, not fidelity. η=(E−E_max)/(E0−E_max) with the full-spectrum edges above. A separate L=1 maximization of F is only a diagnostic.

Search, per (N, Γ), L=0..5: L=1 exhaustive grid over one γ period (spacing ~0.05, plus a focus window |γ|≤6) and local BFGS from the best basins, random starts, and short linear-anneal seeds. L=2 seeds the Pagano shifts δ∈{0.05,0.1,0.2,0.4} on (γ,β), not on γ_W. L≥3 uses the supplement interpolation (linear at depth 3, cubic after that) plus an end-point ±0.2 nudge. Every depth also keeps the exact zero-layer pad of the previous best energy, so the recorded energy cannot rise. Random multistarts and linear-schedule seeds compete with that bootstrap. Shared-γ (one γ, independent β_l) is a second run, not the primary ansatz. It was executed for all Γ at N≤12 and, at larger N, where the ground-state overlap with |+⟩ on the fine grid was ≤0.3.

N-transfer of angles across system size was not used.

## Finite-size target

Γ grid finally used: 0.60, 0.80, 1.00, 1.15, 1.25, 1.35, and 1.40. The extra point is the N=8 edge where ⟨m_z⟩ of the ground state drops through 1/2. The minimum full gap on a 27-point Γ scan sits near the mean-field point and stays O(1):

| N | min gap | Γ | ⟨m_z⟩ | half-cut entropy (bits) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 1.76 | 1.25 | 0.67 | 0.18 |
| 12 | 1.38 | 1.35 | 0.57 | 0.30 |
| 20 | 0.89 | 1.35 | 0.56 | 0.50 |
| 32 | 0.46 | 1.35 | 0.46 | 0.79 |

No gap scaling is fitted. Four sizes do not justify an exponential-gap claim. On the ordered side the ground state is a spin coherent product: best identical-qubit coherent overlap is ≥0.98 for Γ≤1 at N=12 and ≥0.995 at N=32, Γ≤1. Half-cut von Neumann entropy is for qubits `[0, N//2)`, in bits. For a symmetric state every cut of that size agrees. Entropy stays below 0.3 bit at N≤12 except near Γ≈1.35, where it reaches 0.30 (N=12) and 0.79 (N=32). That is macroscopically polarized collective order. It is not evidence of entangled many-body preparation. The symmetric irrep itself cannot hold more than log2(N/2+1) bits across any cut.

## Fidelity at the best energy

F = |⟨ψ|GS⟩|². Marker values below are the fidelity of the lowest-energy schedule, not the highest fidelity found. Bars on the plots are the min-to-max fidelity among local searches, most of which miss the good basin (L=1 endpoint F often ranges from ~0 to the reported value; the near-best-energy subset agrees to 1e-9).

| N | Γ=0.6 | 0.8 | 1.0 | 1.15 | 1.25 | 1.35 | 1.40 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8, L=1 | 0.955 | 0.985 | 0.981 | 0.932 | 0.876 | 0.810 | 0.778 |
| 10, L=1 | 0.931 | 0.973 | 0.986 | 0.939 | 0.862 | 0.762 | 0.712 |
| 12, L=1 | 0.907 | 0.960 | 0.987 | 0.949 | 0.860 | 0.720 | 0.646 |
| 8, L=5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12, L=5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32, L=1 | 0.687 | 0.821 | 0.938 | 0.984 | 0.957 | 0.477 | 0.841 |
| 32, L=5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.891 | 0.999 |

Lowest F1 at the gate sizes is 0.646 (N=12, Γ=1.40). The L=1 fidelity diagnostic is within ~0.02 of the energy optimum on the ordered side (both high). Near the small gap they separate: at N=32, Γ=1.35 the energy optimum has F=0.477 while fidelity maximization reaches 0.655. Even that diagnostic stays far above 0.25. The ordered-side result is not a failed optimizer.

Where F1 is already above 0.9, L=5 adds little because there is a ceiling at 1. Where F1 is lower, the climb is front-loaded: N=12, Γ=1.40 goes 0.646 → 0.976 from L=1 to L=2. N=32, Γ=0.60 goes 0.687 (L=1) → 0.687 (L=2, zero-pad kept; another local minimum reached F=0.793 at a worse energy) → 0.998 at L=3. In this batch the best-energy fidelity never decreased with L. That is an observation about these minima. Energy minimization does not imply it. The one L=2 case above is a concrete counterexample to reading a higher-depth search as a higher overlap: a worse-energy L=2 state had higher overlap than the recorded energy minimum.

η is a poor companion to F here. At N=32, Γ=1.25, the L=0 state |+⟩ has F=0.010 and η=0.981, because E(|+⟩) is already far from E_max on an extensive spectrum. η then moves only from 0.981 to 0.998 while F moves from 0.01 to 0.96.

## Angles and baselines

On the ordered side the L=1 energy minimum is a single large cost angle plus a mixer angle at π/4. Example, N=12, Γ=0.8, no period fold required:

```text
γ = −19.0064,   β = 0.78540 ≈ π/4
γ_W = γ/N^2 = −0.1320,   β_W = −β
```

The same γ_W≈−0.132 appears from N=8 to N=32 at Γ=0.6. It is not claimed as a closed form. The L=5 winner at that same point is five γ's still near −19, with β decreasing (0.96, 0.38, 0.34, 0.28, 0.09). That is a flat cost schedule, not a Pagano ramp that starts near zero. Sum |γ| at L=1 scales as ~0.13 N²; sum |β| stays O(1).

A digitized linear ramp forced to reuse that sum |γ| and sum |β| matches QAOA at L=1 (one layer is the schedule) and collapses at L=5: N=12, Γ=1, matched-ramp F=0.007 against QAOA F=1.000. A separate τ sweep of the Wauters linear Trotter (s_m=m/L, τ from 0.05 to 30, not budget-matched) also stays well below the ordered-side QAOA fidelity (N=20, Γ=0.6, L=5: Trotter F=0.037 at τ=2.15, QAOA F=1). This says the optimum found here is not a short linear ramp. It does not say QAOA beats adiabatic annealing, and it does not test the exponentially small gap discussed by Wauters via Jörg et al. Shared-γ reaches F>0.99 at N=12 on the ordered side but stalls at larger N (N=32, Γ=0.6, L=5: shared F=0.34, independent F=1.00). Independent angles are doing real work there; they are not required to explain the failed low-L1 gate, which already fails at L=1 where the two ansätze coincide.

Statevector replay of the stored L=1 and L=5 angles at N≤12 matches the symmetric-block state with overlap at least 1−1e-14 (56 replays). The block evolution was also checked against `scipy.linalg.expm` during validation.

## What this does not say

- No scaling law and no speedup. L≤5 and seven sizes, with a ceiling at F≈1, are not a test of Wauters' P∝N threshold.
- No reading of the high fidelity as entanglement. Coherent-product overlaps above 0.9 and half-cut entropies below half a bit are collective mean-field control.
- Fidelity was not optimized, except as the L=1 diagnostic. Zero-padding forces energy down, not fidelity up.
- Glass, distinct-index QAOA, and the Seki–Nishimori non-stoquastic deformation were not run.

## Next direction

Stop using this family as a Pagano-style depth curve. The ground state on the requested window is close to a spin coherent product, and one layer with β≈π/4 already lands on it (F1≥0.65 at N≤12, usually ≥0.9). A later family would need a target whose ground state is not that coherent state — a distinct-index interaction without a uniform product ground state — before a low L=1 and a slow L/N climb are even plausible. That was not started.

## Reproducibility

```bash
cd pspin_nov6/collective_p3
python3 validate.py
python3 run_round1.py --out results
```

Code: `pauli_enum.py` (Pauli list and identity), `symmetric.py` (irreps, evolution, entropy), `qaoa_search.py` (bootstrap, shared-γ, anneals), `validate.py`, `run_round1.py`. Numbers: `results/qaoa_round1.csv`, `results/angles.json`, `results/gs_scan.csv`, `results/gate.json`. Plots: `F_vs_invN_by_depth.png`, `F_vs_L_over_N.png`, `gs_phase_diagram.png`. Log: `results/verification.log`. Sources opened: `SOURCE_NOTES.md`.
