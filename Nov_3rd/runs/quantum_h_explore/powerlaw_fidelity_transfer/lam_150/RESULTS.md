# Power-law TFIM fidelity transfer — λ=1.5

**Verdict: needs_edit**

L*=2 median train/hold F≥0.95 on n=6 and n=9 (train 0.984/0.977, hold 0.983/0.972). Frozen median F is 0.911 at n=12 and 0.829 at n=15; n=15 (0.829) is under the 0.85 floor.

Fidelity is the pure-state overlap squared, F = |⟨ψ|GS⟩|². The gate is the median over three seeds of each seed's median over instances. L* is the smallest depth in {1,2,4,8,12,16} with that median ≥ 0.95 on the train and hold splits of both n=6 and n=9, for Walsh SNAP B=2. Frozen F ≥ 0.85 at both n=12 and n=15 is the pre-registered reading of a freeze that stays reasonable. Fourier is trained on the same fidelity loss and is a reference, not the arm the verdict is about.

## Hamiltonian

μ = 0, α = 1.5, J_ij ∈ {±1} on the open chain. Qubit 0 is the MSB. Cavity c owns sites [3c, 3c+3). Cross-cavity pairs stay in the chain; they are not dropped. The circuit phases use Cn = H_ZZ / n. H_ZZ is pure Walsh degree 2, so SNAP trains a single γ_2 per layer. The shell is φ_k = c₁ k + c₂ k² on popcount k=0..3, stored in orthonormal coordinates of that span (the B=2 ablation metric). Paper init φ_k = −β k lies in the span, and at that init SNAP matches Fourier.

Cross-cavity fraction of Σ|K_ij| (sign-independent): n=6 0.358, n=9 0.456, n=12 0.503, n=15 0.532.

## Phase diagnostic

Bitstring peak is max_z |⟨z|GS⟩|². Product gap is (E_prod − E_0)/|E_0|, with E_prod the best XZ product energy found. ⟨X⟩ is (1/n) Σ_i ⟨X_i⟩. F(+) is |⟨+|GS⟩|². ΔE is the gap to the first excited state on n≤9. Comparison λ use the first 32 instance indices; λ=1.5 also reports the train pool [0:64) and the hold pool [64:96).

| λ | n | slice | B | peak | product gap | ⟨X⟩ | F(+) | ΔE | quantumness |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 6 | prefix32 | 32 | 0.335 | 0.009 | 0.424 | 0.274 | 0.0350 | fail |
| 0.50 | 9 | prefix32 | 32 | 0.327 | 0.004 | 0.355 | 0.071 | 0.0034 | fail |
| 1.00 | 6 | prefix32 | 32 | 0.130 | 0.060 | 0.766 | 0.672 | 0.4967 | pass |
| 1.00 | 9 | prefix32 | 32 | 0.094 | 0.041 | 0.707 | 0.456 | 0.2579 | pass |
| 1.50 | 6 | hold | 32 | 0.085 | 0.071 | 0.883 | 0.831 | 1.1782 | pass |
| 1.50 | 6 | prefix32 | 32 | 0.071 | 0.076 | 0.892 | 0.844 | 1.3469 | pass |
| 1.50 | 6 | train | 64 | 0.080 | 0.075 | 0.885 | 0.835 | 1.2480 | pass |
| 1.50 | 9 | hold | 32 | 0.033 | 0.063 | 0.863 | 0.719 | 0.9381 | pass |
| 1.50 | 9 | prefix32 | 32 | 0.030 | 0.067 | 0.872 | 0.736 | 1.0193 | pass |
| 1.50 | 9 | train | 64 | 0.028 | 0.067 | 0.872 | 0.737 | 1.0529 | pass |
| 1.50 | 12 | freeze | 32 | 0.009 | — | 0.866 | 0.652 | — | — |
| 1.50 | 15 | freeze | 32 | 0.003 | — | 0.862 | 0.574 | — | — |
| 2.00 | 6 | prefix32 | 32 | 0.050 | 0.056 | 0.940 | 0.912 | 2.2744 | pass |
| 2.00 | 9 | prefix32 | 32 | 0.015 | 0.059 | 0.930 | 0.850 | 1.9450 | pass |

At λ=1.5 on the shared prefix, n=6 has median peak 0.071, product gap 0.076, ⟨X⟩ 0.892, F(+) 0.844; n=9 has peak 0.030, gap 0.067, ⟨X⟩ 0.872, F(+) 0.736. On that same n=6 prefix the smaller fields are less X-disordered: peak 0.335 (λ=0.50) and 0.130 (λ=1.00), ⟨X⟩ 0.424 and 0.766, F(+) 0.274 and 0.672, product gap 0.009 and 0.060. The cheap λ=2.00 probe on the n=6 prefix goes further: peak 0.050, ⟨X⟩ 0.940, F(+) 0.912.

## Ladder

Train indices [0:64), hold [64:96), freeze n=12 and n=15 indices [0:32). Seeds [0, 1, 2]. L-BFGS 40 rounds × max_iter 8, lr 0.25, minibatch 32 split across n=6 and n=9, checkpoint on min(train median F at n=6, n=9). The hold split is not used to pick the round. No retrain at n=12 or n=15.

| L | arm | train F n6 | hold F n6 | train F n9 | hold F n9 | freeze F n12 | freeze F n15 | Medε n12 | Medε n15 | Med(ε/|E0|) n15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | snap_b2 | 0.967 | 0.965 | 0.950 | 0.937 | 0.884 | 0.809 | 0.5073 | 0.9301 | 0.0366 |
| 1 | fourier | 0.967 | 0.964 | 0.949 | 0.936 | 0.882 | 0.807 | 0.5115 | 0.9354 | 0.0368 |
| 2 | snap_b2 | 0.984 | 0.983 | 0.977 | 0.972 | 0.911 | 0.829 | 0.4421 | 0.8870 | 0.0350 |
| 2 | fourier | 0.984 | 0.984 | 0.976 | 0.971 | 0.910 | 0.828 | 0.4435 | 0.8875 | 0.0350 |
| 4 | snap_b2 | 0.985 | 0.984 | 0.980 | 0.975 | 0.916 | 0.833 | 0.4224 | 0.8629 | 0.0340 |
| 4 | fourier | 0.987 | 0.985 | 0.981 | 0.977 | 0.914 | 0.831 | 0.4178 | 0.8745 | 0.0345 |
| 8 | snap_b2 | 0.986 | 0.985 | 0.984 | 0.979 | 0.925 | 0.848 | 0.3547 | 0.7582 | 0.0299 |
| 8 | fourier | 0.987 | 0.984 | 0.986 | 0.982 | 0.921 | 0.840 | 0.3760 | 0.8092 | 0.0319 |
| 12 | snap_b2 | 0.985 | 0.984 | 0.985 | 0.980 | 0.926 | 0.849 | 0.3521 | 0.7545 | 0.0298 |
| 12 | fourier | 0.987 | 0.984 | 0.986 | 0.982 | 0.921 | 0.839 | 0.3735 | 0.8124 | 0.0320 |
| 16 | snap_b2 | 0.986 | 0.985 | 0.985 | 0.980 | 0.926 | 0.849 | 0.3524 | 0.7568 | 0.0298 |
| 16 | fourier | 0.987 | 0.985 | 0.985 | 0.982 | 0.920 | 0.839 | 0.3791 | 0.8162 | 0.0322 |

## L* and freeze

Walsh B=2 L* = 2. Frozen median F is 0.911 at n=12 and 0.829 at n=15. Per-seed freeze F n=12 0.911, 0.911, 0.912; n=15 0.828, 0.829, 0.830. The median across seeds of each seed's worst instance is 0.793 at n=12 and 0.713 at n=15. Medε at n=15 is 0.8870 (median ε/|E0| = 0.0350).
Fourier at the same L*=2: train/hold F 0.984/0.984 (n6) and 0.976/0.971 (n9); freeze 0.910 / 0.828.
Fourier's own smallest depth with train and hold F≥0.95 on both n is L=2.

## Reading

L*=2 median train/hold F≥0.95 on n=6 and n=9 (train 0.984/0.977, hold 0.983/0.972). Frozen median F is 0.911 at n=12 and 0.829 at n=15; n=15 (0.829) is under the 0.85 floor. L=1 is not L*: its n=9 hold median is 0.937, under 0.95, while n=6 and the n=9 train split already clear the bar. Fourier at L* freezes at 0.910 / 0.828. B=2 and Fourier stay within about 0.01 at every reported cell, so the quadratic shell is not what limits transfer. Extra depth does not close the freeze. At L=16 the n=15 median is 0.849 and n=12 is 0.926. Train fidelity has already flattened near 0.985 by L=4. This is a soft miss rather than a collapse: frozen F at n=15 is 0.829, the median energy error ε/|E0| is 0.035, and n=12 stays at 0.911. The |+⟩ overlap on that n=15 pool is 0.574. The pre-registered floor is 0.85 on both frozen sizes, and n=15 is under it.

Instance streams are `1500003017 + n·100003 + index`, the PR #42 family. n=21 was not loaded. `settings.py`, `qaoa_angles/`, and the Max-E ladder were not written.

_train wall 341s (sum of per-seed train+eval; ground states are cached separately)_
