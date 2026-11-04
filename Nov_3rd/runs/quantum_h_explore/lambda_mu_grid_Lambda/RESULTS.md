# (λ, μ) grid — cavity Λ and shallow R_HVA at L=2

**3 candidates. Best by (low Λ, low R) is λ=0.50, μ=0.00: Λ=0.559, gap=0.110, R_HVA=0.750.**

Round 10 killed the SAFE lock (λ=0.75, μ=−0.50) because the ground state is cavity-trivial: median Λ≈0.943 at n=12/15/18 even though L=2 R_HVA stayed near 0.33. This grid keeps that Hamiltonian family and asks where Λ drops to ≤0.80 without giving up the shallow SNAP lead over HVA.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - \lambda \sum_i X_i + \mu \sum_c \sum_{i<j\in c} X_i X_j
\]

Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. The mixer is BinH Hamming-shell SNAP (Walsh degrees 1..4 on Cn). Readout is the Z histogram for unsat and the X histogram for ⟨X⟩, ⟨XX⟩, and Λ. No Y. n=21 is not loaded.

λ ∈ {0.50, 0.75, 1.00}, μ ∈ {0, −0.15, −0.25, −0.35, −0.50}. In the code the ΣX coefficient is λ_x = −λ and the ΣXX coefficient is μ. SAFE is λ=0.75, μ=−0.50.

## Gate

Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ on the X histogram of the true ground state, equivalently 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩. The reported Λ is the median over cavities and instances. Λ→1 means the cavity has collapsed to one X-magnetization. A Z-basis product state scores 1/3.

Med ε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E_0. R_HVA = Medε_SNAP / Medε_HVA at n=12, L=2. R_F is the same ratio against Fourier QAOA. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state. Z-peak is the median over instances of max_x |⟨x|ψ_0⟩|^2.

- **Candidate (PASS)** when n=12 median Λ ≤ 0.80, n=12 median product gap ≥ 0.03, and n=12 L=2 R_HVA ≤ 0.85.
- **FAIL** when any of those three bars is missed.
- **SAFE control** is λ=0.75, μ=−0.50. It is flagged as the known-trivialized control and is not ranked as a candidate, whatever its numbers.

Rank is low Λ, then low R_HVA. Fourier is trained on the same L-BFGS budget and reported as R_F. It is not part of the gate.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12, the first 128 instances (B≥64).
- L=2 only. Three seeds {0, 1, 2}. Methods: Walsh SNAP, HVA (β^X, β^XX, γ), Fourier (β, γ).
- L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n on that cell's (λ, μ). Each seed restores the round with the best n=9-holdout Med ε.
- Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k.
- Ground states: dense eigh at n=6 and n=9, ARPACK at n=12. Λ and Z-peak use that vector. The product gap is computed on the n=12 freeze only.
- n=15, n=18, and n=21 were not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Grid at n=12, L=2

Λ9 is the median cavity Λ on all 250 n=9 pack rows (train and holdout), from the same (λ, μ). The gate uses Λ12.

| λ | μ | role | Λ12 | Λ9 | gap12 | Z-peak12 | Medε SNAP | Medε HVA | Medε Fourier | R_HVA | R_F |
|--:|--:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.00 | candidate | 0.559 | 0.559 | 0.110 | 0.1569 | 0.8764 | 1.1688 | 1.1687 | 0.750 | 0.750 |
| 0.50 | -0.15 | candidate | 0.741 | 0.745 | 0.188 | 0.0497 | 0.4602 | 0.7246 | 0.7286 | 0.635 | 0.632 |
| 0.50 | -0.25 | fail | 0.824 | 0.828 | 0.195 | 0.0228 | 0.3108 | 0.5452 | 0.5600 | 0.570 | 0.555 |
| 0.50 | -0.35 | fail | 0.876 | 0.876 | 0.160 | 0.0125 | 0.2181 | 0.4388 | 0.4389 | 0.497 | 0.497 |
| 0.50 | -0.50 | fail | 0.918 | 0.919 | 0.110 | 0.0064 | 0.1384 | 0.3354 | 0.3377 | 0.413 | 0.410 |
| 0.75 | 0.00 | candidate | 0.748 | 0.755 | 0.127 | 0.0379 | 0.4264 | 0.6901 | 0.6876 | 0.618 | 0.620 |
| 0.75 | -0.15 | fail | 0.854 | 0.856 | 0.129 | 0.0136 | 0.2383 | 0.4701 | 0.4784 | 0.507 | 0.498 |
| 0.75 | -0.25 | fail | 0.893 | 0.895 | 0.109 | 0.0086 | 0.1664 | 0.3826 | 0.3872 | 0.435 | 0.430 |
| 0.75 | -0.35 | fail | 0.918 | 0.920 | 0.087 | 0.0058 | 0.1253 | 0.3218 | 0.3283 | 0.389 | 0.382 |
| 0.75 | -0.50 | SAFE control | 0.941 | 0.943 | 0.066 | 0.0036 | 0.0855 | 0.2608 | 0.2667 | 0.328 | 0.321 |
| 1.00 | 0.00 | fail | 0.851 | 0.855 | 0.099 | 0.0129 | 0.2342 | 0.4695 | 0.4722 | 0.499 | 0.496 |
| 1.00 | -0.15 | fail | 0.905 | 0.907 | 0.079 | 0.0067 | 0.1386 | 0.3455 | 0.3517 | 0.401 | 0.394 |
| 1.00 | -0.25 | fail | 0.926 | 0.928 | 0.067 | 0.0046 | 0.1029 | 0.2911 | 0.2991 | 0.354 | 0.344 |
| 1.00 | -0.35 | fail | 0.941 | 0.942 | 0.057 | 0.0035 | 0.0805 | 0.2575 | 0.2591 | 0.313 | 0.311 |
| 1.00 | -0.50 | fail | 0.955 | 0.957 | 0.044 | 0.0025 | 0.0574 | 0.2186 | 0.2171 | 0.262 | 0.264 |

## Ranked candidates

| rank | λ | μ | Λ12 | gap12 | Z-peak12 | R_HVA | R_F | paired SNAP<HVA |
|--:|--:|--:|---:|---:|---:|---:|---:|--:|
| 1 | 0.50 | 0.00 | 0.559 | 0.110 | 0.1569 | 0.750 | 0.750 | 384/384 |
| 2 | 0.50 | -0.15 | 0.741 | 0.188 | 0.0497 | 0.635 | 0.632 | 384/384 |
| 3 | 0.75 | 0.00 | 0.748 | 0.127 | 0.0379 | 0.618 | 0.620 | 384/384 |

## SAFE control

λ=0.75, μ=−0.50 is the known-trivialized control from PR #41 (median Λ≈0.943, L=2 R_HVA≈0.328 on a five-seed n=12 freeze). This three-seed rerun measures Λ12=0.941 (min 0.843, max 0.972, fraction of cavities above 0.85 = 0.996), gap12=0.066, Z-peak12=0.0036, R_HVA=0.328, R_F=0.321. It is not ranked as a candidate.

## Cells that miss a bar

- λ=0.50, μ=-0.25: median Λ 0.824 > 0.80. Λ12=0.824, gap=0.195, Z-peak=0.0228, R_HVA=0.570, R_F=0.555.
- λ=0.50, μ=-0.35: median Λ 0.876 > 0.80. Λ12=0.876, gap=0.160, Z-peak=0.0125, R_HVA=0.497, R_F=0.497.
- λ=0.50, μ=-0.50: median Λ 0.918 > 0.80. Λ12=0.918, gap=0.110, Z-peak=0.0064, R_HVA=0.413, R_F=0.410.
- λ=0.75, μ=-0.15: median Λ 0.854 > 0.80. Λ12=0.854, gap=0.129, Z-peak=0.0136, R_HVA=0.507, R_F=0.498.
- λ=0.75, μ=-0.25: median Λ 0.893 > 0.80. Λ12=0.893, gap=0.109, Z-peak=0.0086, R_HVA=0.435, R_F=0.430.
- λ=0.75, μ=-0.35: median Λ 0.918 > 0.80. Λ12=0.918, gap=0.087, Z-peak=0.0058, R_HVA=0.389, R_F=0.382.
- λ=1.00, μ=0.00: median Λ 0.851 > 0.80. Λ12=0.851, gap=0.099, Z-peak=0.0129, R_HVA=0.499, R_F=0.496.
- λ=1.00, μ=-0.15: median Λ 0.905 > 0.80. Λ12=0.905, gap=0.079, Z-peak=0.0067, R_HVA=0.401, R_F=0.394.
- λ=1.00, μ=-0.25: median Λ 0.926 > 0.80. Λ12=0.926, gap=0.067, Z-peak=0.0046, R_HVA=0.354, R_F=0.344.
- λ=1.00, μ=-0.35: median Λ 0.941 > 0.80. Λ12=0.941, gap=0.057, Z-peak=0.0035, R_HVA=0.313, R_F=0.311.
- λ=1.00, μ=-0.50: median Λ 0.955 > 0.80. Λ12=0.955, gap=0.044, Z-peak=0.0025, R_HVA=0.262, R_F=0.264.

## Per-seed n=12 Med ε

| λ | μ | seed | SNAP | HVA | Fourier | R_HVA | paired | best round SNAP | wall SNAP s |
|--:|--:|--:|---:|---:|---:|---:|--:|--:|--:|
| 0.50 | 0.00 | 0 | 0.8764 | 1.1688 | 1.1689 | 0.750 | 128/128 | 38 | 5.5 |
| 0.50 | 0.00 | 1 | 0.8752 | 1.1673 | 1.1682 | 0.750 | 128/128 | 39 | 5.6 |
| 0.50 | 0.00 | 2 | 0.8788 | 1.1694 | 1.1687 | 0.752 | 128/128 | 34 | 5.7 |
| 0.50 | -0.15 | 0 | 0.4602 | 0.7260 | 0.7286 | 0.634 | 128/128 | 36 | 5.2 |
| 0.50 | -0.15 | 1 | 0.4594 | 0.7218 | 0.7287 | 0.637 | 128/128 | 39 | 5.3 |
| 0.50 | -0.15 | 2 | 0.4638 | 0.7246 | 0.7286 | 0.640 | 128/128 | 32 | 5.2 |
| 0.50 | -0.25 | 0 | 0.3110 | 0.5471 | 0.5606 | 0.568 | 128/128 | 28 | 5.0 |
| 0.50 | -0.25 | 1 | 0.3099 | 0.5452 | 0.5595 | 0.569 | 128/128 | 35 | 5.2 |
| 0.50 | -0.25 | 2 | 0.3108 | 0.5439 | 0.5600 | 0.572 | 128/128 | 29 | 5.0 |
| 0.50 | -0.35 | 0 | 0.2180 | 0.4388 | 0.4389 | 0.497 | 128/128 | 39 | 5.0 |
| 0.50 | -0.35 | 1 | 0.2181 | 0.4388 | 0.4392 | 0.497 | 128/128 | 32 | 5.2 |
| 0.50 | -0.35 | 2 | 0.2182 | 0.4403 | 0.4388 | 0.496 | 128/128 | 37 | 5.0 |
| 0.50 | -0.50 | 0 | 0.1389 | 0.3380 | 0.3378 | 0.411 | 128/128 | 21 | 5.1 |
| 0.50 | -0.50 | 1 | 0.1384 | 0.3354 | 0.3377 | 0.413 | 128/128 | 11 | 5.3 |
| 0.50 | -0.50 | 2 | 0.1382 | 0.3310 | 0.3374 | 0.417 | 128/128 | 9 | 5.1 |
| 0.75 | 0.00 | 0 | 0.4264 | 0.6920 | 0.6876 | 0.616 | 128/128 | 35 | 5.5 |
| 0.75 | 0.00 | 1 | 0.4243 | 0.6884 | 0.6878 | 0.616 | 128/128 | 32 | 5.6 |
| 0.75 | 0.00 | 2 | 0.4264 | 0.6901 | 0.6876 | 0.618 | 128/128 | 31 | 5.5 |
| 0.75 | -0.15 | 0 | 0.2383 | 0.4720 | 0.4785 | 0.505 | 128/128 | 39 | 5.2 |
| 0.75 | -0.15 | 1 | 0.2383 | 0.4701 | 0.4784 | 0.507 | 128/128 | 38 | 5.1 |
| 0.75 | -0.15 | 2 | 0.2387 | 0.4699 | 0.4782 | 0.508 | 128/128 | 37 | 5.2 |
| 0.75 | -0.25 | 0 | 0.1660 | 0.3828 | 0.3872 | 0.434 | 128/128 | 39 | 5.5 |
| 0.75 | -0.25 | 1 | 0.1670 | 0.3826 | 0.3889 | 0.437 | 128/128 | 37 | 5.8 |
| 0.75 | -0.25 | 2 | 0.1664 | 0.3820 | 0.3869 | 0.436 | 128/128 | 38 | 5.6 |
| 0.75 | -0.35 | 0 | 0.1253 | 0.3218 | 0.3290 | 0.389 | 128/128 | 39 | 5.1 |
| 0.75 | -0.35 | 1 | 0.1274 | 0.3219 | 0.3282 | 0.396 | 128/128 | 37 | 5.3 |
| 0.75 | -0.35 | 2 | 0.1251 | 0.3207 | 0.3283 | 0.390 | 128/128 | 38 | 5.2 |
| 0.75 | -0.50 | 0 | 0.0851 | 0.2632 | 0.2667 | 0.323 | 128/128 | 36 | 5.8 |
| 0.75 | -0.50 | 1 | 0.0858 | 0.2608 | 0.2667 | 0.329 | 128/128 | 28 | 5.3 |
| 0.75 | -0.50 | 2 | 0.0855 | 0.2555 | 0.2664 | 0.335 | 128/128 | 32 | 5.2 |
| 1.00 | 0.00 | 0 | 0.2349 | 0.4695 | 0.4747 | 0.500 | 128/128 | 34 | 5.4 |
| 1.00 | 0.00 | 1 | 0.2310 | 0.4687 | 0.4722 | 0.493 | 128/128 | 32 | 5.6 |
| 1.00 | 0.00 | 2 | 0.2342 | 0.4697 | 0.4722 | 0.498 | 128/128 | 31 | 5.5 |
| 1.00 | -0.15 | 0 | 0.1386 | 0.3468 | 0.3513 | 0.400 | 128/128 | 34 | 5.4 |
| 1.00 | -0.15 | 1 | 0.1389 | 0.3455 | 0.3517 | 0.402 | 128/128 | 38 | 5.4 |
| 1.00 | -0.15 | 2 | 0.1385 | 0.3451 | 0.3517 | 0.401 | 128/128 | 39 | 5.5 |
| 1.00 | -0.25 | 0 | 0.1029 | 0.2944 | 0.2998 | 0.349 | 128/128 | 34 | 5.3 |
| 1.00 | -0.25 | 1 | 0.1035 | 0.2907 | 0.2990 | 0.356 | 128/128 | 30 | 5.3 |
| 1.00 | -0.25 | 2 | 0.1024 | 0.2911 | 0.2991 | 0.352 | 128/128 | 35 | 5.6 |
| 1.00 | -0.35 | 0 | 0.0805 | 0.2575 | 0.2592 | 0.313 | 128/128 | 38 | 5.3 |
| 1.00 | -0.35 | 1 | 0.0799 | 0.2578 | 0.2591 | 0.310 | 128/128 | 39 | 5.3 |
| 1.00 | -0.35 | 2 | 0.0810 | 0.2570 | 0.2590 | 0.315 | 128/128 | 32 | 5.3 |
| 1.00 | -0.50 | 0 | 0.0574 | 0.2186 | 0.2177 | 0.263 | 128/128 | 39 | 5.6 |
| 1.00 | -0.50 | 1 | 0.0573 | 0.2191 | 0.2171 | 0.261 | 128/128 | 34 | 5.7 |
| 1.00 | -0.50 | 2 | 0.0574 | 0.2180 | 0.2170 | 0.263 | 128/128 | 38 | 5.6 |

## Reading

Across the 15 cells, n=12 median Λ runs from 0.559 to 0.955, the product gap from 0.044 to 0.195, and R_HVA from 0.262 to 0.750. Λ rises, and R_HVA falls, as λ increases or as μ becomes more negative. The three bars are cleared only for small |μ|.

HVA mean |β^XX| runs from 0.125 to 0.288. It stays away from the paper-init value 0 on every cell, including μ=0, so the HVA column is a trained XX mixer. At μ=0, R_HVA and R_F agree to about 0.01: the extra XX angle does not improve on Fourier when the Hamiltonian has no XX term.

Nearest miss on Λ is λ=0.50, μ=-0.25: Λ12=0.824 (bar 0.80), gap=0.195, R_HVA=0.570.

Candidates, lowest Λ first: λ=0.50, μ=0.00 has Λ=0.559, gap=0.110, Z-peak=0.1569, R_HVA=0.750. λ=0.50, μ=-0.15 has Λ=0.741, gap=0.188, Z-peak=0.0497, R_HVA=0.635. λ=0.75, μ=0.00 has Λ=0.748, gap=0.127, Z-peak=0.0379, R_HVA=0.618.

Per-seed JSON is `runs/L2_{method}_s{seed}_{tag}.json`. Ground-state arrays are `cache/diag_{tag}.npz`. Mean |β^XX| per cell is in `summary.json`.

