# NEW_SAFE B=2 depth sweep — Walsh SNAP vs Fourier

**Verdict: needs_edit.** Frozen n=12 L=2 R_Fourier=0.6197, L=8 R_Fourier=0.7877, L=16 R_Fourier=1.2671, median Λ(GS)=0.7477.

Point **NEW_SAFE**: λ=0.75, μ=0. Mixer is the PR #57 quadratic Hamming shell φ_k = c₁ k + c₂ k² (B=2). PR #50's four-phase shell (B=4) won at L=2 (R≈0.62) and lost at L=8 and L=16 (R≈1.9 / 2.2). This file asks whether that split survives when the mixer is the two-coefficient shell. n=21 was not loaded.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i
\]

μ=0, so there is no cavity XX term. Cavity c owns qubits [3c, 3c+3). ε = ⟨H⟩ − E₀ from the Z histogram (unsat) and the X histogram (⟨X⟩). Medε is the median over the five seeds of each seed's median-over-instances ε. R_Fourier = Medε_B2 / Medε_Fourier, with Fourier trained on the same batches. Λ is the median of ⟨(Σ_{i∈c} X_i / 3)²⟩ over cavities and instances of the true ground state. It does not depend on L.

## Mixer

The cost pulse is the locked Walsh grading: four weights on degrees 1..4 of Cn. B=2 has two phase coefficients per layer. φ_k = c₁ k + c₂ k² on popcount k=0..3, in orthonormal coordinates, so L-BFGS sees the same Euclidean metric on φ as the locked shell. The global phase is fixed because φ_0 is not free. Paper init φ_k=−β k sits in this span. Basis reconstruction max |φ_recon − φ_paper| = 3.469e-18.

## Gate

- Train n∈{6,9}, indices [0:200). Checkpoint on the n=9 holdout [200:250).
- Freeze n=12, the first 128 pack instances. Five seeds {0,1,2,3,4} at every L∈{2,4,8,16}. n=21 was not loaded.
- **PASS** when frozen-n12 L=2 R_Fourier ≤ 0.85 and both L=8 and L=16 have R_Fourier ≥ 1.0 (Fourier wins deep).
- **needs_edit** when L=2 softens (R > 0.85) or B=2 wins deep (L=8 or L=16 has R < 1.0). That would change the paper claim.
- **win** at a depth when R ≤ 0.85. **soft** when 0.85 < R < 0.98. **tie** when 0.98 ≤ R ≤ 1.02. **loss** when R > 1.02.
- L=4 is reported and is not part of the gate. A depth with R ≥ 0.98 is not a SNAP win.

## Protocol

Packs: `beat_qaoa_12h/train_packs`. One batch stream per seed (NumPy Generator(seed) and torch.manual_seed(seed)), shared by B=2 and Fourier and by every depth. L-BFGS matches `Nov_3rd.settings` and PR #57: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Medε. Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. B=2 starts on that shell. Medε of five seeds is the ordinary median (the middle seed), not a mean of two. Ground states are the PR #50 cache at the same λ, μ, and pack seeds (dense eigh at n=6 and n=9, ARPACK at n=12). Residuals were checked on load. torch 2.15.0+cu132, numpy 2.5.4. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Frozen n=12

| L | seeds | Medε B=2 | Medε Fourier | R_Fourier | median Λ (GS) | paired B=2<Fourier | seeds R≤0.85 | call |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 2 | 5 | 0.4261 | 0.6876 | 0.6197 | 0.7477 | 640/640 | 5/5 | win |
| 4 | 5 | 0.1958 | 0.3172 | 0.6174 | 0.7477 | 640/640 | 5/5 | win |
| 8 | 5 | 0.0931 | 0.1182 | 0.7877 | 0.7477 | 635/640 | 4/5 | win |
| 16 | 5 | 0.1003 | 0.0792 | 1.2671 | 0.7477 | 8/640 | 0/5 | loss |

Quote check against PR #57 (B=2, five seeds, same protocol) and PR #50 (B=4). R in this file always uses this run's Fourier.

| L | this Medε B=2 | #57 Medε B=2 | this Fourier | #57 Fourier | this R | #57 R | #50 B=4 R | #50 seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.4261 | 0.4261 | 0.6876 | 0.6876 | 0.6197 | 0.6197 | 0.6201 | 5 |
| 4 | 0.1958 | 0.1958 | 0.3172 | 0.3172 | 0.6174 | 0.6174 | 0.8687 | 5 |
| 8 | 0.0931 | — | 0.1182 | — | 0.7877 | — | 1.9232 | 3 |
| 16 | 0.1003 | — | 0.0792 | — | 1.2671 | — | 2.1641 | 2 |

Holdout Medε (checkpoint distribution, B=50). Λ there is the ground state on the diagonalized pack prefix (n=6 and n=9, B=250), not a function of L.

| L | n | Medε B=2 | Medε Fourier | R_Fourier | median Λ (GS) | paired B=2<Fourier | call |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 2 | 6 | 0.2170 | 0.3580 | 0.6061 | 0.7612 | 250/250 | win |
| 2 | 9 | 0.2950 | 0.4722 | 0.6248 | 0.7555 | 250/250 | win |
| 4 | 6 | 0.0919 | 0.1452 | 0.6325 | 0.7612 | 250/250 | win |
| 4 | 9 | 0.1326 | 0.2254 | 0.5885 | 0.7555 | 250/250 | win |
| 8 | 6 | 0.0427 | 0.0533 | 0.8019 | 0.7612 | 235/250 | win |
| 8 | 9 | 0.0623 | 0.0847 | 0.7360 | 0.7555 | 243/250 | win |
| 16 | 6 | 0.0486 | 0.0370 | 1.3138 | 0.7612 | 17/250 | loss |
| 16 | 9 | 0.0701 | 0.0566 | 1.2380 | 0.7555 | 0/250 | loss |

## Ground state

Λ and the Z-peak do not change with L. The n=12 row is the freeze.

| n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | median Z-peak | frac peak≥0.5 | max residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 250 | 0.7612 | 0.5741 | 0.9417 | 0.262 | 0.2022 | 0.000 | 0.00e+00 |
| 9 | 250 | 0.7555 | 0.5223 | 0.9343 | 0.247 | 0.0916 | 0.000 | 0.00e+00 |
| 12 | 128 | 0.7477 | 0.5498 | 0.9117 | 0.230 | 0.0379 | 0.000 | 3.97e-09 |

## Seeds at frozen n=12

### L=2

| seed | Medε B=2 | Medε Fourier | R | best round B=2 | best round Fourier | wall s B=2 | wall s Fourier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4261 | 0.6876 | 0.6197 | 32 | 19 | 5.6 | 6.9 |
| 1 | 0.4259 | 0.6878 | 0.6192 | 16 | 19 | 4.9 | 7.6 |
| 2 | 0.4260 | 0.6876 | 0.6195 | 17 | 32 | 4.9 | 7.4 |
| 3 | 0.4263 | 0.6876 | 0.6200 | 34 | 28 | 4.9 | 7.7 |
| 4 | 0.4262 | 0.6878 | 0.6198 | 17 | 28 | 4.9 | 7.5 |

### L=4

| seed | Medε B=2 | Medε Fourier | R | best round B=2 | best round Fourier | wall s B=2 | wall s Fourier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1962 | 0.3184 | 0.6161 | 38 | 36 | 6.5 | 10.2 |
| 1 | 0.1958 | 0.3169 | 0.6181 | 39 | 22 | 6.5 | 10.5 |
| 2 | 0.1975 | 0.3172 | 0.6226 | 36 | 32 | 6.7 | 10.4 |
| 3 | 0.1949 | 0.3163 | 0.6162 | 35 | 28 | 6.4 | 10.1 |
| 4 | 0.1949 | 0.3181 | 0.6128 | 36 | 26 | 6.5 | 9.7 |

### L=8

| seed | Medε B=2 | Medε Fourier | R | best round B=2 | best round Fourier | wall s B=2 | wall s Fourier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0953 | 0.1221 | 0.7802 | 39 | 39 | 10.5 | 18.3 |
| 1 | 0.0931 | 0.1173 | 0.7944 | 39 | 38 | 10.2 | 18.3 |
| 2 | 0.0970 | 0.1133 | 0.8560 | 39 | 39 | 10.8 | 19.3 |
| 3 | 0.0911 | 0.1328 | 0.6864 | 39 | 39 | 10.4 | 18.9 |
| 4 | 0.0881 | 0.1182 | 0.7453 | 39 | 39 | 10.5 | 18.6 |

### L=16

| seed | Medε B=2 | Medε Fourier | R | best round B=2 | best round Fourier | wall s B=2 | wall s Fourier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0982 | 0.0792 | 1.2408 | 39 | 38 | 16.1 | 33.3 |
| 1 | 0.1068 | 0.0803 | 1.3292 | 39 | 38 | 15.7 | 32.5 |
| 2 | 0.1194 | 0.0753 | 1.5851 | 39 | 39 | 16.3 | 31.2 |
| 3 | 0.0997 | 0.0788 | 1.2657 | 39 | 38 | 16.5 | 31.3 |
| 4 | 0.1003 | 0.0807 | 1.2428 | 39 | 38 | 15.9 | 33.2 |

## Reading

The call is **needs_edit**.

B=2 wins deep at L=8: R_Fourier=0.7877 < 1.0 (Fourier does not win).

L=2 frozen n=12 is win: Medε B=2 0.4261 vs Fourier 0.6876, R=0.6197, 5/5 seeds ≤0.85, B=2 lower on 640/640 paired instances. PR #57 quoted R=0.6197 (Medε 0.4261 / 0.6876). Deltas versus that quote: Medε B=2 +0.0000, Fourier +0.0000, R -0.0000.

L=4 is win and is not a gate: Medε B=2 0.1958 vs Fourier 0.3172, R=0.6174, 5/5 seeds ≤0.85, paired 640/640. PR #57 quoted R=0.6174. PR #50's four-phase shell at this depth was soft (R=0.8687).

L=8 is win: Medε B=2 0.0931 vs Fourier 0.1182, R=0.7877, 4/5 seeds ≤0.85, 0/5 seeds with R≥1, B=2 lower on 635/640. PR #50's B=4 shell, on 3 seeds, had R=1.9232 (Medε 0.2255 / 0.1173). Fourier on the seeds #50 already published differs from this run by at most 0.0000. Every B=2 seed restores round 39.

L=16 is loss: Medε B=2 0.1003 vs Fourier 0.0792, R=1.2671, 0/5 seeds ≤0.85, 5/5 seeds with R≥1, B=2 lower on 8/640. PR #50's B=4 shell, on 2 seeds, had R=2.1641 (Medε 0.1726 / 0.0797). Fourier on the seeds #50 already published differs from this run by at most 0.0000. Every B=2 seed restores round 39.

Median Λ(GS) at frozen n=12 is 0.7477 (n=6 0.7612, n=9 0.7555). The #50 quote is 0.7477. Λ does not depend on L or on B. Holdout R at L=2 is n=6 0.6061, n=9 0.6248.

This file is needs_edit. L=2 does not soften (R=0.620). L=16 still belongs to Fourier (R=1.267). L=8 does not: B=2 wins there (R=0.788≤0.85), so the sentence "L≥8 loses to Fourier" is false for this mixer. The depth where Fourier takes the median moves from 8 to 16.

n=21 was not run.

## Non-claims

This does not retune λ or μ, does not add XX, and does not reseal n=15 or n=18. It does not replace the PR #50 B=4 sentence. L=4 is outside the honesty gate. The deep bar is R≥1.0 on the median of five seed medians, which is a coarser statement than the per-depth win/soft/tie/loss labels. n=21 was not loaded.

Per-seed JSON is `runs/L{L}_{method}_s{seed}.json`. Ground-state arrays are `cache/diag_n{n}.npz`.
