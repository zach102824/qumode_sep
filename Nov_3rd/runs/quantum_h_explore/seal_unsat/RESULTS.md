# RESULTS — unsat λ=0.75 Walsh seal, freeze {6,9} → n=12/15

**Verdict: PASS**

Walsh beats Fourier QAOA at L=2 and L=16 on n=6, n=9, and frozen n=12 (n=12 L=16 R=0.940). n=15 stays under 1.1 (L=16 R=0.914). 2/3 seeds beat at n=12 L=16 (median seed R=0.888). Transfer winner: unsat (unsat via walsh: 4/4 transfer BEATs, n=12 L=16 R=0.940).

## Setup

Stoquastic transverse field on the raw unsat diagonal. The gate is the locked Nov_3rd sequence. Angles for n=12 and n=15 were not in the loss and not in the checkpoint.

```text
H = diag(unsat) − λ Σ_i X_i ,  λ = 0.75
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
ε = (⟨H⟩ − E_0) / n
⟨H⟩ = Σ_z p_Z(z) unsat(z) − λ Σ_i ⟨X_i⟩
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Holdout B=128 per n and transfer B=64 (n=12), B=32 (n=15) are the scale1000 satisfiable prefix (base 54e9). Seeds [0, 1, 2]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. L=16 is longer than the 24-round PR #15 budget: at 24 rounds Fourier's holdout score was still falling by a few 10⁻⁴ per round, so that baseline was not sealed. Early stop is unchanged (8 rounds without a 10⁻⁶ improvement).

Fourier QAOA stores 2L Zhou coefficients (Q=L, paper init β=0.01, γ=−0.01, seed jitter 0.01). power_p stores per-layer γ, c1, c2, c3 and one exponent p (4L+1). Walsh stores four degree weights per layer plus the same shell (7L+1). Both SNAP heads warm-start from the trained Fourier dual (φ_k=−β k, c2=c3=0, p=0); Walsh copies the scalar γ onto degrees 1–4. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier.

## Quantumness

A split is quantum when the median ground-state bitstring peak is below 0.5. E0 is Lanczos on the sparse hypercube Hamiltonian. It matches dense eigh on two train instances at n=6 and n=9.

Worst |ΔE0| Lanczos vs dense = 1.78e-15.

| n | role | B | peak Med | S_cav Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.206 | 0.442 | 0.197 | 0.322 | 7.9e-15 |
| 6 | hold | 128 | 0.202 | 0.418 | 0.198 | 0.323 | 4.4e-15 |
| 9 | train | 128 | 0.089 | 0.513 | 0.191 | 0.323 | 2.5e-10 |
| 9 | hold | 128 | 0.086 | 0.550 | 0.187 | 0.319 | 1.9e-10 |
| 12 | transfer | 64 | 0.038 | 0.617 | 0.183 | 0.313 | 3.6e-09 |
| 15 | transfer | 32 | 0.019 | 0.591 | 0.194 | 0.323 | 1.9e-10 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34). Equal degree weights reproduce the scalar pulse up to a global phase. On the ZZZ champ the same split is pure degree 3, so graded γ is a null knob there and was not trained.

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better. n=6 and n=9 are the selection holdout, so those medians are slightly optimistic. n=12 and n=15 are not.

| method | L | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |
|:-------|--:|--:|:-----|-------:|------:|---------:|----:|----------:|--:|:-----|
| fourier_qaoa | 2 | 6 | hold | 4 | 0.0594 | 0.0000 | 0.0286 | -0.2610 | — |  |
| fourier_qaoa | 2 | 9 | hold | 4 | 0.0574 | 0.0000 | 0.0218 | -0.2633 | — |  |
| fourier_qaoa | 2 | 12 | transfer | 4 | 0.0532 | 0.0000 | 0.0170 | -0.2608 | — |  |
| fourier_qaoa | 2 | 15 | transfer | 4 | 0.0596 | 0.0000 | 0.0132 | -0.2606 | — |  |
| power_p | 2 | 6 | hold | 9 | 0.0543 | 0.0005 | 0.0272 | -0.2679 | 0.915 | BEAT |
| power_p | 2 | 9 | hold | 9 | 0.0551 | 0.0003 | 0.0208 | -0.2653 | 0.961 | BEAT |
| power_p | 2 | 12 | transfer | 9 | 0.0532 | 0.0002 | 0.0169 | -0.2598 | 1.000 | TIE |
| power_p | 2 | 15 | transfer | 9 | 0.0592 | 0.0001 | 0.0133 | -0.2602 | 0.994 | TIE |
| walsh | 2 | 6 | hold | 15 | 0.0398 | 0.0004 | 0.0187 | -0.2801 | 0.670 | BEAT |
| walsh | 2 | 9 | hold | 15 | 0.0404 | 0.0003 | 0.0156 | -0.2798 | 0.705 | BEAT |
| walsh | 2 | 12 | transfer | 15 | 0.0405 | 0.0003 | 0.0154 | -0.2695 | 0.761 | BEAT |
| walsh | 2 | 15 | transfer | 15 | 0.0437 | 0.0003 | 0.0114 | -0.2765 | 0.733 | BEAT |
| fourier_qaoa | 16 | 6 | hold | 32 | 0.0025 | 0.0025 | 0.0017 | -0.3187 | — |  |
| fourier_qaoa | 16 | 9 | hold | 32 | 0.0021 | 0.0025 | 0.0018 | -0.3168 | — |  |
| fourier_qaoa | 16 | 12 | transfer | 32 | 0.0029 | 0.0027 | 0.0025 | -0.3078 | — |  |
| fourier_qaoa | 16 | 15 | transfer | 32 | 0.0030 | 0.0029 | 0.0028 | -0.3177 | — |  |
| power_p | 16 | 6 | hold | 65 | 0.0025 | 0.0021 | 0.0019 | -0.3187 | 0.978 | BEAT |
| power_p | 16 | 9 | hold | 65 | 0.0020 | 0.0022 | 0.0019 | -0.3168 | 0.951 | BEAT |
| power_p | 16 | 12 | transfer | 65 | 0.0029 | 0.0025 | 0.0025 | -0.3078 | 0.990 | TIE |
| power_p | 16 | 15 | transfer | 65 | 0.0029 | 0.0028 | 0.0028 | -0.3177 | 0.992 | TIE |
| walsh | 16 | 6 | hold | 113 | 0.0023 | 0.0019 | 0.0019 | -0.3188 | 0.925 | BEAT |
| walsh | 16 | 9 | hold | 113 | 0.0019 | 0.0019 | 0.0019 | -0.3170 | 0.895 | BEAT |
| walsh | 16 | 12 | transfer | 113 | 0.0027 | 0.0022 | 0.0025 | -0.3080 | 0.940 | BEAT |
| walsh | 16 | 15 | transfer | 113 | 0.0027 | 0.0025 | 0.0028 | -0.3179 | 0.914 | BEAT |

Per-seed median ε, then per-seed R where a baseline exists:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2]: 0.0594, 0.0594, 0.0594
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2]: 0.0574, 0.0573, 0.0574
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2]: 0.0532, 0.0532, 0.0532
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2]: 0.0596, 0.0596, 0.0596
- `power_p` L=2 n=6 seeds [0, 1, 2]: 0.0548, 0.0543, 0.0538 R 0.923, 0.915, 0.906 (median of those R = 0.915)
- `power_p` L=2 n=9 seeds [0, 1, 2]: 0.0554, 0.0551, 0.0548 R 0.966, 0.961, 0.956 (median of those R = 0.961)
- `power_p` L=2 n=12 seeds [0, 1, 2]: 0.0533, 0.0532, 0.0530 R 1.003, 1.000, 0.996 (median of those R = 1.000)
- `power_p` L=2 n=15 seeds [0, 1, 2]: 0.0592, 0.0592, 0.0593 R 0.993, 0.994, 0.995 (median of those R = 0.994)
- `walsh` L=2 n=6 seeds [0, 1, 2]: 0.0402, 0.0398, 0.0394 R 0.677, 0.670, 0.663 (median of those R = 0.670)
- `walsh` L=2 n=9 seeds [0, 1, 2]: 0.0409, 0.0404, 0.0402 R 0.713, 0.705, 0.702 (median of those R = 0.705)
- `walsh` L=2 n=12 seeds [0, 1, 2]: 0.0409, 0.0405, 0.0403 R 0.769, 0.761, 0.757 (median of those R = 0.761)
- `walsh` L=2 n=15 seeds [0, 1, 2]: 0.0440, 0.0437, 0.0435 R 0.738, 0.733, 0.729 (median of those R = 0.733)
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2]: 0.0025, 0.0068, 0.0025
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2]: 0.0021, 0.0063, 0.0020
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2]: 0.0029, 0.0076, 0.0027
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2]: 0.0030, 0.0079, 0.0028
- `power_p` L=16 n=6 seeds [0, 1, 2]: 0.0025, 0.0060, 0.0023 R 0.978, 0.883, 0.956 (median of those R = 0.956)
- `power_p` L=16 n=9 seeds [0, 1, 2]: 0.0020, 0.0057, 0.0019 R 0.951, 0.903, 0.980 (median of those R = 0.951)
- `power_p` L=16 n=12 seeds [0, 1, 2]: 0.0029, 0.0071, 0.0027 R 0.990, 0.946, 0.996 (median of those R = 0.990)
- `power_p` L=16 n=15 seeds [0, 1, 2]: 0.0029, 0.0076, 0.0027 R 0.992, 0.958, 0.981 (median of those R = 0.981)
- `walsh` L=16 n=6 seeds [0, 1, 2]: 0.0020, 0.0054, 0.0023 R 0.787, 0.794, 0.953 (median of those R = 0.794)
- `walsh` L=16 n=9 seeds [0, 1, 2]: 0.0018, 0.0052, 0.0019 R 0.859, 0.815, 0.980 (median of those R = 0.859)
- `walsh` L=16 n=12 seeds [0, 1, 2]: 0.0026, 0.0065, 0.0027 R 0.888, 0.856, 0.996 (median of those R = 0.888)
- `walsh` L=16 n=15 seeds [0, 1, 2]: 0.0027, 0.0070, 0.0027 R 0.898, 0.882, 0.979 (median of those R = 0.898)

## GRAPE of e^{−i H_c}

Cost-only layers (SNAP phase 0) commute, so L of them equal one pulse whose γ is the sum. That pulse is already U_c^X. A grid over that one parameter is the GRAPE optimum for a single control along C_n. The minimum on every split is γ=0, which is the final W applied to |0⟩, the product |+⟩ state. The Hamming-shell SNAP does not commute with the cost layer, and that is the part that moves ε.

| n | role | GRAPE γ | GRAPE Med ε | Walsh L=16 Med ε |
|--:|:-----|--------:|------------:|------------------:|
| 6 | hold | 0.0000 | 0.1976 | 0.0023 |
| 9 | hold | 0.0000 | 0.1873 | 0.0019 |
| 12 | transfer | 0.0000 | 0.1828 | 0.0027 |
| 15 | transfer | 0.0000 | 0.1907 | 0.0027 |

L-BFGS on L redundant cost angles (seed 0, warm-started from the Fourier γ with the mixer removed) is a check that the parameterization collapses, not a second optimum. |Δ⟨H⟩| is the gap between those L layers and one layer whose γ is the sum.

| L | seed | L-BFGS Med ε n=12 | one-pulse |Δ⟨H⟩| | Walsh Med ε | power_p Med ε | Fourier Med ε |
|--:|----:|------------------:|-----------------:|------------:|--------------:|--------------:|
| 2 | 0 | 0.1830 | 9.77e-15 | 0.0409 | 0.0533 | 0.0532 |
| 16 | 0 | 0.9338 | 4.45e-13 | 0.0026 | 0.0029 | 0.0029 |

## Which H_c wins

The two diagonals do not share a residual scale, so the comparison is the residual ratio R against Fourier QAOA on that same H, at the head each diagonal can actually use. ZZZ is pure degree 3, so Walsh collapses to scalar γ and the champ head is power_p (4L+1 parameters). Unsat spreads Walsh mass over degrees 1–4, so the PR #11 head is graded γ (7L+1). power_p on unsat is the scalar control. A transfer cell is one of {12,15}×{2,16}. The winner is the eligible head (n=12 L=16 BEAT) with more transfer BEATs, then the lower n=12 L=16 R. ZZZ numbers are the published PR #15 champ at B_train=32. They were not retrained here.

| H_c | head | source | n=12 L=16 R | transfer BEATs | eligible |
|:----|:-----|:-------|------------:|---------------:|:---------|
| unsat | walsh | this seal | 0.940 | 4/4 | yes |
| unsat | power_p | this seal | 0.990 | 0/4 | no |
| zzz | power_p | PR #15 published, B_train=32 | 0.924 | 2/4 | yes |

**Winner: unsat.** unsat via walsh: 4/4 transfer BEATs, n=12 L=16 R=0.940.

On the scalar power_p head, published ZZZ is BEAT at n=12 L=16 (R=0.924) and this unsat seal is TIE (R=0.990). A power_p-only comparison would pick ZZZ: its L=16 transfer survives and unsat's scalar head does not. Unsat wins the transfer story through Walsh, the head its diagonal actually supports. Walsh seed ratios at n=12 L=16 are 0.888, 0.856, 0.996. Two seeds beat and one ties; the median of those ratios and the ratio of medians are both BEAT. Walsh has 7L+1 parameters, power_p has 4L+1, Fourier QAOA has 2L. The call is which diagonal supports a transferring head, not a matched parameter count.

A 24-round L=16 snapshot, the PR #15 budget, had not stalled: Fourier's holdout score was still falling by a few 10⁻⁴ per round. At that snapshot unsat Walsh n=12 R was 0.897 and power_p was a BEAT at 0.956. The table above is the 48-round budget. The scalar head's transfer BEAT did not survive the longer Fourier baseline.

L=16 Fourier's last-round holdout-score drop, seeds in order, was 2.36e-05, 2.39e-05, 1.60e-05.

Published ZZZ power_p (PR #15) and the earlier unsat Walsh ratios at B_train=32, for the shift this seal measures:

| L | n | ZZZ power_p R | unsat Walsh B=32 R | unsat Walsh this seal R | unsat power_p this seal R |
|--:|--:|---------------:|-------------------:|------------------------:|--------------------------:|
| 2 | 6 | 1.000 TIE | 0.735 | 0.670 | 0.915 |
| 2 | 9 | 0.999 TIE | 0.756 | 0.705 | 0.961 |
| 2 | 12 | 1.000 TIE | 0.831 | 0.761 | 1.000 |
| 2 | 15 | 1.000 TIE | 0.768 | 0.733 | 0.994 |
| 16 | 6 | 0.654 BEAT | 0.807 | 0.925 | 0.978 |
| 16 | 9 | 0.853 BEAT | 0.794 | 0.895 | 0.951 |
| 16 | 12 | 0.924 BEAT | 0.886 | 0.940 | 0.990 |
| 16 | 15 | 0.905 BEAT | 0.848 | 0.914 | 0.992 |

## Rule

PASS if Walsh has R≤0.98 at L=2 and L=16 on holdout n=6, n=9 and frozen n=12; n=15 R stays below 1.10 at both depths; the n=15 ground state is not a bitstring; at least two of three n=12 L=16 seeds beat and the median of those seed ratios is on the same side of 0.98 as the ratio of medians; and the optimized cost-only pulse at n=12 L=16 is worse than 1.05× the Walsh residual. power_p does not gate PASS.

KILL if a gated ground state (n=6, 9, or 12) is a bitstring; if neither head beats Fourier QAOA on both n=6 and n=9 at either depth; if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05); or if the cost-only pulse matches Walsh. Otherwise needs_edit.

Self-check: Fock vs Nov3rdAnsatz max |Δp| Z=2.07e-08, X=4.60e-07; QAOA-point max |Δp|=4.44e-10; X-hist vs Pauli 5.00e-15.

Optimizer time summed over fits 223.8s. This process 28.8s. Branch `cursor/unsat-walsh-seal-35aa`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written or signalled.

