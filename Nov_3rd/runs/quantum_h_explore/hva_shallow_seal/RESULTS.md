# Shallow seal — Walsh SNAP vs HVA at the SAFE point

**Verdict: PASS.** L=2 → PASS (R_HVA=0.3280, 5/5 seeds BEAT, CI 0.3181–0.3416) L=4 → PASS (R_HVA=0.5356, 5/5 seeds BEAT, CI 0.4940–0.5741).

This seal is L∈{2, 4} only. PR #36 already has HVA ahead of SNAP at L=8 (R_HVA=1.372) and L=16 (R_HVA=2.275) under the same Hamiltonian and the same L-BFGS budget. Those depths were not retrained here and are not claimed.

## Gate

The call uses frozen n=12. Med ε is the median over seeds of each seed's median-over-instances excess energy ⟨H⟩−E₀. R_HVA = Medε_SNAP / Medε_HVA.

- **PASS** on a depth when R_HVA ≤ 0.85, at least 3 of 5 seeds have their own n12 R_HVA ≤ 0.85, and the instance-bootstrap 95% interval for R_HVA ends at or below 0.85.
- **needs_edit** when that depth is not a PASS and HVA has not taken it. That is the soft case: the median lead is past 0.85, fewer than 3 seeds clear 0.85, or the instance interval crosses 0.85.
- **KILL** when R_HVA > 1 at n12, or at least 3 of 5 seeds have R_HVA > 1.
- Overall is KILL if either depth is KILL, needs_edit if either depth is needs_edit, and PASS only when both L=2 and L=4 are PASS.
- n=15 is reported and is not the gate. n=18 is a probe and is not the gate.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j
\]

SAFE point **λ=0.75, μ=−0.50**. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. Readout is the Z histogram (unsat) plus the X histogram (⟨X⟩ and ⟨XX⟩). No Y.

## Ansatz

| method | angles / layer | mixer |
|---|---|---|
| Fourier QAOA | γ, β | \(\bigotimes \exp(-i\beta X/2)\) |
| HVA QAOA | γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\) then \(U_X(\beta^X)\). β^XX=0 is Fourier. |
| Walsh SNAP | γ_{w=1..4}, φ_k | BinH Hamming shell. Linear φ_k=−βk is the X mixer. |

Paper init β=β^X=0.01, β^XX=0, γ=γ_w=-0.01, φ_k=−β k. The three inits are the same state. The seed only changes the L-BFGS batches.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n12 first 128, n15 first 64.
- n18 probe: 16 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n18 train pack in the tree. The probe is not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings` and PR #36: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. No Adam.
- Each seed restores the round with the best n9-holdout Med ε.
- Instance 95% CI: 4000 bootstrap draws, resampling instances inside each seed, seeds held fixed. RNG seed 5209.
- n21 phase-2 ladder was not read or written. L≥8 was not trained.

## L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1369 | 0.0952 | 0.0465 | 0.3399 | 0.4887 |
| 9 | holdout (select) | 0.1905 | 0.1600 | 0.0589 | 0.3091 | 0.3680 |
| 12 | freeze (gate) | 0.2664 | 0.2608 | 0.0855 | 0.3210 | 0.3280 |
| 15 | freeze | 0.3330 | 0.3504 | 0.1046 | 0.3143 | 0.2986 |
| 18 | probe | 0.3609 | 0.4133 | 0.1209 | 0.3349 | 0.2925 |

**L=2 n12 gate: PASS.** R_HVA=0.3280 (SNAP 0.0855 / HVA 0.2608). Instance 95% CI 0.3181–0.3416. Seeds BEAT 5/5 (R≤0.85); seeds with R>1: 0/5. Paired n12 instances, SNAP lower ε than HVA: 640/640.

| seed | n12 SNAP | n12 HVA | n12 Fourier | R_HVA | mark | paired SNAP<HVA | best round SNAP | wall SNAP s |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 0.0851 | 0.2632 | 0.2667 | 0.3235 | BEAT | 128/128 | 36 | 6 |
| 1 | 0.0858 | 0.2608 | 0.2667 | 0.3290 | BEAT | 128/128 | 28 | 5 |
| 2 | 0.0855 | 0.2555 | 0.2664 | 0.3348 | BEAT | 128/128 | 32 | 5 |
| 3 | 0.0852 | 0.2621 | 0.2664 | 0.3250 | BEAT | 128/128 | 15 | 5 |
| 4 | 0.0855 | 0.2588 | 0.2664 | 0.3304 | BEAT | 128/128 | 39 | 6 |

HVA mean |β^XX| over layers and seeds: 0.285. SNAP shell RMS off the linear φ_k=a+bk fit, median over layers: 0.2228. Median within-layer std of γ_w: 0.6273.

n15 is not the gate. R_HVA=0.2986 (SNAP 0.1046 / HVA 0.3504).

n18 probe, B=16, not part of the gate. R_HVA=0.2925 (SNAP 0.1209 / HVA 0.4133, Fourier 0.3609).

## L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0452 | 0.0229 | 0.0232 | 0.5142 | 1.0148 |
| 9 | holdout (select) | 0.0586 | 0.0354 | 0.0285 | 0.4866 | 0.8071 |
| 12 | freeze (gate) | 0.0771 | 0.0736 | 0.0394 | 0.5112 | 0.5356 |
| 15 | freeze | 0.0983 | 0.1109 | 0.0450 | 0.4574 | 0.4057 |
| 18 | probe | 0.1114 | 0.1451 | 0.0511 | 0.4592 | 0.3523 |

**L=4 n12 gate: PASS.** R_HVA=0.5356 (SNAP 0.0394 / HVA 0.0736). Instance 95% CI 0.4940–0.5741. Seeds BEAT 5/5 (R≤0.85); seeds with R>1: 0/5. Paired n12 instances, SNAP lower ε than HVA: 640/640.

| seed | n12 SNAP | n12 HVA | n12 Fourier | R_HVA | mark | paired SNAP<HVA | best round SNAP | wall SNAP s |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 0.0394 | 0.0718 | 0.0771 | 0.5492 | BEAT | 128/128 | 38 | 8 |
| 1 | 0.0408 | 0.0764 | 0.0774 | 0.5345 | BEAT | 128/128 | 39 | 9 |
| 2 | 0.0370 | 0.0736 | 0.0760 | 0.5024 | BEAT | 128/128 | 39 | 8 |
| 3 | 0.0531 | 0.0739 | 0.0760 | 0.7192 | BEAT | 128/128 | 39 | 8 |
| 4 | 0.0344 | 0.0732 | 0.0777 | 0.4695 | BEAT | 128/128 | 39 | 9 |

HVA mean |β^XX| over layers and seeds: 0.317. SNAP shell RMS off the linear φ_k=a+bk fit, median over layers: 0.1946. Median within-layer std of γ_w: 0.3159.

n15 is not the gate. R_HVA=0.4057 (SNAP 0.0450 / HVA 0.1109).

n18 probe, B=16, not part of the gate. R_HVA=0.3523 (SNAP 0.0511 / HVA 0.1451, Fourier 0.1114).

## Reading

The comparison is excess energy above the exact ground state of the SAFE Hamiltonian, not a success-probability ratio. A seed BEATs when its own frozen-n12 Med ε ratio is ≤ 0.85.

Seeds 0–2 use the same batches as PR #36 and reproduce that run's frozen n12 Med ε for Fourier, HVA, and SNAP at both depths. Seeds 3 and 4 are new.

At L=4 the widest seed ratio is seed 3 with R_HVA=0.7192 (still a BEAT). Instance intervals end at 0.3416 for L=2 and 0.5741 for L=4, both under 0.85. Seeds BEAT 5/5 at L=2 and 5/5 at L=4.

The n=6 holdout at L=4 has R_HVA=1.0148. That split is not the gate. PR #36 reported 1.016 on the same three-seed holdout.

Frozen n12 is the gate. n15 and the n18 probe are reported beside it and do not move the call. β^XX is away from zero at both depths, so the HVA column is not a Fourier run with a stuck mixer.

L≥8 is outside this seal. PR #36 is the record that HVA takes those depths (L=8 R_HVA=1.372, L=16 R_HVA=2.275). This file does not reopen them.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint.
