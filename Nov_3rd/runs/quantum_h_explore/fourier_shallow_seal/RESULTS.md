# Shallow seal — Walsh SNAP vs plain Fourier QAOA

**Verdict: PASS.** L=2 → PASS (R_Fourier=0.3210, 5/5 seeds BEAT, CI 0.3126–0.3365) L=4 → PASS (R_Fourier=0.5112, 5/5 seeds BEAT, CI 0.4740–0.5464).

This seal is L∈{2, 4} only. The call is R_Fourier on frozen n=12. HVA from PR #38 is a column in the table. PR #39 left L=16 at needs_edit (R_HVA=1.006). Those depths were not trained here and are not claimed.

## Gate

Frozen n=12. Medε is the median over seeds of each seed's median-over-instances excess energy ⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier.

- **PASS** on a depth when R_Fourier ≤ 0.85, at least 4 of 5 seeds have their own n12 R_Fourier ≤ 0.85, and the instance-bootstrap 95% interval ends strictly below 0.95.
- **needs_edit** when that depth is not a PASS and Fourier has not taken it.
- **KILL** when R_Fourier > 0.98 at n12, or at least 4 of 5 seeds have R_Fourier > 0.98.
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
| Fourier QAOA | γ, β | \(\bigotimes_i \exp(-i \beta X_i/2)\). No XX mixer. |
| Walsh SNAP | γ_{w=1..4}, φ_k | BinH Hamming shell. Linear φ_k=−β k is the X mixer. |
| HVA (cited, #38) | γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\) then \(U_X(\beta^X)\). Not trained here. |

Paper init β=0.01, γ=γ_w=-0.01, φ_k=−β k. Fourier and SNAP start on the same state. The seed only changes the L-BFGS batches. HVA in PR #38 used the same init with β^XX=0.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n12 first 128, n15 first 64.
- n18 probe: 16 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. The probe is not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. No Adam.
- Each seed restores the round with the best n9-holdout Medε.
- Instance 95% CI: 4000 bootstrap draws, resampling instances inside each seed, seeds held fixed. RNG seed 5209.
- HVA medians are read from `hva_pr38.json` (PR #38 five-seed shallow seal). They are not refit.
- n21 was not read or written. L≥8 was not trained.

## L=2

| n | role | Medε Fourier | Medε HVA (#38) | Medε SNAP | R_Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1369 | 0.0952 | 0.0465 | 0.3399 | 0.4887 |
| 9 | holdout (select) | 0.1905 | 0.1600 | 0.0589 | 0.3091 | 0.3680 |
| 12 | freeze (gate) | 0.2664 | 0.2608 | 0.0855 | 0.3210 | 0.3280 |
| 15 | freeze | 0.3330 | 0.3504 | 0.1046 | 0.3142 | 0.2986 |
| 18 | probe | 0.3609 | 0.4133 | 0.1209 | 0.3349 | 0.2924 |

**L=2 n12 gate: PASS.** R_Fourier=0.3210 (SNAP 0.0855 / Fourier 0.2664). Instance 95% CI 0.3126–0.3365. Seeds BEAT 5/5 (R≤0.85); seeds with R>0.98: 0/5. Paired n12 instances, SNAP lower ε than Fourier: 640/640.

| seed | n12 SNAP | n12 Fourier | R_Fourier | mark | paired SNAP<Fourier | n12 HVA (#38) | best round SNAP | wall SNAP s |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.0851 | 0.2667 | 0.3193 | BEAT | 128/128 | 0.2632 | 36 | 7 |
| 1 | 0.0858 | 0.2667 | 0.3218 | BEAT | 128/128 | 0.2608 | 28 | 7 |
| 2 | 0.0855 | 0.2664 | 0.3210 | BEAT | 128/128 | 0.2555 | 32 | 7 |
| 3 | 0.0852 | 0.2664 | 0.3197 | BEAT | 128/128 | 0.2621 | 15 | 7 |
| 4 | 0.0855 | 0.2664 | 0.3210 | BEAT | 128/128 | 0.2588 | 39 | 7 |

Fourier mean |β| over layers and seeds: 0.8225. SNAP shell RMS off the linear φ_k=a+bk fit, median over layers: 0.2228. Median within-layer std of γ_w: 0.6275.
Cited HVA mean |β^XX| from PR #38: 0.285. That mixer is absent from the Fourier column.

n15 is not the gate. R_Fourier=0.3142 (SNAP 0.1046 / Fourier 0.3330).

n18 probe, B=16, not part of the gate. R_Fourier=0.3349 (SNAP 0.1209 / Fourier 0.3609).

## L=4

| n | role | Medε Fourier | Medε HVA (#38) | Medε SNAP | R_Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0452 | 0.0229 | 0.0232 | 0.5142 | 1.0148 |
| 9 | holdout (select) | 0.0586 | 0.0354 | 0.0285 | 0.4866 | 0.8071 |
| 12 | freeze (gate) | 0.0771 | 0.0736 | 0.0394 | 0.5112 | 0.5356 |
| 15 | freeze | 0.0983 | 0.1109 | 0.0450 | 0.4574 | 0.4057 |
| 18 | probe | 0.1114 | 0.1451 | 0.0511 | 0.4592 | 0.3523 |

**L=4 n12 gate: PASS.** R_Fourier=0.5112 (SNAP 0.0394 / Fourier 0.0771). Instance 95% CI 0.4740–0.5464. Seeds BEAT 5/5 (R≤0.85); seeds with R>0.98: 0/5. Paired n12 instances, SNAP lower ε than Fourier: 639/640.

| seed | n12 SNAP | n12 Fourier | R_Fourier | mark | paired SNAP<Fourier | n12 HVA (#38) | best round SNAP | wall SNAP s |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.0394 | 0.0771 | 0.5112 | BEAT | 128/128 | 0.0718 | 38 | 9 |
| 1 | 0.0408 | 0.0774 | 0.5276 | BEAT | 128/128 | 0.0764 | 39 | 10 |
| 2 | 0.0370 | 0.0760 | 0.4864 | BEAT | 128/128 | 0.0736 | 39 | 10 |
| 3 | 0.0531 | 0.0760 | 0.6994 | BEAT | 127/128 | 0.0739 | 39 | 10 |
| 4 | 0.0344 | 0.0777 | 0.4424 | BEAT | 128/128 | 0.0732 | 39 | 10 |

Fourier mean |β| over layers and seeds: 0.9668. SNAP shell RMS off the linear φ_k=a+bk fit, median over layers: 0.1946. Median within-layer std of γ_w: 0.3159.
Cited HVA mean |β^XX| from PR #38: 0.317. That mixer is absent from the Fourier column.

n15 is not the gate. R_Fourier=0.4574 (SNAP 0.0450 / Fourier 0.0983).

n18 probe, B=16, not part of the gate. R_Fourier=0.4592 (SNAP 0.0511 / Fourier 0.1114).

## Reading

Excess energy is ⟨H⟩−E₀ on the SAFE Hamiltonian. A seed BEATs when its own frozen-n12 Medε ratio against Fourier is ≤ 0.85.

Fourier's mixer is ⊗ exp(−i β X/2). Saved Fourier checkpoints contain only β and γ. The XX mixer belongs to the HVA column, which is cited from PR #38 and is not this gate.

L=2: widest seed is seed 1 with R_Fourier=0.3218. Instance interval 0.3126–0.3365. Seeds BEAT 5/5.
L=4: widest seed is seed 3 with R_Fourier=0.6994. Instance interval 0.4740–0.5464. Seeds BEAT 5/5.

The n=6 holdout at L=4 has R_Fourier=0.5142. That split is not the gate. Frozen n12 is the gate.

Frozen n12 Medε for Fourier and SNAP matches the PR #38 five-seed trains (max |Δ|=1.116e-12). The cited HVA column is the same n12 slice.

n15 and the n18 probe sit beside the gate and do not move it. L≥8 was not trained. PR #39 left the L=16 rescue at needs_edit (R_HVA=1.006). This file does not claim those depths.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint. Fourier files have `betas` and `gammas` only.
