# λ=0.50, μ=0 shallow seal — Walsh SNAP vs Fourier through n=18

**Verdict: PASS.** All four PASS bars hold.

This is the primary non-trivialized cell from Round11. PR #45 (TF-only) and PR #44 (λ×μ grid) both kept λ=0.50, μ=0: n=12 median cavity Λ=0.559 and L=2 R_Fourier≈0.751, with L=4 a soft 0.904. SAFE μ=−0.50 was killed by Λ≈0.94 (PR #41). μ=−0.25 stayed soft at Λ=0.893 (PR #43). The question here is whether that λ=0.50, μ=0 lead survives a five-seed freeze at n=12 and the n=15 / n=18 probes without the cavity lock drifting up.

## Gate

Med ε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier. A seed BEATs when its own n=12 L=2 R_Fourier ≤ 0.98. Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. The reported Λ is the median over cavities and instances. The product gap is (E_prod − E_0)/|E_0| for the best real XZ product state.

- **PASS** when all four hold: (1) n=12 L=2 R_Fourier ≤ 0.85 and at least 4/5 seeds BEAT; (2) median Λ ≤ 0.80 at n=12 and at n=18; (3) median product gap ≥ 0.03 at n=12 and at n=18; (4) L=2 R_Fourier rises by at most 0.05 on each +3-qubit step n=12→15 and n=15→18.
- **needs_edit** when nothing kills the point and at least one PASS bar is missed. Λ drifting above 0.80 at n=18 is this case.
- **KILL** when n=18 median Λ ≥ 0.90, or when L=2 R_Fourier > 0.98 at n=12, n=15, or n=18.
- **L=4** is reported and is not a PASS bar. A ratio above 0.85 is noted as soft (Round11 n=12 L=4 was about 0.90).

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.50 \sum_i X_i
\]

μ = 0, so there is no intra-cavity XX term. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. The circuit phases Cn = unsat · n/m. ⟨H⟩ uses the raw unsat count from the Z histogram and ⟨X_i⟩ from the X histogram. No Y, and no XX correlator in the energy. The mixer is BinH Hamming-shell SNAP with a Walsh-graded cost. Fourier is the two-angle QAOA. HVA is not trained: β^XX = 0 is Fourier.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128. Probe n=15 first 64.
- n=18: 32 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings` and the Round11 shallow seals: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε. The batch stream matches the Round11 TF-only trainer at λ=0.50.
- Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. Fourier and SNAP start on the same state.
- L≥8 was not trained. n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Ground state: Λ and product gap

Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has collapsed to an effective spin. A Z-basis product state scores Λ=1/3. The pass line is 0.80 and the kill line is 0.90 at n=18. SAFE μ=−0.50 sat near 0.94.

| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | fraction Λ≥0.90 | median product gap | min gap | fraction gap≥0.03 | median Z-peak |
|--:|:-----|--:|--------:|------:|------:|----------------:|----------------:|-------------------:|--------:|------------------:|-------------:|
| 12 | freeze (gate) | 128 | 0.5585 | 0.4037 | 0.8593 | 0.010 | 0.000 | 0.1102 | 0.0413 | 1.000 | 0.1569 |
| 15 | probe | 64 | 0.5521 | 0.4021 | 0.8701 | 0.016 | 0.000 | 0.1017 | 0.0432 | 1.000 | 0.1053 |
| 18 | probe (gate) | 32 | 0.5477 | 0.3936 | 0.7656 | 0.000 | 0.000 | 0.1115 | 0.0460 | 1.000 | 0.0650 |

n=12 median Λ = **0.5585**. n=15 median Λ = **0.5521**. n=18 median Λ = **0.5477**. n=12 median product gap = **0.1102** (n=15 0.1017, n=18 0.1115). The n=12 Z-basis peak of the ground state is 0.1569 (n=15 0.1053, n=18 0.0650). Max ground-state residual is 2.253e-09.

## L=2

| n | role | Medε Fourier | Medε SNAP | R_Fourier |
|--:|:-----|-------------:|----------:|----------:|
| 6 | holdout | 0.6067 | 0.4441 | 0.7319 |
| 9 | holdout (select) | 0.8908 | 0.6892 | 0.7736 |
| 12 | freeze (gate) | 1.1686 | 0.8779 | 0.7512 |
| 15 | probe | 1.4101 | 1.0638 | 0.7545 |
| 18 | probe | 1.7711 | 1.3267 | 0.7491 |

**L=2 n12 R_Fourier = 0.7512** (SNAP 0.8779 / Fourier 1.1686). Seeds BEAT (n12 R≤0.98): 5/5. Probe R is 0.7545 at n=15 and 0.7491 at n=18. Steps of R: n12→n15 +0.0032, n15→n18 -0.0054.

Paired instances with SNAP ε below Fourier: n12 638/640, n15 320/320, n18 160/160. Median shell RMS (distance of φ_k from the best line a+bk) = 0.0452.

| seed | n12 SNAP | n12 Fourier | n12 R | BEAT | paired n12 | n15 R | n18 R | shell RMS | SNAP best round | wall s |
|---:|---:|---:|---:|:----:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8769 | 1.1686 | 0.7504 | yes | 128/128 | 0.7551 | 0.7492 | 0.0845 | 39 | 5 |
| 1 | 0.8767 | 1.1686 | 0.7502 | yes | 128/128 | 0.7560 | 0.7491 | 0.0452 | 34 | 5 |
| 2 | 0.8779 | 1.1686 | 0.7512 | yes | 128/128 | 0.7545 | 0.7497 | 0.0447 | 38 | 5 |
| 3 | 0.8802 | 1.1678 | 0.7537 | yes | 127/128 | 0.7449 | 0.7444 | 0.0447 | 39 | 5 |
| 4 | 0.8789 | 1.1686 | 0.7521 | yes | 127/128 | 0.7513 | 0.7436 | 0.0533 | 24 | 5 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.780, SNAP 0.787. At n18: Fourier 0.781, SNAP 0.789. The gate uses the ground-state Λ, not these.

## L=4

| n | role | Medε Fourier | Medε SNAP | R_Fourier |
|--:|:-----|-------------:|----------:|----------:|
| 6 | holdout | 0.3192 | 0.2956 | 0.9258 |
| 9 | holdout (select) | 0.5022 | 0.4736 | 0.9430 |
| 12 | freeze | 0.6462 | 0.5839 | 0.9037 |
| 15 | probe | 0.7928 | 0.7240 | 0.9133 |
| 18 | probe | 1.0079 | 0.9190 | 0.9118 |

**L=4 n12 R_Fourier = 0.9037** (SNAP 0.5839 / Fourier 0.6462). Seeds BEAT (n12 R≤0.98): 5/5. Probe R is 0.9133 at n=15 and 0.9118 at n=18. Steps of R: n12→n15 +0.0097, n15→n18 -0.0015.

Paired instances with SNAP ε below Fourier: n12 541/640, n15 291/320, n18 144/160. Median shell RMS (distance of φ_k from the best line a+bk) = 0.2157.

| seed | n12 SNAP | n12 Fourier | n12 R | BEAT | paired n12 | n15 R | n18 R | shell RMS | SNAP best round | wall s |
|---:|---:|---:|---:|:----:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5797 | 0.6447 | 0.8992 | yes | 122/128 | 0.9027 | 0.9068 | 0.2157 | 39 | 8 |
| 1 | 0.5936 | 0.6475 | 0.9168 | yes | 111/128 | 0.9308 | 0.9313 | 0.2194 | 39 | 8 |
| 2 | 0.5402 | 0.6461 | 0.8360 | yes | 128/128 | 0.8576 | 0.8744 | 0.1821 | 38 | 6 |
| 3 | 0.5839 | 0.6462 | 0.9037 | yes | 121/128 | 0.9164 | 0.9074 | 0.2158 | 39 | 8 |
| 4 | 0.6317 | 0.6473 | 0.9760 | yes | 59/128 | 0.9966 | 0.9878 | 0.2051 | 39 | 7 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.726, SNAP 0.762. At n18: Fourier 0.722, SNAP 0.764. The gate uses the ground-state Λ, not these.

L=4 note: n12 L=4 R_Fourier=0.9037 is above 0.85, reported as a soft lead and not required for PASS.

## Reading

The call is **PASS**. All four PASS bars hold.

At the n=12 freeze, L=2 R_Fourier=0.7512 (SNAP 0.8779, Fourier 1.1686) with 5/5 seeds at or under the 0.98 BEAT line. The same ratio is 0.7545 at n=15 and 0.7491 at n=18. The +3-qubit steps are +0.0032 and -0.0054 (bar ≤ +0.05 on each step).

Ground-state median Λ is 0.5585 / 0.5521 / 0.5477 at n=12 / 15 / 18. The fraction of cavities above 0.80 is 0.010 / 0.016 / 0.000. The fraction at or above 0.90 is 0.000 / 0.000 / 0.000. The product gap is 0.1102 / 0.1017 / 0.1115 (bar ≥ 0.03 at n=12 and n=18). Z-basis peaks are 0.1569 / 0.1053 / 0.0650.

Λ equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩, computed from the joint X histogram of the ground state. Values near 1 mean the intra-cavity space has collapsed to an effective spin, which is what killed μ=−0.50. Round11's n=12 anchor at this point was Λ=0.5585, gap=0.1102, L=2 R=0.7512, L=4 R=0.9037. The n=12 freeze here reproduces that anchor.

L=4 R_Fourier is 0.9037 / 0.9133 / 0.9118 at n=12 / 15 / 18. n12 L=4 R_Fourier=0.9037 is above 0.85, reported as a soft lead and not required for PASS. The soft tail is seed 4 n12/n15/n18 R=0.9760/0.9966/0.9878.

L=2 shell RMS median is 0.0452; L=4 shell RMS median is 0.2157. A small L=2 shell RMS means the Hamming shell stays close to a linear φ_k = a + bk, so part of the L=2 lead over Fourier is the Walsh-graded cost (four γ weights per layer against Fourier's one).

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n=9-holdout checkpoint. Ground-state Λ, the product gap, and the Z-peak are in `diag_n{n}_{role}.npz`.

