# μ=−0.25 shallow Λ seal — λ=0.75, Walsh SNAP vs HVA vs Fourier

**Verdict: needs_edit.** n12 median Λ > 0.80.

This is the softer-μ candidate after the SAFE lock (μ=−0.50) was killed by cavity Λ. PR #41 reported median Λ≈0.941 at n=12 and 0.943 at n=18, with shallow R_HVA well under 0.5. The question here is whether μ=−0.25 keeps the ground-state cavities from collapsing (Λ≤0.80) while shallow Walsh still beats matched HVA (R_HVA≤0.85, ≥4/5 seeds). Measured n=12 median Λ=0.893, L=2 R_HVA=0.4340 (5/5 BEAT), L=4 R_HVA=0.7581 (5/5 BEAT), n=12 median product gap=0.109.

The gate is the n=12 freeze (B=128). n=15 (B=64) and n=18 (B=32) are probes and do not enter the verdict. L≥8 is not trained. PR #25 is the earlier μ=−0.25 record against Fourier (L=2 n=12 R≈0.483, L=16 n=12 R≈0.342, product gap≈0.110), not against HVA.

## Gate

Med ε is the median over seeds of each seed's median-over-instances excess energy ⟨H⟩−E₀. R_HVA = Medε_SNAP / Medε_HVA. A seed BEATs when its own n=12 R_HVA ≤ 0.98. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state. Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. The reported Λ is the median over cavities and instances.

- **PASS** when, at n=12, L=2 and L=4 R_HVA ≤ 0.85, at least 4/5 seeds BEAT at each depth, median Λ ≤ 0.80, and the median product gap ≥ 0.03.
- **needs_edit** when nothing kills the point and at least one PASS bar is missed. R inside the bar with Λ in (0.80, 0.90) is this case.
- **KILL** when n=12 median Λ ≥ 0.90, or when n=12 L=2 R_HVA > 0.98. Either one kills the candidate.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.25 \sum_{c}\sum_{i<j\in c} X_i X_j
\]

Candidate point **λ=0.75, μ=−0.25**, written unsat − λ ΣX + μ Σ_cav XX. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. Readout is the Z histogram (unsat) plus the X histogram (⟨X⟩, ⟨XX⟩, and Λ). No Y. The mixer is BinH Hamming-shell SNAP. HVA adds one β^XX angle per layer on the same intra-cavity pairs. Fourier is the two-angle QAOA nested at β^XX=0.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128. Probe n=15 first 64.
- n=18: 32 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings` and the PR #41 shallow seal: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε.
- Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k. The three inits are the same state.
- n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Ground state: Λ and product gap

Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has collapsed to an effective spin. A Z-basis product state scores Λ=1/3. The SAFE μ=−0.50 point sat near 0.94. The pass line is 0.80 and the kill line is 0.90.

| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | fraction Λ≥0.90 | median product gap | min gap | fraction gap≥0.03 | median Z-peak |
|--:|:-----|--:|--------:|------:|------:|----------------:|----------------:|-------------------:|--------:|------------------:|-------------:|
| 12 | freeze (gate) | 128 | 0.893 | 0.745 | 0.951 | 0.957 | 0.414 | 0.109 | 0.064 | 1.000 | 0.0086 |
| 15 | probe | 64 | 0.889 | 0.728 | 0.958 | 0.963 | 0.391 | 0.105 | 0.063 | 1.000 | 0.0026 |
| 18 | probe | 32 | 0.893 | 0.742 | 0.950 | 0.948 | 0.411 | 0.107 | 0.080 | 1.000 | 0.0007 |

n=12 median Λ = **0.893**. n=15 median Λ = **0.889**. n=18 median Λ = **0.893**. n=12 median product gap = **0.109** (n=15 0.105, n=18 0.107). The n=12 Z-basis peak of the ground state is 0.0086 (n=18 0.0007).

## L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1939 | 0.1558 | 0.0904 | 0.4663 | 0.5805 |
| 9 | holdout (select) | 0.2695 | 0.2485 | 0.1174 | 0.4354 | 0.4722 |
| 12 | freeze (gate) | 0.3880 | 0.3828 | 0.1662 | 0.4283 | 0.4340 |
| 15 | probe | 0.4679 | 0.4876 | 0.2011 | 0.4299 | 0.4125 |
| 18 | probe | 0.5796 | 0.6277 | 0.2610 | 0.4502 | 0.4158 |

**L=2 n12 R_HVA = 0.4340** (SNAP 0.1662 / HVA 0.3828). Seeds BEAT (n12 R_HVA≤0.98): 5/5. Probe R_HVA is 0.4125 at n=15 and 0.4158 at n=18. Steps of R_HVA: n12→n15 -0.0215, n15→n18 0.0032.

Paired n12 instances with SNAP ε below HVA: 640/640. Paired n18: 160/160. HVA mean |β^XX| = 0.231.

| seed | n12 SNAP | n12 HVA | n12 Fourier | n12 R_HVA | BEAT | paired n12 | n18 R_HVA | best round | wall s |
|---:|---:|---:|---:|---:|:----:|---:|---:|---:|---:|
| 0 | 0.1660 | 0.3828 | 0.3872 | 0.4337 | yes | 128/128 | 0.4144 | 39 | 5 |
| 1 | 0.1670 | 0.3826 | 0.3889 | 0.4366 | yes | 128/128 | 0.4193 | 37 | 5 |
| 2 | 0.1664 | 0.3820 | 0.3869 | 0.4357 | yes | 128/128 | 0.4170 | 38 | 6 |
| 3 | 0.1662 | 0.3831 | 0.3880 | 0.4337 | yes | 128/128 | 0.4170 | 38 | 6 |
| 4 | 0.1660 | 0.3829 | 0.3894 | 0.4335 | yes | 128/128 | 0.4148 | 21 | 5 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.924, HVA 0.922, SNAP 0.919. At n18: Fourier 0.924, HVA 0.920, SNAP 0.918. The gate uses the ground-state Λ, not these.

## L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0715 | 0.0473 | 0.0557 | 0.7791 | 1.1784 |
| 9 | holdout (select) | 0.0973 | 0.0785 | 0.0690 | 0.7094 | 0.8788 |
| 12 | freeze (gate) | 0.1390 | 0.1300 | 0.0985 | 0.7090 | 0.7581 |
| 15 | probe | 0.1661 | 0.1738 | 0.1144 | 0.6890 | 0.6582 |
| 18 | probe | 0.2077 | 0.2405 | 0.1484 | 0.7142 | 0.6168 |

**L=4 n12 R_HVA = 0.7581** (SNAP 0.0985 / HVA 0.1300). Seeds BEAT (n12 R_HVA≤0.98): 5/5. Probe R_HVA is 0.6582 at n=15 and 0.6168 at n=18. Steps of R_HVA: n12→n15 -0.0999, n15→n18 -0.0414.

Paired n12 instances with SNAP ε below HVA: 637/640. Paired n18: 160/160. HVA mean |β^XX| = 0.271.

| seed | n12 SNAP | n12 HVA | n12 Fourier | n12 R_HVA | BEAT | paired n12 | n18 R_HVA | best round | wall s |
|---:|---:|---:|---:|---:|:----:|---:|---:|---:|---:|
| 0 | 0.0985 | 0.1279 | 0.1387 | 0.7706 | yes | 128/128 | 0.6332 | 39 | 8 |
| 1 | 0.0938 | 0.1342 | 0.1388 | 0.6993 | yes | 128/128 | 0.5464 | 39 | 8 |
| 2 | 0.1019 | 0.1300 | 0.1398 | 0.7841 | yes | 128/128 | 0.6189 | 39 | 8 |
| 3 | 0.1094 | 0.1279 | 0.1390 | 0.8550 | yes | 125/128 | 0.7037 | 39 | 8 |
| 4 | 0.0920 | 0.1322 | 0.1400 | 0.6958 | yes | 128/128 | 0.5550 | 39 | 8 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.914, HVA 0.911, SNAP 0.915. At n18: Fourier 0.914, HVA 0.912, SNAP 0.917. The gate uses the ground-state Λ, not these.

## Reading

The call is **needs_edit**. n12 median Λ > 0.80.

At the n=12 freeze, L=2 goes R_HVA=0.4340 (Fourier ratio 0.4283) and L=4 goes R_HVA=0.7581 (Fourier ratio 0.7090). The probes move to L=2 n15/n18 0.4125/0.4158 and L=4 n15/n18 0.6582/0.6168. HVA |β^XX| is 0.231 at L=2 and 0.271 at L=4. L=4 on the n=6 holdout has R_HVA=1.1784, so SNAP does not lead HVA on that small set. The lead is at the n=12 freeze and on both probes.

Ground-state median Λ is 0.893 / 0.889 / 0.893 at n=12 / 15 / 18. The fraction of cavities above 0.80 is 0.957 / 0.963 / 0.948. The fraction at or above 0.90 is 0.414 / 0.391 / 0.411. Λ equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩, computed from the joint X histogram of the ground state. Values near 1 mean the intra-cavity space has collapsed to an effective spin, which is what killed μ=−0.50. The only missed PASS bar is the n=12 median Λ. The same median sits in (0.80, 0.90) at n=15 and n=18, and it stays under the 0.90 kill line at every reported size. Relative to SAFE μ=−0.50 (Λ≈0.94), the softer XX term moved the median down by about 0.05 and did not restore Λ≤0.80.

The product gap is 0.109 / 0.105 / 0.107 at n=12 / 15 / 18 (bar ≥ 0.03 on the n=12 median). Z-basis peaks stay 0.0086 / 0.0026 / 0.0007, so the ground state is not a classical bitstring.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint. Ground-state Λ and the product gap are in `diag_n{n}_{tag}.npz`.

