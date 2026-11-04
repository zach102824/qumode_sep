# Mild-XX shallow seal — λ=0.50, μ=−0.15, Walsh SNAP vs HVA vs Fourier

**Verdict: needs_edit.** n12 L=4 R_HVA > 0.85 (L=4 soft).

This is the mild-XX cell from the λ×μ grid (PR #44): the nonzero-μ point with Λ12≤0.80 and the lowest L=2 R_HVA among those cells. The three-seed L=2 grid measured Λ12=0.741, product gap=0.188, R_HVA=0.635, against R_HVA=0.750 and Λ12=0.559 at the same λ with μ=0. This five-seed seal, with L=4 and an n=18 freeze, measures n=12 median Λ=0.741, n=18 median Λ=0.742, L=2 R_HVA=0.6350 (5/5 BEAT), L=4 R_HVA=0.8677 (5/5 BEAT), n=12 median product gap=0.188, n=18 median product gap=0.185.

Ratios are gated at the n=12 freeze (B=128). Median cavity Λ and the product gap are gated at n=12 and at n=18 (B=32). n=15 (B=64) is frozen and reported, and it does not enter the verdict. L≥8 is not trained. SAFE (PR #41, λ=0.75, μ=−0.50) had median Λ≈0.941 at n=12 and 0.943 at n=18.

## Gate

Med ε is the median over seeds of each seed's median-over-instances excess energy ⟨H⟩−E₀. R_HVA = Medε_SNAP / Medε_HVA. A seed BEATs when its own n=12 R_HVA ≤ 0.98. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state. Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. The reported Λ is the median over cavities and instances.

- **PASS** when L=2 and L=4 R_HVA ≤ 0.85 at n=12, at least 4/5 seeds BEAT at each depth, median Λ ≤ 0.80 at n=12 and n=18, and the median product gap ≥ 0.03 at n=12 and n=18.
- **needs_edit** when nothing kills the point and at least one PASS bar is missed. Λ in (0.80, 0.90), or a soft L=4 (R_HVA > 0.85 or fewer than 4/5 BEAT), is this case.
- **KILL** when n=12 or n=18 median Λ ≥ 0.90, or when n=12 L=2 R_HVA > 0.98. Either one kills the candidate.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.50 \sum_i X_i - 0.15 \sum_{c}\sum_{i<j\in c} X_i X_j
\]

Candidate point **λ=0.50, μ=−0.15**, written unsat − λ ΣX + μ Σ_cav XX. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. Readout is the Z histogram (unsat) plus the X histogram (⟨X⟩, ⟨XX⟩, and Λ). No Y. The mixer is BinH Hamming-shell SNAP. HVA adds one β^XX angle per layer on the same intra-cavity pairs. Fourier is the two-angle QAOA nested at β^XX=0.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128 and n=15 first 64.
- n=18: 32 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings` and the shallow seals: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε.
- Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k. The three inits are the same state.
- n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Ground state: Λ and product gap

Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has collapsed to an effective spin. A Z-basis product state scores Λ=1/3. The SAFE μ=−0.50 point sat near 0.94. The pass line is 0.80 and the kill line is 0.90. Both lines apply at n=12 and at n=18.

| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | fraction Λ≥0.90 | median product gap | min gap | fraction gap≥0.03 | median Z-peak |
|--:|:-----|--:|--------:|------:|------:|----------------:|----------------:|-------------------:|--------:|------------------:|-------------:|
| 12 | freeze (R + Λ gate) | 128 | 0.741 | 0.534 | 0.913 | 0.215 | 0.004 | 0.188 | 0.061 | 1.000 | 0.0497 |
| 15 | freeze (reported) | 64 | 0.740 | 0.489 | 0.910 | 0.234 | 0.003 | 0.186 | 0.093 | 1.000 | 0.0216 |
| 18 | freeze (Λ gate) | 32 | 0.742 | 0.546 | 0.875 | 0.198 | 0.000 | 0.185 | 0.113 | 1.000 | 0.0096 |

n=12 median Λ = **0.741**. n=15 median Λ = **0.740**. n=18 median Λ = **0.742**. n=12 median product gap = **0.188** (n=15 0.186, n=18 0.185). The n=12 Z-basis peak of the ground state is 0.0497 (n=15 0.0216, n=18 0.0096).

## L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.3726 | 0.3329 | 0.2451 | 0.6577 | 0.7362 |
| 9 | holdout (select) | 0.5031 | 0.4834 | 0.3347 | 0.6652 | 0.6924 |
| 12 | freeze (R gate) | 0.7286 | 0.7246 | 0.4602 | 0.6316 | 0.6350 |
| 15 | freeze (reported) | 0.8659 | 0.8865 | 0.5427 | 0.6268 | 0.6122 |
| 18 | freeze (Λ gate) | 1.1074 | 1.1414 | 0.6890 | 0.6222 | 0.6036 |

**L=2 n12 R_HVA = 0.6350** (SNAP 0.4602 / HVA 0.7246). Seeds BEAT (n12 R_HVA≤0.98): 5/5. Freeze R_HVA is 0.6122 at n=15 and 0.6036 at n=18. Steps of R_HVA: n12→n15 -0.0228, n15→n18 -0.0086.

Paired n12 instances with SNAP ε below HVA: 640/640. Paired n18: 160/160. HVA mean |β^XX| = 0.195.

| seed | n12 SNAP | n12 HVA | n12 Fourier | n12 R_HVA | BEAT | paired n12 | n18 R_HVA | best round | wall s |
|---:|---:|---:|---:|---:|:----:|---:|---:|---:|---:|
| 0 | 0.4602 | 0.7260 | 0.7286 | 0.6338 | yes | 128/128 | 0.5990 | 36 | 5 |
| 1 | 0.4594 | 0.7218 | 0.7287 | 0.6365 | yes | 128/128 | 0.6037 | 39 | 6 |
| 2 | 0.4638 | 0.7246 | 0.7286 | 0.6401 | yes | 128/128 | 0.6057 | 32 | 5 |
| 3 | 0.4588 | 0.7238 | 0.7285 | 0.6339 | yes | 128/128 | 0.6014 | 38 | 5 |
| 4 | 0.4620 | 0.7255 | 0.7301 | 0.6368 | yes | 128/128 | 0.6006 | 39 | 5 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.857, HVA 0.858, SNAP 0.853. At n18: Fourier 0.858, HVA 0.858, SNAP 0.854. The gate uses the ground-state Λ, not these.

## L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1743 | 0.1366 | 0.1543 | 0.8856 | 1.1302 |
| 9 | holdout (select) | 0.2538 | 0.2268 | 0.2236 | 0.8810 | 0.9856 |
| 12 | freeze (R gate) | 0.3475 | 0.3416 | 0.2964 | 0.8528 | 0.8677 |
| 15 | freeze (reported) | 0.4247 | 0.4428 | 0.3568 | 0.8402 | 0.8058 |
| 18 | freeze (Λ gate) | 0.5155 | 0.5674 | 0.4446 | 0.8625 | 0.7836 |

**L=4 n12 R_HVA = 0.8677** (SNAP 0.2964 / HVA 0.3416). Seeds BEAT (n12 R_HVA≤0.98): 5/5. Freeze R_HVA is 0.8058 at n=15 and 0.7836 at n=18. Steps of R_HVA: n12→n15 -0.0618, n15→n18 -0.0222.

Paired n12 instances with SNAP ε below HVA: 619/640. Paired n18: 160/160. HVA mean |β^XX| = 0.237.

| seed | n12 SNAP | n12 HVA | n12 Fourier | n12 R_HVA | BEAT | paired n12 | n18 R_HVA | best round | wall s |
|---:|---:|---:|---:|---:|:----:|---:|---:|---:|---:|
| 0 | 0.2964 | 0.3416 | 0.3476 | 0.8677 | yes | 127/128 | 0.7895 | 38 | 8 |
| 1 | 0.3147 | 0.3401 | 0.3475 | 0.9252 | yes | 111/128 | 0.8422 | 39 | 8 |
| 2 | 0.2783 | 0.3401 | 0.3475 | 0.8184 | yes | 128/128 | 0.7262 | 38 | 8 |
| 3 | 0.2993 | 0.3420 | 0.3495 | 0.8754 | yes | 125/128 | 0.7948 | 39 | 8 |
| 4 | 0.2770 | 0.3424 | 0.3451 | 0.8088 | yes | 128/128 | 0.7089 | 39 | 8 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.827, HVA 0.832, SNAP 0.843. At n18: Fourier 0.823, HVA 0.830, SNAP 0.842. The gate uses the ground-state Λ, not these.

## Reading

The call is **needs_edit**. n12 L=4 R_HVA > 0.85 (L=4 soft).

At the n=12 freeze, L=2 goes R_HVA=0.6350 (Fourier ratio 0.6316, 5/5 BEAT) and L=4 goes R_HVA=0.8677 (Fourier ratio 0.8528, 5/5 BEAT). The other freezes move to L=2 n15/n18 0.6122/0.6036 and L=4 n15/n18 0.8058/0.7836. HVA |β^XX| is 0.195 at L=2 and 0.237 at L=4. On the n=6 holdout, L=2 R_HVA=0.7362 and L=4 R_HVA=1.1302 (SNAP does not lead HVA at L=4 on that small set). The ratio gate is the n=12 freeze.

Ground-state median Λ is 0.741 / 0.740 / 0.742 at n=12 / 15 / 18. The fraction of cavities above 0.80 is 0.215 / 0.234 / 0.198. The fraction at or above 0.90 is 0.004 / 0.003 / 0.000. Λ equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩, computed from the joint X histogram of the ground state. Values near 1 mean the intra-cavity space has collapsed to an effective spin, which is what killed SAFE μ=−0.50 (Λ≈0.941 at n=12). Grid #44's three-seed L=2 measurement of this same cell was Λ12=0.741, gap=0.188, R_HVA=0.635. The μ=0 cell at λ=0.50 on that grid had R_HVA=0.750 and Λ12=0.559.

The product gap is 0.188 / 0.186 / 0.185 at n=12 / 15 / 18 (bar ≥ 0.03 on the n=12 and n=18 medians). Z-basis peaks stay 0.0497 / 0.0216 / 0.0096.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint. Ground-state Λ and the product gap are in `diag_n{n}_{tag}.npz`.

