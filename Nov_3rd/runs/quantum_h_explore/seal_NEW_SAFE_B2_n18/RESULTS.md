# NEW_SAFE B=2 Walsh SNAP seal through n=18

**Verdict: PASS.** n12 L=2 R≤0.85, B=2 stays inside the #51 band at n12/n15/n18, and median Λ≤0.80.

Locked point: λ=0.75, μ=0, H = diag(unsat) − 0.75 ΣX. The mixer is the PR #57 quadratic Hamming shell, shared across cavities: φ_k = c₁ k + c₂ k² on popcount k=0..3 (orthonormal coordinates, global phase fixed). The cost pulse is the Walsh grading of degrees 1..4. Fourier QAOA is trained on the same batches. PR #51 sealed the four-phase shell (B=4) through n=18 at R=0.6209 / 0.6016 / 0.5880. PR #57 showed that at the n=12 freeze, L=2 B=2 matches that shell (Medε 0.4261, R=0.6197) and did not probe n=15 or n=18. This file is that probe. n=21 was not loaded.

## Gate

Medε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. R_Fourier = Medε_B2 / Medε_Fourier on this run's matched Fourier. Λ is the median over cavities and instances of ⟨(Σ_{i∈c} X_i / 3)²⟩ on the true ground state's X histogram. It does not depend on B.

- **PASS** when all three hold: (1) n=12 L=2 R_Fourier ≤ 0.85; (2) at each of n=12, n=15, and n=18, Medε_B2 ≤ 1.05 × the PR #51 L=2 Medε **or** |R_Fourier − R_#51| ≤ 0.02, where R_#51 is 0.6209 / 0.6016 / 0.5880; (3) median Λ ≤ 0.80 at n=12, n=15, and n=18.
- **needs_edit** otherwise. **B=2 softens on scale** when n=12 stays inside the #51 band and n=15 or n=18 leaves it.
- L=4, when trained, is reported and is not a PASS bar.

## Hamiltonian and protocol

H = diag(unsat) − 0.75 Σ_i X_i. μ = 0, so there is no intra-cavity XX term. Cavity c owns qubits [3c, 3c+3). Circuit phases use Cn = unsat · n/m. ⟨H⟩ uses the raw unsat count from the Z histogram and ⟨X_i⟩ from the X histogram.

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128. Probe n=15 first 64.
- n=18: 32 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings` and PR #57: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Medε. The batch stream is `numpy` Generator(seed) and `torch.manual_seed(seed)`, shared by Fourier and B=2, independent of depth, matching PR #57.
- Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. B=2 starts on that shell. Basis reconstruction error max |φ_recon − φ_paper| = 3.469e-18.
- n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Ground state

| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | median product gap | median Z-peak |
|--:|:-----|--:|--------:|------:|------:|----------------:|-------------------:|-------------:|
| 12 | freeze (gate) | 128 | 0.7477 | 0.5498 | 0.9117 | 0.230 | 0.1265 | 0.0379 |
| 15 | probe | 64 | 0.7489 | 0.5058 | 0.9142 | 0.269 | 0.1294 | 0.0148 |
| 18 | probe (gate) | 32 | 0.7487 | 0.5687 | 0.8748 | 0.203 | 0.1336 | 0.0064 |

Λ→1 means every cavity has collapsed to one X-magnetization sector. A Z-basis product state scores Λ=1/3. The pass line is 0.80. Max ground-state residual across these pools is 4.928e-09.

## L=2 against the PR #51 table

| n | Medε B=2 | Medε #51 B=4 | Medε ratio | Medε Fourier | Medε #51 Fourier | R_Fourier | R #51 | ΔR | band |
|--:|--------:|-------------:|-----------:|-------------:|-----------------:|----------:|------:|---:|:-----|
| 12 | 0.4261 | 0.4271 | 0.998 | 0.6876 | 0.6879 | 0.6197 | 0.6209 | -0.0012 | Medε |
| 15 | 0.5043 | 0.5066 | 0.995 | 0.8434 | 0.8421 | 0.5979 | 0.6016 | -0.0037 | Medε |
| 18 | 0.6298 | 0.6352 | 0.991 | 1.0807 | 1.0803 | 0.5828 | 0.5880 | -0.0052 | Medε |

**L=2 n12 R_Fourier = 0.6197** (B=2 0.4261 / Fourier 0.6876). Seeds with n12 R≤0.85: 5/5. Seeds with n12 R≤0.98: 5/5. R at n15 is 0.5979, at n18 is 0.5828.

Holdout (checkpoint distribution, not a gate):

| n | Medε B=2 | Medε Fourier | R_Fourier |
|--:|--------:|-------------:|----------:|
| 6 | 0.2170 | 0.3580 | 0.6061 |
| 9 | 0.2950 | 0.4722 | 0.6248 |

Paired instances with B=2 ε below Fourier: n12 640/640, n15 320/320, n18 160/160.

| seed | n12 B=2 | n12 Fourier | n12 R | ≤0.85 | paired n12 | n15 B=2 | n15 R | n18 B=2 | n18 R | best round | wall s |
|---:|---:|---:|---:|:----:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4261 | 0.6876 | 0.6197 | yes | 128/128 | 0.5043 | 0.5975 | 0.6315 | 0.5842 | 32 | 7 |
| 1 | 0.4259 | 0.6878 | 0.6192 | yes | 128/128 | 0.5049 | 0.5991 | 0.6298 | 0.5829 | 16 | 7 |
| 2 | 0.4260 | 0.6876 | 0.6195 | yes | 128/128 | 0.5044 | 0.5981 | 0.6310 | 0.5839 | 17 | 7 |
| 3 | 0.4263 | 0.6876 | 0.6200 | yes | 128/128 | 0.5041 | 0.5978 | 0.6289 | 0.5817 | 34 | 7 |
| 4 | 0.4262 | 0.6878 | 0.6198 | yes | 128/128 | 0.5035 | 0.5972 | 0.6290 | 0.5821 | 17 | 7 |

### Quote check

| source | n | Medε shell | Medε Fourier | R | Λ |
|:-------|--:|----------:|-------------:|--:|--:|
| PR #51 B=4 | 12 | 0.4271 | 0.6879 | 0.6209 | 0.7477 |
| PR #51 B=4 | 15 | 0.5066 | 0.8421 | 0.6016 | 0.7489 |
| PR #51 B=4 | 18 | 0.6352 | 1.0803 | 0.5880 | 0.7487 |
| PR #57 B=2 | 12 | 0.4261 | 0.6876 | 0.6197 | 0.7477 |
| this run B=2 | 12 | 0.4261 | 0.6876 | 0.6197 | 0.7477 |
| this run B=2 | 15 | 0.5043 | 0.8434 | 0.5979 | 0.7489 |
| this run B=2 | 18 | 0.6298 | 1.0807 | 0.5828 | 0.7487 |

Deltas of this run's n=12 against PR #57 (same ansatz and batch stream): Medε +0.0000, Fourier +0.0000, R -0.0000, Λ +0.0000. Deltas against PR #51 use the band table above. The two earlier runs do not share a batch stream, so a Fourier delta against #51 is a different draw, not a bug.

## L=4 (not a PASS bar)

Same training budget as L=2. PR #51's four-phase shell was a soft lead here (n12 R=0.8881) and was not a PASS bar. PR #57 found L=4 B=2 stronger than that shell at n=12 (Medε 0.1958, R=0.6174).

| n | Medε B=2 | Medε #51 L=4 | ratio | Medε Fourier | R_Fourier | R #51 L=4 | ΔR |
|--:|--------:|-------------:|------:|-------------:|----------:|----------:|---:|
| 12 | 0.1958 | 0.2819 | 0.695 | 0.3172 | 0.6174 | 0.8881 | -0.2707 |
| 15 | 0.2434 | 0.3375 | 0.721 | 0.3894 | 0.6252 | 0.8684 | -0.2432 |
| 18 | 0.2959 | 0.4265 | 0.694 | 0.4878 | 0.6067 | 0.8756 | -0.2689 |

L=4 n12 R_Fourier=0.6174 (B=2 0.1958 / Fourier 0.3172). Seeds with n12 R≤0.85: 5/5. Deltas of this run's n=12 against PR #57 L=4 B=2: Medε +0.0000, Fourier +0.0000, R -0.0000. This depth does not change the verdict.

| seed | n12 B=2 | n12 Fourier | n12 R | n15 R | n18 R | best round | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1962 | 0.3184 | 0.6162 | 0.6236 | 0.6106 | 38 | 11 |
| 1 | 0.1958 | 0.3169 | 0.6181 | 0.6262 | 0.6050 | 39 | 11 |
| 2 | 0.1975 | 0.3172 | 0.6226 | 0.6337 | 0.6139 | 36 | 11 |
| 3 | 0.1949 | 0.3163 | 0.6162 | 0.6237 | 0.6055 | 35 | 11 |
| 4 | 0.1949 | 0.3181 | 0.6128 | 0.6305 | 0.6053 | 36 | 11 |

## Reading

The call is **PASS**.

At the n=12 freeze, L=2 R_Fourier=0.6197 (B=2 0.4261, Fourier 0.6876), under 0.85, with 5/5 seeds at or under that line. Medε(B=2) / Medε(#51 B=4) is 0.998 / 0.995 / 0.991 at n=12 / 15 / 18, and ΔR against 0.6209 / 0.6016 / 0.5880 is -0.0012 / -0.0037 / -0.0052. The ratio stays under 1 and does not grow with n, so the two coefficients do not soften on scale. Median Λ is 0.7477 / 0.7489 / 0.7487, all at or under 0.80. Two quadratic SNAP phases carry the locked NEW_SAFE shallow lead through n=18.

L=4 is not part of the gate. Its R_Fourier is 0.6174 / 0.6252 / 0.6067, and Medε stays near 0.695 / 0.721 / 0.694 times the PR #51 L=4 shell. The n=12 row matches PR #57. At this depth the quadratic shell is ahead of the published four-phase shell, and that lead holds through n=18.

## Non-claims

This does not retune λ or μ, does not add XX, and does not train L≥8. It does not replace the PR #51 B=4 sentence with a claim about every instance. The 1.05× bar is on Medε, a median of medians; the 0.02 bar is on R_Fourier. Either one, at each of n=12, n=15, and n=18, is enough for that size. n=21 was not loaded.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n=9-holdout checkpoint. Ground-state Λ, the product gap, and the Z-peak are in `diag_n{n}_{role}.npz`.
