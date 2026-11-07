# NEW_SAFE Walsh-shell coefficient ablation

**Verdict: PASS.** L=2 frozen n=12 full R_Fourier=0.6196, B=4 R_Fourier=0.6201, B=2 R_Fourier=0.6197, Medε(B=2)/Medε(full)=1.000, median Λ(GS)=0.748.

Point **NEW_SAFE**: λ=0.75, μ=0. The quote from PR #50 / #51 / #52 is L=2 R≈0.62 (#50 Medε SNAP/Fourier 0.4264/0.6876, R=0.6201, Λ=0.7477). This file retrains that cell and the coefficient sweep around it. n=21 was not loaded.

## What B is

The cost pulse is the locked Walsh grading in every SNAP arm: four weights on degrees 1..4 of Cn. On these 4-SAT packs the Walsh power above degree 4 is numerical noise, so four degree weights already are the full cost spectrum. B counts free SNAP phase coefficients of the shared cavity mixer.

| arm | phase parameters per layer | what it spans |
|:----|:---------------------------|:--------------|
| B=2 | 2 | φ_k = c1 k + c2 k² on popcount k=0..3, in orthonormal coordinates. Walsh degrees ≤2 of a symmetric phase, global phase fixed because φ_0 is not free. Paper init φ_k=−βk sits in this span exactly. |
| B=4 | 4 | free φ_0..φ_3. The locked Hamming shell (8 parameters per layer once the four γ weights are counted, at L=1). |
| full | 8 | one phase per Fock state of the 3-bit cavity. The Hamming shell is the popcount-tied subspace. |

Fourier is the matched baseline: two angles per layer, same batches, same L-BFGS budget. R_Fourier = Medε_arm / Medε_Fourier. Medε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. Λ is the median of ⟨(Σ_{i∈c} X_i / 3)²⟩ on the true ground state. It does not depend on B or L.

## Gate

- Train n∈{6,9}, indices [0:200). Checkpoint on the n=9 holdout [200:250).
- Freeze n=12, the first 128 pack instances. L=2 is the gate. L=4 is a soft check and does not change the verdict.
- **PASS** when, at L=2 on frozen n=12, full has R_Fourier ≤ 0.85, Medε(B=2) ≤ 1.3 × Medε(full), and B=2 itself has R_Fourier ≤ 0.85.
- **needs_edit** otherwise. B=2 is called soft when its R_Fourier sits in (0.85, 0.98). A soft B=2 is needs_edit even if the 1.3× ratio holds.

Reasons: full R_Fourier=0.6196≤0.85 and B=2 Medε is 1.000× full (≤1.3), with B=2 itself at R=0.6197

## Protocol

Packs: `beat_qaoa_12h/train_packs`. Seeds {0,1,2,3,4} at both depths. One batch stream per seed (NumPy Generator(seed) and torch.manual_seed(seed)), shared by every arm and both depths. L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64. New optimizer each round. Loss = mean ε/n on a 32+32 batch. Each seed restores the round with the best n=9-holdout Medε. Paper init β=0.01, γ=γ_w=−0.01, φ_k=−βk, and the full arm starts on that same shell. Ground states: dense eigh at n=6 and n=9, ARPACK at n=12. torch 2.15.0+cpu, numpy 2.5.4. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

B=2 basis reconstruction of the paper shell: max |φ_recon − φ_paper| = 3.469e-18.

Mean Walsh power of Cn's unsat table by degree, on the diagonalized packs. Degree 0 is a global phase of the cost and is dropped by the grading. Power above degree 4 is the part four γ weights cannot see.

| n | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 |
|--:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.7876 | 0.0571 | 0.0860 | 0.0557 | 0.0135 | 0.0000 | 0.0000 |
| 9 | 0.8551 | 0.0383 | 0.0587 | 0.0382 | 0.0096 | 0.0000 | 0.0000 |
| 12 | 0.8848 | 0.0320 | 0.0456 | 0.0301 | 0.0074 | 0.0000 | 0.0000 |

Degree ≥5 power, summed and then taking the worst of these three packs, is 0.000e+00. Four γ weights already see the whole cost spectrum on 4-SAT, which is why B does not truncate γ.

## Frozen n=12

| L | arm | Medε | Medε Fourier | R_Fourier | Medε / Medε(full) | median Λ (GS) | paired < Fourier | seeds R≤0.85 | call |
|--:|:----|-----:|-------------:|----------:|------------------:|--------------:|-----------------:|-------------:|:-----|
| 2 | B2 | 0.4261 | 0.6876 | 0.6197 | 1.000 | 0.7477 | 640/640 | 5/5 | win |
| 2 | B4 | 0.4264 | 0.6876 | 0.6201 | 1.001 | 0.7477 | 640/640 | 5/5 | win |
| 2 | full | 0.4261 | 0.6876 | 0.6196 | 1.000 | 0.7477 | 640/640 | 5/5 | win |
| 4 | B2 | 0.1958 | 0.3172 | 0.6174 | 0.965 | 0.7477 | 640/640 | 5/5 | win |
| 4 | B4 | 0.2755 | 0.3172 | 0.8687 | 1.358 | 0.7477 | 581/640 | 2/5 | soft |
| 4 | full | 0.2030 | 0.3172 | 0.6399 | 1.000 | 0.7477 | 640/640 | 5/5 | win |

B=2 against the full arm on paired frozen instances (L=2): 408/640 have a lower excess than full. Medε(B=2)/Medε(B=4) at L=2 is 0.999. B=4 is the locked shell, so that ratio is the compression claim against the published ansatz rather than against the 8-phase arm.

## Quote check (B=4 and Fourier, L=2, n=12)

| source | Medε shell | Medε Fourier | R | Λ |
|:-------|----------:|-------------:|--:|--:|
| PR #50 quote | 0.4264 | 0.6876 | 0.6201 | 0.7477 |
| this run, B=4 | 0.4264 | 0.6876 | 0.6201 | 0.7477 |

Deltas (this run minus #50): Medε B=4 -0.0000, Fourier +0.0000, R -0.0000, Λ +0.0000. R in this file always uses this run's Fourier, trained on the same batches as the shells.

## Holdout (checkpoint distribution, B=50)

Λ here is the ground state on the diagonalized pack (n=6 and n=9, B=250), not a property of the circuit.

| L | n | arm | Medε | R_Fourier | median Λ (GS) |
|--:|--:|:----|-----:|----------:|--------------:|
| 2 | 6 | B2 | 0.2170 | 0.6061 | 0.7612 |
| 2 | 6 | B4 | 0.2196 | 0.6135 | 0.7612 |
| 2 | 6 | full | 0.2196 | 0.6134 | 0.7612 |
| 2 | 9 | B2 | 0.2950 | 0.6248 | 0.7555 |
| 2 | 9 | B4 | 0.2978 | 0.6307 | 0.7555 |
| 2 | 9 | full | 0.2961 | 0.6271 | 0.7555 |
| 4 | 6 | B2 | 0.0919 | 0.6325 | 0.7612 |
| 4 | 6 | B4 | 0.1388 | 0.9555 | 0.7612 |
| 4 | 6 | full | 0.1002 | 0.6898 | 0.7612 |
| 4 | 9 | B2 | 0.1326 | 0.5885 | 0.7555 |
| 4 | 9 | B4 | 0.1977 | 0.8773 | 0.7555 |
| 4 | 9 | full | 0.1449 | 0.6430 | 0.7555 |

## Ground state

| n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | median Z-peak | frac peak≥0.5 | max residual |
|--:|--:|--------:|------:|------:|------------:|--------------:|--------------:|-------------:|
| 6 | 250 | 0.7612 | 0.5741 | 0.9417 | 0.262 | 0.2022 | 0.000 | 0.00e+00 |
| 9 | 250 | 0.7555 | 0.5223 | 0.9343 | 0.247 | 0.0916 | 0.000 | 0.00e+00 |
| 12 | 128 | 0.7477 | 0.5498 | 0.9117 | 0.230 | 0.0379 | 0.000 | 3.97e-09 |

## Seeds at frozen n=12

### L=2

| arm | seed | Medε | Medε Fourier | R | best round | wall s | shell std | phys rms |
|:----|-----:|-----:|-------------:|--:|----------:|-------:|----------:|---------:|
| B2 | 0 | 0.4261 | 0.6876 | 0.6197 | 32 | 4.1 | 0.00e+00 | 0.0000 |
| B2 | 1 | 0.4259 | 0.6878 | 0.6192 | 16 | 4.1 | 0.00e+00 | 0.0000 |
| B2 | 2 | 0.4260 | 0.6876 | 0.6195 | 17 | 4.1 | 0.00e+00 | 0.0000 |
| B2 | 3 | 0.4263 | 0.6876 | 0.6200 | 34 | 4.1 | 0.00e+00 | 0.0000 |
| B2 | 4 | 0.4262 | 0.6878 | 0.6198 | 17 | 4.1 | 0.00e+00 | 0.0000 |
| B4 | 0 | 0.4264 | 0.6876 | 0.6202 | 35 | 3.8 | 0.00e+00 | 0.0187 |
| B4 | 1 | 0.4243 | 0.6878 | 0.6169 | 32 | 4.0 | 2.78e-17 | 0.0021 |
| B4 | 2 | 0.4264 | 0.6876 | 0.6201 | 31 | 3.8 | 2.78e-17 | 0.0164 |
| B4 | 3 | 0.4261 | 0.6876 | 0.6196 | 34 | 3.8 | 0.00e+00 | 0.0062 |
| B4 | 4 | 0.4274 | 0.6878 | 0.6215 | 34 | 3.8 | 0.00e+00 | 0.0060 |
| full | 0 | 0.4232 | 0.6876 | 0.6155 | 35 | 4.0 | 1.46e-02 | 0.0060 |
| full | 1 | 0.4267 | 0.6878 | 0.6204 | 39 | 4.0 | 2.26e-02 | 0.0138 |
| full | 2 | 0.4261 | 0.6876 | 0.6196 | 35 | 4.0 | 1.35e-02 | 0.0064 |
| full | 3 | 0.4273 | 0.6876 | 0.6214 | 24 | 3.9 | 2.33e-02 | 0.0031 |
| full | 4 | 0.4260 | 0.6878 | 0.6194 | 34 | 4.0 | 1.65e-02 | 0.0111 |

### L=4

| arm | seed | Medε | Medε Fourier | R | best round | wall s | shell std | phys rms |
|:----|-----:|-----:|-------------:|--:|----------:|-------:|----------:|---------:|
| B2 | 0 | 0.1962 | 0.3184 | 0.6161 | 38 | 5.6 | 0.00e+00 | 0.0000 |
| B2 | 1 | 0.1958 | 0.3169 | 0.6181 | 39 | 5.6 | 1.11e-16 | 0.0000 |
| B2 | 2 | 0.1975 | 0.3172 | 0.6226 | 36 | 5.6 | 2.22e-16 | 0.0000 |
| B2 | 3 | 0.1949 | 0.3163 | 0.6162 | 35 | 5.5 | 2.22e-16 | 0.0000 |
| B2 | 4 | 0.1949 | 0.3181 | 0.6128 | 36 | 5.6 | 2.22e-16 | 0.0000 |
| B4 | 0 | 0.2755 | 0.3184 | 0.8653 | 38 | 5.2 | 2.78e-17 | 0.0368 |
| B4 | 1 | 0.3078 | 0.3169 | 0.9714 | 39 | 5.0 | 1.11e-16 | 0.0380 |
| B4 | 2 | 0.2534 | 0.3172 | 0.7989 | 38 | 5.2 | 1.39e-17 | 0.0412 |
| B4 | 3 | 0.2777 | 0.3163 | 0.8778 | 39 | 5.1 | 2.78e-17 | 0.0297 |
| B4 | 4 | 0.2241 | 0.3181 | 0.7044 | 39 | 5.4 | 5.55e-17 | 0.0397 |
| full | 0 | 0.2029 | 0.3184 | 0.6371 | 38 | 5.2 | 1.29e-02 | 0.0272 |
| full | 1 | 0.2046 | 0.3169 | 0.6459 | 39 | 5.2 | 1.64e-02 | 0.0278 |
| full | 2 | 0.2046 | 0.3172 | 0.6452 | 38 | 5.4 | 1.45e-02 | 0.0280 |
| full | 3 | 0.2004 | 0.3163 | 0.6335 | 39 | 5.3 | 1.48e-02 | 0.0266 |
| full | 4 | 0.2030 | 0.3181 | 0.6380 | 37 | 5.3 | 2.01e-02 | 0.0292 |

## Reading

L=2 full is a win against Fourier (R=0.6196, Medε 0.4261 vs 0.6876). L=2 B=4, the locked shell, is a win (R=0.6201, Medε 0.4264). L=2 B=2 is a win (R=0.6197, Medε 0.4261), 1.000× the full arm and 0.999× the locked shell. On paired instances B=2 is lower than full on 408/640, a tie rather than a sweep.

The three L=2 arms tie, and B=4 matches the PR #50 quote at four decimals on both Medε columns, on R, and on Λ. A constant added to every popcount phase is a global phase of the cavity unitary, so it is not a real coefficient. After removing that constant, B=4's phases sit 0.0099 RMS from span{1, k, k²} (full: 0.0081). The third shell direction is barely used, which is why two coefficients match four. full's largest within-popcount phase std, averaged over seeds, is 0.0181, so the eight Fock phases stay close to a shell.

L=4 is the soft check, not the gate. full R=0.6399 (win), B=4 R=0.8687 (soft), B=2 R=0.6174 (win), Medε(B=2)/Medε(full)=0.965, Medε(B=2)/Medε(B=4)=0.711. B=4 at this depth is the published soft result (Medε 0.2755 / Fourier 0.3172 / R 0.8687). B=2 and full are wins there. Two coefficients are better than the locked four-phase shell under the same budget, which is the opposite of a compression failure. The physical gap of B=4 off span{1, k, k²} grows to 0.0371 RMS (full 0.0278), so L=4 does use a bit of the extra shell direction, and that use does not improve Medε. L=4 stays outside the locked NEW_SAFE sentence.

Median Λ at frozen n=12 is 0.7477. The expectation from the seal is about 0.75, and the pass line used there was 0.80. n=6 / n=9 medians are 0.7612 / 0.7555.

Shell std is the max, over layers and popcount groups, of the std of the eight Fock phases inside one popcount. It is zero for B=2 and B=4 by construction. Phys rms is the layer-averaged RMS of the popcount phases outside span{1, k, k²}. That span is the B=2 family plus the unphysical global phase. A raw distance to span{k, k²} alone is not reported: it is dominated by the global phase.

## Non-claims

This does not retune μ, does not add XX, and does not train L≥8. It does not say that two coefficients beat four on every instance. The 1.3× bar is on Medε, which is a median of medians. n=15 and n=18 were not resealed. n=21 was not loaded.

