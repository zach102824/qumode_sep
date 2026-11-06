# Power-law TFIM fidelity transfer — λ ∈ {0.75, 1.00}

**Verdicts:** λ=0.75 **needs_edit** (L\*=12); λ=1.00 **PASS** (L\*=16).

Fidelity to the exact ground state, not an energy race against QAOA. F = |⟨ψ(θ)|GS⟩|². A depth clears the train bar when the median-over-seeds of the median-over-instances fidelity is at least 0.95 on the n=6 train split, the n=9 train split, and both holdouts. L\* is the smallest such depth. PASS requires that L\* and a frozen n=12 fidelity ≳ 0.70 whose drop from min(train n6, train n9) is ≲ 0.25. needs_edit means train hit ~95% and the holdout or the freeze did not. KILL means no depth through L=16 put both training sizes at ~95%.

## Hamiltonian

$$
H = \sum_{i<j} \frac{J_{ij}}{|i-j|^{1.5}} Z_i Z_j - \lambda \sum_i X_i
$$

μ = 0, α = 1.5, \(J_{ij}\in\{\pm 1\}\) on the open chain. Qubit 0 is the MSB. Cavity \(c\) owns \([3c, 3c+3)\). Bonds that cross a cavity stay in the sum. Circuit phases use \(C_n = H_{ZZ}/n\); \(\langle H\rangle\) and \(E_0\) use the formula above. \(H_{ZZ}\) is pure Walsh degree 2.

Median cross-cavity share of Σ|K_ij| on the pools used here: n6=0.358, n9=0.456, n12=0.503, n15=0.532.

## Phase diagnostic

Bitstring peak is max_z |⟨z|GS⟩|². Product gap is (E_prod − E_0)/|E_0| with the same product-state search as the breadth smoke. ⟨X⟩ is the ground-state mean of Σ_i ⟨X_i⟩ / n. The spectral gap is (E_1 − E_0)/|E_0|. Holdout indices 64..95 at λ=1 reproduce the PR #42 breadth cache. The quantumness bit is that smoke's rule, median peak < 0.5 and median product gap ≥ 0.03. It is not the fidelity verdict.

| λ | n | split | B | peak | product gap | ⟨X⟩ | (E1−E0)/|E0| | E0 | quantumness |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.75 | 6 | train | 64 | 0.2397 | 0.0282 | 0.5960 | 0.0256 | -6.492 | fail |
| 0.75 | 6 | hold | 32 | 0.2609 | 0.0234 | 0.5731 | 0.0204 | -6.540 | fail |
| 0.75 | 6 | all | 96 | 0.2520 | 0.0261 | 0.5882 | 0.0248 | -6.500 | fail |
| 0.75 | 9 | train | 64 | 0.1656 | 0.0213 | 0.5450 | 0.0069 | -10.051 | fail |
| 0.75 | 9 | hold | 32 | 0.1814 | 0.0167 | 0.5192 | 0.0040 | -10.301 | fail |
| 0.75 | 9 | all | 96 | 0.1677 | 0.0212 | 0.5342 | 0.0058 | -10.126 | fail |
| 1.00 | 6 | train | 64 | 0.1515 | 0.0557 | 0.7446 | 0.0577 | -7.508 | pass |
| 1.00 | 6 | hold | 32 | 0.1646 | 0.0474 | 0.7339 | 0.0526 | -7.531 | pass |
| 1.00 | 6 | all | 96 | 0.1590 | 0.0528 | 0.7428 | 0.0562 | -7.509 | pass |
| 1.00 | 9 | train | 64 | 0.0829 | 0.0424 | 0.7113 | 0.0255 | -11.498 | pass |
| 1.00 | 9 | hold | 32 | 0.0917 | 0.0362 | 0.6802 | 0.0177 | -11.626 | pass |
| 1.00 | 9 | all | 96 | 0.0906 | 0.0383 | 0.7006 | 0.0221 | -11.527 | pass |

## Ansatz and budget

Walsh SNAP (b2): each layer has γ_2 on Walsh degree 2 of C_n and two shell coefficients. φ_k = c₁ u_k + c₂ v_k, where {u, v} is the orthonormal basis of span{k, k²} on popcount k=0..3 used by the B=2 ablation. Paper init φ_k = −β k sits in that span, so the init matches Fourier. The full four-phase Hamming shell was not needed for the parameterization. Fourier is the reference column on the same C_n, same batches, same L-BFGS budget. It does not decide the verdict.

| knob | value |
|---|---|
| λ | [0.75, 1.0] |
| μ, α | 0, 1.5 |
| L | [1, 2, 4, 8, 12, 16] |
| seeds | [0, 1, 2] |
| pools | train B=64 indices 0..63, hold B=32 indices 64..95, joint n=6 and n=9 |
| freeze | n=12 B=64 indices 0..63, n=15 B=64 indices 0.., no retrain |
| optimizer | L-BFGS 40×8, lr 0.25, batch 32 (half n6 / half n9), grad clip 5 |
| loss | mean of per-size mean(1 − F) |
| checkpoint | highest min of the four split median fidelities |
| early stop | three rounds in a row with that min ≥ 0.99 |
| instance seed | 1500003017 + n · 100003 + index |

## Ladder

Each entry is the median over the three seeds. The four fidelities are train n6, train n9, hold n6, hold n9. score is their minimum.

### λ = 0.75

| L | b2 train6 | b2 train9 | b2 hold6 | b2 hold9 | b2 score | Fourier train6 | Fourier train9 | Fourier hold6 | Fourier hold9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7721 | 0.5590 | 0.7493 | 0.5142 | 0.5142 | 0.7692 | 0.5537 | 0.7479 | 0.5077 |
| 2 | 0.8851 | 0.7251 | 0.8792 | 0.6745 | 0.6745 | 0.8776 | 0.7069 | 0.8641 | 0.6603 |
| 4 | 0.9305 | 0.8582 | 0.9203 | 0.8439 | 0.8439 | 0.9310 | 0.8398 | 0.9212 | 0.8213 |
| 8 | 0.9494 | 0.9363 | 0.9406 | 0.9356 | 0.9356 | 0.9553 | 0.9313 | 0.9511 | 0.9225 |
| 12 | 0.9593 | 0.9576 | 0.9578 | 0.9581 | 0.9576 | 0.9661 | 0.9376 | 0.9600 | 0.9331 |
| 16 | 0.9646 | 0.9630 | 0.9655 | 0.9622 | 0.9622 | 0.9655 | 0.9376 | 0.9605 | 0.9310 |

Med ε = ⟨H⟩ − E_0 at the fidelity checkpoint (b2 / fourier), holdout.

| L | n6 b2 | n6 Fourier | n9 b2 | n9 Fourier |
|---:|---:|---:|---:|---:|
| 1 | 0.9287 | 0.9197 | 1.2343 | 1.2181 |
| 2 | 0.4987 | 0.4892 | 0.7068 | 0.7245 |
| 4 | 0.2766 | 0.3001 | 0.3417 | 0.3683 |
| 8 | 0.1409 | 0.1145 | 0.1260 | 0.1388 |
| 12 | 0.1107 | 0.0895 | 0.0831 | 0.1267 |
| 16 | 0.0900 | 0.0914 | 0.0755 | 0.1210 |

L\* = 12. Best score is L=16 (0.9622). Smallest L with both train fidelities ≥ 0.95: 12. Frozen depth L=12. Fourier does not clear the same four-way 0.95 bar at any L≤16.

Per seed at the frozen depth, b2 then Fourier (train6, train9, hold6, hold9):

- b2 seed 0: 0.9603, 0.9600, 0.9578, 0.9596 (best round 36)
- b2 seed 1: 0.9556, 0.9563, 0.9533, 0.9567 (best round 37)
- b2 seed 2: 0.9593, 0.9576, 0.9585, 0.9581 (best round 38)
- Fourier seed 0: 0.9651, 0.9376, 0.9590, 0.9331 (best round 36)
- Fourier seed 1: 0.9661, 0.9466, 0.9600, 0.9406 (best round 37)
- Fourier seed 2: 0.9672, 0.9359, 0.9619, 0.9288 (best round 36)

### λ = 1.00

| L | b2 train6 | b2 train9 | b2 hold6 | b2 hold9 | b2 score | Fourier train6 | Fourier train9 | Fourier hold6 | Fourier hold9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8899 | 0.7917 | 0.8753 | 0.7388 | 0.7388 | 0.8896 | 0.7880 | 0.8732 | 0.7345 |
| 2 | 0.9443 | 0.8846 | 0.9419 | 0.8492 | 0.8492 | 0.9448 | 0.8813 | 0.9421 | 0.8394 |
| 4 | 0.9598 | 0.9373 | 0.9514 | 0.9192 | 0.9192 | 0.9637 | 0.9327 | 0.9558 | 0.9167 |
| 8 | 0.9593 | 0.9594 | 0.9478 | 0.9492 | 0.9478 | 0.9652 | 0.9572 | 0.9558 | 0.9515 |
| 12 | 0.9694 | 0.9593 | 0.9646 | 0.9460 | 0.9460 | 0.9716 | 0.9547 | 0.9677 | 0.9471 |
| 16 | 0.9629 | 0.9666 | 0.9597 | 0.9576 | 0.9576 | 0.9719 | 0.9560 | 0.9665 | 0.9475 |

Med ε = ⟨H⟩ − E_0 at the fidelity checkpoint (b2 / fourier), holdout.

| L | n6 b2 | n6 Fourier | n9 b2 | n9 Fourier |
|---:|---:|---:|---:|---:|
| 1 | 0.5778 | 0.5697 | 0.6987 | 0.7001 |
| 2 | 0.2838 | 0.2920 | 0.3732 | 0.3755 |
| 4 | 0.1613 | 0.1546 | 0.1993 | 0.2102 |
| 8 | 0.1621 | 0.1164 | 0.1240 | 0.1272 |
| 12 | 0.1069 | 0.0941 | 0.1256 | 0.1351 |
| 16 | 0.1084 | 0.1000 | 0.0995 | 0.1332 |

L\* = 16. Best score is L=16 (0.9576). Smallest L with both train fidelities ≥ 0.95: 8. Frozen depth L=16. Fourier first clears the same four-way 0.95 bar at L=8.

Per seed at the frozen depth, b2 then Fourier (train6, train9, hold6, hold9):

- b2 seed 0: 0.9612, 0.9687, 0.9595, 0.9576 (best round 36)
- b2 seed 1: 0.9629, 0.9656, 0.9597, 0.9598 (best round 39)
- b2 seed 2: 0.9653, 0.9666, 0.9630, 0.9568 (best round 35)
- Fourier seed 0: 0.9728, 0.9533, 0.9666, 0.9452 (best round 32)
- Fourier seed 1: 0.9663, 0.9591, 0.9635, 0.9535 (best round 39)
- Fourier seed 2: 0.9719, 0.9560, 0.9665, 0.9475 (best round 39)

## Freeze

Angles are the fidelity checkpoint from n∈{6, 9}. n=12 and n=15 are new draws from the same seed formula. No retrain. Drop = min(train n6, train n9) − n12 F, using the median-over-seeds numbers at the frozen depth.

| λ | verdict | L | n12 F b2 | drop | n15 F b2 | n12 F Fourier | n15 F Fourier | n12 Medε b2 | n12 E0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | needs_edit | 12 | 0.6971 | 0.2605 | 0.3863 | 0.6500 | 0.3472 | 0.6499 | -13.575 |
| 1.00 | PASS | 16 | 0.7950 | 0.1679 | 0.5735 | 0.7821 | 0.5560 | 0.5543 | -15.441 |

**λ=0.75: needs_edit.** train/hold cleared 0.95 at L*=12 but the freeze missed the n12 bar (F=0.6971, drop=0.2605)

n12 Medε = 0.6499 against median E0 = -13.575; n15 Medε = 1.6444 against median E0 = -17.308. Across instances, the seed-median n12 fidelity has lower quartile 0.610, minimum 0.488, and fraction ≥ 0.70 equal to 0.47. n15 lower quartile is 0.288.

Side measurement at L=16, not used for the verdict. That depth also clears train and hold. Frozen b2 n12 F=0.7132, drop=0.2498, n15 F=0.3968 (lower quartile 0.629, fraction ≥ 0.70 is 0.56). Fourier at the same angles-depth has n12 F=0.6570 and n15 F=0.3537.

- b2 seed 0: n12 F=0.6980 (ε=0.6499), n15 F=0.3863 (ε=1.6444)
- b2 seed 1: n12 F=0.6930 (ε=0.6558), n15 F=0.3850 (ε=1.6446)
- b2 seed 2: n12 F=0.6971 (ε=0.6487), n15 F=0.3877 (ε=1.6416)

**λ=1.00: PASS.** L*=16 clears train and hold F≥0.95. Frozen n12 F=0.7950, drop=0.1679.

n12 Medε = 0.5543 against median E0 = -15.441; n15 Medε = 1.3096 against median E0 = -19.531. Across instances, the seed-median n12 fidelity has lower quartile 0.721, minimum 0.533, and fraction ≥ 0.70 equal to 0.81. n15 lower quartile is 0.487.

- b2 seed 0: n12 F=0.8005 (ε=0.5497), n15 F=0.5808 (ε=1.2906)
- b2 seed 1: n12 F=0.7915 (ε=0.5868), n15 F=0.5686 (ε=1.3448)
- b2 seed 2: n12 F=0.7950 (ε=0.5543), n15 F=0.5735 (ε=1.3096)

## Repro

Helpers are `Nov_3rd/powerlaw_tfim.py` and `Nov_3rd/hva_safe.py` from the PR #42 family, unchanged. Runner: `Nov_3rd/scripts/run_powerlaw_fidelity_lam075_100.py`. Protocol `powerlaw_fidelity_lam075_100_v1`.
