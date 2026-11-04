# Power-law long-range Ising — breadth smoke

**Verdict: needs_edit.** No cell clears its bar, and the L=4 train sweep is not a KILL.

This is a 3-seed smoke at paper depth L∈{2, 4}. It is not the 5-seed shallow seal used for the unsat SAFE point.

## Hamiltonian

Second non-CSP family. The diagonal is a spin glass, not an unsat count.

$$
H = \sum_{i<j} \frac{J_{ij}}{|i-j|^{1.5}} Z_i Z_j - \lambda \sum_i X_i + \mu \sum_c \sum_{i<j\in c} X_i X_j
$$

\(J_{ij}\in\{\pm 1\}\) is drawn once per instance on the open chain. Qubit \(i\) sits at site \(i\), qubit 0 is the MSB, cavity \(c\) owns \([3c, 3c+3)\). Bonds that cross a cavity boundary stay in the sum. Cross-cavity share of \(\sum|K_{ij}|\) is n6=0.358, n9=0.456, n12=0.503.

The circuit phases \(U_C(\gamma)=\exp(-i\gamma H_{ZZ}/n)\). Dividing by \(n\) only rescales \(\gamma\); \(\langle H\rangle\) and \(E_0\) use the formula above. \(H_{ZZ}\) is pure Walsh degree 2, so Walsh SNAP trains \(\gamma_2\) only. The Hamming shell \(\varphi_k\) is free. Linear \(\varphi_k=-\beta k\) is the Fourier X mixer.

Readout is the Z histogram for every ZZ and the X histogram for \(\langle X\rangle\) and intra-cavity \(\langle XX\rangle\). No Y.

## Gate

Med \(\varepsilon\) is the median over seeds of each seed's median-over-instances excess \(\langle H\rangle-E_0\). \(R = \mathrm{Med}\varepsilon_{\mathrm{SNAP}} / \mathrm{Med}\varepsilon_{\mathrm{baseline}}\).

- Quantumness on the frozen n=12 pool: median bitstring peak < 0.5 and median product gap \((E_{\mathrm{prod}}-E_0)/|E_0|\) ≥ 0.03. Otherwise that (λ, μ) is needs_edit.
- Driver PASS (μ=−0.5): some baseline in {Fourier, HVA} has n12 R ≤ 0.90, n6 and n9 R ≤ 0.98, and n12 R ≤ max(n6, n9) + 0.10 (transfer held). Either baseline is enough.
- Pure-family PASS (μ=0): the same rule against Fourier at n12 R ≤ 0.98.
- KILL: at L=4, every (λ, μ) has Fourier R > 0.98 at both n6 and n9. A driver or pure PASS outranks KILL.
- n6 and n9 ratios are holdout. n12 is the freeze. n=21 was not loaded.

**5-seed seal:** not warranted. No quantum cell is within 0.03 of its n12 bar with the lead still holding at larger n. A 5-seed seal would not change the smoke call.

## Setup

| knob | value |
|---|---|
| α | 1.5 |
| λ, μ | [0.5, 1.0] × [0.0, -0.5] |
| L | [2, 4] |
| seeds | [0, 1, 2] |
| pools | train B=64 (n6 and n9, indices 0..), holdout B=32, freeze n12 B=64 |
| optimizer | L-BFGS 40×8, lr 0.25, batch 32 (half n6 / half n9), grad clip 5 |
| init | β=β^X=0.01, β^XX=0, γ=γ_2=−0.01, φ_k=−β k |
| checkpoint | lowest n9-holdout Med ε |
| instance seed | 1500003017 + n * 100003 + index |

## Quantumness

Median bitstring peak and median product gap. The pass bit is the n=12 row.

| λ | μ | n6 hold | n9 hold | n12 freeze |
|---:|---:|---|---|---|
| 0.50 | -0.50 | 0.146 / 0.125 (pass, peak<0.5 on 100%, gap≥0.03 on 100%) | 0.072 / 0.101 (pass, peak<0.5 on 100%, gap≥0.03 on 100%) | 0.028 / 0.104 (pass, peak<0.5 on 100%, gap≥0.03 on 100%) |
| 0.50 | 0.00 | 0.383 / 0.006 (fail, peak<0.5 on 100%, gap≥0.03 on 12%) | 0.313 / 0.004 (fail, peak<0.5 on 100%, gap≥0.03 on 3%) | 0.251 / 0.006 (fail, peak<0.5 on 100%, gap≥0.03 on 3%) |
| 1.00 | -0.50 | 0.078 / 0.093 (pass, peak<0.5 on 100%, gap≥0.03 on 100%) | 0.026 / 0.093 (pass, peak<0.5 on 100%, gap≥0.03 on 100%) | 0.007 / 0.094 (pass, peak<0.5 on 100%, gap≥0.03 on 100%) |
| 1.00 | 0.00 | 0.165 / 0.047 (pass, peak<0.5 on 100%, gap≥0.03 on 78%) | 0.092 / 0.036 (pass, peak<0.5 on 100%, gap≥0.03 on 66%) | 0.042 / 0.037 (pass, peak<0.5 on 100%, gap≥0.03 on 70%) |

## Ratios

R < 1 means Walsh SNAP has the lower excess energy.

### L=2

| λ | μ | n | Medε Fourier | Medε HVA | Medε SNAP | R Fourier | R HVA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | -0.50 | 6 hold | 0.1567 | 0.1484 | 0.1484 | 0.9468 | 0.9999 |
| 0.50 | -0.50 | 9 hold | 0.3603 | 0.3429 | 0.3480 | 0.9659 | 1.0148 |
| 0.50 | -0.50 | 12 freeze | 0.8100 | 0.7702 | 0.7901 | 0.9754 | 1.0258 |
| 0.50 | 0.00 | 6 hold | 0.6369 | — | 0.5690 | 0.8934 | — |
| 0.50 | 0.00 | 9 hold | 1.3080 | — | 1.1996 | 0.9171 | — |
| 0.50 | 0.00 | 12 freeze | 2.4030 | — | 2.3106 | 0.9615 | — |
| 1.00 | -0.50 | 6 hold | 0.0597 | 0.0542 | 0.0583 | 0.9780 | 1.0765 |
| 1.00 | -0.50 | 9 hold | 0.1373 | 0.1306 | 0.1348 | 0.9815 | 1.0325 |
| 1.00 | -0.50 | 12 freeze | 0.4249 | 0.4227 | 0.4304 | 1.0131 | 1.0183 |
| 1.00 | 0.00 | 6 hold | 0.1642 | — | 0.1642 | 0.9999 | — |
| 1.00 | 0.00 | 9 hold | 0.4360 | — | 0.4228 | 0.9699 | — |
| 1.00 | 0.00 | 12 freeze | 0.9492 | — | 0.9407 | 0.9910 | — |

| λ | μ | cell | seeds n12 R Fourier | seeds n12 R HVA | shell RMS | mean \|β^XX\| |
|---:|---:|---|---|---|---:|---:|
| 0.50 | -0.50 | no baseline has n12 R≤0.90 with train R≤0.98 and transfer held | 0.985, 0.971, 0.975 | 1.026, 1.026, 1.026 | 0.2949 | 0.1480 |
| 0.50 | 0.00 | quantumness fails (median peak or product gap) | 0.971, 0.965, 0.962 | — | 0.0610 | — |
| 1.00 | -0.50 | no baseline has n12 R≤0.90 with train R≤0.98 and transfer held | 1.017, 1.093, 1.009 | 1.029, 1.004, 1.015 | 0.3233 | 0.0930 |
| 1.00 | 0.00 | pure family misses n12 R≤0.98 with train R≤0.98 and transfer held | 1.007, 0.978, 1.088 | — | 0.0627 | — |

### L=4

| λ | μ | n | Medε Fourier | Medε HVA | Medε SNAP | R Fourier | R HVA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | -0.50 | 6 hold | 0.1235 | 0.0691 | 0.0789 | 0.6391 | 1.1414 |
| 0.50 | -0.50 | 9 hold | 0.2689 | 0.2141 | 0.2087 | 0.7762 | 0.9751 |
| 0.50 | -0.50 | 12 freeze | 0.7198 | 0.6847 | 0.6746 | 0.9372 | 0.9852 |
| 0.50 | 0.00 | 6 hold | 0.2994 | — | 0.2314 | 0.7729 | — |
| 0.50 | 0.00 | 9 hold | 0.6518 | — | 0.5443 | 0.8350 | — |
| 0.50 | 0.00 | 12 freeze | 1.6355 | — | 1.6605 | 1.0153 | — |
| 1.00 | -0.50 | 6 hold | 0.0541 | 0.0410 | 0.0389 | 0.7191 | 0.9504 |
| 1.00 | -0.50 | 9 hold | 0.1266 | 0.1058 | 0.1049 | 0.8289 | 0.9920 |
| 1.00 | -0.50 | 12 freeze | 0.4118 | 0.3966 | 0.3974 | 0.9648 | 1.0019 |
| 1.00 | 0.00 | 6 hold | 0.0920 | — | 0.0900 | 0.9783 | — |
| 1.00 | 0.00 | 9 hold | 0.2551 | — | 0.2668 | 1.0461 | — |
| 1.00 | 0.00 | 12 freeze | 0.7844 | — | 0.8076 | 1.0296 | — |

| λ | μ | cell | seeds n12 R Fourier | seeds n12 R HVA | shell RMS | mean \|β^XX\| |
|---:|---:|---|---|---|---:|---:|
| 0.50 | -0.50 | no baseline has n12 R≤0.90 with train R≤0.98 and transfer held | 0.911, 0.938, 0.951 | 0.980, 0.976, 1.006 | 0.2795 | 0.1836 |
| 0.50 | 0.00 | quantumness fails (median peak or product gap) | 1.037, 1.015, 1.008 | — | 0.1023 | — |
| 1.00 | -0.50 | no baseline has n12 R≤0.90 with train R≤0.98 and transfer held | 0.972, 0.960, 1.028 | 1.006, 1.002, 0.997 | 0.2181 | 0.1205 |
| 1.00 | 0.00 | pure family misses n12 R≤0.98 with train R≤0.98 and transfer held | 1.036, 1.028, 1.014 | — | 0.1462 | — |

## L=16 light probe

Light probe at the best driver cell of the smoke (λ=0.50, μ=-0.50, reference L=4). Seed 0 only, 20 L-BFGS rounds, same pools. Not in the verdict.

| n | Medε Fourier | Medε HVA | Medε SNAP | R Fourier | R HVA |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.1195 | 0.0543 | 0.0516 | 0.4320 | 0.9504 |
| 9 | 0.2509 | 0.2038 | 0.1796 | 0.7156 | 0.8811 |
| 12 | 0.6908 | 0.6678 | 0.6376 | 0.9231 | 0.9548 |

The probe is one seed and a shorter L-BFGS budget. It is not part of the verdict.

## Reading

No cell clears its bar, and the L=4 train sweep is not a KILL. Best frozen n12 Fourier ratio is 0.9372 at λ=0.50 μ=-0.50 L=4 (SNAP Medε 0.6746, Fourier 0.7198). The same cell is 0.6391 at n6 and 0.7762 at n9. Best n12 HVA ratio is 0.9852 at λ=0.50 μ=-0.50 L=4 (HVA Medε 0.6847). Quantum n12 ground states (peak < 0.5 and product gap ≥ 0.03): λ=0.50 μ=-0.50, λ=1.00 μ=-0.50, λ=1.00 μ=0.00. Product-gap failures, so those cells cannot PASS: λ=0.50 μ=0.00. A spread-out bitstring peak is not enough when a product state already reaches the ground energy. H_ZZ is pure Walsh degree 2, so the SNAP cost has one live γ per layer. Any lead over Fourier has to come from the Hamming shell. On the μ=−0.5 cells the shell does leave the X mixer (median shell RMS off φ_k=a+bk is 0.2919; median mean |β^XX| is 0.1338), and SNAP still does not clear 0.90 against Fourier or against HVA. Ratio without quantumness: λ=0.50 μ=0 L=2 would be a pure-family ratio (0.9615) and fails the product gap. Every seed at that best Fourier cell is above 0.90 (min 0.911). Two more seeds do not reopen a 0.90 driver PASS.

Per-seed JSON is `L{L}_{method}_s{seed}_l{λ}_m{μ}.json` in this folder. Angles are the restored n9-holdout checkpoint. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.
