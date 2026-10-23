# Oct 23 Part 1: Two-Cavity Experimental Circuit (`n = 6` and `n = 8`, <= 5 Blocks Total)

*Last updated: 2026-10-23 21:55:29 (UTC+8). All SNAP gates strictly restricted to computational Fock levels `0..D-1` (`0` phase on leaked levels `f >= D`); all Fock leakage (`f >= D` or ancilla in `|e>`) counted as failure and verified at `N_Fock = 40` (`n=6`) and `N_Fock = 56` (`n=8`).*

## 1. Classical WalkSATlm Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)

| Problem Size `n` | Layout | WalkSATlm Tries (Med / B90) | WalkSATlm Lookups (Med / B90) | Random Guess `2^n / n_sol` (Med / B90) |
|---|---|---|---|---|
| **n = 6** | 2 cavities x 3 bits (`D = 8`) | **1.691 / 2.816** | **8.466 / 16.039** | 64.0 / 64.0 |
| **n = 8** | 2 cavities x 4 bits (`D = 16`) | **2.014 / 3.739** | **13.989 / 29.396** | 128.0 / 256.0 |

## 2. State Preparation Fidelity from Vacuum (`0..D-1` Computational SNAP Phases Only)

| Gate & Cavity Size | `k = 1` Block `F` (leak) | `k = 2` Blocks `F` (leak) | `k = 3` Blocks `F` (leak) |
|---|---|---|---|
| SNAP+D (`K=2, D=4`) | 90.77% (2.81%) | 99.82% (0.16%) | 100.00% (0.00%) |
| ECD ancilla (`K=2, D=4`) | 90.77% (2.81%) | 93.91% (2.15%) | 92.94% (2.21%) |
| ECD hybrid (`1t+2c, 2D=8`) | 90.77% (2.81%) | 95.11% (1.43%) | 98.90% (0.60%) |
| SNAP+D (`K=3, D=8`) | 88.60% (1.51%) | 99.78% (0.12%) | 100.00% (0.00%) |
| ECD ancilla (`K=3, D=8`) | 88.60% (1.51%) | 88.61% (1.52%) | 90.61% (1.65%) |
| ECD hybrid (`1t+3c, 2D=16`) | 88.60% (1.51%) | 92.88% (1.08%) | 97.16% (0.72%) |
| SNAP+D (`K=4, D=16`) | 77.96% (0.64%) | 98.85% (0.29%) | 100.00% (0.00%) |
| ECD ancilla (`K=4, D=16`) | 77.96% (0.64%) | 77.96% (0.64%) | 91.26% (4.30%) |
| ECD hybrid (`1t+4c, 2D=32`) | 77.96% (0.64%) | 94.27% (1.56%) | 96.73% (0.82%) |

## 3. Category A: Pure Shared-Parameter L-BFGS-B (`relabel = none`, Fixed Global Cost `exp(-i gamma_l (n/m) C(x))`)

*One shared parameter vector across all formulas; canonical variable ordering (`relabel = none`); exact unweighted global cost phase.*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | Val Med / B90 | Test Med / B90 (`1/p`) | `NF_chk` Med / B90 | Mean Leak | Beats WS Tries / Lookups |
|---|---|---|---|---|---|---|---|---|---|---|
| `n6_0p5x1_csd_none` | 6 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 6.15 / 16.45 | **7.10 / 13.12** | 7.10 / 13.12 | 5.36% | no / YES |
| `n6_0p5x1_ecd_none` | 6 | `0p5x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `none` | 8.52 / 48.70 | **8.23 / 34.50** | 8.23 / 34.50 | 2.13% | no / med only |
| `n6_0p5x1_sdc_none` | 6 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `none` | 6.17 / 16.24 | **7.07 / 13.13** | 7.07 / 13.13 | 5.46% | no / YES |
| `n6_1p2x2_csd_none` | 6 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 8.03 / 18.26 | **8.15 / 17.83** | 8.15 / 17.83 | 5.62% | no / med only |
| `n6_1p4x1_ecd_none` | 6 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `none` | 6.82 / 37.80 | **6.78 / 30.57** | 6.78 / 30.57 | 1.82% | no / med only |
| `n6_1p4x1_ecdt_none` | 6 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 2000 | `none` | 12.24 / 27.97 | **12.73 / 24.76** | 12.73 / 24.76 | 4.77% | no / no |
| `n6_2p3x1_csd_none` | 6 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 7.32 / 17.41 | **7.66 / 14.11** | 7.66 / 14.11 | 7.92% | no / YES |
| `n6_ntrain10000_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 6.23 / 16.26 | **7.09 / 13.12** | 7.09 / 13.12 | 6.08% | no / YES |
| `n6_ntrain2000_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 6.08 / 17.11 | **7.14 / 12.83** | 7.14 / 12.83 | 4.70% | no / YES |
| `n6_ntrain5000_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `none` | 6.15 / 15.99 | **7.00 / 13.08** | 7.00 / 13.08 | 4.70% | no / YES |
| `n6_ntrain500_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `none` | 6.06 / 16.09 | **7.23 / 13.00** | 7.23 / 13.00 | 6.46% | no / YES |
| `smoke_n6` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `none` | 8.74 / 23.11 | **10.24 / 21.51** | 10.24 / 21.51 | 2.29% | no / no |
| `smoke_n6_ecd` | 6 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 500 | `none` | 8.30 / 49.97 | **7.92 / 35.62** | 7.92 / 35.62 | 1.61% | no / med only |
| `smoke_n6_ecdt` | 6 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 500 | `none` | 15.07 / 37.18 | **15.44 / 34.33** | 15.44 / 34.33 | 8.57% | no / no |
| `smoke_n6_none_sdir` | 6 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 7.33 / 17.19 | **7.75 / 14.14** | 7.75 / 14.14 | 8.35% | no / YES |
| `n8_0p5x1_csd_none` | 8 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 36.40 / 77.95 | **40.39 / 78.65** | 40.39 / 78.65 | 5.51% | no / no |
| `n8_0p5x1_sdc_none` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 2000 | `none` | 26.30 / 62.42 | **25.55 / 53.58** | 25.55 / 53.58 | 6.06% | no / no |
| `n8_1p2x2_csd_none` | 8 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 47.61 / 147.56 | **43.44 / 157.36** | 43.44 / 157.36 | 3.26% | no / no |
| `n8_2p3x1_csd_none` | 8 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 19.72 / 46.26 | **19.46 / 47.95** | 19.46 / 47.95 | 3.18% | no / no |
| `n8_ntrain10000_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 20.67 / 51.24 | **20.55 / 49.37** | 20.55 / 49.37 | 4.28% | no / no |
| `n8_ntrain2000_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 20.42 / 47.37 | **20.24 / 54.07** | 20.24 / 54.07 | 4.15% | no / no |
| `n8_ntrain5000_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `none` | 20.34 / 46.60 | **20.14 / 52.49** | 20.14 / 52.49 | 4.00% | no / no |
| `n8_ntrain500_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `none` | 20.21 / 50.35 | **19.79 / 50.47** | 19.79 / 50.47 | 4.18% | no / no |

## 4. Category B: Shared-Parameter L-BFGS-B + Compile-Time Variable Relabeling (Fixed Global Cost)

*One shared parameter vector across all formulas + compile-time factor-graph variable ordering/polarity alignment (`bias`, `biaspol`, `bppol`, `bpdec`, `mpw4`, `rr4`); exact unweighted global cost phase.*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | Val Med / B90 | Test Med / B90 (`1/p`) | `NF_chk` Med / B90 | Mean Leak | Beats WS Tries / Lookups |
|---|---|---|---|---|---|---|---|---|---|---|
| `n6_0p5x1_csd_bpdec` | 6 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 1.92 / 16.61 | **1.91 / 7.32** | 1.91 / 7.32 | 2.49% | no / YES |
| `n6_0p5x1_sdc_bpdec` | 6 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 2.01 / 16.17 | **2.01 / 7.57** | 2.01 / 7.57 | 2.86% | no / YES |
| `n6_0p5x1_sdc_mpw4` | 6 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `mpw4` | 1.10 / 1.18 | **1.09 / 1.20** | 1.09 / 1.20 | 0.20% | YES / YES |
| `n6_1p2x2_csd_bpdec` | 6 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 1.83 / 42.01 | **1.80 / 17.11** | 1.80 / 17.11 | 1.31% | no / med only |
| `n6_1p4x1_bias` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bias` | 6.26 / 15.31 | **7.23 / 15.75** | 7.23 / 15.75 | 6.74% | no / YES |
| `n6_1p4x1_biaspol` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `biaspol` | 5.26 / 14.38 | **5.98 / 14.40** | 5.98 / 14.40 | 6.11% | no / YES |
| `n6_1p4x1_bppol` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bppol` | 2.16 / 10.69 | **2.19 / 9.01** | 2.19 / 9.01 | 3.32% | no / YES |
| `n6_1p4x1_csd_mpw4` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `mpw4` | 1.11 / 1.21 | **1.12 / 1.21** | 1.12 / 1.21 | 0.56% | YES / YES |
| `n6_1p4x1_ecd_bpdec` | 6 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `bpdec` | 1.61 / 266.93 | **1.64 / 125.37** | 1.64 / 125.30 | 6.49% | med only / med only |
| `n6_1p4x1_ecdt_bpdec` | 6 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 2000 | `bpdec` | 1.97 / 27.86 | **2.00 / 13.94** | 2.00 / 13.94 | 1.63% | no / YES |
| `n6_1p4x1_msdec_sdir` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.17 / 14.62 | **2.15 / 10.09** | 2.15 / 10.09 | 1.05% | no / YES |
| `n6_2p3x1_csd_bpdec` | 6 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 1.93 / 29.36 | **1.92 / 13.26** | 1.92 / 13.26 | 4.49% | no / YES |
| `n6_ntrain10000_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 2.07 / 15.48 | **2.08 / 7.46** | 2.08 / 7.46 | 2.84% | no / YES |
| `n6_ntrain2000_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.03 / 17.45 | **2.06 / 15.97** | 2.06 / 15.97 | 3.74% | no / YES |
| `n6_ntrain5000_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `bpdec` | 2.19 / 16.69 | **2.23 / 13.68** | 2.23 / 13.68 | 3.08% | no / YES |
| `n6_ntrain500_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bpdec` | 1.87 / 20.84 | **1.89 / 17.67** | 1.89 / 17.67 | 2.56% | no / med only |
| `smoke_n6_0p5x1` | 6 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bppol` | 1.97 / 26.72 | **1.95 / 30.71** | 1.95 / 30.71 | 2.41% | no / med only |
| `smoke_n6_0p5x1_rr4` | 6 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `rr4` | 4.32 / 7.87 | **4.23 / 7.27** (r0: 1.95/30.70, best4: 1.74/2.66) | 4.23 / 7.27 | 2.35% | no / YES |
| `smoke_n6_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bpdec` | 1.82 / 32.80 | **1.83 / 19.82** | 1.83 / 19.82 | 2.47% | no / med only |
| `smoke_n6_bpdec_rr4` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `rr4` | 3.29 / 7.44 | **3.57 / 7.47** (r0: 1.80/50.22, best4: 1.78/2.10) | 3.57 / 7.47 | 2.54% | no / YES |
| `smoke_n6_bpdec_sdir` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.03 / 14.23 | **2.05 / 22.23** | 2.05 / 22.23 | 1.40% | no / med only |
| `smoke_n6_bppol` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bppol` | 2.50 / 29.07 | **2.50 / 22.49** | 2.50 / 22.49 | 3.08% | no / med only |
| `smoke_n6_rr4` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `rr4` | 5.06 / 7.76 | **5.50 / 8.02** (r0: 2.04/35.80, best4: 1.96/2.28) | 5.50 / 8.02 | 4.75% | no / YES |
| `n8_0p5x1_csd_bpdec` | 8 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.43 / 471.42 | **2.40 / 393.16** | 2.40 / 393.16 | 2.38% | no / med only |
| `n8_0p5x1_sdc_bpdec` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.35 / 619.62 | **2.38 / 336.04** | 2.38 / 336.04 | 2.34% | no / med only |
| `n8_0p5x1_sdc_mpw4` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `mpw4` | 1.57 / 5.57 | **1.52 / 1.70** | 1.52 / 1.70 | 0.53% | YES / YES |
| `n8_0p5x1_sdc_msdec_sdir` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.50 / 182.02 | **2.51 / 178.55** | 2.51 / 178.55 | 1.74% | no / med only |
| `n8_1p2x2_csd_bpdec` | 8 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.24 / 922.05 | **2.27 / 428.81** | 2.27 / 428.81 | 2.26% | no / med only |
| `n8_1p4x1_bias` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bias` | 69.57 / 199.22 | **58.67 / 184.84** | 58.67 / 184.84 | 5.36% | no / no |
| `n8_1p4x1_biaspol` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `biaspol` | 18.05 / 146.56 | **21.46 / 232.15** | 21.46 / 232.15 | 4.17% | no / no |
| `n8_1p4x1_csd_mpw4` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `mpw4` | 1.57 / 3.00 | **1.56 / 1.72** | 1.56 / 1.72 | 0.78% | YES / YES |
| `n8_1p4x1_msdec_sdir` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.67 / 181.94 | **2.69 / 128.71** | 2.69 / 128.71 | 1.96% | no / med only |
| `n8_2p3x1_csd_bpdec` | 8 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.52 / 740.22 | **2.48 / 847.67** | 2.48 / 847.67 | 4.68% | no / med only |
| `n8_ntrain10000_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 2.38 / 291.58 | **2.40 / 388.81** | 2.40 / 388.81 | 2.31% | no / med only |
| `n8_ntrain2000_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 2.46 / 435.25 | **2.53 / 264.35** | 2.53 / 264.35 | 2.97% | no / med only |
| `n8_ntrain5000_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `bpdec` | 2.31 / 347.75 | **2.33 / 381.05** | 2.33 / 381.05 | 2.12% | no / med only |
| `n8_ntrain500_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bpdec` | 2.66 / 918.70 | **2.66 / 459.05** | 2.66 / 459.05 | 5.26% | no / med only |

## 5. Category C: Pure GNN from Raw Factor-Graph Degrees (`--pure_gnn --relabel none`, Zero Precomputed BP/Decimation)

*GNN takes ONLY raw factor-graph degree features `fv[:, :, 0:3] = [1, pos_deg, neg_deg]` (`fv[:, :, 3:7] = 0`) with canonical variable ordering (`relabel = none`); compares fixed global cost (`fixed_cost`) vs. clause-weighted global cost (`clause_w`).*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | Val Med / B90 | Test Med / B90 (`1/p`) | `NF_chk` Med / B90 | Mean Leak | Beats WS Tries / Lookups |
|---|---|---|---|---|---|---|---|---|---|---|
| `gnn5_n6_pure_nocw` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 6.14 / 12.51 | **6.14 / 11.21** | 6.14 / 11.21 | 6.15% | no / YES |

## 6. Category D: BP/Decimation-Assisted GNN (`bpdec`, `mpw4`, `bpfeat_canon`)

*Combines factor-graph message-passing / decimation features with GNN per-formula SNAP/displacement modulation (both `fixed_cost` and `clause_w`).*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | Val Med / B90 | Test Med / B90 (`1/p`) | `NF_chk` Med / B90 | Mean Leak | Beats WS Tries / Lookups |
|---|---|---|---|---|---|---|---|---|---|---|
| `gnn5_n6_0p5x1_sdc_bpdec_pcap` | 6 | `0p5x1` (`sdc`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.06 / 4.38 | **1.06 / 3.71** | 1.06 / 3.71 | 0.59% | med only / YES |
| `gnn5_n6_0p5x1_sdc_prod` | 6 | `0p5x1` (`sdc`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.04 / 4.14 | **1.04 / 2.72** (RR4: 2.00/3.63, best4: 1.04/1.21) | 1.04 / 2.72 | 0.42% | YES / YES |
| `gnn5_n6_1p4x1_bpdec_pcap` | 6 | `1p4x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.13 / 4.27 | **1.10 / 3.65** (RR4: 2.75/5.34, best4: 1.10/2.14) | 1.10 / 3.65 | 3.11% | med only / YES |
| `gnn5_n6_1p4x1_bpfeat_canonical` | 6 | `1p4x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpfeat_canon` | 1.37 / 7.79 | **1.38 / 5.90** | 1.38 / 5.90 | 4.80% | med only / YES |
| `gnn5_n6_2p3x1_bpdec_pcap` | 6 | `2p3x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.20 / 5.32 | **1.20 / 3.97** | 1.20 / 3.97 | 4.57% | med only / YES |
| `gnn5_n6_2p3x1_bpfeat_canonical` | 6 | `2p3x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpfeat_canon` | 2.28 / 12.40 | **2.00 / 11.09** | 2.00 / 11.09 | 8.91% | no / YES |
| `smoke_n6_gnn` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 1.26 / 13.88 | **1.26 / 11.30** | 1.26 / 11.30 | 2.73% | med only / YES |
| `smoke_n6_gnn_rr4` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `rr4` | 1.26 / 13.88 | **1.26 / 11.30** (RR4: 2.59/10.48, best4: 1.25/8.28) | 1.26 / 11.30 | 2.73% | med only / YES |
| `smoke_n6_none_gnn` | 6 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 5.06 / 15.71 | **5.78 / 12.43** | 5.78 / 12.43 | 10.52% | no / YES |
| `gnn5_n8_0p5x1_sdc_bpdec_prod` | 8 | `0p5x1` (`sdc`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.87 / 44.26 | **1.74 / 43.90** (RR4: 4.54/27.98, best4: 1.73/15.71) | 1.74 / 43.90 | 1.58% | med only / med only |
| `gnn5_n8_1p4x1_bpdec_pcap` | 8 | `1p4x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.58 / 55.88 | **1.59 / 44.81** | 1.59 / 44.81 | 4.64% | med only / med only |
| `gnn5_n8_2p3x1_bpdec_pcap` | 8 | `2p3x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpdec` | 1.61 / 70.00 | **1.60 / 64.17** | 1.60 / 63.91 | 7.79% | med only / med only |
| `gnn5_n8_2p3x1_bpfeat_canonical` | 8 | `2p3x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpfeat_canon` | 2.01 / 33.28 | **2.12 / 48.02** | 2.12 / 48.02 | 4.68% | no / med only |
| `gnn5_n8_2p3x1_bpfeat_prod` | 8 | `2p3x1` (`csd`) | `sd` (`clause_w`) | 10000 | `bpfeat_canon` | 4.80 / 33.66 | **5.64 / 46.16** | 5.64 / 46.16 | 3.74% | no / med only |
