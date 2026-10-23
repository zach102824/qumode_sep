# Oct 23 Part 1: Two-Cavity Experimental Circuit (`n = 6` and `n = 8`, <= 5 Blocks Total)

*Last updated: 2026-10-23 22:28:56 (UTC+8). All SNAP gates strictly restricted to computational Fock levels `0..D-1` (`0` phase on leaked levels `f >= D`); all Fock leakage (`f >= D` or ancilla in `|e>`) counted as failure and verified at `N_Fock = 40` (`n=6`) and `N_Fock = 56` (`n=8`). Every row is verified to reproduce its saved `p_test` to `< 1e-6` under current code.*

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

*One shared parameter vector across all formulas; canonical variable ordering (`relabel = none`, `|0,0>` pre-solve = 1% random baseline); exact unweighted global cost phase. Genuine quantum phase interference (`NF_chk` Med / B90 vs `Zero-Oracle` `gamma_l=0`).*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | `|0,0>` Pre-Solve | Zero-Oracle (`gamma=0`) Med / B90 | Val Med / B90 | Test (`NF_chk`) Med / B90 | Unsolved Subset (`Succ[0,0]=0`) Med / B90 | Mean Leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `n6_0p5x1_csd_none` | 6 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 60.69 / 148.07 | 6.15 / 16.45 | **7.10 / 13.12** | 7.12 / 13.17 (99) | 5.36% |
| `n6_0p5x1_ecd_none` | 6 | `0p5x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `none` | 1% | 48.89 / 240.27 | 8.52 / 48.70 | **8.23 / 34.50** | 8.24 / 34.55 (99) | 2.13% |
| `n6_0p5x1_sdc_none` | 6 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 60.36 / 145.90 | 6.17 / 16.24 | **7.07 / 13.13** | 7.10 / 13.19 (99) | 5.46% |
| `n6_1p2x2_csd_none` | 6 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 68.43 / 86.48 | 8.03 / 18.26 | **8.15 / 17.83** | 8.26 / 17.86 (99) | 5.62% |
| `n6_1p4x1_ecd_none` | 6 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `none` | 1% | 44.14 / 243.93 | 6.82 / 37.80 | **6.78 / 30.57** | 6.98 / 30.70 (99) | 1.82% |
| `n6_1p4x1_ecdt_none` | 6 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 2000 | `none` | 1% | 57.23 / 93.01 | 12.24 / 27.97 | **12.73 / 24.76** | 12.74 / 24.82 (99) | 4.77% |
| `n6_2p3x1_csd_none` | 6 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 54.84 / 135.73 | 7.32 / 17.41 | **7.66 / 14.11** | 7.68 / 14.16 (99) | 7.92% |
| `n6_ntrain10000_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 61.58 / 136.69 | 6.23 / 16.26 | **7.09 / 13.12** | 7.12 / 13.21 (99) | 6.08% |
| `n6_ntrain2000_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 1% | 59.67 / 164.55 | 6.08 / 17.11 | **7.14 / 12.83** | 7.15 / 12.83 (99) | 4.70% |
| `n6_ntrain5000_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `none` | 1% | 58.64 / 152.45 | 6.15 / 15.99 | **7.00 / 13.08** | 7.00 / 13.14 (99) | 4.70% |
| `n6_ntrain500_none` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `none` | 1% | 60.20 / 130.12 | 6.06 / 16.09 | **7.23 / 13.00** | 7.29 / 13.01 (99) | 6.46% |
| `n8_0p5x1_csd_none` | 8 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 170.51 / 376.70 | 36.40 / 77.95 | **40.39 / 78.65** | 39.45 / 73.11 (99) | 5.51% |
| `n8_0p5x1_ecd_none` | 8 | `0p5x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `none` | 1% | 166.62 / 610.22 | 36.85 / 136.82 | **31.82 / 140.71** | 31.82 / 119.61 (99) | 21.64% |
| `n8_0p5x1_sdc_none` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 2000 | `none` | 1% | 146.34 / 286.01 | 26.30 / 62.42 | **25.55 / 53.58** | 25.24 / 52.90 (99) | 6.06% |
| `n8_1p2x2_csd_none` | 8 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 1% | 162.29 / 437.95 | 47.61 / 147.56 | **43.44 / 157.36** | 43.43 / 157.56 (99) | 3.26% |
| `n8_1p4x1_ecd_none` | 8 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `none` | 1% | 189.62 / 447.11 | 41.87 / 122.12 | **38.99 / 122.00** | 38.53 / 114.67 (99) | 20.26% |
| `n8_1p4x1_ecdt_none` | 8 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 2000 | `none` | 1% | 137.53 / 615.39 | 19.78 / 95.30 | **19.73 / 95.99** | 19.64 / 92.19 (99) | 5.33% |
| `n8_2p3x1_csd_none` | 8 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 1% | 138.08 / 316.24 | 19.72 / 46.26 | **19.46 / 47.95** | 19.18 / 47.02 (99) | 3.18% |
| `n8_ntrain10000_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 156.96 / 385.03 | 20.67 / 51.24 | **20.55 / 49.37** | 20.40 / 48.87 (99) | 4.28% |
| `n8_ntrain2000_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 1% | 170.27 / 398.66 | 20.42 / 47.37 | **20.24 / 54.07** | 20.22 / 50.90 (99) | 4.15% |
| `n8_ntrain5000_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `none` | 1% | 161.62 / 396.45 | 20.34 / 46.60 | **20.14 / 52.49** | 20.06 / 51.69 (99) | 4.00% |
| `n8_ntrain500_none` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `none` | 1% | 171.96 / 306.86 | 20.21 / 50.35 | **19.79 / 50.47** | 18.80 / 49.23 (99) | 4.18% |

## 4. Category C: Pure GNN from Raw Factor-Graph Degrees (`--pure_gnn --relabel none`, Zero Classical Pre-Solving)

*GNN takes ONLY raw factor-graph degree features `fv[:, :, 0:3] = [1, pos_deg, neg_deg]` (`fv[:, :, 3:7] = 0`) with canonical variable ordering (`relabel = none`, `|0,0>` pre-solve = 1% random baseline). Compares fixed global cost (`fixed_cost`) vs. clause-weighted global cost (`clause_w`).*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | `|0,0>` Pre-Solve | Zero-Oracle (`gamma=0`) Med / B90 | Val Med / B90 | Test (`NF_chk`) Med / B90 | Unsolved Subset (`Succ[0,0]=0`) Med / B90 | Mean Leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gnn5_n6_0p5x1_sdc_pure_cw` | 6 | `0p5x1` (`sdc`) | `sd` (`clause_w`) | 10000 | `none` | 1% | 42.37 / 120.78 | 4.81 / 10.98 | **5.03 / 8.87** | 5.05 / 8.95 (99) | 4.80% |
| `gnn5_n6_0p5x1_sdc_pure_nocw` | 6 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 59.20 / 133.41 | 5.47 / 10.81 | **5.23 / 10.61** | 5.25 / 10.61 (99) | 4.82% |
| `gnn5_n6_pure_cw` | 6 | `1p4x1` (`csd`) | `sd` (`clause_w`) | 10000 | `none` | 1% | 45.16 / 153.60 | 5.09 / 11.60 | **5.16 / 9.21** | 5.17 / 9.25 (99) | 5.74% |
| `gnn5_n6_pure_nocw` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `none` | 1% | 61.97 / 138.87 | 6.14 / 12.51 | **6.14 / 11.21** | 6.20 / 11.21 (99) | 6.15% |
| `gnn5_n8_2p3x1_pure_cw` | 8 | `2p3x1` (`csd`) | `sd` (`clause_w`) | 10000 | `none` | 1% | 83.08 / 252.28 | 15.83 / 35.79 | **15.95 / 38.27** | 15.76 / 34.59 (99) | 4.35% |
| `gnn5_n8_2p3x1_pure_nocw` | 8 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 114.55 / 317.16 | 17.07 / 38.79 | **17.70 / 41.79** | 17.52 / 40.05 (99) | 4.19% |
| `gnn5_n8_pure_cw` | 8 | `1p4x1` (`csd`) | `sd` (`clause_w`) | 10000 | `none` | 1% | 85.37 / 293.88 | 11.53 / 35.07 | **12.14 / 35.87** | 11.98 / 35.18 (99) | 5.22% |
| `gnn5_n8_pure_nocw` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `none` | 1% | 99.82 / 388.04 | 12.41 / 41.87 | **13.97 / 37.35** | 13.91 / 36.94 (99) | 5.68% |

## 5. Category B: Shared-Parameter L-BFGS-B + Compile-Time Variable Relabeling (Classical-Decimation-Assisted)

*Note: `bias` (`p00 = 1%`) only reorders variables by literal degree difference; `biaspol` (`p00 = 14%` at `n=6`, `7%` at `n=8`) aligns literal majority sign to `|0,0>`; `bppol` (`p00 = 81% / 66%`) and `bpdec` (`p00 = 88% / 78%`) run 45-step / 114-step factor-graph message passing (7,200–36,024 clause operations, exceeding WalkSATlm lookups) to pre-solve formulas into `|0,0>`, which reduces median `1/p` on pre-solved formulas but performs poorly on the `Unsolved Subset (`Succ[0,0]==0`)`.*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | `|0,0>` Pre-Solve | Zero-Oracle (`gamma=0`) Med / B90 | Val Med / B90 | Test (`NF_chk`) Med / B90 | Unsolved Subset (`Succ[0,0]=0`) Med / B90 | Mean Leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `n6_0p5x1_csd_bpdec` | 6 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 88% | 5.62 / 38.12 | 1.92 / 16.61 | **1.91 / 7.32** | 26.92 / 60.19 (12) | 2.49% |
| `n6_0p5x1_sdc_bpdec` | 6 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 88% | 4.20 / 39.53 | 2.01 / 16.17 | **2.01 / 7.57** | 37.51 / 46.40 (12) | 2.86% |
| `n6_1p2x2_csd_bpdec` | 6 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 88% | 3.87 / 59.63 | 1.83 / 42.01 | **1.80 / 17.11** | 58.39 / 72.22 (12) | 1.31% |
| `n6_1p4x1_bias` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bias` | 1% | 60.56 / 213.83 | 6.26 / 15.31 | **7.23 / 15.75** | 7.29 / 15.81 (99) | 6.74% |
| `n6_1p4x1_biaspol` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `biaspol` | 14% | 34.36 / 173.75 | 5.26 / 14.38 | **5.98 / 14.40** | 6.21 / 15.07 (86) | 6.11% |
| `n6_1p4x1_bppol` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bppol` | 81% | 5.17 / 36.98 | 2.16 / 10.69 | **2.19 / 9.01** | 10.08 / 59.31 (19) | 3.32% |
| `n6_1p4x1_ecd_bpdec` | 6 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 2000 | `bpdec` | 88% | 1.63 / 10466.08 | 1.61 / 266.93 | **1.64 / 125.30** | 672.00 / 1322.96 (12) | 6.49% |
| `n6_1p4x1_ecdt_bpdec` | 6 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 2000 | `bpdec` | 88% | 4.32 / 36.61 | 1.97 / 27.86 | **2.00 / 13.94** | 48.34 / 223.08 (12) | 1.63% |
| `n6_1p4x1_msdec_sdir` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 88% | 8.85 / 37.44 | 2.17 / 14.62 | **2.15 / 10.09** | 21.88 / 918.21 (12) | 1.05% |
| `n6_2p3x1_csd_bpdec` | 6 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 88% | 2.30 / 42.55 | 1.93 / 29.36 | **1.92 / 13.26** | 24.16 / 38.85 (12) | 4.49% |
| `n6_ntrain10000_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 88% | 4.43 / 46.80 | 2.07 / 15.48 | **2.08 / 7.46** | 36.25 / 51.16 (12) | 2.84% |
| `n6_ntrain2000_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 88% | 4.44 / 52.17 | 1.76 / 25.44 | **1.77 / 11.27** | 24.89 / 39.81 (12) | 1.82% |
| `n6_ntrain5000_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `bpdec` | 88% | 4.52 / 59.92 | 1.76 / 22.78 | **1.76 / 8.95** | 27.53 / 69.20 (12) | 2.39% |
| `n6_ntrain500_bpdec` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bpdec` | 88% | 3.58 / 35.85 | 1.62 / 41.16 | **1.62 / 18.75** | 54.68 / 125.37 (12) | 3.76% |
| `n8_0p5x1_csd_bpdec` | 8 | `0p5x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 5.45 / 760.51 | 2.43 / 471.42 | **2.40 / 393.16** | 364.58 / 810.70 (22) | 2.38% |
| `n8_0p5x1_sdc_bpdec` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 5.48 / 673.95 | 2.35 / 619.62 | **2.38 / 336.04** | 290.14 / 959.98 (22) | 2.34% |
| `n8_0p5x1_sdc_msdec_sdir` | 8 | `0p5x1` (`sdc`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 8.32 / 355.13 | 2.50 / 182.02 | **2.51 / 178.55** | 160.39 / 828.62 (22) | 1.74% |
| `n8_1p2x2_csd_bpdec` | 8 | `1p2x2` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 2.12 / 2854.96 | 2.24 / 922.05 | **2.27 / 428.81** | 413.98 / 1146.89 (22) | 2.26% |
| `n8_1p4x1_bias` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bias` | 1% | 178.53 / 519.16 | 69.57 / 199.22 | **58.67 / 184.84** | 57.69 / 185.89 (99) | 5.36% |
| `n8_1p4x1_biaspol` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `biaspol` | 7% | 87.15 / 378.50 | 18.05 / 146.56 | **21.46 / 232.15** | 22.03 / 236.79 (93) | 4.17% |
| `n8_1p4x1_bppol` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bppol` | 66% | 5.22 / 512.97 | 2.91 / 184.01 | **2.82 / 148.84** | 51.00 / 458.75 (34) | 3.73% |
| `n8_1p4x1_ecd_bpdec` | 8 | `1p4x1` (`csd`) | `ecd` (`fixed_cost`) | 800 | `bpdec` | 78% | 1.85 / 1000000000000.00 | 2.04 / 2572.85 | **2.09 / 1142.51** | 1017.74 / 24411.20 (22) | 6.16% |
| `n8_1p4x1_ecdt_bpdec` | 8 | `1p4x1` (`csd`) | `ecdt` (`fixed_cost`) | 800 | `bpdec` | 78% | 7.01 / 1098.95 | 3.32 / 1178.00 | **3.28 / 771.67** | 720.87 / 18930.84 (22) | 9.85% |
| `n8_1p4x1_msdec_sdir` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 4.90 / 552.17 | 2.67 / 181.94 | **2.69 / 128.71** | 123.23 / 2814.83 (22) | 1.96% |
| `n8_2p3x1_csd_bpdec` | 8 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 3.45 / 1200.76 | 2.52 / 740.22 | **2.48 / 847.67** | 766.39 / 2478.21 (22) | 4.68% |
| `n8_ntrain10000_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 78% | 3.17 / 924.85 | 2.38 / 291.58 | **2.40 / 388.81** | 331.42 / 715.23 (22) | 2.31% |
| `n8_ntrain2000_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 2000 | `bpdec` | 78% | 3.50 / 892.33 | 2.46 / 435.25 | **2.53 / 264.35** | 249.38 / 882.30 (22) | 2.97% |
| `n8_ntrain5000_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 5000 | `bpdec` | 78% | 3.00 / 1620.37 | 2.31 / 347.75 | **2.33 / 381.05** | 343.38 / 927.03 (22) | 2.12% |
| `n8_ntrain500_bpdec` | 8 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 500 | `bpdec` | 78% | 4.22 / 812.57 | 2.66 / 918.70 | **2.66 / 459.05** | 419.62 / 27986.31 (22) | 5.26% |

## 6. Category D: Classical-Decimation-Assisted GNN (`bpdec`)

*Combines 114-step factor-graph decimation (`p00 = 88%` at `n=6`, `78%` at `n=8`) with GNN per-formula modulation.*

| Tag | `n` | Schedule | Gate (`Cost`) | `N_train` | Relabel | `|0,0>` Pre-Solve | Zero-Oracle (`gamma=0`) Med / B90 | Val Med / B90 | Test (`NF_chk`) Med / B90 | Unsolved Subset (`Succ[0,0]=0`) Med / B90 | Mean Leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gnn5_n6_0p5x1_sdc_bpdec_cw` | 6 | `0p5x1` (`sdc`) | `sd` (`clause_w`) | 10000 | `bpdec` | 88% | 1.50 / 10.93 | 1.09 / 4.04 | **1.09 / 3.38** | 5.27 / 10.97 (12) | 0.73% |
| `gnn5_n6_1p4x1_bpdec_nocw` | 6 | `1p4x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 88% | 1.43 / 10.88 | 1.24 / 7.95 | **1.21 / 4.16** | 8.12 / 14.52 (12) | 3.26% |
| `gnn5_n8_0p5x1_sdc_bpdec_cw` | 8 | `0p5x1` (`sdc`) | `sd` (`clause_w`) | 10000 | `bpdec` | 78% | 2.83 / 97.50 | 1.76 / 30.17 | **1.80 / 30.40** | 28.38 / 441.92 (22) | 1.61% |
| `gnn5_n8_2p3x1_bpdec_nocw` | 8 | `2p3x1` (`csd`) | `sd` (`fixed_cost`) | 10000 | `bpdec` | 78% | 2.81 / 140.08 | 2.42 / 89.24 | **2.39 / 64.97** | 64.32 / 121.79 (22) | 5.32% |
