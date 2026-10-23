# Oct 23 Exploration — Part 2: Multi-Cavity Scaling (`K = 4` Bits/Cavity, `n = 8, 12, 16, 20`)

## 0. Experimental Protocol & Audit Compliance
- **Encoding**: `K = 4` bits per cavity (`D = 2^4 = 16` computational Fock states `0..15`, `C = n / 4 in {2, 3, 4, 5}` cavities for `n in {8, 12, 16, 20}`, cutoff `N_Fock = 40`).
- **Strict Computational-Subspace SNAP**: All SNAP gates (both in `k_prep = 2` state preparation and in all `L` circuit layers) act **exclusively on computational Fock states `0..15`**, with strictly `0` phase on leaked Fock levels `f >= 16`.
- **Leakage Accounted as Failure**: Any Fock population outside `0..15` in any cavity is excluded from `p_succ` and reported explicitly as `Leak%`.
- **No Classical Pre-Solving (`relabel = none`)**: Canonical variable ordering and polarity (`flips = 0`), with raw factor-graph degree features only (`fv[:, :, 3:7] = 0`). The vacuum `|0,0,...,0>` pre-solve rate is `1%` at `n = 8` and `0%` at `n = 12, 16, 20`.
- **Size-Transferable Training**: Shared parameters and GNN weights are trained **strictly on `n = 8, 12` training instances** (`TRB = 51e9`, selected on `n = 12` validation `VAB = 53e9`) and evaluated **zero-shot at `n = 16, 20`** on 100 unseen test instances per `n` (`TEB = 52e9`).

### Classical Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)
| Classical Baseline / Reference | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WalkSATlm Tries** (`p = 0.30`, 500 runs/inst) | `2.01 / 3.74` | `2.92 / 6.28` | `4.55 / 9.09` | `5.63 / 14.2` | **`1.092^n / 1.116^n`** |
| **WalkSATlm Clause Lookups** (`p = 0.30`, 500 runs/inst) | `13.99 / 29.40` | `32.51 / 76.12` | `78.93 / 214.08` | `111.79 / 356.11` | **`1.191^n / 1.236^n`** |
| **Uniform Random Guessing (`2^n` / Solutions)** | `42.67 / 128.00` | `585.14 / 2048.00` | `9362.29 / 32768.00` | `149796.6 / 524288.0` | **`1.970^n / 2.000^n`** |

## 1A. Stage-1 Ideal Cavity Gray Mixer `U_gray(β)` + Global Cost Only (`2L` params `(β_l, γ_l)`; no free SNAP θ)

> **Stage-1 only:** exact `U_gray(β_l)` + cost phase `γ_l` in the computational subspace (`2L` schedule parameters). This is **not** physical per-cavity `0..15` SNAP synthesis — that is Stage-2 / §1B. ★T = dual-beats WalkSATlm tries (Med **and** B90); ✓L = dual-beats WalkSATlm lookups. `p2_gray_L15` at `n=20` is **not** ★T (`B90 = 14.52 ≰ 14.2`).

| Tag | Mode | Arch / Init | `L` | `cw` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | Zero-Oracle `n=12 / n=20` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `p2_gray_L12` | `shared` | `Shared-2L` | 12 | no | **1.50 / 3.92** (✓L) | **2.24 / 11.66** (✓L) | **3.16 / 20.20** (✓L) | **5.92 / 22.91** (✓L) | **`1.1180^n / 1.1575^n`** | `2018` / `311066` |
| `p2_gray_L15` | `shared` | `Shared-2L` | 15 | no | **1.29 / 3.08** (★T) | **1.83 / 8.25** (✓L) | **2.60 / 13.73** (✓L) | **3.87 / 14.52** (✓L) | **`1.0955^n / 1.1376^n`** | `1791` / `351532` |

## 1B. Direct Physical `k = 1, 2, 3` `SNAP+D` Blocks per Layer — Shared Parameters (L-BFGS-B on `n = 8 + 12`)
| Tag | `L` | `k` (`SNAP+D`/layer) | Blocks/Cav (`2 + L*k`) | `N_tr(8,12)` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | `n=12` Zero-Oracle | `n=20` Leak% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `p2_shared_L4_k2` | 4 | 2 | 10 | `2000,500` | 14.13 / 33.16 | 60.84 / 176.39 | 284.81 / 1000.85 | 1124.18 / 3447.02 | **`1.4432^n / 1.4795^n`** | `1915/5451` | 7.29% |
| `p2_shared_L6_k1` | 6 | 1 | 8 | `2000,500` | **11.98 / 32.17** | 52.89 / 194.88 | 201.01 / 661.52 | 837.33 / 3819.40 | **`1.4218^n / 1.4752^n`** | `1988/5393` | 14.37% |
| `p2_shared_L6_k2` | 6 | 2 | 14 | `2000,500` | **11.52 / 26.09** (✓L) | 45.74 / 141.96 | 180.17 / 608.29 | 557.85 / 2261.75 | **`1.3844^n / 1.4493^n`** | `1875/10237` | 11.62% |
| `p2_shared_L8_k1` | 8 | 1 | 10 | `2000,500` | **9.79 / 24.68** (✓L) | 38.99 / 125.76 | 135.39 / 429.53 | 540.35 / 2487.57 | **`1.3937^n / 1.4574^n`** | `2054/6085` | 14.61% |
| `p2_shared_L8_k2` | 8 | 2 | 18 | `2000,500` | **9.03 / 22.54** (✓L) | **32.30 / 82.26** | 101.57 / 390.21 | 346.75 / 1074.84 | **`1.3529^n / 1.3892^n`** | `2096/13507` | 9.45% |
| `p2_shared_L10_k1` | 10 | 1 | 12 | `2000,500` | **8.34 / 22.26** (✓L) | 32.70 / 104.58 | 102.82 / 314.13 | 397.61 / 1775.78 | **`1.3750^n / 1.4275^n`** | `1964/7012` | 14.63% |
| `p2_shared_L10_k2` | 10 | 2 | 22 | `2000,500` | **8.27 / 22.11** (✓L) | **26.89 / 82.53** | 85.37 / 315.78 | 292.10 / 797.87 | **`1.3447^n / 1.3532^n`** | `2138/15869` | 10.49% |
| `p2_shared_L12_k1` | 12 | 1 | 14 | `2000,500` | **7.53 / 20.26** (✓L) | **28.47 / 95.48** | 88.17 / 330.38 | 328.28 / 1366.22 | **`1.3653^n / 1.4146^n`** | `1880/8622` | 15.40% |
| `p2_shared_L12_k2` | 12 | 2 | 26 | `2000,500` | **6.59 / 16.10** (✓L) | **20.25 / 77.80** | **48.49 / 173.04** | 150.68 / 518.76 | **`1.2925^n / 1.3237^n`** | `2312/28664` | 11.81% |
| `p2_shared_L15_k1` | 15 | 1 | 17 | `2000,500` | **7.20 / 20.11** (✓L) | **26.13 / 85.97** | 77.32 / 283.45 | 298.63 / 1074.23 | **`1.3587^n / 1.3885^n`** | `1764/9332` | 15.21% |
| `p2_shared_L15_k2` | 15 | 2 | 32 | `2000,500` | **7.16 / 17.62** (✓L) | **24.13 / 62.53** (✓L) | **64.56 / 231.25** | 220.68 / 675.62 | **`1.3255^n / 1.3582^n`** | `2244/57090` | 18.91% |
| `p2_shared_L15_k3` | 15 | 3 | 48 | `2000,500` | **5.63 / 13.29** (✓L) | **16.43 / 48.39** (✓L) | **40.39 / 127.98** (✓L) | 112.56 / 345.85 | **`1.2803^n / 1.3083^n`** | `2786/65516` | 13.91% |

## 2. Pure Factor-Graph GNN Scaling (`relabel = none`, Trained on `n = 8, 12` -> Zero-Shot `n = 16, 20`)
| Tag | `L` | `k` | Arch (`H`) | `clause_w` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | `n=12` Zero-Oracle | `n=20` Leak% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

