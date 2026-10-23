# Wavefunction Failure Diagnosis: Why the 16-Level Native Ansatz Fails on 4-SAT B90 and 8-SAT

This report analyzes the full final and intermediate 2ⁿ-state wavefunctions across all 100 unseen test instances at n = 8, 12, 16, 20 for both **4-SAT** (`results/clean16/diagnosis_data.json`) and **8-SAT** (`results/clean16_8sat/diagnosis_data.json`), comparing:
- **QAOA L4 (`Q`)**: All-to-all exact problem Hamiltonian H_C with 4 layers (8 global parameters).
- **Global Cost L4 (`G`)**: All-to-all clause-weighted H_C + GNN per-layer parameters.
- **Pure Chain-Projected QAOA L4 (`QC_block`, `QC_r0`, `QC_RR`)**: Exact 1D cavity-chain projection Proj_chain(E) = E₁ + E₂ − E_ov with no neural network (`QC_block` = identity variable order; `QC_r0` = min-cut placement; `QC_RR` = R=4 round-robin).
- **Native `ckpt_110` (`N_r0`, `N_RR`)**: Full OpenEvolve winner (`Proj_pv(E_w) + corr(dS, dB) + grover(dS)`).
- **Component Ablations of `ckpt_110` (at r=0)**:
  - `N_pv_r0`: `Proj_pv(E_w)` only (pv-weighted chain projection of GNN-weighted clauses; `dS = 0, dB = 0`).
  - `N_corr_r0`: `Proj_pv(E_w) + corr(dS, dB)` (without Grover phase boost).
  - `N_grover_r0`: `Proj_pv(E_w) + grover(dS)` (without learned residual `dS, dB` corrections).
  - `N_unif_dSdB`: Uniform projection `Proj_unif(E_w) + dS + dB` (replacing learned `pv` with `pv = 0.5`).
  - `N_unif_w`: Uniform projection `Proj_unif(E_w)` only (`pv = 0.5, dS = 0, dB = 0`).

---

## Executive Summary: Locking vs. Smearing

The wavefunction diagnostics reveal **two distinct physical failure regimes**—one in **4-SAT** driven by **over-concentration (locking into false attractors / ghost minima)**, and a second in **8-SAT** driven by **catastrophic projection collapse (71% Hamiltonian variance lost at n=20) combined with GNN cue failure**:

1. **4-SAT Failure Mode — Hyper-Concentration into Ghost Minima (`N_eff` is 4.6× SMALLER than QAOA at n=20):**
   - On 4-SAT, Native `ckpt_110` does **not** smear out. At n = 20 (2²⁰ = 1,048,576 states), All-to-All QAOA L4 has median participation ratio `N_eff = 2,836.3` (`H(P)/n = 0.737`), whereas Native `ckpt_110` (`N_r0`) collapses the wavefunction to **`N_eff = 619.7`** (`H(P)/n = 0.612`)—**4.6× more concentrated** than QAOA.
   - Native concentrates **27.3% of the entire wavefunction** into the top 100 wrong bitstrings (`P_top100_wrong = 0.2728` vs QAOA's `0.1230`) and **8.02%** into just the top 10 wrong bitstrings (`P_top10_wrong = 0.0802` vs QAOA's `0.0259`).
   - On the **10 hardest formulas (Hard10 / B90 tail)** at n = 20, Native still packs `7.65%` of all probability mass into 10 wrong bitstrings (`N_eff = 856.6`), but the true solution probability `p_sol` is starved (`P_top10_wrong / p_sol = 80,572×` vs `8.38×` for QAOA).
   - **Why it locks onto the wrong state:** Even with min-cut placement (`r=0`), the 1D chain projection drops `35.9%` of the 4-SAT energy variance at n = 20 (`VarKept = 0.6414`), creating a median of **`275` uniform ghost minima** (mean `3,728` wrong bitstrings whose projected energy is strictly lower than the true satisfying assignment). While the GNN's bit-marginal prior `pv` (`76.4%` bit accuracy on 4-SAT) cuts the median ghost-minimum count tenfold (`275 → 28.5`, reducing `Med 1/p` from `1,675.9` to `342.6`), whenever `pv` mispredicts a cluster of `~3.7–4.7` variables on a hard instance, `Proj_pv(E_w)` + `dS/dB` + Grover amplifies a false attractor (`E_top1 = 2.23` violated clauses, `PE_z = -3.37 σ`) that beats the true ground state in projected energy **74% of the time**.

2. **8-SAT Failure Mode — Projection Annihilation + Smearing Across 314,555 Ghost Minima + Broken GNN Cues:**
   - Because an 8-SAT clause spans 8 variables across `C = 3, 4, 5` cavities, any clause touching 3 or more cavities (or non-adjacent cavities) has **zero** exact projection onto the 1D nearest-neighbor chain. At n = 20 (`C = 5`), the chain projection destroys **70.8% of the Hamiltonian variance** (`VarKept = 0.2918`, `Corr(Proj(E), E) = 0.540`), leaving `argmin Proj(E)` as a true solution **0.0% of the time** (its mean true energy is `E = 7.10` violated clauses) and creating **`6,712` median uniform ghost minima** (mean `37,129`).
   - Simultaneously, on 8-SAT (`m ≈ 176.5 n` clauses) the GNN's bit-marginal predictions `pv` collapse to **56%–59% bit accuracy** (`pv_conf = 0.20`, barely above random 50% coin flips), and the Grover top-quantile mark (`mark > 0.5`) hits the true solution's cavity level **0.0% to 0.2% of the time**.
   - Consequently, at n = 20 8-SAT the wavefunction is **both 3.4× more smeared out** than QAOA (`Med N_eff = 17,638` for `N_r0` and `15,694` for `QC_r0` vs `5,190` for QAOA; `97.6%` of all probability mass sits at `E ≥ 5` violated clauses) **and** every state it amplifies is a high-energy ghost minimum (`E_top1 = 6.78` violated clauses, `PE_z = -3.50 σ`, beating the true solution in projected energy **93% of the time**). Moreover, adding `ckpt_110`'s `dS/dB` tables on 8-SAT **hurts** performance (`N_r0` median `1/p = 96,305` vs `N_pv_r0` `61,529` and pure `QC_r0` `50,128`), exploding median effective ghost minima to **314,555** (30% of the entire Hilbert space) and causing Layer 4 at n = 16 to actively **decrease** `p_sol` (`0.00026 → 0.00019`).

---

## 1. Concentration vs. Smearing Across n = 8, 12, 16, 20

Metrics over the 100 test instances (`B90 1/p` = 90th percentile of `1/p`, matching `OCT23_DATA.md`; `Hard10` = median over the 10 hardest instances with highest `1/p` for that method):

### 1.1 4-SAT: Wavefunction Concentration (`H(P)/n`, `N_eff`) and False-Peak Mass (`P_top10_wrong`)

| n | Method | Med 1/p | B90 1/p (Q90) | Med H(P)/n | Med N_eff (All / Hard10) | Med P_top1_wrong | Med P_top10_wrong (All / Hard10) | Med P_top100_wrong | Med Ratio P_top10_w / p_sol (All / Hard10) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | QAOA L4 (`Q`) | 5.78 | 10.26 | 0.7140 | 23.9 / 31.7 | 0.06531 | 0.36085 / 0.4449 | 0.7734 | 2.21× / 5.5× |
| 8 | Global Cost L4 (`G`) | 5.27 | 9.12 | 0.7214 | 23.6 / 30.2 | 0.06444 | 0.35251 / 0.4295 | 0.7462 | 1.94× / 4.9× |
| 8 | Chain-Proj QAOA (`QC_r0`) | 5.94 | 10.70 | 0.7005 | 22.9 / 29.0 | 0.06851 | 0.38355 / 0.4691 | 0.7819 | 2.39× / 5.9× |
| 8 | Native `Proj_pv` only (`N_pv_r0`) | 5.71 | 9.89 | 0.7239 | 25.6 / 31.8 | 0.07105 | 0.36628 / 0.4455 | 0.7575 | 2.09× / 5.9× |
| 8 | Native `ckpt_110` (`N_r0`) | 4.34 | 9.13 | 0.6855 | 18.5 / 25.4 | 0.06755 | 0.35576 / 0.4628 | 0.7140 | 1.53× / 5.1× |
| 12 | QAOA L4 (`Q`) | 12.85 | 33.15 | 0.7255 | 118.1 / 146.6 | 0.02796 | 0.16867 / 0.1807 | 0.5303 | 2.17× / 7.4× |
| 12 | Global Cost L4 (`G`) | 11.46 | 26.45 | 0.7262 | 112.4 / 146.3 | 0.02720 | 0.16839 / 0.1746 | 0.5212 | 2.01× / 5.9× |
| 12 | Chain-Proj QAOA (`QC_r0`) | 29.77 | 107.54 | 0.6851 | 97.4 / 96.4 | 0.03982 | 0.22552 / 0.2369 | 0.6345 | 7.11× / 36.3× |
| 12 | Native `Proj_pv` only (`N_pv_r0`) | 20.93 | 85.86 | 0.6890 | 100.4 / 119.8 | 0.03754 | 0.20948 / 0.2115 | 0.6293 | 4.25× / 39.2× |
| 12 | Native `ckpt_110` (`N_r0`) | 12.57 | 102.70 | 0.6471 | 61.9 / 78.6 | 0.05061 | 0.25678 / 0.2651 | 0.6503 | 3.33× / 59.7× |
| 16 | QAOA L4 (`Q`) | 34.11 | 84.89 | 0.7329 | 596.5 / 687.0 | 0.00979 | 0.06678 / 0.0709 | 0.2831 | 2.46× / 7.6× |
| 16 | Global Cost L4 (`G`) | 31.67 | 69.69 | 0.7301 | 538.2 / 647.3 | 0.00992 | 0.06909 / 0.0717 | 0.2839 | 2.17× / 6.2× |
| 16 | Chain-Proj QAOA (`QC_r0`) | 204.82 | 2809.62 | 0.6797 | 402.5 / 454.7 | 0.01863 | 0.10103 / 0.0968 | 0.3496 | 16.97× / 2402.7× |
| 16 | Native `Proj_pv` only (`N_pv_r0`) | 83.68 | 3032.98 | 0.6693 | 381.7 / 422.9 | 0.01635 | 0.09933 / 0.0969 | 0.3610 | 8.01× / 5848.9× |
| 16 | Native `ckpt_110` (`N_r0`) | 35.78 | 3319.66 | 0.6172 | 186.0 / 309.8 | 0.02915 | 0.15514 / 0.1245 | 0.4467 | 5.42× / 9076.5× |
| 20 | QAOA L4 (`Q`) | 77.58 | 198.43 | 0.7369 | 2836.3 / 3153.2 | 0.00391 | 0.02593 / 0.0285 | 0.1230 | 1.86× / 8.4× |
| 20 | Global Cost L4 (`G`) | 66.08 | 171.90 | 0.7369 | 2715.0 / 3345.0 | 0.00391 | 0.02625 / 0.0287 | 0.1253 | 1.69× / 7.5× |
| 20 | Chain-Proj QAOA (`QC_r0`) | 1675.86 | 73766.18 | 0.6806 | 1947.0 / 1948.2 | 0.00647 | 0.03955 / 0.0387 | 0.1617 | 65.11× / 24438.1× |
| 20 | Native `Proj_pv` only (`N_pv_r0`) | 342.58 | 24768.24 | 0.6693 | 1908.6 / 2197.2 | 0.00659 | 0.03860 / 0.0329 | 0.1671 | 14.13× / 10600.1× |
| 20 | Native `ckpt_110` (`N_r0`) | 195.18 | 25109.19 | 0.6115 | 619.7 / 856.6 | 0.01452 | 0.08020 / 0.0765 | 0.2728 | 13.77× / 80571.6× |

### 1.2 8-SAT: Wavefunction Concentration (`H(P)/n`, `N_eff`) and False-Peak Mass (`P_top10_wrong`)

| n | Method | Med 1/p | B90 1/p (Q90) | Med H(P)/n | Med N_eff (All / Hard10) | Med P_top1_wrong | Med P_top10_wrong (All / Hard10) | Med P_top100_wrong | Med Ratio P_top10_w / p_sol (All / Hard10) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | QAOA L4 (`Q`) | 7.33 | 17.72 | 0.7031 | 23.6 / 31.6 | 0.08656 | 0.43187 / 0.4622 | 0.8211 | 3.33× / 10.7× |
| 8 | Global Cost L4 (`G`) | 7.62 | 17.22 | 0.7041 | 24.4 / 31.7 | 0.08290 | 0.42776 / 0.4636 | 0.8255 | 3.41× / 10.1× |
| 8 | Chain-Proj QAOA (`QC_r0`) | 8.78 | 18.90 | 0.7417 | 30.7 / 32.9 | 0.06866 | 0.39223 / 0.4483 | 0.8254 | 3.52× / 9.7× |
| 8 | Native `Proj_pv` only (`N_pv_r0`) | 9.36 | 20.46 | 0.7397 | 30.2 / 32.4 | 0.07171 | 0.40348 / 0.4544 | 0.8328 | 3.81× / 11.3× |
| 8 | Native `ckpt_110` (`N_r0`) | 8.96 | 20.46 | 0.7400 | 30.1 / 32.0 | 0.07334 | 0.40359 / 0.4593 | 0.8281 | 3.65× / 11.7× |
| 12 | QAOA L4 (`Q`) | 25.45 | 68.94 | 0.7477 | 147.4 / 167.8 | 0.02829 | 0.16929 / 0.1741 | 0.5382 | 4.36× / 18.2× |
| 12 | Global Cost L4 (`G`) | 25.88 | 63.67 | 0.7385 | 147.6 / 167.0 | 0.02727 | 0.16733 / 0.1718 | 0.5473 | 4.16× / 16.3× |
| 12 | Chain-Proj QAOA (`QC_r0`) | 155.80 | 1349.41 | 0.7477 | 177.0 / 175.5 | 0.02967 | 0.16328 / 0.1707 | 0.5453 | 26.24× / 601.4× |
| 12 | Native `Proj_pv` only (`N_pv_r0`) | 143.43 | 1713.03 | 0.7294 | 148.6 / 153.4 | 0.03172 | 0.17980 / 0.1893 | 0.5839 | 25.64× / 1511.7× |
| 12 | Native `ckpt_110` (`N_r0`) | 143.27 | 1726.09 | 0.7290 | 143.8 / 142.1 | 0.03135 | 0.18863 / 0.1983 | 0.5830 | 26.43× / 1133.6× |
| 16 | QAOA L4 (`Q`) | 104.63 | 268.60 | 0.7750 | 849.6 / 927.2 | 0.01029 | 0.06326 / 0.0649 | 0.2477 | 6.64× / 22.0× |
| 16 | Global Cost L4 (`G`) | 98.03 | 234.59 | 0.7590 | 833.9 / 919.1 | 0.00996 | 0.06249 / 0.0629 | 0.2510 | 6.19× / 19.3× |
| 16 | Chain-Proj QAOA (`QC_r0`) | 3642.42 | 49291.23 | 0.7798 | 1508.9 / 1384.9 | 0.00723 | 0.04449 / 0.0530 | 0.1820 | 186.65× / 5255.5× |
| 16 | Native `Proj_pv` only (`N_pv_r0`) | 2748.94 | 83685.10 | 0.7485 | 1003.7 / 1100.6 | 0.00993 | 0.05715 / 0.0536 | 0.2264 | 181.71× / 14504.1× |
| 16 | Native `ckpt_110` (`N_r0`) | 5381.65 | 89753.90 | 0.7929 | 1686.9 / 1802.1 | 0.00676 | 0.04166 / 0.0424 | 0.1789 | 200.42× / 12760.2× |
| 20 | QAOA L4 (`Q`) | 256.77 | 849.54 | 0.7929 | 5190.1 / 5691.1 | 0.00334 | 0.02139 / 0.0208 | 0.0951 | 5.70× / 25.5× |
| 20 | Global Cost L4 (`G`) | 245.78 | 805.65 | 0.7724 | 4942.3 / 5352.4 | 0.00322 | 0.02064 / 0.0203 | 0.0955 | 5.39× / 21.1× |
| 20 | Chain-Proj QAOA (`QC_r0`) | 50128.39 | 534327.13 | 0.8097 | 15694.3 / 19472.0 | 0.00131 | 0.00940 / 0.0073 | 0.0472 | 441.46× / 18512.8× |
| 20 | Native `Proj_pv` only (`N_pv_r0`) | 61528.84 | 1220837.84 | 0.7750 | 9095.6 / 10388.7 | 0.00205 | 0.01365 / 0.0116 | 0.0662 | 755.31× / 43763.2× |
| 20 | Native `ckpt_110` (`N_r0`) | 96304.55 | 2518346.72 | 0.8142 | 17637.7 / 15539.4 | 0.00129 | 0.00891 / 0.0094 | 0.0429 | 932.77× / 72923.2× |

### 1.3 Easy-10 vs. Median-10 vs. Hard-10 (B90 Tail) at n = 20 (`N_r0` vs. `Q`)

Sorting the 100 test instances at n = 20 by each method's `1/p` (`Easy10` = ranks 0..9, `Med10` = ranks 45..54, `Hard10` = ranks 90..99):

| Dataset | Method | Slice (10 inst) | Med 1/p | Med p_sol | Med N_eff | Med P_top10_wrong | Med Ratio P_top10_w / p_sol | Mean E_top1_wrong | Mean PE_z(top1_w) | Mean d_H(top1_w, sol) | Mean pv bit acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-SAT | QAOA L4 (`Q`) | Easy10 | 25.8 | 0.038753 | 2624.4 | 0.0244 | 0.7× | 1.00 | -2.39 | 5.60 | 88.5% |
| 4-SAT | QAOA L4 (`Q`) | Med10 | 77.6 | 0.012894 | 2936.0 | 0.0247 | 1.9× | 1.10 | -2.23 | 7.10 | 79.0% |
| 4-SAT | QAOA L4 (`Q`) | Hard10 (B90) | 314.3 | 0.003182 | 3153.2 | 0.0285 | 8.4× | 1.00 | -2.74 | 9.80 | 60.0% |
| 4-SAT | Chain-Proj (`QC_r0`) | Easy10 | 85.4 | 0.011727 | 1277.9 | 0.0539 | 4.5× | 2.70 | -3.85 | 3.40 | 91.0% |
| 4-SAT | Chain-Proj (`QC_r0`) | Med10 | 1675.9 | 0.000597 | 1917.8 | 0.0426 | 66.3× | 3.90 | -3.84 | 4.90 | 79.5% |
| 4-SAT | Chain-Proj (`QC_r0`) | Hard10 (B90) | 712591.2 | 0.000002 | 1948.2 | 0.0387 | 24438.1× | 4.40 | -3.77 | 7.50 | 62.0% |
| 4-SAT | Native (`N_r0`) | Easy10 | 11.7 | 0.085824 | 234.0 | 0.0979 | 1.1× | 1.80 | -3.27 | 2.00 | 92.0% |
| 4-SAT | Native (`N_r0`) | Med10 | 195.2 | 0.005143 | 570.9 | 0.0914 | 15.1× | 2.10 | -3.44 | 4.20 | 79.0% |
| 4-SAT | Native (`N_r0`) | Hard10 (B90) | 948776.3 | 0.000003 | 856.6 | 0.0765 | 80571.6× | 3.30 | -3.38 | 10.20 | 57.0% |
| 8-SAT | QAOA L4 (`Q`) | Easy10 | 93.5 | 0.010696 | 4196.3 | 0.0231 | 2.1× | 1.30 | -2.43 | 6.10 | 66.0% |
| 8-SAT | QAOA L4 (`Q`) | Med10 | 256.8 | 0.003895 | 4738.9 | 0.0229 | 6.0× | 1.30 | -2.58 | 7.90 | 63.0% |
| 8-SAT | QAOA L4 (`Q`) | Hard10 (B90) | 1104.1 | 0.000908 | 5691.1 | 0.0208 | 25.5× | 1.40 | -2.41 | 11.40 | 52.5% |
| 8-SAT | Chain-Proj (`QC_r0`) | Easy10 | 2309.8 | 0.000434 | 11621.0 | 0.0113 | 26.1× | 6.40 | -4.17 | 6.60 | 59.0% |
| 8-SAT | Chain-Proj (`QC_r0`) | Med10 | 50128.4 | 0.000020 | 19307.6 | 0.0083 | 357.2× | 7.40 | -4.11 | 6.00 | 58.5% |
| 8-SAT | Chain-Proj (`QC_r0`) | Hard10 (B90) | 2364248.2 | 0.000001 | 19472.0 | 0.0073 | 18512.8× | 7.00 | -3.85 | 8.40 | 52.5% |
| 8-SAT | Native (`N_r0`) | Easy10 | 2539.6 | 0.000396 | 13215.5 | 0.0104 | 26.2× | 7.10 | -3.60 | 5.20 | 62.5% |
| 8-SAT | Native (`N_r0`) | Med10 | 96304.5 | 0.000010 | 19944.3 | 0.0089 | 932.8× | 5.70 | -3.57 | 6.80 | 60.0% |
| 8-SAT | Native (`N_r0`) | Hard10 (B90) | 8271007.8 | 0.000000 | 15539.4 | 0.0094 | 72923.2× | 6.60 | -3.42 | 9.30 | 53.5% |

---

## 2. True Energy vs. Projected Energy of Wrong States ("Ghost Minima")

A **ghost minimum** is an unsatisfying bitstring (`E(z) > 0`) whose 1D chain-projected energy is **strictly lower** than the minimum projected energy among all true satisfying assignments (`min_{z*: E(z*)=0} Proj(E)(z*)`).

| Dataset | n | VarKept (block / min-cut r=0) | Corr(Proj(E), E) (r=0) | argmin Proj(E) is True Sol | Mean True E at argmin Proj(E) | Med / Mean Ghost Minima `N_ghost` (Uniform `Proj_unif(E)`) | Med Ghost Minima (`Proj_pv(E)`) | Med Ghost Minima (`ckpt_110` Layer-4 Effective Cost) |
|---|---|---|---|---|---|---|---|---|
| 4-SAT | 8 | 1.0000 / 1.0000 | 1.0000 | 100.0% | 0.00 | 0.0 / 0.0 | 0.0 | 0.0 |
| 4-SAT | 12 | 0.7222 / 0.8303 | 0.9111 | 28.0% | 1.14 | 2.0 / 5.7 | 0.0 | 0.0 |
| 4-SAT | 16 | 0.6010 / 0.7183 | 0.8474 | 8.0% | 1.98 | 21.0 / 254.4 | 2.0 | 2.0 |
| 4-SAT | 20 | 0.5223 / 0.6414 | 0.8006 | 3.0% | 3.46 | 275.0 / 3728.2 | 28.5 | 22.0 |
| 8-SAT | 8 | 1.0000 / 1.0000 | 1.0000 | 100.0% | 0.00 | 0.0 / 0.0 | 0.0 | 0.0 |
| 8-SAT | 12 | 0.5149 / 0.5986 | 0.7735 | 8.0% | 2.22 | 28.5 / 81.2 | 22.5 | 839.0 |
| 8-SAT | 16 | 0.3252 / 0.3962 | 0.6292 | 1.0% | 4.77 | 565.0 / 1920.1 | 537.0 | 19435.5 |
| 8-SAT | 20 | 0.2344 / 0.2918 | 0.5398 | 0.0% | 7.10 | 6712.0 / 37129.4 | 8676.5 | 314554.5 |

### 2.2 Energy Profile and Energy-Shell Distribution of the Top Wrong States

| Dataset | n | Method | Mean True E (top1 / top10w) | Mean Proj Energy z-score `PE_z(top1)` | % Instances Where `top1_wrong` Beats True Sol in `Proj_unif(E)` / `Proj_pv(E)` | Probability Mass by True Energy Shell: `E=0` / `E=1` / `E=2` / `E=3..4` / `E≥5` |
|---|---|---|---|---|---|---|
| 4-SAT | 8 | QAOA L4 (`Q`) | 1.04 / 1.30 | -1.83 σ | 0% / 0% | 0.208 / 0.290 / 0.235 / 0.170 / 0.096 |
| 4-SAT | 8 | Chain-Proj (`QC_r0`) | 1.04 / 1.32 | -1.83 σ | 0% / 0% | 0.207 / 0.302 / 0.250 / 0.155 / 0.087 |
| 4-SAT | 8 | Native `Proj_pv` (`N_pv_r0`) | 1.08 / 1.43 | -1.81 σ | 0% / 0% | 0.201 / 0.259 / 0.239 / 0.203 / 0.098 |
| 4-SAT | 8 | Native `ckpt_110` (`N_r0`) | 1.04 / 1.29 | -1.83 σ | 0% / 0% | 0.268 / 0.284 / 0.206 / 0.158 / 0.084 |
| 4-SAT | 12 | QAOA L4 (`Q`) | 1.04 / 1.21 | -2.11 σ | 21% / 21% | 0.092 / 0.201 / 0.239 / 0.264 / 0.203 |
| 4-SAT | 12 | Chain-Proj (`QC_r0`) | 1.57 / 1.88 | -2.77 σ | 64% / 35% | 0.052 / 0.144 / 0.236 / 0.378 / 0.191 |
| 4-SAT | 12 | Native `Proj_pv` (`N_pv_r0`) | 1.72 / 2.04 | -2.52 σ | 34% / 39% | 0.063 / 0.140 / 0.200 / 0.346 / 0.251 |
| 4-SAT | 12 | Native `ckpt_110` (`N_r0`) | 1.22 / 1.63 | -2.60 σ | 45% / 39% | 0.103 / 0.200 / 0.227 / 0.299 / 0.172 |
| 4-SAT | 16 | QAOA L4 (`Q`) | 1.02 / 1.17 | -2.31 σ | 33% / 33% | 0.036 / 0.120 / 0.192 / 0.323 / 0.329 |
| 4-SAT | 16 | Chain-Proj (`QC_r0`) | 2.29 / 2.56 | -3.33 σ | 85% / 49% | 0.011 / 0.053 / 0.119 / 0.367 / 0.451 |
| 4-SAT | 16 | Native `Proj_pv` (`N_pv_r0`) | 2.29 / 2.67 | -2.90 σ | 48% / 50% | 0.023 / 0.068 / 0.122 / 0.311 / 0.477 |
| 4-SAT | 16 | Native `ckpt_110` (`N_r0`) | 1.48 / 1.85 | -3.07 σ | 61% / 61% | 0.049 / 0.121 / 0.180 / 0.329 / 0.322 |
| 4-SAT | 20 | QAOA L4 (`Q`) | 1.03 / 1.21 | -2.52 σ | 29% / 29% | 0.016 / 0.061 / 0.127 / 0.319 / 0.477 |
| 4-SAT | 20 | Chain-Proj (`QC_r0`) | 3.57 / 3.85 | -3.73 σ | 94% / 52% | 0.002 / 0.011 / 0.040 / 0.223 / 0.724 |
| 4-SAT | 20 | Native `Proj_pv` (`N_pv_r0`) | 3.10 / 3.56 | -3.16 σ | 58% / 63% | 0.007 / 0.024 / 0.057 / 0.218 / 0.694 |
| 4-SAT | 20 | Native `ckpt_110` (`N_r0`) | 2.23 / 2.42 | -3.37 σ | 72% / 74% | 0.018 / 0.051 / 0.104 / 0.300 / 0.527 |
| 8-SAT | 8 | QAOA L4 (`Q`) | 1.08 / 1.47 | -1.91 σ | 0% / 0% | 0.145 / 0.278 / 0.253 / 0.231 / 0.093 |
| 8-SAT | 8 | Chain-Proj (`QC_r0`) | 1.06 / 1.43 | -1.92 σ | 0% / 0% | 0.121 / 0.257 / 0.259 / 0.241 / 0.122 |
| 8-SAT | 8 | Native `Proj_pv` (`N_pv_r0`) | 1.07 / 1.46 | -1.92 σ | 0% / 0% | 0.117 / 0.258 / 0.260 / 0.235 / 0.130 |
| 8-SAT | 8 | Native `ckpt_110` (`N_r0`) | 1.07 / 1.46 | -1.92 σ | 0% / 0% | 0.118 / 0.259 / 0.259 / 0.233 / 0.131 |
| 8-SAT | 12 | QAOA L4 (`Q`) | 1.14 / 1.47 | -2.03 σ | 37% / 37% | 0.046 / 0.129 / 0.221 / 0.340 / 0.265 |
| 8-SAT | 12 | Chain-Proj (`QC_r0`) | 2.68 / 2.99 | -2.93 σ | 89% / 89% | 0.011 / 0.047 / 0.116 / 0.369 / 0.457 |
| 8-SAT | 12 | Native `Proj_pv` (`N_pv_r0`) | 2.66 / 2.94 | -2.83 σ | 87% / 94% | 0.012 / 0.052 / 0.125 / 0.373 / 0.439 |
| 8-SAT | 12 | Native `ckpt_110` (`N_r0`) | 2.61 / 2.89 | -2.85 σ | 89% / 92% | 0.012 / 0.053 / 0.128 / 0.372 / 0.435 |
| 8-SAT | 16 | QAOA L4 (`Q`) | 1.28 / 1.56 | -2.40 σ | 64% / 64% | 0.012 / 0.050 / 0.123 / 0.329 / 0.486 |
| 8-SAT | 16 | Chain-Proj (`QC_r0`) | 4.93 / 5.03 | -3.55 σ | 96% / 94% | 0.001 / 0.005 / 0.019 / 0.130 / 0.845 |
| 8-SAT | 16 | Native `Proj_pv` (`N_pv_r0`) | 4.71 / 4.72 | -3.41 σ | 93% / 96% | 0.001 / 0.006 / 0.023 / 0.151 / 0.818 |
| 8-SAT | 16 | Native `ckpt_110` (`N_r0`) | 4.62 / 4.86 | -3.24 σ | 89% / 92% | 0.001 / 0.004 / 0.016 / 0.113 / 0.865 |
| 8-SAT | 20 | QAOA L4 (`Q`) | 1.36 / 1.65 | -2.42 σ | 68% / 68% | 0.004 / 0.019 / 0.059 / 0.245 / 0.672 |
| 8-SAT | 20 | Chain-Proj (`QC_r0`) | 6.93 / 7.14 | -4.00 σ | 100% / 96% | 0.000 / 0.001 / 0.003 / 0.027 / 0.969 |
| 8-SAT | 20 | Native `Proj_pv` (`N_pv_r0`) | 6.51 / 6.66 | -3.75 σ | 98% / 100% | 0.000 / 0.001 / 0.004 / 0.036 / 0.960 |
| 8-SAT | 20 | Native `ckpt_110` (`N_r0`) | 6.78 / 7.01 | -3.50 σ | 90% / 93% | 0.000 / 0.000 / 0.002 / 0.022 / 0.976 |

---

## 3. Hamming Distance Structure & GNN Cue Accuracy (`pv`, `E_loc`, Grover Mark)

Where do the top wrong states sit in Hamming space relative to (a) the nearest true satisfying assignment `z*`, (b) the GNN's bit-marginal sign prediction `x_pv = (pv > 0.5)`, (c) the cavity-wise local optimistic minimum `x_E1 = argmin E_loc`, and (d) the global minimum of the projected Hamiltonian `argmin Proj_unif(E)`?

| Dataset | n | GNN `pv` Bit Acc (`d_H(pv, sol)`) | `E(x_pv)` | Local Min `d_H(x_E1, sol)` (`E(x_E1)`) | % True-Sol Cavities in Grover Mark | Method | Mean `d_H(top1_w, sol)` | Wrong Cavities (of C) | Wrong Bits per Wrong Cavity | Mean `d_H(top1_w, pv / E1 / argmin_PE)` |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-SAT | 8 | 78.1% (1.75) | 1.40 | 1.75 (1.62) | 6.5% | QAOA (`Q`) | 3.70 | 1.87 / 2 | 2.00 / 4 | 3.37 / 3.24 / 4.40 |
| 4-SAT | 8 | 78.1% (1.75) | 1.40 | 1.75 (1.62) | 6.5% | Chain (`QC_r0`) | 3.65 | 1.88 / 2 | 1.97 / 4 | 3.32 / 3.17 / 4.35 |
| 4-SAT | 8 | 78.1% (1.75) | 1.40 | 1.75 (1.62) | 6.5% | `N_pv_r0` | 4.18 | 1.90 / 2 | 2.23 / 4 | 4.16 / 3.99 / 4.75 |
| 4-SAT | 8 | 78.1% (1.75) | 1.40 | 1.75 (1.62) | 6.5% | `N_r0` | 3.36 | 1.84 / 2 | 1.85 / 4 | 2.57 / 2.62 / 3.92 |
| 4-SAT | 12 | 76.0% (2.88) | 2.22 | 3.03 (2.67) | 8.0% | QAOA (`Q`) | 5.17 | 2.74 / 3 | 1.87 / 4 | 4.48 / 5.75 / 5.04 |
| 4-SAT | 12 | 76.0% (2.88) | 2.22 | 3.03 (2.67) | 8.0% | Chain (`QC_r0`) | 3.58 | 2.11 / 3 | 1.61 / 4 | 3.35 / 3.66 / 2.25 |
| 4-SAT | 12 | 76.0% (2.88) | 2.22 | 3.03 (2.67) | 8.0% | `N_pv_r0` | 4.35 | 2.27 / 3 | 1.85 / 4 | 5.02 / 4.95 / 4.50 |
| 4-SAT | 12 | 76.0% (2.88) | 2.22 | 3.03 (2.67) | 8.0% | `N_r0` | 3.28 | 2.01 / 3 | 1.58 / 4 | 2.81 / 3.04 / 3.17 |
| 4-SAT | 16 | 76.1% (3.82) | 2.96 | 4.09 (3.33) | 8.5% | QAOA (`Q`) | 6.06 | 3.28 / 4 | 1.84 / 4 | 5.78 / 8.16 / 6.10 |
| 4-SAT | 16 | 76.1% (3.82) | 2.96 | 4.09 (3.33) | 8.5% | Chain (`QC_r0`) | 4.20 | 2.49 / 4 | 1.62 / 4 | 3.92 / 4.44 / 2.22 |
| 4-SAT | 16 | 76.1% (3.82) | 2.96 | 4.09 (3.33) | 8.5% | `N_pv_r0` | 4.40 | 2.46 / 4 | 1.68 / 4 | 5.32 / 5.44 / 5.26 |
| 4-SAT | 16 | 76.1% (3.82) | 2.96 | 4.09 (3.33) | 8.5% | `N_r0` | 3.65 | 2.19 / 4 | 1.52 / 4 | 3.07 / 3.63 / 3.67 |
| 4-SAT | 20 | 76.4% (4.72) | 3.74 | 5.05 (4.54) | 10.6% | QAOA (`Q`) | 7.29 | 4.00 / 5 | 1.79 / 4 | 6.89 / 9.55 / 7.00 |
| 4-SAT | 20 | 76.4% (4.72) | 3.74 | 5.05 (4.54) | 10.6% | Chain (`QC_r0`) | 5.71 | 3.30 / 5 | 1.67 / 4 | 5.34 / 5.96 / 2.81 |
| 4-SAT | 20 | 76.4% (4.72) | 3.74 | 5.05 (4.54) | 10.6% | `N_pv_r0` | 6.12 | 3.27 / 5 | 1.77 / 4 | 6.52 / 6.80 / 6.41 |
| 4-SAT | 20 | 76.4% (4.72) | 3.74 | 5.05 (4.54) | 10.6% | `N_r0` | 4.68 | 2.85 / 5 | 1.57 / 4 | 3.67 / 4.15 / 4.80 |
| 8-SAT | 8 | 61.1% (3.11) | 4.42 | 3.14 (3.91) | 0.0% | QAOA (`Q`) | 3.35 | 1.77 / 2 | 1.84 / 4 | 3.38 / 3.60 / 3.70 |
| 8-SAT | 8 | 61.1% (3.11) | 4.42 | 3.14 (3.91) | 0.0% | Chain (`QC_r0`) | 3.89 | 1.90 / 2 | 2.04 / 4 | 3.76 / 3.76 / 4.12 |
| 8-SAT | 8 | 61.1% (3.11) | 4.42 | 3.14 (3.91) | 0.0% | `N_pv_r0` | 3.92 | 1.92 / 2 | 2.05 / 4 | 3.83 / 3.79 / 4.15 |
| 8-SAT | 8 | 61.1% (3.11) | 4.42 | 3.14 (3.91) | 0.0% | `N_r0` | 3.91 | 1.91 / 2 | 2.06 / 4 | 3.76 / 3.76 / 4.16 |
| 8-SAT | 12 | 56.1% (5.27) | 7.09 | 5.11 (5.99) | 0.0% | QAOA (`Q`) | 4.82 | 2.58 / 3 | 1.80 / 4 | 4.76 / 6.07 / 4.66 |
| 8-SAT | 12 | 56.1% (5.27) | 7.09 | 5.11 (5.99) | 0.0% | Chain (`QC_r0`) | 4.19 | 2.30 / 3 | 1.77 / 4 | 5.03 / 5.26 / 2.40 |
| 8-SAT | 12 | 56.1% (5.27) | 7.09 | 5.11 (5.99) | 0.0% | `N_pv_r0` | 4.45 | 2.37 / 3 | 1.82 / 4 | 5.16 / 5.23 / 3.17 |
| 8-SAT | 12 | 56.1% (5.27) | 7.09 | 5.11 (5.99) | 0.0% | `N_r0` | 4.27 | 2.34 / 3 | 1.75 / 4 | 4.79 / 5.00 / 2.90 |
| 8-SAT | 16 | 59.1% (6.54) | 9.17 | 6.71 (8.70) | 0.0% | QAOA (`Q`) | 5.99 | 3.20 / 4 | 1.80 / 4 | 6.59 / 7.94 / 6.45 |
| 8-SAT | 16 | 59.1% (6.54) | 9.17 | 6.71 (8.70) | 0.0% | Chain (`QC_r0`) | 6.67 | 3.30 / 4 | 1.97 / 4 | 6.46 / 6.89 / 2.90 |
| 8-SAT | 16 | 59.1% (6.54) | 9.17 | 6.71 (8.70) | 0.0% | `N_pv_r0` | 6.25 | 3.20 / 4 | 1.90 / 4 | 6.69 / 7.08 / 4.13 |
| 8-SAT | 16 | 59.1% (6.54) | 9.17 | 6.71 (8.70) | 0.0% | `N_r0` | 6.22 | 3.14 / 4 | 1.95 / 4 | 6.21 / 6.42 / 4.97 |
| 8-SAT | 20 | 59.4% (8.11) | 11.92 | 8.28 (10.88) | 0.2% | QAOA (`Q`) | 7.74 | 4.20 / 5 | 1.80 / 4 | 8.12 / 10.21 / 7.48 |
| 8-SAT | 20 | 59.4% (8.11) | 11.92 | 8.28 (10.88) | 0.2% | Chain (`QC_r0`) | 7.59 | 4.03 / 5 | 1.86 / 4 | 8.06 / 8.35 / 2.82 |
| 8-SAT | 20 | 59.4% (8.11) | 11.92 | 8.28 (10.88) | 0.2% | `N_pv_r0` | 7.72 | 4.10 / 5 | 1.86 / 4 | 8.48 / 8.71 / 5.14 |
| 8-SAT | 20 | 59.4% (8.11) | 11.92 | 8.28 (10.88) | 0.2% | `N_r0` | 7.80 | 4.16 / 5 | 1.86 / 4 | 7.60 / 7.75 / 5.78 |

---

## 4. Layer-by-Layer Trajectory (l = 1..4) and Component Ablations of `ckpt_110`

### 4.1 Layer-by-Layer Evolution at n = 16 and n = 20

| Dataset | n | Method | Med `p_sol` (l = 1 → 2 → 3 → 4) | Med `N_eff` (l = 1 → 2 → 3 → 4) | Mean True `<E>` (l = 1 → 2 → 3 → 4) | Mean `<Proj(E)_z>` (l = 1 → 2 → 3 → 4) | Med `P_top10_wrong` (l = 1 → 2 → 3 → 4) |
|---|---|---|---|---|---|---|---|
| 4-SAT | 16 | QAOA L4 (`Q`) | 0.00053 → 0.00406 → 0.01307 → 0.02932 | 22142 → 4257 → 1608 → 596 | 7.88 → 5.27 → 4.58 → 3.94 | -0.76 → -1.32 → -1.36 → -1.46 | 0.0033 → 0.0170 → 0.0351 → 0.0668 |
| 4-SAT | 16 | Chain-Proj (`QC_r0`) | 0.00034 → 0.00167 → 0.00337 → 0.00488 | 22212 → 4239 → 1361 → 402 | 7.75 → 5.83 → 5.15 → 4.45 | -0.85 → -1.63 → -1.92 → -2.23 | 0.0033 → 0.0182 → 0.0455 → 0.1010 |
| 4-SAT | 16 | Native `Proj_pv` (`N_pv_r0`) | 0.00031 → 0.00250 → 0.00640 → 0.01199 | 31106 → 4980 → 1106 → 382 | 8.63 → 6.33 → 5.29 → 4.61 | -0.49 → -1.32 → -1.72 → -1.97 | 0.0022 → 0.0156 → 0.0498 → 0.0993 |
| 4-SAT | 16 | Native `ckpt_110` (`N_r0`) | 0.00099 → 0.00664 → 0.01379 → 0.02796 | 9712 → 1328 → 492 → 186 | 7.12 → 5.01 → 4.36 → 3.71 | -1.00 → -1.73 → -1.98 → -2.21 | 0.0092 → 0.0445 → 0.0887 → 0.1551 |
| 4-SAT | 20 | QAOA L4 (`Q`) | 0.00009 → 0.00128 → 0.00533 → 0.01289 | 270642 → 34455 → 9779 → 2836 | 9.90 → 6.64 → 5.76 → 4.99 | -0.84 → -1.41 → -1.42 → -1.52 | 0.0005 → 0.0042 → 0.0113 → 0.0259 |
| 4-SAT | 20 | Chain-Proj (`QC_r0`) | 0.00004 → 0.00022 → 0.00048 → 0.00060 | 281876 → 35721 → 8935 → 1947 | 9.80 → 7.61 → 6.86 → 6.00 | -0.98 → -1.84 → -2.16 → -2.52 | 0.0005 → 0.0041 → 0.0135 → 0.0396 |
| 4-SAT | 20 | Native `Proj_pv` (`N_pv_r0`) | 0.00004 → 0.00043 → 0.00139 → 0.00292 | 409899 → 41978 → 7450 → 1909 | 10.75 → 8.13 → 6.97 → 6.10 | -0.59 → -1.47 → -1.90 → -2.19 | 0.0003 → 0.0036 → 0.0154 → 0.0386 |
| 4-SAT | 20 | Native `ckpt_110` (`N_r0`) | 0.00017 → 0.00119 → 0.00288 → 0.00514 | 80102 → 7309 → 2174 → 620 | 8.85 → 6.45 → 5.79 → 4.98 | -1.18 → -1.94 → -2.20 → -2.46 | 0.0022 → 0.0158 → 0.0367 → 0.0802 |
| 8-SAT | 16 | QAOA L4 (`Q`) | 0.00040 → 0.00174 → 0.00493 → 0.00956 | 25003 → 7635 → 2388 → 850 | 8.77 → 7.44 → 6.47 → 5.49 | -0.56 → -0.89 → -1.06 → -1.18 | 0.0028 → 0.0124 → 0.0323 → 0.0633 |
| 8-SAT | 16 | Chain-Proj (`QC_r0`) | 0.00005 → 0.00013 → 0.00019 → 0.00027 | 42991 → 19404 → 6628 → 1509 | 10.29 → 8.92 → 8.20 → 7.17 | -0.36 → -1.05 → -1.43 → -1.99 | 0.0012 → 0.0043 → 0.0155 → 0.0445 |
| 8-SAT | 16 | Native `Proj_pv` (`N_pv_r0`) | 0.00005 → 0.00018 → 0.00027 → 0.00036 | 42634 → 14879 → 4115 → 1004 | 10.30 → 8.70 → 7.87 → 6.93 | -0.35 → -1.14 → -1.55 → -2.01 | 0.0011 → 0.0055 → 0.0214 → 0.0571 |
| 8-SAT | 16 | Native `ckpt_110` (`N_r0`) | 0.00005 → 0.00017 → 0.00026 → 0.00019 | 26987 → 12931 → 3853 → 1687 | 10.01 → 8.60 → 7.86 → 7.62 | -0.48 → -1.16 → -1.54 → -1.66 | 0.0026 → 0.0068 → 0.0229 → 0.0417 |
| 8-SAT | 20 | QAOA L4 (`Q`) | 0.00007 → 0.00048 → 0.00168 → 0.00389 | 319174 → 76168 → 18972 → 5190 | 11.03 → 9.39 → 8.22 → 7.04 | -0.54 → -0.88 → -1.05 → -1.15 | 0.0004 → 0.0026 → 0.0088 → 0.0214 |
| 8-SAT | 20 | Chain-Proj (`QC_r0`) | 0.00000 → 0.00001 → 0.00001 → 0.00002 | 680220 → 307344 → 92663 → 15694 | 12.89 → 11.68 → 10.93 → 9.84 | -0.44 → -1.07 → -1.48 → -2.09 | 0.0001 → 0.0005 → 0.0023 → 0.0094 |
| 8-SAT | 20 | Native `Proj_pv` (`N_pv_r0`) | 0.00000 → 0.00001 → 0.00002 → 0.00002 | 674647 → 229301 → 52628 → 9096 | 12.91 → 11.46 → 10.56 → 9.50 | -0.42 → -1.15 → -1.60 → -2.13 | 0.0001 → 0.0006 → 0.0036 → 0.0136 |
| 8-SAT | 20 | Native `ckpt_110` (`N_r0`) | 0.00000 → 0.00001 → 0.00002 → 0.00001 | 245178 → 168208 → 50106 → 17638 | 12.52 → 11.36 → 10.62 → 10.46 | -0.60 → -1.18 → -1.56 → -1.64 | 0.0009 → 0.0010 → 0.0038 → 0.0089 |

### 4.2 Component Ablation of `ckpt_110` (Placement r = 0, Med 1/p and B90 Q90 1/p)

| Dataset | Variant | Description | n = 8 (Med / B90) | n = 12 (Med / B90) | n = 16 (Med / B90) | n = 20 (Med / B90) | n = 20 Med `N_eff` | n = 20 Med `P_top10_w` |
|---|---|---|---|---|---|---|---|---|
| 4-SAT | `Q` | All-to-all QAOA L4 (exact H_C) | 5.78 / 10.26 | 12.85 / 33.15 | 34.11 / 84.89 | 77.58 / 198.43 | 2836.3 | 0.0259 |
| 4-SAT | `G` | All-to-all Global Cost L4 (GNN weights) | 5.27 / 9.12 | 11.46 / 26.45 | 31.67 / 69.69 | 66.08 / 171.90 | 2715.0 | 0.0263 |
| 4-SAT | `QC_block` | Chain-Projected QAOA L4 (block order, no NN) | 5.94 / 10.70 | 38.69 / 189.81 | 450.39 / 7795.50 | 3225.57 / 153982.50 | 2229.8 | 0.0371 |
| 4-SAT | `QC_r0` | Chain-Projected QAOA L4 (min-cut r=0, no NN) | 5.94 / 10.70 | 29.77 / 107.54 | 204.82 / 2809.62 | 1675.86 / 73766.18 | 1947.0 | 0.0396 |
| 4-SAT | `N_unif_w` | `Proj_unif(E_w)` only (`pv=0.5, dS=0, dB=0`) | 5.71 / 9.89 | 29.84 / 96.94 | 163.06 / 3383.60 | 1543.57 / 49183.96 | 2626.8 | 0.0320 |
| 4-SAT | `N_unif_dSdB` | `Proj_unif(E_w) + dS + dB` (`pv=0.5`) | 4.34 / 9.13 | 20.81 / 80.66 | 84.45 / 2215.43 | 775.86 / 44606.49 | 874.5 | 0.0676 |
| 4-SAT | `N_pv_r0` | `Proj_pv(E_w)` only (`dS=0, dB=0`) | 5.71 / 9.89 | 20.93 / 85.86 | 83.68 / 3032.98 | 342.58 / 24768.24 | 1908.6 | 0.0386 |
| 4-SAT | `N_grover_r0` | `Proj_pv(E_w) + grover(dS)` (`corr=0`) | 5.28 / 9.03 | 15.97 / 86.21 | 60.09 / 3028.67 | 250.74 / 23528.02 | 1247.5 | 0.0499 |
| 4-SAT | `N_corr_r0` | `Proj_pv(E_w) + corr(dS, dB)` (`grover=0`) | 4.92 / 9.11 | 13.52 / 104.15 | 44.43 / 3000.93 | 192.80 / 21876.11 | 909.1 | 0.0648 |
| 4-SAT | `N_r0` | Full `ckpt_110` (`Proj_pv(E_w) + corr + grover`) | 4.34 / 9.13 | 12.57 / 102.70 | 35.78 / 3319.66 | 195.18 / 25109.19 | 619.7 | 0.0802 |
| 8-SAT | `Q` | All-to-all QAOA L4 (exact H_C) | 7.33 / 17.72 | 25.45 / 68.94 | 104.63 / 268.60 | 256.77 / 849.54 | 5190.1 | 0.0214 |
| 8-SAT | `G` | All-to-all Global Cost L4 (GNN weights) | 7.62 / 17.22 | 25.88 / 63.67 | 98.03 / 234.59 | 245.78 / 805.65 | 4942.3 | 0.0206 |
| 8-SAT | `QC_block` | Chain-Projected QAOA L4 (block order, no NN) | 8.78 / 18.90 | 157.57 / 2143.78 | 4904.88 / 47550.47 | 77936.78 / 775045.50 | 25012.0 | 0.0068 |
| 8-SAT | `QC_r0` | Chain-Projected QAOA L4 (min-cut r=0, no NN) | 8.78 / 18.90 | 155.80 / 1349.41 | 3642.42 / 49291.23 | 50128.39 / 534327.13 | 15694.3 | 0.0094 |
| 8-SAT | `N_unif_w` | `Proj_unif(E_w)` only (`pv=0.5, dS=0, dB=0`) | 9.36 / 20.46 | 143.72 / 1196.69 | 3603.39 / 82329.92 | 49212.09 / 975563.50 | 10280.3 | 0.0128 |
| 8-SAT | `N_unif_dSdB` | `Proj_unif(E_w) + dS + dB` (`pv=0.5`) | 8.96 / 20.46 | 148.08 / 1272.45 | 6196.47 / 85180.97 | 117057.25 / 2073240.85 | 17555.4 | 0.0081 |
| 8-SAT | `N_pv_r0` | `Proj_pv(E_w)` only (`dS=0, dB=0`) | 9.36 / 20.46 | 143.43 / 1713.03 | 2748.94 / 83685.10 | 61528.84 / 1220837.84 | 9095.6 | 0.0136 |
| 8-SAT | `N_grover_r0` | `Proj_pv(E_w) + grover(dS)` (`corr=0`) | 9.02 / 20.54 | 142.25 / 1651.50 | 2801.45 / 72976.88 | 51158.40 / 1126312.48 | 9210.9 | 0.0142 |
| 8-SAT | `N_corr_r0` | `Proj_pv(E_w) + corr(dS, dB)` (`grover=0`) | 9.24 / 20.38 | 139.77 / 1759.64 | 5061.49 / 114889.97 | 102455.04 / 1785160.31 | 18387.9 | 0.0082 |
| 8-SAT | `N_r0` | Full `ckpt_110` (`Proj_pv(E_w) + corr + grover`) | 8.96 / 20.46 | 143.27 / 1726.09 | 5381.65 / 89753.90 | 96304.55 / 2518346.72 | 17637.7 | 0.0089 |

---

## 5. Across R = 4 Placements: Do Different Variable-to-Cavity Permutations Lock into the Same or Different Wrong States?

Each formula is evaluated across R = 4 variable-to-cavity placements (`r = 0` spectral min-cut + `r = 1, 2, 3` seeded random permutations), with top-10 wrong bitstrings mapped back to the **canonical variable ordering** so they can be compared directly across placements:

| Dataset | n | Family | Med / B90 1/p (`r=0` min-cut) | Med / B90 1/p (`R=4` Round-Robin) | Med / B90 1/p (Best-of-4 `min_r 1/p_r`) | Med Spread `max_r p_r / min_r p_r` | Mean Distinct Top-1 Wrong States (of 4) | Mean Pairwise `d_H(top1_r1, top1_r2)` | Mean Pairwise Top-10 Overlap | Mean Union Size of Top-10 Sets (of 40) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-SAT | 8 | Chain-Proj QAOA (`QC`) | 5.94 / 10.70 | 5.94 / 10.70 | 5.94 / 10.70 | 1.00× | 1.00 / 4 | 0.00 | 100.0% | 10.0 / 40 |
| 4-SAT | 8 | Native `Proj_pv` (`N_pv`) | 5.71 / 9.89 | 5.71 / 9.89 | 5.71 / 9.89 | 1.00× | 1.00 / 4 | 0.00 | 100.0% | 10.0 / 40 |
| 4-SAT | 8 | Native `ckpt_110` (`N_full`) | 4.34 / 9.13 | 4.38 / 9.37 | 4.22 / 8.93 | 1.07× | 1.49 / 4 | 0.92 | 89.4% | 12.0 / 40 |
| 4-SAT | 12 | Chain-Proj QAOA (`QC`) | 29.77 / 107.54 | 35.07 / 112.55 | 21.38 / 44.64 | 4.48× | 3.72 / 4 | 4.12 | 14.0% | 33.0 / 40 |
| 4-SAT | 12 | Native `Proj_pv` (`N_pv`) | 20.93 / 85.86 | 20.39 / 94.49 | 13.51 / 43.48 | 2.84× | 3.69 / 4 | 4.98 | 14.3% | 32.9 / 40 |
| 4-SAT | 12 | Native `ckpt_110` (`N_full`) | 12.57 / 102.70 | 12.76 / 89.02 | 8.55 / 37.51 | 2.67× | 3.35 / 4 | 3.14 | 26.6% | 28.4 / 40 |
| 4-SAT | 16 | Chain-Proj QAOA (`QC`) | 204.82 / 2809.62 | 180.53 / 2122.56 | 85.02 / 717.26 | 15.79× | 3.90 / 4 | 5.17 | 4.5% | 37.6 / 40 |
| 4-SAT | 16 | Native `Proj_pv` (`N_pv`) | 83.68 / 3032.98 | 70.72 / 745.79 | 38.19 / 282.38 | 6.48× | 3.89 / 4 | 5.56 | 6.6% | 36.6 / 40 |
| 4-SAT | 16 | Native `ckpt_110` (`N_full`) | 35.78 / 3319.66 | 43.62 / 871.65 | 24.89 / 300.50 | 6.64× | 3.57 / 4 | 3.28 | 18.6% | 31.7 / 40 |
| 4-SAT | 20 | Chain-Proj QAOA (`QC`) | 1675.86 / 73766.18 | 1864.22 / 21563.85 | 775.92 / 8701.33 | 15.56× | 3.97 / 4 | 6.68 | 1.2% | 39.3 / 40 |
| 4-SAT | 20 | Native `Proj_pv` (`N_pv`) | 342.58 / 24768.24 | 396.79 / 6377.77 | 214.58 / 2549.07 | 11.47× | 3.94 / 4 | 6.76 | 3.3% | 38.1 / 40 |
| 4-SAT | 20 | Native `ckpt_110` (`N_full`) | 195.18 / 25109.19 | 210.04 / 9013.09 | 96.17 / 3292.88 | 9.46× | 3.76 / 4 | 3.96 | 11.7% | 34.4 / 40 |
| 8-SAT | 8 | Chain-Proj QAOA (`QC`) | 8.78 / 18.90 | 8.78 / 18.90 | 8.78 / 18.90 | 1.00× | 1.00 / 4 | 0.00 | 100.0% | 10.0 / 40 |
| 8-SAT | 8 | Native `Proj_pv` (`N_pv`) | 9.36 / 20.46 | 9.36 / 20.46 | 9.36 / 20.46 | 1.00× | 1.00 / 4 | 0.00 | 100.0% | 10.0 / 40 |
| 8-SAT | 8 | Native `ckpt_110` (`N_full`) | 8.96 / 20.46 | 8.98 / 19.77 | 8.84 / 19.18 | 1.04× | 1.17 / 4 | 0.34 | 96.7% | 10.6 / 40 |
| 8-SAT | 12 | Chain-Proj QAOA (`QC`) | 155.80 / 1349.41 | 145.27 / 678.77 | 80.30 / 388.42 | 9.09× | 3.79 / 4 | 4.43 | 11.4% | 34.1 / 40 |
| 8-SAT | 12 | Native `Proj_pv` (`N_pv`) | 143.43 / 1713.03 | 128.73 / 743.42 | 63.17 / 313.82 | 12.11× | 3.76 / 4 | 4.61 | 11.6% | 34.2 / 40 |
| 8-SAT | 12 | Native `ckpt_110` (`N_full`) | 143.27 / 1726.09 | 129.84 / 780.99 | 64.94 / 309.61 | 11.25× | 3.81 / 4 | 4.42 | 12.5% | 33.8 / 40 |
| 8-SAT | 16 | Chain-Proj QAOA (`QC`) | 3642.42 / 49291.23 | 2837.74 / 16713.06 | 1398.66 / 7567.09 | 13.27× | 3.96 / 4 | 6.16 | 3.3% | 38.2 / 40 |
| 8-SAT | 16 | Native `Proj_pv` (`N_pv`) | 2748.94 / 83685.10 | 2272.25 / 27375.22 | 1053.22 / 7336.65 | 17.82× | 3.96 / 4 | 6.01 | 3.7% | 38.0 / 40 |
| 8-SAT | 16 | Native `ckpt_110` (`N_full`) | 5381.65 / 89753.90 | 3072.58 / 42276.07 | 1399.04 / 15616.58 | 21.47× | 3.96 / 4 | 5.67 | 3.2% | 38.2 / 40 |
| 8-SAT | 20 | Chain-Proj QAOA (`QC`) | 50128.39 / 534327.13 | 39512.74 / 413155.02 | 18526.37 / 182936.95 | 14.72× | 4.00 / 4 | 7.47 | 0.7% | 39.6 / 40 |
| 8-SAT | 20 | Native `Proj_pv` (`N_pv`) | 61528.84 / 1220837.84 | 26211.20 / 614121.45 | 9762.93 / 217057.54 | 21.69× | 3.99 / 4 | 7.46 | 1.1% | 39.4 / 40 |
| 8-SAT | 20 | Native `ckpt_110` (`N_full`) | 96304.55 / 2518346.72 | 66704.83 / 999636.19 | 26179.02 / 418269.98 | 33.27× | 4.00 / 4 | 7.19 | 0.6% | 39.6 / 40 |

### Key Takeaways from the R = 4 Placement Analysis
1. **In Pure Chain-Projected QAOA (`QC`) and `N_pv`, different placements lock into completely different ghost minima:** At n = 20, the 4 placements produce `3.97 / 4` distinct top-1 wrong states (`pairwise d_H = 6.68` on 4-SAT, `7.47` on 8-SAT), with only `1.2%` (`4-SAT`) and `0.7%` (`8-SAT`) overlap between their top-10 wrong-state sets (`39.3 / 40` and `39.6 / 40` distinct states in the union). Because each permutation cuts a different subset of clauses, its ghost minima sit in different parts of the hypercube.
2. **In Native `ckpt_110` on 4-SAT, the placement-invariant GNN bias `pv` pulls all 4 placements toward the SAME wrong cluster on hard instances:** Because `pv` depends only on the variable-clause graph (not the cavity permutation), `ckpt_110`'s top-1 wrong states across the 4 placements have **10× higher top-10 overlap** at n = 20 (`11.7%` vs `1.2%` for `QC`, with pairwise `d_H = 3.96` vs `6.68`). When `pv` is right (easy/median formulas), all 4 placements reinforce the true basin; when `pv` is wrong (the B90 tail), all 4 placements get dragged toward the same `x_pv` false attractor, explaining why `R=4` round-robin cannot rescue the B90 tail on 4-SAT (`B90 = 9,013` for `N_full` vs `6,030` for `N_pv` at n = 20).
