# oct_explore log (times CST)

Problem: random 4-SAT, r=9.93, m=round(9.93n), 4 distinct vars, random signs (F-sat generator of
noiseless/sat4_poisson_run.py), all satisfiable instances kept, success p = Born prob on all E=0 strings.
Seeds: train 3.1e10+1e7n+k (100 inst at n=8), test 3.2e10+1e7n+k (100 at n=8,10,12,20; 200 at n=14).
Metric: median and 90th percentile (B90) of 1/p over test instances. Random guess = 2^n/#solutions.
Code: `sim.py` (JAX simulator), `train.py` (train n=8 with layer growth, freeze, test), `qaoa_ref.py` (QAOA reference), `report.py`.

Circuit: registers = (transmon, cavity) pairs, no coupling between them. Per layer, parameters SHARED by all
cavities (so the same vector runs on 2 or 3 cavities): transmon R(theta,phi) -> ECD(beta_c) with
beta_c=(bre+i bim)*sqrt(2^k_c) -> exp(-i gamma C(x)) on the joint computational space.
'raw' cost C = E*n/m (density-normalised clause count), 'std' C = (E-mean)/std. Gray-coded Fock levels.
Start: uniform superposition over the computational space (QAOA analogue; vacuum start tested separately).
Cavity truncation 2^k + max(8, 2^k/4) levels; checked: padding to +64 changes median 1/p by <1e-5 (n=8, 14).
Training: L-BFGS on exact JAX gradients, L = 1,2,3,4,6,8 grown from the previous L (+ interpolated + random restarts).

## Exp 0 (17:40): objective = mean p (as specified) -> degenerate
Mean-p training makes the circuit dump all probability on instances where it is easy: train mean p rises
(0.11 at L=8) but median 1/p ~ 1e5-1e10 (half the instances get p~0). Unusable for median/B90. Switched to
the mean of log p (geometric mean), which is what the median metric rewards. Results kept in results/meanp_obj/.

## Step 0 + Step A (17:46-18:11): logp objective
Median / B90 of 1/p, params trained at n=8 only, frozen:

| run | L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|---|
| T_raw (transmon bit) | 4 | 32/93 | 109/578 | 742/5943 | 644/3050 | 288600/2.2e6 |
| T_raw | 8 | 17/45 | 60/203 | 258/1391 | 245/1079 | 55599/824795 |
| T_std | 8 | 16/46 | 48/308 | 327/3694 | 208/888 | (not run, OOM) |
| C_raw (cavity only 4/4) | 4 | 24/96 | 83/659 | 836/6235 | - | - |
| C_raw | 8 | 14/42 | 50/268 | 361/3999 | - | - |
| QAOA (X mixer, same training) | 4 | 5/10 | 8/20 | 13/33 | 21/60 | 85/238 |
| QAOA | 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| QAOA | 14 | 1.6/3.6 | 2.0/5.7 | 2.6/12.9 | 3.5/17.1 | 7.0/92.3 |
| random guess | - | 128/256 | 512/1024 | 2048/4096 | 8192/16384 | 349525/1.05e6 |
| SATLike (F-sat) | - | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 (fit) |

Findings:
- Step 0: transmon-bit layout is NOT clearly worse than cavity-only (n=8 L=8 17 vs 14; n=12 258 vs 361).
  Keep the transmon-bit layout (n=20 fits under the 6-bit cap).
- raw vs std cost normalisation: no meaningful difference.
- Transfer loss is small: training directly at n=14 (40 inst) gives L=2 median 1541 vs 1704 transferred.
  So the problem is not transfer, it is expressivity of the ECD mixer.
- The ECD circuit is much weaker than QAOA: ~10x at n=8 L=8, 40x at n=14, ~3000x at n=20 (near random).
- Trained displacements are tiny (|b|~0.05); the optimiser mostly uses the transmon rotations, i.e. it mixes
  the transmon bits and barely the cavity bits. Displacement hops n<->n+1 with amplitude ~beta*sqrt(n): very
  inhomogeneous across Fock levels and not a Hamming-local bit-flip mixer, so the cavity bits are poorly explored.
  The more bits sit in cavities (n=20: 17/20), the closer to random.

## Next (18:12): mixer expressivity probes at n=8 -> 12,14 (L=1,2,4,6)
E_bin (binary code), E_vac (vacuum start), E_snap2 (per-layer size-free cavity SNAP phase
phi(n)=s1 u + s2 u^2, u=n/sqrt(2^k), part of the allowed arbitrary diagonal, instance independent).

### Results (18:18): median/B90 1/p, train n=8
| run | L | n=8 | n=12 | n=14 |
|---|---|---|---|---|
| E_bin (binary code) | 6 | 22/70 | 755/4940 | 394/3521 |
| E_vac (vacuum start) | 6 | 26/54 | 2389/33923 | 609/6285 |
| E_snap2 (+cavity SNAP) | 6 | 15/29 | 1512/14065 | 530/3599 |
| T_raw (baseline, Gray, uniform) | 6 | 19/45 | 408/1828 | 358/1722 |
- Gray >= binary, uniform start > vacuum: keep Gray + uniform.
- SNAP helps at n=8 (L=2: 32 vs 52) but transfers worse: it overfits the k=3 cavity. Next: train SNAP jointly
  on n=8+10 (two cavity sizes, forces size-free SNAP) and a deeper baseline (L up to 16) to see whether the ECD
  circuit keeps improving with depth at n=14.

## Finished jobs (18:36–18:48 CST)

### Direct train at n=14 (T_raw_direct14)
| L | transferred from n=8 | trained at n=14 |
|---|---|---|
| 1 | 7287/18825 | 7475/18218 |
| 2 | 1704/5541 | 1913/4500 |
| 4 | 644/3050 | 462/2427 |

Transfer loss ≤1.4× at L=4. Expressivity, not transfer, is the bottleneck. Depth alone (T_raw_deep) only gets
n=14 L=16 to 195/2298 — still 4× SATLike median (50) and 30× QAOA.

### Joint SNAP n=8+10 (E_snap2_8and10)
n=14 L=6: 276/957 (vs baseline L=8 245/1079). Small help, not enough.

### Gray-bit X mixer upper bound (BF_gray) — KEY RESULT
Mixer = transmon R_x(β_tm) on every register + exact ∑_j X_j on Gray-coded cavity bits
(shared β_cav), then cost phase. Same training/test protocol. This is what ECD *should*
approximate if it mixed the cavity hypercube.

| L | n=8 | n=10 | n=12 | n=14 | n=20 | vs SATLike n=20 |
|---|---|---|---|---|---|---|
| 4 | 5.9/12 | 10.5/31 | 15/50 | 26/85 | 123/374 | ≈1× median |
| 6 | 3.6/8 | 5.3/17 | 7.5/30 | 11.5/47 | 38/174 | 3× better |
| 8 | 2.4/6 | 3.5/10 | 4.8/23 | 7.0/36 | **17/132** | **7× better** |
| QAOA L=8 | 2.5/5 | 3.5/7 | 4.6/16 | 6.3/18 | 16/54 | — |
| ECD L=8 | 17/45 | 60/203 | 258/1391 | 245/1079 | 55599/825k | 450× worse |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 | — |

Conclusion: a Hamming-local cavity mixer transfers like QAOA and beats SATLike at n=20.
ECD fails because a displacement only walks the Gray *path* (Fock n↔n±1), mixing time
O(2^k), not the hypercube (mixing O(k)). Next: (1) rebalance to k=1 cavities where ECD
≈ bit flip; (2) synthesize Gray X via ECD+SNAP if rebalance isn't enough.

## Running (18:44–)
R_raw: layout R = more registers, k=1 at n=8 (4×(1+1)), k=2 at n=12/14, k=4 at n=20.

## Mixer probes continued (18:48–19:16 CST)

### T_bmax (force |b|≤1.5 via tanh): no gain
Trained |b| still collapses to 0.02–0.35. Test n=14 L=8: 333/1355 (vs T_raw 245/1079).
Optimizer does not want large ECD displacements.

### E_snap4 joint 8+10: worse
n=14 L=8: 1441/7831. Heavy SNAP hurts; discarded.

### R layout (more registers, fewer cavity bits) — in progress
Fixed bit-count bug (n=12/14 layouts). Partial from earlier valid n=8/10 run (k=1 cavities):
n=8 L=2: 33/75 (vs T 52/126); n=10 L=2: 103/223 (vs T 236/624). Rebalancing helps at
small n. Full R_raw restarted 19:16 CST with corrected layouts.

### BF_gray remains the only method matching QAOA
n=20 L=8 median 17 vs SATLike ~124 (7× better), vs ECD 55599.
Interpretation: need Hamming-local cavity mixer (Gray X), synthesizable by ECD+SNAP/GRAPE
as a compiled block; plain ECD displacement is the wrong mixer.

## Hardware model lock-in (19:27 CST) — theory first

Zach: each layer = **local ECD** on each cavity–transmon + an **arbitrary** (not just
diagonal) unitary on the **joint cavity space** via GRAPE. No JP. Bits may sit on cavities
and/or control transmons. Find the best transferring unitary first; make it GRAPE-friendly later.

### Roles of each operation
- **ECD alone** only mixes each cavity with its own transmon (conditional displacement:
  Fock walk n↔n±1). Mixing time on a k-bit cavity is O(2^k) along the Gray path, not O(k)
  on the hypercube. ECD cannot couple cavities to each other.
- **Joint cavity unitary (GRAPE)** is the *only* place that can (i) Hamming-mix cavity bits,
  (ii) mix across cavities, and (iii) apply the cost. Theory target used here:
  product of local Gray-X on each cavity (QAOA-analogue hypercube mixer) × cost diagonal D(C).
- **ECD is optional** once GRAPE owns cavity mixing. Useful ECD roles left: entangle / move
  information involving **transmon bits**, or cheap local prep. Ablate ECD on/off to check.
- Plain ECD-as-mixer fails (T_raw n=20 L=8 median 55599). Exact Gray-X + cost (BF_gray)
  matches QAOA and beats SATLike (n=20 L=8 median **17** vs ~124).

### Next experiments (`theory_u.py`)
1. **U_ecd_raw**: ECD on + Gray-X joint U + raw cost (does ECD help when Gray-X is exact?).
2. **U_poly2 / U_soft / U_step**: ECD off, stronger size-free diagonals in U (L focus ≈4–8).
3. Keep **R_raw** (rebalanced k) running as ECD-only baseline with more transmon bits.
4. Killed `fit_ecd_bitflip` (ECD+SNAP ≈ Gray-X synthesis): low fidelity (~0.3), GRAPE-friendly
   path deferred until theory unitary is settled.

Code: `theory_u.py`. Detached setsid/nohup, ≤8 cores, never touch `*.STOP`.

## Theory unitary interim (19:53 CST)

Roles confirmed by ablation:
- **Joint Gray-X + raw cost** (BF_gray / U_ecd_raw) transfers like QAOA.
- **ECD on top of exact Gray-X** neither helps nor hurts much at n=8–14 (ECD optional
  when GRAPE owns cavity mixing; keep ECD only if transmon bits need it).
- **Stronger diagonal poly2 OVERFITS**: n=8 L=8 median 1.8 (best yet) but n=14 L=8
  3040 and n=20 L=4 already 61k — nonlinear Cn^2 fits n=8 and does not transfer.
  Raw cost D=γ Cn remains the transferable diagonal. Soft/step next; if they overfit
  too, stick to raw. Joint-train 8+10 only if a size-free nonlinear form is found.

### U_poly2 (partial 19:53 CST)
| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 128.0/256.0 | 512.0/1024.0 | 2048.0/4096.0 | 8192.0/16384.0 | 349525.5/1048576.2 |
| 2 | 17.1/32.7 | 41.0/95.2 | 110.1/241.1 | 330.2/922.6 | 13726.5/42363.2 |
| 4 | 4.2/8.9 | 9.8/24.0 | 34.7/160.5 | 260.3/2174.0 | 61602.2/467728.2 |
| 6 | 2.4/5.0 | 5.9/14.1 | 48.4/330.8 | 1015.5/17653.8 | 282876.2/2081991.4 |
| 8 | 1.8/3.6 | 4.4/12.8 | 75.4/770.5 | 3040.5/42312.3 | - |

### U_ecd_raw (partial 19:53 CST)
| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 137.6/267.8 | 538.3/1168.2 | 2216.5/4676.6 | 7317.5/17758.8 | 423503.6/1691140.2 |
| 2 | 20.7/38.0 | 44.9/112.2 | 97.9/209.9 | 204.7/503.4 | 2109.4/6856.9 |
| 4 | 6.4/13.5 | 12.5/38.7 | 23.3/68.6 | 31.6/110.8 | - |
| 6 | 3.7/7.4 | 5.2/16.6 | 7.7/28.6 | 11.7/47.3 | - |
| 8 | 2.5/6.1 | 3.5/9.9 | 5.3/20.0 | 6.9/36.3 | - |


Compare: BF_gray L=8 n=14=7.0/36, n=20=17/132; QAOA L=8 n=20=16/54; SATLike n=20~124/~530.
