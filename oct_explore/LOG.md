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

## U_poly2 finished (19:54 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 128.0/256.0 | 512.0/1024.0 | 2048.0/4096.0 | 8192.0/16384.0 | 349525.5/1048576.2 |
| 2 | 17.1/32.7 | 41.0/95.2 | 110.1/241.1 | 330.2/922.6 | 13726.5/42363.2 |
| 4 | 4.2/8.9 | 9.8/24.0 | 34.7/160.5 | 260.3/2174.0 | 61602.2/467728.2 |
| 6 | 2.4/5.0 | 5.9/14.1 | 48.4/330.8 | 1015.5/17653.8 | 282876.2/2081991.4 |
| 8 | 1.8/3.6 | 4.4/12.8 | 75.4/770.5 | 3040.5/42312.3 | 468511.5/3850968.2 |

Key: n=14 L=4: 260.3/2174.0; n=20 L=4: 61602.2/467728.2; n=14 L=8: 3040.5/42312.3; n=20 L=8: 468511.5/3850968.2

## U_ecd_raw finished (20:02 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 137.6/267.8 | 538.3/1168.2 | 2216.5/4676.6 | 7317.5/17758.8 | 423503.6/1691140.2 |
| 2 | 20.7/38.0 | 44.9/112.2 | 97.9/209.9 | 204.7/503.4 | 2109.4/6856.9 |
| 4 | 6.4/13.5 | 12.5/38.7 | 23.3/68.6 | 31.6/110.8 | 610.5/6259.7 |
| 6 | 3.7/7.4 | 5.2/16.6 | 7.7/28.6 | 11.7/47.3 | 47.3/194.0 |
| 8 | 2.5/6.1 | 3.5/9.9 | 5.3/20.0 | 6.9/36.3 | 21.3/169.6 |

Key: n=14 L=4: 31.6/110.8; n=20 L=4: 610.5/6259.7; n=14 L=8: 6.9/36.3; n=20 L=8: 21.3/169.6

## Theory conclusion (20:08 CST)

Best transferring unitary under the hardware model (local ECD + arbitrary joint cavity U):

| circuit | L | n=8 | n=14 | n=20 | vs SATLike n=20 |
|---|---|---|---|---|---|
| **Gray-X + raw cost (BF_gray, ECD off)** | 4 | 5.9/12 | 26/85 | **123/374** | ≈1× |
| **Gray-X + raw cost** | 8 | 2.4/6 | 7/36 | **17/132** | **7× better** |
| ECD on + Gray-X + raw (U_ecd_raw) | 8 | 2.5/6 | 6.9/36 | 21/170 | 6× better |
| QAOA L=8 | 8 | 2.5/5 | 6.3/18 | 16/54 | — |
| poly2 / soft diagonals | 8 | 1.5–1.8 | 600–3000 | >1e5 | overfit, fail |
| plain ECD mixer (T_raw) | 8 | 17/45 | 245/1079 | 55599 | fail |

**Takeaways**
1. Joint cavity U = ∏_cav exp(-i β/2 ∑_j X_j^Gray) · exp(-i γ C) is the theory target. It
   matches QAOA and beats SATLike at n=20. Make this GRAPE-friendly next (not ECD+SNAP per cavity).
2. ECD is optional once that U owns mixing: U_ecd_raw ≈ BF_gray (n=20 L=8: 21 vs 17). Use ECD
   only for transmon-bit entanglement/prep if needed.
3. Stronger size-free diagonals (poly2, soft) improve n=8 train but **destroy transfer**. Stick
   to raw D=γ·(E n/m). Joint 8+10 poly2 still running as a last check.
4. Focus L≈4 already competitive with SATLike at n=20 (median 123 ≈ 124); L=8 is clearly better.

Next: U_cav (cavity-only bits, ECD off), U_L4deep (more restarts at L≤4), resume R_raw, then GRAPE compile plan.

## U_soft finished (20:14 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 128.0/256.0 | 512.0/1024.0 | 2048.0/4096.0 | 8192.0/16384.0 | 349525.5/1048576.2 |
| 2 | 16.8/31.9 | 41.3/91.3 | 108.8/245.9 | 341.6/900.7 | 20997.7/85456.8 |
| 4 | 4.2/8.6 | 9.7/21.9 | 38.9/157.2 | 300.5/1750.0 | 158145.1/807230.5 |
| 6 | 2.1/4.2 | 6.1/18.5 | 53.1/256.0 | 552.3/1923.4 | 137009.3/419532.5 |
| 8 | 1.5/2.6 | 5.5/22.5 | 69.5/249.2 | 605.1/2246.5 | 111593.4/342554.2 |

## U_step finished (20:14 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 128.0/256.0 | 512.0/1024.0 | 2048.0/4096.0 | 8192.0/16384.0 | 349525.5/1048576.2 |
| 2 | 17.6/34.3 | 48.0/103.7 | 145.9/342.2 | 516.8/1527.6 | 70220.2/362383.1 |
| 4 | 4.2/8.4 | 10.4/24.1 | 48.4/200.2 | 426.1/2094.4 | 141637.8/562705.9 |
| 6 | 2.1/4.0 | 6.4/18.0 | 49.2/206.6 | 394.3/1326.2 | 77556.7/259120.0 |
| 8 | 1.7/3.1 | 5.1/16.4 | 42.9/156.6 | 282.0/985.3 | 46449.5/161081.6 |

## U_poly2_810 finished (20:21 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 17.3/33.3 | 38.0/89.7 | 91.1/198.2 | 252.4/671.6 | 8610.2/25465.7 |
| 4 | 4.5/9.4 | 8.1/19.1 | 15.6/52.7 | 58.4/262.8 | 23201.1/207598.2 |
| 6 | 2.6/5.3 | 3.9/9.3 | 8.3/36.5 | 50.7/320.6 | 87379.0/455586.0 |
| 8 | 1.9/3.7 | 2.4/6.2 | 6.2/31.0 | 74.4/538.4 | 159599.6/898223.2 |

## U_poly2_810 finished (20:23 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 17.3/33.3 | 38.0/89.7 | 91.1/198.2 | 252.4/671.6 | 8610.2/25465.7 |
| 4 | 4.5/9.4 | 8.1/19.1 | 15.6/52.7 | 58.4/262.8 | 23201.1/207598.2 |
| 6 | 2.6/5.3 | 3.9/9.3 | 8.3/36.5 | 50.7/320.6 | 87379.0/455586.0 |
| 8 | 1.9/3.7 | 2.4/6.2 | 6.2/31.0 | 74.4/538.4 | 159599.6/898223.2 |

## U_L4deep finished (20:23 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 1 | 128.0/256.0 | 512.0/1024.0 | 2048.0/4096.0 | 8192.0/16384.0 | 349525.5/1048576.2 |
| 2 | 20.1/39.0 | 43.8/113.1 | 91.8/199.4 | 200.4/536.2 | 2155.0/5261.5 |
| 3 | 9.1/19.0 | 18.5/55.3 | 28.8/84.5 | 54.6/168.7 | 336.1/971.0 |
| 4 | 5.9/12.2 | 10.5/30.8 | 15.1/49.5 | 25.9/85.2 | 122.7/373.6 |

## U_cav partial (in progress)

| L | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| 1 | 128.0/256.0 | 512.0/1024.0 | 2048.0/4096.0 | 8192.0/16384.0 |
| 2 | 20.0/39.1 | 43.7/113.1 | 91.0/200.5 | 199.5/533.4 |
| 4 | 5.9/12.4 | 10.4/30.8 | 15.3/49.6 | 25.9/84.6 |
| 6 | 3.6/7.7 | 5.1/15.8 | 7.6/29.9 | 11.4/45.1 |
| 8 | 2.5/6.0 | 3.4/9.8 | 4.8/23.8 | 6.8/36.1 |

## Broad scan (20:23 CST) — explore MANY options before locking

Zach update: keep ECD every layer; one joint cavity U per layer (mixer+cost folded in).
Hardware: local ECD + arbitrary joint cavity unitary (GRAPE later). Theory first.

`theory_u.py` extended with `--u_form/--u_order/--code/--mixer`. Short screens L=2,4,8.

| tag | what |
|---|---|
| S_gxcost | ECD + U=GrayX·cost (order A; ≈ U_ecd_raw) |
| S_costgx | ECD + U=cost·GrayX (order B) |
| S_sexp | ECD + U=exp(-i(aX+bC)) single generator (test≤12) |
| S_gxonly | ECD + U=GrayX only (no cost) ablation |
| S_costonly | ECD + U=cost only (no GrayX) ablation |
| S_bin | ECD + U=binary-X·cost |
| S_joint | ECD + U=joint concatenated-bit GrayX·cost |
| S_R | layout R (more regs / fewer cav bits) + ECD + GrayX·cost |

Still finishing: U_cav, R_raw (ECD-only R baseline). Soft/poly2 confirmed overfit.
Do not lock onto one circuit yet — wait for compact comparison table vs SATLike.

## S_gxcost finished (20:39 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 20.7/38.0 | 44.9/112.2 | 97.9/209.6 | 204.7/502.9 | 2109.1/6859.4 |
| 4 | 5.9/12.4 | 10.9/32.4 | 16.7/60.2 | 26.3/92.3 | 131.6/447.3 |
| 8 | 2.5/6.1 | 3.5/10.0 | 5.0/22.8 | 6.9/34.8 | 20.0/164.0 |

## Broad-scan highlight (20:39 CST): S_gxcost + schedule effect

ECD + U=GrayX·cost (product, order gx_cost), train L=2→4→8 (no L=1/6):

| L | n=8 | n=14 | n=20 | vs SATLike n=20 |
|---|---|---|---|---|
| 4 | 5.9/12 | 26/92 | **132/447** | ≈1× |
| 8 | 2.5/6 | 6.9/35 | **20/164** | **6× better** |

Key: earlier U_ecd_raw L=4 n=20 was **611** (same circuit, but grown via L=1,2,4,6,8).
Skipping L=1/6 lands L=4 near BF_gray (123). Schedule/init matters for ECD+GrayX.
S_costgx (order cost·GrayX) training; other S_* queued. Not locking yet.

## S_costgx finished (20:59 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 10.8/23.0 | 22.4/58.4 | 36.6/110.9 | 75.0/208.3 | 490.8/1688.3 |
| 4 | 4.8/9.9 | 8.1/22.5 | 11.7/44.9 | 18.2/69.4 | 79.0/416.4 |
| 8 | 2.2/5.1 | 3.2/8.9 | 4.5/18.8 | 6.0/32.1 | 17.1/128.6 |

## Decision from data (21:11 CST) — prefer cost·GrayX; add SNAP

**S_costgx beats S_gxcost** (same ECD+GrayX+raw, only U order flipped):

| run | L | n=8 | n=14 | n=20 | vs SATLike n=20 |
|---|---|---|---|---|---|
| **S_costgx** (cost·GrayX) | 4 | 4.8/10 | 18/69 | **79/416** | **1.6× better** |
| **S_costgx** | 8 | 2.2/5 | 6.0/32 | **17/129** | **7× better** |
| S_gxcost (GrayX·cost) | 4 | 5.9/12 | 26/92 | 132/447 | ≈1× |
| S_gxcost | 8 | 2.5/6 | 6.9/35 | 20/164 | 6× |
| BF_gray (no ECD) | 8 | 2.4/6 | 7/36 | 17/132 | 7× |
| SATLike | — | 16/70 | 50/208 | ~124/~530 | — |

Ablations so far: **S_gxonly** (no cost) ≈ random — cost phase required.
S_sexp abandoned (NaN / non-transferring). Soft/poly2 still out.

**Next (data-driven):** keep ECD; default U order = cost·GrayX; screen binary/joint/R;
add **SNAP-per-cavity + joint mixer U** (`--snap 2/4`, local and joint mixer): S_snap2, S_snap2j, S_snap4.

## S_gxonly finished (21:15 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 150.1/379.5 | 875.8/20052.5 | 2221.0/15242.3 | 10054.3/164839.3 | 501083.7/4502829.0 |
| 4 | 151.5/368.5 | 1075.7/5929.5 | 3657.0/42632.9 | 9396.9/81914.4 | 880049.6/20089352.0 |
| 8 | 140.9/409.0 | 676.8/12140.5 | 2577.1/23331.6 | 11093.0/142565.5 | 1184760.5/53469548.0 |

## S_costonly finished (21:23 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 52.9/131.2 | 244.3/636.6 | 1057.1/4456.0 | 1667.3/5764.7 | 120230.3/5458849.0 |
| 4 | 32.4/92.4 | 122.6/504.2 | 634.8/3353.4 | 707.0/3196.4 | 182438.2/1747960.4 |
| 8 | 16.8/43.9 | 61.6/306.0 | 422.0/2266.3 | 286.2/1091.0 | 72457.9/557709.6 |

## S_bin finished (21:31 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 10.8/23.0 | 22.6/58.3 | 36.0/103.7 | 73.9/202.9 | 448.4/1525.0 |
| 4 | 4.8/9.8 | 8.1/23.3 | 11.4/40.9 | 18.3/70.1 | 75.4/354.0 |
| 8 | 2.2/5.2 | 3.1/9.0 | 4.5/26.7 | 6.2/36.5 | 24.6/409.8 |

## S_joint crashed at n=20 test (21:37 CST) — dropped as redundant

Joint mixer (X on the concatenated cavity bits) OOM'd building a dense 2^17 matrix (256 GiB) for n=20.
It is mathematically identical to the local mixer: sum_j X_j is separable across cavities, so
exp(-i b/2 sum X) = product of per-cavity exps. Data agrees — S_joint vs S_costgx (med 1/p / B90):

| L | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| 8 joint | 2.2/5.0 | 3.1/8.9 | 4.6/18.8 | 5.9/32.3 |
| 8 local (S_costgx) | 2.2/5.1 | 3.2/8.9 | 4.5/18.8 | 6.0/32.1 |

So S_joint and S_snap2j are removed from the queue; `theory_u.py` now only builds the dense joint
mixer for ≤14 cavity bits. A truly non-separable joint mixer would need cross-cavity terms.

**Next (data-driven):** L 4→8 cut n=20 median 1/p 79→17 (S_costgx), the biggest single lever so far,
so queued **S_deep** (same S_costgx recipe, warm-grown L=8→12→16). Watchdog v6 restarted.

## S_snap2 finished (22:03 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 10.8/22.9 | 22.7/59.1 | 36.7/107.6 | 74.5/209.2 | 504.2/1694.2 |
| 4 | 4.8/9.8 | 8.2/23.4 | 14.8/57.5 | 18.6/70.8 | 230.3/1638.3 |
| 8 | 2.2/4.9 | 3.1/9.9 | 4.9/24.5 | 6.2/36.1 | 42.1/311.2 |

## Pivot (23:20 CST): native qumode U — Gray-X is reference only

Zach: less QAOA-like Gray-X; prefer ECD + SNAP + displacement/phase.
Joint cavity U from SNAP/displacement generators (not Pauli-X). Goal: transfer ≈ SATLike.

Gray-X reference (do not chase further):
| run | L | n=20 | vs SATLike |
|---|---|---|---|
| **S_costgx** (ECD+cost·GrayX) | 8 | **17/129** | 7× better |
| S_snap2 (SNAP+GrayX) | 8 | 42/311 | SNAP on GrayX hurts |
| S_bin | 8 | 25/410 | Gray≈binary |
| S_costonly (no mixer) | 8 | 72k | fail |

New code: `native_u.py`. Forms: d_snap, snap_d, d_snap_d, snap_cost, gen_sum, bs_snap.
Queue N_*: no Gray-X. Watchdog v7. Gray-X deep/snap4/R killed to free cores.

## S_costgx finished (23:22 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 10.8/23.0 | 22.4/58.4 | 36.6/110.9 | 75.0/208.3 | 490.8/1688.3 |
| 4 | 4.8/9.9 | 8.1/22.5 | 11.7/44.9 | 18.2/69.4 | 79.0/416.4 |
| 8 | 2.2/5.1 | 3.2/8.9 | 4.5/18.8 | 6.0/32.1 | 17.1/128.6 |

## S_snap2 finished (23:22 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 10.8/22.9 | 22.7/59.1 | 36.7/107.6 | 74.5/209.2 | 504.2/1694.2 |
| 4 | 4.8/9.8 | 8.2/23.4 | 14.8/57.5 | 18.6/70.8 | 230.3/1638.3 |
| 8 | 2.2/4.9 | 3.1/9.9 | 4.9/24.5 | 6.2/36.1 | 42.1/311.2 |

## S_bin finished (23:22 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 10.8/23.0 | 22.6/58.3 | 36.0/103.7 | 73.9/202.9 | 448.4/1525.0 |
| 4 | 4.8/9.8 | 8.1/23.3 | 11.4/40.9 | 18.3/70.1 | 75.4/354.0 |
| 8 | 2.2/5.2 | 3.1/9.0 | 4.5/26.7 | 6.2/36.5 | 24.6/409.8 |

## S_costonly finished (23:22 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 52.9/131.2 | 244.3/636.6 | 1057.1/4456.0 | 1667.3/5764.7 | 120230.3/5458849.0 |
| 4 | 32.4/92.4 | 122.6/504.2 | 634.8/3353.4 | 707.0/3196.4 | 182438.2/1747960.4 |
| 8 | 16.8/43.9 | 61.6/306.0 | 422.0/2266.3 | 286.2/1091.0 | 72457.9/557709.6 |

## N_dsnap finished (23:48 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 31.9/64.9 | 136.8/499.7 | 1094.3/13953.8 | 658.5/2515.9 | 187018.0/3651962.8 |
| 4 | 13.2/25.9 | 48.6/190.2 | 355.5/6744.5 | 157.6/833.9 | 205849.5/3007296.0 |
| 8 | 7.4/20.2 | 26.5/168.1 | 308.2/7657.9 | 83.2/935.9 | 135013.9/2789492.8 |

## N_snapd finished (23:48 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 31.7/64.3 | 138.5/492.6 | 1119.5/12678.8 | 642.8/2347.6 | 179461.5/5026711.0 |
| 4 | 13.4/27.1 | 46.8/179.8 | 367.7/5603.5 | 157.3/821.9 | 168118.1/4426818.0 |
| 8 | 7.2/20.8 | 26.3/168.8 | 309.5/3875.8 | 85.4/1083.2 | 156554.6/2434791.5 |

## N_dsnapd finished (23:55 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 29.6/64.6 | 130.3/421.5 | 1615.0/11148.4 | 650.4/2098.8 | 183483.4/1475564.9 |
| 4 | 11.5/27.2 | 40.9/164.3 | 635.9/4686.6 | 152.5/699.3 | 273976.0/5214825.0 |
| 8 | 6.2/16.6 | 21.3/85.5 | 275.4/3361.4 | 65.3/307.5 | 166909.3/2036027.1 |

## N_snapcost finished (00:07 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 54.7/123.6 | 240.2/659.0 | 961.8/3658.9 | 1700.0/5974.7 | 159141.8/3154740.5 |
| 4 | 30.9/93.9 | 140.0/590.8 | 951.9/7547.6 | 770.0/4027.5 | 403853.0/3797509.5 |
| 8 | 18.6/49.6 | 112.8/433.6 | 1131.0/9935.4 | 631.6/2750.5 | 346198.0/2414298.2 |

## N_gensum finished (01:02 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 37.7/98.5 | 185.6/557.0 | 1698.6/8250.8 | 895.1/2872.1 | 418009.6/6725540.0 |
| 4 | 14.0/27.4 | 63.9/262.3 | 419.0/2906.9 | 199.2/1026.0 | 109922.3/786212.9 |
| 8 | 6.7/16.6 | 35.1/138.9 | 341.0/9144.0 | 77.6/366.3 | 103155.3/1057179.1 |

## N_dsnap4 finished (01:47 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 34.9/68.0 | 227.8/1253.1 | 1385.0/13114.5 | 1263.7/8509.9 | 321809.9/6908660.0 |
| 4 | 15.2/32.0 | 112.8/857.9 | 739.7/13458.5 | 440.4/6379.6 | 287191.8/5389980.5 |
| 8 | 7.2/23.7 | 93.5/970.7 | 918.9/12570.6 | 317.3/4563.5 | 425331.4/2907537.2 |

## N_dsnapd4 finished (04:58 CST)

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 32.4/68.4 | 163.7/641.3 | 1828.6/14560.3 | 874.4/3306.7 | 2810032.0/132729576.0 |
| 4 | 30.3/68.0 | 295.4/3810.8 | 3702.2/25811.2 | 1516.3/18328.1 | 944079.2/10642523.0 |
| 8 | 40.3/222.9 | 516.8/8389.7 | 2871.1/38731.9 | 3109.9/45367.7 | 1098072.2/7804058.0 |

## Check-in 05:50 CST Sat — native forms summary + box freeze

Native joint U (no Gray-X), n=20 best median/B90 1/p at L=8: N_gensum 103k/1.06M, N_dsnap 135k, N_snapd 157k,
N_dsnapd 167k, N_snapcost 346k, N_dsnap4 425k, N_dsnapd4 1.1M (gets worse with L, training fails).
SATLike ~124/~530; Gray-X ref S_costgx 17/129; cost-only (no mixer) 72k. So every native form so far is
~1000× worse than SATLike at n=20 and no better than having no mixer. snap=4 hurts vs snap=2.

**Box freeze found:** the box is suspended whenever no agent turn is active. Watchdog `sleep 120` loops
took 38–48 min; N_dsnap_R (launched 01:00) has only finished 3 n=8 L=2 restarts by 05:49 (N_dsnap did 6 in 46 s);
process start times show ~4h20m of the last ~4h50m was frozen. Jobs only advance during check-ins.
Still running: N_dsnap_R, N_bssnap (last two in native queue).

## QAOA reference cross-check vs PRX Quantum 5, 030348 (Sat Oct 17, 08:50 CST)

Ran the exact `qaoa_ref.make()` circuit (the code path that produced QAOA_raw; `qaoa_paper_check.py` execs it without training)
on qaoa_ksat's saved random 8-SAT instances (paper ensemble: Poisson m, r=176.54, literals with replacement, SAT only) with
the published angles. Convention map: paper layer exp(+i γ H/2)·Rx(β), Rx=exp(-iβX/2) ⇒ ours b=β, g=-γ/2 on Cn=E.

| n | p | N | oct_explore mean p / med TTS | qaoa_ksat mean p / med TTS | paper | max per-instance |Δp| |
|---|---|---|---|---|---|---|
| 8 | 14 | 40 | 0.3688 / 2.81 | 0.3688 / 2.81 | Table II fit 2.92 (extrap.) | 4e-7 |
| 12 | 14 | 40 | 0.1765 / 6.10 | 0.1765 / 6.10 | Table II 7.20 | 2e-7 |
| 12 | 60 | 40 | **0.5576** / 1.51 | 0.5576 / 1.51 | **mean ≈0.56** (text), Table II 1.72 | 6e-7 |
| 20 | 14 | 40 | 0.0559 / 25.7 | 0.0559 / 25.7 | Table II 43.9 | 2e-7 |
| 20 | 60 | 40 | 0.2574 / 5.02 | 0.2574 / 5.02 | Table II 9.18 | 6e-7 |

Per-instance agreement to float32 precision. Eq. (13) sign taken literally gives mean p ≈ 4e-5 at n=8 (below random 7e-3),
confirming the sign pitfall. Independent float64 numpy re-evaluation (qaoa_ksat.circuit) of QAOA_raw L=14 angles on the first
24 of our n=20 4-SAT test instances: max rel. diff 5e-6 vs stored p.

4-SAT sanity (`qaoa_paper_k4.py`): paper's published k=4, r=9.93 angles on OUR 4-SAT test sets vs our trained QAOA_raw
(median / B90 1/p):

| n | paper k4 p=8 | QAOA_raw L=8 | paper k4 p=10 | QAOA_raw L=10 |
|---|---|---|---|---|
| 8 | 2.0 / 5.3 | 2.5 / 4.5 | 1.6 / 4.2 | 2.2 / 4.2 |
| 14 | 5.7 / 37.0 | 6.3 / 18.3 | 4.1 / 27.2 | 5.5 / 16.8 |
| 20 | 13.7 / 121.9 | 16.4 / 54.2 | 8.8 / 141.6 | 13.2 / 48.4 |

Paper angles (optimised for mean p at n→∞) give a slightly better median; our mean-log-p training gives a 2-3× tighter B90.
Same ballpark, so QAOA_raw is a fair (not inflated, not crippled) baseline. L=14 n=20 median 7.0 on 4-SAT vs 25.7 on 8-SAT at
p=14 is consistent with k=4 being easier (log2 slope 0.18/var vs 0.26 for 8-SAT in qaoa_ksat). **Verdict: correct, no bug.**
Note for PART 2: QAOA with paper angles is a stronger median reference (13.7 at L=8, 8.8 at L=10, n=20).

Stopped N_bssnap and N_dsnap_R (Zach: focus on ECD + cost·GrayX). Partial results kept in results/N_bssnap_partial.json
(n=14 L=8 434/2224, n=20 not reached) and results/N_dsnap_R_partial.json (train L=2 only, med 27.9).

## G_qfe finished (Sat 08:56 CST)  [layout=T code=gray ntrain=80 nt2=[] final_ecd=True ecd2=False init=qaoa obj=meanlog alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 4 | 5.4/9.9 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| 8 | 2.5/4.5 | 3.6/7.3 | 4.8/15.4 | 6.6/17.3 | 16.4/59.2 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## G_qfe810 finished (Sat 08:57 CST)  [layout=T code=gray ntrain=80 nt2=[10] final_ecd=True ecd2=False init=qaoa obj=meanlog alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 4 | 5.4/10.1 | 8.9/19.1 | 13.4/32.8 | 21.7/54.9 | 83.1/247.8 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| 8 | 2.5/4.6 | 3.5/8.6 | 5.0/18.0 | 6.7/18.8 | 21.3/89.3 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## G_qfe_mix finished (Sat 08:57 CST)  [layout=T code=gray ntrain=80 nt2=[] final_ecd=True ecd2=False init=qaoa obj=mix alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 4 | 5.4/9.7 | 9.0/20.2 | 15.7/35.9 | 21.8/57.1 | 128.9/476.1 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| 8 | 2.6/4.5 | 3.7/7.6 | 5.2/15.8 | 7.1/18.3 | 24.1/100.2 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## Check-in 09:05 CST Sat — QAOA-init + final ECD is a new best

| run (L=8) | n=14 med/B90 | n=20 med/B90 |
|---|---|---|
| **G_qfe** (QAOA_raw init, final ECD, meanlog, train n=8) | 6.6/**17.3** | **16.4/59.2** |
| G_qfe810 (+n=10 in train) | 6.7/18.8 | 21.3/89.3 |
| G_qfe_mix (mix obj) | 7.1/18.3 | 24.1/100.2 |
| S_costgx (previous best, grown init) | 6.0/32.1 | 17.1/128.6 |
| QAOA_raw | 6.3/18.3 | 16.4/54.2 |

G_qfe halves the n=20 B90 tail vs S_costgx (128.6→59.2) and ties QAOA's n=20 median; beats QAOA's n=14 B90 slightly.
Adding n=10 to training and the mix objective both hurt n=20 transfer. G_qfe_mix was OOM-killed once at its n=20 test
(3 n=20 tests at ~2.7 GB each), watchdog resumed it fine.
**Next:** G_qfe recipe at L=10,14 (G_qfe1014) vs QAOA_raw L=10/14; G_fe and G_q ablations running.

## PART 2 start — ECD + cost·GrayX improvement (Sat 09:05 CST)

New driver `costgx.py` (same physics as theory_u.py S_costgx; resumable per L / per test) + `watchdog_cg.sh` (v8, reads
`queue_cg.txt` each minute, ≤3 jobs × 2 threads, deadline Mon 09:00 CST) + `cg_report.py` (tables vs QAOA_raw, SATLike).

Diagnosis of S_costgx: trained ECD β ≈ 0 (|β| < 0.01 every layer), so ECD = transmon X and the circuit is QAOA with a
separate transmon angle θ_l (alternating ≈ π±0.6) and cavity Gray-X angle. It contains QAOA exactly
(θ_l = b_{l-1} − π, φ = 0, β = 0, verified: maxiter-0 QAOA init reproduces QAOA_raw 2.5/4.5 at n=8), but its own
train loss (1.024) beats QAOA (1.084) while n=20 B90 is 2.4× worse → overfit to n=8, not a missing capability.
Also: transmon bits only get L−1 useful mixes (first R+ECD acts on |+>), so added `--final_ecd` (one trailing R+ECD).

First batch (median/B90 1/p, n=20 on 100 inst, n=14 on 200):

| run | L | n=14 | n=20 | note |
|---|---|---|---|---|
| S_costgx (old best) | 4 | 18.2/69.4 | 79.0/416.4 | |
| G_qfe (QAOA init + final ECD) | 4 | 22.8/57.5 | 103.5/358.6 | |
| G_qfe810 (+ train n=8+10) | 4 | 21.7/54.9 | 83.1/247.8 | ≈ QAOA tail |
| G_qfe_mix (+ CVaR20 term) | 4 | 21.8/57.1 | 128.9/476.1 | tail obj hurts |
| QAOA_raw | 4 | 20.8/59.6 | 84.9/238.1 | |
| S_costgx (old best) | 8 | 6.0/32.1 | 17.1/128.6 | |
| **G_qfe** | 8 | 6.6/17.3 | **16.4/59.2** | B90 128.6 → 59.2 |
| G_qfe810 | 8 | 6.7/18.8 | 21.3/89.3 | |
| G_qfe_mix | 8 | 7.1/18.3 | 24.1/100.2 | |
| QAOA_raw | 8 | 6.3/18.3 | 16.4/54.2 | |

Reading: the tail gap was a training-basin/overfit issue; starting in QAOA's basin with a final ECD fixes most of it
(L=8 now ties QAOA). Tail-weighted (CVaR) training at n=8 does not transfer. n=8 train loss is a bad predictor of
n=20, so next: iterate/candidate selection on held-out validation sets at n=12,14 (seed base 3.3e10, disjoint from
test 3.2e10) — `--val_n`. Queued: G_fe (grow init + final ECD, isolates final-ECD effect), G_q (QAOA init, no final
ECD), G_qfe_deep (L=10/12/14), G_qfe_val (val selection, grow+QAOA cands), G_qfe81012 (train n=8+10+12).

## G_q finished (Sat 09:03 CST)  [layout=T code=gray ntrain=80 nt2=[] final_ecd=False ecd2=False init=qaoa obj=meanlog alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 4 | 5.8/10.5 | 10.4/23.9 | 16.0/42.6 | 24.7/69.5 | 102.1/413.1 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| 8 | 2.6/4.6 | 3.7/7.8 | 5.1/16.6 | 7.1/18.3 | 17.1/62.6 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## G_fe finished (Sat 09:10 CST)  [layout=T code=gray ntrain=80 nt2=[] final_ecd=True ecd2=False init=grow obj=meanlog alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |
| QAOA 2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| 4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| 8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## G_qfe12 finished (Sat 09:12 CST)  [layout=T code=gray ntrain=80 nt2=[] final_ecd=True ecd2=False init=qaoa obj=meanlog alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 12 | 1.8/3.6 | 2.4/5.7 | 3.0/11.1 | 4.0/15.1 | 9.3/50.6 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## G_qfe1014 finished (Sat 09:14 CST)  [layout=T code=gray ntrain=80 nt2=[] final_ecd=True ecd2=False init=qaoa obj=meanlog alpha=0.2 w=1.0 beta_l2=0.0]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 10 | 2.2/4.2 | 3.1/6.5 | 4.0/12.2 | 5.6/15.7 | 13.7/52.1 |
| QAOA 10 | 2.2/4.2 | 3.1/6.7 | 3.9/12.3 | 5.5/16.8 | 13.2/48.4 |
| 14 | 1.6/3.7 | 2.0/5.7 | 2.9/12.6 | 3.5/18.5 | 9.3/89.7 |
| QAOA 14 | 1.6/3.6 | 2.0/5.7 | 2.6/12.9 | 3.5/17.1 | 7.0/92.3 |
| SATLike | 16/70 | 26/111 | 36/157 | 50/208 | ~124/~530 |

## PART 3 start — NN parameter predictor for ECD + cost·GrayX (Sat 09:18 CST)

G_* costgx queue stopped per Zach. New direction: **instance-conditioned neural net**, deliberately *not* a QAOA correction.

Design (`nn_costgx.py`):
- Circuit = ECD + cost·GrayX + final ECD (same hardware model), **no QAOA warm-start**.
- ECD soft floor + pull toward `|β|≈β0` so displacements stay meaningful (do not collapse to 0).
- **Learned diagonal**: phase = g₀·Cn + g₁·Cn² (+ g₂·Cn³ with `--rich 3`).
- **Per-bit Gray-X**: independent angle per Gray digit (pad to 6), not one shared β.
- **Instance-conditioned MLP** (clause/energy features + n) → additive δ on a shared base (`--mode cond`).
- Curriculum: train {8} → test 10; add 10 → test 12; … through 14; always eval n=20.
- Held-out test seed base 3.2e10 (same as G_qfe / QAOA_raw).

Running now (detached, ≤2×4 threads, deadline Mon 09:00 CST):
- `NN_best`: cond + per_bit + rich=2 + β0=0.25, L∈{4,8}, stages 8→10→12→14
- `NN_best_r3`: same with rich=3, stronger ECD pull
- `NN_agn` queued: size-extrapolation ablation (n→params)

Target: beat QAOA_raw n=20 L=4 (84.9/238.1) and L=8 (16.4/54.2). First table as soon as stage-8 L=4/8 tests land.

## NN_best stage-8 L=4 early (Sat 09:25 CST) — transfer diagnosis

cond+per_bit+rich2, trained only at n=8. |β| held at 0.25 (ECD not collapsed).

| L | n=8 train | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| NN 4 | 5.5/10.2 | 12.4/29.3 | 92.5/687.3 | 158.9/1112.1 | 716307.1/3758644.0 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 5.4/9.9 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |

Reading: n=8 matches QAOA, but instance-δ overfits — n≥12 collapses. Next: curriculum +10/+12,
shared (no δ), and tiny δ_scale=0.08. L=8 training in progress.


## NN stage-8 diagnosis (Sat 09:31 CST)

| run | L | n=8 | n=10 | n=12 | n=14 | n=20 | |β| |
|---|---|---|---|---|---|---|---|
| NN_best cond+per_bit+rich2 | 4 | 5.5/10.2 | 12.4/29.3 | 92.5/687 | 159/1112 | 7e5/4e6 | 0.25 |
| NN_best | 8 | 3.4/5.5 | 20.8/98 | 550/7247 | 1648/11127 | (testing) | 0.26 |
| NN_sh shared+per_bit+rich2 | 4 | 7.8/11.5 | 31.8/64 | 185/615 | 161/475 | 29549/1e5 | 0.07 |
| QAOA_raw | 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238 | 0 |
| QAOA_raw | 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 | 0 |

Forced ECD+|β|≈0.25 matches n=8 but **destroys transfer** (Fock leakage / wrong scale at large k).
rich=2 (Cn²) also hurts. Killed NN_best; launched **NN_v2**: shared, joint train n=8+10+12,
per_bit GrayX, rich=1, soft ECD cap bmax=0.15, tiny β pull. Goal: beat QAOA via richer mixer
without breaking size transfer.

## NN check-in 09:31 CST Sat — stage-8 done; queue pivoted

Stage-8 (train n=8 only) finished for L=4 and L=8. No QAOA beat; transfer still collapses.

| run | L | n=8 train | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|---|
| NN_best (cond δ=0.4) | 4 | 5.5/10.2 | 12.4/29.3 | 92.5/687.3 | 158.9/1112.1 | 716307/3.8e6 |
| NN_best | 8 | 3.4/5.5 | 20.8/98.4 | 549.5/7247 | 1647.8/11127 | 575649/3.4e6 |
| NN_sh (shared, no δ) | 4 | 7.8/11.5 | 31.8/63.9 | 185.2/614.7 | 160.6/474.9 | 29549/112241 |
| NN_best_r3 (rich3) | 4 | 7.9/12.1 | 29.2/75.0 | 402.7/2381 | 179.7/669.5 | 406415/4.8e6 |
| QAOA_raw | 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | **84.9/238.1** |
| QAOA_raw | 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | **16.4/54.2** |

Notes: ECD |β| held ~0.25 on cond runs; shared NN_sh collapsed |β|→0.07. Shared beats cond on n=20 L=4
(29549 vs 716k) but still ≫ QAOA. NN_best died mid stage 8+10 (no ALLDONE); NN_best_r3 died mid L=8 train.
Queue now: NN_sh (running), NN_v2 shared+joint β0=0.1 (running), NN_v2c / NN_v3 queued. Watchdog up, ≤2 jobs.


## NN_v2 L=4 joint 8+10+12 (Sat 09:39 CST)

shared + per_bit GrayX + rich=1 + bmax=0.15. ECD collapsed to |β|≈0.01 (optimizer choice).

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| NN_v2 4 | 10.4/34.2 | 14.7/53.7 | 24.4/91.4 | 116.6/447.0 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |

Joint multi-n training restored transfer (n=20 now 116/447 vs broken 1e5). Still ~1.4× QAOA median.
L=8 training. Next: joint +14, QAOA-basin-free grow with more restarts, tiny instance δ on frozen shared base.

## NN_v2 L=8 done (Sat 10:12 CST)

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| NN_v2 8 (train 2.9/5.8) | 3.5/9.8 | 4.9/27.2 | 6.8/34.3 | 21.6/122.0 |
| QAOA 8 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | **16.4/54.2** |

Joint training transfers at L=8 too (n=20 21.6/122), but still 1.3× QAOA median, 2.3× B90. ECD |β|→0.012 again.
NN_v2 ALLDONE; NN_v2b (joint +14) running, watchdog starting NN_cc (costcond) next.

## NN_v2 (Sat 10:12 CST)  [mode=shared stages=[8, 10, 12] ntrain=80 hidden=64 depth=2 delta_scale=0.3 rich=1 per_cav=False per_bit=True final_ecd=True beta0=0.1 beta_min=0.0 beta_w=0.05]

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| 4 | 10.4/34.2 | 14.7/53.7 | 24.4/91.4 | 116.6/447.0 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |
| 8 | 3.5/9.8 | 4.9/27.2 | 6.8/34.3 | 21.6/122.0 |
| QAOA 8 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_qfe 8 | 3.6/7.3 | 4.8/15.4 | 6.6/17.3 | 16.4/59.2 |

## NN next (Sat 10:20 CST) — stop ECD collapse

NN_v2 L=8 n=20 = 21.6/122 (no QAOA beat). Optimizer still drives |β|→0.01 (near-QAOA).
Root cause: T layout n=20 is [6,6,5]; disp scales β by √(2^k)/2, so |β|=0.25 → α≈1 on k=6 and destroys transfer.

New knobs in `nn_costgx.py`:
- `--fix_beta`: lock |β|=beta0 (phase free) — ECD cannot collapse
- `--phys_beta`: β is physical α (undo per-cavity √(2^k) scale) — same ECD across n
- `--ebins K`: free diagonal cost, one phase per round(Cn) bin (not γ·H)

Queued after NN_v2b / NN_cc: NN_fix04 (α=0.04), NN_fix06 (α=0.06), NN_ebin (12 bins + fix04), NN_ccfix.
Target still QAOA n=20 L4 84.9/238 and L8 16.4/54. Stay quiet until a clear beat.

## Check-in (Sat 10:40 CST)

Running: NN_v2b (shared joint 8–14, L4 restarts) and NN_cc (costcond). No new results; no QAOA beat yet.
Found NN_v2b launched with old code where shared-mode restarts were identical (cand0–2 all 7.864→2.335) — wasted ~17 min each.
Fix: restarts already use variants 0/1/2; now r≥3 adds seeded jitter so all restarts differ. Helper `logs/requeue_v2b.sh` kills NN_v2b once its L4 tests are saved; watchdog relaunches it (L4 skipped, L8 trains with distinct restarts).

## NN_v2b L=4 interim (Sat 10:48 CST) — best NN so far, still no QAOA beat

shared+joint 8+10+12+14, ntrain=100, |β|→0.004.

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| NN_v2b 4 | 8.9/25.7 | 12.3/45.9 | 20.8/74.3 | **91.7/397.4** |
| NN_v2 4 | 10.4/34.2 | 14.7/53.7 | 24.4/91.4 | 116.6/447.0 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | **84.9/238.1** |

+14 joint training closed most of the median gap (1.08× QAOA). B90 still 1.7×. ECD collapsed again.
L=8 training (requeued with distinct restart jitter). NN_cc L4 cand0 train-loss 2.13 (better than shared 2.33).

## Check-in (Sat 11:02 CST) — NN_cc L=4: beats QAOA at n≤14, not at n=20

costcond (MLP on cost features → per-instance γ offsets), joint 8+10+12+14, rich=2.

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| NN_cc 4 | **6.8/16.2** | **10.4/30.6** | **16.5/54.6** | 122.1/505.7 |
| NN_v2b 4 | 8.9/25.7 | 12.3/45.9 | 20.8/74.3 | 91.7/397.4 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |

First NN that beats QAOA on both median and B90 at n=10/12/14 (L=4), but the cost-conditioned offsets extrapolate badly to n=20 (overfits n≤14 feature range). |β|→0.009 again.
NN_cc was OOM-killed right after its n20 test (box has only ~4.4 GB free); L4 results were saved and the watchdog relaunched it straight into L=8.
Guard added to `nn_costgx.py` (new launches): n≥20 evals take `/tmp/nn_n20.lock` and use chunk=2, so two jobs never run big evals at once.
Running: NN_v2b L8 (distinct restarts), NN_cc L8.

## SNAP instead of ECD (Sat 11:10 CST) — design

Zach: maybe use SNAP instead of ECD. ECD collapsed to |β|≈0 in ECD+cost·GrayX and NN arms.
Prior S_snap2 kept ECD+SNAP and hurt n=20 (42/311 vs S_costgx 17/129). Native N_* (no GrayX) failed at n=20.

**New arm `snap_costgx.py` (NO ECD):**
each layer = R(th,ph) on transmons + **local SNAP** on each cavity + joint U = cost · GrayX.

Kinds:
- `csnap` (default): conditional SNAP |0⟩⊗SNAP(s0)+|1⟩⊗SNAP(s1) — true ECD replacement
- `uncond`: unconditional size-free poly SNAP
- `free`: free phases on logical Fock levels

Smoke SNAP_smoke L=2 n=8: med1/p=16.5/37.6, **snap_mag=0.92** (does not collapse).

Queued (watchdog_snap waits for NN to free RAM): SNAP_cs4, SNAP_cs4j (joint 8+10+12+14), SNAP_uc4, SNAP_free8, SNAP_cs4jR.
Target still QAOA n=20 L4 84.9/238 and L8 16.4/54. NN jobs left running.

## SNAP_smoke (Sat 11:10 CST)  [kind=csnap snap=2 layout=T nt2=[]]

| L | n=8 |
|---|---|
| SNAP_smoke 2 | 16.5/37.6 |
| QAOA 2 | 9.0/18.8 |
| G_qfe 2 | — |

L=2 snap_mag=0.9236 (mean |s0−s1| for csnap, else mean |s|).

## SNAP queue unblocked (Sat 11:32 CST)
watchdog_snap was waiting for both NN jobs to finish (could be hours). SNAP training at n≤14 is light, so watchdog v2 now runs 1 SNAP job (3 threads) alongside the 2 NN jobs when RAM allows; `snap_costgx.py` n=20 tests now share `/tmp/nn_n20.lock` with NN (chunk=2), so only one big eval runs at a time.
Running: NN_v2b L8, NN_cc L8, SNAP_cs4.

## SNAP_cs4 n=8-only (Sat 11:57 CST)  [kind=csnap snap=4 layout=T nt2=[]]

Train n=8 only (next queue item is joint SNAP_cs4j). snap_mag stays nonzero (L2 0.62 → L8 0.39). Transfer fails hard — confirms need for joint multi-n.

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| SNAP_cs4 2 | 21.0/50.6 | 129/937 | 379/4591 | 448/3476 | 101848/1.7e6 |
| SNAP_cs4 4 | 9.6/19.8 | 28.8/100 | 241/1898 | 82.5/350 | **23698/280906** |
| SNAP_cs4 8 | 6.4/14.4 | 349/2600 | 3444/19792 | 723/4439 | (n20 L8 running) |
| QAOA 4 | — | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | **84.9/238.1** |
| QAOA 8 | — | — | — | — | **16.4/54.2** |

No QAOA beat. Running: NN_v2b L8, NN_fix04 (fixed |β|=0.04 + phys_beta), SNAP_cs4 n20 L8 then SNAP_cs4j.
## Check-in (Sat 11:57 CST)

Alive: NN_v2b (L8 train), NN_fix04 just launched (fix_beta+phys_beta), SNAP_cs4 (n20 L8 eval under shared lock), both watchdogs. NN_cc deferred (L4 n20=122/506 + OOM on L8). PID 2888494 untouched. No n=20 QAOA beat yet — stay quiet.

## SNAP_cs4 (Sat 11:58 CST)  [kind=csnap snap=4 layout=T nt2=[]]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| SNAP_cs4 2 | 21.0/50.6 | 129.4/937.5 | 378.6/4591.4 | 447.9/3475.7 | 101848.4/1734683.1 |
| QAOA 2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_qfe 2 | — | — | — | — | — |
| SNAP_cs4 4 | 9.6/19.8 | 28.8/100.4 | 240.9/1897.8 | 82.5/350.3 | 23697.9/280906.1 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 5.4/9.9 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |
| SNAP_cs4 8 | 6.4/14.4 | 349.3/2600.4 | 3444.1/19791.9 | 723.5/4439.5 | 791073.4/6005532.5 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_qfe 8 | 2.5/4.5 | 3.6/7.3 | 4.8/15.4 | 6.6/17.3 | 16.4/59.2 |

L=2 snap_mag=0.6237 (mean |s0−s1| for csnap, else mean |s|).

L=4 snap_mag=0.5316 (mean |s0−s1| for csnap, else mean |s|).

L=8 snap_mag=0.3911 (mean |s0−s1| for csnap, else mean |s|).

## Status 2026-10-17 11:59 CST — native bias

Running: NN_v2b (L8), NN_fix04 (fixed |β|=0.04), SNAP_cs4j (joint n=8+10+12+14).
Killed deferred NN_cc L8 relaunch (L4 already 122/506 @n=20). Queued NN_ebin, SNAP_cs4jeb (ebins=12), SNAP_free8j, SNAP_uc4j.

SNAP_cs4 (n=8-only train) done — SNAP strength holds (~0.39–0.62) but transfer fails hard:
| L | n=8 | n=14 | n=20 |
|---|---|---|---|
| 4 | 9.6/19.8 | 82.5/350 | 23698/280906 |
| 8 | 6.4/14.4 | 723/4440 | 791073/6.0e6 |
QAOA L4/L8 n=20 = 84.9/238 / 16.4/54. No beat. Next hope: SNAP_cs4j joint multi-n.

## Check-in (Sat 12:14 CST)

Alive: NN_v2b (L8 train, ~57m CPU-busy), NN_fix04 (L4 cand1), SNAP_cs4j (L2 cand2+), watchdog_nn + watchdog_snap. PID 2888494 untouched. Mem ~2.2 GiB avail — no new launch.
Best n=20 so far still short of QAOA (L4 84.9/238, L8 16.4/54): S_costgx L4 79/416 (med ok, B90 miss), S_costgx L8 17.1/129, NN_v2 L8 21.6/122, NN_v2b L4 91.7/397. No beat — stay quiet.

## NN_v2b (Sat 12:18 CST)  [mode=shared stages=[8, 10, 12, 14] ntrain=100 hidden=64 depth=2 delta_scale=0.3 rich=1 per_cav=False per_bit=True final_ecd=True beta0=0.05 beta_min=0.0 beta_w=0.0 bmax=0.2]

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| 4 | 8.9/25.7 | 12.3/45.9 | 20.8/74.3 | 91.7/397.4 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |

## NN_v2b aborted mid-L8 (2026-10-17 12:18:09 CST)
L4 complete (n20 91.7/397, |beta|=0.004). Killed L8 to free RAM for SNAP_cs4j + NN_fix04/ebin (more native).

## Status 2026-10-17 12:22 CST

**NN_fix04 L4** (fixed |β|=0.04, held!): train |beta|=0.040. Transfer:
| n | fix04 | QAOA | NN_v2b (β→0) |
|---|---|---|---|
| 10 | 7.7/20.4 | 8.4/19.7 | 8.9/25.7 |
| 12 | 13.1/46.1 | 12.8/32.8 | 12.3/45.9 |
| 14 | **17.8**/75.8 | 20.8/59.6 | 20.8/74.3 |
| 20 | 120.1/515.2 | **84.9/238** | 91.7/397 |
Beats QAOA median at n=14; loses at n=20 (~1.4× med, ~2.2× B90). No n=20 beat. L8 running.

**SNAP_cs4** (n=8-only): snap_mag holds 0.39–0.62 but n=20 fails (L4 23700/281k). Pushed.
**SNAP_cs4j** L2 train 25.0/56.4 snap_mag=0.43; L4/L8+tests running (joint n=8..14).
**NN_ebin** launched (ebins=12 + fix β). Queued: SNAP_cs4jeb, free8j, uc4j, cs4jm.

Killed NN_v2b mid-L8 (|β|=0.004 near-QAOA) and deferred NN_cc to free RAM. Watchdogs healthy. No QAOA beat → no Zach ping.

## Check-in (2026-10-17 12:23 CST)

Alive: NN_fix04 (L8 after L4 n20=120/515), NN_ebin (ebins=12+fixβ), SNAP_cs4j (joint L2 done snap_mag=0.43; L4+). watchdog_nn+snap. PID 2888494 untouched. No G_*/watchdog_cg restart.
Queue iterate from data: NN_fix02 / NN_ebin02 before larger |β|; SNAP_cs4jr2 (rich=2 cost) added; fix06/cc deferred. snap_report now → LOG.md.
No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## Check-in (2026-10-17 12:32 CST)

Alive: NN_fix04 (L8 after OOM-relaunch @12:25; L4 n20=120.1/515.2 held), NN_ebin (L4 cand1+), SNAP_cs4j (L2 done snap_mag=0.43 train 25.0/56.4; L4 cand1+), watchdog_nn+snap. PID 2888494 untouched. MemAvail ~2.0 GiB — no new launch (3 trainers). No G_*/watchdog_cg restart.
No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## OOM guard (2026-10-17 12:34 CST)

OOM killed NN_fix04 once (~12:25); relaunched and resumed L8 (L4 checkpoint kept: n20=120/515).
Paused NN_ebin to free RAM. Now: SNAP_cs4j + NN_fix04 only. Watchdogs: ≤1 NN while SNAP runs.
No n=20 QAOA beat → stay quiet.

## Check-in (2026-10-17 12:52 CST)

Alive: NN_fix04 (L8 cand0 done → cand1+; L4 n20=120.1/515.2 held), SNAP_cs4j (L2 done snap_mag=0.43; L4 cand2 done @12:50, cand3+/L8 pending), watchdog_nn+snap. PID 2888494 untouched. MemAvail ~4.1 GiB. NN_ebin stays paused (OOM guard). No G_*/watchdog_cg restart.
No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## Check-in (2026-10-17 13:14 CST)

Alive: NN_fix04 (L8 cand1 done @13:07 → cand2+; L4 n20=120.1/515.2 held, |β|=0.040), SNAP_cs4j (L2/L4 train done; L4 train 19.56/33.95 snap_mag=0.26; L8 cand0+ in progress — tests after all L), watchdog_nn+snap. PID 2888494 untouched. MemAvail ~2.2 GiB — no new launch (OOM guard: ≤1 NN while SNAP). No G_*/watchdog_cg restart. NN_ebin stays paused.
No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## Check-in (2026-10-17 13:16 CST)

Alive: NN_fix04 (L8 train; L4 n20=120.1/515.2), SNAP_cs4j (L8 after L4 train 19.6/34 snap_mag=0.26; tests pending), watchdog_nn+snap+tu. PID 2888494 untouched. No G_*/watchdog_cg.
Data move: S_bin beat QAOA **median** at n20 L4 (75.4 vs 84.9) but B90 354>238 — queued binary SNAP (cs4jB/Bm), snap_min/fix_snap floors, and idle-only S_binj/C_binmix (joint+mix) via watchdog_tu.
No clear n=20 beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.


## Check-in (2026-10-17 13:30 CST)

Alive: NN_fix04 (L8 cand2 done @13:28 → train pick/tests next; L4 n20=120.1/515.2 held, |β|=0.040), SNAP_cs4j (L8 cand0 done @13:20 after L4 train 19.6/34 snap_mag=0.26; cand1+/tests pending), watchdog_nn+snap+tu. PID 2888494 untouched. MemAvail ~2.2 GiB — no new launch (OOM guard: ≤1 NN while SNAP). No G_*/watchdog_cg restart. NN_ebin stays paused; tu idle-only.
No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## NN_fix04 (Sat 13:59 CST)  [mode=shared stages=[8, 10, 12, 14] ntrain=100 hidden=64 depth=2 delta_scale=0.3 rich=1 per_cav=False per_bit=True final_ecd=True beta0=0.04 beta_min=0.0 beta_w=0.0 bmax=0.0 fix_beta=True phys_beta=True ebins=0]

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| 4 | 7.7/20.4 | 13.1/46.1 | 17.8/75.8 | 120.1/515.2 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |
| 8 | 3.2/10.6 | 4.9/28.7 | 6.9/32.5 | 29.6/208.0 |
| QAOA 8 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_qfe 8 | 3.6/7.3 | 4.8/15.4 | 6.6/17.3 | 16.4/59.2 |

## Check-in (2026-10-17 13:55–14:00 CST)

Alive: SNAP_cs4j (L8 cand2+ after L4 train 19.6/34 snap_mag=0.26; tests pending), NN_ebin (just launched by watchdog after NN_fix04 ALLDONE @13:56), watchdog_nn+snap+tu. PID 2888494 untouched. MemAvail ~4.2 GiB. No G_*/watchdog_cg restart.

**NN_fix04 DONE** L8 n20 = **29.6/208** vs QAOA 16.4/54.2 — no beat (~1.8× med, ~3.8× B90). L4 was 120.1/515.2. Fixed |β|=0.040 held. Beats QAOA med at n=14 L4 only; loses n=20 both L.
No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## NN_fix04 done (2026-10-17 14:01 CST)

Fixed |β|=0.04 held. L8 n20 **29.6/208** vs QAOA 16.4/54 (no beat; ~1.8× med, ~3.8× B90). Worse transfer than NN_v2 L8 21.6/122 (β→0).
| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| 4 | 7.7/20.4 | 13.1/46.1 | 17.8/75.8 | 120.1/515.2 |
| 8 | 3.2/10.6 | 4.9/28.7 | 6.9/32.5 | 29.6/208.0 |
| QAOA 8 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | **16.4/54.2** |

Next: NN_ebin (free diagonal cost, fixβ) running; queued soft-β ebin + binary. SNAP_cs4j L8 cand2 done (~4.53 best so far), cand3+/tests pending. No Zach ping.


## Queue re-aim (2026-10-17 14:05 CST)

S_bin L4 n20 **75.4/354** beat QAOA med (84.9) — ECD was ~0 there too; win is **binary X mixer**, not ECD.
B90 gap is the target. Reordered:
- SNAP: cs4jBm (binary+mix) → cs4jB → fix/min snap → ebin/rich
- TU (idle-only): C_binmix / C_bincvar / S_bin0j (ecd=0) / S_binsnapj
- NN: bin04 / ebinB next after ebin

Alive: SNAP_cs4j (L8 cand3+), NN_ebin (L4), watchdogs nn/snap/tu. PID 2888494 untouched. No G_*/watchdog_cg.
No n=20 QAOA beat yet → stay quiet.

## Check-in (2026-10-17 14:14 CST)

Alive: SNAP_cs4j (n20 tests after L8 train 18.95/33.61 snap_mag=0.15 collapsing; n20 L2=55264/580193 so far; L4/L8 pending), NN_ebin (L4 train 42.1/101 |β|=0.040; n10=51.7/145 n12=75.9/197 n14=154/468; n20/L8 pending), watchdog_nn+snap+tu. PID 2888494 untouched. MemAvail ~2.1 GiB — no new launch (OOM guard: ≤1 NN while SNAP). No G_*/watchdog_cg restart.

No n=20 QAOA beat (need L4≤84.9/238 or L8≤16.4/54) → stay quiet.

## SNAP_cs4j (Sat 14:18 CST)  [kind=csnap snap=4 layout=T nt2=[10, 12, 14]]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| SNAP_cs4j 2 | 19.9/40.8 | 65.8/489.1 | 303.2/3125.2 | 324.6/1658.0 | 55264.0/580193.5 |
| QAOA 2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_qfe 2 | — | — | — | — | — |
| SNAP_cs4j 4 | 17.0/33.2 | 59.4/456.0 | 286.9/2430.8 | 256.3/1282.5 | 48501.8/1067110.8 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 5.4/9.9 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |
| SNAP_cs4j 8 | 16.4/33.1 | 60.4/400.7 | 255.5/2554.1 | 249.7/1042.3 | 63827.1/631355.6 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_qfe 8 | 2.5/4.5 | 3.6/7.3 | 4.8/15.4 | 6.6/17.3 | 16.4/59.2 |

L=2 snap_mag=0.4259 (mean |s0−s1| for csnap, else mean |s|).

L=4 snap_mag=0.2600 (mean |s0−s1| for csnap, else mean |s|).

L=8 snap_mag=0.1499 (mean |s0−s1| for csnap, else mean |s|).

## SNAP_cs4j (Sat 14:18 CST)  [kind=csnap snap=4 layout=T nt2=[10, 12, 14]]

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| SNAP_cs4j 2 | 19.9/40.8 | 65.8/489.1 | 303.2/3125.2 | 324.6/1658.0 | 55264.0/580193.5 |
| QAOA 2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_qfe 2 | — | — | — | — | — |
| SNAP_cs4j 4 | 17.0/33.2 | 59.4/456.0 | 286.9/2430.8 | 256.3/1282.5 | 48501.8/1067110.8 |
| QAOA 4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 5.4/9.9 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |
| SNAP_cs4j 8 | 16.4/33.1 | 60.4/400.7 | 255.5/2554.1 | 249.7/1042.3 | 63827.1/631355.6 |
| QAOA 8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_qfe 8 | 2.5/4.5 | 3.6/7.3 | 4.8/15.4 | 6.6/17.3 | 16.4/59.2 |

L=2 snap_mag=0.4259 (mean |s0−s1| for csnap, else mean |s|).

L=4 snap_mag=0.2600 (mean |s0−s1| for csnap, else mean |s|).

L=8 snap_mag=0.1499 (mean |s0−s1| for csnap, else mean |s|).


## SNAP_cs4j DONE (2026-10-17 14:18 CST)

Joint csnap=4. Train OK (L2 25.0/56.4 snap_mag=0.43 → L4 19.6/34 mag=0.26 → L8 19.0/33.6 mag=0.15) but **transfer collapsed** — snap_mag shrinks with L; n20 useless.

| L | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| 2 | 19.9/40.8 | 65.8/489 | 303/3125 | 325/1658 | 55264/580194 |
| 4 | 17.0/33.2 | 59.4/456 | 287/2431 | 256/1283 | 48502/1067111 |
| 8 | 16.4/33.1 | 60.4/401 | 256/2554 | 250/1042 | 63827/631356 |
| QAOA 4 | | | | | **84.9/238** |
| QAOA 8 | | | | | **16.4/54** |

No beat. Watchdog already launched **SNAP_cs4jBm** (binary+mix, next queue priority). NN_ebin L4 n20=2821/9490 (also no beat; L8 training). PID 2888494 untouched.

## Check-in (2026-10-17 14:22 CST)

Alive: SNAP_cs4jBm (just started), NN_ebin (L4 done n20=2821/9490; L8+), watchdog_nn+snap+tu. MemAvail ~2.2 GiB. No G_*/watchdog_cg. No n=20 QAOA beat → stay quiet.

## SNAP_cs4j DONE + NN_ebin abort (2026-10-17 14:25 CST)

**SNAP_cs4j** (joint n=8..14, gray, csnap=4): snap_mag shrank 0.43→0.26→0.15. n=20 failed hard.
| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| 2 | 65.8/489 | 303/3125 | 325/1658 | 55264/580194 |
| 4 | 59.4/456 | 287/2431 | 256/1283 | 48502/1.07e6 |
| 8 | 60.4/401 | 256/2554 | 250/1042 | 63827/631356 |
| QAOA 8 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | **16.4/54.2** |

Joint helped train (L8 train 19.0/33.6) but did **not** transfer. Gray SNAP≠binary X.

**NN_ebin** L4 n20 **2821/9489** (ebins=12 overfit). Aborted before L8; freed for binary.

**Now running:** SNAP_cs4jBm (binary+mix) — best bet from S_bin med-win. Next NN: NN_bin04.
No n=20 QAOA beat → stay quiet.

## NN_ebin (Sat 14:25 CST)  [mode=shared stages=[8, 10, 12, 14] ntrain=80 hidden=64 depth=2 delta_scale=0.3 rich=12 per_cav=False per_bit=True final_ecd=True beta0=0.04 beta_min=0.0 beta_w=0.0 bmax=0.0 fix_beta=True phys_beta=True ebins=12]

| L | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|
| 4 | 51.7/144.8 | 75.9/196.5 | 154.3/467.5 | 2821.1/9489.7 |
| QAOA 4 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_qfe 4 | 9.2/20.5 | 14.6/37.6 | 22.8/57.5 | 103.5/358.6 |

## Pivot to C_binmix (2026-10-17 14:29 CST)

SNAP_cs4jBm L2 train **35.3/117.7** snap_mag=0.24 — worse than gray cs4j L2 (25/56). Aborted; commented out of SNAP queue.
Launched **C_binmix** (binary cost·X + mix α=0.2 w=1.5, joint n=8..14) and **C_bincvar** — direct attack on S_bin B90 gap (75.4/354 vs QAOA 84.9/238).
NN_bin04 still running. No SNAP until these finish. PID 2888494 untouched.

## Check-in (2026-10-17 14:35 CST)

Alive: NN_bin04 (curriculum build), C_binmix, C_bincvar, SNAP_cs4jB (L2 training; cand0 7.50→6.30), watchdogs nn+snap+tu. PID 2888494 untouched. MemAvail ~1.9 GiB; load ~19 on 8 cores (busy but progressing). No finished n=20 this window; no QAOA beat → stay quiet.

## Direction change (Sat 14:57 CST)
Zach: don't force ECD to work. If it's useless, find a natural way. No more locked/floored ECD displacement (`--fix_beta`) or forced SNAP strength (`--fix_snap`, `--snap_min`) runs; those queued entries are marked DROPPED. Remaining work lets each design use what actually helps (ECD free to go to zero or off, binary/Gray mixers, free diagonal costs, unforced SNAP).

## Check-in (2026-10-17 14:58 CST)

Only C_binmix was alive (L=4 train cand1 loss 23.0→7.99), leaving ~4 cores idle. Relaunched **C_bincvar** (binary cost·X, CVaR α=0.15, ECD unforced) and started **S_bin0j** (binary cost·X, ECD off, joint n=8..14), the natural no-ECD test after Zach's "don't force ECD" note. NN_bin04/SNAP_cs4jB stay paused (forced-ECD dropped / held). PID 2888494 untouched. No finished n=20 this window; no QAOA beat → stay quiet.

## NN back in (Sat 15:03 CST)
Zach: keep trying versions, with the network involved. Un-held NN_bin_soft (binary mixer, ECD free, no lock) and NN_ebinB_soft (binary mixer + learned energy-bin cost, ECD free). watchdog_nn launches them when RAM allows.

## QAOA start removed (Sat 15:08 CST)
Zach: remove the QAOA-start idea. All `--init qaoa` / `--init both` entries in queue_cg.txt are marked REMOVED and will not run again. Current NN, SNAP and binary-X runs already start from grown/random init (no QAOA angles). G_qfe results stay in the log for reference only.

## 15:12 CST restart (v2): Zach stopped everything and confirmed the plan
Found structure (per layer: local ECD per cavity+transmon, then one joint U = cost phase · Gray-X; plus final ECD), trained from scratch at n=8 with BFGS, frozen and tested at n=10/12/14/20 vs QAOA at equal L. No QAOA init, nothing forced on ECD. Variations only on this setup: L (2/4/8/10), Gray vs binary, more restarts (16), objective (meanlog/CVaR/mix), n=12 validation for iterate selection, ECD2. Baseline G_fe (4 restarts): n=20 L4 116.1/387.7, L8 20.3/118.8 vs QAOA 84.9/238.1, 16.4/54.2. Queue: queue_v2.txt, runner watchdog_v2.sh (2 jobs x 4 threads).

## 15:18 CST v3 build plan (approved)
Zach approved the v3 build plan and asked to "explore how to improve once you find some data". v2 queue replaced (watchdog_v2.sh + R_g16/R_b16 killed at ~15:20 CST; partial results/R_*.json kept).
Fixed circuit: per layer local R+ECD per (cavity, transmon) register, then ONE joint cavity U = X-mixer . exp(-i phase); plus final R+ECD. Training size n=8 only; validation n=10/12 (seeds 3.3e10) only selects among trained candidates; test n=8/10/12/14/20 (seeds 3.2e10). No QAOA init, ECD fully free.
New code in `oct_explore/v3/`:
- `v3core.py` batched simulator (per-instance params native; per-digit X mixers; weighted-clause phase). Verified bit-exact vs costgx on G_fe L4 (n=8,10 p agree to 1e-7).
- `v3_plain.py` PLAN A: screen `--ninit 200` random inits (300 it, 128 inst) -> refine top 16 (L2) / 12 + 10 grown-from-previous-L (L4/8/10) to convergence (maxiter 2000) on 1000 inst -> keep top 10 by train loss -> winner by val n=10+12 (60 each) -> test.
- `gnn_v3.py` PLAN B: bipartite var–clause graph, sign-typed edges, 3 rounds message passing (node-shared weights), heads: global (per-layer ECD/R/mixer/gamma deltas on a random base), per-clause weights w_{c,l} (phase = (n/m) sum_c w_{c,l}[c violated]), per-variable angles (cavity digit -> its Gray/binary digit X angle in U_l; transmon var -> its R theta in block l). Heads zero-init so each init starts as a random plain circuit. L-BFGS-B over all weights (H=64: 85k weights).
Smoke timings (2 threads/job): plain n=8 1000 inst ~0.2 s/it L2, ~0.4 L4, ~0.8 L8; screening (128 inst) ~0.04 s/it L2; n=20 test 100 inst ~1 min/L. GNN L4 H64 1000 inst ~1.5-2 s/it.
Scale-downs vs plan (runtime): screening uses 128 inst x 300 it (full 1000-inst BFGS only for the refined top 12-16 + grown); GNN uses 1000 train inst (not 2000) and 6 inits (L4) / 4 inits (L8), maxiter 1000, instead of 20 inits.
Runner `v3/run_v3.sh` (4 jobs x 2 threads, MemAvailable >= 3000 MB gate, deps via `after`), queue `v3/queue_v3.txt`, logs `logs/v3/`, results `results/v3/`. Per-job table appended here automatically.

## Check-in (2026-10-17 15:34 CST)
v3 runner alive (run_v3.sh), 4 jobs x 2 threads on 8 cores: A_g2 (screen 30/200), A_b2 (screen 20/200), B_g4_iii and B_g4_i (GNN building). MemAvail ~6 GiB. No finished jobs yet; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in + queue revision (2026-10-17 16:05 CST)
A_g2 / A_b2 DONE (L2, 200 inits): the L2 landscape is easy, ~half of the 200 screened inits reach the same basin (refined train loss 2.2885-2.2886). n=20 L2 winner med/B90 1/p: gray 319.9/1097.5, binary 332.6/1036.1 (gray ~ binary). No QAOA beat (targets are L4/8/10).
Random-base GNN (B_g4_iii, B_g4_i) init 0 after 1000 L-BFGS its (~26 min): train loss 3.926 (clause_w+var_mix) vs 4.556 (global only), both far worse than the plain L2 optimum 2.289. Reason: the random base starts in a bad basin (screen median ~4.4-4.8) and 85k weights don't escape it. So: 
- B_g4_iii / B_g4_i cut to --ninit 1 (keep init 0, run val/test only) as the from-scratch-base GNN reference.
- All further GNN runs (*_A tags) start from the from-scratch n=8 plain keep-list (`--init_from results/v3/A_*.json`, heads zero-init => exactly the plain optimum at it 0, so GNN can only improve train loss). Still no QAOA anywhere. Added L2 GNN (B_g2_iii_A, B_g2_i_A) to test this quickly; ablations (global / +clause / +clause+var, H32/H128, binary) moved to the _A bases at L4/8/10.
- Stray old-queue random-base launches at 16:00-16:01 (B_g4_iii_h128, B_b4_iii, B_g8_i) were stopped; logs in logs/v3/aborted/. Runner now caps GNN at 2 concurrent and uses --chunk 125 (an OOM killed GNN workers at 4 concurrent).

### A_g2 finished 2026-10-17 16:04 CST  (L=2)

`code=gray L=2 ninit=200 screen_n=128 screen_iter=300 refine=16 keep=10 ntrain=1000 maxiter=2000 grow_wait_h=8.0 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] obj=meanlog alpha=0.2 w=1.0 chunk=500`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| A_g2 winner scr48 (train 2.289, val 3.109) | 9.2/19.0 | 18.7/50.2 | 30.5/84.7 | 55.9/166.6 | 319.9/1097.5 |
| A_g2 trainbest scr147 (train 2.289, val 3.112) | 9.2/19.0 | 18.9/49.9 | 31.0/86.8 | 55.8/166.8 | 316.1/1126.3 |
| QAOA_raw L=2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_fe L=2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |

n=20 winner vs QAOA L=2: median 319.9 vs 336.4, B90 1097.5 vs 971.4 -> no beat

### A_b2 finished 2026-10-17 16:04 CST  (L=2)

`code=binary L=2 ninit=200 screen_n=128 screen_iter=300 refine=16 keep=10 ntrain=1000 maxiter=2000 grow_wait_h=8.0 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] obj=meanlog alpha=0.2 w=1.0 chunk=500`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| A_b2 winner scr114 (train 2.289, val 3.110) | 9.2/19.1 | 18.6/50.5 | 29.6/84.9 | 55.6/172.0 | 332.6/1036.1 |
| A_b2 trainbest scr51 (train 2.289, val 3.111) | 9.2/19.0 | 18.6/50.7 | 29.6/85.1 | 55.5/172.6 | 334.2/1035.5 |
| QAOA_raw L=2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_fe L=2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |

n=20 winner vs QAOA L=2: median 332.6 vs 336.4, B90 1036.1 vs 971.4 -> no beat

### B_g4_iii finished 2026-10-17 16:04 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True ninit=6 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] chunk=250`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_iii winner 0 (train 3.926, val 8.023) | 37.0/428.6 | 1450.1/19931.1 | 5178.2/65609.5 | 13279.0/240575.4 | 1509185.3/22838228.6 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 1509185.3 vs 84.9, B90 22838228.6 vs 238.1 -> no beat

### B_g4_i finished 2026-10-17 16:04 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 ninit=6 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] chunk=250`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_i winner 0 (train 4.556, val 8.303) | 105.6/533.3 | 1050.3/16685.3 | 6039.7/49578.2 | 30110.1/548365.6 | 1332162.3/20293751.1 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 1332162.3 vs 84.9, B90 20293751.1 vs 238.1 -> no beat

## Check-in (2026-10-17 16:18 CST)
v3 runner alive (run_v3.sh pid 3365113), 4 jobs on ~7.7 cores: A_g4 / A_b4 (L4 screen 20/200; screen best train-subset loss 1.839 / 1.840 vs L2 optimum 2.289), B_g2_iii_A / B_g2_i_A (GNN from the plain L2 keep-list, init 0 training). MemAvail ~2 GiB so the 3000 MB gate holds further launches; queue unchanged. No new finished jobs; no n=20 QAOA beat. Note: ps etime lags log wall-clock by ~6 min (box likely paused briefly ~16:09-16:15); jobs resumed fine. PID 2888494 untouched.

## Check-in (2026-10-17 16:30 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs on ~6.6 cores. A_g4 screen 40/200 (best train-subset loss 1.593, p10 2.155), A_b4 screen 30/200 (best 1.607); L4 screening runs ~1 init/min with the two GNN jobs alongside, so the L4 screens finish ~19:00 CST before refine. B_g2_iii_A / B_g2_i_A still on init 0 (no per-iteration log lines; previous GNN inits took ~26 min per 1000 its). MemAvail ~2 GiB, gate holds the queue; queue unchanged. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 16:52 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs on ~6.6 cores, MemAvail ~2 GiB (3000 MB gate holds the queue; queue unchanged). A_g4 / A_b4 L4 screens at 50/200 (best train-subset loss 1.556 / 1.607); no screen log line since 16:33 even though CPU time keeps rising, consistent with another brief box pause (ps etime again lags wall-clock). First GNN-from-plain result: B_g2_iii_A (clause_w + var_mix) init 0 trained 2.2886 -> 2.0360 in 1000 L-BFGS its (2815 s), an 11% train-loss drop below the plain L2 optimum, so per-instance clause weights / mixer angles do add expressivity at n=8; val/test pending after the remaining inits. B_g2_i_A (global only) still on init 0. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 17:12 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs, MemAvail ~2 GiB (gate holds queue; queue unchanged). A_g4 screen 80/200 (best 1.556, p10 2.016), A_b4 screen 70/200 (best 1.607, p10 2.060); screening ~1 init/20-40 s when not paused, another apparent box pause 16:33-16:53. GNN-from-plain L2 init 0 both done: global-only (B_g2_i_A) 2.2886 -> 2.2038 (-3.7%) vs clause_w+var_mix (B_g2_iii_A) 2.2886 -> 2.0360 (-11%), so most of the GNN gain at n=8 comes from the per-clause weights / per-variable mixer heads, not the global angle head. Inits 1-3 come from the same plain basin, so they mostly test GNN optimizer variance; left running since all next jobs wait on A_g4/A_b4 anyway. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 17:35 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs (A_g4, A_b4, B_g2_iii_A, B_g2_i_A), CPU time still advancing; MemAvail ~2.2 GiB so the 3000 MB gate keeps holding the queue (queue unchanged). Last screen lines 17:15/17:16 (A_g4 90/200 best 1.556 p10 1.982; A_b4 90/200 best 1.607 p10 2.051); no new lines since, matching another box pause (~17:16-17:34; process etime now lags wall-clock by ~58 min total since 16:02). GNN-from-plain L2 jobs on inits 1-3. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 17:52 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs, CPU advancing, MemAvail ~2 GiB (3000 MB gate holds queue; queue unchanged). A_g4 screen 100/200 (best 1.556, p10 1.948), A_b4 100/200 (best 1.607, p10 2.035). GNN-from-plain L2 init 1: clause_w+var_mix 2.2886 -> 2.0253, global-only 2.2886 -> 2.2226, consistent with init 0 (per-clause/per-variable heads carry the gain). No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 18:11 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs, CPU advancing, MemAvail ~2 GiB (3000 MB gate holds queue; queue unchanged). A_g4 screen 120/200 (best 1.556, p10 1.839), A_b4 110/200 (best 1.607, p10 2.034); the gray-code L4 tail keeps improving (p10 1.95 -> 1.84) while binary has flattened. GNN-from-plain L2 jobs on inits 2-3 (init 1 already logged). No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 18:33 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs (A_g4, A_b4, B_g2_iii_A, B_g2_i_A), CPU advancing, MemAvail ~2 GiB (3000 MB gate holds queue; queue unchanged). A_g4 screen 130/200 (best 1.556, p10 1.838), A_b4 120/200 (best 1.607, p10 2.034); both tails have flattened. GNN-from-plain L2 jobs on inits 2-3. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 18:54 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs (A_g4, A_b4, B_g2_iii_A, B_g2_i_A), CPU advancing, MemAvail ~2 GiB free / 6 GiB avail per `free` (3000 MB gate behaviour unchanged; queue unchanged). A_g4 screen 160/200 (best 1.556, p10 1.838, flat), A_b4 160/200 (best 1.607, p10 1.958, binary tail improving again 2.034 -> 1.958). GNN-from-plain L2 init 2: clause_w+var_mix 2.2886 -> 2.0422, global-only 2.2886 -> 2.2072, same pattern as inits 0-1; init 3 running, then val/test. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 19:16 CST)
v3 runner alive (run_v3.sh pid 3365113), same 4 jobs (A_g4, A_b4, B_g2_iii_A, B_g2_i_A), CPU advancing, MemAvail ~2 GiB (3000 MB gate holds queue; queue unchanged). A_g4 screen 180/200 (best 1.556, p10 1.836), A_b4 180/200 (best 1.607, p10 1.958); last screen lines ~18:58-19:00, so another apparent box pause since then. L4 screens should finish within ~10-20 min of active time and move to refine (12). GNN-from-plain L2 jobs on init 3, then val/test. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

### B_g2_i_A finished 2026-10-17 19:20 CST  (L=2)

`code=gray L=2 hidden=64 rounds=3 ninit=4 ntrain=1000 maxiter=1000 init_from=results/v3/A_g2.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g2_i_A winner 0 (train 2.204, val 4.699) | 8.7/21.9 | 31.2/411.1 | 310.7/1540.6 | 75.6/322.2 | 176275.6/1892119.9 |
| QAOA_raw L=2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_fe L=2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |

n=20 winner vs QAOA L=2: median 176275.6 vs 336.4, B90 1892119.9 vs 971.4 -> no beat

### B_g2_iii_A finished 2026-10-17 19:35 CST  (L=2)

`code=gray L=2 hidden=64 rounds=3 clause_w=True var_mix=True ninit=4 ntrain=1000 maxiter=1000 init_from=results/v3/A_g2.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g2_iii_A winner 1 (train 2.025, val 3.531) | 6.8/20.2 | 15.8/52.1 | 68.8/477.1 | 40.7/205.0 | 22801.5/462798.0 |
| QAOA_raw L=2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_fe L=2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |

n=20 winner vs QAOA L=2: median 22801.5 vs 336.4, B90 462798.0 vs 971.4 -> no beat

## Check-in (2026-10-17 19:40 CST)
**Runner bug fixed.** `ready()` in `v3/run_v3.sh` grepped the result JSON for the string `"keep"`, which also matches the `args.keep` option, so dependents launched before their parent finished. Effect: at 19:20 B_g4_iii_A / B_g4_i_A launched and crashed at once (`KeyError: 'keep'`, A_g4 still screening), and A_g8 / A_b8 launched early (harmless: v3_plain screens first and waits internally for the parent keep-list before growth/refine). `ready()` now checks for a top-level `keep` key; runner restarted (new pid in runner.log), running jobs untouched. The two B_g4_*_A jobs will relaunch automatically once A_g4 writes its keep-list.

**GNN-from-plain L2 results (both done).** Plain A_g2 (shared angles) is QAOA-level everywhere (n=20 319.9/1097.5 vs QAOA 336.4/971.4). GNN clause_w+var_mix (B_g2_iii_A) beats QAOA L2 at n=8 (6.8 vs 9.0), n=10 (15.8 vs 18.4) and n=14 (40.7 vs 54.8 median), but collapses at n=12 (68.8/477) and n=20 (22801/462798); global-only (B_g2_i_A) collapses too (n=12 310.7, n=20 176276).
**Diagnosis:** the non-monotone pattern (n=14 fine, n=12/n=20 broken) matches the cavity layout, not n. Layout ks: n=8 [3,3], n=10 [4,4], n=12 [5,5], n=14 [4,4,3], n=20 [6,6,5]. The GNN var features include hardware placement (digit_j/6, k_c/6, is_cavity), and n=8 training only ever sees k=3, so k=5/6 cavities are out of distribution and the heads output garbage there (n=14's k=4,4,3 stays close to training).
**Follow-up queued:** added `--feat {full,tq,formula}` to `v3/gnn_v3.py` (tq = clause degrees + is_transmon; formula = clause degrees only, size- and placement-free). Queued B_g2_iii_A_f (formula) and B_g2_iii_A_tq (tq), L2 clause_w+var_mix from the A_g2 keep-list, 2 inits each, ahead of the L4 GNN jobs. If they transfer to n=20, the L4/L8 GNN queue entries will be switched to the size-free features.
Running: A_g4 / A_b4 (L4 screens ~190/200, then refine), A_g8 / A_b8 (L8 screens). No n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 19:53 CST)
v3 runner alive (restarted runner pid 3407180, dependency fix in place), 4 jobs at MAXJ=4 on ~7.4 cores: A_g4 / A_b4 (L4 screens 190/200, best 1.556 / 1.607, then refine 12), A_g8 / A_b8 (L8 screens 10/200, best train-subset loss 1.307 gray / 1.437 binary, already below the L4 bests). MemAvail ~2 GiB, so the size-free-feature GNN tests (B_g2_iii_A_f / _tq) wait for a slot; queue unchanged. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 20:18 CST)
v3 runner alive (run_v3.sh pid 3407180), same 4 jobs (A_g4 / A_b4 L4 screens 190/200, best 1.556 / 1.607; A_g8 / A_b8 L8 screens 10/200, best 1.307 / 1.437), ~7.4 cores busy. No new log lines since ~19:42 and runner etime (14.5 min) lags its 19:40 start by ~24 min, so another box pause; processes intact. MemAvail ~4 GiB (`free`), MAXJ=4 slots full, so the size-free-feature GNN tests (B_g2_iii_A_f / _tq) still wait; queue unchanged. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 20:38 CST)
v3 runner alive (restarted pid 3419014). Fixed a silent failure mode: A_g4/A_b4 log paths were recreated at 19:37 while the jobs still held the old (deleted) inodes, so refine/ALLDONE lines vanished from the visible logs and `done_()` would have missed completion → risked a full relaunch. `run_v3.sh` now (1) treats a result JSON with `keep`+`winner`+`test.winner[20]` as done, and (2) each loop refreshes any visible log from a live deleted-fd. Jobs untouched.
Running (4/4 slots, ~7.5 cores): A_g4 / A_b4 in refine (~11/22 cands each after 200-init screen; best refined train loss ~1.570 gray / ~1.570 binary, same basin), A_g8 / A_b8 L8 screens 30/200 and 20/200 (best train-subset 1.307 / 1.437, already below L4 screen bests). MemAvail ~6 GiB; size-free GNN tests (B_g2_iii_A_f / _tq) wait for a free slot. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 20:52 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g4 / A_b4 in refine (latest grown candidates converge to 1.783 gray / 1.570 binary, no improvement on the ~1.570 refine best), A_g8 / A_b8 L8 screens 30/200 (best 1.307 / 1.437). Last log lines ~20:43, runner etime lags wall-clock by ~6 min, so a short box pause again; processes intact. MemAvail ~2 GiB; size-free GNN tests (B_g2_iii_A_f / _tq) still wait for a slot, queue unchanged. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

## Check-in (2026-10-17 21:17 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g4 / A_b4 still in refine/growth (gray grown cands 1.75-1.86, binary grown cands all reconverge to 1.5696, same basin as refine best ~1.570), A_g8 / A_b8 L8 screens 50/200 (gray best 1.307, p10 2.25; binary best improved 1.437 -> 1.385, p10 1.99). Last log lines ~21:01 and runner etime (19 min) lags its 20:38 start by ~20 min, so another box pause; processes intact. MemAvail ~2 GiB; size-free GNN tests (B_g2_iii_A_f / _tq) still wait for a slot, queue unchanged. No new finished jobs; no n=20 QAOA beat. PID 2888494 untouched.

### A_b4 finished 2026-10-17 21:20 CST  (L=4)

`code=binary L=4 ninit=200 screen_n=128 screen_iter=300 refine=12 keep=10 ntrain=1000 maxiter=2000 grow_from=A_b2 grow_wait_h=8.0 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] obj=meanlog alpha=0.2 w=1.0 chunk=500`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| A_b4 winner scr22 (train 1.570, val 2.128) | 4.3/9.1 | 7.2/20.5 | 10.0/36.6 | 16.0/63.6 | 62.4/250.5 |
| A_b4 trainbest grow_scr114 (train 1.570, val 2.129) | 4.3/9.1 | 7.2/20.5 | 10.0/36.7 | 16.0/63.5 | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 62.4 vs 84.9, B90 250.5 vs 238.1 -> no beat

## Check-in (2026-10-17 21:40 CST)
**First n=20 median beat vs QAOA at equal L.** A_b4 (plain shared-angle ECD → cost·binary-X, L=4, trained from scratch at n=8 with BFGS, 200 inits + growth from A_b2, frozen) tests at n=20 (100 instances) at median/B90 1/p = 62.4/250.5 vs QAOA L4 84.9/238.1: median ~26% better, B90 ~5% worse. It also beats QAOA L4 on median at every n (8: 4.3 vs 5.3, 10: 7.2 vs 8.4, 12: 10.0 vs 12.8, 14: 16.0 vs 20.8); B90 is better at n=8/10 and slightly worse from n=12 on. The auto-report line says "no beat" because it requires both metrics; under the standing order a median beat counts, so this is handed to Zach.
**Reasoning for follow-ups:** binary-X currently looks at least as good as Gray-X at L4 (A_g4 still refining; gray grown candidates 1.75–1.86 vs binary 1.570). The queue already has A_b8 (L8 binary, screen 50/200), A_b10, and B_b4_iii_A (GNN from the A_b4 keep-list). Once the size-free-feature GNN tests (B_g2_iii_A_f launched 21:39, B_g2_iii_A_tq next) report, the binary GNN entries will switch to whichever feature set transfers to n=20.
Running (4/4 slots): A_g4 (L4 refine/growth), A_g8 / A_b8 (L8 screens 50/200, best 1.307 / 1.385), B_g2_iii_A_f. MemAvail ~2 GiB. PID 2888494 untouched.

## Check-in (2026-10-17 21:53 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g4 (L4 growth, grown cands reconverge to ~1.857, no gain over refine best ~1.570), A_g8 / A_b8 (L8 screens 60/200; best 1.307 / 1.385, p10 improving 2.00 / 1.83), B_g2_iii_A_f (size-free "formula" GNN features, L2, started 21:39, 84k weights/model). B_g2_iii_A_tq waits for a slot; MemAvail ~2 GiB. No new finished jobs since A_b4 (already reported: n=20 median 62.4 vs QAOA L4 84.9, B90 250.5 vs 238.1). Queue unchanged. PID 2888494 untouched.

### A_g4 finished 2026-10-17 21:56 CST  (L=4)

`code=gray L=4 ninit=200 screen_n=128 screen_iter=300 refine=12 keep=10 ntrain=1000 maxiter=2000 grow_from=A_g2 grow_wait_h=8.0 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] obj=meanlog alpha=0.2 w=1.0 chunk=500`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| A_g4 winner scr34 (train 1.570, val 2.128) | 4.4/9.0 | 7.2/20.7 | 10.4/39.3 | 16.1/62.4 | 60.7/243.5 |
| A_g4 trainbest scr49 (train 1.570, val 2.128) | 4.4/9.0 | 7.2/20.7 | 10.4/39.4 | 16.1/62.5 | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 60.7 vs 84.9, B90 243.5 vs 238.1 -> no beat

## Check-in (2026-10-17 22:19 CST)
**A_g4 (Gray-X, L=4) is a second n=20 median beat vs QAOA L4**, slightly ahead of A_b4: winner scr34 tests at n=20 median/B90 1/p = 60.7/243.5 vs QAOA 84.9/238.1 (median ~29% better, B90 ~2% worse); its train-best scr49 finished its n=20 test at 22:17 with 61.0/242.8 (B90 within 2% of QAOA). Gray and binary L4 land in the same train basin (1.570) and give near-identical frozen transfer, so the mixer code is not the lever at L4; the remaining gap is the B90 tail at n>=12.
**Reasoning for next steps:** the B90 gap is small, so L8 (A_g8 / A_b8 screens 70/200, best 1.307 / 1.385 vs 1.570 at L4) should clear both QAOA L8 targets only if transfer holds; the GNN L4 jobs from the A_g4 keep-list (B_g4_iii_A, B_g4_i_A) are now unblocked but wait behind the size-free-feature tests, which decide which features the L4/L8 GNN runs use. Queue unchanged.
Running (4/4 slots): A_g8, A_b8 (L8 screens 70/200), B_g2_iii_A_f (started 21:39), B_g2_iii_A_tq (launched 22:18). MemAvail ~3 GiB. PID 2888494 untouched.

## Check-in (2026-10-17 22:40 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 80/200 (best train-subset 1.307 gray / 1.385 binary, p10 1.82 / 1.78; ~10 inits per 15-27 min wall, so screens finish ~03:00-05:00 CST, then refine/growth), B_g2_iii_A_f (formula features, since 21:39) and B_g2_iii_A_tq (tq features, since 22:18) still training, no epoch lines yet. Visible A_g8/A_b8 logs confirmed in sync with the live deleted fds. MemAvail ~2 GiB. No new finished jobs; no new n=20 result beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-17 22:59 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 80/200 (best 1.307 gray / 1.385 binary, p10 1.82 / 1.78), B_g2_iii_A_f (formula features) finished init 0 of 2 (train loss 2.289 -> 2.124 after 1000 it, 3763 s), B_g2_iii_A_tq (tq features, since 22:18) still on init 0. Load ~18-20 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 result beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-17 23:10 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 L8 screen 90/200 (best 1.307, p10 1.82), A_b8 L8 screen 90/200 (best improved 1.385 -> 1.330, p10 1.78 -> 1.65; binary now close to gray at L8), B_g2_iii_A_f (formula features) on init 1 of 2 after init 0 2.289 -> 2.124, B_g2_iii_A_tq (tq features) still on init 0. Load ~17-20 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 result beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-17 23:38 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 100/200 (best 1.307 gray / 1.330 binary, p10 1.82 / 1.65), B_g2_iii_A_f (formula features) on init 1 of 2 (init 0 2.289 -> 2.124), B_g2_iii_A_tq (tq features) init 0 done 2.289 -> 2.027 (on par with the full-feature B_g2_iii_A train 2.025, so dropping placement features costs little on n=8 train; n=20 transfer is the real test), now on init 1. Load high, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 result beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-17 23:56 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy: A_g8 / A_b8 L8 screens 100/200 (best 1.307 gray / 1.330 binary, p10 1.82 / 1.65), B_g2_iii_A_f (formula features) finished both inits (2.124 / 2.122, worse on n=8 train than tq's 2.027) and is moving to val/test, B_g2_iii_A_tq on init 1. Load ~22-30 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 result beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

### B_g2_iii_A_f finished 2026-10-17 23:58 CST  (L=2)

`code=gray L=2 hidden=64 rounds=3 clause_w=True var_mix=True ninit=2 ntrain=1000 maxiter=1000 init_from=results/v3/A_g2.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g2_iii_A_f winner 0 (train 2.124, val 3.040) | 7.8/17.6 | 15.8/38.5 | 28.8/78.1 | 44.4/138.7 | 323.5/1475.2 |
| B_g2_iii_A_f trainbest 1 (train 2.122, val 3.052) | 7.8/18.0 | 15.5/36.8 | 26.3/72.3 | 46.2/150.2 | 353.6/1497.6 |
| QAOA_raw L=2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_fe L=2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |

n=20 winner vs QAOA L=2: median 323.5 vs 336.4, B90 1475.2 vs 971.4 -> no beat

## Check-in (2026-10-18 00:13 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 110/200 (best 1.307 gray / 1.330 binary; p10 improving to 1.70 / 1.54), B_g2_iii_A_tq on init 1 (init 0 2.289 -> 2.027), B_g4_iii_A launched 23:58 (GNN L4 from the A_g4 keep-list, 85k weights/model; the KeyError at the top of its log is the stale 19:20 attempt from before A_g4 had finished, the new run started cleanly). B_g2_iii_A_f (formula features) n=20 test: 323.5/1475.2 vs QAOA L2 336.4/971.4, median slightly better but B90 much worse, so the formula features do not fix the GNN tail; tq features (better n=8 train, 2.027 vs 2.124) decide next. Load ~22 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 00:36 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 120/200 (best 1.307 gray / 1.330 binary; p10 1.70 / 1.54), B_g4_iii_A (GNN L4, 85k weights/model) training init 0 since 23:58, B_g2_iii_A_tq in frozen testing. Early tq-feature transfer: n=8 7.0/21.3, n=10 15.6/48.1, n=12 37.7/141.1 (median/B90 1/p) vs QAOA L2 9.0/18.8, 18.4/55.0, 28.7/84.0, so the tq features train better on n=8 (2.027) but transfer worse than full or formula features by n=12; the GNN tail problem is not a feature-set issue. Plain shared-angle circuits (A_b4/A_g4) remain the n=20 leaders. Load ~28 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

### B_g2_iii_A_tq finished 2026-10-18 00:38 CST  (L=2)

`code=gray L=2 hidden=64 rounds=3 clause_w=True var_mix=True ninit=2 ntrain=1000 maxiter=1000 init_from=results/v3/A_g2.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=tq chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g2_iii_A_tq winner 0 (train 2.027, val 3.171) | 7.0/21.3 | 15.6/48.1 | 37.7/141.1 | 40.2/174.0 | 2816.1/26892.8 |
| QAOA_raw L=2 | 9.0/18.8 | 18.4/55.0 | 28.7/84.0 | 54.8/166.2 | 336.4/971.4 |
| G_fe L=2 | 11.6/21.3 | 21.9/43.2 | 43.2/90.5 | 75.4/169.5 | 664.0/1501.0 |

n=20 winner vs QAOA L=2: median 2816.1 vs 336.4, B90 26892.8 vs 971.4 -> no beat

## Check-in (2026-10-18 00:59 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 130/200 (best 1.307 gray / 1.330 binary; p10 1.70 / 1.54), B_g4_iii_A (GNN L4 full, since 23:58) and B_g4_i_A (GNN L4 angles-only, launched 00:38 after B_g2_iii_A_tq finished) on init 0. B_g2_iii_A_tq final n=20 test collapses (2816/26893 vs QAOA L2 336/971) despite the best n=8 train (2.027): of the three L2 GNN feature sets, lower n=8 train loss means worse n=20 transfer, i.e. the GNN overfits n=8 structure. **Reasoning:** the GNN's extra freedom is the problem at n=20, not the feature choice, so B_g4_i_A (no per-clause weights / per-variable mixers) is the key L4 ablation; if it transfers like A_g4 while B_g4_iii_A degrades, follow-ups should shrink the GNN's output freedom (angles only, smaller hidden) rather than add features. Queue unchanged (already holds B_g4_ii_A and the h32/h128 size ablations). Load ~16-20 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new n=20 beat beyond A_b4/A_g4 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 01:15 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 140/200 (best 1.307 gray / 1.330 binary; p10 1.70 / 1.54, screen medians drifting down 3.49 -> 3.36 gray), B_g4_iii_A (GNN L4 full, since 23:58) and B_g4_i_A (GNN L4 angles-only, since 00:38) still on init 0, no epoch lines yet. Load ~23-32 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 01:30 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 150/200 (best 1.307 gray / 1.330 binary; binary p10 improved 1.54 -> 1.50), B_g4_iii_A (GNN L4 full) finished init 0 with n=8 train 1.570 -> 1.354 (4621 s), a large drop below the plain A_g4 basin (1.570); given the L2 pattern (lower n=8 train, worse n=20 transfer) this needs the frozen n=20 test before reading it as progress. B_g4_i_A (angles-only) still on init 0. Load ~24-31 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 01:55 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 160/200 (best 1.307 gray / 1.330 binary; p10 1.70 / 1.50; at ~10 inits per 16-25 min the screens finish ~03:00 CST, then refine/growth), B_g4_iii_A (GNN L4 full) on init 1 after init 0 1.570 -> 1.354, B_g4_i_A (GNN L4 angles-only) finished init 0 with 1.570 -> 1.517 (3321 s). As expected the angles-only GNN moves much less below the plain A_g4 basin than the full GNN (1.517 vs 1.354); per the L2 pattern the smaller drop may transfer better, so the frozen n=20 tests of both decide whether follow-ups should shrink GNN output freedom. Load ~15-20 on 8 cores, MemAvail ~5 GiB, but the 4-job / 2-GNN caps hold (next GNN jobs wait for a GNN slot, A_g10/A_b10 wait for A_g8/A_b8 keep-lists). No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 02:20 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 170/200 (best 1.307 gray / 1.330 binary; p10 1.67 / 1.50), finishing ~03:00-03:30 CST before refine/growth. B_g4_iii_A (GNN L4 full) on init 1 (init 0 1.570 -> 1.354, init 1 due ~02:35), B_g4_i_A (angles-only) on init 1 (init 0 1.570 -> 1.517). Load ~15-19 on 8 cores, MemAvail ~2 GiB, so no extra slot. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 02:31 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 170/200 (best 1.307 gray / 1.330 binary; p10 1.67 / 1.50), B_g4_iii_A (GNN L4 full) on init 1 (init 0 1.570 -> 1.354), B_g4_i_A (angles-only) on init 1 (init 0 1.570 -> 1.517). Load ~17-19 on 8 cores, MemAvail ~5 GiB, caps hold. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 02:52 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 L8 screens 180/200 (best 1.307 gray / 1.330 binary; p10 1.66 / 1.49), finishing ~03:30 CST before refine/growth. B_g4_iii_A (GNN L4 full) finished init 1 with 1.570 -> 1.351 (matches init 0's 1.354, so the full-GNN L4 basin is reproducible), now on init 2 of 4. B_g4_i_A (angles-only) on init 1 (init 0 1.570 -> 1.517). Load ~16-19 on 8 cores, MemAvail ~2 GiB, caps hold. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 03:12 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 / A_b8 finished their 200-init L8 screens at 03:01 (best train-subset 1.307 gray / 1.330 binary; p10 1.66 / 1.48) and are now refining 22 cands on 1000 inst (maxiter 2000), then growth from A_g4/A_b4. B_g4_iii_A (GNN L4 full) has 2/4 inits done (1.570 -> 1.354 / 1.351) and is on init 2; B_g4_i_A (angles-only) has 2/3 inits done (1.570 -> 1.517 / 1.522) and is on init 2. Synced A_g8/A_b8 visible logs from deleted fds. Load ~21-23 on 8 cores, MemAvail ~2.3 GiB (<3500 MB gate), so caps hold and no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 03:38 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (first done: scr5 1.3255 -> 1.2732 in 289 it), A_b8 refining (first cand still running), B_g4_iii_A (GNN L4 full) on init 2 of 4 (1.354 / 1.351 so far), B_g4_i_A (angles-only) on init 2 of 3 (1.517 / 1.522). Load ~14-18 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 03:57 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (scr5 1.3255 -> 1.2732; next cand in progress), A_b8 refining (first done: scr83 1.3495 -> 1.1517 in 641 it, the best L8 n=8 train loss so far, binary below gray's 1.273), B_g4_iii_A (GNN L4 full) finished init 2 with 1.570 -> 1.363 (three inits now 1.354 / 1.351 / 1.363, a stable basin), on init 3 of 4; B_g4_i_A (angles-only) on init 2 of 3 (1.517 / 1.522). Load ~20-27 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 04:12 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (scr5 1.3255 -> 1.2732, scr171 1.4022 -> 1.0827 in 752 it, now the best L8 n=8 train loss, gray back below binary), A_b8 refining (scr83 1.3495 -> 1.1517, scr179 1.3874 -> 1.1974). Refinement gains are large and the screen rank is a poor predictor (scr171 was not near the screen top), so the full 22-cand refine is worth its time before growth. B_g4_iii_A (GNN L4 full) on init 3 of 4 (1.354 / 1.351 / 1.363), B_g4_i_A (angles-only) on init 2 of 3 (1.517 / 1.522). Load ~18-19 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

### B_g4_i_A finished 2026-10-18 04:14 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=full chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_i_A winner 2 (train 1.517, val 3.541) | 4.3/9.2 | 9.8/31.7 | 118.9/2777.8 | 19.6/87.1 | 117559.2/1703765.6 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 117559.2 vs 84.9, B90 1703765.6 vs 238.1 -> no beat

## Check-in (2026-10-18 04:30 CST)
v3 runner alive (run_v3.sh pid 3419014). B_g4_i_A (GNN L4 angles-only, full features) finished 04:14 and collapsed at n=20 (117559/1703766 vs QAOA L4 84.9/238.1) and already at n=12 (118.9/2778), even though it moved little off the plain A_g4 basin at n=8 (1.517). **Reasoning / direction change within the rules:** every GNN run with the default `--feat full` (hardware-placement features: cavity count k and digit position, where n=8 only ever shows k=3) has collapsed at n=20: L2 angles-only 176276, L2 full 22802, L2 tq 2816, L4 angles-only 117559. The only GNN that transferred is L2 `--feat formula` (degrees only): 323.5/1475 vs QAOA L2 336.4/971, a median edge at n=20. So the output-freedom ablation is confounded by the features, and the queued L4/L8/L10 GNN jobs were all still on full features. Stopped B_g4_ii_A (full, 15 min in, PID 3502929) and switched every unstarted GNN job to `--feat formula` (tags suffixed `_f`), with B_g4_iii_A_f then B_g4_i_A_f then B_g4_ii_A_f first in line for the free GNN slot; B_g8_*_f wait for the A_g8 keep-list. B_g4_iii_A (full, init 3 of 4) is left to finish as the full-feature L4 reference. Plain A_g8 / A_b8 still refining 22 cands (best n=8 train 1.083 gray scr171, 1.152 binary scr83). Load ~29 on 8 cores, MemAvail ~2 GiB. No new n=20 beat beyond A_b4/A_g4 (already reported). PID 2888494 untouched. Backup of previous queue: v3/queue_v3.bak_0430.

## Check-in (2026-10-18 04:55 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (4 done: scr5 1.273, scr171 1.083, scr118 1.216, scr53 1.250; best still scr171), A_b8 refining (3 done: scr83 1.152, scr179 1.197, scr36 1.323), B_g4_iii_A (GNN L4 full features, reference) on init 3 of 4 (1.354 / 1.351 / 1.363), B_g4_iii_A_f (GNN L4 full outputs, formula features) started 04:30 on init 0. Synced A_g8/A_b8 visible logs from deleted fds. Load ~15-18 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

### B_g4_iii_A finished 2026-10-18 05:01 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True ninit=4 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=full chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_iii_A winner 3 (train 1.352, val 3.440) | 3.4/7.5 | 7.9/25.3 | 97.3/1338.9 | 15.6/57.2 | 129679.9/2568903.1 |
| B_g4_iii_A trainbest 1 (train 1.351, val 4.727) | 3.5/7.6 | 18.8/260.2 | 449.8/2328.8 | 25.2/182.4 | 214966.5/1659342.8 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 129679.9 vs 84.9, B90 2568903.1 vs 238.1 -> no beat

## Check-in (2026-10-18 05:13 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing. B_g4_iii_A (GNN L4 full outputs, full features, reference) finished 05:01: n=8 train 1.352, but n=12 97.3/1339 and n=20 129680/2568903 vs QAOA L4 84.9/238.1, no beat. This is the fifth full-feature GNN to collapse at n=20, which confirms the 04:30 switch of all remaining GNN jobs to `--feat formula`. B_g4_i_A_f (angles-only, formula features) took the freed GNN slot at 05:01; B_g4_iii_A_f (full outputs, formula) on init 0 since 04:30. Plain refines continue: A_g8 5/22 done (best scr171 1.083), A_b8 5/22 done (best scr83 1.152; scr190 1.227, scr7 1.276). Load ~15-18 on 8 cores, MemAvail ~3 GiB (<3500 MB gate), caps hold, no new launches. No new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 05:34 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (6 done; new scr29 1.546 -> 1.206, best still scr171 1.083), A_b8 refining (6 done; scr116 1.472 -> 1.452, best still scr83 1.152). B_g4_iii_A_f (GNN L4 full outputs, formula features) finished init 0 with 1.570 -> 1.402 (2722 s), a smaller drop than the full-feature version (1.354), consistent with the formula features carrying less n=8-specific placement information; its n=20 test will show whether this transfers better. B_g4_i_A_f (angles-only, formula) on init 0 since 05:01. Synced A_g8/A_b8 visible logs from deleted fds. Load ~20-28 on 8 cores, MemAvail ~3-4 GiB, caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 05:55 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (7 done, best still scr171 1.083; current cand running since 05:14), A_b8 refining (7 done, best still scr83 1.152), B_g4_iii_A_f (GNN L4 full outputs, formula features) on init 1 of 3 (init 0 1.570 -> 1.402), B_g4_i_A_f (angles-only, formula) on init 0 since 05:01. Synced A_g8/A_b8 visible logs from deleted fds. Load ~15-18 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 06:16 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (8 done; scr45 1.583 -> 1.488, best still scr171 1.083), A_b8 refining (8 done; scr88 1.487 -> 1.375, best still scr83 1.152). Refine pace is ~25 min/cand, so the remaining 14 cands need roughly 6 more hours before growth starts; kept as is since screen rank has not predicted refined loss. B_g4_iii_A_f (GNN L4 full outputs, formula features) on init 1 of 3 (init 0 1.570 -> 1.402), B_g4_i_A_f (angles-only, formula) on init 0 since 05:01. Synced A_g8/A_b8 visible logs from deleted fds. Load ~17 on 8 cores, MemAvail ~3 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 06:36 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (8 done, best still scr171 1.083; current cand running since 05:56), A_b8 refining (8 done, best still scr83 1.152). B_g4_i_A_f (GNN L4 angles-only, formula features) finished init 0 with 1.570 -> 1.549 (4635 s), barely moving off the plain A_g4 basin, like the full-feature angles-only run (1.517); on init 1 of 3. B_g4_iii_A_f (full outputs, formula) on init 1 of 3 (init 0 1.570 -> 1.402). Load ~16-18 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 07:20 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (10 done; new scr103 1.553, scr71 1.686 -> 1.246; best still scr171 1.083), A_b8 refining (9 done; new scr162 1.428; best still scr83 1.152). B_g4_iii_A_f (GNN L4 full outputs, formula features) finished init 1 with 1.570 -> 1.407 (5894 s), matching init 0's 1.402, so the formula-feature L4 basin is reproducible; on init 2 of 3. B_g4_i_A_f (angles-only, formula) on init 1 of 3 (init 0 1.549). Synced A_g8/A_b8 visible logs from deleted fds. Load ~17 on 8 cores, MemAvail ~2.3 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 07:34 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (10 done, best still scr171 1.083), A_b8 refining (9 done, best still scr83 1.152). B_g4_i_A_f (GNN L4 angles-only, formula features) finished init 1 with 1.570 -> 1.5485 (3841 s), matching init 0's 1.549, so angles-only freedom again barely leaves the plain A_g4 basin; on init 2 of 3. B_g4_iii_A_f (full outputs, formula) on init 2 of 3 (1.402 / 1.407). Load ~20-28 on 8 cores, MemAvail ~4 GiB, caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 07:56 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing: A_g8 refining 22 cands on 1000 inst (10 done, best still scr171 1.083), A_b8 refining (10 done; new scr48 1.514 -> 1.155, nearly tying the binary best scr83 1.152). B_g4_iii_A_f (GNN L4 full outputs, formula features) on init 2 of 3 (1.402 / 1.407), B_g4_i_A_f (angles-only, formula) on init 2 of 3 (1.549 / 1.549). Synced A_g8/A_b8 visible logs from deleted fds. Load ~15-19 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; no new n=20 beat beyond A_b4/A_g4 (already reported). Queue unchanged. PID 2888494 untouched.

### B_g4_iii_A_f finished 2026-10-18 08:09 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_iii_A_f winner 2 (train 1.409, val 2.031) | 3.6/7.1 | 6.4/15.7 | 7.9/29.7 | 13.5/43.9 | 56.0/217.2 |
| B_g4_iii_A_f trainbest 0 (train 1.402, val 2.040) | 3.5/7.2 | 6.5/14.8 | 8.4/28.2 | 13.8/40.5 | 53.5/183.7 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 56.0 vs 84.9, B90 217.2 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-18 08:13 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy. **B_g4_iii_A_f (GNN L4, full outputs: per-layer angles + per-clause cost weights + per-variable mixer angles, formula/degree features, started from the from-scratch A_g4 keep-list, trained at n=8 only) finished 08:09 and is the first v3 run to beat QAOA L4 at n=20 on both metrics:** val-selected winner (init 2) 56.0/217.2 vs QAOA 84.9/238.1 (median -34%, B90 -9%); it also beats QAOA at n=8/10/12/14 (3.6/7.1, 6.4/15.7, 7.9/29.7, 13.5/43.9). The train-best init 0 (not used as winner, since validation selects) gives 53.5/183.7. The earlier plain A_g4/A_b4 had beaten the median only (62.4/250.5, 60.7/243.5). **Reasoning:** this confirms the 04:30 diagnosis that the n=20 collapses came from hardware-placement features, not from the GNN's extra output freedom; with size-free features the extra freedom transfers and helps, while angles-only (B_g4_i_A_f, 1.549 at n=8) barely leaves the plain basin. The queue already holds the natural follow-ups on formula features (clause-weights-only B_g4_ii_A_f now running since 08:09, binary B_b4_iii_A_f, hidden 32/128 ablations, and L8/L10 versions once the A_g8/A_g10 keep-lists exist), so the queue is unchanged. Load ~16 on 8 cores, MemAvail ~5 GiB. PID 2888494 untouched.

## Check-in (2026-10-18 08:33 CST)
v3 runner alive (run_v3.sh pid 3419014), 4/4 slots busy with CPU time advancing. Plain refines moved on: A_g8 now 13 done (scr25 1.684 -> 1.355, scr188 1.709 -> 1.679) and the first grown cand grow_scr49 went 3.940 -> **0.952**, the best n=8 L8 train loss so far (previous best scr171 1.083). A_b8 13 done (scr81 1.358, scr141 1.388) with grown cands grow_grow_scr114 and grow_grow_scr25 both at **0.952** as well. Gray and binary landing on the identical 0.9519 from different seeds suggests one shared L8 basin reached by growth; its n=10/12 validation and n=20 test will show whether it carries over. B_g4_i_A_f (GNN L4 angles-only, formula features) is in its test phase (winner n=12 10.4/43.5); B_g4_ii_A_f (clause weights only, formula) training since 08:09. Synced A_g8/A_b8 visible logs from deleted fds. Load ~16 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), caps hold, no new launches. No new finished jobs; the B_g4_iii_A_f n=20 beat was reported at 08:13. Queue unchanged. PID 2888494 untouched.

### B_g4_i_A_f finished 2026-10-18 08:33 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_i_A_f winner 1 (train 1.548, val 2.224) | 4.3/9.3 | 6.9/21.5 | 10.4/43.5 | 16.3/70.6 | 78.4/354.4 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 78.4 vs 84.9, B90 354.4 vs 238.1 -> no beat

## 08:43 CST SNAP-for-ECD (Zach 08:43, corrected 08:44: standard SNAP; 08:45: GNN tunes transmon + SNAP)
**Why:** in the winner B_g4_iii_A_f the trained ECD |beta| is ~0.002-0.01 (max 0.03), so local ECD is effectively transmon R + flip. Replace ECD by a standard SNAP.
**Design (v3/gnn_snap.py):** local block per register, layer l<L: S_c (standard SNAP, cavity only, no transmon dependence) -> R(th,ph) on transmon -> X on transmon (X = exactly ECD(beta=0)). Joint U unchanged (per-digit X mixer . exp(-i cost phase)); final block R + X. Size-free SNAP: theta_k = sum_j a_j b_j(k) + sum_{i<j} c_ij b_i(k) b_j(k), b_j(k) = logical (Gray) digit j of Fock k, pads 0. GNN: a_{v,l} = sa_l + h_v.Ws[:,l] (cavity-digit variable head), c_{ij,l} = sc_l + sum_h h_i h_j Wp[h,l]/H; transmon R stays network output (shared base + per-variable head e_v on the transmon variable, global head for ph), as in B_g4_iii_A_f. All SNAP params zero-init, nothing forced. Plain (--plain): base angles + shared sa_l, sc_l only.
**Absorbability note:** a cavity-only diagonal SNAP commutes with R/X on the transmon, so S in block l is exactly a set of extra diagonal terms in layer l's cost phase. Placing it after the mixer just moves it to the next layer's cost phase (it commutes through R/X there too). In the final block it sits right before computational-basis measurement and cannot change p, so it is omitted. So standard SNAP adds no capability beyond "arbitrary diagonal". What it adds are new terms the clause-weight cost cannot express: per-variable local fields (a) and intra-cavity digit couplings (c). Any gain over the control measures the value of those terms. The nonstandard transmon-conditional version was not built.
**Step-0 check (200 n=8 train instances):** loss with SNAP=0 equals the ECD circuit with beta:=0 on the network output, to within 6e-7 for all 3 warm inits (#2 1.415793 vs 1.415793; #0 1.414651; #1 1.418105). Dropping the trained small beta costs +0.006 to +0.010 in loss (ECD trained beta: 1.407597 / 1.404765 / 1.412154). So beta~0.01 is not exactly useless, and the SNAP run starts slightly behind the ECD warm start. The SNAP-param gradients are nonzero at step 0 (|grad Ws| 1.4-1.7).
**Smoke timing (2 threads, under load):** GNN 125 inst x 5 its = 4 s (+3 s JIT); n=20 eval of 100 test instances = 60 s. Plain 125 inst x 20 its = 2-4 s. 5-it smoke: n=20 48.8/175.6 (not a result, 125 inst).
**Queue (front, 08:48):** S_g4_snap_f (main: GNN + standard SNAP + pairs, warm from B_g4_iii_A_f runs 2,0,1, 1000 its); S_g4_ctrl_f (control: same warm start, ECD circuit, 1000 its, which separates SNAP from simply training longer); S_g4_plainsnap (plain shared-angle SNAP from the A_g4 keep-list, 5 inits, 2000 its); S_g4_snap_fromP_f (GNN SNAP from the S_g4_plainsnap keep, after it). SNAP magnitude (mean |a| and phase rms per layer at each test n) is saved in results json "snap_mag" and logged.

### 08:52 CST SNAP-for-ECD: final spec (Zach 08:43–08:48) — 3 SNAP parameterisations, queue replaced
**Ordering:** SNAP_c -> R(th,ph) -> X on the transmon, per register, layers l<L; final block R + X. A cavity-only diagonal commutes with every transmon op and with the cost phase. The only non-diagonal cavity op is the X mixer inside the joint U, so every placement of a standard SNAP in the local block (before or after R, or after the mixer) folds into a cost phase. In the final block it is invisible to measurement. The only placement that cannot be absorbed is inside the mixer (between digit X rotations), which is not a local-block SNAP and not built. So all three forms test whether the extra diagonal terms they add help, beyond the clause-weight cost.
**Forms, phases per SNAP (n=8 k=3 | n=20 k=6,6,5):**
- (a) lin: k → 3 | 6,6,5.
- (b) pair: k+k(k-1)/2 → 6 | 21,21,15.
- (c) free: 2^k → 8 | 64,64,32, produced by a size-free deep-set MLP head: z_f = mean_j[b_j(f)U1_j + (1-b_j(f))U0_j], U = u + h_v·P (M=32), theta_f = tanh(z_f+q)·Wo, with Wo zero-init.
- In plain mode (no node info) the free head can only depend on each level's Hamming weight (k+1 distinct values). That is why the non-transferable plain table "level" (8 phases × 2 cavities × 4 layers = 64 numbers, n=8 val/test only) is added to give the true free-SNAP n=8 train loss.
- **Trainable params added:** GNN lin +260, pair +520, free +4,320 (85,276 → 85,536 / 85,796 / 89,596). Plain lin 32, pair 36, free 252, level 92 (28 base + SNAP).
**Step-0 checks:**
- GNN: B_g4_iii_A_f runs #2/#0/#1, 200 instances, all three forms: SNAP=0 equals ECD with β:=0 on the network output within 6e-7 (1.415793 / 1.414651 / 1.418105). The trained-β loss is 1.407597 / 1.404765 / 1.412154, so dropping β costs 0.006–0.010.
- Plain: A_g4 keep top 5, 1,000 instances: SNAP=0 equals β:=0 within 6e-7 (1.570294…1.570454) vs trained β 1.569865–1.569950, so β costs only 0.0004 there.
- Gradients on SNAP at step 0 are nonzero: Ws 1.4–1.7, Wo 0.33–0.54.
**Smoke (2 threads, loaded box):** GNN gradient on 200 instances 0.17–0.23 s/eval after a 3 s JIT. Plain 250 instances × 30 its: 7–9 s. Plain 30-it n=8 loss (250 inst) from 1.5717: lin 1.5689, pair 1.5687, free 1.5684, level 1.5553.
**Queue (front, replaces the 08:48 S_g4_* lines; nothing had launched):** S_g4_snapP_f, S_g4_snapF_f, S_g4_ctrl_f, S_g4_plainL, S_g4_plainP, S_g4_plainF, S_g4_plainLev8, S_g4_snapL_f.

## Check-in (2026-10-18 08:56 CST)
v3 runner restarted (run_v3.sh new pid 3564313; previous 3419014 stopped cleanly). All 4 job processes survived the restart: A_g8 (PID 3405870), A_b8 (3406154), B_g4_ii_A_f (3544785), B_b4_iii_A_f (3551522).
**Progress:** A_g8 / A_b8 still in the 22-cand refine+grow phase (screen done 03:01). Grown basin at n=8 train loss **0.9519** (gray grow_scr49 / binary grow_grow_scr114 and siblings); ~17/22 gray and ~19/22 binary cands done, so both should finish within ~1–2 h, then frozen val/test. B_g4_ii_A_f (clause_w only, formula) finished init 0: 1.570 -> 1.440 (2280 s), on init 1 of 3. B_b4_iii_A_f (binary full outputs, formula, since 08:33) still on init 0.
**Runner fix:** MAXGNN now counts `gnn_snap.py` as well as `gnn_v3.py`, so the front-of-queue SNAP jobs cannot stack with the two running GNNs into 4 neural jobs when A_g8/A_b8 free slots. MemAvail ~5.6 GiB, load ~20 on 8 cores, caps hold. SNAP queue (S_g4_snapP_f / snapF_f / ctrl_f / plainL/P/F/Lev8 / snapL_f) still waiting for a GNN slot. No new finished n=20 result; B_g4_iii_A_f 56.0/217.2 vs QAOA L4 84.9/238.1 remains the best (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 09:19 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU time advancing. A_g8 18/22 refine+grow cands done (latest grow_scr171 3.85 -> 1.011, grow_grow_scr53 -> 0.9522); A_b8 21/22 done (four more grown cands at 0.9519-0.9521, same shared L8 basin). Both should enter frozen val/test soon. B_g4_ii_A_f (clause_w only, formula) on init 1 of 3 (init 0 1.440). B_b4_iii_A_f (binary full outputs, formula) finished init 0: 1.570 -> **1.401** (1548 s), matching the gray full-output winner's train loss (1.402-1.409), on init 1 of 3. Load ~18.9 on 8 cores, MemAvail ~2.5 GiB (<3500 MB gate), caps hold, no new launches. SNAP queue (S_g4_snapP_f / snapF_f / ctrl_f / plainL/P/F/Lev8 / snapL_f) still waiting for a GNN slot. No new finished jobs; best n=20 still B_g4_iii_A_f 56.0/217.2 vs QAOA L4 84.9/238.1 (already reported). Queue unchanged. PID 2888494 untouched.

### A_b8 finished 2026-10-18 09:21 CST  (L=8)

`code=binary L=8 ninit=200 screen_n=128 screen_iter=300 refine=12 keep=10 ntrain=1000 maxiter=2000 grow_from=A_b4 grow_wait_h=8.0 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] obj=meanlog alpha=0.2 w=1.0 chunk=500`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| A_b8 winner grow_scr22 (train 0.952, val 1.323) | 2.0/4.9 | 2.9/8.4 | 4.0/21.4 | 5.7/33.4 | 14.6/119.3 |
| A_b8 trainbest grow_grow_scr25 (train 0.952, val 1.324) | 2.0/4.9 | 2.9/8.4 | 4.0/21.3 | 5.7/33.1 | - |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 14.6 vs 16.4, B90 119.3 vs 54.2 -> no beat

## Check-in (2026-10-18 09:38 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy. **A_b8 finished 09:21** (plain binary L8, from-scratch n=8, val-selected winner grow_scr22, train 0.952): n=20 **14.6/119.3 vs QAOA L8 16.4/54.2**, so it beats the QAOA L8 median (-11%) but the B90 is 2.2x worse. It also beats QAOA medians at n=8-14 (2.0/2.9/4.0/5.7 vs 2.5/3.5/4.6/6.3) while B90 is worse from n=10 on (8.4 vs 7.3, 21.4 vs 15.5, 33.4 vs 18.3). **Reasoning:** the shared 0.952 grown L8 basin is a better typical-instance circuit but its hard-instance tail does not transfer; one shared angle set at L8 fits n=8 too tightly. At L4 the formula-feature GNN with full outputs (B_g4_iii_A_f) took the plain A_g4 B90 from 250.5 to 217.2 and beat QAOA on both metrics, so per-instance cost weights and mixer angles are the natural fix for the tail. **Queue change:** added B_b8_iii_A_f (binary L8 GNN, full outputs, formula features, from the A_b8 keep-list, 3 inits) right after the SNAP block; B_g8_iii_A_f is already queued behind A_g8. A_b10 (plain L10 grown from A_b8) launched 09:23 into the free plain slot because both GNN slots are taken by B_g4_ii_A_f (init 1/3) and B_b4_iii_A_f (init 1/3, init 0 1.401). A_g8 still refining (~18/22). SNAP queue still waiting for a GNN slot. MemAvail ~2 GiB (<3500 MB gate), no new launches. PID 2888494 untouched.

## Check-in (2026-10-18 09:58 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing: A_g8 (plain gray L8, 19/22 refine+grow cands; latest grow_scr82 5.97 -> 1.996, best still the shared 0.952 grown basin), A_b10 (plain binary L10 grown from A_b8, screen phase since 09:23), B_g4_ii_A_f (clause_w only, formula; init 1 1.570 -> 1.438, matches init 0 1.440, on init 2 of 3), B_b4_iii_A_f (binary full outputs, formula; init 0 1.401, on init 1 of 3). Clause weights alone (1.438-1.440) stay well above full outputs (1.401-1.409), so per-variable mixer angles carry most of the GNN gain at L4. Synced A_g8 / B_g4_ii_A_f visible logs from deleted fds. Load ~17 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), no new launches. SNAP queue and B_b8_iii_A_f still waiting for a GNN slot. No new finished jobs; best n=20 still B_g4_iii_A_f 56.0/217.2 vs QAOA L4 84.9/238.1 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 10:12 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing: A_g8 (plain gray L8, 19/22 refine+grow cands, best still the shared 0.952 grown basin), A_b10 (plain binary L10 grown from A_b8, screen 10/200 at 09:59, best 1.860; ~3.7 min/init under load, so the screen alone runs ~12 h), B_g4_ii_A_f (clause_w only, formula; init 2 of 3, inits 0/1 1.440/1.438), B_b4_iii_A_f (binary full outputs, formula; init 1 of 3 running 72 min vs 26 min for init 0, slowed by load ~24). Synced A_g8 / B_g4_ii_A_f visible logs from deleted fds. MemAvail ~2.4 GiB (<3500 MB gate), no new launches. SNAP queue and B_b8_iii_A_f still waiting for a GNN slot. No new finished jobs; best n=20 still B_g4_iii_A_f 56.0/217.2 vs QAOA L4 84.9/238.1 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 10:34 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing: A_g8 (plain gray L8; grow_scr119 4.82 -> 1.412 at 10:13, best still the shared 0.952 grown basin), A_b10 (plain binary L10 grown from A_b8, screen 20/200, best 1.860), B_g4_ii_A_f (clause_w only, formula; init 2 of 3, inits 0/1 1.440/1.438), B_b4_iii_A_f (binary full outputs, formula; init 1 1.570 -> **1.404** at 10:15, matching init 0 1.401, now on init 2 of 3). Binary full outputs reproduce the gray winner's train loss (1.402-1.409), so its frozen n=20 test is the next thing to watch. Synced A_g8 / B_g4_ii_A_f visible logs from deleted fds. Load ~15-18 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), no new launches. SNAP queue and B_b8_iii_A_f still waiting for a GNN slot. No new finished jobs; best n=20 still B_g4_iii_A_f 56.0/217.2 vs QAOA L4 84.9/238.1 (already reported). Queue unchanged. PID 2888494 untouched.

## Check-in (2026-10-18 10:59 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing: A_g8 (plain gray L8; grow_grow_scr96 4.77 -> 1.492 at 10:36, best still the shared 0.952 grown basin, ~20/22 cands), A_b10 (plain binary L10 grown from A_b8, screen 20/200, best 1.860), B_g4_ii_A_f (clause_w only, formula; **all 3 inits trained**: 1.440 / 1.438 / 1.435, now in frozen val/test), B_b4_iii_A_f (binary full outputs, formula; inits 0/1 1.401/1.404, on init 2 of 3). Clause-weights-only stays ~0.03 above full outputs in train loss, so expect its n=20 test to land behind B_g4_iii_A_f. Synced A_g8 / B_g4_ii_A_f visible logs from deleted fds. Load ~18 on 8 cores, MemAvail ~2 GiB (<3500 MB gate), no new launches. SNAP queue and B_b8_iii_A_f still waiting for a GNN slot (one frees when B_g4_ii_A_f finishes its test). No new finished jobs; best n=20 still B_g4_iii_A_f 56.0/217.2 vs QAOA L4 84.9/238.1 (already reported). Queue unchanged. PID 2888494 untouched.

### A_g8 finished 2026-10-18 11:18 CST  (L=8)

`code=gray L=8 ninit=200 screen_n=128 screen_iter=300 refine=12 keep=10 ntrain=1000 maxiter=2000 grow_from=A_g4 grow_wait_h=8.0 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] obj=meanlog alpha=0.2 w=1.0 chunk=500`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| A_g8 winner grow_scr49 (train 0.952, val 1.327) | 2.1/4.9 | 2.8/8.2 | 4.0/20.3 | 5.6/32.2 | 14.6/113.1 |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 14.6 vs 16.4, B90 113.1 vs 54.2 -> no beat

### B_g4_ii_A_f finished 2026-10-18 11:18 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_ii_A_f winner 1 (train 1.438, val 2.045) | 3.7/7.4 | 6.3/15.5 | 8.4/30.0 | 13.8/40.8 | 58.2/197.2 |
| B_g4_ii_A_f trainbest 2 (train 1.435, val 2.076) | 3.7/7.3 | 6.4/15.4 | 8.2/29.0 | 13.8/40.5 | 62.8/253.4 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 58.2 vs 84.9, B90 197.2 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-18 11:20 CST)
Two jobs finished, both finalized and pushed by the runner (run_v3.sh pid 3564313).
- **B_g4_ii_A_f** (GNN, clause weights only, formula features, L4): n=20 winner **58.2/197.2 vs QAOA L4 84.9/238.1, beats QAOA on both** and has the lowest L4 n=20 B90 of any val-selected winner so far (B_g4_iii_A_f winner 56.0/217.2). Its job exited before the deleted-fd log sync, so the visible log lacked ALLDONE and the runner could not see it was done (GNN jsons have "runs", not "keep"). The json was complete (both roles at n=8–20), so ALLDONE was appended by hand and the runner finalized it at 11:18.
- **A_g8** (plain gray L8, from scratch, val-selected grow_scr49, train 0.952): n=20 **14.6/113.1 vs QAOA L8 16.4/54.2**, which beats the median (-11%) but has a 2.1x worse B90. Same picture as A_b8 (14.6/119.3), so gray and binary land in the same shared L8 basin.
**Reasoning:** clause-weights-only trains 0.03 worse at n=8 (1.435–1.440 vs 1.402–1.409 with mixer angles) yet ties at n=20. The n=20 B90 spread across inits of one config is large (iii 183.7–217.2, ii 197.2–253.4), so the L4 GNN variants differ by less than init-to-init noise at n=20, and val with 60 instances at n=10/12 does not reliably pick the n=20 tail. Next candidate improvement (rules allow it): a larger n=10/12 val set for selection among already-trained runs. The L8 plain tail problem is what B_g8_iii_A_f (now ready, queued) and B_b8_iii_A_f target.
**Slots:** A_g10 launched 11:18 into the freed plain slot (grown from A_g8). Running: A_b10, A_g10, B_b4_iii_A_f (init 2/3), S_g4_snapP_f (since 11:02). GNN slots full, so B_g8_iii_A_f and the remaining SNAP jobs wait. PID 2888494 untouched.
