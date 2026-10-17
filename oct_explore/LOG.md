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
