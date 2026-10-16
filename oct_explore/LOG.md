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
