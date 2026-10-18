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

### B_b4_iii_A_f finished 2026-10-18 11:40 CST  (L=4)

`code=binary L=4 hidden=64 rounds=3 clause_w=True var_mix=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_b4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_b4_iii_A_f winner 0 (train 1.401, val 2.057) | 3.6/6.9 | 6.2/15.4 | 9.2/28.0 | 13.2/42.3 | 71.7/531.8 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 71.7 vs 84.9, B90 531.8 vs 238.1 -> no beat

## Check-in (2026-10-18 11:41 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing.
- **B_b4_iii_A_f finished 11:40** (binary L4 full outputs, formula): n=20 **71.7/531.8 vs QAOA L4 84.9/238.1**, median beat but B90 2.2x worse -> no beat (runner finalized + pushed). Train losses 1.401/1.404/1.403 match gray full-output, but the n=20 tail is far worse than gray B_g4_iii_A_f (56.0/217.2) or B_g4_ii_A_f (58.2/197.2). **Reasoning:** binary encoding + instance-adaptive GNN overfits the n=8 train basin the same way as plain L8; gray formula GNN remains the only solid L4 transfer. No new binary L4 GNN variants queued.
- **Slots:** S_g4_snapF_f launched 11:39 into the freed GNN slot. Running: A_b10 (screen 50/200, best 1.380), A_g10 (screen 10/200, best 1.568), S_g4_snapP_f (~15 min on first warm init), S_g4_snapF_f (just started). B_g8_iii_A_f / B_b8_iii_A_f and remaining SNAP (ctrl, plainL/P/F/Lev8, snapL) wait for a GNN slot. MemAvail ~6 GiB, load ~21 on 8 cores, caps hold. Best n=20 dual-beat still B_g4_iii_A_f 56.0/217.2 / B_g4_ii_A_f 58.2/197.2 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 12:01 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing. MemAvail ~2.4 GiB (<3500 MB gate), load ~22–28 on 8 cores, caps hold, no new launches. Synced A_b10 visible log from deleted fd.
- **A_b10** (plain binary L10 grown from A_b8): screen **60/200**, best **1.332** (improved from 1.380 at 50/200). ~3–4 min/init under load → screen finish ~8–10 h more.
- **A_g10** (plain gray L10 grown from A_g8): screen **30/200**, best **1.568**.
- **S_g4_snapP_f** (GNN standard SNAP, pair form, warm from B_g4_iii_A_f #2/#0/#1): **init 0 done** 1.4152 → **1.2115** (1000 it, 2485 s). Large n=8 train drop vs ECD warm (~1.40) and vs B_g4_iii_A_f train (~1.40); shared SNAP sa/sc nonzero (sa ≈ −0.005…−0.025, sc ≈ −0.019…−0.031). On init 1 of 3.
- **S_g4_snapF_f** (GNN standard SNAP, free form): started 11:39, still on first warm init (89 596 params; no json yet).
**Queue waiting (GNN slots full):** S_g4_ctrl_f, S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check. Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 12:11 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing. MemAvail ~2.5 GiB (<3500 MB gate), load ~18–20 on 8 cores, caps hold, no new launches. Synced A_b10 visible log from deleted fd (still at screen 70/200).
- **A_b10** (plain binary L10 grown from A_b8): screen **70/200**, best **1.332** (unchanged since 60/200). ~2–3 min/init under load → screen finish ~6–8 h more, then refine+grow.
- **A_g10** (plain gray L10 grown from A_g8): screen **30/200**, best **1.568**.
- **S_g4_snapP_f** (GNN standard SNAP, pair form, warm from B_g4_iii_A_f #2/#0/#1): init 0 done 1.4152 → **1.2115** (1000 it, 2485 s); still on init 1 of 3 (~27 min since init 0 finished; first init was ~41 min).
- **S_g4_snapF_f** (GNN standard SNAP, free form, 89 596 params): started 11:39, still on first warm init, no json yet (~31 min elapsed; pair’s first init took ~41 min so this is on track).
**Queue waiting (GNN slots full):** S_g4_ctrl_f, S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check. Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 12:16 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing. MemAvail ~5.8 GiB, load ~21 on 8 cores, caps hold, no new launches. Synced A_b10 visible log from deleted fd.
- **A_b10** (plain binary L10 grown from A_b8): screen **80/200**, best **1.332** (unchanged since 60/200). ~2–3 min/init under load → screen finish ~4–5 h more, then refine+grow.
- **A_g10** (plain gray L10 grown from A_g8): screen **40/200**, best **1.568**.
- **S_g4_snapP_f** (GNN standard SNAP, pair form, warm from B_g4_iii_A_f #2/#0/#1): init 0 done 1.4152 → **1.2115**; init 1 done 1.4085 → **1.1852** (1000 it, 1908 s); on init 2 of 3. Pair SNAP keeps lowering n=8 train well below the ECD warm (~1.40) and B_g4_iii_A_f (~1.40).
- **S_g4_snapF_f** (GNN standard SNAP, free form): init 0 done 1.4152 → **1.1826** (1000 it, 1967 s); best SNAP n=8 train so far; on init 1 of 3.
**Reasoning (interim, n=8 train only):** free and pair SNAP both move ~0.23 off the ECD warm basin in 1000 BFGS steps, so the extra diagonal terms are being used. Frozen n=20 tests after all 3 warm inits will say whether this transfers (or overfits like binary L4 / plain L8). Queue waiting unchanged (GNN slots full): S_g4_ctrl_f, S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check. Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 12:34 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU time advancing. MemAvail ~2.5 GiB (<3500 MB gate), load ~21–22 on 8 cores, caps hold, no new launches. Synced A_b10 visible log from deleted fd.
- **A_b10** (plain binary L10 grown from A_b8): screen **90/200**, best **1.332** (unchanged since 60/200). ~1–2 min/init under load → screen finish ~3–4 h more, then refine+grow.
- **A_g10** (plain gray L10 grown from A_g8): screen **50/200**, best **1.568**.
- **S_g4_snapP_f** (GNN standard SNAP, pair form, warm from B_g4_iii_A_f #2/#0/#1): init 0 1.4152 → **1.2115**; init 1 1.4085 → **1.1852**; on init 2 of 3 (~19 min since init 1 finished).
- **S_g4_snapF_f** (GNN standard SNAP, free form): init 0 1.4152 → **1.1826**; on init 1 of 3 (~22 min since init 0 finished; free first init was ~33 min).
**Queue waiting (GNN slots full):** S_g4_ctrl_f, S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check. Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1 (already reported). PID 2888494 untouched.

## Check-in (2026-10-18 12:52 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU time advancing. MemAvail ~2622 MB (<3500 MB gate), load ~20.13 20.79 20.63 on 8 cores, caps hold, no new launches. Synced A_b10 visible log from deleted fd.
- **A_b10** (plain binary L10 grown from A_b8): screen **100/200**, best **1.332** (unchanged since 60/200). ~2–3 min/init under load → screen finish ~3–4 h more, then refine+grow.
- **A_g10** (plain gray L10 grown from A_g8): screen **60/200**, best **1.568**.
- **S_g4_snapP_f** (GNN standard SNAP, pair form, warm from B_g4_iii_A_f #2/#0/#1): init 0 1.4152 → **1.2115**; init 1 1.4085 → **1.1852**; on init 2 of 3 (~36 min since init 1 finished; pair inits ~32–41 min).
- **S_g4_snapF_f** (GNN standard SNAP, free form): init 0 1.4152 → **1.1826**; on init 1 of 3 (~39 min since init 0 finished; free first init was ~33 min).
**Queue waiting (GNN slots full):** S_g4_ctrl_f, S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check. Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1 (already reported). PID 2888494 untouched.

### S_g4_snapP_f finished 2026-10-18 13:24 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True snap=std snap_pair=True snap_form=pair free_m=32 warm=results/v3/B_g4_iii_A_f.json warm_ids=[2, 0, 1] ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_snapP_f winner 0 (train 1.212, val 1.870) | 2.8/7.8 | 5.1/19.7 | 6.8/50.7 | 11.7/76.6 | 35.6/398.3 |
| S_g4_snapP_f trainbest 1 (train 1.185, val 1.878) | 2.8/8.3 | 4.9/17.0 | 6.5/41.0 | 11.6/76.6 | 35.1/534.9 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 35.6 vs 84.9, B90 398.3 vs 238.1 -> no beat

## Check-in (2026-10-18 13:25 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy. PID 2888494 untouched.

**S_g4_snapP_f finished 13:24** (GNN standard SNAP, pair form, warm from B_g4_iii_A_f #2/#0/#1, 1000 BFGS its): n=20 winner **35.6/398.3 vs QAOA L4 84.9/238.1** — median −58% (best L4 n=20 median so far, ahead of B_g4_iii_A_f 56.0) but B90 1.7× worse → **no dual beat** (runner finalized + pushed). Train losses 1.212 / 1.185 / 1.212; val-selected init 0. SNAP phase-rms/layer at n=20 ~0.62/0.26/0.21/0.25 (nonzero). Train-best init 1 is 35.1/534.9 (even worse tail).

**Crash fix:** first snapP pass trained all 3 inits then died at test n=10 with `jax.errors.UnexpectedTracerError` — `bits()` had cached `jnp.asarray` tables inside jitted `nf()` for a new layout. Patched `v3/gnn_snap.py` to cache numpy only and `jnp.asarray` at use. Relaunched snapP (resumed tests from n=10) and killed/relaunched snapF (inits 0–1 saved; was mid init 2 under the old binary) so free-form would not hit the same leak on val. Fix rode along in the snapP results commit.

**Slots now:** A_b10 (plain binary L10 screen ~110/200, best 1.332), A_g10 (plain gray L10 screen 80/200, best 1.568), S_g4_snapF_f (resumed at init 2 of 3 since 13:21), S_g4_ctrl_f (ECD control, same warm start / 1000 its, launched 13:24 — separates SNAP from simply training longer). Queue waiting: S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep).

**Reasoning:** pair SNAP clearly moves the typical-instance median (n=8 train 1.40→1.19–1.21 and n=20 median 56→35.6) via extra diagonal terms, but the hard-instance B90 blows out the same way as plain L8 / binary L4 GNN. Ctrl + free + plain SNAP forms already queued will say whether the median gain is SNAP-specific and whether any form keeps the B90. No new queue lines yet — wait for ctrl/free before changing direction. Best dual-beat remains B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2**. MemAvail ~5.6 GiB, load ~20 on 8 cores, caps hold.


## Check-in (2026-10-18 13:34 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing. MemAvail ~2464 MB (<3500 MB gate), load ~19–20 on 8 cores, caps hold, no new launches. Synced A_b10 / A_g10 / S_g4_snapF_f visible logs from deleted fds. PID 2888494 untouched.
- **A_b10** (plain binary L10 grown from A_b8): screen **120/200**, best **1.332** (unchanged since 60/200). ~2–3 min/init under load → screen finish ~3–4 h more, then refine+grow.
- **A_g10** (plain gray L10 grown from A_g8): screen **80/200**, best **1.568**.
- **S_g4_snapF_f** (GNN standard SNAP, free form; relaunched 13:21 after tracer fix with inits 0–1 saved): init 0 1.4152 → **1.1826**; init 1 1.4085 → **1.1546**; mid init 2 of 3 (~9 min into resume; free inits ~33–42 min). Best SNAP n=8 train so far (1.155).
- **S_g4_ctrl_f** (ECD control, same warm / 1000 its, launched 13:24): still on first warm init, no json yet (~6 min).
**Queue waiting (GNN slots full + mem gate):** S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check (snapP already reported: median 35.6, B90 398.3, no dual beat). Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1. Wait for free + ctrl n=20 before queue changes.

## Check-in (2026-10-18 13:59 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU advancing. Synced A_b10 / A_g10 / S_g4_snapF_f visible logs from deleted fds. PID 2888494 untouched. MemAvail ~6.1 GiB, load ~19–20 on 8 cores, caps hold (slots full; no new launches needed).
- **A_b10** (plain binary L10 grown from A_b8): screen **130/200**, best **1.332** (unchanged since 60/200). ~2–3 min/init under load → screen finish ~2.5–3.5 h more, then refine+grow.
- **A_g10** (plain gray L10 grown from A_g8): screen **90/200**, best **1.540** (improved from 1.568 at 80/200).
- **S_g4_snapF_f** (GNN standard SNAP, free form; relaunched 13:21 after tracer fix with inits 0–1 saved): init 0 1.4152 → **1.1826**; init 1 1.4085 → **1.1546**; mid init 2 of 3 (~38 min into resume; free inits ~33–42 min). Best SNAP n=8 train so far (1.155). Json still has 2 runs (writes after each init).
- **S_g4_ctrl_f** (ECD control, same warm / 1000 its, launched 13:24): still on first warm init, no json yet (~35 min; on track vs pair/free ~33–41 min).
**Queue waiting (GNN slots full):** S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check (snapP already reported: median 35.6, B90 398.3, no dual beat). Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1. Wait for free + ctrl n=20 before queue changes.

## Check-in (2026-10-18 14:19 CST)
v3 runner alive (run_v3.sh pid 3564313), 4/4 slots busy with CPU time advancing. Synced A_b10 / A_g10 / S_g4_snapF_f visible logs from deleted fds. PID 2888494 untouched. MemAvail ~2429 MiB (<3500 MB gate), load ~19.70 on 8 cores, caps hold (slots full; no new launches needed).
- **A_b10** (plain binary L10 grown from A_b8): screen **130/200**, best **1.332** (unchanged since 60/200). ~2–4 min/init under load → screen finish ~2–4 h more, then refine+grow. Last json write ~40 min ago (mid next 10-init batch; CPU still advancing).
- **A_g10** (plain gray L10 grown from A_g8): screen **100/200**, best **1.540**.
- **S_g4_snapF_f** (GNN standard SNAP, free form; resumed 13:21 after tracer fix with inits 0–1 saved): init 0 1.4152 → **1.1826**; init 1 1.4085 → **1.1546**; mid init 2 of 3 (~58 min into resume; free init 2 took 3452 s under load). Best SNAP n=8 train so far (1.155). Json still has 2 runs.
- **S_g4_ctrl_f** (ECD control, same warm / 1000 its, launched 13:24): still on first warm init, no json yet (~55 min; on track vs pair/free ~33–41 min unloaded, slower under 4-way load).
**Queue waiting (GNN slots full + mem gate):** S_g4_plainL/P/F/Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f (needs A_g10 keep). No new finished n=20 result this check (snapP already reported: median 35.6, B90 398.3, no dual beat). Best dual-beat still B_g4_iii_A_f **56.0/217.2** and B_g4_ii_A_f **58.2/197.2** vs QAOA L4 84.9/238.1. Wait for free + ctrl n=20 before queue changes.

### S_g4_snapF_f finished 2026-10-18 14:23 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True snap=std snap_form=free free_m=32 warm=results/v3/B_g4_iii_A_f.json warm_ids=[2, 0, 1] ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_snapF_f winner 0 (train 1.183, val 1.887) | 2.7/7.5 | 5.0/19.7 | 7.1/31.6 | 11.6/85.6 | 37.3/211.7 |
| S_g4_snapF_f trainbest 1 (train 1.155, val 1.919) | 2.9/8.6 | 5.1/19.2 | 7.0/47.9 | 12.5/79.1 | 38.5/274.2 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 37.3 vs 84.9, B90 211.7 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-18 14:23 CST)
**S_g4_snapF_f finished** (report above): n=20 winner **37.3/211.7 vs QAOA L4 84.9/238.1** → **dual beat** (median −56%, B90 −11%). Beats prior dual-beat medians (B_g4_iii_A_f 56.0, B_g4_ii_A_f 58.2); B90 between B_g4_ii_A_f 197.2 and B_g4_iii_A_f 217.2. Free SNAP used nonzero phases (n=20 rms/layer ~0.35/0.18/0.13/0.11). Train-best 38.5/274.2 (B90 miss).
**Runner glitch:** deleted log fd had no on-disk ALLDONE, so `done_` missed and runner relaunched snapF at 14:22 — killed the duplicate; wrote ALLDONE into `logs/v3/S_g4_snapF_f.log`. PID 2888494 untouched.
**Slots now:** A_b10 (screen 130/200, best 1.332), A_g10 (100/200, best 1.540), S_g4_ctrl_f (ECD control, still training). One GNN slot free; MemAvail ~6039 MiB. Queue next: S_g4_plainL/P/F/Lev8, S_g4_snapL_f, then B_b8_iii_A_f / B_g8… Waiting on ctrl n=20 to separate SNAP from extra BFGS; no queue rewrite yet beyond letting the runner drain the existing SNAP/plain/lin block.

### S_g4_plainL finished 2026-10-18 14:25 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 snap=std snap_form=lin free_m=32 plain=True init_from=results/v3/A_g4.json ninit=5 ntrain=1000 maxiter=2000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=250`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_plainL winner 0 (train 1.570, val 2.122) | 4.4/9.0 | 7.1/20.2 | 10.3/36.5 | 16.0/61.3 | 59.1/234.1 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 59.1 vs 84.9, B90 234.1 vs 238.1 -> **BEATS QAOA (both)**

### S_g4_plainP finished 2026-10-18 14:31 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 snap=std snap_pair=True snap_form=pair free_m=32 plain=True init_from=results/v3/A_g4.json ninit=5 ntrain=1000 maxiter=2000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=250`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_plainP winner 3 (train 1.570, val 2.120) | 4.4/9.1 | 7.1/21.5 | 10.7/36.8 | 16.0/65.5 | 60.2/238.0 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 60.2 vs 84.9, B90 238.0 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-18 14:35 CST)
v3 runner alive (run_v3.sh pid 3646460), 4/4 slots busy with CPU advancing. Synced A_b10 / A_g10 / S_g4_ctrl_f visible logs from deleted fds. PID 2888494 untouched. MemAvail ~5 GiB, load ~19.4 on 8 cores, caps hold.
- **S_g4_plainL / S_g4_plainP finished** (plain shared-angle standard SNAP, lin / pair form, from the A_g4 keep-list, 5 inits): BFGS converged in 15–21 its with SNAP phases only ~0.01–0.04 rad; train 1.5703 → 1.5701 / 1.5698 (all 5 inits collapse to the same point). n=20 **59.1/234.1** (lin) and **60.2/238.0** (pair) vs QAOA L4 84.9/238.1 → formally dual beats, but these are a tiny perturbation of plain A_g4 (60.7/243.5); the B90 margin (−1.7% / −0.04%) is inside instance noise. **Reading:** a single shared SNAP phase set gives almost nothing at L4; the SNAP gain needs per-instance phases from the network.
- **S_g4_ctrl_f** (ECD control, same warm start, 1000 its, no SNAP): init 0 1.4086 → **1.3777** vs free SNAP 1.18 / pair SNAP 1.19–1.21 on the same warm start → the SNAP median gain is SNAP-specific, not just longer training. Inits 1–2 still running.
- **S_g4_plainF** (plain free-form SNAP) launched 14:31.
- **A_b10** screen 150/200, best 1.332. **A_g10** screen 120/200, best **1.331** (from 1.540).
**Queue:** unchanged (S_g4_Lev8, S_g4_snapL_f, B_b8_iii_A_f, B_g8_iii_A_f, B_g4_iii_A_h32/h128_f, B_g8_i_A_f, B_g10_iii_A_f). Best dual-beat remains S_g4_snapF_f **37.3/211.7** (already reported). Next lever once ctrl finishes: network SNAP at L8 off the A_g8/B_g8 line, since the network-SNAP gain is the one that transfers.

### S_g4_plainF finished 2026-10-18 14:36 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 snap=std snap_form=free free_m=32 plain=True init_from=results/v3/A_g4.json ninit=5 ntrain=1000 maxiter=2000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=250`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_plainF winner 2 (train 1.569, val 2.120) | 4.4/9.2 | 7.1/20.2 | 10.3/37.1 | 16.1/61.9 | 59.6/233.6 |
| S_g4_plainF trainbest 0 (train 1.569, val 2.120) | 4.4/9.2 | 7.2/20.2 | 10.2/37.2 | 16.2/61.9 | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 59.6 vs 84.9, B90 233.6 vs 238.1 -> **BEATS QAOA (both)**

### S_g4_plainLev8 finished 2026-10-18 14:56 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 snap=std snap_form=level free_m=32 plain=True init_from=results/v3/A_g4.json ninit=5 ntrain=1000 maxiter=2000 val=[8] val_count=60 test=[8] feat=formula chunk=250`

| row | n=8 |
|---|---|
| S_g4_plainLev8 winner 0 (train 1.567, val 1.483) | 4.4/8.7 |
| S_g4_plainLev8 trainbest 1 (train 1.567, val 1.483) | 4.4/8.7 |
| QAOA_raw L=4 | 5.3/9.9 |
| G_fe L=4 | 5.4/9.8 |

## Check-in (2026-10-18 14:57 CST)
v3 runner alive (run_v3.sh pid 3646460), 4/4 slots busy (A_b10, A_g10, S_g4_ctrl_f, S_g4_plainLev8), CPU advancing; load ~26 on 8 cores, MemAvail ~2.5 GiB (below the 3500 MB launch floor, so new launches hold until a slot frees). Synced A_b10 / A_g10 / S_g4_ctrl_f visible logs from deleted fds. PID 2888494 untouched.
- **S_g4_plainF** (14:36, above): plain free SNAP also ~0 gain (train 1.5703 → 1.569, n=20 59.6/233.6, ≈ plain A_g4 60.7/243.5). All three plain shared-SNAP forms (lin/pair/free) agree: shared SNAP phases add nothing at L4.
- **S_g4_plainLev8** (n=8-only level table, val/test n=8 only): 4/5 inits done, all 1.5703 → 1.5668. Slightly more than lin/pair/free but it is an n=8 lookup table and cannot be tested at larger n; diagnostic only.
- **S_g4_ctrl_f**: init 0 1.4086 → 1.3777 (vs free SNAP 1.18); inits 1–2 running (~1 h each).
- **A_b10** screen 170/200 best 1.332; **A_g10** 130/200 best 1.331.
**Queue change (backup `v3/queue_v3.bak_1456`):** the SNAP gain is per-instance (network) only, so carry network free SNAP to L8 from the from-scratch plain keep-lists (no QAOA, zero-init SNAP/heads, nothing forced): added `S_g8_snapF_A_f` (gray, after A_g8) and `S_b8_snapF_A_f` (binary, after A_b8) right after S_g4_snapL_f, ahead of the no-SNAP L8 GNNs (B_b8_iii_A_f / B_g8_iii_A_f stay queued as their controls). Targets QAOA L8 n=20 16.4/54.2; A_b8 plain already beats the L8 median (14.6) but misses B90 (119.3), which is where per-instance SNAP helped at L4 (median −33% vs B_g4_iii_A_f).

## Check-in (2026-10-18 15:16 CST)
v3 runner alive (run_v3.sh pid 3646460), 3/4 slots busy (A_b10, A_g10, S_g4_ctrl_f), CPU advancing. MemAvail ~2.7 GiB (box-wide use from other processes; our 3 jobs ~1.8 GiB RSS), below the 3500 MB launch floor, so the runner is correctly holding S_g4_snapL_f and the rest of the queue; floor left as is. PID 2888494 untouched.
- **S_g4_ctrl_f** (ECD control, no SNAP): init 1 1.4016 → **1.3699** (init 0 1.3777) vs network free SNAP 1.18 on the same warm start — the SNAP gain stays SNAP-specific. Init 2 running.
- **A_b10** screen 180/200, best 1.332 → **1.324**. **A_g10** 140/200, best 1.331.
No new n=20 result; best dual-beat remains S_g4_snapF_f 37.3/211.7 (already reported). Queue unchanged.

## 15:29 CST refocus on GNN-SNAP (2026-10-18)
Zach approved refocusing the v3 queue on the working case and **named the approach GNN-SNAP**: the formula GNN on the variable–clause graph that outputs, per instance, standard-SNAP phases (free form), transmon angles, clause weights and per-variable mixer angles (run tags `S_*_snapF_*` / `S_*_snapP_*` / `S_*_snapL_*` = GNN-SNAP with free / pairwise / per-digit SNAP). Current best GNN-SNAP: **S_g4_snapF_f** (L4) n=20 37.3/211.7 vs QAOA L4 84.9/238.1 (train-best init 38.5/274.2). Rules unchanged: n=8 L-BFGS-B training only, n=10/12 select only, no QAOA starts, SNAP zero-init. PID 2888494 and *.STOP files untouched.
- **Stopped (SIGTERM, own PIDs verified by cmdline) the plain L10 screens:** A_b10 (pid 3571566) at screen 190/200, partial best n=8 loss **1.3243** (p10 1.641, median 4.366); A_g10 (pid 3590664) at screen 150/200, partial best **1.3312** (p10 1.783, median 3.665). Partial screens stay in `results/v3/A_b10.json` / `A_g10.json` (no keep, no refine, no n=20). Their queue lines and B_g10_iii_A_f (needs A_g10 keep) are commented `#PARKED-1529` so the runner cannot relaunch them.
- **S_g4_ctrl_f** (ECD control, no SNAP) left running (pid 3630951, init 2 of 3).
- **Queue reordered** (backup `v3/queue_v3.bak_1530`): R_g4_snapF_val → S_g8_snapF_A_f (gray, warm A_g8 keep) → S_b8_snapF_A_f (binary, warm A_b8 keep) → S_g4_snapF_s1 (seed 1, warm B_g4_iii_A_f #2 0 1) → S_g4_snapF_s2 (seed 2, warm #2 0) → R_g4_snapF_val2 (after s1,s2) → S_g4_snapL_f → no-SNAP L8 controls B_b8_iii_A_f, B_g8_iii_A_f, B_g8_i_A_f → hidden-size ablations B_g4_iii_A_h32_f/h128_f. The 5 new L4 inits reuse the 3 warm GNN bases (the only no-QAOA warm bases) with new random free-SNAP-head input projections; SNAP outputs zero-init as before; separate result names.
- **Large-validation re-selection** `v3/reselect_snap.py` (selection only, no training): every trained init scored on a FRESH validation set, 600 instances each at n=10 and n=12, seed base 36e9 (disjoint from train 31e9, n=20 test 32e9 and the old 60-instance val 33e9). **Criterion fixed now, before any re-pick n=20 number: minimise mean over n∈{10,12} of (log median 1/p + log B90 1/p)/2; tie-break mean −log p.** Only the re-picked init is then tested at n=20 (same 100 TEB instances). R_g4_snapF_val = the 3 S_g4_snapF_f inits; R_g4_snapF_val2 = those + the 5 new inits. Smoke-tested on 4 instances (result discarded).
- **Runner** (backup `v3/run_v3.sh.bak_1530`, restarted as pid 3668898; old runner 3646460 stopped, jobs untouched): `after` may be a comma list and a dep counts as ready when it is done (GNN jobs write no keep); MAXGNN 2 → 4 because the queue is now all-GNN (still ≤4 jobs × 2 threads = 8 cores); post-launch settle 30 → 90 s so the next memory reading includes the new job. Memory gate 3500 MB unchanged.
- **Blocker: memory.** MemAvailable ~3.1–3.5 GB after stopping the L10 screens (box-wide; our only job S_g4_ctrl_f ~0.65 GB). The rest is other agents' processes on the shared box (~5.2 GB exec-daemon/node, ~1 GB Xvfb across 15 desktops); not touched. Everything holds until S_g4_ctrl_f finishes (~15:45–16:15) or memory frees.

## Check-in (2026-10-18 15:37 CST)
Runner alive (run_v3.sh pid 3668898). Only job: S_g4_ctrl_f (pid 3630951), init 2 of 3 running since 15:02 (init 1 took ~37 min, so it should finish within minutes). MemAvailable ~3.3 GB (box-wide; our job ~0.7 GB), just under the 3500 MB floor, so the GNN-SNAP queue (R_g4_snapF_val first) is correctly holding; it frees ~0.7 GB when ctrl ends, which should clear the gate. Floor left unchanged (shared 16 GB box, no swap). PID 2888494 and *.STOP untouched. No new n=20 result; best GNN-SNAP stays S_g4_snapF_f 37.3/211.7.

### S_g4_ctrl_f finished 2026-10-18 15:46 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True snap=none snap_form=lin free_m=32 warm=results/v3/B_g4_iii_A_f.json warm_ids=[2, 0, 1] ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_ctrl_f winner 2 (train 1.378, val 2.086) | 3.6/7.2 | 6.6/15.9 | 9.5/30.0 | 14.2/42.7 | 63.3/253.0 |
| S_g4_ctrl_f trainbest 1 (train 1.370, val 2.122) | 3.6/7.0 | 6.5/16.0 | 8.6/37.2 | 14.0/40.2 | 73.0/255.9 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 63.3 vs 84.9, B90 253.0 vs 238.1 -> no beat

### R_g4_snapF_val finished 2026-10-18 15:47 CST  (L=4)

`L=4 src=['S_g4_snapF_f'] nval=600 vn=[10, 12] vbase=36000000000 criterion=min mean_n[(log median 1/p + log B90 1/p)/2] over fresh val n=[10, 12] (600 inst each, base 36000000000); tie-break mean -log p`

| row | n=20 |
|---|---|
| R_g4_snapF_val winner 0 (train 1.183, val 2.448) | 37.3/211.7 |
| QAOA_raw L=4 | 84.9/238.1 |
| G_fe L=4 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 37.3 vs 84.9, B90 211.7 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-18 15:55 CST)
Runner alive (run_v3.sh pid 3668898). 3 GNN-SNAP jobs running since 15:40–15:44 (each ~0.7 GB RSS, ~1.8 cores): **S_g8_snapF_A_f** (pid 3675472, gray L8 warm A_g8 keep), **S_b8_snapF_A_f** (pid 3676022, binary L8 warm A_b8 keep), **S_g4_snapF_s1** (pid 3676466, seed 1). All on init 0, no iterations logged yet. Load ~10 on 8 cores; MemAvailable ~2.5 GB (box-wide), under the 3500 MB floor, so S_g4_snapF_s2 and the rest correctly hold. Floor unchanged. PID 2888494 and *.STOP untouched.
- **S_g4_ctrl_f** final (15:46, above): no-SNAP ECD control n=20 63.3/253.0 (no beat) vs GNN-SNAP 37.3/211.7 on the same warm start; confirms the gain is from per-instance SNAP.
- **R_g4_snapF_val** (600-instance fresh val at n=10/12) re-picked init 0 = the same init as before, so n=20 stays **37.3/211.7** (beats QAOA L4 both; already reported, not a new result).
No new n=20 result. Queue unchanged.

## 16:09 CST architecture scan (approved)

Zach approved Phase 1 (network-architecture scan). Implemented and queued 2026-10-18 ~16:14 CST.

**Rules unchanged:** train only at n=8, L-BFGS-B, no QAOA-angle starts, SNAP zero-init nothing forced, formula features, free-SNAP + clause_w + var_mix heads.

**Code**
- `v3/arch_nets.py` — backbones: mp / mp128 / attn / tf / ds / mlp (same heads).
- `v3/gnn_snap.py` — new `--arch` flag; default `mp` is code-identical to prior GNN-SNAP (89596 params; S_g4_snapF_f still loads).
- `v3/reselect_arch.py` — multi-arch validation at n=10/14/20 (600/300/100), VBASE=37e9; criterion fixed before TEST; TEST = winner only on TEB n=20.
- `v3/analyze_arch.py` — scaffolding: SNAP/weight correlations, transfer n=8→14/20, B90-tail traits, ablation hooks.
- `v3/run_v3.sh` — `done_()` accepts winner+any test n (Phase-1 jobs omit n=20); runner restarted (jobs kept).

**Queue** (backup `v3/queue_v3.bak_1609`); Phase 1 at FRONT:
1. N_g4_mp128, N_g4_attn, N_g4_tf, N_g4_ds, N_g4_mlp (each 3 inits, init_from A_g4, Gray L=4, free SNAP)
2. R_arch_scan (srcs include baseline **S_g4_snapF_f ≡ N_g4_mp64**)
3. A_arch_scan (analysis)
4. Then existing GNN-SNAP (L8 / L4 seeds / re-pick / lin SNAP / controls) — running jobs **not killed**

**Smoke (16:12 CST):** all 6 arches SNAP≡ECD(β=0) at step 0; param counts mp=89596, attn=187900, tf=118524, ds=40316, mlp=32508, mp128=539900.

**Running at queue rewrite:** S_g8_snapF_A_f, S_b8_snapF_A_f, S_g4_snapF_s1, S_g4_snapF_s2 (MAXJ=4). Phase-1 launches when a slot + ≥3500 MB free.

**ETA:** first arch (N_g4_*) ~1–2 h after launch (3× BFGS @ n=8); full Phase-1 comparison overnight (~7–9 h wall with 1–2 concurrent jobs). Official pick = R_arch_scan leaderboard + TEB n=20 of winner only.


## Check-in (2026-10-18 16:17 CST)
Runner alive (run_v3.sh pid 3688928), 4/4 slots busy, CPU advancing. MemAvailable ~5.5 GiB, load ~14 on 8 cores, caps hold. PID 2888494 and *.STOP untouched. No new n=20 finish this check.

**Running (all init 0 done → mid init 1):**
- **S_g8_snapF_A_f** (gray L8 free SNAP from A_g8): init 0 **0.953→0.673** (1000 it, ~33 min). Promising vs L4 train ~1.18.
- **S_b8_snapF_A_f** (binary L8): init 0 **0.953→0.676**.
- **S_g4_snapF_s1** (seed 1, warm B_g4 #2/0/1): init 0 **1.415→1.191** (near S_g4_snapF_f's 1.183).
- **S_g4_snapF_s2** (seed 2, warm #2/0): init 0 **1.415→1.193**.

**Phase 1 arch scan** still at queue front (N_g4_mp128/attn/tf/ds/mlp → R_arch_scan → A_arch_scan); launches when a slot frees (s1/s2 L4 should finish first). Best dual-beat remains **S_g4_snapF_f 37.3/211.7** vs QAOA L4 84.9/238.1. No queue rewrite.

## Check-in (2026-10-18 16:35 CST)
Runner alive (run_v3.sh pid 3688928), 4/4 slots busy, CPU advancing (~150% each). MemAvailable ~2.4 GiB (below 3500 MB gate), load ~12–15 on 8 cores, caps hold — Phase-1 N_g4_* correctly waits until a slot frees. PID 2888494 and *.STOP untouched. No new n=20 finish this check.

**Running:**
- **S_g8_snapF_A_f** (gray L8 free SNAP from A_g8): init 0 **0.953→0.673** (1000 it); mid init 1 of 3. Json has 1 run.
- **S_b8_snapF_A_f** (binary L8): init 0 **0.953→0.676**; mid init 1 of 3.
- **S_g4_snapF_s1** (seed 1): init 0 **1.415→1.191**; init 1 **1.408→1.161** (near prior train-best 1.155); mid init 2 of 3.
- **S_g4_snapF_s2** (seed 2): init 0 **1.415→1.193**; mid init 1 of 2.

**Phase 1 arch scan** still at queue front (N_g4_mp128/attn/tf/ds/mlp → R_arch_scan → A_arch_scan). L4 seeds should free first (~16:55–17:05); L8 inits ~34 min each. Best dual-beat remains **S_g4_snapF_f 37.3/211.7** vs QAOA L4 84.9/238.1. No queue rewrite.


### S_g4_snapF_s2 finished 2026-10-18 16:59 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True snap=std snap_form=free free_m=32 warm=results/v3/B_g4_iii_A_f.json warm_ids=[2, 0] ninit=2 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula seed=2 chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_snapF_s2 winner 1 (train 1.142, val 1.890) | 2.9/8.5 | 5.3/18.8 | 7.0/33.4 | 12.3/82.8 | 40.3/255.0 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 40.3 vs 84.9, B90 255.0 vs 238.1 -> no beat

## Check-in (2026-10-18 16:58 CST)
Runner alive (run_v3.sh pid 3688928), 4/4 slots busy, CPU advancing. MemAvailable ~5.5 GiB, load ~12–15 on 8 cores, caps hold. PID 2888494 and *.STOP untouched.

**Finished:** **S_g4_snapF_s2** (seed 2, warm B_g4 #2/0) n=20 **40.3/255.0** — beats QAOA L4 median (84.9) but loses B90 (238.1); worse than best dual-beat **S_g4_snapF_f 37.3/211.7**. Early finalize had raced before n=20 was written; LOG corrected and `done_()` now requires n=20 when `args.test` includes 20.

**Running:**
- **S_g8_snapF_A_f**: init 0 **0.953→0.673**; mid init 1 of 3.
- **S_b8_snapF_A_f**: init 0 **0.953→0.676**; mid init 1 of 3.
- **S_g4_snapF_s1**: inits 0–1 done (1.191 / **1.161**); mid init 2 of 3.
- **S_g4_snapL_f**: just launched (lin SNAP, warm B_g4 #2/0/1).

**Phase 1 arch scan** still at queue front (N_g4_mp128/attn/tf/ds/mlp → R_arch_scan → A_arch_scan); next free slot + ≥3500 MB launches N_g4_mp128 (s1 L4 should free first). Queue unchanged. No WakeParent (no new dual-beat / no crash).


### S_g4_snapF_s1 finished 2026-10-18 17:14 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 clause_w=True var_mix=True snap=std snap_form=free free_m=32 warm=results/v3/B_g4_iii_A_f.json warm_ids=[2, 0, 1] ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula seed=1 chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_snapF_s1 winner 0 (train 1.191, val 1.882) | 2.9/8.9 | 5.2/17.9 | 6.4/38.9 | 11.1/80.8 | 37.5/182.7 |
| S_g4_snapF_s1 trainbest 1 (train 1.161, val 1.897) | 2.9/9.6 | 5.4/20.5 | 6.5/40.3 | 12.2/72.2 | 37.2/234.7 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 37.5 vs 84.9, B90 182.7 vs 238.1 -> **BEATS QAOA (both)**
n=20 trainbest vs QAOA L=4: median 37.2 vs 84.9, B90 234.7 vs 238.1 -> median beat only (B90 close)

## Check-in (2026-10-18 17:16 CST)
Runner alive (run_v3.sh pid 3704086, restarted 16:59 with done_ fix), 4/4 slots busy (s1 process still finishing trainbest tests). MemAvailable ~2.7 GiB, load ~12–15 on 8 cores. PID 2888494 and *.STOP untouched.

**New best dual-beat: S_g4_snapF_s1** (seed 1, warm B_g4 #2/0/1, winner init 0 picked on old n=10/12 val) n=20 L4 **37.5/182.7** vs QAOA 84.9/238.1 — B90 improves on S_g4_snapF_f (37.3/211.7) by ~14%, median about equal. Caveat: winner chosen by n=10/12 val, not the newer n=10/14/20 criterion; re-pick of s1 starts should be folded into R_arch_scan/reselect before calling it official.

**Running:** S_g8_snapF_A_f (inits 0.673/0.686, init 2 of 3), S_b8_snapF_A_f (0.676/0.671, init 2 of 3), S_g4_snapL_f (lin SNAP, init 0). Phase 1 arch scan (N_g4_mp128 first) launches on next free slot + ≥3500 MB. Queue unchanged.

## Check-in (2026-10-18 17:34 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing. MemAvailable ~4.7 GiB, load high on 8 cores, MINMEM/MAXJ caps hold. PID 2888494 and *.STOP untouched. No new n=20 finish this check (s1 trainbest n=10–20 now complete on disk; winner unchanged 37.5/182.7).

**Running:**
- **S_g8_snapF_A_f**: inits 0–1 done (0.673 / 0.686); mid init 2 of 3. Train loss ~0.67 vs L4 ~1.18 — promising for L8 B90 gap.
- **S_b8_snapF_A_f**: inits 0–1 done (0.676 / 0.671); mid init 2 of 3.
- **S_g4_snapL_f** (lin SNAP): init 0 **1.415→1.218** (worse train than free SNAP ~1.18); shared SNAP sa ~0.003–0.019 rad (near-zero, like prior shared-SNAP controls). Mid init 1 of 3.
- **N_g4_mp128** (Phase 1 deeper/wider GNN): started 17:15, 539900 params, still on init 0 (no loss lines yet).

**Queue:** Phase 1 continues (attn/tf/ds/mlp → R_arch_scan → A_arch_scan) ahead of R_g4_snapF_val2 and L8 no-SNAP controls. Best dual-beat remains **S_g4_snapF_s1 37.5/182.7** (caveat: old n=10/12 val pick). No queue rewrite. No WakeParent (s1 beat already logged/pushed at 17:14–17:16).

### S_g8_snapF_A_f finished 2026-10-18 17:55 CST  (L=8)

`code=gray L=8 hidden=64 rounds=3 clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g8.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g8_snapF_A_f winner 1 (train 0.686, val 1.102) | 1.7/3.8 | 2.3/7.1 | 3.1/10.6 | 4.5/18.8 | 9.1/51.6 |
| S_g8_snapF_A_f trainbest 2 (train 0.651, val 1.154) | 1.8/4.5 | 2.5/8.2 | 3.1/17.0 | 4.6/24.7 | 9.3/58.8 |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 9.1 vs 16.4, B90 51.6 vs 54.2 -> **BEATS QAOA (both)**

### S_b8_snapF_A_f finished 2026-10-18 17:59 CST  (L=8)

`code=binary L=8 hidden=64 rounds=3 clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_b8.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_b8_snapF_A_f winner 1 (train 0.671, val 1.119) | 1.7/4.9 | 2.5/8.3 | 2.9/14.7 | 4.7/20.2 | 8.1/74.7 |
| S_b8_snapF_A_f trainbest 2 (train 0.649, val 1.126) | 1.8/5.0 | 2.4/6.8 | 2.9/12.9 | 4.8/23.2 | 9.2/87.7 |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 8.1 vs 16.4, B90 74.7 vs 54.2 -> no beat

## Check-in (2026-10-18 18:02 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing. MemAvailable ~4.0 GiB, load high on 8 cores, MINMEM/MAXJ caps hold. PID 2888494 and *.STOP untouched.

**New L8 dual-beat: S_g8_snapF_A_f** (gray L8 free SNAP, warm A_g8 keep) n=20 **9.1/51.6** vs QAOA L8 **16.4/54.2** — beats both (median −44%, B90 −4.8%). Trainbest init 2 is 9.3/58.8 (median beat only). Closes the prior L8 B90 gap (plain A_b8 was 14.6/119.3 median-only). Winner picked on old n=10/12 val (init 1); trainbest lost B90 — same caveat as L4 s1.

**S_b8_snapF_A_f** (binary L8 free SNAP, warm A_b8): n=20 **8.1/74.7** — beats QAOA median, loses B90 (74.7 vs 54.2). Better median than gray S_g8 but worse tail; improves on plain A_b8 B90 119.3 → 74.7 still short of QAOA. Trainbest 9.2/87.7.

**Running:**
- **S_g4_snapL_f** (lin SNAP): inits 0–1 done (1.218 / **1.180**); mid init 2. Shared sa still ~0.01–0.04 rad.
- **N_g4_mp128** (Phase 1, 539900 params): still init 0 (~45+ min, no loss line yet — expected slower).
- **N_g4_attn** (187900 params): launched 17:56, init 0.
- **N_g4_tf** (118524 params): launched 18:01, init 0.

**Queue next:** N_g4_ds, N_g4_mlp → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Best dual-beats: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. WakeParent for the new L8 dual-beat.

## Check-in (2026-10-18 18:19 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing (~134–195% each). MemAvailable ~4.2 GiB (above 3500 MB gate; was ~2.5 GiB earlier this check), load high on 8 cores, MINMEM/MAXJ caps hold. PID 2888494 and *.STOP untouched. No new finish since 18:02 check-in; no queue rewrite; no restart.

**Running:**
- **S_g4_snapL_f** (lin SNAP, warm B_g4 #2/0/1): inits 0–1 done (1.218 / **1.180**); mid init 2 of 3 (~50 min elapsed). Train still near free-SNAP 1.18 on init 1; shared SNAP phases remain small in prior inits.
- **N_g4_mp128** (Phase 1, 539900 params): init 0 **1.5703→1.1489** (1000 it, 2848 s); mid init 1 of 3. Train better than baseline free-SNAP ~1.18 so far — wait for all 3 inits + R_arch_scan before ranking.
- **N_g4_attn** (187900 params): ~15 min elapsed, still init 0 (no loss line yet; CPU ~153%, RSS ~0.8 GB).
- **N_g4_tf** (118524 params): ~11 min elapsed, still init 0 (CPU ~134%, RSS ~0.75 GB).

**Queue next:** N_g4_ds, N_g4_mlp → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6** (both already logged/pushed). No WakeParent this check.

### S_g4_snapL_f finished 2026-10-18 18:21 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=mp clause_w=True var_mix=True snap=std snap_form=lin free_m=32 warm=results/v3/B_g4_iii_A_f.json warm_ids=[2, 0, 1] ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g4_snapL_f winner 2 (train 1.212, val 1.870) | 3.0/8.6 | 4.9/17.3 | 6.5/51.2 | 11.6/78.7 | 37.8/455.5 |
| S_g4_snapL_f trainbest 1 (train 1.180, val 1.875) | 2.8/8.0 | 4.9/18.2 | 6.3/37.2 | 11.7/78.1 | 37.1/441.4 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 37.8 vs 84.9, B90 455.5 vs 238.1 -> no beat
n=20 trainbest vs QAOA L=4: median 37.1 vs 84.9, B90 441.4 vs 238.1 -> median beat only (B90 ~2× QAOA)

(Finalize raced: trainbest n=20 landed after report_v3; row patched 18:35 CST. Shared lin sa stayed ~0.01–0.04 rad; per-instance SNAP phase-rms at n=20 ~0.57 — lin form hurts B90 vs free SNAP.)

## Check-in (2026-10-18 18:35 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing (~117–194% each). MemAvailable ~2.5 GiB (below 3500 MB gate) — correctly holding **N_g4_mlp**. Load ~18 on 8 cores. PID 2888494 and *.STOP untouched. No new dual-beat this check.

**Finished (already tabled above; trainbest n=20 patched):** **S_g4_snapL_f** n=20 winner **37.8/455.5**, trainbest **37.1/441.4** — median beats QAOA L4 (84.9) but B90 ~2× worse (238.1). Worse than free-SNAP bests (s1 37.5/182.7, snapF_f 37.3/211.7). Lin SNAP not competitive on tail.

**Phase 1 running (all L4 free SNAP, init A_g4, formula):**
- **N_g4_mp128** (539900 params): init 0 **1.570→1.149** (1000 it, 2848 s); mid init 1 of 3 (~45 min elapsed on job; ~1.0 GB RSS).
- **N_g4_attn** (187900): ~20 min, still init 0 (empty loss line; ~0.86 GB).
- **N_g4_tf** (118524): ~16 min, still init 0 (~0.67 GB).
- **N_g4_ds** (40316): launched 18:22, init 0 (~0.64 GB).

**Queued next:** N_g4_mlp → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Baseline for scan = S_g4_snapF_f ≡ mp64. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No queue rewrite; no restart; no WakeParent.

## Check-in (2026-10-18 18:56 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing (~118–194% each). MemAvailable ~2.7 GiB (below 3500 MB gate) — correctly holding **N_g4_mlp**. Load ~20 on 8 cores. PID 2888494 and *.STOP untouched. No new finish since S_g4_snapL_f (18:21); no queue rewrite; no restart.

**Phase 1 running (all L4 free SNAP, init A_g4, formula, test 8/10/12/14):**
- **N_g4_mp128** (539900 params, ~1.06 GB RSS, ~55 min): init 0 **1.570→1.149** (1000 it, 2848 s); mid init 1 of 3. Train still better than baseline free-SNAP ~1.18 on init 0.
- **N_g4_attn** (187900, ~0.86 GB, ~60 min): init 0 **1.570→1.304** (1000 it, 3563 s) — worse train than mp64 baseline ~1.18; mid init 1 of 3.
- **N_g4_tf** (118524, ~0.74 GB, ~55 min): still init 0 (empty loss line; CPU ~136%).
- **N_g4_ds** (40316, ~0.58 GB, ~34 min): header logged; mid init 0.

**Queued next:** N_g4_mlp (gated on mem+slot) → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Baseline for scan = S_g4_snapF_f ≡ mp64. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent (no new n=20 dual-beat / no crash).

## Check-in (2026-10-18 19:19 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing (~119–194% each). MemAvailable ~2.7 GiB (below 3500 MB gate) — correctly holding **N_g4_mlp**. Load ~14–17 on 8 cores. PID 2888494 and *.STOP untouched. No new finish since S_g4_snapL_f (18:21); no queue rewrite; no restart.

**Phase 1 running (all L4 free SNAP, init A_g4, formula, test 8/10/12/14):**
- **N_g4_mp128** (539900 params, ~1.05 GB RSS, ~66 min): init 0 **1.570→1.149** (1000 it, 2848 s); init 1 **1.570→1.160** (1000 it, 3634 s); mid init 2 of 3. Both finished inits beat baseline free-SNAP train ~1.18 so far.
- **N_g4_attn** (187900, ~0.87 GB, ~42 min): init 0 **1.570→1.304** (1000 it, 3563 s) — worse than mp64 ~1.18; mid init 1 of 3.
- **N_g4_tf** (118524, ~0.74 GB, ~38 min): init 0 **1.570→1.405** (1000 it, 3856 s) — worse train; mid init 1 of 3.
- **N_g4_ds** (40316, ~0.60 GB, ~23 min): init 0 **1.570→1.438** (1000 it, 2227 s) — worst train so far; mid init 1 of 3.

**Early train read (not official — wait for all 3 inits + R_arch_scan):** mp128 best so far; attn/tf/ds lag baseline on n=8 loss. Official pick remains R_arch_scan (n=10/14/20 val) + TEB n=20 of winner only.

**Queued next:** N_g4_mlp (gated on mem+slot) → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Baseline for scan = S_g4_snapF_f ≡ mp64. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent (no new n=20 dual-beat / no crash).

## Check-in (2026-10-18 19:39 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing (~118–193% each). MemAvailable fluctuated ~2.3→3.9 GiB this check (above/below 3500 MB gate); **N_g4_mlp** still held by MAXJ=4 (no free slot). Load ~16 on 8 cores. PID 2888494 and *.STOP untouched. No new finish since S_g4_snapL_f (18:21); no queue rewrite; no restart.

**Phase 1 running (all L4 free SNAP, init A_g4, formula, test 8/10/12/14):**
- **N_g4_mp128** (539900 params, ~1.0 GB RSS, ~72 min): init 0 **1.570→1.149** (1000 it, 2848 s); init 1 **1.570→1.160** (1000 it, 3634 s); mid init 2 of 3. Both finished inits still beat baseline free-SNAP train ~1.18.
- **N_g4_attn** (187900, ~0.88 GB, ~47 min): init 0 **1.570→1.304** (1000 it, 3563 s) — worse than mp64 ~1.18; mid init 1 of 3.
- **N_g4_tf** (118524, ~0.72 GB, ~44 min): init 0 **1.570→1.405** (1000 it, 3856 s) — worse train; mid init 1 of 3.
- **N_g4_ds** (40316, ~0.62 GB, ~29 min): init 0 **1.570→1.438** (1000 it, 2227 s) — worst train so far; mid init 1 of 3.

**Early train read (not official — wait for all 3 inits + R_arch_scan):** mp128 best so far; attn/tf/ds lag baseline on n=8 loss. Official pick remains R_arch_scan (n=10/14/20 val) + TEB n=20 of winner only.

**Queued next:** N_g4_mlp (needs free slot + mem≥3500) → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Baseline for scan = S_g4_snapF_f ≡ mp64. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent (no new n=20 dual-beat / no crash).

## Check-in (2026-10-18 19:54 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing. MemAvailable ~2566.7 GiB (below 3500 MB gate) — correctly holding **N_g4_mlp** (also no free slot). Load ~13.76 on 8 cores. PID 2888494 and *.STOP untouched. No new finish since S_g4_snapL_f (18:21); no queue rewrite; no restart.

**Phase 1 running (all L4 free SNAP, init A_g4, formula, test 8/10/12/14):**
- **N_g4_mp128** (539900 params, ~1.02 GB RSS): init 0 **1.570→1.149** (1000 it, 2848 s); init 1 **1.570→1.160** (1000 it, 3634 s); mid init 2 of 3. Both finished inits still beat baseline free-SNAP train ~1.18.
- **N_g4_attn** (187900, ~0.88 GB): init 0 **1.570→1.304** (1000 it, 3563 s) — worse than mp64 ~1.18; mid init 1 of 3.
- **N_g4_tf** (118524, ~0.71 GB): init 0 **1.570→1.405** (1000 it, 3856 s) — worse train; mid init 1 of 3.
- **N_g4_ds** (40316, ~0.62 GB): init 0 **1.570→1.438** (1000 it, 2227 s); init 1 **1.570→1.441** (1000 it, 2443 s); mid init 2 of 3 — worst train so far.

**Early train read (not official — wait for all 3 inits + R_arch_scan):** mp128 best so far; attn/tf/ds lag baseline on n=8 loss. Official pick remains R_arch_scan (n=10/14/20 val) + TEB n=20 of winner only.

**Queued next:** N_g4_mlp (needs free slot + mem≥3500) → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Baseline for scan = S_g4_snapF_f ≡ mp64. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent (no new n=20 dual-beat / no crash).

## Check-in (2026-10-18 20:17 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy (mp128 ~193%, attn ~156%, tf ~137%, ds ~117% CPU). MemAvailable ~4 GiB but no free slot (MAXJ=4) — **N_g4_mlp** still queued. Load ~14 on 8 cores. PID 2888494 and *.STOP untouched. No new finish since S_g4_snapL_f (18:21); no new init lines since 19:54 check (mp128 init 2, attn/tf init 1, ds init 2 in progress). No queue rewrite; no restart.

Queued next unchanged: N_g4_mlp → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent.

## Check-in (2026-10-18 20:39 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy, CPU advancing. MemAvailable ~2.4 GiB (below 3500 MB gate) and no free slot — **N_g4_mlp** still queued. Load ~14 on 8 cores. PID 2888494 and *.STOP untouched. No new finish since S_g4_snapL_f (18:21); no queue rewrite; no restart.

New init line: **N_g4_attn** init 1 **1.570→1.308** (1000 it, 4994 s) — same as init 0 (1.304), still well behind mp64 ~1.18. Others unchanged: mp128 inits 1.149/1.160 (init 2 running), tf 1.405 (init 1 running), ds 1.438/1.441 (init 2 running, should finish soon).

Queued next unchanged: N_g4_mlp → R_arch_scan → A_arch_scan, then R_g4_snapF_val2 / L8 no-SNAP controls. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent.

### N_g4_ds finished 2026-10-18 20:43 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=ds clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_ds winner 2 (train 1.436, val 2.020) | 4.1/8.1 | 6.1/18.8 | 9.2/32.4 | 14.2/58.2 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

## Check-in (2026-10-18 20:56 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy. **N_g4_ds** finished 20:43 (winner init 2, train 1.436, val 2.020; n=14 14.2/58.2 vs QAOA L4 20.8/59.6 — beats QAOA at n≤14 but train loss clearly worst of the scan, ~1.44 vs mp64 ~1.18). Runner then launched **N_g4_mlp** at 20:43 (mem 5650 MB). MemAvailable ~5.2 GiB, load ~14.5 on 8 cores. PID 2888494 and *.STOP untouched. No queue rewrite; no restart.

Running: mp128 init 2 (inits 1.149/1.160), attn init 2 (1.304/1.308), tf init 1 (1.405), mlp init 0. Queued next unchanged: R_arch_scan → A_arch_scan (n=10/14/20 val pick), then R_g4_snapF_val2 / L8 no-SNAP controls. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent.

### N_g4_mp128 finished 2026-10-18 20:57 CST  (L=4)

`code=gray L=4 hidden=128 rounds=5 arch=mp128 clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_mp128 winner 0 (train 1.149, val 1.904) | 3.1/8.2 | 5.2/18.2 | 7.1/36.8 | 12.2/79.0 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

### R_g4_snapF_val2 finished 2026-10-18 21:00 CST  (L=4)

`L=4 src=['S_g4_snapF_f', 'S_g4_snapF_s1', 'S_g4_snapF_s2'] nval=600 vn=[10, 12] vbase=36000000000 criterion=min mean_n[(log median 1/p + log B90 1/p)/2] over fresh val n=[10, 12] (600 inst each, base 36000000000); tie-break mean -log p`

| row | n=20 |
|---|---|
| R_g4_snapF_val2 winner 3 (train 1.191, val 2.440) | 37.5/182.7 |
| QAOA_raw L=4 | 84.9/238.1 |
| G_fe L=4 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 37.5 vs 84.9, B90 182.7 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-18 21:12 CST)
Runner alive (run_v3.sh pid 3704086), 4/4 slots busy: N_g4_attn (init 2), N_g4_tf (init 1+), N_g4_mlp (init 0), B_b8_iii_A_f (L8 binary no-SNAP control, launched 21:00). MemAvailable ~2.8 GiB, load ~13 on 8 cores. PID 2888494 and *.STOP untouched. No queue rewrite; no restart.

New finishes since 20:56:
- **N_g4_mp128** (20:57): winner init 0, train **1.149** (best n=8 loss of the scan, below mp64 ~1.18), val 1.904. But it generalizes worse at larger n: n=14 **12.2/79.0** vs QAOA L4 20.8/59.6 (median better, B90 worse; ds got 14.2/58.2). Hunch: the wider/deeper net overfits n=8 structure, hurting the tail. Official comparison waits for R_arch_scan (n=10/14/20 val).
- **R_g4_snapF_val2** (21:00): 600-instance n=10/12 re-pick across S_g4_snapF_f/s1/s2 picks the same start as before; n=20 **37.5/182.7** = existing best S_g4_snapF_s1 (not a new result).

Queued next unchanged: R_arch_scan (after attn/tf/mlp) → A_arch_scan, then B_g8_iii_A_f / B_g8_i_A_f controls. Improvement idea held for after R_arch_scan: if mp128's tail problem holds at n=20, try mp64 depth/width in between or weight decay, within Phase 2. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent.

### N_g4_attn finished 2026-10-18 21:16 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=attn clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_attn winner 1 (train 1.308, val 1.886) | 3.1/7.9 | 5.3/15.3 | 7.5/30.7 | 12.8/50.2 |
| N_g4_attn trainbest 2 (train 1.300, val 1.886) | 3.1/7.6 | 5.3/14.3 | - | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

## Classical baselines (Sun 21:45 CST)  [results/v3/classical_baselines.json, code v3/classical/]

Exact v3 TEST instances: `sim.instances(n, N, TEB=32e9)` (N=200 at n=14, else 100) — same set as every v3 n=20 report. Target = all satisfying strings (E==0), same success set as `v3core.succ_D`. Per instance: mean over 200 independent runs of cost to first hit; then median/B90 across instances. Tuning only on TRAIN (base 31e9). WalkSAT = SKC, p=0.5 (best on n=20 train; n=8 train degenerate, prefers p=0.7). SA = single-flip Metropolis, linear beta, restart until hit.
**Units differ:** quantum = 1/p (expected shots); classical flips = accepted single-var moves; evals = WalkSAT 1+4/step break-count evals, SA 1/proposal; random = 2^n/nsol samples.

| method | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| GNN-SNAP L4 (shots) | 2.9/8.9 | 5.2/17.9 | 6.4/38.9 | 11.1/80.8 | **37.5/182.7** |
| GNN-SNAP L8 (shots) | 1.7/3.8 | 2.3/7.1 | 3.1/10.6 | 4.5/18.8 | **9.1/51.6** |
| QAOA L4 / L8 (shots) | 5.3/9.9 / 2.5/4.5 | 8.4/19.7 / 3.5/7.3 | 12.8/32.8 / 4.6/15.5 | 20.8/59.6 / 6.3/18.3 | 84.9/238.1 / 16.4/54.2 |
| random guess (samples) | 128/256 | 512/1024 | 2048/4096 | 8192/16384 | 349525/1048576 |
| WalkSAT p=.5 flips | 15.1/43.5 | 28.5/61.2 | 43.7/118.1 | 60.8/148.6 | 135.1/383.6 |
| WalkSAT p=.5 evals | 61.4/175.0 | 115.1/245.7 | 175.8/473.2 | 244.2/595.4 | 541.5/1535.3 |
| SA (n=8-train tuned) flips | 30.8/59.8 | 58.9/103.3 | 97.5/219.3 | 157.3/343.3 | 627.3/1519.7 |
| SA (n=8-train tuned) evals | 84.7/173.9 | 163.2/309.7 | 282.8/658.7 | 458.1/1032.2 | 1900.9/4572.6 |
| SA (n=20-train tuned) flips | 34.6/83.8 | 56.3/117.3 | 81.9/228.5 | 123.6/269.8 | 288.8/770.3 |
| SA (n=20-train tuned) evals | 103.8/319.0 | 190.9/469.9 | 299.2/937.6 | 472.2/1239.8 | 1222.1/3567.4 |

Per-instance Spearman at n=20 (GNN 1/p vs classical cost): L4 0.83 vs WalkSAT flips, 0.72 vs 1/nsol; L8 0.87 / 0.74 — the same instances are hard for both.
Caveats: units are not equivalent (a shot is one full L-layer circuit + measurement; a flip is O(degree~40) clause updates). Only cost-to-solution counts compared, no wall-clock/hardware time. Scaling n=8→20 of median: WalkSAT flips ×9, GNN L8 ×5.4, GNN L4 ×13, QAOA L4 ×16. Earlier "SATLike 124/530 at n=20" row was a fit on a different instance family; WalkSAT here is measured on the actual test set (135/384 flips).

## Check-in (2026-10-18 21:33 CST)
Runner alive (run_v3.sh pid 3704086), 3/4 slots busy: N_g4_tf (init 2 of 3), N_g4_mlp (init 2 of 3), B_b8_iii_A_f (L8 binary no-SNAP control). MemAvailable ~2.5 GiB (below 3500 MB gate) — runner correctly holding B_g8_iii_A_f / B_g8_i_A_f / B_g4_iii_A_h32_f / B_g4_iii_A_h128_f. Load ~12.6 on 8 cores. PID 2888494 and *.STOP untouched. No queue rewrite; no restart.

New finish since 21:12: **N_g4_attn** (21:16) winner init 1, train 1.308 (worse n=8 fit than mp64 ~1.18 / mp128 1.149), val 1.886 (best n=10/12 val of the scan so far). n=14 **12.8/50.2** vs QAOA L4 20.8/59.6 — beats QAOA on both at n=14 and has the best n=14 B90 of the scan (mp128 79.0, ds 58.2). Read so far: lower n=8 train loss does not translate to better large-n tails (mp128 fits best, generalizes worst); attention's weaker fit but better tail supports the overfit hunch. Official pick still waits for R_arch_scan (n=10/14/20 val).

Progress: N_g4_tf inits 1.405/1.414 (worst-but-ds fit), N_g4_mlp inits 1.424/1.442. R_arch_scan launches after tf + mlp finish (~1–2 h). Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. No WakeParent.

### N_g4_tf finished 2026-10-18 21:40 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=tf clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_tf winner 0 (train 1.405, val 2.004) | 3.6/6.9 | 6.5/13.8 | 8.0/29.5 | 13.7/43.3 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

### N_g4_mlp finished 2026-10-18 21:40 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=mlp clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_mlp winner 2 (train 1.439, val 1.984) | 3.8/7.9 | 6.1/17.0 | 8.9/29.7 | 14.0/48.2 |
| N_g4_mlp trainbest 0 (train 1.424, val 1.995) | 3.9/7.3 | 6.2/16.9 | 8.8/31.3 | 15.0/45.4 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

## Check-in (2026-10-18 21:59 CST)
Runner alive (run_v3.sh pid 3704086). N_g4_tf and N_g4_mlp finished 21:40, so all five Phase-1 arch jobs are done and R_arch_scan became ready, but all 4 runner slots had already gone to lower-priority controls (B_b8_iii_A_f, B_g8_iii_A_f, B_g8_i_A_f, B_g4_iii_A_h32_f, launched 21:34–21:37 while R_arch_scan was still blocked). Since R_arch_scan is the Phase-1 decision and MemAvailable was ~5.2 GB (>3500 gate), launched it manually (same detached setsid nohup env as the runner; runner sees it via `--tag`).
Two crashes on launch, both in `v3/reselect_arch.py`, both fixed: (1) `import sim` failed because `sim.py` lives in `oct_explore/` and the script didn't add it to `sys.path`; (2) exec'ing the `gnn_snap.py` library part needs `--tag` (it was filtered out) — now passes a dummy `_lib_reselect_arch` tag (library only defines `save()`, never writes on import). Third launch 21:54 running fine: ~1 min per init, 18 inits, ETA ~22:15 then n=20 test of the winner; runner will finalize/push, then A_arch_scan.
First rows (S_g4_snapF_f, mp): S = 3.243 / 3.267 / 3.290; n=20 val 33.2/201.6, 36.4/178.1, 34.8/195.3. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7**, L8 **S_g8_snapF_A_f 9.1/51.6**. PID 2888494 and *.STOP untouched. No WakeParent.

### R_arch_scan finished 2026-10-18 22:22 CST  (L=4)

`L=4 src=['S_g4_snapF_f', 'N_g4_mp128', 'N_g4_attn', 'N_g4_tf', 'N_g4_ds', 'N_g4_mlp'] vn=[10, 14, 20] nval=[600, 300, 100] vbase=37000000000 criterion=min mean_n[(log median 1/p + log B90 1/p)/2] over fresh val n=[10, 14, 20] counts=[600, 300, 100] base=37000000000; tie-break mean -log p; TEST only winner on TEB n=20`

| row | n=20 |
|---|---|
| R_arch_scan winner 7 (train 1.308, val 3.184) | 42.8/172.7 |
| QAOA_raw L=4 | 84.9/238.1 |
| G_fe L=4 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 42.8 vs 84.9, B90 172.7 vs 238.1 -> **BEATS QAOA (both)**

## Phase 1 result (2026-10-18 22:22 CST) — attention wins

Official pick = `R_arch_scan` (fixed score S = mean_n[(log median + log B90)/2] on fresh val n=10/14/20 @ 600/300/100, VBASE=37e9; TEB n=20 of winner only).

**Winner:** cand 7 = **N_g4_attn#1** (arch=attn, train 1.308, S=3.1842).
**TEB n=20 L4:** **42.8/172.7** vs QAOA L4 **84.9/238.1** → **dual-beat** (median and B90).
vs prior best L4 dual-beat **S_g4_snapF_s1 37.5/182.7**: median worse (42.8 vs 37.5), **B90 better** (172.7 vs 182.7).

**Best-S per arch (val):**
| arch | best S | tag# | val n=20 |
|---|---|---|---|
| **attn** | **3.184** | N_g4_attn#1 | 34.2/149.4 |
| mp (baseline) | 3.243 | S_g4_snapF_f#0 | 33.2/201.6 |
| mp128 | 3.245 | N_g4_mp128#1 | 37.5/179.0 |
| mlp | 3.256 | N_g4_mlp#2 | 42.1/143.5 |
| tf | 3.262 | N_g4_tf#0 | 40.8/159.5 |
| ds | 3.348 | N_g4_ds#2 | 47.3/165.6 |

**Why attn:** top-3 leaderboard are all attn inits. Lower n=8 train loss (mp128 1.149, mp 1.18) did **not** win — attn's weaker fit (~1.30) generalizes to better n=14/20 tails (val B90 143–159 vs mp 178–202). mp128 overfits. ds/mlp/tf lag. Supports Phase-1 hypothesis that capacity≠generalization for free-SNAP heads.

**Phase 2 queued (22:25 CST):** improve attn past 37.3/211.7 (and toward/past s1 37.5/182.7):
- N_g4_attn_s1 / s2 (seeds 1/2, init_from A_g4, L4 free SNAP)
- N_g4_attn_r5 (rounds=5 deeper attn)
- S_g8_attn_A_f (L8 free SNAP attn from A_g8)
- R_attn_val (reselect across N_g4_attn + s1/s2/r5 on VBASE=38e9)
Rules unchanged: train n=8 only, no QAOA angles, SNAP zero-init, no forced ECD. Backup `v3/queue_v3.bak_2224`.

## Check-in (2026-10-18 22:25 CST)
Runner alive (run_v3.sh pid 3704086). R_arch_scan finished 22:22, finalized+pushed by runner (`577265c6`). 4/4 slots still on L8/h32 no-SNAP controls (B_b8_iii_A_f init 0–1 done; B_g8_iii_A_f / B_g8_i_A_f / B_g4_iii_A_h32_f mid-train). MemAvailable ~5.0 GiB after R_arch_scan freed ~2 GB. Next free slot → **A_arch_scan**, then Phase-2 attn jobs. PID 2888494 and *.STOP untouched.

**NEW dual-beat (WakeParent):** R_arch_scan TEB **N_g4_attn#1 42.8/172.7** vs QAOA L4 84.9/238.1. Best L4 dual-beats now: s1 **37.5/182.7** (best median), attn **42.8/172.7** (best B90). L8 unchanged **S_g8_snapF_A_f 9.1/51.6**.

### B_g4_iii_A_h32_f finished 2026-10-18 22:39 CST  (L=4)

`code=gray L=4 hidden=32 rounds=3 clause_w=True var_mix=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_iii_A_h32_f winner 2 (train 1.406, val 2.020) | 3.6/7.0 | 6.4/15.3 | 8.3/28.8 | 12.8/46.5 | 55.9/277.5 |
| B_g4_iii_A_h32_f trainbest 1 (train 1.397, val 2.022) | 3.6/7.5 | 6.2/15.1 | 8.5/28.1 | 13.3/40.8 | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 55.9 vs 84.9, B90 277.5 vs 238.1 -> no beat

## 22:45 CST core cap 6 (Zach)
- Zach: for the rest of the exploration use at most 6 cores (was 8) to leave room for other tasks.
- v3/run_v3.sh (backup v3/run_v3.sh.bak_2245): MAXJ 4->3, MAXGNN 4->3 (3 jobs x 2 threads = 6); every newly launched job is wrapped in `taskset -c 0-5`.
- The 4 jobs already running (B_b8_iii_A_f 3775185, B_g8_iii_A_f 3808009, B_g8_i_A_f 3808670, N_g4_attn_s1 3843966) were not killed; all their threads were pinned to cores 0-5 (`taskset -a -p -c 0-5`), so the campaign is on <=6 cores immediately. The runner will not launch again until fewer than 3 jobs run.
- Runner restarted 22:46 CST as PID 3845590 (old 3704086 stopped; it recognises running jobs via pgrep on --tag).

## 22:49 CST fix A_arch_scan --tag crash
- `v3/analyze_arch.py`: inject dummy `--tag _lib_analyze_arch` when execing gnn_snap lib (same as reselect_arch 21:58); smoke past arg parse OK; queue line unchanged, runner will relaunch.

### B_b8_iii_A_f finished 2026-10-18 22:54 CST  (L=8)

`code=binary L=8 hidden=64 rounds=3 clause_w=True var_mix=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_b8.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_b8_iii_A_f winner 0 (train 0.805, val 1.187) | 1.9/3.7 | 2.6/6.2 | 3.5/13.5 | 4.8/17.7 | 13.0/100.7 |
| B_b8_iii_A_f trainbest 2 (train 0.789, val 1.188) | 1.9/3.6 | 2.6/6.1 | 3.3/12.6 | 4.7/20.7 | - |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 13.0 vs 16.4, B90 100.7 vs 54.2 -> no beat

## Check-in (2026-10-18 22:56 CST)
Runner alive (run_v3.sh pid 3845590), MAXJ=3 / cores 0-5 (Zach 6-core cap). After B_b8_iii_A_f exited, **3/3 slots** busy: B_g8_iii_A_f (init 2 of 3), B_g8_i_A_f (init 2 of 3), N_g4_attn_s1 (Phase-2 attn seed, training). MemAvailable ~6.1 GiB. PID 2888494 and *.STOP untouched. No queue rewrite; no kill of L8 no-SNAP controls (let them finish; next free slot → A_arch_scan then Phase-2).

Finished since 22:45:
- **B_g4_iii_A_h32_f** (22:39): n=20 **55.9/277.5** vs QAOA L4 84.9/238.1 — median-only, no dual-beat (already in LOG).
- **B_b8_iii_A_f** (22:54): n=20 L8 **13.0/100.7** vs QAOA L8 16.4/54.2 — median beats QAOA, B90 loses badly; vs SNAP L8 best **9.1/51.6** much worse. Confirms L8 gain needs free SNAP. Runner finalized+pushed (`d79e6b79`).

**A_arch_scan:** crashed 22:40 on missing `--tag` (pre-fix); `analyze_arch.py` fixed 22:49 (`a6f66fb7`). No result JSON yet — runner will relaunch when n_run < 3 (after a B_g8 finishes). Phase-2 still queued: N_g4_attn_s2, N_g4_attn_r5, S_g8_attn_A_f, R_attn_val.

Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new dual-beat / no improve-over-best → no WakeParent.

### B_g8_i_A_f finished 2026-10-18 23:16 CST  (L=8)

`code=gray L=8 hidden=64 rounds=3 ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g8.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g8_i_A_f winner 0 (train 0.927, val 1.401) | 2.0/5.3 | 2.7/9.0 | 4.1/26.2 | 5.7/36.9 | 19.1/229.2 |
| B_g8_i_A_f trainbest 1 (train 0.917, val 1.430) | 2.0/5.5 | 2.9/9.7 | 4.3/24.2 | 5.9/44.7 | - |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 19.1 vs 16.4, B90 229.2 vs 54.2 -> no beat

### B_g8_iii_A_f finished 2026-10-18 23:19 CST  (L=8)

`code=gray L=8 hidden=64 rounds=3 clause_w=True var_mix=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g8.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g8_iii_A_f winner 2 (train 0.798, val 1.182) | 1.9/3.6 | 2.6/6.2 | 3.4/13.8 | 4.7/19.3 | 11.3/136.1 |
| B_g8_iii_A_f trainbest 1 (train 0.795, val 1.189) | 1.9/3.6 | 2.6/6.5 | 3.3/14.8 | 4.7/17.6 | - |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 11.3 vs 16.4, B90 136.1 vs 54.2 -> no beat

## Check-in (2026-10-18 23:23 CST)
Runner restarted 23:23 as pid **3872547** (was 3845590) after a double-launch bug: `running()` only matched `--tag`, so `A_arch_scan` (`analyze_arch.py`, no `--tag`) was relaunched at 23:22 while the 23:18 instance was still live. Patched `v3/run_v3.sh` (`bak_2322`): treat `pgrep analyze_arch.py` as running for tag `A_arch_scan`. Killed the duplicate (3872150); kept first A_arch (3869925). MAXJ=3 / cores 0-5 unchanged. PID 2888494 and *.STOP untouched.

**3/3 slots now:** N_g4_attn_s1 (Phase-2 seed1, init 0 done 1.286 @23:06; inits 1–2), A_arch_scan (relaunch 23:18, past the 22:49 `--tag` fix; probing arches), **N_g4_attn_s2** (just launched). MemAvailable ~5.1 GiB.

Finished since 22:56:
- **B_g8_i_A_f** (23:16): n=20 L8 **19.1/229.2** vs QAOA 16.4/54.2 — loses both (no-SNAP gray L8 without clause_w/var_mix).
- **B_g8_iii_A_f** (23:19): n=20 L8 **11.3/136.1** vs QAOA 16.4/54.2 — median beats QAOA, B90 loses; vs SNAP L8 best **9.1/51.6** worse on both. Same story as B_b8: L8 needs free SNAP.

Queued next: N_g4_attn_r5, S_g8_attn_A_f, R_attn_val (after s1/s2/r5). Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** / attn **42.8/172.7**; L8 **S_g8_snapF_A_f 9.1/51.6**. No new dual-beat improving on best → no WakeParent.

### A_arch_scan finished 2026-10-18 23:25 CST  (L=4)

`L=4 tags=['S_g4_snapF_f', 'N_g4_mp128', 'N_g4_attn', 'N_g4_tf', 'N_g4_ds', 'N_g4_mlp']`

| row | n=8 |
|---|---|
| A_arch_scan winner 0 | 0.0/0.0 |
| QAOA_raw L=4 | 5.3/9.9 |
| G_fe L=4 | 5.4/9.8 |

### A_arch_scan analysis (2026-10-18 23:25–23:38 CST)

Probed all 6 Phase-1 nets (`analyze_arch_summary.json`; n_ok=6). Stub TEB row is placeholder (analysis job, not a circuit test).

| arch | train loss | phase_rms n=20 | w_rms n=20 | transfer ph20/ph8 | TEB B90 (known) | tail nsol | body nsol |
|---|---|---|---|---|---|---|---|
| **attn** | 1.308 | **0.149** | 1.181 | **0.751** | **172.7** | 1.2 | 4.9 |
| mp (baseline) | 1.183 | 0.205 | 1.037 | 0.809 | 211.7 | 1.2 | 4.9 |
| mp128 | 1.149 | 0.194 | 1.249 | 0.834 | 242.7 | 1.4 | 4.9 |
| tf | 1.405 | 0.065 | 0.877 | 0.998 | ~152–160 val | 1.1 | 4.9 |
| ds | 1.436 | 0.139 | 0.828 | 0.799 | ~192 val | 1.5 | 4.9 |
| mlp | 1.439 | 0.124 | 0.667 | 0.788 | ~189 val | 1.5 | 4.9 |

**Takeaways (why keep attn for Phase 2):**
1. **attn damps SNAP phases most when scaling n=8→20** (transfer 0.75) yet still uses them (ph20=0.15 ≫ tf’s 0.065). mp/mp128 keep larger phases and worse B90 — overconfident SNAP at large n.
2. **Hard-tail instances are shared across arches** (tail mean nsol ~1.1–1.5 vs body ~4.9). Architecture does not invent easy tails; attn’s edge is better B90 on the same hard set.
3. **tf under-uses SNAP** (phase collapse) — explains why low B90 on val did not win the fixed score (median suffers).
4. Suggested ablations (degree-channel zero, sign shuffle, var permute, drop heads) recorded in summary; hold until after Phase-2 seed/r5/L8 attn + R_attn_val. Tail-weighted loss / more n=8 train still on the Phase-2 menu.

## Check-in (2026-10-18 23:38 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap). Re-pinned all threads of the 3 live jobs to `taskset -c 0-5` (s1 had idle PSR reports on 6/7; affinity mask already 3f). MemAvailable ~2.4 GiB (tight but stable; MINMEM=3500 so no 4th launch). PID 2888494 and *.STOP untouched. No old-queue restart.

**3/3 slots (Phase 2):**
- **N_g4_attn_s1** (pid 3843966, ~53 min): init 0 → 1.286, init 1 → 1.288; on init 2. Params 187900.
- **N_g4_attn_s2** (pid 3872605, ~13 min): training init 0.
- **N_g4_attn_r5** (pid 3874429, ~10 min, rounds=5, 302844 params): training init 0.

Finished since 23:23: **A_arch_scan** (23:25) — analysis only; runner finalized+pushed stub (`47daa390`). This check-in adds the probe table + pushes `analyze_arch_summary.json` / per-arch analyze_*.json.

Queued next (unchanged): **S_g8_attn_A_f** (after a slot frees; dep A_g8 ready), then **R_attn_val** (after s1/s2/r5). Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-18 23:54 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap). Re-pinned all threads of the 3 live jobs to `taskset -c 0-5`. MemAvailable ~5.9 GiB. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2, still training):**
- **N_g4_attn_s1** (pid 3843966, ~72 min): init 0 → 1.286, init 1 → 1.288; on init 2.
- **N_g4_attn_s2** (pid 3872605, ~31 min): init 0 → **1.280** (best train so far among attn seeds; original N_g4_attn#1 was 1.308); on init 1+.
- **N_g4_attn_r5** (pid 3874429, ~28 min, rounds=5, 302844 params): still on init 0 (no JSON yet).

No finishes since 23:38. Next free slot → **S_g8_attn_A_f** (A_g8 ready); **R_attn_val** waits on s1/s2/r5. Behind that: B_g4_iii_A_h128_f.

Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

### N_g4_attn_s1 finished 2026-10-18 23:58 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=attn clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula seed=1 chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_attn_s1 winner 1 (train 1.288, val 1.888) | 3.1/7.5 | 5.6/15.8 | 7.6/30.5 | 12.9/48.7 |
| N_g4_attn_s1 trainbest 0 (train 1.286, val 1.893) | 3.2/7.4 | 5.4/17.3 | 7.7/35.1 | 13.1/50.0 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

## Check-in (2026-10-19 00:15 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap). Re-pinned all threads of the 3 live jobs to `taskset -c 0-5`. MemAvailable ~5.8 GiB. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2):**
- **N_g4_attn_s2** (pid 3872605, ~43 min): init 0 → **1.280**; on init 1+.
- **N_g4_attn_r5** (pid 3874429, ~40 min, rounds=5, 302844 params): init 0 → **1.276**; on init 1+.
- **S_g8_attn_A_f** (pid 3895655, ~12 min, L8 free-SNAP attn from A_g8): training init 0 (launched 23:58 after s1 freed a slot).

Finished since 23:54: **N_g4_attn_s1** (23:58) — train n=8 only, test through n=14 (no TEB n=20 by design). Winner init1 train 1.288 / val 1.888; n=14 **12.9/48.7** vs QAOA L4 20.8/59.6. Slightly better n=8 train than Phase-1 N_g4_attn#1 (1.308). Runner finalized+pushed (`451fc993`). Official n=20 for Phase-2 L4 still waits on **R_attn_val** (needs s1/s2/r5).

Queued next: **R_attn_val** after s2+r5 finish; then B_g4_iii_A_h128_f. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 00:30 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap); all 3 jobs' affinity confirmed `0-5`. MemAvailable ~5.4 GiB. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2, still training):**
- **N_g4_attn_s2** (pid 3872605, ~58 min): init 0 → 1.280, init 1 → 1.299; on init 2.
- **N_g4_attn_r5** (pid 3874429, ~55 min, rounds=5): init 0 → **1.276** (best n=8 train among attn variants so far); on init 1.
- **S_g8_attn_A_f** (pid 3895655, ~27 min, L8): training init 0 (no JSON yet).

No finishes since 00:15. Queued next: **R_attn_val** after s2+r5; then B_g4_iii_A_h128_f. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 00:54 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap); all 3 jobs' affinity confirmed `0-5`. MemAvailable ~4.3 GiB. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2, still training):**
- **N_g4_attn_s2** (pid 3872605, ~64 min): init 0 → 1.280, init 1 → 1.299; on init 2.
- **N_g4_attn_r5** (pid 3874429, ~61 min, rounds=5): init 0 → **1.276**; on init 1 (no further JSON yet).
- **S_g8_attn_A_f** (pid 3895655, ~34 min, L8): init 0 → **0.714** (1000 it, ~55 min); on init 1+.

No finishes since 00:15 (s1). Queued next: **R_attn_val** after s2+r5; then B_g4_iii_A_h128_f. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 01:17 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5` (PSRs in 0–5 only). MemAvailable ~2.3 GiB (below MINMEM=3500 → no 4th launch). Synced deleted log inode for N_g4_attn_s2. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2, still training):**
- **N_g4_attn_s2** (pid 3872605, ~73 min): init 0 → **1.280**, init 1 → 1.299; on init 2.
- **N_g4_attn_r5** (pid 3874429, ~70 min, rounds=5): init 0 → 1.276, init 1 → **1.262** (best n=8 train among attn variants so far); on init 2.
- **S_g8_attn_A_f** (pid 3895655, ~42 min, L8): init 0 → **0.714**; on init 1+.

No finishes since 00:15 (s1). Queued next: **R_attn_val** after s2+r5; then B_g4_iii_A_h128_f. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

### N_g4_attn_s2 finished 2026-10-19 01:19 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=attn clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula seed=2 chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_attn_s2 winner 2 (train 1.288, val 1.880) | 3.0/7.6 | 5.3/14.4 | 7.4/29.7 | 12.5/51.1 |
| N_g4_attn_s2 trainbest 0 (train 1.280, val 1.894) | 3.2/7.7 | 5.4/17.0 | 7.7/29.6 | 12.9/51.6 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

## Check-in (2026-10-19 01:33 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of live jobs re-pinned `taskset -c 0-5`. MemAvailable ~2.5 GiB (below MINMEM=3500 → runner holding **B_g4_iii_A_h128_f**). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**2/3 slots (Phase 2):**
- **N_g4_attn_r5** (pid 3874429, ~76 min, rounds=5): init 0 → 1.276, init 1 → **1.262** (best n=8 train among attn variants); on init 2 (JSON has 2/3 runs; empty log file but CPU ~187% and JSON updates — healthy).
- **S_g8_attn_A_f** (pid 3895655, ~48 min, L8 free-SNAP attn): init 0 → **0.714**; on init 1+ (JSON 1/3 runs).

Finished since 01:17: **N_g4_attn_s2** (01:19) — train n=8 only, test through n=14. Winner init2 train 1.288 / val 1.880; n=14 **12.5/51.1** vs QAOA L4 20.8/59.6. Runner finalized+pushed (`8934dce6`). Official n=20 for Phase-2 L4 still waits on **R_attn_val** (needs r5).

Queued next: **R_attn_val** after r5 finishes (s1+s2 ready); h128 held on mem. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 02:00 CST)
Runner alive (run_v3.sh pid **3872547**), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of live jobs re-pinned `taskset -c 0-5` (0 on cores 6–7). MemAvailable ~5.4Gi (load 11.58, 10.64, 11.25). PID 2888494 and *.STOP untouched. No old-queue restart.

**3/3 slots:**
- **N_g4_attn_r5** (pid 3874429, ~2026-10-19 02:00 CST): 2/3 inits loss=[1.276, 1.262] — best n=8 train among attn variants so far (init1 **1.262**).
- **S_g8_attn_A_f** (pid 3895655): 1/3 inits loss=[0.714] — init0 **0.714**.
- **B_g4_iii_A_h128_f** (pid 3931759, launched 01:58 when mem cleared past MINMEM=3500): no-SNAP H=128 control (queue order behind Phase-2; R_attn_val still blocked on r5).

Finished since 01:33: none new (s2 already logged). Official Phase-2 L4 n=20 still waits on **R_attn_val** (needs r5).

**Queue wave-2 appended** (backup `v3/queue_v3.bak_0159`) after R_attn_val — existing CLI only:
- **N_g4_attn_n2k** (`--ntrain 2000`) more training instances
- **N_g4_attn_s3** (seed 3)
- **N_g4_attn_fm64** (`--free_m 64`) larger free-SNAP head
- **R_attn_val2** (VBASE=39e9) reselect across wave-1+2 after those finish
Tail-weighted loss / ensembles **not queued** (need code changes in `gnn_snap.py`); SNAP lin/pair-with-attn held (prior lin already lost to free on mp).

Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 02:18 CST)
Runner alive (run_v3.sh pid **3872547**, ~1h33m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5` (0 off-cap). MemAvailable ~2.3 GiB (below MINMEM=3500 → no 4th launch). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **N_g4_attn_r5** (pid 3874429, ~90 min, rounds=5): 2/3 inits loss=[1.276, **1.262**] — still best n=8 train among attn variants; on init 2.
- **S_g8_attn_A_f** (pid 3895655, ~63 min, L8 free-SNAP attn): 1/3 inits loss=[**0.714**]; on init 1+.
- **B_g4_iii_A_h128_f** (pid 3931759, ~10 min): no-SNAP H=128 control; started (334364 weights), no JSON runs yet.

Finished since 02:00: none. Official Phase-2 L4 n=20 still waits on **R_attn_val** (needs r5). Wave-2 already queued behind that: N_g4_attn_n2k / s3 / fm64 / R_attn_val2.

Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 02:39 CST)
Runner alive (run_v3.sh pid **3872547**, ~1h39m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5` (PSRs only 0–5). MemAvailable ~2.5 GiB (below MINMEM=3500 → no 4th launch). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **N_g4_attn_r5** (pid 3874429, ~96 min, rounds=5): 2/3 inits loss=[1.276, **1.262**] — still best n=8 train among attn variants; on init 2.
- **S_g8_attn_A_f** (pid 3895655, ~68 min, L8 free-SNAP attn): 2/3 inits loss=[0.714, **0.703**]; on init 2. Train losses still behind prior mp **S_g8_snapF_A_f** (best train 0.651 → n=20 9.1/51.6).
- **B_g4_iii_A_h128_f** (pid 3931759, ~40 min): no-SNAP H=128 control; weights started, no JSON runs yet.

Finished since 02:18: none. Official Phase-2 L4 n=20 still waits on **R_attn_val** (needs r5). Wave-2 already queued behind that: N_g4_attn_n2k / s3 / fm64 / R_attn_val2.

Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

### N_g4_attn_r5 finished 2026-10-19 02:39 CST  (L=4)

`code=gray L=4 hidden=64 rounds=5 arch=attn clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_attn_r5 winner 0 (train 1.276, val 1.907) | 3.2/8.5 | 5.7/15.7 | 7.6/26.7 | 13.5/54.2 |
| N_g4_attn_r5 trainbest 2 (train 1.222, val 1.961) | 3.6/9.8 | 5.9/18.5 | 7.1/35.6 | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |

## Check-in (2026-10-19 02:54 CST)
Runner alive (run_v3.sh pid **3872547**, ~1h48m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5` (0 off-cap). Synced deleted log inode for S_g8_attn_A_f. MemAvailable ~5.0 GiB. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **S_g8_attn_A_f** (pid 3895655, ~77 min, L8 free-SNAP attn): 2/3 inits loss=[0.714, **0.703**]; on init 2. Train still behind prior mp **S_g8_snapF_A_f** (best train 0.651 → n=20 9.1/51.6).
- **B_g4_iii_A_h128_f** (pid 3931759, ~56 min): no-SNAP H=128 control; 1/3 inits loss=[**1.409**]; on init 1+.
- **R_attn_val** (pid 3946728, launched 02:54 when mem cleared past MINMEM after r5 exit): Phase-2 L4 reselect over N_g4_attn / s1 / s2 / r5 on fresh n=10/14/20 val (VBASE=38e9); will TEB-test winner at n=20.

Finished since 02:39: **N_g4_attn_r5** (already finalized+pushed `b287a7ac`) — n=14 winner **13.5/54.2** vs QAOA L4 20.8/59.6; rounds=5 best n=8 train among attn variants (trainbest 1.222) but winner picked by old n=10/12 val. Official Phase-2 L4 n=20 now running via **R_attn_val**.

Queued next (wave-2, after a slot frees): N_g4_attn_n2k / s3 / fm64 / R_attn_val2. Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 03:11 CST)
Runner alive (run_v3.sh pid **3872547**, ~1h56m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5` (0 off-cap). MemAvailable ~2.5 GiB (below MINMEM=3500 → no 4th launch). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **S_g8_attn_A_f** (pid 3895655, ~85 min, L8 free-SNAP attn): 2/3 inits loss=[0.714, **0.703**]; on init 2. Train still behind prior mp **S_g8_snapF_A_f** (best train 0.651 → n=20 9.1/51.6).
- **B_g4_iii_A_h128_f** (pid 3931759, ~33 min): no-SNAP H=128 control; 1/3 inits loss=[**1.409**]; on init 1+.
- **R_attn_val** (pid 3946728, ~9 min): Phase-2 L4 reselect mid-flight — 10/12 cands done. Best val S so far **N_g4_attn_s2#1** S=3.2922 (val n=20 **47.2/161.6**); next are r5#1/#2 then TEB winner. Note: val n=20 is not official TEB.

Finished since 02:54: none. Wave-2 still queued: N_g4_attn_n2k / s3 / fm64 / R_attn_val2.

Best dual-beats unchanged: L4 **S_g4_snapF_s1 37.5/182.7** (median) / attn **42.8/172.7** (B90); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 TEB dual-beat improving on best → no WakeParent.

### R_attn_val finished 2026-10-19 03:14 CST  (L=4)

`L=4 src=['N_g4_attn', 'N_g4_attn_s1', 'N_g4_attn_s2', 'N_g4_attn_r5'] vn=[10, 14, 20] nval=[600, 300, 100] vbase=38000000000 criterion=min mean_n[(log median 1/p + log B90 1/p)/2] over fresh val n=[10, 14, 20] counts=[600, 300, 100] base=38000000000; tie-break mean -log p; TEST only winner on TEB n=20`

| row | n=20 |
|---|---|
| R_attn_val winner 7 (train 1.299, val 3.292) | 42.9/167.2 |
| QAOA_raw L=4 | 84.9/238.1 |
| G_fe L=4 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 42.9 vs 84.9, B90 167.2 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-19 03:35 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h02m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5`. MemAvailable ~2.5 GiB (below MINMEM=3500 → no 4th launch). Synced deleted log inode for S_g8_attn_A_f. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **S_g8_attn_A_f** (pid 3895655, ~92 min, L8 free-SNAP attn): 2/3 inits loss=[0.714, **0.703**]; on init 2 (healthy CPU; train still behind prior mp **S_g8_snapF_A_f** best train 0.651 → n=20 9.1/51.6).
- **B_g4_iii_A_h128_f** (pid 3931759, ~97 min): no-SNAP H=128 control; 1/3 inits loss=[**1.409**]; on init 1+.
- **N_g4_attn_n2k** (pid 3954426, ~20 min, `--ntrain 2000`): wave-2; started, no JSON runs yet.

Finished since 03:11: **R_attn_val** (03:14, already finalized+pushed `b1ac8bc0`) — Phase-2 L4 reselect winner **N_g4_attn_s2#1** (train 1.299, val S=3.292). Official TEB n=20 **42.9/167.2** vs QAOA L4 84.9/238.1 → **dual-beats QAOA**. Improves L4 B90 best (was Phase-1 attn 42.8/172.7); median still behind **S_g4_snapF_s1 37.5/182.7**. → **WakeParent**.

Queued next (wave-2): N_g4_attn_s3 / fm64 / R_attn_val2 (after n2k + deps). Best dual-beats now: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **R_attn_val (attn_s2#1) 42.9/167.2**; L8 **S_g8_snapF_A_f 9.1/51.6**.

### S_g8_attn_A_f finished 2026-10-19 03:52 CST  (L=8)

`code=gray L=8 hidden=64 rounds=3 arch=attn clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g8.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| S_g8_attn_A_f winner 1 (train 0.703, val 1.127) | 1.8/4.2 | 2.4/6.7 | 3.0/14.3 | 4.9/21.3 | 9.0/56.6 |
| S_g8_attn_A_f trainbest 2 (train 0.695, val 1.142) | 1.7/4.5 | 2.5/7.5 | 3.1/9.6 | - | - |
| QAOA_raw L=8 | 2.5/4.5 | 3.5/7.3 | 4.6/15.5 | 6.3/18.3 | 16.4/54.2 |
| G_fe L=8 | 2.5/5.9 | 3.5/10.7 | 5.3/19.2 | 7.4/29.7 | 20.3/118.8 |

n=20 winner vs QAOA L=8: median 9.0 vs 16.4, B90 56.6 vs 54.2 -> no beat

## Check-in (2026-10-19 04:00 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h17m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5`. Synced deleted log inode for S_g8_attn_A_f before exit. MemAvailable ~2.5 GiB (below MINMEM=3500 → no 4th launch). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **B_g4_iii_A_h128_f** (pid 3931759, ~109 min): no-SNAP H=128 control; 2/3 inits loss=[1.409, **1.406**]; on init 2.
- **N_g4_attn_n2k** (pid 3954426, ~45 min, `--ntrain 2000`): wave-2; still on init 0 (no JSON yet; 2× train set → longer).
- **N_g4_attn_s3** (pid 3961946, launched 03:54 after S_g8 freed a slot, seed 3): started, params 187900.

Finished since 03:35: **S_g8_attn_A_f** (03:52, finalized+pushed `934976db`) — L8 free-SNAP attn. Winner init1 train 0.703 / val 1.127; trainbest init2 0.695. Official TEB n=20 **9.0/56.6** vs QAOA L8 16.4/54.2 → **median beats QAOA, B90 loses** (56.6 > 54.2). Median edges prior mp **S_g8_snapF_A_f 9.1/51.6** but B90 worse → L8 dual-beat best **unchanged** at 9.1/51.6. No WakeParent (not dual-beat QAOA).

Queued next (wave-2): N_g4_attn_fm64 / R_attn_val2 (after n2k+s3+fm64). Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **R_attn_val (attn_s2#1) 42.9/167.2**; L8 **S_g8_snapF_A_f 9.1/51.6**.

## Check-in (2026-10-19 04:13 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h21m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5` (PSRs only 0–5). MemAvailable ~2.6 GiB (below MINMEM=3500 → no 4th launch). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **B_g4_iii_A_h128_f** (pid 3931759, ~115 min): no-SNAP H=128 control; 2/3 inits loss=[1.409, **1.406**]; on init 2.
- **N_g4_attn_n2k** (pid 3954426, ~58 min, `--ntrain 2000`): wave-2; still on init 0 (2× train set → longer; healthy CPU ~172%).
- **N_g4_attn_s3** (pid 3961946, ~19 min, seed 3): still on init 0; params 187900.

Finished since 04:00: none new. S_g8_attn_A_f JSON gained trainbest TEB n=14/20 after prior push (trainbest n=20 **9.3/66.0**, worse than winner **9.0/56.6**). Queued next (wave-2): N_g4_attn_fm64 / R_attn_val2 (after n2k+s3+fm64).

Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **R_attn_val (attn_s2#1) 42.9/167.2**; L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 04:41 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h31m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5` (0 off-cap). MemAvailable ~2.6 GiB (below MINMEM=3500 → no 4th launch). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots:**
- **B_g4_iii_A_h128_f** (pid 3931759, ~163 min): no-SNAP H=128 control; 2/3 inits loss=[1.409, **1.406**]; on init 2 (~48 min into this init; prior inits ~54–60 min).
- **N_g4_attn_n2k** (pid 3954426, ~86 min, `--ntrain 2000`): wave-2; still on init 0 (no JSON yet; 2× train set → longer; healthy CPU ~171%).
- **N_g4_attn_s3** (pid 3961946, ~47 min, seed 3): still on init 0; params 187900; healthy CPU ~171%.

Finished since 04:13: none. Queued next (wave-2): **N_g4_attn_fm64** when a slot frees (mem gate); then **R_attn_val2** after n2k+s3+fm64.

Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **R_attn_val (attn_s2#1) 42.9/167.2**; L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

### B_g4_iii_A_h128_f finished 2026-10-19 04:44 CST  (L=4)

`code=gray L=4 hidden=128 rounds=3 clause_w=True var_mix=True ninit=3 ntrain=1000 maxiter=1000 init_from=results/v3/A_g4.json val=[10, 12] val_count=60 test=[8, 10, 12, 14, 20] feat=formula chunk=125`

| row | n=8 | n=10 | n=12 | n=14 | n=20 |
|---|---|---|---|---|---|
| B_g4_iii_A_h128_f winner 0 (train 1.409, val 2.030) | 3.5/7.2 | 6.1/14.4 | 8.3/31.9 | 13.7/40.8 | 53.1/166.3 |
| B_g4_iii_A_h128_f trainbest 1 (train 1.406, val 2.040) | 3.6/7.1 | 6.4/14.6 | 8.9/32.2 | 13.7/42.7 | - |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 | 84.9/238.1 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 | 116.1/387.7 |

n=20 winner vs QAOA L=4: median 53.1 vs 84.9, B90 166.3 vs 238.1 -> **BEATS QAOA (both)**

## Check-in (2026-10-19 04:54 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h40m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5`. MemAvailable ~5.7 GiB after h128 exit + fm64 launch. PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2 wave-2):**
- **N_g4_attn_n2k** (pid 3954426, ~100 min, `--ntrain 2000`): still on init 0 (no JSON yet; 2× train set → longer; healthy CPU ~171%).
- **N_g4_attn_s3** (pid 3961946, ~61 min, seed 3): init 0 → **1.310** (1000 it, ~50 min); on init 1+.
- **N_g4_attn_fm64** (pid 3977567, launched 04:52 after h128 freed a slot, `--free_m 64`): started, params 192220; no JSON yet.

Finished since 04:41: **B_g4_iii_A_h128_f** (04:44, finalized+pushed) — no-SNAP H=128 control. Winner init0 train 1.409 / val 2.030; official TEB n=20 **53.1/166.3** vs QAOA L4 84.9/238.1 → **dual-beats QAOA**. Median behind **S_g4_snapF_s1 37.5** and attn **42.9**; B90 **166.3** edges prior L4 B90 best **R_attn_val 167.2**. → **WakeParent**.

Queued next: **R_attn_val2** after n2k+s3+fm64. Best dual-beats now: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **B_g4_iii_A_h128_f 53.1/166.3** (attn still best dual-beat package 42.9/167.2); L8 **S_g8_snapF_A_f 9.1/51.6**.

## Check-in (2026-10-19 05:20 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h50m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5` (PSRs only 0–5). MemAvailable ~2.7 GiB (below MINMEM=3500 → no 4th launch). Runner stdout fd still on deleted `runner.log` inode (say→deleted; disk log last updated by check-ins); loop healthy (`sleep 60` child). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2 wave-2):**
- **N_g4_attn_n2k** (pid 3954426, ~126 min wall since LAUNCH 03:14, `--ntrain 2000`): init 0 → **1.3085** (1000 it, 7412s ≈2h4m; mid-check JSON at 05:18); on init 1+. Train loss ≈ baseline attn (1.30), behind s1/s2/r5 bests.
- **N_g4_attn_s3** (pid 3961946, ~86 min, seed 3): init 0 → **1.310** (1000 it, ~50 min); on init 1+.
- **N_g4_attn_fm64** (pid 3977567, ~28 min, `--free_m 64`): still on init 0; params 192220; healthy CPU ~178%.

Finished since 04:54: none (jobs still mid-train). Mid-check: **N_g4_attn_n2k** init0 landed (1.3085). h128 JSON/log gained trainbest TEB n=20 after prior push (trainbest n=20 **53.1/229.9**, worse B90 than winner **53.1/166.3** — no change to bests). Queued next: **R_attn_val2** after n2k+s3+fm64.

Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **B_g4_iii_A_h128_f 53.1/166.3** (attn dual-beat package still **R_attn_val 42.9/167.2**); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 05:32 CST)
Runner alive (run_v3.sh pid **3872547**, ~2h58m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5` (0 off-cap). MemAvailable ~2.4 GiB (below MINMEM=3500 → no 4th launch). Runner stdout fd still on deleted `runner.log` inode (say→deleted; disk log updated by check-ins); loop healthy (`sleep 60` child). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart.

**3/3 slots (Phase 2 wave-2):**
- **N_g4_attn_n2k** (pid 3954426, ~59 min wall this check / ~2h18m since LAUNCH 03:14, `--ntrain 2000`): init 0 → **1.3085** (1000 it, 7412s); on init 1+ (JSON 1/3 runs). Train loss ≈ baseline attn.
- **N_g4_attn_s3** (pid 3961946, ~48 min, seed 3): init 0 → 1.310; init 1 → **1.293** (best so far this job; JSON 2/3 runs); on init 2+.
- **N_g4_attn_fm64** (pid 3977567, ~22 min, `--free_m 64`): still on init 0; params 192220; no JSON yet; healthy CPU ~178%.

Finished since 05:20: none (jobs still mid-train). Progress: **s3** landed init1 **1.293**. Queued next: **R_attn_val2** after n2k+s3+fm64 (only pending undones besides the 3 running).

Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **B_g4_iii_A_h128_f 53.1/166.3** (attn dual-beat package still **R_attn_val 42.9/167.2**); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 05:55 CST)
Runner alive (run_v3.sh pid **3872547**, ~3h10m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs re-pinned `taskset -c 0-5` (0 off-cap). MemAvailable ~2.7 GiB (below MINMEM=3500 → no 4th launch). Runner loop healthy (`sleep 60` child). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart. Campaign continues (deadline Mon Oct 19 09:00 CST; ~3h5m left).

**3/3 slots (Phase 2 wave-2):**
- **N_g4_attn_n2k** (pid 3954426, ~2h41m since LAUNCH 03:14, `--ntrain 2000`): init 0 → **1.3085** (1000 it, 7412s); on init 1+ (JSON 1/3 runs). Train loss ≈ baseline attn.
- **N_g4_attn_s3** (pid 3961946, ~2h01m, seed 3): init 0 → 1.310; init 1 → **1.293**; on init 2+ (JSON 2/3 runs).
- **N_g4_attn_fm64** (pid 3977567, ~63 min, `--free_m 64`): init 0 → **1.2915** (1000 it, 2565s; landed 05:34); on init 1+. Best n=8 train among wave-2 so far (edges s3 1.293; still behind r5 trainbest 1.222 / Phase-2 val winners).

Finished since 05:32: none (jobs still mid-train). Progress: **fm64** landed init0 **1.2915**. Queued next: **R_attn_val2** after n2k+s3+fm64 (only pending undones besides the 3 running).

Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **B_g4_iii_A_h128_f 53.1/166.3** (attn dual-beat package still **R_attn_val 42.9/167.2**); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

## Check-in (2026-10-19 06:15 CST)
Runner alive (run_v3.sh pid **3872547**, ~3h20m), MAXJ=3 / cores 0-5 (Zach 6-core cap); all threads of the 3 live jobs confirmed affinity `0-5` (PSRs only 0–5). MemAvailable ~2.6 GiB (below MINMEM=3500 → no 4th launch). Runner stdout fd still on deleted `runner.log` inode (disk log last updated by check-ins); loop healthy (`sleep 60` child). PID 2888494 and *.STOP untouched. No queue rewrite; no old-queue restart. Campaign continues (deadline Mon Oct 19 09:00 CST; ~2h45m left).

**3/3 slots (Phase 2 wave-2):**
- **N_g4_attn_n2k** (pid 3954426, ~3h00m since LAUNCH 03:14, `--ntrain 2000`): init 0 → **1.3085** (1000 it, 7412s); on init 1+ (JSON 1/3 runs). Train loss ≈ baseline attn.
- **N_g4_attn_s3** (pid 3961946, ~2h20m, seed 3): init 0 → 1.310; init 1 → **1.293**; on init 2+ (JSON 2/3 runs).
- **N_g4_attn_fm64** (pid 3977567, ~1h23m, `--free_m 64`): init 0 → **1.2915** (1000 it, 2565s); on init 1+ (JSON 1/3 runs). Still best n=8 train among wave-2 so far.

Finished since 05:55: none (jobs still mid-train). No new JSON losses since fm64 init0 / s3 init1. Queued next: **R_attn_val2** after n2k+s3+fm64 (only pending undones besides the 3 running).

Best dual-beats unchanged: L4 median **S_g4_snapF_s1 37.5/182.7** / B90 **B_g4_iii_A_h128_f 53.1/166.3** (attn dual-beat package still **R_attn_val 42.9/167.2**); L8 **S_g8_snapF_A_f 9.1/51.6**. No new n=20 dual-beat improving on best → no WakeParent.

### N_g4_attn_s3 finished 2026-10-19 06:18 CST  (L=4)

`code=gray L=4 hidden=64 rounds=3 arch=attn clause_w=True var_mix=True snap=std snap_form=free free_m=32 init_from=results/v3/A_g4.json ninit=3 ntrain=1000 maxiter=1000 val=[10, 12] val_count=60 test=[8, 10, 12, 14] feat=formula seed=3 chunk=125`

| row | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| N_g4_attn_s3 winner 0 (train 1.310, val 1.871) | 3.1/7.5 | 5.6/16.4 | 7.3/30.4 | 13.3/51.6 |
| N_g4_attn_s3 trainbest 2 (train 1.287, val 1.905) | 3.1/7.3 | 5.4/16.9 | 7.5/30.7 | 13.4/56.5 |
| QAOA_raw L=4 | 5.3/9.9 | 8.4/19.7 | 12.8/32.8 | 20.8/59.6 |
| G_fe L=4 | 5.4/9.8 | 9.3/20.4 | 14.9/39.1 | 22.9/57.3 |
