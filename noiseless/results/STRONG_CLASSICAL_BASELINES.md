# Strong classical baselines vs hybrid lookups (wmaxsat, 70 strict-sweep instances)

Run 2026-10-13, 03:46-03:53 CST (box 11:46-11:53 CEST), 8 cores, detached setsid nohup queue. Script `noiseless/strong_classical.py`,
analysis `noiseless/analyze_strong_classical.py`, raw data `logs/strong/{tune,eval}.jsonl`, chosen configs `logs/strong/best_cfg.json`.

## Method
- Eval instances: `Hamiltonians/wmaxsat/default_nXX_{0..9}.json`, n=8..20 (same 70 as SA_ALONE_SCALING / EXACT_MAXSAT_SCALING).
  Ground truth by exhaustive 2^n enumeration; asserted unique argmin = file gs, E_gs matches.
- Tuning: separate held-out set, same generator (wmaxsat_gen.py clause rule), seeds 100000n+70000+k, 10 unique-GS instances per n,
  20 runs each (200 runs per config), tuning cap 4x untuned-SA B90. Config chosen per n by min held-out B90 (tie: success at cap, mean).
  Grids: SA T0 in {0.5,1,2,4}, Tend in {0.05,0.2,0.5,1.0} (Tend<T0), chain length L in {1,2,4,8,16,32,64}*n, random restart after each chain
  (91 configs/n); WalkSAT noise p in {0,0.05,0.1,0.2,0.3,0.5}, maxflips in {n,4n,16n,64n,inf} (30/n);
  SATLike weight init/increment unit in {1, original w}, soft weight bound swt in {2,5,20,100,1000}, smoothing prob sp in {0,0.01,0.1}, BMS t in {3,15} (60/n).
- Eval: tuned config, 200 runs per n (20 seeds x 10 instances), cap min(40*2^n, 4e6) lookups; 0 runs unsolved at cap except 1 SATLike run at n=12.
- Algorithms (numba): SA = random-restart Metropolis, geometric T0->Tend over L single-flip proposals (absolute temperatures, integer weights 1-5).
  WalkSAT = weighted SKC: random unsat clause, break (weighted) of its 4 vars, freebie if break 0, else noise p random var / min break.
  SATLike = faithful SATLike-3.0-style dynamic clause weighting re-implementation (the C source was NOT built): score with dynamic weights,
  goodvars = score>0, BMS pick best of t sampled goodvars (age tie-break); if none, weight update (prob sp: smooth satisfied clauses down by one unit,
  else bump unsat clauses by one unit up to swt units) then best var of a random unsat clause. No restarts.

## Lookup convention (state clearly)
- **strict** (as requested): every candidate string whose cost is scored = 1 lookup, repeats count; every fresh random start = 1.
  SA: 1 per proposal. WalkSAT: 4 per step. SATLike: min(t,|goodvars|) per BMS step, 4 per stuck step. Hit = first time the GS string is scored.
- **full-neighbourhood**: 1 + n*(flips+1), i.e. every step scores all n one-flip neighbours. This is the honest count for SATLike, whose goodvar set
  needs the score of every variable each step (maintained incrementally in real SATLike); strict undercounts SATLike. For SA the two are the same column.
- Flips (accepted moves) to hit are reported (B90 flips = 90th-percentile flips).
- Hybrid lookups 152..740 exclude its 505..2005 quantum evals.

## Results

### Tuned SA

| n | tuned config | succ @ hybrid budget (strict) | B90 strict | B90/hybrid | median strict | B90 flips | B90 full-neighbourhood (1+n*(flips+1)) | succ @ hybrid budget (full-nbhd) | B90 full/hybrid | unsolved@cap |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | {"T0": 0.5, "T1": 0.2, "L": 16} | 0.655 | 319 | 2.10 | 98 | 70 | 319 | 0.655 | 2.10 | 0 |
| 10 | {"T0": 1.0, "T1": 0.05, "L": 20} | 0.420 | 726 | 3.82 | 243 | 161 | 726 | 0.420 | 3.82 | 0 |
| 12 | {"T0": 2.0, "T1": 1.0, "L": 96} | 0.260 | 1925 | 8.44 | 420 | 383 | 1925 | 0.260 | 8.44 | 0 |
| 14 | {"T0": 2.0, "T1": 1.0, "L": 112} | 0.205 | 2557 | 9.61 | 774 | 558 | 2557 | 0.205 | 9.61 | 0 |
| 16 | {"T0": 2.0, "T1": 0.05, "L": 64} | 0.180 | 4073 | 10.18 | 1082 | 705 | 4073 | 0.180 | 10.18 | 0 |
| 18 | {"T0": 2.0, "T1": 1.0, "L": 288} | 0.230 | 7190 | 12.89 | 1774 | 1375 | 7190 | 0.230 | 12.89 | 0 |
| 20 | {"T0": 2.0, "T1": 1.0, "L": 320} | 0.175 | 9511 | 12.85 | 2560 | 1810 | 9511 | 0.175 | 12.85 | 0 |

B90 strict: exp a=47.022 b=1.3194 R2(log)=0.9650 R2(lin)=0.9037; power a=0.1532 k=3.702 R2(log)=0.9912 R2(lin)=0.9898

B90 full-nbhd: exp a=47.022 b=1.3194 R2(log)=0.9650 R2(lin)=0.9037; power a=0.1532 k=3.702 R2(log)=0.9912 R2(lin)=0.9898

### Weighted WalkSAT

| n | tuned config | succ @ hybrid budget (strict) | B90 strict | B90/hybrid | median strict | B90 flips | B90 full-neighbourhood (1+n*(flips+1)) | succ @ hybrid budget (full-nbhd) | B90 full/hybrid | unsolved@cap |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | {"p": 0.3, "maxf": 8} | 0.780 | 255 | 1.68 | 62 | 61 | 497 | 0.545 | 3.27 | 0 |
| 10 | {"p": 0.5, "maxf": 10} | 0.655 | 412 | 2.17 | 115 | 100 | 1011 | 0.385 | 5.32 | 0 |
| 12 | {"p": 0.2, "maxf": 12} | 0.510 | 848 | 3.72 | 224 | 207 | 2497 | 0.205 | 10.95 | 0 |
| 14 | {"p": 0.05, "maxf": 14} | 0.500 | 1045 | 3.93 | 262 | 256 | 3599 | 0.195 | 13.53 | 0 |
| 16 | {"p": 0.1, "maxf": 16} | 0.430 | 1577 | 3.94 | 510 | 387 | 6209 | 0.140 | 15.52 | 0 |
| 18 | {"p": 0.0, "maxf": 18} | 0.440 | 2604 | 4.67 | 650 | 641 | 11557 | 0.080 | 20.71 | 0 |
| 20 | {"p": 0.5, "maxf": 320} | 0.370 | 3784 | 5.11 | 1070 | 945 | 18921 | 0.080 | 25.57 | 0 |

B90 strict: exp a=47.492 b=1.2479 R2(log)=0.9880 R2(lin)=0.9923; power a=0.5437 k=2.917 R2(log)=0.9872 R2(lin)=0.9762

B90 full-nbhd: exp a=53.418 b=1.3475 R2(log)=0.9894 R2(lin)=0.9854; power a=0.1269 k=3.938 R2(log)=0.9927 R2(lin)=0.9797

### SATLike-style

| n | tuned config | succ @ hybrid budget (strict) | B90 strict | B90/hybrid | median strict | B90 flips | B90 full-neighbourhood (1+n*(flips+1)) | succ @ hybrid budget (full-nbhd) | B90 full/hybrid | unsolved@cap |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 3} | 0.900 | 151 | 0.99 | 33 | 47 | 385 | 0.670 | 2.53 | 0 |
| 10 | {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 3} | 0.950 | 161 | 0.85 | 37 | 51 | 521 | 0.600 | 2.74 | 0 |
| 12 | {"mode": 1, "swt": 20, "sp": 0.0, "t_bms": 3} | 0.815 | 347 | 1.52 | 92 | 122 | 1477 | 0.280 | 6.48 | 1 |
| 14 | {"mode": 0, "swt": 100, "sp": 0.0, "t_bms": 15} | 0.880 | 297 | 1.12 | 86 | 100 | 1415 | 0.360 | 5.32 | 0 |
| 16 | {"mode": 0, "swt": 100, "sp": 0.0, "t_bms": 15} | 0.855 | 526 | 1.31 | 123 | 178 | 2865 | 0.345 | 7.16 | 0 |
| 18 | {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 15} | 0.800 | 994 | 1.78 | 237 | 349 | 6301 | 0.220 | 11.29 | 0 |
| 20 | {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 3} | 0.900 | 702 | 0.95 | 218 | 249 | 5001 | 0.235 | 6.76 | 0 |

B90 strict: exp a=42.195 b=1.1674 R2(log)=0.8771 R2(lin)=0.7167; power a=1.8613 k=2.038 R2(log)=0.8753 R2(lin)=0.7606

B90 full-nbhd: exp a=59.843 b=1.2690 R2(log)=0.9334 R2(lin)=0.7408; power a=0.4736 k=3.150 R2(log)=0.9395 R2(lin)=0.8275

Hybrid lookups: exp b=1.1426 R2(log)=0.9792; power k=1.714 R2(log)=0.9321

### Head-to-head (B90 strict / full-nbhd vs hybrid lookups; hybrid succ 10/10, mean p(GS) listed)

| n | hybrid lookups | hybrid p(GS) | untuned SA B90 | tuned SA B90 | WalkSAT B90 strict/full | SATLike B90 strict/full |
|---|---|---|---|---|---|---|
| 8 | 152 | 0.987 | 462 | 319 | 255 / 497 | 151 / 385 |
| 10 | 190 | 0.986 | 1005 | 726 | 412 / 1011 | 161 / 521 |
| 12 | 228 | 0.99 | 2127 | 1925 | 848 / 2497 | 347 / 1477 |
| 14 | 266 | 0.961 | 3735 | 2557 | 1045 / 3599 | 297 / 1415 |
| 16 | 400 | 0.972 | 5234 | 4073 | 1577 / 6209 | 526 / 2865 |
| 18 | 558 | 0.963 | 9909 | 7190 | 2604 / 11557 | 994 / 6301 |
| 20 | 740 | 0.957 | 18236 | 9511 | 3784 / 18921 | 702 / 5001 |

## Verdict
- **YES, a strong heuristic matches or beats the hybrid on lookups under the strict convention: SATLike-style B90 = 151 vs 152 (n=8), 161 vs 190 (n=10),
  702 vs 740 (n=20)**, and is within 1.12x-1.78x at n=12-18. Success at the hybrid budget is 0.800-0.950 (hybrid 1.00, mean p(GS) 0.957-0.990).
  Strict SATLike scaling b=1.1674 vs hybrid 1.1426 (SATLike fit poor, R2(log) 0.8771).
- Under the full-neighbourhood count SATLike needs 2.53x-11.29x the hybrid lookups (B90 385..6301; b=1.2690).
- Tuned SA: 2.10x-12.89x hybrid (B90 319..9511), vs untuned 3.04x-24.64x; tuning halves n=20 (18236 -> 9511).
- WalkSAT strict: 1.68x-5.11x hybrid (B90 255..3784, b=1.2479); never matches.
- So the "3x-25x fewer lookups" claim only holds against untuned/tuned SA. Against SATLike the honest statement is: comparable (strict count)
  to 2.5x-11x fewer (full-neighbourhood count), and the hybrid's advantage hinges on the counting convention.

## Caveats
- SATLike is a re-implementation, not the official binary; hard-clause machinery is irrelevant (all soft). Held-out tuning per n on 10 instances;
  several chosen configs sit on grid edges (SA Tend=1.0, WalkSAT p=0.0/0.5), so slightly better tunes may exist.
- 10 eval instances per n, n<=20; fits over 7 points. Noiseless hybrid; its quantum evals are not converted to lookups.
- Times: CST taken as box CEST - 8h, consistent with SA_ALONE_SCALING.md.
