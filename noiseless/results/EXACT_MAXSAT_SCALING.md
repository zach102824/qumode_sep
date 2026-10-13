# Exact MaxSAT solver cost scaling (PySAT RC2 / FM)

Run 2026-10-13 ~17:14 CST, single core, `noiseless/exact_maxsat_scaling.py` (raw: `logs/3cav/exact_maxsat/results.jsonl`), analysis `noiseless/analyze_exact_maxsat.py`. PySAT 1.9.dev15, SAT oracle Glucose 4 (`g4`).

**Instances.** n=8..20: exactly the strict-sweep set `Hamiltonians/wmaxsat/default_nNN_{0..9}.json` (as loaded by `run_relayout_kbit.py --ham-set wmaxsat`). Weighted MAX-4-SAT, m = round(9.93 n) clauses, integer weights 1..5. The cost is literally a sum of weights of violated clauses, so each clause maps directly to a WCNF soft clause (literal +v if sign>0 else −v, weight w); no hard clauses, no weight scaling. **Verification:** for all 70 instances both RC2 and FM returned the exhaustive-enumeration ground string `gs` and cost `E_gs` (70/70 each).
n=24..64 (RC2; FM only to 32): **NEW instances**, 10 per n, same clause generator as `Hamiltonians/wmaxsat_gen.py` (seed 100000n+50000+k) but WITHOUT the unique-ground-state filter (no enumeration), so no ground-truth check; optimum is the solver's proven optimum.

Metrics per instance: wall time = median of 5 repeats (formula build excluded, solver build included); SAT calls = calls to oracle solve/solve_limited; cores = non-empty unsat cores (RC2); conflicts/decisions/propagations = Glucose accum_stats summed over oracles. LSU was tried but hung on these (no-hard-clause) formulas and was dropped.

### RC2 (Glucose4 oracle)

| n | instances | new? | GS match | median t (ms) | mean t (ms) | median SAT calls | mean SAT calls | median cores | median conflicts | mean conflicts | median decisions | median propagations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 10 | no | 10/10 | 0.198 | 0.224 | 1 | 1.2 | 0.0 | 4.5 | 6.9 | 9.5 | 100.5 |
| 10 | 10 | no | 10/10 | 0.224 | 0.276 | 1 | 1.3 | 0.0 | 7 | 11.4 | 13.5 | 142.5 |
| 12 | 10 | no | 10/10 | 0.479 | 0.464 | 1.5 | 1.5 | 0.5 | 27.5 | 26.2 | 33.5 | 400 |
| 14 | 10 | no | 10/10 | 0.282 | 0.331 | 1 | 1.1 | 0.0 | 22.5 | 23.1 | 27 | 259 |
| 16 | 10 | no | 10/10 | 0.298 | 0.419 | 1 | 1.3 | 0.0 | 22 | 32.1 | 29.5 | 291 |
| 18 | 10 | no | 10/10 | 0.869 | 0.888 | 2 | 1.8 | 1.0 | 77 | 78.9 | 89.5 | 1204 |
| 20 | 10 | no | 10/10 | 0.453 | 0.526 | 1 | 1.1 | 0.0 | 42.5 | 52.0 | 52 | 471.5 |
| 24 | 10 | yes | n/a (new) | 0.587 | 1.033 | 1 | 1.4 | 0.0 | 89 | 118.6 | 98.5 | 837 |
| 28 | 10 | yes | n/a (new) | 1.277 | 1.714 | 1.5 | 1.6 | 0.5 | 197 | 237.0 | 220 | 2218.5 |
| 32 | 10 | yes | n/a (new) | 1.846 | 3.375 | 1.5 | 1.7 | 0.5 | 304.5 | 407.2 | 339 | 4060 |
| 40 | 10 | yes | n/a (new) | 2.560 | 6.612 | 1 | 1.5 | 0.0 | 558.5 | 920.5 | 638.5 | 6068.5 |
| 48 | 10 | yes | n/a (new) | 30.082 | 57.960 | 1.5 | 1.6 | 0.5 | 2986.5 | 4299.2 | 3586.5 | 54229 |
| 56 | 10 | yes | n/a (new) | 53.696 | 185.909 | 1.5 | 1.6 | 0.5 | 5326.5 | 8947.5 | 6043.5 | 145950 |
| 64 | 10 | yes | n/a (new) | 451.427 | 741.357 | 2 | 1.8 | 1.0 | 25129 | 33395.2 | 28713.5 | 700732 |

- fit n=8-20 (original), median t_med: 0.0001091*1.0878^n (R2 0.500); 1.935e-05*n^1.12 (R2 0.511)

- fit n=8-20 (original), median calls: 0.9155*1.0177^n (R2 0.073); 0.6141*n^0.25 (R2 0.084)

- fit n=8-20 (original), median conflicts: 1.199*1.2238^n (R2 0.783); 0.01677*n^2.73 (R2 0.826)

- fit all n, median t_med: 4.907e-05*1.1359^n (R2 0.941); 7.435e-08*n^3.20 (R2 0.793)

- fit all n, median calls: 1.021*1.0080^n (R2 0.279); 0.6642*n^0.21 (R2 0.252)

- fit all n, median conflicts: 2.841*1.1514^n (R2 0.975); 0.0009963*n^3.79 (R2 0.941)

### FM (Glucose4 oracle)

| n | instances | new? | GS match | median t (ms) | mean t (ms) | median SAT calls | mean SAT calls | median cores | median conflicts | mean conflicts | median decisions | median propagations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 10 | no | 10/10 | 0.236 | 0.316 | 2 | 2.2 | - | 4.5 | 9.9 | 10.5 | 100.5 |
| 10 | 10 | no | 10/10 | 0.312 | 0.528 | 2 | 2.3 | - | 7 | 27.6 | 14.5 | 142.5 |
| 12 | 10 | no | 10/10 | 0.928 | 1.066 | 2.5 | 2.5 | - | 30.5 | 44.2 | 63 | 330 |
| 14 | 10 | no | 10/10 | 0.455 | 0.660 | 2 | 2.1 | - | 22.5 | 25.0 | 28 | 259 |
| 16 | 10 | no | 10/10 | 0.528 | 1.163 | 2 | 2.3 | - | 22 | 39.2 | 30.5 | 291 |
| 18 | 10 | no | 10/10 | 3.414 | 3.824 | 3 | 2.8 | - | 94.5 | 79.2 | 246 | 1497.5 |
| 20 | 10 | no | 10/10 | 0.663 | 1.244 | 2 | 2.1 | - | 42.5 | 51.4 | 53 | 471.5 |
| 24 | 10 | yes | n/a (new) | 0.898 | 4.778 | 2 | 2.4 | - | 89 | 110.4 | 99.5 | 837 |
| 28 | 10 | yes | n/a (new) | 6.645 | 10.762 | 2.5 | 2.6 | - | 205.5 | 199.1 | 358.5 | 2980.5 |
| 32 | 10 | yes | n/a (new) | 9.166 | 19.102 | 2.5 | 2.7 | - | 329 | 307.9 | 525.5 | 5783 |

- fit n=8-20 (original), median t_med: 0.0001002*1.1396^n (R2 0.416); 6.277e-06*n^1.77 (R2 0.440)

- fit n=8-20 (original), median calls: 1.889*1.0106^n (R2 0.078); 1.491*n^0.15 (R2 0.089)

- fit n=8-20 (original), median conflicts: 1.161*1.2305^n (R2 0.743); 0.01438*n^2.81 (R2 0.786)

- fit all n, median t_med: 8.621e-05*1.1486^n (R2 0.730); 1.429e-06*n^2.35 (R2 0.685)

- fit all n, median calls: 1.966*1.0069^n (R2 0.131); 1.584*n^0.12 (R2 0.133)

- fit all n, median conflicts: 2.003*1.1794^n (R2 0.886); 0.01048*n^2.93 (R2 0.911)

## Notes / caveats

- **These instances are near-satisfiable.** Ratio 9.93 is ~the random 4-SAT threshold; optimum costs are 0–2 at every n (n=8–20 originals: 0 for 49/70, 1 for 19, 2 for 2). An exact solver therefore usually proves optimality in 1–2 SAT calls with 0–1 cores; RC2 at n=20 takes ~0.45 ms median. Over n=8–20 the time is flat/noisy (fits R² ≈ 0.5, meaningless); the real growth signal is in SAT-solver conflicts (~1.22^n on n=8–20, ~1.15^n over n=8–64) and in time at n≥40 (0.0026 s at n=40 → 0.45 s at n=64 median).
- **No "lookups".** Exact solvers never enumerate candidate strings; they reason over clauses (unit propagation, conflict learning, cores). The closest analogue to our lookups is the number of SAT-solver *decisions* (each a partial-assignment branch, ~candidate visited) or conflicts; median decisions at n=20 = 52 (RC2) vs our hybrid's 740 lookups — but a decision is a partial assignment plus propagation, not one full-string cost evaluation, so this is not a like-for-like count. Propagations (471 median at n=20) are the closest to "elementary operations".
- **Implication for the pitch:** on this instance family a classical exact solver is cheap and fast at n≤64. Any quantum-advantage argument must be against heuristic lookups (SA), or use harder instance families (larger optimum cost / further above threshold, or structured), not against exact solvers here.
- Fits are over medians of 10 instances; exponential vs power law cannot be strongly distinguished at this range.
