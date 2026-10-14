# Fresh instances: no-classical-search quantum recipe vs SATLike

Updated 2026-10-14 23:11 CST. Instances: Hamiltonians/wmaxsat_fresh (first 100 unique-GS of satlike_manyinst block 0, seed 900000000+1000000n+k; verified: SATLike hits on these 100 reproduce a manyinst block exactly for every n). Quantum: QM_NOCLS=1, 3n final settings (E=2,2,2,2,4,6,8; 2cav n<=12, 3cav --ecd-fast n>=14; r150, sanoise 0.3, exact cost), 1 trial per instance (logs/fresh/run.sh). SATLike: one run per instance, tuned cfg; all = 2000 instances per n (n=20: 6000). Matched = smallest SATLike lookup count solving the same fraction of the same 100 instances as the quantum run. Fits a*b^n, R2 in log space, n=8..18 (n=20 running, ETA 00:55 CST 2026-10-15).

| n | quantum success | mean p(GS) | lookups | evals | total checks | SATLike, same 100, at quantum's rate | SATLike 99%, same 100 | SATLike 99%, all | SATLike median, all | instances (all) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 100/100 | 0.986 | 73 | 505 | 578 | 388 | 200 | 246 | 31 | 2000 |
| 10 | 99/100 | 0.977 | 91 | 585 | 676 | 395 | 395 | 424 | 50 | 2000 |
| 12 | 99/100 | 0.976 | 109 | 705 | 814 | 701 | 701 | 594 | 89 | 2000 |
| 14 | 100/100 | 0.965 | 127 | 727 | 854 | 844 | 727 | 1083 | 113 | 2000 |
| 16 | 100/100 | 0.967 | 241 | 1153 | 1394 | 2822 | 1136 | 1694 | 142 | 2000 |
| 18 | 100/100 | 0.968 | 379 | 1579 | 1958 | 1740 | 836 | 1694 | 178 | 2000 |

- quantum total checks: 202.9 * 1.126^n, R2 0.926
- SATLike at quantum rate, same 100: 70.1 * 1.214^n, R2 0.827
- SATLike 99%, all: 52.4 * 1.228^n, R2 0.961
- SATLike median, all: 9.1 * 1.188^n, R2 0.955
