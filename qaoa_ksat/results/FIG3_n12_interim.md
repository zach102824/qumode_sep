# Fig. 3 WalkSATlm, n=12 interim (2026-10-22)

**Main finding: the authors' raw data shows the unit.** The Zenodo data (record 7764484, used by
github.com/sami-b95/qaoa_ksat_paper_data `load_walksatlm_benchmarks`) sets
`walksatlm_runtime = num_trials`. Each record has `p=0.15, w1=6, w2=5, max_flips = n`. So a
"formula evaluation" is **one try of n flips**, not one flip. Recomputing their medians from that data
gives n=12: 12.0, and the n=12..20 fit is **log2 = -0.3085 + 0.3246 n**, which is exactly Table II.

Our rerun: C WalkSATlm with Cai's WalkSATlm2013 pick rule (`pick_bbreak_gmake`, with p/w1/w2
as parameters). Ensemble CNF(12, 8, r=176.54): Poisson m, literals uniform with replacement,
brute-force filtered to satisfiable (2000/3442 kept). One run per instance, median over 2000
instances.

| counter (n=12) | median |
|---|---|
| tries, max_flips = n = 12, fresh random start each try (**paper protocol**) | **11** |
| tries, max_flips = 2n | 7 |
| tries, max_flips = 4n | 4 |
| flips + initial assignments, cutoff n (1 eval per flip and per start) | 141 |
| flips only, cutoff n | 130 |
| flips, no restart (sample still running: 88 instances) | 202 |
| paper Table II fit | 12.1 |
| paper raw data median | 12.0 |

Per clause check is not instrumented in the C code. Our earlier Python "lookups" ran about 8× flips,
i.e. about 1000+ here.

Gap explained: the paper's unit is n-flip tries, so 1 paper unit ≈ n flips. 12.1 × 12 ≈ 145 flips,
which matches our 141. Our old 255 flips at n=12 came from a no-restart Python walk. That is a
counting/protocol difference, not a weaker solver.
