# SA-alone scaling vs hybrid lookups (wmaxsat, Gray-code sweep instances)

Run 2026-10-13, 03:03-03:08 CST (box 11:03-11:08 CEST), 8 cores, ~5 min. Script `noiseless/sa_alone_scaling.py`,
analysis `noiseless/analyze_sa_alone_scaling.py`, raw data `logs/3cav/saalone/hits.jsonl` (n, inst, seed, first-hit lookup).

Method
- Instances: `ham_paths(n, "wmaxsat")` + `energies_for(..., "wmaxsat")` from run_relayout_kbit.py, inst 0-9, n=8..20 (same as strict sweep).
- SA: exact re-implementation of `classical_layout.sa_sample` (QM_SA_COUNT=raw: every evaluation counts, repeats count;
  64 counted probes once per run; restart chains of 8n steps from a fresh random start, T 2.0 -> 0.05 x probe std).
  sa_sample's loop is already an unbounded restart loop, and the budget only truncates the RNG stream, so the lookup index of the
  first GS evaluation is budget-independent; one long run per (inst, seed) gives success at every budget. Verified against
  sa_sample on n=8-12, 150 runs, at budgets h-1 and h: 0 mismatches (except GS hit inside the 64 probes, which sa_sample
  always performs; all budgets here are >=152, so it does not matter).
- 20 seeds per instance (seed 20261013+1000n+100inst+s) -> 200 runs per n. Cap 40*2^n lookups; 0 runs hit the cap.
- B90: smallest budget with success >= 0.9 over 200 runs, found by doubling from 64 then integer bisection on the empirical curve.

## Results
| n | hybrid lookups | SA-alone success @ hybrid budget (200 runs) | per-instance range | @2x | @4x | B90 (SA-alone) | B90/hybrid | median lookups to GS | mean | random 2^n ln10 | B90/2^n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 152 | 0.535 | 0.30-0.70 | 0.765 | 0.960 | 462 | 3.04 | 144 | 219 | 589 | 1.8047 |
| 10 | 190 | 0.385 | 0.15-0.60 | 0.640 | 0.850 | 1005 | 5.29 | 273 | 422 | 2358 | 0.9814 |
| 12 | 228 | 0.245 | 0.10-0.35 | 0.450 | 0.630 | 2127 | 9.33 | 606 | 981 | 9431 | 0.5193 |
| 14 | 266 | 0.140 | 0.05-0.20 | 0.300 | 0.485 | 3735 | 14.04 | 1120 | 1556 | 37726 | 0.2280 |
| 16 | 400 | 0.080 | 0.00-0.20 | 0.230 | 0.475 | 5234 | 13.09 | 1700 | 2598 | 150902 | 0.0799 |
| 18 | 558 | 0.125 | 0.00-0.25 | 0.255 | 0.430 | 9909 | 17.76 | 3084 | 4164 | 603609 | 0.0378 |
| 20 | 740 | 0.060 | 0.00-0.15 | 0.125 | 0.250 | 18236 | 24.64 | 6712 | 9415 | 2414435 | 0.0174 |

Exponential fit B90 = a*b^n: a = 51.782, b = 1.3427 (= 2^0.425n), R^2(log) = 0.9901, R^2(lin) = 0.9947
Power-law fit B90 = a*n^k: a = 0.1331, k = 3.888, R^2(log) = 0.9920, R^2(lin) = 0.9568
Hybrid lookups for comparison: power-law k = 1.714 (R^2 log 0.9321); exp b = 1.1426 (R^2 log 0.9792)

Hybrid success at the same lookup budgets: 1.00 at n=8..20 (strict sweep; n=20 = noise 0.3/r150 screen), plus quantum evals.

## Caveats
- SA is the project's SA, not tuned (schedule, 8n chain length, restarts, probes untouched). A tuned SA, tabu search, WalkSAT-style
  local search or an exact MaxSAT solver would do better; this is a fair "same classical component, no quantum" ablation, not a
  best-classical bound.
- n range 8-20 is short; exponential and power-law fit about equally well (R^2 ~0.99 in log space), so the growth class is not
  established. Exponential b=1.34 is well below random search (b=2).
- Hybrid lookups exclude its quantum evals (505-2005 circuit evals, ~1000 shots each); there is no exact exchange rate.
- 10 instances per n; per-instance success varies strongly (see range column).
