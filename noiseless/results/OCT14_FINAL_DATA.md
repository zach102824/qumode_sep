# Oct 14 final data (2026-10-14)

Weighted MaxSAT (random 4-SAT, clause/variable ratio 9.93, weights 1-5, unique ground state), Gray code, noiseless simulation.
10 instances per n (same 70 instances as all earlier tables), 1 trial each. Ground truth from exhaustive 2^n enumeration.

## Recipe (one rule for every n)
- Classical stage: SATLike-style dynamic local search (after Cai & Lei, SATLike 3.0, AAAI 2020; our own implementation in `noiseless/cls_search.py`), budget 3n raw lookups. Its lowest-cost distinct strings center the explore runs.
- Explore runs: E = 2, 2, 2, 2, 4, 6, 8 for n = 8, 10, 12, 14, 16, 18, 20; layout `fixed_low`; layout noise 0.3 from run 2 on.
- Readout: K = 2n strings per round, over E+1 rounds.
- Fix-up: one Hamming-1 center per round (n lookups), always performed and counted.
- Relabel: one round, r = 150.
- Hardware: n=8-12 two cavities + two transmons (xL2, xs 25/35/50); n=14-20 three cavities + three transmons (xL3, xs35), cavity bits 4/4/3, 5/4/4, 5/5/5, 6/6/5.
- Lookup counting: every evaluated string counts, repeats included; identical cost in every trial.

Lookups = 3n (classical) + 2n(E+1) (readout) + n(E+1) (fix-up).

## Results

| n | E | classical | readout | fix-up | lookups | lookups / 2^n | evals | success | mean p(GS) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 24 | 48 | 24 | 96 | 37.5% | 505 | 10/10 | 0.987 |
| 10 | 2 | 30 | 60 | 30 | 120 | 11.7% | 585 | 10/10 | 0.988 |
| 12 | 2 | 36 | 72 | 36 | 144 | 3.52% | 705 | 10/10 | 0.991 |
| 14 | 2 | 42 | 84 | 42 | 168 | 1.03% | 727 | 10/10 | 0.961 |
| 16 | 4 | 48 | 160 | 80 | 288 | 0.439% | 1153 | 10/10 | 0.972 |
| 18 | 6 | 54 | 252 | 126 | 432 | 0.165% | 1579 | 10/10 | 0.963 |
| 20 | 8 | 60 | 360 | 180 | 600 | 0.0572% | 2005 | 10/10 | 0.959 |

n=20 p(GS) per instance 0-9: 0.955, 0.968, 0.957, 0.967, 0.943, 0.963, 0.966, 0.960, 0.945, 0.963.

Compared with the Oct 13 final recipe (SA 10n), lookups drop from 152, 190, 228, 266, 400, 558, 740 by 7n each; evals unchanged.

## Classical stage alone (its best string = ground state, out of 10)
n=8..20: 1, 4, 5, 3, 2, 1, 2. The string handed to the relabel round was the ground state in 10/10 trials at every n.

## Caveats
- 10-trial screens; a 5n-budget version of the same search failed one n=20 instance (0.90 success), so n=20 results are sensitive to single hard instances.
- Noiseless; quantum evals are not counted as lookups.
- An exact MaxSAT solver (RC2) solves these instances easily (EXACT_MAXSAT_SCALING.md). Standalone SATLike-style search needs 702 lookups (strict count) for 0.9 success at n=20 (STRONG_CLASSICAL_BASELINES.md).

## Data
- Per-trial records: `noiseless/results/relayout_runs/ovn_sl3/n*.jsonl`
- Log of all variants tried: `noiseless/results/OVERNIGHT_IMPROVEMENTS.md`
