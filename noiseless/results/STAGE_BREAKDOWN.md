# Stage-by-stage breakdown of the final-table runs (Oct 12, 2026)

All numbers are recomputed from the saved per-trial records (`noiseless/results/relayout_runs/kbit/*.jsonl`); no new jobs were run. Plain absolute values, averaged over trials. Each trial = one random instance (10 instances per n) with its own seed.

Stages:
- **SA**: fraction of trials where the SA layout's best string (`sa_best_is_ground`) is the ground state (GS).
- **Explore**: after the E explore runs with top-48 readout + Hamming-1 fix-up of the 3 lowest-energy states. *Run find* = fraction of explore runs whose fix-up found the GS. *Guess* = fraction of trials where the current best guess (the relabel center) is the GS. *Argmax* = best-so-far argmax = GS, and p(GS) of that state.
- **R1, R2**: best-so-far after each relabel round (argmax success / mean p(GS)).
- Evals are cumulative circuit evaluations. Lookups: only SA lookups and the trial total are stored; per-stage lookups after SA are not recorded.

| n | device, run | trials | SA: best = GS | SA lookups | Explore: run find | Explore: guess = GS | Explore: argmax success / p(GS) | evals after explore | R1 success / p(GS) | evals after R1 | R2 success / p(GS) | evals after R2 | total lookups |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 cav, E1 xL2 xs25 sab32 R1 | 100 | 0.20 | 57 | 1.00 | 1.00 | 0.35 / 0.184 | 102 | 1.00 / 0.987 | 403 | – | – | 126 |
| 10 | 2 cav, E1 xL2 xs35 sab50 R1 | 50 | 0.02 | 62 | 1.00 | 1.00 | 0.36 / 0.127 | 142 | 1.00 / 0.990 | 443 | – | – | 163 |
| 12 | 2 cav, E1 xL2 xs50 sab72 R1 | 50 | 0.00 | 72 | 0.98 | 0.98 | 0.18 / 0.040 | 202 | 0.98 / 0.967 | 503 | – | – | 192 |
| 14 | 3 cav 4/4/3, E2 r100 sab64 R2 | 50 | 0.00 | 64 | 1.00 | 1.00 | 0.58 / 0.060 | 426 | 1.00 / 0.909 | 627 | 1.00 / 0.932 | 828 | 257 |
| 16 | 3 cav 5/4/4, E4 r100 sab128 R2 | 50 | 0.02 | 128 | 0.95 | 1.00 | 0.66 / 0.074 | 852 | 1.00 / 0.909 | 1053 | 1.00 / 0.934 | 1254 | 435 |
| 18 | 3 cav 5/5/5, E4 r100 sab162 R2 (10-trial screen; confirm pending) | 10 | 0.00 | 162 | 0.73 | 0.90 | 0.20 / 0.006 | 852 | 1.00 / 0.930 | 1053 | 1.00 / 0.941 | 1254 | 519 |

Notes:
- SA alone almost never lands on the GS (0–20%). Its job is only to seed layouts; the explore + top-K + fix-up step is what finds the GS (guess = GS in 98–100% of trials for n ≤ 16, 90% at n = 18 screen), and the relabel round turns that guess into a high-probability state (p(GS) 0.006–0.18 → 0.91–0.99).
- `sa_sample` always spends 64 random probes before annealing, so with sab ≤ 64 (n = 8, 10, 14) the "SA" is effectively 64 random lookups (that is why n = 8 shows 57 and n = 10 shows 62 distinct lookups despite sab 32 / 50).
- At n = 18, 1 of 10 trials had a wrong guess after explore, yet R1 still reached the GS (the top-K readout and fix-up run again inside the relabel round).
- Not recorded: whether the GS was anywhere among the SA candidates; lookups after the explore and relabel stages. Recomputing those would need a code change and a re-run.
- n = 8 file holds 100 trials (two 50-trial seeds); n = 20 is still running.

## Classical lookups for the ONE-relabel-round table (added Oct 12, 2026)

Lookups = distinct strings whose energy was evaluated (`n_lookups_distinct`). Only the SA count and the trial total are stored per trial; the split between explore and relabel round 1 is **not** recorded, and the readout distributions are not saved, so it can't be replayed classically.

How the through-round-1 totals were obtained:
- n = 8, 10, 12: the runs are already R=1 → totals are exact.
- n = 18: an R=1 run with the identical settings (E4 r100 sab162, same 10 seeds) exists. Checked per seed: its round-1 p(GS) matches the R=2 run's round-1 p(GS) exactly, so the runs are identical through round 1 and its total is the exact through-round-1 count (10-trial screen).
- n = 14 (sab64) and n = 16 (sab128): no matched R=1 run. Estimated as R=2 total minus the round-2 increment. That increment is measured from paired R=1/R=2 runs with the same seeds: +15.0 per trial at n=14 (sab98; range 7–21) and +15.4 at n=18 (sab162; range 10–28). Used 15 for both.
- Explore vs R1 split: not recorded. As a rough proxy, R1 adds about as many new strings as R2 (~15 in the 3-cav runs). This is an estimate, not a measurement.

| n | SA lookups | explore (top-K + fix-up) | relabel round 1 | total through R1 | post-SA (explore+R1) | status |
|---|---|---|---|---|---|---|
| 8 | 57 | not recorded | not recorded | 126 | 69 | total exact (100 trials) |
| 10 | 62 | not recorded | not recorded | 163 | 101 | total exact (50) |
| 12 | 72 | not recorded | not recorded | 192 | 120 | total exact (50) |
| 14 | 64 | ~163 (est.) | ~15 (est.) | ~242 | ~178 | total estimated (257 − 15) |
| 16 | 128 | ~277 (est.) | ~15 (est.) | ~420 | ~292 | total estimated (435 − 15) |
| 18 | 162 | ~327 (est.) | ~15 (est.) | 504 | 342 | total exact (10-trial screen, matched R=1 run) |

For reference: the R=1 n=14 sab98 screen gives 274; the R=1 n=18 sab64 50-trial confirm gives 399 (0.96 / 0.879, fails the bar).
