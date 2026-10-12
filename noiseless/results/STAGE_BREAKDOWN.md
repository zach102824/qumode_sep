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
- At n = 18 the R1 success 1.00 is the second relabel center; at n = 18 one trial's guess was wrong after explore but the R1 relabel round still reached the GS.
- Not recorded: whether the GS was anywhere among the SA candidates; lookups after the explore and relabel stages. Recomputing those would need a code change and a re-run.
- n = 8 file holds 100 trials (two 50-trial seeds); n = 20 is still running.
