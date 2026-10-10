# Improvement search at n = 12 and n = 14 (wmaxsat, Gray code)

Common settings: L = 4, s = 50, r = 150, R = 2 relabel rounds, lr = 0.05, xeta = 16, reta = 2, 8 workers.
Evals per trial = 404 E + 602. Trials = 10 instances x 5 starts = 50 unless noted.
Lookups = distinct classical energy lookups per trial, INCLUDING the SA pre-pass.

## Idea 1: SA-chosen layout (explore-mask sa)

Before any circuit, a short simulated-annealing pass (budget sab lookups, default n^2) collects low-energy
states. Their bit marginals choose each explore run's layout and starting bit flips
(noiseless/classical_layout.py). Modes: unc = uncertain bits on high-impact slots;
fix = fixed (backbone) bits to low slots (samode fixed_low); nz = per-run layout noise.
The SA best state itself is the ground state in only 12% (n=12) and 2-4% (n=14) of trials, so SA alone does not solve it.

### Baselines (random layouts)

| n | E | success | mean p(GS) | evals |
|---|---|---|---|---|
| 12 | 5 | 0.92 | 0.914 | 2622 |
| 14 | 9 | 0.72 | 0.719 | 4238 |
| 14 | 13 | 0.82 | 0.817 | 5854 |
| 14 | 17 | 0.94 | 0.933 | 7470 |

### n = 12, SA layout

| setting | E | trials | find per explore run | success | mean p(GS) | evals | lookups (SA part) |
|---|---|---|---|---|---|---|---|
| sab144 nz0.15 unc | 5 | 50 | 0.54 | 0.90 | 0.894 | 2622 | 176 (144) |
| sab144 nz0.15 fix | 5 | 50 | 0.38 | 0.94 | 0.935 | 2622 | 188 (144) |
| sab144 nz0.15 fix | 4 | 50 | 0.39 | 0.90 | 0.898 | 2218 | 181 (144) |
| sab144 nz0.6 fix | 4 | 50 | 0.41 | 0.88 | 0.879 | 2218 | 181 (144) |
| sab144 nz0.3 fix, adaptive centre | 4 | 50 | 0.69 | 0.86 | 0.858 | 2218 | 167 (144) |
| sab144 nz0.15 fix | 3 | 50 | 0.36 | 0.82 | 0.820 | 1814 | 176 (144) |
| sab144 nz0.3 unc | 3 | 50 | 0.55 | 0.82 | 0.818 | 1814 | 167 (144) |
| sab144 nz0.3 fix, adaptive centre | 3 | 50 | 0.63 | 0.78 | 0.780 | 1814 | 166 (144) |

### n = 14, SA layout

| setting | E | trials | find per explore run | success | mean p(GS) | evals | lookups (SA part) |
|---|---|---|---|---|---|---|---|
| sab196 nz0.15 unc | 9 | 30 | 0.29 | 0.73 | 0.727 | 4238 | 272 (196) |
| sab196 nz0.3 unc | 9 | 30 | 0.25 | 0.70 | 0.696 | 4238 | 282 (196) |
| sab196 nz0.6 unc | 9 | 30 | 0.24 | 0.83 | 0.828 | 4238 | 286 (196) |
| sab98 nz0.3 unc | 9 | 30 | 0.21 | 0.80 | 0.794 | 4238 | 201 (98) |
| sab196 nz0.15 fix | 7 | 50 | 0.17 | 0.70 | 0.702 | 3430 | 279 (196) |
| sab196 nz0.15 fix | 9 | 50 | 0.17 | 0.82 | 0.818 | 4238 | 301 (196) |
| sab196 nz0.6 fix | 9 | 50 | 0.17 | 0.84 | 0.836 | 4238 | 299 (196) |
| sab98 nz0.15 fix | 9 | 50 | 0.15 | 0.84 | 0.836 | 4238 | 211 (98) |
| sab196 nz0.3 fix, adaptive centre | 9 | 50 | 0.40 | 0.48 | 0.479 | 4238 | 233 (196) |
| sab196 nz0.15 fix | 11 | 50 | 0.18 | 0.88 | 0.875 | 5046 | 319 (196) |
| sab196 nz0.15 fix (confirm) | 13 | 50 | 0.16 | 0.90 | 0.897 | 5854 | 340 (196) |

Notes
- The "adaptive centre" variant re-centres each later layout on the current best guess. It re-finds the same
  state (high find rate) but locks onto wrong guesses, so final quality drops. Not useful.
- One earlier run (n14 adaptive nz0.6) collided with the plain nz0.6 checkpoint tag and is excluded;
  the tag now carries the full samode name.
- Find per explore run is the fraction of explore runs whose Hamming-1-polished peak is the ground state.

### Takeaway so far
Confirmed: n=14 reaches 0.90 / 0.897 at E=13, 5854 evals (old recipe needed E=17, 7470 evals for 0.94 / 0.933; E=13 old gave 0.82 / 0.817).
SA layout saves about one explore run at n=12 (0.90 at E=4, 2218 evals, versus E=5 baseline)
and about four at n=14 (0.88 at E=11, 5046 evals, versus about E=15 baseline). Classical cost is n^2 SA lookups
on top of the usual ones (about 180 lookups total at n=12, about 300 at n=14).

## Status (Oct 10, 2026, about 17:00 Shanghai)

Best settings
- n=12: SA layout, sab144, nz0.15, samode fixed_low, E=4: 0.90 / 0.898, 2218 evals, about 181 lookups. E=5: 0.94 / 0.935, 2622 evals.
- n=14: SA layout, sab196, nz0.15, samode fixed_low, E=13: 0.90 / 0.897, 5854 evals, about 340 lookups.
  Eval scaling for about 0.9 from n=12 (2218) to n=14 (5854): about 1.62 per added variable, still the n=14 jump.

New code options (committed): --xstop S / --xstopq Q (stop an explore run after growth stage S if its mean energy
is worse than the Q quantile of earlier runs in that trial, padded with identity layers; saves evals),
--thr EPS (relabel rounds use cost min(E - E(guess) - EPS*std, 0), so only states at or below the guess score).
Top-K readout already exists as --rawk K --polt T.

Running detached (setsid nohup), in order
- logs/improve/queue5.sh (n14 sab98 E11 and E13, n12 sab72 E4 and E5; writes logs/improve/queue5.done)
- logs/improve/queue6.sh waits for queue5.done, then 30-trial screens: n14 E11 with thr0.05, xstop1, rawk16 polt2, thr0;
  n12 E4 with thr0.05, xstop1, rawk16 polt2; n14 E15 with xstop1. Outputs logs/improve/NAME.json, checkpoints in
  noiseless/results/relayout_runs/kbit/. Compare against trials 0-2 of the matching 50-trial SA runs.

Next ideas
- Confirm at 50 trials whichever of thr / xstop / rawk helps, then stack them.
- Early stop plus more explore runs at equal evals (for example n14 E15 with xstop1 against E13 plain).
- Warm-start explore runs from the best previous parameters (not yet implemented).
- Diverse layouts: keep and mutate layouts whose runs reached low energy (the re-centring variant failed; mutation of the layout itself is untested).

### Half SA budget (50 trials, sab = n^2/2)
| n | sab | E | success | mean p(GS) | evals |
|---|---|---|---|---|---|
| 12 | 72 | 4 | 0.90 | 0.894 | 2218 |
| 12 | 72 | 5 | 0.90 | 0.894 | 2622 |
| 14 | 98 | 11 | 0.86 | 0.857 | 5046 |
| 14 | 98 | 13 | 0.90 | 0.895 | 5854 |

Halving the SA budget gives the same quality at both n, so classical lookups drop by about n^2/2 for free.

### Status update (Oct 10, 2026, about 17:00 Shanghai)
- Running: logs/improve/queue6.sh (30-trial screens of thr, xstop, rawk), then logs/improve/queue7.sh
  (50 trials: n14 sab49 E13, n12 sab36 E4, n14 E13 sab98 thr0.05, n14 E15 sab98 xstop1, n12 E4 sab72 thr0.05).
- Next ideas unchanged: warm starts, layout mutation, stacking whichever of thr / xstop / rawk helps.

## Queue6 screens (30 trials, paired with trials 0-2 of the matching 50-trial SA run; Oct 10, 2026, about 17:00 Shanghai)
Same trials and seeds in both columns (noiseless/analyze_paired.py). Baseline = SA layout, nz0.15, fixed_low, sab = n^2.

| n | E | variant | success | mean p(GS) | evals | lookups | baseline success / p(GS) / evals / lookups |
|---|---|---|---|---|---|---|---|
| 14 | 11 | thr 0.05 | 0.90 | 0.895 | 5046 | 318 | 0.90 / 0.895 / 5046 / 317 |
| 14 | 11 | xstop 1 (early stop) | 0.77 | 0.764 | 4137 | 325 | 0.90 / 0.895 / 5046 / 317 |
| 14 | 11 | thr 0 | 0.67 | 0.441 | 5046 | 331 | 0.90 / 0.895 / 5046 / 317 |
| 14 | 11 | rawk 16, polt 2 | 1.00 | 0.993 | 5046 | 407 | 0.90 / 0.895 / 5046 / 317 |
| 14 | 15 | xstop 1 | 0.83 | 0.831 | 5315 | 369 | E13 plain: 0.90 / 0.899 / 5854 / 337 |
| 12 | 4 | thr 0.05 | 0.93 | 0.928 | 2218 | 179 | 0.93 / 0.928 / 2218 / 178 |
| 12 | 4 | xstop 1 | 0.90 | 0.894 | 1949 | 182 | 0.93 / 0.928 / 2218 / 178 |
| 12 | 4 | rawk 16, polt 2 | 1.00 | 0.993 | 2218 | 237 | 0.93 / 0.928 / 2218 / 178 |

Reading
- thr 0.05 does nothing (same numbers); thr 0 is clearly harmful. Drop thr.
- xstop (early stop of explore runs after growth stage 1, then more explore runs) loses quality: n14 E15 xstop gives 0.83 at 5315 evals,
  worse than plain E11 at 5046 evals (0.90) and E13 at 5854 evals. Early stopping discards runs that would have found the answer. Drop xstop.
- rawk 16 with polt 2 (energy-ranked readout pool: 16 most probable states per round, 1 lookup each, Hamming-1 fix-up of the 2 lowest-energy ones)
  is the only clear win: 1.00 success on 30 of 30 trials at both n, at the same evals. Extra lookups: about 90 at n=14 and 60 at n=12.
  This needs a 50-trial confirmation with half-size SA (sab = n^2/2) and fewer explore runs (queue8).

## Code added this round
- --xwarm S / --xmut M: warm-started explore runs. After the first explore run, each later explore run starts from the final parameters of the
  lowest-energy earlier explore run, on that run's layout with M random slot swaps (same centre state), and runs one stage of S steps
  at full depth instead of a full 4-stage growth (about 101 evals instead of 404 at S = 50). Tag suffix _xwSmM.
- noiseless/analyze_paired.py: paired comparison of checkpoints on shared (inst, trial) pairs.
- Fixed a crash in the explore early-stop check when the cool stage is on (stages missing).

## Queued (detached, chained by done-files)
- queue7 (running): 50 trials, n14 sab49 E13, n12 sab36 E4, n14 E13 sab98 thr0.05, n14 E15 sab98 xstop1, n12 E4 sab72 thr0.05.
- queue8 (waits for queue7.done): rawk16 polt2 at 50 trials with sab = n^2/2 and fewer explore runs:
  n14 E11, E9, E7 (sab98), n12 E4, E3, E2 (sab72); rawk8; sab = n^2/4; polt3; rawk plus thr0.05.
- queue9 (waits for queue8.done): warm-started explore runs (30 trials): n14 E13 xwarm50 m2 with and without rawk, xwarm100 m4, n14 E9 m1, n12 E5.

## Queue7 (50 trials, finished)
| n | E | SA budget | variant | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|---|
| 14 | 13 | 49 (n^2/4) | plain | 0.92 | 0.914 | 5854 | 232 |
| 14 | 13 | 98 | thr 0.05 | 0.90 | 0.895 | 5854 | 256 |
| 14 | 15 | 98 | xstop 1 | 0.92 | 0.914 | 5171 | 291 |
| 12 | 4 | 36 (n^2/4) | plain | 0.88 | 0.879 | 2218 | 109 |
| 12 | 4 | 72 | thr 0.05 | 0.90 | 0.894 | 2218 | 115 |

- n=14 E15 with xstop 1 at sab98 reaches 0.92 / 0.914 at 5171 evals (about 12% fewer than plain E13 at 5854), with 291 lookups.
  At 30 trials xstop with sab196 looked worse (0.83), so this is within noise of 50-trial resolution; the gain is small.
- A quarter-size SA budget (sab = n^2/4) is still fine at n=14 (0.92 / 0.914 at E13, 232 lookups) and costs about 0.02 at n=12.
- thr 0.05 again changes nothing.

## Queue8 results so far (rawk 16, polt 2, 50 trials; Oct 10, 2026, about 17:15 Shanghai)
Settings: wmaxsat, Gray code, L4 s50, r150, R2, explore with SA layouts (fixed_low), SA budget sab = n^2/2 (98 at n=14, 72 at n=12).
"Lookups" = all classical energy lookups (SA + readout pool + fix-up). n^2 is 144 at n=12 and 196 at n=14.

| n | E | variant | trials | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|---|
| 12 | 5 | plain SA baseline | 50 | 0.92 | 0.914 | 2622 | n/a (about n^2) |
| 12 | 4 | rawk16 polt2, sab72 | 50 | 1.00 | 0.993 | 2218 | 174 |
| 12 | 3 | rawk16 polt2, sab72 | 50 | 1.00 | 0.993 | 1814 | 162 |
| 14 | 17 | plain baseline | 50 | 0.94 | 0.933 | 7470 | n/a (about n^2) |
| 14 | 13 | plain SA, sab98 | 50 | 0.90 | 0.895 | 5854 | 256 |
| 14 | 11 | rawk16 polt2, sab98 | 50 | 0.96 | 0.953 | 5046 | 322 |
| 14 | 9 | rawk16 polt2, sab98 | 50 | 0.94 | 0.933 | 4238 | 295 |
| 14 | 7 | rawk16 polt2, sab98 | 24 of 50 so far | 1.00 | 0.993 | 3430 | 263 |

Reading
- Confirmed at 50 trials: n=12 with E=3 gives 1.00 success and 0.993 mean p(GS) at 1814 evals, which is 31% fewer evals than the E=5 baseline (2622) and 18% fewer than the SA-layout baseline at E=4 (2218). Lookups 162, about 1.1 n^2.
- n=14 with E=9 gives 0.94 / 0.933 at 4238 evals, the same quality as the E=17 baseline (7470 evals) with 43% fewer evals, and 16% fewer than the SA-layout baseline at E=11 (5046, 0.86 / 0.857). Lookups 295, about 1.5 n^2.
- n=14 E=11 with rawk16 gives 0.96 / 0.953 at 5046 evals (the earlier 30-trial 1.00 was optimistic at full-size SA).
- Half-size SA (sab = n^2/2) costs a little at n=14 (0.96 vs 1.00 at E=11, 50 vs 30 trials, not the same trials). Rest of queue8 still running (E=7 and E=2 points, rawk8, quarter SA, polt3, thr).

## Queued after queue8 (all detached, chained by done-files)
- queue9 (30 trials): warm-started explore runs, with and without rawk16.
- queue10 (50 trials, rawk16 polt2): n14 E5, E6; n12 E1; n14 E7 sab49; n12 E2 sab36; layout diversity via SA noise 0.4 (n14 E7, n12 E2); rawk24 (n14 E7); top-K 3 (n14 E7).
- queue11 (50 trials, rawk16 polt2): cheaper explore runs (xs 35, xL 3) at n14 E9 and n12 E3; warm-started explore runs (n14 E9, n12 E4); n14 E7 polt3; n14 E7 rawk8.

## Queue8 / queue9 / queue10 results (Oct 10, 2026, about 18:20 Shanghai)
All rawk 16, polt 2 unless noted; wmaxsat, Gray code, L4 s50, r150, R2, SA layouts (fixed_low); 50 trials unless noted.
Lookups = distinct classical lookups per trial including SA, readout pool and fix-up (n^2 = 144 at n=12, 196 at n=14).

| n | E | variant | trials | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|---|
| 14 | 7 | sab98 | 50 | 0.94 | 0.934 | 3430 | 266 |
| 14 | 5 | sab98 | 50 | 0.84 | 0.834 | 2622 | 236 |
| 14 | 9 | sab49 (n^2/4) | 50 | 0.96 | 0.953 | 4238 | about 200 |
| 14 | 9 | sab98, rawk 8 | 50 | 0.92 | 0.913 | 4238 | 227 |
| 14 | 9 | sab98, polt 3 | 50 | 0.94 | 0.933 | 4238 | n/a (same as polt 2) |
| 14 | 9 | sab98, thr 0.05 | 50 | 0.94 | 0.933 | 4238 | n/a |
| 12 | 2 | sab72 | 50 | 0.90 | 0.894 | 1410 | 146 |
| 12 | 3 | sab72, rawk 8 | 50 | 0.94 | 0.934 | 1814 | 138 |
| 12 | 3 | sab36 (n^2/4) | 50 | 0.96 | 0.953 | 1814 | 109-156 |

Warm-started explore runs (xwarm / xmut, 30 trials) are clearly bad:
| n | E | variant | success | mean p(GS) | evals |
|---|---|---|---|---|---|
| 14 | 13 | xwarm 50 mut 2, plain | 0.27 | 0.265 | 2218 |
| 14 | 13 | xwarm 50 mut 2, rawk16 | 0.33 | 0.331 | 2218 |
| 14 | 13 | xwarm 100 mut 4, rawk16 | 0.40 | 0.398 | 3418 |
| 14 | 9 | xwarm 50 mut 1, rawk16 | 0.30 | 0.299 | 1814 |
| 12 | 5 | xwarm 50 mut 2, rawk16 | 0.63 | 0.629 | 1410 |
| 12 | 5 | xwarm 50 mut 2, plain | 0.37 | 0.366 | 1410 |
Warm starts drop the layout diversity and the growth stages that find the answer; per eval they are worse than
cold explore runs with the SA layout. Drop xwarm.

Confirmed (50 trials) best results so far, against the baselines (n=12 E5: 0.92 / 0.914 at 2622; n=14 E17: 0.94 / 0.933 at 7470):
- n=12: E=3, rawk16 polt2, sab72: 1.00 success, 0.993 mean p(GS), 1814 evals (31% fewer evals than 2622), 162 lookups.
  E=2: 0.90 / 0.894 at 1410 evals (46% fewer), 146 lookups.
- n=14: E=7, rawk16 polt2, sab98: 0.94 / 0.934 at 3430 evals (54% fewer than 7470), 266 lookups (1.4 n^2).
  E=9 with sab49: 0.96 / 0.953 at 4238 evals, about 200 lookups (about n^2).
- Rawk size: rawk 16 beats rawk 8 (n14 E9: 0.94 vs 0.92; n12 E3: 1.00 vs 0.94). polt 3 and thr give nothing beyond polt 2.
- Evals scaling of the best points (n=12 E3: 1814, n=14 E7: 3430): ratio 1.89 over two steps in n, about 1.37 per unit n.
  The E needed still grows from 3 to 7 because single explore runs find the answer only about 16% of the time at n=14.

## Queued (Oct 10, 2026, about 18:20 Shanghai; all detached, chained by done-files)
- queue10 (running, then queue11): n14 E6 (running), n12 E1, n14 E7 sab49, n12 E2 sab36, SA noise 0.4 diversity (n14 E7, n12 E2), rawk24 (n14 E7), top-K 3 (n14 E7).
- queue11: cheaper explore runs (xs 35, xL 3) at n14 E9 and n12 E3; warm-started explore runs at 50 trials (expected bad, per the 30-trial results);
  n14 E7 polt3; n14 E7 rawk8.
- queue12 (new, waits for queue11.done): cheaper relabel stage with rawk16 polt2: R1 (one relabel round) and r100 at n14 E7 and n12 E3;
  n14 E6 rawk24, n12 E2 rawk24; n14 E7 polt1; n14 E7 sab49 rawk32.
- Expected total: about 6 more hours of compute at 8 cores. Next check-in: read queue10/11/12 outputs in logs/improve and record them here.

## Queue10 / queue11 / queue12 confirmed (50 trials; Oct 10, 2026, about 20:55 Shanghai)

All SA layout fixed_low, rawk polt 2 unless noted; wmaxsat, Gray.

| n | E | setting | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|
| 14 | 7 | rawk24 | 0.98 | 0.974 | 3430 | 321 |
| 14 | 9 | xL3, rawk16 | 0.98 | 0.973 | 3329 | 294 |
| 14 | 9 | xs35, rawk16 | 0.96 | 0.953 | 3158 | 302 |
| 14 | 7 | R1, rawk16 | 0.94 | 0.931 | 3129 | 263 |
| 14 | 6 | rawk24 | 0.90 | 0.893 | 3026 | 301 |
| 14 | 7 | r100, rawk16 | 0.94 | 0.924 | 3230 | 266 |
| 14 | 7 | polt1, rawk16 | 0.94 | 0.934 | 3430 | 240 |
| 14 | 7 | sab49, rawk32 | 0.92 | 0.914 | 3430 | 349 |
| 14 | 7 | nz0.4, rawk16 | 0.94 | 0.934 | 3430 | 266 |
| 12 | 3 | R1, rawk16 | 1.00 | 0.988 | 1513 | 159 |
| 12 | 2 | nz0.4, rawk16 | 0.96 | 0.953 | 1410 | 147 |
| 12 | 2 | sab36, rawk16 | 0.94 | 0.933 | 1410 | 141 |
| 12 | 2 | rawk24 | 0.92 | 0.913 | 1410 | 168 |
| 12 | 3 | r100, rawk16 | 1.00 | 0.982 | 1614 | 162 |
| 12 | 3 | xL3, rawk16 | 0.90 | 0.895 | 1511 | n/a |
| 12 | 3 | xs35, rawk16 | 0.94 | 0.934 | 1454 | n/a |

Reading
- rawk24 at n14 E7 is the quality leader (0.98 / 0.974 at 3430).
- One relabel round (R1) keeps quality and cuts about 301 evals (n14 E7: 3129; n12 E3: 1513).
- Shallower explore (xL3) and fewer steps (xs35) cut evals at matched E without losing quality.
- Matching cheapest 0.9+ points before the next stack (n12 at 1410, n14 at 3129): about 1.49 per added variable.

## Stacking winners (queues 13-16; started Oct 10, 2026, about 20:53 Shanghai)

Priority: cut the n12 to n14 step toward about 1.12^n by stacking R1 + rawk24 + cheaper explore (xL3 / xs35) at lower E, plus layout mutation.

Code added: `--lmut N` (cold layout mutation). After explore run 0, later explore runs use the lowest-energy earlier layout with N random slot swaps and a full cold growth (no warm parameters; distinct from the failed xwarm). Tag suffix `_lmN`.

### Early screen (30 trials)
| n | E | setting | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|
| 14 | 5 | R1, rawk24 | 0.90 | 0.890 | 2321 | 272 |
| 14 | 6 | R1, rawk24 | 0.93 | 0.922 | 2725 | 295 |
| 14 | 7 | R1, rawk24 | 1.00 | 0.990 | 3129 | 315 |
| 14 | 4 | R1, rawk24 | 0.87 | 0.854 | 1917 | 252 |
| 14 | 7 | rawk24, xL3 | 0.97 | 0.958 | 2723 | 320 |
| 14 | 5 | rawk24, xL3 | 0.90 | 0.893 | 2117 | 278 |
| 14 | 7 | R1, rawk24, xL3 | 0.97 | 0.949 | 2422 | 315 |

n14 E5 R1 rawk24 already matches 0.9 at 2321 evals (vs 3129 previous cheapest n14 0.9+). Against n12 at 1410 that is about 1.28 per added variable.
n14 E4 R1 rawk24 and xL3 stacks are screened; 50-trial confirms are in queue14.
E7 rawk24 xL3 at 30 trials: see table. 50-trial confirm queued.

### Running (detached, chained)
- queue13 (running): 30-trial screens of R1+rawk24 at E=4,5,6,7; xL3+rawk24; xs35+rawk24; nz0.4 stacks; n12 R1/nz40.
- queue14 (waits for queue13): 50-trial confirms of the same stacks at E=4..7, including xL3+R1 and xs35+R1.
- queue15 (waits for queue14): more aggressive confirms (rawk32, xL3+R1 at E7/E9, n12 E1).
- queue16 (waits for queue15): `--lmut` 2 and 4 screens then 50-trial confirms.

### Best confirmed so far (50 trials)
- n12: E3 R1 rawk16 1.00 / 0.988 at 1513; E2 nz0.4 0.96 / 0.953 at 1410.
- n14: E7 rawk24 0.98 / 0.974 at 3430; E9 xL3 0.98 / 0.973 at 3329; E7 R1 0.94 / 0.931 at 3129; E9 xs35 0.96 / 0.953 at 3158.

### Next ideas
- Confirm n14 E5 R1 rawk24 at 50 trials (target about 2321 evals for 0.9).
- Stack xL3 or xs35 on top of R1+rawk24 at E=4..6 (target under 2000 evals).
- Layout mutation (`--lmut`) vs sanoise diversity head-to-head.
- If E5 clears 0.9 at 50 trials, try E4 and E3 with the full stack.


---

**Superseded for the ladder view by [IMPROVE_N8_N16.md](IMPROVE_N8_N16.md)** (Oct 10, 2026 ~23:20 Shanghai). This file keeps the early n=12/14 idea history.
