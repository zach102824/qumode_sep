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
