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

Notes
- The "adaptive centre" variant re-centres each later layout on the current best guess. It re-finds the same
  state (high find rate) but locks onto wrong guesses, so final quality drops. Not useful.
- One earlier run (n14 adaptive nz0.6) collided with the plain nz0.6 checkpoint tag and is excluded;
  the tag now carries the full samode name.
- Find per explore run is the fraction of explore runs whose Hamming-1-polished peak is the ground state.

### Takeaway so far
SA layout saves about one explore run at n=12 (0.90 at E=4, 2218 evals, versus E=5 baseline)
and about four at n=14 (0.88 at E=11, 5046 evals, versus about E=15 baseline). Classical cost is n^2 SA lookups
on top of the usual ones (about 180 lookups total at n=12, about 300 at n=14).
