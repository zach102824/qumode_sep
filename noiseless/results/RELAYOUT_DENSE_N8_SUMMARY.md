# Relayout, denser n=8 set: all settings at 500 trials

- Hamiltonians: `Hamiltonians/four_sat_scaling.py` n=8 set, 20 instances, 25 to 30 clauses each (mean 27.6, m/n about 3.45), unique planted ground state.
- Simulation: noiseless, 24 Fock levels per cavity (8 code levels + 16). Population above Fock 7 counts as leakage and counts against p(GS).
- Runner: `noiseless/run_relayout_kbit.py --n 8 --ham-set scaling --trials 25 --workers 8 --nf-check 0`, lr 0.1, seed 20260917. That is 500 trials per setting (20 H x 25), with paired seeds across settings.
- Setting LxsY-rZ-RW: round-0 growth to L layers with s SPSA-Adam steps per stage, then R relabel rounds of r steps each. evals = mean function evaluations per trial.
- Mean leakage = mean over trials of the population above the code space in the selected final state.

| setting | success | mean p(GS) | mean leakage above code space | evals | trials |
|---|---|---|---|---|---|
| L4s10-r200-R4 | 0.876 | 0.848 | 0.0365 | 1688 | 500 |
| L4s25-r200-R4 | 0.928 | 0.911 | 0.0151 | 1808 | 500 |
| L4s25-r200-R8 | 0.958 | 0.946 | 0.0096 | 3412 | 500 |
| L4s25-r300-R4 | 0.954 | 0.932 | 0.0125 | 2608 | 500 |
| L4s25-r300-R8 | 0.968 | 0.965 | 0.0068 | 5012 | 500 |
| L4s50-r200-R8 | 0.964 | 0.959 | 0.0073 | 3612 | 500 |
| L4s50-r300-R8 | 0.984 | 0.978 | 0.0036 | 5212 | 500 |
| L4s25-r400-R8 | 0.988 | 0.980 | 0.0020 | 6612 | 500 |
| L4s50-r400-R8 | 0.996 | 0.989 | 0.0005 | 6812 | 500 |

L4s10-r200-R4 is the headline setting. Only L4s50-r400-R8 reaches the headline level (0.996 / 0.989), at about 4x the headline's evals.
