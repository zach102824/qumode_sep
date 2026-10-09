# Relayout, denser n=10 set

- Hamiltonians: `Hamiltonians/four_sat_scaling.py` n=10 set, 20 instances (29 to 38 clauses, mean 32.5), unique planted ground state.
- Simulation: noiseless, 32 Fock levels per cavity (16 code levels + 16). Population above Fock 15 counts as leakage and counts against p(GS).
- Runner: `noiseless/run_relayout_kbit.py --n 10 --ham-set scaling --workers 8 --nf-check 0`, lr 0.1, cavity-only XOR, seed 20260917, paired seeds across settings.
- Screens: `--trials 5` (100 trials = 20 H x 5). Confirm: `--trials 25` (500 trials = 20 H x 25).
- Setting LxsY-rZ-RW: round-0 growth to L layers with s SPSA-Adam steps per stage, then R relabel rounds of r steps each. evals = mean function evaluations per trial.
- Mean leakage = mean over trials of the population above the code space in the selected final state.

| setting | success | mean p(GS) | mean leakage above code space | evals | trials |
|---|---|---|---|---|---|
| L4s50-r400-R8 | 0.710 | 0.705 | 0.0444 | 6812 | 100 |
| L4s50-r800-R8 | 0.820 | 0.760 | 0.0458 | 13212 | 100 |
| L4s100-r800-R8 | 0.880 | 0.858 | 0.0166 | 13612 | 100 |
| L4s100-r800-R12 | 0.890 | 0.878 | 0.0200 | 20016 | 100 |
| L4s100-r800-R16 | 0.886 | 0.876 | 0.0237 | 26420 | 500 |
| L4s100-r1600-R8 | 0.850 | 0.806 | 0.0433 | 26412 | 100 |
| L6s100-r600-R8 | 0.800 | 0.629 | 0.0624 | 10814 | 100 |
| L6s100-r800-R8 | 0.820 | 0.716 | 0.0363 | 14014 | 100 |
| L8s100-r600-R8 | 0.800 | 0.362 | 0.0845 | 11216 | 100 |
| L8s100-r800-R8 | 0.800 | 0.409 | 0.0950 | 14416 | 100 |

No setting reached mean p(GS) 0.90. Only L4s100-r800-R16 was confirmed at 500 trials; the L4s100-r800-R12 confirm was stopped before any new trials (cost), so it stays at 100.
