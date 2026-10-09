## Recommended K=1 recipe and per-n best (100 trials, 20 H × 5 seeds)

Recipe: L4, s=50, `--explore-mask perm --xeta 16 --reta 2`, exploit R=4 × r=200, top-1 candidate per round,
Hamming-1 fix-up. Only E grows with n.

| n | E | success | mean p(GS) | leakage | evals | distinct lookups | lookups/2^n |
|---|---|---|---|---|---|---|---|
| 8 | 2 | 1.000 | 0.991 | 0.0000 | 2412 | 10 | 0.040 |
| 10 | 3 | 1.000 | 0.990 | 0.0000 | 2816 | 20 | 0.020 |
| 12 | 5 | 1.000 | 0.989 | 0.0000 | 3624 | 46 | 0.011 |

Evals = 404·E + 1604. E = 2, 3, 5 at n = 8, 10, 12 gives 2412 → 2816 → 3624, i.e. ×1.17 and ×1.29 per +2 qubits
(≈ 1.08^n to 1.14^n). Distinct lookups ≈ 10 → 20 → 46 (≈ 1–4% of 2^n, falling with n).
At n = 14 the discovery rate per run drops sharply (see finding 3), so the n=14 E is still being scanned (E = 10, 13).

## Status / next steps (for resuming)

- Detached drivers (resumable; finished trials are skipped): `logs/queue_m.sh` (n=14 E10/E13 K1, E6K4, E8K2;
  n12 E4K2, n10 E2K2), then `logs/queue_n.sh` (minimal E per n incl. n8 E1 / n10 E2 / n12 E6; cheaper explore
  steps xs=35; exploit shape r250R3 / r150R5 / lr 0.07, 0.15; xeta 24/32; random vs perm at η×16; a second
  seed block (trials 5–9, tag suffix `_seedB`) for the recommended n = 8/10/12 settings; n=16 probe with E8).
  `logs/disc_l.sh`: η anneal schedules per growth stage (4,16,64,64 and 16,16,16,64) at n = 12/14.
- Rebuild this file with `noiseless/results/build_explore_md.sh` (hand-written head/tail + generated table).
- Open ideas: a cold final "concentration" stage so top-1 catches the GS at n ≥ 14 (GS often rank 2–8);
  fit E(n) once n = 14 numbers are in.
