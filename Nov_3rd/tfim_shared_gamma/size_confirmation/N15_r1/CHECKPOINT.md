# N=15 checkpoint — workload stopped before completion

Stopped on request at 2026-11-08 03:25 UTC, before any p=12 line was printed and before `results/` was written. The runner stores records only at the end, so the angle vectors and the per-start table are not on disk. This file is the stdout ledger. It is not a finished comparison and it is not a win or a loss.

Preregistration commit `771465f3`. Manifest `de5f47b1bd1b4e3cfced8694c892d41f8cab9eb43f41ed285c3fccff6c9b0892`. Packing `dcd102b8bc45bec941dc9f3e474781ba9cf19c12dd893d068e83d50e09d41074`.

Krylov spectrum printed before training: E0 = -10.857659100961786, Emax = 32.45041919208127, full gap = 0.00043523124187672124.

Candidate, 48 chains, maxfun 40, all six depths completed:

| L | best error | F | chains | cumulative optimizer evals |
|---|---|---|---|---|
| 1 | 1.450583 | 0.0631 | 48 | 2037 |
| 2 | 1.368291 | 0.0716 | 48 | 4038 |
| 3 | 0.969281 | 0.1232 | 48 | 6058 |
| 4 | 0.869658 | 0.1488 | 48 | 8083 |
| 5 | 0.757495 | 0.1966 | 48 | 10103 |
| 6 | 0.674749 | 0.2458 | 48 | 12120 |

QAOA floor was frozen from that ledger at 4 × 12120 = 48480, then the ladder started. Printed before the stop:

| p | best error | F | starts | cumulative QAOA evals |
|---|---|---|---|---|
| 1 | 2.340133 | 0.0334 | 48 | 913 |
| 2 | 1.543415 | 0.0633 | 48 | 2081 |
| 3 | 1.225548 | 0.0870 | 48 | 3410 |
| 4 | 1.005934 | 0.1176 | 48 | 4795 |
| 5 | 0.854689 | 0.1472 | 48 | 6562 |
| 6 | 0.745707 | 0.1781 | 48 | 8493 |
| 7 | 0.655724 | 0.2100 | 48 | 10454 |
| 8 | 0.590727 | 0.2422 | 48 | 12473 |
| 9 | 0.531964 | 0.2766 | 48 | 14493 |
| 10 | 0.480591 | 0.3113 | 48 | 16512 |
| 11 | 0.438027 | 0.3406 | 48 | 18532 |

p=12 had not printed. Extras had not started. Shortfall against the frozen floor is the entire remainder, 48480 − 18532 = 29948, plus whatever p=12 would have added. No further N=15 or N=18 run is authorized.
