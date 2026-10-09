## Best K=1 settings per n so far (~0.99 / ~0.99)

| n | setting | success | mean p(GS) | evals | distinct lookups | lookups/2^n |
|---|---|---|---|---|---|---|
| 8 | L4s50_r200_R4_E2K1perm_xeta16 | 1.000 | 0.989 | 2412 | 10 | 0.040 |
| 10 | L4s50_r200_R4_E3K1perm_xeta16 | 1.000 | 0.988 | 2816 | 20 | 0.020 |
| 12 | L4s50_r200_R4_E5K1perm_xeta16_reta2 | 1.000 | 0.989 | 3624 | 46 | 0.011 |

Trend so far: exploit cost fixed (R=4 × r=200 = 1604 evals); explore runs E ≈ 2, 3, 4–5 at n = 8, 10, 12 (≈ +1 run per +2 qubits,
404 evals each), total evals 2412 → 2816 → 3220–3624, i.e. ≈ 1.08–1.1^n.

## Status / next steps (for resuming)

- Driver scripts in `logs/queue_*.sh` (detached, resumable: finished trials are skipped).
- Next: exploit η scale (reta 2–8), cheaper exploit (r=150, R=3), n=14 (E5–7, xeta16), possibly n=16;
  then fit E(n), η(n) and refresh the per-n best table with distinct lookup fractions.
