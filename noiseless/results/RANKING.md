# Noiseless SPSA ranking

Updated: 2026-09-17 10:09 UTC

## Fleet #1 (all 6 U × L*∈{2,3,4} × 20 H × 5 trials × 200 SPSA steps)

Metric: ground-bitstring success (argmax bitstring == GS) and mean p(GS).
No jointly trained prep; fixed U frozen; random ECD init.

| Rank | U | L* | Success rate | Mean p(GS) | Trials |
|-----:|---|---:|-------------:|-----------:|-------:|
| 1 | `bs_pi4` | 4 | 0.970 | 0.1595 | 100 |
| 2 | `bs_pi2` | 4 | 0.930 | 0.1648 | 100 |
| 3 | `identity` | 4 | 0.910 | 0.2622 | 100 |
| 4 | `snap_b_pi` | 4 | 0.910 | 0.2553 | 100 |
| 5 | `cz_nm` | 4 | 0.900 | 0.1566 | 100 |
| 6 | `snap_a_pi` | 4 | 0.890 | 0.2835 | 100 |
| 7 | `bs_pi2` | 3 | 0.820 | 0.1710 | 100 |
| 8 | `snap_b_pi` | 3 | 0.780 | 0.1992 | 100 |
| 9 | `bs_pi4` | 3 | 0.780 | 0.1326 | 100 |
| 10 | `cz_nm` | 3 | 0.740 | 0.1633 | 100 |
| 11 | `snap_a_pi` | 3 | 0.730 | 0.1926 | 100 |
| 12 | `identity` | 3 | 0.690 | 0.1940 | 100 |
| 13 | `bs_pi2` | 2 | 0.660 | 0.1460 | 100 |
| 14 | `bs_pi4` | 2 | 0.620 | 0.1274 | 100 |
| 15 | `cz_nm` | 2 | 0.500 | 0.1320 | 100 |
| 16 | `snap_a_pi` | 2 | 0.450 | 0.1489 | 100 |
| 17 | `identity` | 2 | 0.450 | 0.1464 | 100 |
| 18 | `snap_b_pi` | 2 | 0.430 | 0.1363 | 100 |

**Best so far:** `bs_pi4` at L*=4 (success=0.970, mean p(GS)=0.1595).

Note: `identity` / SNAP at L*=4 have higher mean p(GS) but slightly lower success rate than `bs_pi4`.

Artifacts:
- `noiseless/results/fleet1_20260917T100720Z.json`
- `noiseless/results/fleet1_20260917T100720Z_summary.json`
