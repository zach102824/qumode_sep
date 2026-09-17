# Noiseless SPSA ranking

Updated: 2026-09-17 10:55 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless dim-256.

## Headline

- **Best success rate:** `bs_pi4` at **L\*=6** — success=1.000, mean p(GS)=0.2062 (fleet3, 500 trials × 600 SPSA steps).

### Fleet3 full table (top U only, L∈{4,5,6}, 25 trials × 600 steps)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `bs_pi4` | 6 | 1.000 | 0.2062 | 500 |
| 2 | `bs_pi2` | 6 | 0.998 | 0.2204 | 500 |
| 3 | `identity` | 6 | 0.994 | 0.5160 | 500 |
| 4 | `cz_nm` | 6 | 0.990 | 0.2370 | 500 |
| 5 | `identity` | 5 | 0.988 | 0.4321 | 500 |
| 6 | `bs_pi4` | 5 | 0.988 | 0.2177 | 500 |
| 7 | `cz_nm` | 5 | 0.980 | 0.2413 | 500 |
| 8 | `bs_pi2` | 5 | 0.978 | 0.2191 | 500 |
| 9 | `bs_pi4` | 4 | 0.968 | 0.2054 | 500 |
| 10 | `bs_pi2` | 4 | 0.952 | 0.2164 | 500 |
| 11 | `cz_nm` | 4 | 0.938 | 0.2246 | 500 |
| 12 | `identity` | 4 | 0.916 | 0.3351 | 500 |

### High mean-p(GS) (fleet2, includes SNAP)

- `snap_a_pi` L*=5: mean p=0.4035, success=0.977
- `snap_b_pi` L*=5: mean p=0.3995, success=0.967
- `identity` L*=5: mean p=0.3937, success=0.983
- `snap_a_pi` L*=4: mean p=0.3122, success=0.897
- `snap_b_pi` L*=4: mean p=0.3120, success=0.923
- `identity` L*=4: mean p=0.3072, success=0.923

## Fleet history

- Fleet1 breadth: `fleet1_*_summary.json`
- Fleet2 deeper L=3..5: `fleet2_20260917T102419Z_summary.json`
- Fleet3 winners L=4..6: `fleet3_20260917T105442Z_summary.json`
