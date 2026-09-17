# Noiseless SPSA ranking

Updated: 2026-09-17 14:04 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless.

## Headline result

**Best overall:** `identity` at **L\*=9**
— success rate **1.000**, mean p(GS) **0.7354**
(fleet5, 800 trials × 1000 SPSA steps, 20 Hamiltonians).

Deeper ECD stacks with an **identity** (or SNAP) bus dominate beamsplitter U on mean ground probability while keeping perfect/near-perfect bitstring success.

## Fleet5 (identity/SNAP, L∈{6,7,8,9}, 40×1000)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `identity` | 9 | 1.000 | 0.7354 | 800 |
| 2 | `snap_a_pi` | 9 | 1.000 | 0.7304 | 800 |
| 3 | `snap_b_pi` | 9 | 1.000 | 0.7289 | 800 |
| 4 | `snap_a_pi` | 8 | 1.000 | 0.6916 | 800 |
| 5 | `identity` | 8 | 1.000 | 0.6868 | 800 |
| 6 | `snap_b_pi` | 8 | 1.000 | 0.6850 | 800 |
| 7 | `snap_a_pi` | 7 | 1.000 | 0.6362 | 800 |
| 8 | `identity` | 7 | 1.000 | 0.6356 | 800 |
| 9 | `snap_b_pi` | 7 | 0.998 | 0.6284 | 800 |
| 10 | `snap_b_pi` | 6 | 0.996 | 0.5610 | 800 |
| 11 | `identity` | 6 | 0.995 | 0.5696 | 800 |
| 12 | `snap_a_pi` | 6 | 0.995 | 0.5586 | 800 |

## Prior fleets (selected)

### Fleet4_p
- `identity` L=7: succ=1.000, mean_p=0.6093
- `snap_a_pi` L=7: succ=1.000, mean_p=0.6093
- `snap_b_pi` L=7: succ=0.998, mean_p=0.6017
- `identity` L=6: succ=0.998, mean_p=0.5456
- `snap_b_pi` L=6: succ=0.998, mean_p=0.5390
- `snap_a_pi` L=6: succ=0.993, mean_p=0.5351

### Fleet4_bs (`bs_pi4`)
- L=6: succ=1.000, mean_p=0.2196
- L=8: succ=1.000, mean_p=0.2124
- L=7: succ=0.999, mean_p=0.2142
- L=5: succ=0.991, mean_p=0.2294
