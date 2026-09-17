# Noiseless SPSA ranking

Updated: 2026-09-17 18:51 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless.

## Headline result

**Best overall:** `snap_b_pi` at **L\*=16**
— success rate **1.000**, mean p(GS) **0.9174**
(fleet7, 600 trials × 1200 SPSA steps, 20 Hamiltonians).

Deeper ECD stacks with **SNAP / identity** buses raise mean ground probability at perfect/near-perfect bitstring success.

## Fleet7 (U=identity,snap_a_pi,snap_b_pi, L∈{10,12,14,16}, 30×1200)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `snap_b_pi` | 16 | 1.000 | 0.9174 | 600 |
| 2 | `identity` | 16 | 1.000 | 0.9172 | 600 |
| 3 | `snap_a_pi` | 16 | 1.000 | 0.9166 | 600 |
| 4 | `snap_a_pi` | 14 | 1.000 | 0.8903 | 600 |
| 5 | `snap_b_pi` | 14 | 1.000 | 0.8881 | 600 |
| 6 | `identity` | 14 | 1.000 | 0.8868 | 600 |
| 7 | `snap_a_pi` | 12 | 1.000 | 0.8454 | 600 |
| 8 | `snap_b_pi` | 12 | 1.000 | 0.8445 | 600 |
| 9 | `identity` | 12 | 1.000 | 0.8428 | 600 |
| 10 | `identity` | 10 | 1.000 | 0.7896 | 600 |
| 11 | `snap_b_pi` | 10 | 1.000 | 0.7878 | 600 |
| 12 | `snap_a_pi` | 10 | 1.000 | 0.7861 | 600 |

## Prior fleets (selected)

### Fleet6
- `snap_a_pi` L=12: succ=1.000, mean_p=0.8454
- `identity` L=12: succ=1.000, mean_p=0.8428
- `identity` L=10: succ=1.000, mean_p=0.7896
- `snap_a_pi` L=10: succ=1.000, mean_p=0.7861
- `identity` L=9: succ=1.000, mean_p=0.7569
- `snap_a_pi` L=9: succ=1.000, mean_p=0.7521

### Fleet5
- `identity` L=9: succ=1.000, mean_p=0.7354
- `snap_a_pi` L=9: succ=1.000, mean_p=0.7304
- `snap_b_pi` L=9: succ=1.000, mean_p=0.7289
- `snap_a_pi` L=8: succ=1.000, mean_p=0.6916
- `identity` L=8: succ=1.000, mean_p=0.6868
- `snap_b_pi` L=8: succ=1.000, mean_p=0.6850

### Fleet4_p
- `identity` L=7: succ=1.000, mean_p=0.6093
- `snap_a_pi` L=7: succ=1.000, mean_p=0.6093
- `snap_b_pi` L=7: succ=0.998, mean_p=0.6017
- `identity` L=6: succ=0.998, mean_p=0.5456

### Fleet4_bs (`bs_pi4`)
- L=6: succ=1.000, mean_p=0.2196
- L=8: succ=1.000, mean_p=0.2124
- L=7: succ=0.999, mean_p=0.2142
- L=5: succ=0.991, mean_p=0.2294
