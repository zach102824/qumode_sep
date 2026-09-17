# Noiseless SPSA ranking

Updated: 2026-09-17 15:39 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless.

## Headline result

**Best overall:** `snap_a_pi` at **L\*=12**
— success rate **1.000**, mean p(GS) **0.8454**
(fleet6, 600 trials × 1200 SPSA steps, 20 Hamiltonians).

Deeper ECD stacks with **SNAP / identity** buses continue to raise mean ground probability at perfect bitstring success; `snap_a_pi` now leads at L*=12.

## Fleet6 (identity/snap_a_pi, L∈{8,9,10,12}, 30×1200)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `snap_a_pi` | 12 | 1.000 | 0.8454 | 600 |
| 2 | `identity` | 12 | 1.000 | 0.8428 | 600 |
| 3 | `identity` | 10 | 1.000 | 0.7896 | 600 |
| 4 | `snap_a_pi` | 10 | 1.000 | 0.7861 | 600 |
| 5 | `identity` | 9 | 1.000 | 0.7569 | 600 |
| 6 | `snap_a_pi` | 9 | 1.000 | 0.7521 | 600 |
| 7 | `snap_a_pi` | 8 | 1.000 | 0.7121 | 600 |
| 8 | `identity` | 8 | 1.000 | 0.7056 | 600 |

## Prior fleets (selected)

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

### Fleet4_bs (`bs_pi4`)
- L=6: succ=1.000, mean_p=0.2196
- L=8: succ=1.000, mean_p=0.2124
- L=7: succ=0.999, mean_p=0.2142
- L=5: succ=0.991, mean_p=0.2294
