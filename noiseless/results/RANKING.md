# Noiseless SPSA ranking

Updated: 2026-09-17 12:08 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless 256-dim.

## Headline

- **Overall best (success then mean p):** `identity` @ **L*=7**
  — success=1.000, mean p(GS)=0.6093.
- **Highest mean p(GS):** `identity` @ L*=7
  — mean p=0.6093, success=1.000.

Interpretation: deeper ECD stacks with **identity** bus beat beamsplitter U on mean ground-state probability while matching perfect bitstring success.

## Fleet4_p (identity / SNAP, L∈{5,6,7}, 30×800)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `identity` | 7 | 1.000 | 0.6093 | 600 |
| 2 | `snap_a_pi` | 7 | 1.000 | 0.6093 | 600 |
| 3 | `snap_b_pi` | 7 | 0.998 | 0.6017 | 600 |
| 4 | `identity` | 6 | 0.998 | 0.5456 | 600 |
| 5 | `snap_b_pi` | 6 | 0.998 | 0.5390 | 600 |
| 6 | `snap_a_pi` | 6 | 0.993 | 0.5351 | 600 |
| 7 | `identity` | 5 | 0.988 | 0.4547 | 600 |
| 8 | `snap_a_pi` | 5 | 0.982 | 0.4590 | 600 |
| 9 | `snap_b_pi` | 5 | 0.977 | 0.4603 | 600 |

## Fleet4_bs (`bs_pi4` depth, 40×800)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `bs_pi4` | 6 | 1.000 | 0.2196 | 800 |
| 2 | `bs_pi4` | 8 | 1.000 | 0.2124 | 800 |
| 3 | `bs_pi4` | 7 | 0.999 | 0.2142 | 800 |
| 4 | `bs_pi4` | 5 | 0.991 | 0.2294 | 800 |

## Fleet3 (mixed winners)

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
