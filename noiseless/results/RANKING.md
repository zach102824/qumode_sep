# Noiseless SPSA ranking

Updated: 2026-09-17 11:24 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless.

## Headline

- **Best bitstring success:** `bs_pi4` at **L*=6** — success=1.000, mean p(GS)=0.2196 (fleet4_bs, 800 trials × 800 steps).
- **Best mean p(GS) in fleet3:** `identity` L*=6 — mean p=0.5160, success=0.994.

## Fleet4_bs (`bs_pi4` depth scan)

| Rank | U | L* | Success | Mean p(GS) | N |
|-----:|---|---:|--------:|-----------:|--:|
| 1 | `bs_pi4` | 6 | 1.000 | 0.2196 | 800 |
| 2 | `bs_pi4` | 8 | 1.000 | 0.2124 | 800 |
| 3 | `bs_pi4` | 7 | 0.999 | 0.2142 | 800 |
| 4 | `bs_pi4` | 5 | 0.991 | 0.2294 | 800 |

## Fleet3 (winners L∈{4,5,6})

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

