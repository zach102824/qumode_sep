# Noiseless SPSA ranking

Updated: 2026-09-18 04:16 UTC

Constraints: random ECD init; fixed U never trained; no joint prep; noiseless.

**Scope:** only L* ≤ 5. Deeper fleets (L*≥6, including L*~22) were removed from the repo.

## Headline (best among L*≤5 by success, then mean p(GS))

**Best overall:** `bs_pi4` at **L\*=5**
— success rate **0.991**, mean p(GS) **0.2294**
(source `fleet4_bs`, N=800).

## Full L*≤5 table

| Rank | U | L* | Success | Mean p(GS) | N | Source |
|-----:|---|---:|--------:|-----------:|--:|--------|
| 1 | `bs_pi4` | 5 | 0.991 | 0.2294 | 800 | `fleet4_bs` |
| 2 | `bs_pi4` | 5 | 0.990 | 0.1950 | 300 | `fleet2` |
| 3 | `identity` | 5 | 0.988 | 0.4547 | 600 | `fleet4_p` |
| 4 | `identity` | 5 | 0.988 | 0.4321 | 500 | `fleet3` |
| 5 | `bs_pi4` | 5 | 0.988 | 0.2177 | 500 | `fleet3` |
| 6 | `identity` | 5 | 0.983 | 0.3937 | 300 | `fleet2` |
| 7 | `bs_pi2` | 5 | 0.983 | 0.2014 | 300 | `fleet2` |
| 8 | `snap_a_pi` | 5 | 0.982 | 0.4590 | 600 | `fleet4_p` |
| 9 | `cz_nm` | 5 | 0.980 | 0.2413 | 500 | `fleet3` |
| 10 | `cz_nm` | 5 | 0.980 | 0.2171 | 300 | `fleet2` |
| 11 | `bs_pi2` | 5 | 0.978 | 0.2191 | 500 | `fleet3` |
| 12 | `snap_b_pi` | 5 | 0.977 | 0.4603 | 600 | `fleet4_p` |
| 13 | `snap_a_pi` | 5 | 0.977 | 0.4035 | 300 | `fleet2` |
| 14 | `bs_pi4` | 4 | 0.977 | 0.1862 | 300 | `fleet2` |
| 15 | `bs_pi4` | 4 | 0.970 | 0.1595 | 100 | `fleet1` |
| 16 | `bs_pi4` | 4 | 0.968 | 0.2054 | 500 | `fleet3` |
| 17 | `snap_b_pi` | 5 | 0.967 | 0.3995 | 300 | `fleet2` |
| 18 | `bs_pi2` | 4 | 0.952 | 0.2164 | 500 | `fleet3` |
| 19 | `bs_pi2` | 4 | 0.950 | 0.2000 | 300 | `fleet2` |
| 20 | `cz_nm` | 4 | 0.938 | 0.2246 | 500 | `fleet3` |
| 21 | `cz_nm` | 4 | 0.937 | 0.2012 | 300 | `fleet2` |
| 22 | `bs_pi2` | 4 | 0.930 | 0.1648 | 100 | `fleet1` |
| 23 | `snap_b_pi` | 4 | 0.923 | 0.3120 | 300 | `fleet2` |
| 24 | `identity` | 4 | 0.923 | 0.3072 | 300 | `fleet2` |
| 25 | `identity` | 4 | 0.916 | 0.3351 | 500 | `fleet3` |
| 26 | `identity` | 4 | 0.910 | 0.2622 | 100 | `fleet1` |
| 27 | `snap_b_pi` | 4 | 0.910 | 0.2553 | 100 | `fleet1` |
| 28 | `cz_nm` | 4 | 0.900 | 0.1566 | 100 | `fleet1` |
| 29 | `snap_a_pi` | 4 | 0.897 | 0.3122 | 300 | `fleet2` |
| 30 | `snap_a_pi` | 4 | 0.890 | 0.2835 | 100 | `fleet1` |
| 31 | `bs_pi2` | 3 | 0.843 | 0.1839 | 300 | `fleet2` |
| 32 | `bs_pi4` | 3 | 0.827 | 0.1589 | 300 | `fleet2` |
| 33 | `bs_pi2` | 3 | 0.820 | 0.1710 | 100 | `fleet1` |
| 34 | `snap_b_pi` | 3 | 0.780 | 0.1992 | 100 | `fleet1` |
| 35 | `bs_pi4` | 3 | 0.780 | 0.1326 | 100 | `fleet1` |
| 36 | `cz_nm` | 3 | 0.763 | 0.1837 | 300 | `fleet2` |
| 37 | `snap_b_pi` | 3 | 0.757 | 0.2324 | 300 | `fleet2` |
| 38 | `snap_a_pi` | 3 | 0.747 | 0.2285 | 300 | `fleet2` |
| 39 | `cz_nm` | 3 | 0.740 | 0.1633 | 100 | `fleet1` |
| 40 | `snap_a_pi` | 3 | 0.730 | 0.1926 | 100 | `fleet1` |
| 41 | `identity` | 3 | 0.720 | 0.2241 | 300 | `fleet2` |
| 42 | `identity` | 3 | 0.690 | 0.1940 | 100 | `fleet1` |
| 43 | `bs_pi2` | 2 | 0.660 | 0.1460 | 100 | `fleet1` |
| 44 | `bs_pi4` | 2 | 0.620 | 0.1274 | 100 | `fleet1` |
| 45 | `cz_nm` | 2 | 0.500 | 0.1320 | 100 | `fleet1` |
| 46 | `snap_a_pi` | 2 | 0.450 | 0.1489 | 100 | `fleet1` |
| 47 | `identity` | 2 | 0.450 | 0.1464 | 100 | `fleet1` |
| 48 | `snap_b_pi` | 2 | 0.430 | 0.1363 | 100 | `fleet1` |


## Phase bake-off (200 steps)

Tag `fleet_phase_bakeoff`: U ∈ {`cz_nm`, `ck_pi2`, `ck_pi4`, `cphase_nn`, `bs_pi4`}, L* ∈ {3,4},
25 trials × 200 SPSA steps × 20 Hamiltonians (N=500 per cell); random ECD init; no prep SPSA.

### Phase gates ranked by L*=4 success, then mean p(GS)

| Rank | U | L*=4 Success | L*=4 Mean p(GS) | L*=3 Success | L*=3 Mean p(GS) |
|-----:|---|-------------:|----------------:|-------------:|----------------:|
| 1 | `ck_pi2` | 0.958 | 0.1529 | 0.820 | 0.1473 |
| 2 | `cphase_nn` | 0.950 | 0.1745 | 0.784 | 0.1524 |
| 3 | `ck_pi4` | 0.940 | 0.1620 | 0.774 | 0.1525 |
| 4 | `cz_nm` | 0.922 | 0.1663 | 0.760 | 0.1591 |

### `bs_pi4` reference (same settings)

| U | L* | Success | Mean p(GS) | N |
|---|---:|--------:|-----------:|--:|
| `bs_pi4` | 4 | 0.958 | 0.1563 | 500 |
| `bs_pi4` | 3 | 0.828 | 0.1391 | 500 |

**Phase winner:** `ck_pi2` at L*=4 (success **0.958**, mean p(GS) **0.1529**). Gap to `bs_pi4` under identical settings: success Δ=+0.000, mean p(GS) Δ=+0.0034.


## θ bake-off — beamsplitter angle (200 steps, L*=4)

Tag `fleet_bs_theta`: U ∈ {`bs_pi6` (θ=π/6), `bs_pi4` (θ=π/4), `bs_pi3` (θ=π/3)}, L*=4 only,
25 trials × 200 SPSA steps × 20 Hamiltonians (N=500 per θ); random ECD init; Gibbs SPSA; no prep.

### θ ranked by success, then mean p(GS)

| Rank | U (θ) | Success | Mean p(GS) | Mean of per-H best-of-25 p(GS) | Mean of per-H top-3 p(GS) | N |
|-----:|-------|--------:|-----------:|-------------------------------:|--------------------------:|--:|
| 1 | `bs_pi4` (π/4) | 0.958 | 0.1563 | 0.2785 | 0.2540 | 500 |
| 2 | `bs_pi6` (π/6) | 0.958 | 0.1552 | 0.2941 | 0.2517 | 500 |
| 3 | `bs_pi3` (π/3) | 0.956 | 0.1543 | 0.2860 | 0.2543 | 500 |

**θ winner:** `bs_pi4` (tied success with `bs_pi6`, higher mean p(GS) by Δ=+0.0011).

**Best-of-N vs mean for `bs_pi4`:** mean p(GS) **0.1563** → mean of per-H best-of-25 **0.2785**
(lift **+0.1222**, **1.78×**). Mean of per-H top-3 = **0.2540** (1.62× vs mean).

## Notes

- `bs_pi4` leads bitstring success at L*=4–5; identity/SNAP lead mean p(GS) at the same depths.
- Among θ∈{π/6,π/4,π/3} at L*=4/200 steps, `bs_pi4` remains best (success-tied with π/6); best-of-25 lifts π/4 mean p(GS) by ~1.8×.
- Identity is kept as a no-mixing baseline only.
