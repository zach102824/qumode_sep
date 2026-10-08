# Noiseless SPSA ranking

Updated: 2026-09-29 12:20 Asia/Shanghai

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


## ECD vs QAOA bake-off (matched param count, 200 steps)

Tag `fleet_ecd_vs_qaoa`: three arms × param tiers {16,24,32} × 25 trials × 200 SPSA steps × 20 Hamiltonians
(N=500 per cell; 4500 jobs). Random init; Gibbs SPSA; no prep.

- **ECD**: local ECD ‖ + fixed `bs_pi4`, L* ∈ {2,3,4}
- **QAOA-full**: 8-qubit QAOA, p ∈ {8,12,16}, H_P = full 4-SAT diagonal; H_M = Σ X_j; start |+⟩^⊗8
- **QAOA-NN**: same circuit; H_P = 1-local Z + ring-NN ZZ only (Gibbs on NN spectrum); score p(GS)/success on **true full** GS

### By param tier (success, then mean p(GS))

#### 16 parameters

| Rank | Arm | Depth | Success | Mean p(GS) | N |
|-----:|-----|------:|--------:|-----------:|--:|
| 1 | QAOA-full | p=8 | 0.708 | 0.0618 | 500 |
| 2 | ECD (`bs_pi4`) | L*=2 | 0.624 | 0.1245 | 500 |
| 3 | QAOA-NN | p=8 | 0.000 | 0.0017 | 500 |

**Winner @ 16:** `qaoa_full` (succ=0.708, mean p(GS)=0.0618).

#### 24 parameters

| Rank | Arm | Depth | Success | Mean p(GS) | N |
|-----:|-----|------:|--------:|-----------:|--:|
| 1 | ECD (`bs_pi4`) | L*=3 | 0.866 | 0.1489 | 500 |
| 2 | QAOA-full | p=12 | 0.554 | 0.0526 | 500 |
| 3 | QAOA-NN | p=12 | 0.000 | 0.0022 | 500 |

**Winner @ 24:** `ecd` (succ=0.866, mean p(GS)=0.1489).

#### 32 parameters

| Rank | Arm | Depth | Success | Mean p(GS) | N |
|-----:|-----|------:|--------:|-----------:|--:|
| 1 | ECD (`bs_pi4`) | L*=4 | 0.954 | 0.1554 | 500 |
| 2 | QAOA-full | p=16 | 0.452 | 0.0413 | 500 |
| 3 | QAOA-NN | p=16 | 0.000 | 0.0022 | 500 |

**Winner @ 32:** `ecd` (succ=0.954, mean p(GS)=0.1554).

### Best-of-25 mean p(GS) (per-H max, then mean over H)

| Tier | Arm | Mean p(GS) | Mean best-of-25 p(GS) | Mean top-3 p(GS) |
|-----:|-----|-----------:|----------------------:|-----------------:|
| 16 | ECD (`bs_pi4`) | 0.1245 | 0.2206 | 0.1986 |
| 16 | QAOA-full | 0.0618 | 0.1546 | 0.1340 |
| 16 | QAOA-NN | 0.0017 | 0.0100 | 0.0066 |
| 24 | ECD (`bs_pi4`) | 0.1489 | 0.2886 | 0.2532 |
| 24 | QAOA-full | 0.0526 | 0.1520 | 0.1275 |
| 24 | QAOA-NN | 0.0022 | 0.0131 | 0.0082 |
| 32 | ECD (`bs_pi4`) | 0.1554 | 0.2885 | 0.2493 |
| 32 | QAOA-full | 0.0413 | 0.1281 | 0.1060 |
| 32 | QAOA-NN | 0.0022 | 0.0143 | 0.0086 |

### NN truncation stats (Pauli Z expansion → ring NN)

Across 20 Hamiltonians: mean fraction of terms dropped **0.764** (range 0.681–0.837); mean fraction of |coeff| L1 dropped **0.711**. Kept: weight-1 Z + ring-NN ZZ; dropped: non-NN ZZ + weight≥3.

**Headline:** QAOA-full wins only at 16 params; ECD (`bs_pi4`) wins at 24 and 32. QAOA-NN never recovers the true GS (success 0.000 at all tiers) under heavy truncation.

**Figure:** single-panel success-rate comparison at fixed 200 SPSA steps — [`figures/ecd_vs_qaoa_200_success.png`](figures/ecd_vs_qaoa_200_success.png) (ECD improves with L*; QAOA-full degrades with p).

### jp vs separable baselines: removed (2026-10-08)

The uncontrolled jp-vs-identity/HEA comparison that lived here (sources `jp_L2_20260929T041311Z`,
`jp_L3_20260929T041643Z`, `jp_B0_20260928T042134Z`, `JP_GATE_SUMMARY.md`) was removed: arms differed
in seeds/inits and budgets, so the mixed result (jp ahead on success, identity ahead on p(GS)) was not
interpretable. The controlled replacement is `noiseless/run_entanglement_control.py` (paired seeds across
arms, γ dose–response sweep, product-gate control, per-layer entanglement logging). Old data is in git history.


## QAOA-full @ 800 SPSA steps (flat budget)

Tag `fleet_qaoa_full_800`: QAOA-full only, p ∈ {8,12,16} (16/24/32 params),
25 trials × **800** SPSA steps × 20 Hamiltonians (N=500 per tier; 1500 jobs).
Same hyperparameters as `fleet_ecd_vs_qaoa` except steps (Gibbs + SampledTailEta,
Uniform[0,π) init, `spsa_a = 0.2·√(37/n)`). ECD and QAOA-NN not re-run.

### vs QAOA-full@200 and ECD@200 at matched param tiers

| Tier | Depth | QAOA-full@800 succ | QAOA@800 mean p(GS) | QAOA-full@200 succ | QAOA@200 mean p(GS) | ECD@200 succ | ECD@200 mean p(GS) |
|-----:|------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------:|-------------------:|
| 16 | p=8 / L*=2 | 1.000 | 0.1549 | 0.708 | 0.0618 | 0.624 | 0.1245 |
| 24 | p=12 / L*=3 | 1.000 | 0.2022 | 0.554 | 0.0526 | 0.866 | 0.1489 |
| 32 | p=16 / L*=4 | 0.998 | 0.2057 | 0.452 | 0.0413 | 0.954 | 0.1554 |

### Best-of-25 mean p(GS) (QAOA-full @ 800)

| Tier | Depth | Mean p(GS) | Mean best-of-25 p(GS) | Mean top-3 p(GS) |
|-----:|------:|-----------:|----------------------:|-----------------:|
| 16 | p=8 | 0.1549 | 0.2665 | 0.2411 |
| 24 | p=12 | 0.2022 | 0.3094 | 0.2890 |
| 32 | p=16 | 0.2057 | 0.3288 | 0.2940 |

**Depth trend @ 800 steps:** mean p(GS) **increases** with depth (0.1549 → 0.2022 → 0.2057), reversing the *decrease* seen at 200 steps (0.0618 → 0.0526 → 0.0413). Success saturates (~1.000 / 1.000 / 0.998).

**vs ECD@200:** QAOA-full@800 beats ECD on success at every tier; beats ECD on mean p(GS) at 24 and 32 params (and at 16: 0.1549 vs ECD 0.1245).

## Notes

- `bs_pi4` leads bitstring success at L*=4–5; identity/SNAP lead mean p(GS) at the same depths.
- Among θ∈{π/6,π/4,π/3} at L*=4/200 steps, `bs_pi4` remains best (success-tied with π/6); best-of-25 lifts π/4 mean p(GS) by ~1.8×.
- Identity is kept as a no-mixing baseline only.
- Flat 800 SPSA steps flip the QAOA-full depth trend: deeper p now helps; QAOA-full@800 saturates success and surpasses ECD@200 mean p(GS) at 24/32.

## jp: Gray encoding and layer growth (seed 20260917, Gibbs-only SPSA)

20 four_sat × 25 trials per cell. A = binary random init (existing); B = Gray encoding, random init, 200 steps;
C/D = layer growth from L=1 (transparent appended last layer β=0, θ=π, φ=0 + σ=0.05 kick) with **200 SPSA steps per
stage** (total 400/600/800 for L=2/3/4), binary / Gray. Full details: `ENCODING_GROW_SUMMARY.md`.

| L | A@200 succ / p̄ | A@400 | A@800 | B gray@200 | C grow (binary) | D grow (gray) |
|---|---|---|---|---|---|---|
| 2 | 0.456 / 0.130 | 0.486 / 0.149 | 0.462 / 0.162 | **0.538** / 0.143 | 0.486 / **0.205** (400) | 0.438 / 0.201 (400) |
| 3 | 0.774 / 0.158 | 0.804 / 0.190 | **0.820** / 0.221 | 0.762 / 0.169 | 0.602 / **0.305** (600) | 0.564 / 0.288 (600) |
| 4 | 0.932 / 0.172 | 0.946 / 0.213 | **0.950** / 0.256 | 0.908 / 0.178 | 0.752 / **0.382** (800) | 0.762 / 0.375 (800) |

- Growth gives +40–50% mean p(GS) over random init at matched total steps (L4@800: 0.382 vs 0.256; frac p(GS)>0.5 0.354 vs
  0.072) but loses success at L≥3: failures lock onto the first-excited level. Gray alone is a wash (L2 success +0.08,
  L3/L4 −0.01/−0.02).

## jp: SPSA-Adam (same SPSA gradient + eval count, Adam update lr 0.05; seed 20260917)

20 four_sat × 25 trials, binary, λ=0 Gibbs. A = random init 200 steps (401 evals); C = growth from L=1, 200 steps per
stage (802/1203/1604 evals); fresh Adam state per stage. Full details: `ADAM_SUMMARY.md`.

| L | A SPSA succ / p̄ | A Adam | C SPSA grow | C Adam grow |
|---|---|---|---|---|
| 2 | 0.456 / 0.130 | 0.462 / 0.138 | 0.486 / 0.205 | **0.510** / **0.222** |
| 3 | 0.774 / 0.158 | 0.760 / 0.184 | 0.602 / 0.305 | **0.772** / **0.341** |
| 4 | **0.932** / 0.172 | 0.930 / 0.208 | 0.752 / 0.382 | 0.852 / **0.397** |

- Same cost, free win: Adam keeps random-init success and adds +0.03–0.04 mean p(GS) at L3/L4; with growth it lifts
  success at L3 to random-init level (0.772) and at L4 to 0.852 while keeping growth's p(GS) edge. Remaining growth
  failures still lock onto the first-excited level (97%).

## jp: grow + SPSA-Adam hyper-parameter tuning (final L=4, 800 SPSA steps = 1604 evals; seed 20260917)

Screen: four_sat_000..004 × 10 trials, one knob at a time (Adam lr / per-stage lr schedule, step split, kick σ, η scale
schedule, SPSA c, start depth), then combos; full: 20 four_sat × 25 trials. Full details: `GROW_ADAM_TUNING_SUMMARY.md`.

| arm (L=4, 1604 evals) | success | mean p(GS) | median | frac>0.5 |
|---|---|---|---|---|
| C_adam (lr 0.05, 200×4) | 0.852 | 0.397 | 0.396 | 0.288 |
| lr schedule 0.5,0.2,0.05,0.02 | 0.978 | 0.542 | 0.518 | 0.594 |
| + η 0.5,0.7,1.5,2 + c 0.1 (screen combo) | 0.974 | 0.537 | 0.518 | 0.568 |
| **+ steps 100/150/250/300** | **0.982** | **0.547** | **0.519** | **0.608** |

- Decaying per-stage Adam lr (big early, small late) is the whole effect (+0.145 paired p(GS), 69 vs 6 success flips);
  split / η / c are within noise on top, large kicks are redundant with it. Tuned growth now beats random-init L4
  success (0.93). Remaining failures are first-excited near-ties with p(GS) ≈ 0.24 (not collapsed).
