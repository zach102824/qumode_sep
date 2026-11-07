# LN-LA smoke — load-normalized graded H_c × load-adaptive cavity SNAP

Opus Idea 2. Label: **graded H_c** (positive Walsh weights from cavity load).
Angles γ, c1–c3, p stay frozen. The grid only moves α and β.

```
ρ_j = clause incidences on the 3 bits of cavity j
r_j = ρ_j / mean_j ρ_j
γ_S = γ_ℓ · Π_{j in cav(S)} r_j^(−α / |cav(S)|)
φ_k^(j) = r_j^β · [c1 k + a_n (c2 k² + c3 k³)]
a_n = (8/(n−1))^p
α = β = 0  →  frozen power_p champ
```

Train grid: fresh250 n∈{6,9}, first B=100, score (Med6 + 2 Med9) / 3, grid 9×9 on [−1, 1].
Decisive CORE: fresh packs base 73000000000, n∈{6,9,12,15}, B=1000. Not the train split.

## Frozen champs

| L | checkpoint | p | note |
|---|---|---|---|
| 2 | `Nov_3rd/runs/beat_qaoa_12h/train69/power_p/pfix0.15/params_L2_seed0.pt` | 0.1500 | no fresh250 L=2 (ladder is 1,4,…,32); best-holdout power_p on the older train{6,9} packs, p fixed at 0.15, weights w6=0.25 w9=1 |
| 16 | `Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/power_p/params_L16_seed0.pt` | 0.0729 | fresh250 power_p seed0, free p, weights w6=1 w9=2 |

## Train grid

### L=2

Champ cell α=β=0: score 9.6195 (Med6 5.5795, Med9 11.6395).
Best free cell: α=-0.25, β=1.00, score 9.5705, train gain 0.510%.

Weighted train score (lower is better). `*` marks the selected cell.

| β \\ α | -1.00 | -0.75 | -0.50 | -0.25 | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 9.6575 | 9.6252 | 9.5962 | 9.5705* | 9.5731 | 9.5779 | 9.5838 | 9.5906 | 9.6175 |
| 0.75 | 9.6586 | 9.6326 | 9.6110 | 9.5935 | 9.5937 | 9.5982 | 9.6037 | 9.6103 | 9.6193 |
| 0.50 | 9.6576 | 9.6319 | 9.6104 | 9.5932 | 9.6008 | 9.6185 | 9.6252 | 9.6314 | 9.6390 |
| 0.25 | 9.6621 | 9.6366 | 9.6154 | 9.5983 | 9.6020 | 9.6212 | 9.6421 | 9.6540 | 9.6637 |
| 0.00 | 9.6718 | 9.6466 | 9.6256 | 9.6088 | 9.6195 | 9.6500 | 9.6835 | 9.7084 | 9.7142 |
| -0.25 | 9.6867 | 9.6617 | 9.6436 | 9.6637 | 9.6933 | 9.7260 | 9.7409 | 9.7458 | 9.7530 |
| -0.50 | 9.7213 | 9.7312 | 9.7183 | 9.7210 | 9.7372 | 9.7554 | 9.7755 | 9.7874 | 9.7960 |
| -0.75 | 9.7278 | 9.7383 | 9.7201 | 9.7200 | 9.7379 | 9.7578 | 9.7795 | 9.8030 | 9.8283 |
| -1.00 | 9.7078 | 9.7199 | 9.7251 | 9.7228 | 9.7424 | 9.7640 | 9.7875 | 9.8045 | 9.8209 |

Heatmap: `grid_L2.svg`.

Tied lines β = κα (best cell on that line):

| κ | α | β | score | train gain vs champ |
|---|---|---|---|---|
| -1.0 | -0.25 | 0.25 | 9.5983 | 0.221% |
| -0.5 | -0.50 | 0.25 | 9.6154 | 0.043% |
| 0.0 | -0.25 | 0.00 | 9.6088 | 0.112% |
| 0.5 | 0.00 | 0.00 | 9.6195 | 0.000% |
| 1.0 | 0.75 | 0.75 | 9.6103 | 0.096% |

Best nonzero-κ tie: α=-0.25, β=0.25, train gain 0.221%.

### L=16

Champ cell α=β=0: score 1.3442 (Med6 1.1623, Med9 1.4351).
Best free cell: α=1.00, β=0.00, score 1.3025, train gain 3.099%.

Weighted train score (lower is better). `*` marks the selected cell.

| β \\ α | -1.00 | -0.75 | -0.50 | -0.25 | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.3467 | 1.3429 | 1.3416 | 1.3344 | 1.3307 | 1.3199 | 1.3141 | 1.3185 | 1.3280 |
| 0.75 | 1.3498 | 1.3477 | 1.3475 | 1.3445 | 1.3356 | 1.3325 | 1.3079 | 1.3118 | 1.3178 |
| 0.50 | 1.3575 | 1.3432 | 1.3489 | 1.3507 | 1.3397 | 1.3347 | 1.3227 | 1.3082 | 1.3098 |
| 0.25 | 1.3557 | 1.3491 | 1.3443 | 1.3485 | 1.3430 | 1.3279 | 1.3249 | 1.3089 | 1.3081 |
| 0.00 | 1.3480 | 1.3557 | 1.3408 | 1.3435 | 1.3442 | 1.3333 | 1.3191 | 1.3155 | 1.3025* |
| -0.25 | 1.3462 | 1.3499 | 1.3487 | 1.3405 | 1.3466 | 1.3425 | 1.3266 | 1.3114 | 1.3095 |
| -0.50 | 1.3494 | 1.3467 | 1.3485 | 1.3390 | 1.3423 | 1.3473 | 1.3391 | 1.3207 | 1.3105 |
| -0.75 | 1.3463 | 1.3503 | 1.3458 | 1.3443 | 1.3387 | 1.3443 | 1.3445 | 1.3321 | 1.3154 |
| -1.00 | 1.3429 | 1.3489 | 1.3466 | 1.3454 | 1.3479 | 1.3402 | 1.3471 | 1.3419 | 1.3258 |

Heatmap: `grid_L16.svg`.

Tied lines β = κα (best cell on that line):

| κ | α | β | score | train gain vs champ |
|---|---|---|---|---|
| -1.0 | 1.00 | -1.00 | 1.3258 | 1.367% |
| -0.5 | 1.00 | -0.50 | 1.3105 | 2.507% |
| 0.0 | 1.00 | 0.00 | 1.3025 | 3.099% |
| 0.5 | 1.00 | 0.50 | 1.3098 | 2.558% |
| 1.0 | 0.75 | 0.75 | 1.3118 | 2.408% |

Best nonzero-κ tie: α=1.00, β=0.50, train gain 2.558%.

## Fresh CORE

### L=2

| n | LN-LA Med | champ Med | gain | QAOA Med | vs champ | vs QAOA | LN-LA B90 | champ B90 | QAOA B90 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 5.0872 | 5.0845 | -0.053% | 5.3099 | UNDER | BEAT | 9.7505 | 9.7160 | 10.1902 |
| 9 | 12.4388 | 12.4440 | 0.042% | 12.7044 | BEAT | BEAT | 27.9566 | 27.8519 | 28.2343 |
| 12 | 31.0698 | 30.5545 | -1.686% | 31.4913 | UNDER | BEAT | 85.1778 | 83.8034 | 84.1581 |
| 15 | 77.7356 | 75.4220 | -3.068% | 76.7327 | UNDER | UNDER | 230.2659 | 226.9628 | 218.1138 |

Free cell α=-0.25, β=1.00.
**Verdict (free): kill.** n15 Med gain -3.068% < 0.5%

Tied cell α=-0.25, β=0.25.
**Verdict (tied β=κα): kill.** n15 Med gain -1.504% < 0.5%

QAOA column is trained on the same fresh250 packs (paper init, 200 Adam steps, batch 32+32, seed 0) because the ladder has no L=2 QAOA. Historical n6-sealed L=2 QAOA Med on this CORE: n6 5.2926, n9 12.6869, n12 31.3630, n15 76.4037.

### L=16

| n | LN-LA Med | champ Med | gain | QAOA Med | vs champ | vs QAOA | LN-LA B90 | champ B90 | QAOA B90 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.1558 | 1.1563 | 0.040% | 1.2249 | BEAT | BEAT | 1.7241 | 1.7339 | 2.2095 |
| 9 | 1.4125 | 1.4011 | -0.817% | 1.5787 | UNDER | BEAT | 3.4242 | 3.5363 | 4.1449 |
| 12 | 1.8916 | 1.8665 | -1.349% | 2.1285 | UNDER | BEAT | 7.0596 | 7.1214 | 8.4368 |
| 15 | 2.6336 | 2.6126 | -0.801% | 3.0158 | UNDER | BEAT | 13.9585 | 13.4302 | 14.3683 |

Free cell α=1.00, β=0.00.
**Verdict (free): kill.** n15 Med gain -0.801% < 0.5%

Tied cell α=1.00, β=0.50.
**Verdict (tied β=κα): kill.** n15 Med gain -1.463% < 0.5%

QAOA column re-evaluates the frozen fresh250 L=16 angles (`Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/qaoa_angles/L16/angles.json`) on this fresh CORE, not the published scale1000 medians.

## Reading

L=2: selected α=-0.25, β=1.00 (train-score gain 0.510%). The collapse kill does not fire (|α| and |β| are not both under 0.05). Fresh n15 Med gain is -3.068%, which is under 0.5% and is the kill.
L=16: selected α=1.00, β=0.00 (train-score gain 3.099%). The collapse kill does not fire (|α| and |β| are not both under 0.05). Fresh n15 Med gain is -0.801%, which is under 0.5% and is the kill.

L=16's train pick sits on the α=+1 edge with β=0, so that cell changes only the cost weights. Its best tied line (β = 0.5 α) is worse on fresh n15. L=2's train pick sits on the β=+1 edge. Both fresh n15 Medians went up versus the champ, so a wider (α, β) box is not supported.

r_j is a narrow feature: on the train slice the median max(r)−min(r) is about 0.05 at n=6 and 0.10 at n=9. On the fresh CORE it is about 0.05, 0.12, 0.15, 0.18 at n=6, 9, 12, 15. The L=16 train score improved about 3% (lower Med) on B=100 and then reversed on the fresh packs (n9, n12, and n15 Med all worse). That is the failure mode called out for this idea: load can fit the training instances, and n=6 barely spreads r_j.

The frozen champs still have lower Med than same-L QAOA on every fresh CORE n. LN-LA does not improve them. At L=16 the selected cell still beats QAOA and loses to the champ. At L=2 it loses to the champ at n=15 and also loses to QAOA there.

## Verdict

**kill**

Both L=2 and L=16 hit a kill rule on the fresh CORE.

Kill rule (Opus): both |α| and |β| < 0.05 at the selected cell, or n15 Med gain < 0.5% versus the frozen champ on this fresh CORE.
Ship requires clearing that rule and Med < same-L QAOA Med on every n in {6,9,12,15}.

