# FTREP H: LE replay of FT L=2 holdout 10144–10151

Same alloc.py `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / recipe `347ac23369db`.
FT holdout table from `bd9effe3fecc`. Seeds 10144–10151 intentional replay.
N=15 L=2 Q=2048 Gate R. N=18 optional report. Energy selects. F post hoc. GR not patched. No candidacy revival.

Runs 32 ok 32 failures 0. Charged 65536. Outside cap 0. Rejected B* 0. Missing 0.

Gate R N15 mean ΔE -0.03109 med +0.00279 [-0.08639,+0.01749] tails [-0.16703,+0.03498] 4/8: **fail**.
FT H1 (bd9effe3f) mean ΔE -0.0311 CI [-0.083, 0.014] 4/8: fail.
Decision: **HOLDOUT_INSTABILITY CONFIRMED on LE**.

N18 report mean ΔE -0.06505 med -0.03606 [-0.18490,+0.04258] tails [-0.36786,+0.13912] 5/8.
FT H2 mean ΔE -0.0650 CI [-0.184, 0.045] 5/8: fail.

## Side-by-side vs FT holdout N15 Q=2048 (`bd9effe3f`)

| seed | LE ΔE | FT ΔE | LE−FT | LE start0 | FT start0 |
|---|---|---|---|---|---|
| 10144 | +0.03498 | +0.03498 | +0.00000 | 0 | 0 |
| 10145 | -0.16703 | -0.16703 | -0.00000 | 2 | 2 |
| 10146 | +0.02556 | +0.02556 | +0.00000 | 3 | 3 |
| 10147 | -0.01998 | -0.01998 | +0.00000 | 2 | 2 |
| 10148 | -0.05707 | -0.05707 | +0.00000 | 0 | 0 |
| 10149 | +0.02850 | +0.02850 | -0.00000 | 0 | 0 |
| 10150 | -0.12753 | -0.12753 | -0.00000 | 1 | 1 |
| 10151 | +0.03386 | +0.03386 | +0.00000 | 0 | 0 |

## Side-by-side vs FT holdout N18 Q=2048 (`bd9effe3f`)

| seed | LE ΔE | FT ΔE | LE−FT | LE start0 | FT start0 |
|---|---|---|---|---|---|
| 10144 | -0.12300 | -0.12300 | +0.00000 | 0 | 0 |
| 10145 | +0.10451 | +0.10451 | -0.00000 | 0 | 0 |
| 10146 | +0.05753 | +0.05753 | -0.00000 | 0 | 0 |
| 10147 | -0.01015 | -0.01015 | -0.00000 | 0 | 0 |
| 10148 | -0.06197 | -0.06197 | -0.00000 | 2 | 2 |
| 10149 | +0.13912 | +0.13912 | +0.00000 | 0 | 0 |
| 10150 | -0.36786 | -0.36786 | +0.00000 | 3 | 3 |
| 10151 | -0.25858 | -0.25858 | +0.00000 | 0 | 0 |

N15 B* wall 10.4. N15 fit wall 28.52101159095764. N18 B* wall 103.6. N18 fit wall 259.91836071014404. Sum cell wall 1026.7s. Sum cell CPU 1025.7s. Peak RSS KiB 265460.
