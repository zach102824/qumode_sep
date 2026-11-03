# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 23:02 Asia/Shanghai_

BEAT = R≤0.98. Gates locked. Idea-gen R1=Opus; R2+=Grok 4.6.

## L=16 BOX — **ALL_CORE_BEAT = True** (honest scale1000 B=1000)

Train ONLY {6,9}. Eval CORE {6,9,12,15}.

### ratio69_w9 L=16 seed0 (box)

| n | QAOA Med | Med | R | QAOA B90 | B90 | verdict |
|---|----------|-----|---|----------|-----|---------|
| 6 | 1.226 | 1.157 | **0.943** | 2.300 | 1.686 | BEAT |
| 9 | 1.579 | 1.407 | **0.891** | 3.901 | 3.239 | BEAT |
| 12 | 2.071 | 1.825 | **0.881** | 7.027 | 5.821 | BEAT |
| 15 | 2.888 | 2.584 | **0.895** | 11.423 | 9.694 | BEAT |

**ALL_CORE_BEAT** = True

### ratio69_w9 L=16 seed1 (box)

| n | QAOA Med | Med | R | QAOA B90 | B90 | verdict |
|---|----------|-----|---|----------|-----|---------|
| 6 | 1.226 | 1.145 | **0.934** | 2.300 | 1.734 | BEAT |
| 9 | 1.579 | 1.410 | **0.893** | 3.901 | 3.305 | BEAT |
| 12 | 2.071 | 1.850 | **0.893** | 7.027 | 6.010 | BEAT |
| 15 | 2.888 | 2.596 | **0.899** | 11.423 | 10.036 | BEAT |

**ALL_CORE_BEAT** = True

### power_p L=16 seed0 (box) — best n15 so far

| n | QAOA Med | Med | R | QAOA B90 | B90 | verdict |
|---|----------|-----|---|----------|-----|---------|
| 6 | 1.226 | 1.155 | **0.942** | 2.300 | 1.685 | BEAT |
| 9 | 1.579 | 1.406 | **0.890** | 3.901 | 3.227 | BEAT |
| 12 | 2.071 | 1.817 | **0.878** | 7.027 | 5.796 | BEAT |
| 15 | 2.888 | 2.549 | **0.883** | 11.423 | 9.527 | BEAT |

**ALL_CORE_BEAT** = True

## Box L=2 (shallow) — no ALL_CORE_BEAT yet

### ratio69_w9 L=2 seed0
| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 5.326 | 5.100 | **0.958** | BEAT |
| 9 | 12.642 | 12.339 | **0.976** | BEAT |
| 12 | 29.606 | 28.985 | **0.979** | BEAT |
| 15 | 76.088 | 75.976 | 0.999 | TIE |

### trust69 L=2 seed0 ρ=0.1
| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 5.326 | 5.184 | **0.973** | BEAT |
| 9 | 12.642 | 12.597 | 0.997 | TIE |
| 12 | 29.606 | 29.343 | 0.991 | TIE |
| 15 | 76.088 | 75.562 | 0.993 | TIE |

## L=2 trust-ball push (2026-11-03 box run 2) — **ALL_CORE_BEAT = False** everywhere

Train ONLY {6,9}; γ,c1 frozen at QAOA-L2, c2,c3 in ℓ2 ball ρ. Eval scale1000 B=1000 (regen, see
`scale1000_regen/`). Best n15 R this round = **0.9878** (n9med ρ=0.03 seed3) — TIE, not BEAT.

### Dense trust ball (`scripts/train69_dense_trust_ball.py`, full-batch Adam 300, soft-median, w6=1 w9=2)

Selected = rank 0 by train-only score; best-n15 is over the top-12 (eval-picked, not a claim).

| run | sel R6 | sel R9 | sel R12 | sel R15 | sel ALL_CORE_BEAT | any in top-12 | best n15 R (top-12) |
|-----|--------|--------|---------|---------|-------------------|---------------|---------------------|
| ρ=0.05, 64 starts | **0.9820** | 1.0026 | 0.9953 | 0.9880 | False | False | 0.9880 |
| ρ=0.08, 48 starts | **0.9759** | 0.9973 | 0.9918 | 0.9931 | False | False | 0.9909 |

All starts converge to the same basin (top-12 R spread ≤0.003); multi-start is not the bottleneck.

### n9med (`train_beat_train69.py --idea trust69 --w6 0 --w9 1 --holdout_frac 0.3 --adam_steps 800`)

| seed | ρ=0.03 R6 / R9 / R12 / R15 | ρ=0.08 R6 / R9 / R12 / R15 |
|------|----------------------------|----------------------------|
| 0 | 0.9862 / 0.9999 / 0.9981 / 0.9886 | 0.9854 / 1.0000 / 0.9903 / 0.9894 |
| 1 | 0.9903 / 0.9970 / 0.9931 / 0.9886 | 0.9796 / 1.0012 / 0.9913 / 0.9911 |
| 2 | 0.9844 / 1.0008 / 0.9991 / 1.0000 | 0.9747 / 1.0002 / 0.9919 / 0.9917 |
| 3 | 0.9871 / 0.9964 / 0.9937 / **0.9878** | 0.9781 / 1.0011 / 0.9918 / 0.9906 |
| 4 | 0.9829 / 0.9997 / 0.9988 / 0.9912 | 0.9749 / 0.9999 / 0.9922 / 0.9914 |
| 5 | 0.9904 / 0.9966 / 0.9935 / 0.9889 | 0.9775 / 0.9989 / 0.9904 / 0.9907 |
| 6 | 0.9901 / 0.9976 / 0.9938 / 0.9887 | 0.9775 / 0.9997 / 0.9902 / 0.9909 |
| 7 | 0.9860 / 1.0006 / 0.9961 / 0.9895 | 0.9770 / 1.0007 / 0.9920 / 0.9906 |

0/16 ALL_CORE_BEAT. n9 eval R ≈1.00 in every run despite held-out train n9 median log R −0.006…−0.017.

### Why n9 stalls at R≈1.00 — objective ≠ gate statistic

Losses rank by *median of per-instance ratios*; the gate is *ratio of medians*. Dense ρ=0.05 rank0
(`trust69_dense/objective_gap_check.log`):

| n | split | median(inv/inv_Q) | median(inv)/median(inv_Q) |
|---|-------|-------------------|---------------------------|
| 9 | train | 0.9885 | 0.9832 |
| 9 | scale1000 | **0.9869** | **1.0026** |

The per-instance gain transfers to scale1000 n9; it just doesn't land on instances near the
median of 1/p. Next lever (not run): train on a soft-quantile of inv_p itself vs QAOA's median.

## Cloud L=16 — still RUNNING (no branch push yet; box already has ALL_CORE_BEAT)

- Idea1 ratio69_w9: https://cursor.com/agents/bc-7fa7da80-0efd-4d0f-b392-95c2d9fb5866 `cursor/train69-idea1-ratio69-w9-l16-66bb`

- Idea2 power_p: https://cursor.com/agents/bc-883dadcf-dbf5-4b78-9195-ab7d5d48272f `cursor/train69-idea2-power-p-l16-3c46`

Conversation stalled at setup; box results are authoritative for R1 L16.

