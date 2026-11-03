# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-04 00:40 Asia/Shanghai — L=2 CAPACITY FINAL_

BEAT = R≤0.98. Gates locked. R1=Opus; R2+=Grok 4.6.

## L=16 — **DONE / ALL_CORE_BEAT** under train{6,9}

### Champion: **power_p** learned seed0 (box ≡ cloud)

Edges ratio69_w9 on n12/n15. p_final≈0.0515. Params: `champion/params_L16_train69_power_p.pt`.

| n | QAOA Med | Med | R | QAOA B90 | B90 |
|---|----------|-----|---|----------|-----|
| 6 | 1.2261 | 1.1555 | **0.9424** | 2.300 | 1.685 |
| 9 | 1.5787 | 1.4056 | **0.8904** | 3.901 | 3.227 |
| 12 | 2.0707 | 1.8175 | **0.8777** | 7.027 | 5.796 |
| 15 | 2.8881 | 2.5488 | **0.8825** | 11.423 | 9.527 |

### power_p learned seeds 0–2

| seed | p | n6 R | n9 R | n12 R | n15 R | ALL |
|------|---|------|------|-------|-------|-----|
| 0 | 0.0515 | 0.9424 | 0.8904 | 0.8777 | 0.8825 | True |
| 1 | 0.0520 | 0.9336 | 0.8930 | 0.8896 | 0.8908 | True |
| 2 | 0.0438 | 0.9350 | 0.8935 | 0.8833 | 0.8851 | True |

### power_p p0 sweep (seed0) — all ALL_CORE_BEAT

| tag | p_final | n6 | n9 | n12 | n15 |
|-----|---------|----|----|-----|-----|
| p-0.5 | -0.0162 | 0.9427 | 0.8915 | 0.8827 | 0.8941 |
| p0 | 0.0515 | 0.9424 | 0.8904 | 0.8777 | 0.8825 |
| p0.5 | 0.1038 | 0.9399 | 0.8960 | 0.8901 | 0.8961 |
| p1 | 0.4101 | 0.9462 | 0.8909 | 0.8758 | 0.8817 |

(p1-init ends n15 R=**0.8817**, slight edge over learned-s0 0.8825; champ kept as learned-s0 per primary recipe.)

### ratio69_w9 L=16 seeds 0–5 (also ALL_CORE; runner-up)

| seed | n6 | n9 | n12 | n15 |
|------|----|----|-----|-----|
| 0 | 0.9434 | 0.8912 | 0.8812 | 0.8947 |
| 1 | 0.9336 | 0.8934 | 0.8933 | 0.8987 |
| 2 | 0.9359 | 0.8940 | 0.8850 | 0.8884 |
| 3 | 0.9342 | 0.8952 | 0.8844 | 0.8894 |
| 4 | 0.9339 | 0.8905 | 0.8881 | 0.8990 |
| 5 | 0.9376 | 0.9008 | 0.8994 | 0.9135 |

## Cloud L2 harvest (bc-ca109616 → main)

Branch `cursor/beat-qaoa-l2-train69-28cb` tip harvested 2026-11-04 ~00:00 Asia/Shanghai.

- `train69_dense_trust_ball.py` (box) + `train69_dense_trust_ball_batched.py` (cloud batched forward)
- `explore_OCT29/lib/qaoa.py` restored (QAOA L2 Med/B90 match recorded; scale1000 regen SCB≠DATA_MANIFEST sha)
- n9med ρ0.03/0.08 × seeds 0–7 tables under `trust69/n9med_rho0.03|0.08/`
- Best cloud n15 R = **0.9878** (n9med ρ0.03 s3). Dense ρ0.05/0.08: all starts → same basin.
- Objective gap confirmed: median(inv/inv_Q) ≠ Med(inv)/Med(inv_Q). Next: `--loss med_align`.

## L=2 — NO ALL_CORE (delta-ball warm) (_live 2026-11-04 00:50 Asia/Shanghai_)

**ALL_CORE_BEAT = false.**

`train69_med_align_warm.py` projects (c2,c3) onto an ℓ2 ball **centered on the init**, not the QAOA origin. Knee init `pareto_fine/t5_rs0/params_L2_a0.70_s3.pt` has ||(c2,c3)||=0.044. After ρ=0.03 med_align, ||delta||≤0.026 (an origin ball of radius 0.03 would have forced ||c||≤0.03). Best of that run:

| n | R | verdict |
|---|---|---|
| 6 | 0.9589 | BEAT |
| 9 | 0.9806 | TIE |
| 12 | 0.9862 | TIE |
| 15 | **0.98496** | TIE |

Eval: `trust69/med_align_delta_rho0.03_q0.5/eval_L2_seed1.json`.

Among points with n6 and n9 both BEAT, published pareto α=0.80 s1 is still the knee (n15 0.9853, n12 0.9858). That checkpoint is not in git. A replay of the current blend script’s a=0.80 s1 lands at n15 0.9963; delta polish ρ=0.02 s5 of it keeps n6/n9 BEAT at n15 0.9863 (`med_align_delta_rho0.02_q0.5/eval_L2_seed5.json`).

QAOA-warm med_align (origin ball, separate from Idea C): trust ρ0.05/0.08 ×4, n9-only ρ0.05 ×4, ratio69_w9 ×3 — no ALL_CORE; best n15 0.9879 with n9≈1. Box soft_q delta q∈{0.35,0.4} 24/24 False.

| n15 R | n6 | n9 | n12 | ALL | path |
|-------|----|----|-----|-----|------|
| 0.9850 | 0.9589 | 0.9806 | 0.9862 | False | trust69/med_align_delta_rho0.03_q0.5/eval_L2_seed1.json |
| 0.9853 | 0.9573 | 0.9782 | 0.9858 | False | pareto_blend/eval_L2_a0.80_s1.json (box; pt not in git) |
| 0.9850 | 0.9598 | 0.9822 | 0.9817 | False | pareto_fine/t5_rs0/eval_L2_a0.70_s3.json |
| 0.9865 | 0.9873 | 1.0016 | 0.9957 | False | trust69/n9med_rho005/eval_L2_seed7.json |
| 0.9878 | — | — | — | False | cloud n9med ρ0.03 s3 (pre-med_align best) |

## Cloud (closed)
- L16 Idea1+Idea2 FINISHED. Champ=power_p s0.
- L2 dense+n9med bc-ca109616 FINISHED no-BEAT.
- L2 Round3 denser bc-5198772e FINISHED no-BEAT.
- L2 med_align bc-0fa5a09f FINISHED no-BEAT.

Details: `CAPACITY_L2_EVIDENCE.md`. **L=2 shallow hunt wrapped.**
