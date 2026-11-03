# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 23:18 Asia/Shanghai_

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

## L=2 — NO ALL_CORE yet (_live 2026-11-04 00:42 Asia/Shanghai_)

**ALL_CORE_BEAT = false.** Best n15 **0.9850** (pareto_fine t5×ratio69 s0, α=0.70 seed3) then med_align delta polish ρ=0.03 seed1 → n15 **0.98496**. Eval: `trust69/med_align_delta_rho0.03_q0.5/eval_L2_seed1.json`. n6 0.9589 BEAT, n9 0.9806 TIE, n12 0.9862 TIE, n15 0.9850 TIE. Params: `trust69/med_align_delta_rho0.03_q0.5/params_L2_seed1.pt`.

Idea A QAOA-warm med_align (trust ρ0.05/0.08 ×4, n9-only ρ0.05 ×4, ratio69_w9 ×3): no ALL_CORE; best of those n15 is 0.9879 (trust ρ0.08 s3) with n9 still ~1.00. Idea C from reproduced pareto a=0.80 s1 (that replay is n15 0.9963, not the box 0.9853 pt, which is not in git) at ρ=0.02 s5 keeps n6/n9 BEAT with n12 0.9857 and n15 0.9863.

Prev box pareto_blend α=0.80 s1 was n15 **0.9853** with n6/n9 BEAT and n12 0.9858. This fine grid edges n15 by 0.0003 and pulls n12 to 0.9817, but n9 slips back above 0.98. When n6, n9, and n12 are all BEAT, best n15 is **0.9891** (t5×ratio s1 α=0.70 s3). n9-median Adam inside the δ-ball moves off the n15 knee.

Fine grid: α∈{0.65,0.70,0.72,0.75,0.78,0.80,0.82,0.85} × 6 polish seeds, pairs n9med {7,5,2} × ratio69 {s0,s1} (s7×s1 plus s5/s2 × both). 280 CORE evals, 0 ALL_CORE. Detail: `pareto_fine/RESULTS.md`.

| n15 R | n6 | n9 | n12 | ALL | path |
|-------|----|----|-----|-----|------|
| 0.9850 | 0.9598 | 0.9822 | 0.9817 | False | pareto_fine/t5_rs0/eval_L2_a0.70_s3.json |
| 0.9856 | 0.9596 | 0.9831 | 0.9824 | False | pareto_fine/t7_rs1/eval_raw_a0.65.json |
| 0.9865 | 0.9873 | 1.0016 | 0.9957 | False | trust69/n9med_rho005/eval_L2_seed7.json |
| 0.9866 | 0.9871 | 1.0018 | 0.9953 | False | trust69/n9med_rho005/eval_L2_seed5.json |
| 0.9867 | 0.9888 | 1.0009 | 0.9953 | False | trust69/n9med_rho005/eval_L2_seed2.json |
| 0.9871 | 0.9859 | 1.0018 | 0.9944 | False | trust69/rho0.05/eval_L2_seed0.json |
| 0.9882 | 0.9795 | 1.0015 | 0.9933 | False | alpha_ray/eval_L2_n9med_s0_s1.json |
| 0.9883 | 0.9815 | 1.0026 | 0.9962 | False | trust69/n9med_rho005/eval_L2_seed0.json |
| 0.9884 | 0.9900 | 0.9964 | 0.9915 | False | trust69/n9med_rho005/eval_L2_seed3.json |
| 0.9885 | 0.9896 | 0.9980 | 0.9925 | False | trust69/n9med_rho005/eval_L2_seed1.json |
| 0.9888 | 0.9852 | 0.9988 | 0.9991 | False | trust69/n9med_rho0.03/eval_L2_seed0.json |
| 0.9893 | 0.9824 | 1.0012 | 0.9973 | False | alpha_ray/eval_L2_n9med_s0_s0.json |
| 0.9894 | 0.9901 | 0.9973 | 0.9933 | False | trust69/n9med_rho005/eval_L2_seed6.json |
| 0.9910 | 0.9739 | 1.0016 | 0.9925 | False | alpha_ray/eval_L2_trust_rho005_s0.json |
| 0.9936 | 0.9813 | 1.0028 | 0.9963 | False | trust69/n9med_rho005/eval_L2_seed4.json |
| 0.9982 | 0.9570 | 0.9765 | 0.9786 | False | alpha_ray/eval_L2_ratio69_s0_s1.json |

pareto_fine DONE (this branch). Box may still be on n9med ρ0.03/0.08, dense_trust, and the original pareto_blend.

## Cloud
- L16 Idea1+Idea2 FINISHED. Champ=power_p s0.
- L2 dense+n9med: bc-ca109616 RUNNING.

## L=2 med_align (Idea A) — 2026-11-04 Asia/Shanghai

Gate = Med(inv)/QAOA_Med. Loss `--loss med_align` = log(soft_quantile(inv_p)/QAOA_Med_train).

| variant | best knee max(R12,R15) | notes |
|---------|------------------------|-------|
| trust69 QAOA-warm ρ0.05–0.12 | ~0.990–0.997 | R9≈1.00 same basin as dense |
| n9med med_align | ~0.989–0.993 | R9 still ~1.00 on CORE |
| origin-ball "warm" ρ0.04 (BUG) | 0.996 (3 BEAT n6/9/12) | projected toward QAOA origin; Pareto slide |
| champ↔warm blend | **0.9858 at a=1** | smooth front; champ dominates knee |
| delta-ball around champ ρ0.01 | 0.9866 best so far | no beat of champ knee |
| radial ε on champ | **0.9858 at ε=0** | ε≠0 raises knee; local min at champ |

**ALL_CORE_BEAT = False.** Champ still `pareto_blend/eval_L2_a0.80_s1.json` (n6=0.9573 n9=0.9782 n12=0.9858 n15=0.9853).
Cloud bc-0fa5a09f + bc-5198772e still RUNNING on med_align. Capacity NOT declared.

## L=2 CAPACITY NEAR-MISS (2026-11-04 ~00:25 Asia/Shanghai)

**ALL_CORE_BEAT = False.** Empirical basin claim under locked gates/train{6,9}.
Best: pareto α=0.80 s1 — n6=0.9573 BEAT, n9=0.9782 BEAT, n12=0.9858 TIE, n15=0.9853 TIE (knee 0.9858).
≥163 n6&n9-BEAT CORE evals; radial ε local min at champ; med_align/dense/n9med/hinge/soft_q=0.4 do not improve knee.
Details: `CAPACITY_L2_EVIDENCE.md`.

