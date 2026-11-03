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

## L=2 — NO ALL_CORE yet (_live 2026-11-03 23:24 Asia/Shanghai_)

Best n15 so far **0.9867** (n9med ρ0.05 s2). Alpha_ray L16-distill rays DEAD. Shrink_p BEATs n6/9(/12) but n15≥1.00.

| n15 R | n6 | n9 | n12 | ALL | path |
|-------|----|----|-----|-----|------|
| 0.9867 | 0.9888 | 1.0009 | 0.9953 | False | trust69/n9med_rho005/eval_L2_seed2.json |
| 0.9871 | 0.9859 | 1.0018 | 0.9944 | False | trust69/rho0.05/eval_L2_seed0.json |
| 0.9882 | 0.9795 | 1.0015 | 0.9933 | False | alpha_ray/eval_L2_n9med_s0_s1.json |
| 0.9883 | 0.9815 | 1.0026 | 0.9962 | False | trust69/n9med_rho005/eval_L2_seed0.json |
| 0.9884 | 0.9900 | 0.9964 | 0.9915 | False | trust69/n9med_rho005/eval_L2_seed3.json |
| 0.9885 | 0.9896 | 0.9980 | 0.9925 | False | trust69/n9med_rho005/eval_L2_seed1.json |
| 0.9893 | 0.9824 | 1.0012 | 0.9973 | False | alpha_ray/eval_L2_n9med_s0_s0.json |
| 0.9910 | 0.9739 | 1.0016 | 0.9925 | False | alpha_ray/eval_L2_trust_rho005_s0.json |
| 0.9924 | 0.9768 | 0.9993 | 0.9906 | False | alpha_ray/eval_L2_trust_rho005_s1.json |
| 0.9931 | 0.9733 | 0.9965 | 0.9911 | False | trust69/eval_L2_seed0.json |
| 0.9936 | 0.9813 | 1.0028 | 0.9963 | False | trust69/n9med_rho005/eval_L2_seed4.json |
| 0.9940 | 0.9719 | 0.9919 | 0.9900 | False | alpha_ray/eval_L2_trust_rho01_s1.json |
| 0.9963 | 0.9697 | 0.9952 | 0.9912 | False | alpha_ray/eval_L2_trust_rho01_s0.json |
| 0.9977 | 0.9705 | 0.9912 | 0.9853 | False | trust69/rho0.2/eval_L2_seed0.json |
| 0.9982 | 0.9570 | 0.9765 | 0.9786 | False | alpha_ray/eval_L2_ratio69_s0_s1.json |

Running: n9med ρ0.05 s5–7 → then ρ∈{0.03,0.08}; dense trust-ball search next.

## Cloud
- L16 Idea1+Idea2 FINISHED. Champ=power_p s0.

