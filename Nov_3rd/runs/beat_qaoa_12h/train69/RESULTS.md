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

## L=2 — CAPACITY NEAR-MISS FINAL (2026-11-04 ~00:40 Asia/Shanghai)

**ALL_CORE_BEAT = False.** Soft_q clouds + box soft_q closed; declare stands.

Champ unchanged: pareto_blend α=0.80 s1 — n6 **0.9573 BEAT**, n9 **0.9782 BEAT**, n12 **0.9858 TIE**, n15 **0.9853 TIE** (knee **0.9858**).
≥230 n6&n9-BEAT CORE evals on box; **0** ALL_CORE across box + clouds.

### Soft_q cloud harvest
- **bc-5198772e** Round3 denser → `cursor/train69-l2-round3-denser-76da`: pareto_fine 280 evals; best n15=0.9850 with n9 TIE; among n6&n9 BEAT, knee ≥0.9867; n6+n9+n12 BEAT forces n15≥0.9891. **no ALL_CORE**.
- **bc-0fa5a09f** med_align → `cursor/l2-med-align-sweeps-b563`: 40 soft_q=0.5 + Ideas C/B/E; n15≥0.985 on every primary start; published knee unchanged. **no ALL_CORE**.
- Box soft_q_delta q∈{0.4,0.35}×ρ∈{0.015,0.025,0.04}×s0–3: **24/24** False; best softq n6&n9-BEAT knee ≈0.9894.

### Earlier L=2 snapshot (pre-finalize)
Best n15 among n6&n9 BEAT remains **0.9853**. Dense/n9med/med_align/hinge/radial did not improve the knee.

| n15 R | n6 | n9 | n12 | ALL | path |
|-------|----|----|-----|-----|------|
| 0.9853 | 0.9573 | 0.9782 | 0.9858 | False | pareto_blend/eval_L2_a0.80_s1.json |
| 0.9865 | 0.9873 | 1.0016 | 0.9957 | False | trust69/n9med_rho005/eval_L2_seed7.json |
| 0.9878 | — | — | — | False | cloud n9med ρ0.03 s3 (pre-med_align best) |

## Cloud (closed)
- L16 Idea1+Idea2 FINISHED. Champ=power_p s0.
- L2 dense+n9med bc-ca109616 FINISHED no-BEAT.
- L2 Round3 denser bc-5198772e FINISHED no-BEAT.
- L2 med_align bc-0fa5a09f FINISHED no-BEAT.

Details: `CAPACITY_L2_EVIDENCE.md`. **L=2 shallow hunt wrapped.**
