# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 16:35 UTC — L=2 med_align near-miss capacity_

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

## L=2 — near-miss capacity (med_align hunt, no ALL_CORE)

Gates locked. Eval scale1000 B=1000 CORE n∈{6,9,12,15}. BEAT = R≤0.98.
No ALL_CORE_BEAT. Published pareto knee is unchanged: α=0.80 s1 n6=0.9573 n9=0.9782 n12=0.9858 n15=0.9853 (params pt not on main; endpoints re-blended below).

**Declare:** L=2 under train{6,9} is a near-miss capacity at that knee. 40 independent `--loss med_align` soft_q=0.5 starts (trust69 ρ∈{0.05,0.08,0.12} seeds 0–5, n9med w6=0 w9=1 ρ∈{0.03,0.05,0.08} seeds 0–5, ratio69_w9 seeds 0–3) leave **n15≥0.985 on every start**. 39/40 also leave n12≥0.982. The exception is ratio69_w9 seed3 (n12=0.9775 BEAT, n15=1.0001). Idea C/B/E did not move n15 under 0.985, so the knee is frozen.

### Idea A — med_align soft_q=0.5 (40 starts, 0 fail)

Best n15 is still above the published knee. Summary: `med_align_sweep_summary_q0.5.json`.

| n15 | n6 | n9 | n12 | path |
|-----|----|----|-----|------|
| 0.9876 | 0.9868 | 0.9966 | 0.9944 | trust69/med_align_n9_rho0.03/eval_L2_seed5.json |
| 0.9877 | 0.9867 | 1.0008 | 0.9927 | trust69/med_align_n9_rho0.05/eval_L2_seed5.json |
| 0.9879 | 0.9854 | 0.9981 | 0.9878 | trust69/med_align_rho0.08/eval_L2_seed3.json |
| 1.0001 | 0.9639 | 0.9758 | 0.9775 | ratio69_w9/med_align/eval_L2_seed3.json |

### Idea C — warm-start α=0.80 blend, freeze γ,c1, med_align on c2,c3

Polished `params_L2_train69_pareto_a080_s1.pt` is not on main. Init is the unpolished blend 0.2·n9med s7 + 0.8·ratio69 s0 s1 (`pareto_blend/params_L2_a080_blend.pt`). Raw blend eval: n6=0.9605 n9=0.9778 n12=0.9806 n15=0.9903. Polish ρ∈{0.02,0.04} × seeds 0–5 (adam 200) and ρ=0.04 adam 250 seeds 0–3. Same seed matches across ρ (step stays inside ρ=0.02).

Best joint (n6 and n9 BEAT): **n12=0.9829 n15=0.9878** (`trust69/med_align_pareto_a080_r0.02/eval_L2_seed2.json`). Best n12 BEAT is seed5 n12=0.9783 with n15=0.9966. No ALL_CORE.

### Idea B — soft_q∈{0.4,0.35}, trust69 ρ=0.08, seeds 0–3

8 starts. Best n15=0.9910 (q=0.35 seed0, n6=0.9775 BEAT, n9=0.9989). No n9 BEAT. No ALL_CORE.

### Idea E — α∈[0.74,0.84] step 0.01, 50-step polish, δ=0.02

Endpoints n9med s7 × ratio s1, polish seeds 0–1 (22 evals). Several n6/n9/n12 BEAT (best n12=0.9761 at α=0.84 s0) with n15≥0.996. Best max(n12,n15) is α=0.82 s1: n6=0.9572 n9=0.9766 n12=0.9807 n15=0.9908. Knee does not cross 0.98 on n12 and n15 together.

## L=2 — pre-med_align snapshot (_2026-11-03 23:35 Asia/Shanghai_)

Best n15 then **0.9853** (pareto_blend α=0.80 s1; n6/n9 BEAT, n12 0.9858 TIE). Still the best joint n15. n9med ρ0.05 seeds 0–7 were already done.

| n15 R | n6 | n9 | n12 | ALL | path |
|-------|----|----|-----|-----|------|
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

Running: n9med ρ0.03/0.08; dense_trust ρ0.05→0.08; cloud bc-ca109616; R3 pareto_blend+dense_micro.

## Cloud
- L16 Idea1+Idea2 FINISHED. Champ=power_p s0.
- L2 dense+n9med: bc-ca109616 DONE (harvested above).

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
