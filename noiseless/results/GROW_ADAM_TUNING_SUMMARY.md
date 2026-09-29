# jp grow + SPSA-Adam: hyper-parameter tuning at fixed budget (final L=4, 800 SPSA steps)

Updated Asia/Shanghai: 2026-09-29 17:40 CST. All fleets run 17:18–17:35 CST (09:18–09:35 UTC), 8 workers,
OMP_NUM_THREADS=1. jp, binary encoding, λ1 = λ = 0 Gibbs, `--optimizer spsa_adam --grow`, growth L=1→4 (unless noted),
**800 SPSA steps total = 1604 cost evals/trial for every setting** (start depth 2: 3 stages → 1603).

## New flags (all opt-in; defaults bit-for-bit = commit 28f9ec6)

| flag | meaning |
|---|---|
| `--grow-steps-schedule a,b,c,d` | SPSA steps per growth stage (overrides `--steps` / `--grow-steps-per-stage`) |
| `--grow-lr-schedule l1,l2,l3,l4` | Adam lr per stage (overrides `--adam-lr`) |
| `--grow-eta-scale f1,f2,f3,f4` | per-stage multiplier on the sampled-tail η: η_used = f · η_ctrl (controller/EMA state untouched) |
| `--grow-c-schedule c1,c2,c3,c4` | SPSA perturbation c per stage (overrides `--spsa-c`) |

Each list must have one entry per stage (`L − grow_start + 1`); mismatches error out. `grow_trial()` gets matching
`steps_schedule / lr_schedule / eta_scale_schedule / c_schedule` kwargs; `NoiselessSimulator.eta_scale` (default 1.0).
Per-stage records now also log `spsa_c`, `adam_lr`, `eta_scale`.

**η convention:** `gibbs_objective = −log Σ p·exp(−η(E−Emin)) + η·Emin`, so η is an **inverse temperature**. η→0: cost
→ η·⟨E⟩ (hot, mean energy); η→∞: cost → −log p(GS) + η·Emin (cold, GS-probability focused). **Hotter = factor < 1.**
Base η is the sampled-tail controller (ln 20 / (q25 − q05), EMA 0.35, refreshed every 5 steps), restarted each stage.

Tests: `noiseless/tests/test_grow_tuning.py` (9 tests: golden hashes vs 28f9ec6 for spsa and spsa_adam growth incl.
start 2 / kick 0.2 / steps-per-stage; explicit default schedules == legacy bit-for-bit; per-stage step/eval counts and
start-2 budget; length validation; lr/c applied per stage with earlier stages unchanged; η scale multiplies the
controller output and is restored; η inverse-temperature limits; runner/CLI). Full suite: **84 passed**. The screen
baseline reproduces the C_adam L=4 records (x and p(GS)) exactly for all 50 overlapping trials.

## Screening

Subset: **four_sat_000..004** (`--max-h 5`, the first five files) × 10 trials, `--seed 20260917`, `--layers 4`
(same per-trial seeds / x0 as C_adam). This subset is harder than average (C_adam on it: success 0.784, p̄ 0.343 over
25 trials vs 0.852 / 0.397 on all 20), which helps resolve success. Paired Δp = same instance + trial + seed + x0 vs the
screen baseline. Noise: stderr of mean p(GS) ≈ 0.02–0.03, success ≈ ±0.05 per setting (n = 50); paired Δp stderr ≈
0.015–0.03. Round 1 was one-knob-at-a-time as planned; because lr dominated, I added follow-up single-knob points
(more lr schedules, kick σ 1.0 / 2.0) before combining.

Setting legend (everything else = baseline: lr 0.05, 200×4 steps, kick 0.05, no η scale, c 0.15, start 1):
`lr0pXX` = constant Adam lr; `lrs_hi_lo` 0.1,0.1,0.1,0.03; `lrs_decay` 0.2,0.1,0.05,0.02; `lrs_decay_lo`
0.1,0.05,0.03,0.02; `lrs_decay_01` 0.2,0.1,0.05,0.01; `lrs_decay_hi` 0.3,0.15,0.05,0.02; **`lrs_decay_vhi`
0.5,0.2,0.05,0.02**; `lrs_1p0` 1.0,0.3,0.05,0.02; `lrs_vhi_s3hi` 0.5,0.2,0.1,0.02; `lrs_vhi_last01/03` 0.5,0.2,0.05,0.01/0.03;
`split_*` step schedule; `kickX` kick σ; `eta_hot_early` 0.5,0.7,1,1; `eta_cold_late` 1,1,1.5,2; `eta_hot_cold`
0.5,0.7,1.5,2; `c0pX` constant c; `cs_last0p05` 0.15,0.15,0.15,0.05; `cs_taper` 0.15,0.15,0.1,0.05; `start2` L=2→4,
267/267/266 steps.

| setting | success ± se | mean p(GS) ± se | median | frac>0.5 | paired Δp ± se | frac higher | flips f→s / s→f | evals |
|---|---|---|---|---|---|---|---|---|
| lrs_decay_vhi | 0.940 ± 0.034 | 0.5331 ± 0.0190 | 0.5224 | 0.640 | +0.2173 ± 0.0277 | 0.86 | 10 / 1 | 1604 |
| lrs_vhi_last01 | 0.940 ± 0.034 | 0.5275 ± 0.0192 | 0.5143 | 0.600 | +0.2117 ± 0.0268 | 0.86 | 10 / 1 | 1604 |
| lrs_1p0 | 0.940 ± 0.034 | 0.5140 ± 0.0201 | 0.5175 | 0.580 | +0.1982 ± 0.0370 | 0.78 | 12 / 3 | 1604 |
| lrs_vhi_last03 | 0.940 ± 0.034 | 0.5100 ± 0.0185 | 0.5175 | 0.580 | +0.1942 ± 0.0280 | 0.84 | 10 / 1 | 1604 |
| lrs_vhi_s3hi | 0.940 ± 0.034 | 0.4995 ± 0.0198 | 0.5050 | 0.500 | +0.1837 ± 0.0283 | 0.82 | 10 / 1 | 1604 |
| lrs_decay_hi | 0.940 ± 0.034 | 0.4672 ± 0.0203 | 0.5036 | 0.540 | +0.1514 ± 0.0252 | 0.84 | 10 / 1 | 1604 |
| lrs_decay | 0.860 ± 0.049 | 0.4283 ± 0.0277 | 0.5094 | 0.560 | +0.1125 ± 0.0169 | 0.78 | 6 / 1 | 1604 |
| lrs_decay_lo | 0.820 ± 0.054 | 0.4200 ± 0.0302 | 0.4990 | 0.500 | +0.1042 ± 0.0176 | 0.74 | 4 / 1 | 1604 |
| lrs_decay_01 | 0.860 ± 0.049 | 0.4174 ± 0.0271 | 0.4969 | 0.420 | +0.1016 ± 0.0171 | 0.76 | 6 / 1 | 1604 |
| kick0p5 | 0.800 ± 0.057 | 0.4169 ± 0.0299 | 0.4728 | 0.420 | +0.1011 ± 0.0238 | 0.80 | 4 / 2 | 1604 |
| lrs_hi_lo | 0.900 ± 0.042 | 0.4031 ± 0.0242 | 0.4155 | 0.320 | +0.0873 ± 0.0220 | 0.76 | 9 / 2 | 1604 |
| lr0p02 | 0.760 ± 0.060 | 0.3922 ± 0.0330 | 0.4978 | 0.460 | +0.0764 ± 0.0174 | 0.74 | 2 / 2 | 1604 |
| kick1p0 | 0.980 ± 0.020 | 0.3643 ± 0.0175 | 0.3740 | 0.120 | +0.0485 ± 0.0264 | 0.58 | 11 / 0 | 1604 |
| c0p1 | 0.840 ± 0.052 | 0.3507 ± 0.0260 | 0.3868 | 0.220 | +0.0349 ± 0.0137 | 0.64 | 4 / 0 | 1604 |
| split_100_150_250_300 | 0.860 ± 0.049 | 0.3488 ± 0.0274 | 0.3840 | 0.200 | +0.0330 ± 0.0191 | 0.52 | 5 / 0 | 1604 |
| eta_hot_cold | 0.880 ± 0.046 | 0.3482 ± 0.0263 | 0.3636 | 0.220 | +0.0324 ± 0.0185 | 0.58 | 6 / 0 | 1604 |
| cs_taper | 0.840 ± 0.052 | 0.3402 ± 0.0260 | 0.3619 | 0.280 | +0.0244 ± 0.0166 | 0.60 | 5 / 1 | 1604 |
| eta_cold_late | 0.860 ± 0.049 | 0.3400 ± 0.0266 | 0.3650 | 0.260 | +0.0242 ± 0.0143 | 0.56 | 5 / 0 | 1604 |
| kick0p2 | 0.760 ± 0.060 | 0.3353 ± 0.0277 | 0.3547 | 0.200 | +0.0195 ± 0.0138 | 0.58 | 1 / 1 | 1604 |
| c0p05 | 0.800 ± 0.057 | 0.3352 ± 0.0289 | 0.3678 | 0.300 | +0.0194 ± 0.0132 | 0.62 | 3 / 1 | 1604 |
| cs_last0p05 | 0.800 ± 0.057 | 0.3244 ± 0.0262 | 0.3648 | 0.140 | +0.0085 ± 0.0114 | 0.66 | 3 / 1 | 1604 |
| eta_hot_early | 0.800 ± 0.057 | 0.3231 ± 0.0275 | 0.3548 | 0.180 | +0.0073 ± 0.0148 | 0.54 | 2 / 0 | 1604 |
| start2 | 0.780 ± 0.059 | 0.3201 ± 0.0239 | 0.3114 | 0.220 | +0.0043 ± 0.0324 | 0.46 | 9 / 8 | 1603 |
| base | 0.760 ± 0.060 | 0.3158 ± 0.0273 | 0.3505 | 0.180 | +0.0000 ± 0.0000 | 0.00 | 0 / 0 | 1604 |
| split_50_100_250_400 | 0.860 ± 0.049 | 0.3149 ± 0.0230 | 0.3463 | 0.100 | -0.0009 ± 0.0218 | 0.50 | 5 / 0 | 1604 |
| kick2p0 | 0.940 ± 0.034 | 0.2722 ± 0.0168 | 0.2689 | 0.060 | -0.0436 ± 0.0285 | 0.40 | 10 / 1 | 1604 |
| lr0p1 | 0.800 ± 0.057 | 0.2134 ± 0.0184 | 0.2054 | 0.020 | -0.1024 ± 0.0277 | 0.36 | 9 / 7 | 1604 |
| lr0p2 | 0.720 ± 0.063 | 0.1164 ± 0.0115 | 0.1145 | 0.000 | -0.1994 ± 0.0296 | 0.20 | 10 / 12 | 1604 |

Screen wall: 10–14 s per 50-trial setting (split schedules slightly slower: more steps at the deeper, costlier stages).

### Per-knob winners (primary mean p(GS), success must not drop)

1. **Adam lr → decaying per-stage schedule** is by far the largest knob. Constant lr is a trade-off: 0.02 helps p(GS)
   (+0.08, success flat), 0.1 / 0.2 destroy it (Adam steps too large to polish). Hot early / small late gets both:
   0.5,0.2,0.05,0.02 gives +0.217 ± 0.028 paired, success 0.76 → 0.94. The final-stage lr is flat over 0.01–0.03; the
   first-stage lr keeps helping up to ~0.5 (1.0 no better).
2. Step split: 100/150/250/300 (+0.033 ± 0.019, success 0.86); 50/100/250/400 no p(GS) gain.
3. Kick σ: 0.5 (+0.10 ± 0.024, success 0.80). Large kicks (1.0, 2.0) push success to 0.94–0.98 but cost p(GS)
   (σ=2 below baseline) — they re-randomize the new layer, i.e. more exploration, less inherited progress.
4. η schedule: hot-early + cold-late 0.5,0.7,1.5,2 (+0.032 ± 0.019, success 0.88); hot-early alone ≈ 0, cold-late alone +0.024.
5. SPSA c: constant 0.1 (+0.035 ± 0.014, success 0.84); last-stage-only small c ≈ 0.
6. Start depth 2: neutral (+0.004 ± 0.032, 9/8 flips) → keep start 1.

Knobs 2, 4, 5 are each ≈ 1–2 stderr; only the lr schedule (and kick 0.5 on the old lr) is clearly significant.

## Combos (same screen subset)

`lr_*` = lr 0.2,0.1,0.05,0.02; `lrhi_*` = 0.3,0.15,0.05,0.02; `lrvhi_*` = 0.5,0.2,0.05,0.02; `kick*` σ; `eta` =
0.5,0.7,1.5,2; `split` = 100/150/250/300; `c` = 0.1; `all5` = lr_decay + kick 0.5 + eta + split + c.

| setting | success ± se | mean p(GS) ± se | median | frac>0.5 | paired Δp ± se | frac higher | flips f→s / s→f | evals |
|---|---|---|---|---|---|---|---|---|
| lrvhi_eta_c | 0.980 ± 0.020 | 0.5320 ± 0.0181 | 0.5307 | 0.660 | +0.2162 ± 0.0308 | 0.86 | 11 / 0 | 1604 |
| lrvhi_c | 0.960 ± 0.028 | 0.5307 ± 0.0191 | 0.5323 | 0.660 | +0.2148 ± 0.0281 | 0.86 | 11 / 1 | 1604 |
| lrvhi_split | 0.980 ± 0.020 | 0.5206 ± 0.0171 | 0.5311 | 0.660 | +0.2048 ± 0.0305 | 0.90 | 12 / 1 | 1604 |
| lrvhi_eta | 0.960 ± 0.028 | 0.5119 ± 0.0190 | 0.5235 | 0.640 | +0.1961 ± 0.0270 | 0.88 | 11 / 1 | 1604 |
| lrvhi_eta_split | 0.980 ± 0.020 | 0.5056 ± 0.0161 | 0.5108 | 0.600 | +0.1898 ± 0.0287 | 0.88 | 11 / 0 | 1604 |
| lrvhi_kick0p5 | 0.960 ± 0.028 | 0.5019 ± 0.0204 | 0.5375 | 0.580 | +0.1861 ± 0.0329 | 0.82 | 11 / 1 | 1604 |
| lrhi_eta | 0.960 ± 0.028 | 0.4906 ± 0.0186 | 0.5046 | 0.600 | +0.1748 ± 0.0233 | 0.92 | 10 / 0 | 1604 |
| lrhi_split | 0.900 ± 0.042 | 0.4789 ± 0.0241 | 0.5132 | 0.640 | +0.1631 ± 0.0280 | 0.82 | 9 / 2 | 1604 |
| lrhi_c | 0.920 ± 0.038 | 0.4715 ± 0.0205 | 0.5060 | 0.540 | +0.1557 ± 0.0211 | 0.90 | 10 / 2 | 1604 |
| lr_split | 0.860 ± 0.049 | 0.4523 ± 0.0283 | 0.5201 | 0.600 | +0.1365 ± 0.0225 | 0.86 | 5 / 0 | 1604 |
| lrhi_kick0p5 | 0.940 ± 0.034 | 0.4502 ± 0.0236 | 0.4740 | 0.400 | +0.1343 ± 0.0284 | 0.80 | 9 / 0 | 1604 |
| all5 | 0.900 ± 0.042 | 0.4412 ± 0.0283 | 0.4859 | 0.440 | +0.1254 ± 0.0244 | 0.80 | 8 / 1 | 1604 |
| lr_c | 0.860 ± 0.049 | 0.4388 ± 0.0268 | 0.5005 | 0.500 | +0.1230 ± 0.0180 | 0.76 | 6 / 1 | 1604 |
| lr_kick_split | 0.840 ± 0.052 | 0.4381 ± 0.0306 | 0.4946 | 0.480 | +0.1223 ± 0.0233 | 0.76 | 5 / 1 | 1604 |
| lr_eta | 0.880 ± 0.046 | 0.4298 ± 0.0272 | 0.5013 | 0.520 | +0.1140 ± 0.0195 | 0.80 | 7 / 1 | 1604 |
| lr_kick | 0.880 ± 0.046 | 0.4222 ± 0.0277 | 0.4581 | 0.400 | +0.1063 ± 0.0236 | 0.72 | 7 / 1 | 1604 |
| lr_kick_eta | 0.860 ± 0.049 | 0.4145 ± 0.0285 | 0.4685 | 0.380 | +0.0987 ± 0.0248 | 0.70 | 6 / 1 | 1604 |
| lrhi_kick1_split | 0.920 ± 0.038 | 0.4091 ± 0.0229 | 0.4058 | 0.340 | +0.0933 ± 0.0274 | 0.64 | 9 / 1 | 1604 |
| lrhi_kick1_eta | 1.000 ± 0.000 | 0.3788 ± 0.0221 | 0.3681 | 0.220 | +0.0630 ± 0.0321 | 0.60 | 12 / 0 | 1604 |
| lr_kick_c | 0.840 ± 0.052 | 0.3751 ± 0.0292 | 0.3898 | 0.280 | +0.0593 ± 0.0224 | 0.68 | 4 / 0 | 1604 |
| lrhi_kick1 | 1.000 ± 0.000 | 0.3391 ± 0.0190 | 0.3405 | 0.060 | +0.0233 ± 0.0321 | 0.50 | 12 / 0 | 1604 |

Interactions: **kick is redundant with (or hurts) the lr schedule** — once stage lr is high, the large early Adam steps
already move the new layer, and kick 0.5/1.0 on top lowers p(GS) (lr_kick < lr, lrvhi_kick0p5 < lrvhi; kick 1.0 gives
success 1.00 but p̄ 0.34–0.38). `all5` (0.441) is worse than lr_decay + any single knob. With the best lr schedule, η /
split / c add only ≈ ±0.01–0.02 (within noise): lrvhi 0.533 vs lrvhi_eta_c 0.532, lrvhi_c 0.531, lrvhi_split 0.521.
Chosen combo = best on the screen: **lr 0.5,0.2,0.05,0.02 + η scale 0.5,0.7,1.5,2 + c 0.1** (0.98 / 0.532); because
the extras looked like noise, the lr schedule alone and lr + split were also full-run.

## Full fleet (20 four_sat × 25 trials = 500, seed 20260917, final L=4, 1604 evals/trial)

```
python noiseless/run_u_sweep.py --u-names jp --layers 4 --trials 25 --steps 800 --workers 8 --seed 20260917 \
  --lambda1 0 --encoding binary --optimizer spsa_adam --grow --grow-lr-schedule 0.5,0.2,0.05,0.02 [extras] --tag grow_tune_F_<name>
noiseless/run_grow_adam_full.sh; python noiseless/analyze_grow_tuning.py   # -> grow_tune_stats_summary.json
```

| arm | success | mean p(GS) ± se | median p(GS) | frac p(GS)>0.5 | paired Δp vs C_adam ± se | frac higher | flips f→s / s→f | evals/trial | fleet wall | wall/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| C_adam (baseline, lr 0.05, 200×4, kick 0.05, c 0.15) | 0.852 | 0.3973 ± 0.0111 | 0.3957 | 0.288 | — | — | — | 1604 | (176 s, L2+3+4 fleet) | 1.43 s |
| **lr_sched** (lr 0.5,0.2,0.05,0.02 only; single best knob) | 0.978 | 0.5421 ± 0.0090 | 0.5179 | 0.594 | +0.1448 ± 0.0085 | 0.86 | 69 / 6 | 1604 | 88 s | 1.38 s |
| combo (lr sched + η 0.5,0.7,1.5,2 + c 0.1; screen winner) | 0.974 | 0.5367 ± 0.0095 | 0.5179 | 0.568 | +0.1394 ± 0.0085 | 0.86 | 70 / 9 | 1604 | 101 s | 1.57 s |
| **lr_sched_split** (lr sched + steps 100/150/250/300) | **0.982** | **0.5467 ± 0.0089** | **0.5188** | **0.608** | **+0.1494 ± 0.0086** | 0.89 | 71 / 6 | 1604 | 100 s | 1.57 s |

Paired among the tuned arms: lr_sched_split − lr_sched = +0.0046 ± 0.0055, combo − lr_sched = −0.0054 ± 0.0060 → the
three are statistically tied; **the gain is essentially all from the lr schedule.** For reference, random-init Adam /
SPSA at L4 reach 0.930 / 0.932 success (p̄ 0.21 / 0.17), so tuned growth now beats random init on success too.

Held-out check (instances not in the screen, four_sat_005..019, 375 trials): lr_sched 0.979 / 0.545, lr_sched_split
0.984 / 0.553, combo 0.976 / 0.540 vs C_adam 0.875 / 0.415; on the screen instances 0.976 / 0.532 vs 0.784 / 0.343.
The improvement is not an artefact of tuning on the screen subset.

Per-stage success / mean p(GS) (L1 → L4):
- C_adam: 0.114 / 0.092 → 0.528 / 0.230 → 0.778 / 0.341 → 0.852 / 0.397
- lr_sched: 0.170 / 0.136 → 0.626 / 0.231 → 0.914 / 0.428 → 0.978 / 0.542
- lr_sched_split: 0.228 / 0.135 → 0.598 / 0.230 → 0.922 / 0.422 → 0.982 / 0.547
- combo: 0.132 / 0.129 → 0.574 / 0.222 → 0.904 / 0.424 → 0.974 / 0.537

The big lr at L1/L2 finds the right basin more often (stage-3 success 0.78 → 0.91), and the small lr at L3/L4 lets
Adam polish (final-stage p̄ gain +0.11–0.12 vs +0.056 for C_adam).

### Failure modes (argmax energy level of failed trials; level 1 = first excited)

| arm | n fail / 500 | frac first-excited | mean p(GS) in failures | median p(GS) in failures |
|---|---|---|---|---|
| C_adam | 74 | 0.97 | 0.037 | ≈ 0 |
| lr_sched | 11 | 1.00 | 0.240 | 0.299 |
| combo | 13 | 1.00 | 0.235 | 0.270 |
| lr_sched_split | 9 | 1.00 | 0.231 | 0.241 |

Remaining failures are all first-excited, but they are no longer collapsed: the GS keeps ~0.24–0.30 weight and just
loses the argmax to the first-excited state (near-ties), unlike C_adam failures where p(GS) ≈ 0.

## Recommendation

New grow+Adam default candidate: `--grow-lr-schedule 0.5,0.2,0.05,0.02` (optionally `--grow-steps-schedule
100,150,250,300`); leave kick 0.05, c 0.15, no η scale, start 1. C_adam L4 goes from 0.852 / 0.397 / 0.396 / 0.288 to
0.978–0.982 / 0.542–0.547 / 0.518–0.519 / 0.594–0.608 (success / mean / median / frac>0.5) at the same 1604 evals.
Library defaults are unchanged (bit-for-bit legacy).

## Caveats

- 50-trial screen: small knobs (split, η, c) are ≈ 1–2 σ and did not survive combination; their "winners" are
  tentative. Kick and lr interact strongly (kick is redundant with a high early lr).
- lr schedule tuned only for this 4-stage / 800-step setting; β1/β2, carrying Adam moments across stages, and
  other depths were not tested.
- Start depth 2 used 1603 evals (one fewer final eval).

Files: scripts `noiseless/run_grow_adam_screen.sh`, `run_grow_adam_combos.sh`, `run_grow_adam_combos2.sh`,
`run_grow_adam_combos3.sh`, `run_grow_adam_full.sh`; analysis `noiseless/analyze_grow_tuning.py` →
`grow_tune_stats_summary.json`; per-run `grow_tune_{S,X,F}_*_summary.json` (full JSONs gitignored). Full runs:
`grow_tune_F_combo_20260929T093137Z`, `grow_tune_F_lr_sched_20260929T093305Z`, `grow_tune_F_lr_sched_split_20260929T093444Z`.
