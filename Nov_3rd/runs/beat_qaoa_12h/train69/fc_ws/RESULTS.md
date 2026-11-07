# FC-WS RESULTS — footprint-class Walsh × commutator-matched shell

**Verdict: kill.**

Joint FC-WS does not beat footprint-only and Walsh-degree together, and the shell coupling κ does not survive holdout often enough to be a transfer lever. The prescribed pooling edit is not the failure: class `(1,1,1)` has mean Walsh mass ≈0.27 on the n=9 train pack (0 at n=6), so it is constrained. No new ALL_CORE_BEAT beyond the existing Walsh-degree baseline.

## What was trained

Graded H_c, labeled as such. Six footprint classes

`(1), (2), (1,1), (3), (2,1), (1,1,1)`

boost the Walsh piece of `U_c^X` by `(1+δ_C)`. Shell orders `μ_t` of the frozen champ phases are rescaled by `(1+κ Õ_t)` for `t=1,2,3` and mapped back with the inverse Krawtchouk transform (`M[t,k]=K_t(k;3)`, `M²=8I`). `Õ_t` is the Walsh-mass-weighted anticommutator table from Opus Idea 1. Degree ≥4 (4-SAT residual) stays at weight 1. `δ=0, κ=0` recovers the warm schedule (checked: scalar champ and Walsh-degree `γ_w` match `inv_p`).

Train `{6,9}` only, `max_train=100`, batch 32, w6=1, w9=2, 20% n9 holdout, Adam 400 (lr 1e-2) + LBFGS. Schedule frozen. 6 δ + κ, no pooling.

| warm | schedule |
|------|----------|
| champ, L=2 | pareto a0.80 s1 (`champion/params_L2_train69_pareto_a080_s1.pt`) |
| walsh, L=2 | Walsh-degree seed1 (`ucx_bit_a/walsh_degree/params_L2_seed1_B100.pt`) |
| champ, L=16 | sealed power_p seed0 (`champion/params_L16_train69_power_p.pt`), learn only the 7 globals |

CORE is scale1000, SCB=54e9. L=2 and L=16 QAOA Med match `qaoa_angles/L{2,16}/results.json`. Re-eval of the published checkpoints matches the recorded Med (Walsh seed1 n15 Med 72.797, R 0.9567; L=2 champ n15 R 0.9853; L=16 power_p Med 1.1555 / 1.4056 / 1.8175 / 2.5488).

Raw BEAT = Med < QAOA Med. Legacy BEAT = R≤0.98.

Several seeds restore Adam step 0 (`δ=±0.01`, `κ=0`) because the n9 holdout never improved. Those checkpoints are a one-step perturbation of the warm start, not a second basin.

## L=2 B=100 gate

Gate: joint beats footprint (same seed) **and** Walsh-degree seed1 on both n12 and n15 Med.

**No seed passes.** Champ-warm joint beats footprint on n12 and n15 for seeds 0 and 2, and still loses to Walsh seed1 on n15 (Med 72.6 and 72.3 vs 70.9). Walsh-warm joint does not beat Walsh seed1 on n15 at B=100.

## L=2 B=1000 CORE

QAOA Med: n6 5.3257, n9 12.6416, n12 29.6063, n15 76.0878.

| warm | arm | seed | κ | n6 Med (R) | n9 Med (R) | n12 Med (R) | n15 Med (R) | knee R | legacy ALL |
|------|-----|------|---|------------|------------|-------------|-------------|--------|------------|
| — | walsh | 0 | — | 4.976 (0.9343) | 11.977 (0.9474) | 28.320 (0.9566) | 73.006 (0.9595) | 0.9595 | yes |
| — | walsh | 1 | — | 5.023 (0.9432) | 11.968 (0.9467) | 28.035 (0.9469) | 72.797 (0.9567) | **0.9567** | yes |
| — | walsh | 2 | — | 4.997 (0.9383) | 11.922 (0.9431) | 28.210 (0.9528) | 73.180 (0.9618) | 0.9618 | yes |
| champ | champ | 0 | 0 | 5.098 (0.9573) | 12.366 (0.9782) | 29.187 (0.9858) | 74.969 (0.9853) | 0.9858 | no |
| champ | footprint | 0 | 0 | 5.090 (0.9557) | 12.254 (0.9693) | 29.300 (0.9897) | 75.859 (0.9970) | 0.9970 | no |
| champ | footprint | 2 | 0 | 5.107 (0.9589) | 12.313 (0.9740) | 29.178 (0.9855) | 75.092 (0.9869) | 0.9869 | no |
| champ | joint | 0 | +1.07 | 5.082 (0.9542) | 12.334 (0.9757) | 29.316 (0.9902) | 75.561 (0.9931) | 0.9931 | no |
| champ | joint | 2 | +1.06 | 5.087 (0.9551) | 12.336 (0.9759) | 29.040 (0.9809) | 75.262 (0.9892) | 0.9892 | no |
| walsh | footprint | 2 | 0 | 4.974 (0.9340) | 11.964 (0.9464) | 28.099 (0.9491) | 72.680 (0.9552) | 0.9552 | yes |
| walsh | joint | 0 | +0.17 | 5.048 (0.9479) | 12.096 (0.9568) | 28.527 (0.9636) | 74.482 (0.9789) | 0.9789 | yes |
| walsh | joint | 2 | −0.33 | 4.973 (0.9338) | 11.982 (0.9478) | 28.118 (0.9497) | 72.928 (0.9585) | 0.9585 | yes |

Seed 1 on every learned arm restored Adam step 0 (`|δ_C|=0.01`, `κ=0`). Walsh-warm seed 1 therefore matches footprint exactly (n12 Med 27.914, n15 72.629). That n15 edge over Walsh seed1 (72.797) is inside the paired bootstrap 95% CI (n12 diff CI [−0.25, 0.30], n15 [−0.65, 0.99]).

Paired bootstrap of Med_footprint − Med_joint, 2000 resamples: **0/6 learned seeds** have joint better than footprint at 95% on both n12 and n15. The nonzero-κ runs do not beat Walsh seed1 on n15.

Champ-warm joint improves n9 R by about 0.002 and **worsens** n15 R (seed 0: +0.0025 vs −0.0078; seed 2: +0.0024 vs −0.0039). That is the train-size kill (n15 gain smaller than n9 gain). `|κ|<0.02` on 1/3 seeds per warm, not 2/3.

## L=16 frozen power_p, B=1000

QAOA Med: n6 1.2261, n9 1.5787, n12 2.0707, n15 2.8881. Champ is the sealed power_p schedule with `δ=κ=0`.

| arm | seed | κ | n6 Med (R) | n9 Med (R) | n12 Med (R) | n15 Med (R) |
|-----|------|---|------------|------------|-------------|-------------|
| champ | 0 | 0 | 1.1555 (0.9424) | 1.4056 (0.8904) | 1.8175 (0.8777) | 2.5488 (0.8825) |
| footprint | 1 | 0 | 1.1571 (0.9437) | 1.4354 (0.9093) | 1.8770 (0.9065) | 2.5846 (0.8949) |
| joint | 0 | 0 | 1.1560 (0.9428) | 1.4074 (0.8915) | 1.8157 (0.8768) | 2.5497 (0.8829) |
| joint | 1 | +1.38 | 1.1514 (0.9390) | 1.4123 (0.8946) | 1.8348 (0.8861) | 2.5149 (0.8708) |
| joint | 2 | 0 | 1.1563 (0.9430) | 1.4075 (0.8916) | 1.8175 (0.8777) | 2.5593 (0.8862) |

`κ→0` on **2 of 3** joint seeds (holdout kept Adam step 0). That is a kill.

The one seed that keeps κ (seed 1, κ=+1.38, δ≈[0.00, 0.03, 0.01, 0.17, 0.12, 0.07]) lowers n15 Med from 2.549 to 2.515 and raises n12 from 1.818 to 1.835. Paired bootstrap vs the champ: n15 CI for Med_champ−Med_joint is [−0.010, 0.086] (includes 0); n12 CI [−0.043, 0.023] (point estimate favors the champ). n9 R gets worse (−0.004). Not a transfer win.

All of these L=16 rows are raw ALL_CORE_BEAT because the frozen power_p champ already is. Joint does not widen that margin on n12 and n15 together.

## Kill checklist

| criterion | result |
|-----------|--------|
| L=2 gate: (c) beats (b) and (d) on n12 and n15 Med | fail (0/6 seeds at B=100; B=1000 agrees) |
| (c) ≤ (b) inside bootstrap 95% CI on n12 and n15 | yes — 0/6 L=2 seeds are significantly better; L=16 seed 1 beats a *worse* footprint but not the champ |
| κ→0 in ≥2 of 3 seeds | L=2: 1/3. L=16: **2/3** |
| n15 gain < n9 gain | yes on every champ-warm L=2 joint seed; L=16 seed 1 gains n15 and loses n9/n12 |
| `(1,1,1)` underconstrained | no — n=9 mass ≈0.27; pooling not applied |

## Reproduce

```bash
python3 Nov_3rd/tests/test_fc_ws.py
python3 Nov_3rd/scripts/train69_fc_ws.py --mode train --arm joint --warm champ --L 2 --seed 0 \
  --max_train 100 --eval_B 100 --threads 1
python3 Nov_3rd/scripts/train69_fc_ws.py --mode eval --eval_pt <params.pt> --eval_B 1000
```

Code: `Nov_3rd/fc_ws.py`, `Nov_3rd/ansatz.py` (`fcws=` optional; champ / power_p / Walsh paths unchanged), `Nov_3rd/scripts/train69_fc_ws.py`.
