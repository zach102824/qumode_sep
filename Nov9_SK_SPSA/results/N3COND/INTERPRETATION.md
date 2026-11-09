# N3COND reading after N9 and N12

Development only. Original Nov3rd, shared gamma, full s1/s2/s3, B* only, SPSA only. C0 is the existing scalar gain. C1 is the frozen axial diagonal scale. QAOA p=L uses the same split and the same Q=qL. Fidelity is post hoc. Dev seeds 10010–10013 froze the scale. Holdout seeds 10014–10017 did not.

## What ran

| batch | pid | clock | fits | failures | comparison queries |
|---|---|---|---|---|---|
| N9 | 7281 | 11.9 s | 192 | 0 | 212992 |
| N12 | 8169 | 52.0 s | 192 | 0 | 212992 |

Axial scale queries: 832 at N9 and 832 at N12, dev only. The N9 simultaneous-probe file (512 queries, lambda identically 1) is kept and is not an input. One uncharged diagnostic state rebuild per fit. GS preparation is classical and not charged. All eight instances at each N have E1 odd, a positive even gap, and a small residual.

## Primary cell

N9, L3, q=1024, holdout, Nov3 C1−C0: median ΔF −0.012, mean ΔF −0.020, 95% CI [−0.068, +0.019], median ΔE +0.023, mean ΔE +0.043, CI [+0.002, +0.103]. The adoption rule fails. The energy CI is entirely worse for C1.

## N12, same cell, not the primary endpoint

Holdout L3 q=1024, paired C1−C0: median ΔF +0.044, mean +0.032, CI [−0.004, +0.060], 3/4 higher F; median ΔE −0.046, mean −0.020, CI [−0.062, +0.050], 3/4 lower energy. The mean intervals include 0. The marginal energy medians go the other way (−7.842 for C0, −7.820 for C1), so the paired median is the one that matches the rule. The secondary holdout cell N12 L3 q=512 has median ΔF +0.052 and a mean CI [+0.010, +0.074] that excludes 0, with median ΔE −0.042 and a mean CI that still includes 0. That cell was not the adoption rule. It is not a license to edit the N9 result.

N15 and N18 stay off. The gate needs both holdout median ΔF values ≥ 0.02, and N9 is −0.012.

## Where C1 loses

At L1 and at q=128 the diagonal freeze is worse on energy. Holdout Nov3 L1 q=1024: N9 median ΔF −0.024 and median ΔE +0.045 (4/4 higher energy); N12 median ΔF −0.001 and median ΔE +0.004, with the mean ΔE CI entirely positive. Holdout L1 q=128: N9 median ΔF −0.106, N12 median ΔF −0.042, both 0/4 on F and 4/4 higher energy. QAOA C1−C0 stays within about 0.01 in median F at q=1024.

The axial medians say why the scale is large. At N12 L1, |σ ∂E| is about 0.43 for gamma, 0.43 for s1, and 0.002 for s3. s3 hits the clip of 4. Freezing that scale and the gain, while discarding the per-start calibration that C0 uses, is a different optimizer, and on the primary cell it is a worse one.

## Scalar Nov3 against scalar QAOA

Holdout, C0, q=1024. L1: N9 median F 0.693 vs 0.575, ΔF +0.122, 4/4, median ΔE −0.202. N12 median F 0.631 vs 0.488, ΔF +0.138, 4/4, median ΔE −0.339. L3: N9 median ΔF +0.034 with median ΔE +0.015 (CI crosses 0). N12 median ΔF −0.033 and median ΔE +0.108, 4/4 higher energy for Nov3. The L1 edge is present on this seed band. The L3 comparison is mixed and, at N12, favors QAOA on energy. These seeds are not GR's, and the numbers are not a change to that benchmark.

## What not to do next inside this stream

Do not adopt C1. Do not rerun the holdout under a new clip. Do not start N15. A milder scale, or a per-start gain with only s3 rescaled, would be a new preregistration and needs a seed band that is not 10010–10017. arXiv:2204.02923 is not that preregistration: its entangler is a free matrix under natural gradient, and natural gradient is outside this stream.
