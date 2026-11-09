# N3INIT v4

Development candidate only. Not a benchmark adoption. QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)` on every layer.
Seeds 10018–10025 are development at N=9/12. N=15 uses the same band because the width is new.

New queries 774144. Historical queries counted once 905216. Total research queries 1679360. Failures 0.

| N | L | q | Q | n | mean E center | median | min | max | mean E equal | mean E QAOA | mean Δ self | self CI | mean Δ pair | pair CI | wins vs equal | wins vs QAOA | mean F center | mean F QAOA | start0 Nov3 | start0 QAOA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 9 | 4 | 512 | 2048 | 8 | -7.1041 | -7.1022 | -8.9189 | -5.3875 | -7.0921 | -7.1783 | -0.0120 | [-0.036, +0.000] | +0.0743 | [+0.011, +0.147] | 1/8 | 2/8 | 0.785 | 0.810 | 1/8 | 1/8 |
| 9 | 4 | 1024 | 4096 | 8 | -7.1417 | -7.1341 | -8.9456 | -5.4076 | -7.1393 | -7.1941 | -0.0023 | [-0.007, +0.000] | +0.0524 | [-0.005, +0.123] | 1/8 | 4/8 | 0.804 | 0.818 | 1/8 | 1/8 |
| 9 | 6 | 512 | 3072 | 8 | -7.0881 | -6.9656 | -9.3792 | -5.4643 | -7.0329 | -7.2309 | -0.0551 | [-0.156, +0.049] | +0.1428 | [-0.015, +0.299] | 4/8 | 3/8 | 0.785 | 0.834 | 5/8 | 1/8 |
| 9 | 6 | 1024 | 6144 | 8 | -7.1647 | -7.0621 | -9.3946 | -5.4674 | -7.1212 | -7.2454 | -0.0435 | [-0.121, +0.019] | +0.0807 | [-0.032, +0.189] | 4/8 | 3/8 | 0.830 | 0.842 | 5/8 | 1/8 |
| 12 | 4 | 512 | 2048 | 8 | -10.4188 | -9.8375 | -14.3483 | -7.6730 | -10.3630 | -10.5192 | -0.0558 | [-0.136, +0.008] | +0.1005 | [+0.032, +0.168] | 3/8 | 2/8 | 0.746 | 0.767 | 3/8 | 1/8 |
| 12 | 4 | 1024 | 4096 | 8 | -10.4597 | -9.8668 | -14.3735 | -7.6952 | -10.4348 | -10.5553 | -0.0249 | [-0.079, +0.013] | +0.0956 | [+0.048, +0.144] | 3/8 | 1/8 | 0.757 | 0.786 | 4/8 | 1/8 |
| 12 | 6 | 512 | 3072 | 8 | -10.3964 | -9.6586 | -14.1012 | -7.4900 | -10.3822 | -10.5757 | -0.0142 | [-0.141, +0.131] | +0.1793 | [-0.011, +0.367] | 3/8 | 4/8 | 0.745 | 0.798 | 3/8 | 2/8 |
| 12 | 6 | 1024 | 6144 | 8 | -10.4804 | -9.8553 | -14.1795 | -7.5635 | -10.4891 | -10.6085 | +0.0087 | [-0.109, +0.143] | +0.1281 | [-0.016, +0.278] | 3/8 | 4/8 | 0.778 | 0.816 | 3/8 | 3/8 |
| 15 | 1 | 512 | 512 | 8 | -12.2136 | -12.7931 | -13.9671 | -10.1608 | -12.2245 | -11.9648 | +0.0109 | [+0.000, +0.025] | -0.2488 | [-0.325, -0.159] | 0/8 | 7/8 | 0.602 | 0.499 | 1/8 | 0/8 |
| 15 | 1 | 1024 | 1024 | 8 | -12.2929 | -12.8612 | -14.1601 | -10.1713 | -12.2938 | -11.9648 | +0.0009 | [+0.000, +0.003] | -0.3281 | [-0.362, -0.290] | 0/8 | 8/8 | 0.633 | 0.499 | 0/8 | 0/8 |

Paired median, min, and max are the seedwise energy differences. A win is a strict energy improvement. Ties use absolute difference at most 1e-8.

| N | L | q | median Δ self | min Δ self | max Δ self | ties vs equal | median Δ pair | min Δ pair | max Δ pair | ties vs QAOA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 4 | 512 | +0.0000 | -0.0958 | +0.0000 | 7 | +0.0150 | -0.0232 | +0.2756 | 0 |
| 9 | 4 | 1024 | +0.0000 | -0.0187 | +0.0000 | 7 | -0.0027 | -0.0332 | +0.2486 | 0 |
| 9 | 6 | 512 | -0.0423 | -0.3291 | +0.2240 | 3 | +0.1653 | -0.1707 | +0.4227 | 0 |
| 9 | 6 | 1024 | -0.0305 | -0.2569 | +0.1149 | 3 | +0.0917 | -0.1792 | +0.2895 | 0 |
| 12 | 4 | 512 | +0.0000 | -0.2656 | +0.0463 | 4 | +0.1002 | -0.0573 | +0.2453 | 0 |
| 12 | 4 | 1024 | +0.0000 | -0.1967 | +0.0485 | 3 | +0.0979 | -0.0056 | +0.1918 | 0 |
| 12 | 6 | 512 | +0.0000 | -0.2823 | +0.4377 | 2 | +0.1354 | -0.1138 | +0.5905 | 0 |
| 12 | 6 | 1024 | +0.0000 | -0.2499 | +0.3943 | 2 | +0.0971 | -0.1174 | +0.4626 | 0 |
| 15 | 1 | 512 | +0.0000 | +0.0000 | +0.0576 | 6 | -0.2921 | -0.3751 | +0.0033 | 0 |
| 15 | 1 | 1024 | +0.0000 | +0.0000 | +0.0068 | 7 | -0.3345 | -0.4078 | -0.2300 | 0 |

## Pooled N=9 and N=12

Seeds are resampled inside each N. N is held. This is the preregistered depth pool, computed after every single-N interval.

| L | q | n | mean Δ self | median | CI | mean Δ pair | median | CI | center_helps | beats_fair_qaoa |
|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| 4 | 512 | 16 | -0.0339 | +0.0000 | [-0.076, +0.001] | +0.0874 | +0.0635 | [+0.040, +0.139] | `negative_point_ci_includes_0` | `positive_ci_excludes_0` |
| 4 | 1024 | 16 | -0.0136 | +0.0000 | [-0.041, +0.006] | +0.0740 | +0.0565 | [+0.037, +0.115] | `negative_point_ci_includes_0` | `positive_ci_excludes_0` |
| 6 | 512 | 16 | -0.0347 | +0.0000 | [-0.120, +0.055] | +0.1611 | +0.1548 | [+0.041, +0.291] | `negative_point_ci_includes_0` | `positive_ci_excludes_0` |
| 6 | 1024 | 16 | -0.0174 | +0.0000 | [-0.086, +0.057] | +0.1044 | +0.0917 | [+0.013, +0.194] | `negative_point_ci_includes_0` | `positive_ci_excludes_0` |

## Read flags

- N=9 L=4 q=512: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_ci_excludes_0`.
- N=9 L=4 q=1024: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_point_ci_includes_0`.
- N=9 L=6 q=512: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_point_ci_includes_0`.
- N=9 L=6 q=1024: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_point_ci_includes_0`.
- N=12 L=4 q=512: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_ci_excludes_0`.
- N=12 L=4 q=1024: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_ci_excludes_0`.
- N=12 L=6 q=512: center_helps `negative_point_ci_includes_0`; beats_fair_qaoa `positive_point_ci_includes_0`.
- N=12 L=6 q=1024: center_helps `positive_point_ci_includes_0`; beats_fair_qaoa `positive_point_ci_includes_0`.
- N=15 L=1 q=512: center_helps `positive_point_ci_includes_0`; beats_fair_qaoa `negative_ci_excludes_0`.
- N=15 L=1 q=1024: center_helps `positive_point_ci_includes_0`; beats_fair_qaoa `negative_ci_excludes_0`.

No depth cell, and no N=9/12 pool, has a confirmed center-versus-equal4 energy edge. The median paired self-difference is 0 in every depth row: most seeds tie because a later start is selected.
Pooled over N=9 and N=12, mean d_pair is positive at L=4 and L=6 for both budgets, and each of those four intervals excludes 0. On this development band the derived QAOA center has lower energy than nov3_center4. That comparison is energy only. It does not adopt either arm into the benchmark.
At N=15 L1, mean d_pair is negative and the interval excludes 0 at both budgets, with 7/8 and 8/8 strict energy wins for Nov3. Start 0 is selected once at q=512 and never at q=1024, and every nonzero self-difference is nonnegative, so the coherent center is not what produces that shallow edge.
The N=15 QAOA selected energies at q=512 and q=1024 agree to about 1e-5 or closer on every seed, so the printed means match. The parameter vectors are not copies.

Cell time 1099.9s on the recorded cells. Workers 4. Procedural cap of each cell equals its own Q.
