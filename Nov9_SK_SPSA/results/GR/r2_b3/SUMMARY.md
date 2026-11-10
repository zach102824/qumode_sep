# R2 B3 matched Nov3rd vs QAOA at B*

Entropy case only. Recipe `N3X_matched_v1`: 4 starts, caps 128 and 1024, every initial, calibration, and selection energy call inside the cap, monitoring 0. Fidelity is post hoc on the selected state.
This is not the local cumulative continuation (2048/3072 queries) and it is not the earlier b = 1 / 1.5 reference.

## External local feedback, different recipe

Four dev instances per N, L = 3, cumulative charged queries, energy selection. Not merged with the table below. No scaling claim from these four.

| N | Nov3 F | QAOA F | cumulative queries |
|---:|---:|---:|---:|
| 9 | 0.8326 | 0.7823 | 2048 |
| 12 | 0.7753 | 0.7700 | 3072 |
| 15 | 0.7033 | 0.7468 | 2048 |

Local Nov3 at N = 9: all 4 instances F ≥ 0.7 and 3 of 4 F ≥ 0.8. Local’s next step is a common 3072/4096 budget and fresh N = 15 dev seeds. That extension is not this run.

## This confirmation

Seeds 10000–10007. Same J and same B* for both methods. Nov3rd has one shared γ, two feature angles, and 3 shell phases per layer (6 controls at L = 1). The shell chart is the polynomial `s1 p + s2 C(p,2) + s3 [p=3]`, not the Walsh `t1,t2,t3` chart. QAOA has 2L independent angles, `exp(-i γ H_ZZ) exp(-i β Σ X)`.
Per start at cap 1024: 1 init + 8 calibration + 122×2 iteration + 3 selection = 256, times 4 = 1024. At cap 128 the iteration count is 10. The selectable points are the init, the last iterate, the mean of the last min(64, T) iterates, and the midpoint iterate. Lowest energy wins. Gains use c = 0.1 and a median initial step of 0.5σ.

| N | L | cap | Nov3 med F | QAOA med F | med ΔF | F wins | Nov3 med E | QAOA med E | med ΔE | Nov3 η | Nov3 η_E | F+ | frac Nov3 ≥0.7 | ≥0.8 | frac QAOA ≥0.7 | ≥0.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 128 | 0.6678 | 0.6254 | 0.0507 | 7/8 | -2.6337 | -2.6792 | -0.0470 | 0.473 | 0.712 | 0.308 | 0.50 | 0.00 | 0.25 | 0.00 |
| 6 | 1 | 1024 | 0.7845 | 0.6401 | 0.1451 | 8/8 | -2.8617 | -2.7316 | -0.1878 | 0.674 | 0.796 | 0.308 | 0.88 | 0.38 | 0.25 | 0.00 |
| 6 | 2 | 128 | 0.6858 | 0.6640 | 0.0038 | 4/8 | -2.6778 | -2.7408 | 0.0323 | 0.554 | 0.709 | 0.308 | 0.38 | 0.00 | 0.25 | 0.00 |
| 6 | 2 | 1024 | 0.8450 | 0.7533 | 0.0739 | 8/8 | -2.8745 | -2.8077 | -0.0866 | 0.745 | 0.871 | 0.308 | 1.00 | 0.88 | 0.75 | 0.12 |
| 6 | 3 | 128 | 0.6429 | 0.6867 | -0.0241 | 3/8 | -2.6248 | -2.7396 | 0.0561 | 0.443 | 0.630 | 0.308 | 0.12 | 0.00 | 0.25 | 0.00 |
| 6 | 3 | 1024 | 0.8653 | 0.7541 | 0.0691 | 7/8 | -2.8936 | -2.8130 | -0.0581 | 0.796 | 0.874 | 0.308 | 0.88 | 0.88 | 0.75 | 0.25 |
| 9 | 1 | 128 | 0.5672 | 0.5621 | 0.0008 | 4/8 | -6.1800 | -6.1014 | 0.0174 | 0.409 | 0.629 | 0.291 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 1 | 1024 | 0.7098 | 0.5829 | 0.1397 | 8/8 | -6.4453 | -6.1463 | -0.2365 | 0.592 | 0.779 | 0.291 | 0.62 | 0.00 | 0.00 | 0.00 |
| 9 | 2 | 128 | 0.5933 | 0.6189 | -0.0113 | 3/8 | -5.9948 | -6.1885 | 0.1524 | 0.442 | 0.631 | 0.291 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 2 | 1024 | 0.7425 | 0.6572 | 0.0962 | 8/8 | -6.5179 | -6.2989 | -0.1508 | 0.630 | 0.811 | 0.291 | 1.00 | 0.00 | 0.25 | 0.00 |
| 9 | 3 | 128 | 0.4972 | 0.6149 | -0.1151 | 1/8 | -5.8080 | -6.2276 | 0.2949 | 0.285 | 0.535 | 0.291 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 3 | 1024 | 0.7352 | 0.7175 | 0.0500 | 5/8 | -6.5009 | -6.4152 | -0.0499 | 0.630 | 0.784 | 0.291 | 0.75 | 0.12 | 0.62 | 0.00 |
| 12 | 1 | 128 | 0.4700 | 0.4383 | 0.0275 | 5/8 | -8.0461 | -8.1251 | 0.0097 | 0.343 | 0.594 | 0.206 | 0.00 | 0.00 | 0.00 | 0.00 |
| 12 | 1 | 1024 | 0.6586 | 0.5209 | 0.1403 | 8/8 | -8.6312 | -8.3006 | -0.3202 | 0.555 | 0.793 | 0.206 | 0.25 | 0.00 | 0.00 | 0.00 |
| 12 | 2 | 128 | 0.4706 | 0.5280 | -0.0174 | 4/8 | -7.9012 | -8.4571 | 0.0605 | 0.411 | 0.704 | 0.206 | 0.00 | 0.00 | 0.00 | 0.00 |
| 12 | 2 | 1024 | 0.6729 | 0.6014 | 0.0543 | 8/8 | -8.7307 | -8.6307 | -0.1342 | 0.581 | 0.794 | 0.206 | 0.38 | 0.00 | 0.25 | 0.00 |
| 12 | 3 | 128 | 0.3895 | 0.5508 | -0.1022 | 1/8 | -7.7708 | -8.4691 | 0.4383 | 0.268 | 0.532 | 0.206 | 0.00 | 0.00 | 0.12 | 0.00 |
| 12 | 3 | 1024 | 0.6179 | 0.6444 | 0.0268 | 5/8 | -8.6748 | -8.7423 | 0.0094 | 0.546 | 0.800 | 0.206 | 0.12 | 0.12 | 0.38 | 0.00 |

## Tails at cap 1024

| N | L | method | min F | max F | min η_E |
|---:|---:|---|---:|---:|---:|
| 6 | 1 | Nov3rd | 0.6924 | 0.8291 | 0.721 |
| 6 | 1 | QAOA | 0.4061 | 0.7490 | 0.520 |
| 6 | 2 | Nov3rd | 0.7583 | 0.9106 | 0.812 |
| 6 | 2 | QAOA | 0.4423 | 0.8231 | 0.573 |
| 6 | 3 | Nov3rd | 0.6915 | 0.9284 | 0.791 |
| 6 | 3 | QAOA | 0.6360 | 0.9219 | 0.718 |
| 9 | 1 | Nov3rd | 0.6455 | 0.7586 | 0.713 |
| 9 | 1 | QAOA | 0.4808 | 0.6178 | 0.531 |
| 9 | 2 | Nov3rd | 0.7065 | 0.7959 | 0.752 |
| 9 | 2 | QAOA | 0.5638 | 0.7158 | 0.616 |
| 9 | 3 | Nov3rd | 0.6503 | 0.8271 | 0.741 |
| 9 | 3 | QAOA | 0.6256 | 0.7696 | 0.677 |
| 12 | 1 | Nov3rd | 0.2360 | 0.7254 | 0.641 |
| 12 | 1 | QAOA | 0.1647 | 0.6017 | 0.560 |
| 12 | 2 | Nov3rd | 0.2845 | 0.7591 | 0.711 |
| 12 | 2 | QAOA | 0.2389 | 0.7387 | 0.669 |
| 12 | 3 | Nov3rd | 0.3939 | 0.8006 | 0.713 |
| 12 | 3 | QAOA | 0.2561 | 0.7667 | 0.700 |

Charged optimization queries: 165888 = 1152 init + 9216 calibration + 152064 iteration + 3456 selection. Monitoring: 0. Every stored run totals its cap. The 432 curvature calls are outside that cap and do not select.

At L = 1 and the zero parameter vector, the 2×2 energy Hessian in (γ, s1) for Nov3rd and (γ, β) for QAOA has a negative smaller eigenvalue on every instance (median about −5.5/−9.4/−12.9 for Nov3rd and −13/−23/−31 for QAOA at N = 6/9/12). That is this chart’s curvature, not the full Hessian in the LE note.

The read rule in `NEXT_IDEAS_R2.md` asks for a median L = 1 fidelity advantage of at least 0.05 on at least 6 of 8 seeds at N = 12 and N = 15. At N = 12, cap 1024, the paired median ΔF is 0.140 and Nov3rd wins 8/8. N = 15 was not optimized. One Nov3rd L = 1 energy at N = 15 and b = 0.5, which is not a B* instance, took 0.009 s, about 10 s projected for 1024 queries. A fresh-seed N = 15 extension is not this recipe.

At cap 1024 and N = 12, L = 3, the paired median ΔF is +0.027 (5/8) while the unpaired median fidelities are 0.618 and 0.644. Paired and unpaired summaries are not the same number.

Imaginary-axis displacement is LE’s assignment. The real-α basin audit is FT’s assignment.

