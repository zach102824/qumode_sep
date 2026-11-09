# N21L1 — N=21 L=1 ESTIMATE_NEGATIVE_CI; GR unpatched

Shallow scaling, not adoption. Same unmodified FT v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10432–10439**. N=21 L=1 Q=1024 T=122. Four equal SPSA blocks, frozen gains, center-4 starts. Nov3 vs QAOA p=1. Energy selects. F post hoc. Original Nov3, SPSA only, entropy-peak B*, equal-layer p=L. No Nov9/SYK. No L2+. No start-count / freeze-order / shell-unshare (those levers closed at PR99/101/102).

**Label N21-L1 ESTIMATE_NEGATIVE_CI.** Mean ΔE CI entirely <0 on n=8. Not a GR write. Not an adoption.

| contrast | mean ΔE | CI | wins | mean F Nov3 | mean F QAOA | ΔE/N |
|---|---|---|---|---|---|---|
| Nov3 − QAOA (L=1 Q=1024) | −0.406 | [−0.548, −0.292] | 8/8 | 0.565 | 0.446 | **−0.0193** |

ΔE/N vs local: R25 N18 −0.0229, R16 −0.0184. This N=21 value (−0.0193) sits between them with the same sign. The L=1 Nov3 energy edge per spin does not collapse between N=18 and N=21.

| seed | B* | ΔE | ΔE/N | F Nov3 | F QAOA |
|---|---|---|---|---|---|
| 10432 | 0.247 | −0.839 | −0.0400 | 0.075 | 0.034 |
| 10433 | 0.727 | −0.445 | −0.0212 | 0.594 | 0.447 |
| 10434 | 0.860 | −0.337 | −0.0161 | 0.658 | 0.529 |
| 10435 | 0.930 | −0.253 | −0.0120 | 0.636 | 0.552 |
| 10436 | 0.831 | −0.221 | −0.0105 | 0.675 | 0.566 |
| 10437 | 0.846 | −0.528 | −0.0251 | 0.613 | 0.428 |
| 10438 | 0.810 | −0.305 | −0.0145 | 0.653 | 0.532 |
| 10439 | 0.980 | −0.321 | −0.0153 | 0.613 | 0.483 |

Seed 10432 is the B* outlier (0.247 vs 0.73–0.98) with a near-degenerate odd gap (1.7×10⁻⁶). It still passed B* hard checks and is a Nov3 energy win. Removing it would not change the CI-entirely-<0 label.

Post-hoc F: mean ΔF +0.118 [+0.090, +0.145], 8/8 Nov3 higher. F never selects.

**Memory.** Local OOM was ~2 GiB/model ×8. This run used **2 workers, BLAS1**. No OOM, no serial fallback. B* peak RSS 1245060 KiB (~1.19 GiB/worker); fit peak 951028 KiB (~0.91 GiB/worker). Do not treat this as license for 8-wide N=21.

16/16 fits ok. B* rejects 0. Charged 16384. Outside-cap 0. Recipe hash matched. Local R26 10416–31, LE ≤10407, FT ≤10415 / 10368–75, holdout 10144–51, validation, and final unused. No L2+, no GR patch, no candidacy, no final.
