# L3TS — FAIL. Freezing Uc mid-run does not recover L=3. GR unpatched.

Dev-only. N=15 L=3 Q=3072. Exclusive unread seeds **10360–10367**. Original Nov3 shared-Uc + SNAP shells. No pair-phase. JOINT and QAOA are unmodified FT v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`). TWO_STAGE split was frozen before fits: first Q=1536 joint center-4 (T=186, 4 starts); freeze Uc at the single best-so-far; remaining Q=1536 shell-only (T=378, 2 equal blocks, method_id 80). Energy selects. F post hoc.

**Gate FAIL (both primary and secondary). Reject. Do not patch GR.**

| contrast | mean ΔE | CI | wins | mean F a | mean F b | result |
|---|---|---|---|---|---|---|
| TWO_STAGE − QAOA (primary) | +0.029 | [−0.086, +0.124] | 3/8 | 0.700 | 0.695 | **FAIL** (CI not entirely <0) |
| TWO_STAGE vs JOINT (secondary) | mean E −14.240 vs −14.249 | — | — | 0.700 | 0.721 | **FAIL** (TWO_STAGE worse) |
| JOINT − QAOA (report) | +0.019 | [−0.092, +0.117] | 3/8 | 0.721 | 0.695 | not a gate |

Primary: TWO_STAGE does not beat matched QAOA p=3. Mean ΔE is positive (Nov3 higher energy). CI covers 0. Same L=3 death as the center-4 depth curve; spending the second half of Q on shells with Uc frozen does not recover it.

Secondary: TWO_STAGE mean energy is slightly worse than full-budget JOINT (−14.240 vs −14.249). Freezing Uc after a half-budget joint stage is not better than keeping all params live for the full Q. JOINT itself also fails to beat QAOA at L=3, as in FTREP D/D18.

24/24 fits ok. B* rejects 0. Charged 73728. Outside-cap 0. Recipe hash matched. Local R25 10344+, FT pair-phase 10328–35, LE ≤10327, holdout, validation, and final unused. No adoption, no GR patch, no more center-4 depth-curve sampling. Stop after this matrix.
