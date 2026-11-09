# L3TIE — FAIL. Layer-tied SNAP does not recover L=3. GR unpatched.

Dev-only. N=15 L=3 Q=3072. Exclusive unread seeds **10552–10559** (not R31 10536–51, R30 10504–19, FT 10520–35 / 10568+, LE 10432–39, L3S8 10400–07, holdout/validation/final). 10560–67 held unused in the same band. Original shared-Uc Nov3. Not start-count. Not freeze-order. Not per-cavity shell-unshare. Not two-stage. JOINT and QAOA are unmodified FT v4 `run_equal4` (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`). TIE uses v4 `run_equal4` L=1 geometry (dim 6, T=378) expanded to L=3 by repeating one SNAP triple. Energy selects. F post hoc.

**Gate FAIL (both primary and secondary). Reject. Do not patch GR.**

| contrast | mean ΔE | CI | wins | mean F a | mean F b | result |
|---|---|---|---|---|---|---|
| TIE − QAOA (primary, both dim 6) | +0.132 | [+0.011, +0.313] | 1/8 | 0.648 | 0.674 | **FAIL** (CI entirely >0) |
| TIE vs JOINT (secondary) | mean E −12.741 vs −12.846 | — | — | 0.648 | 0.680 | **FAIL** (TIE worse) |
| JOINT − QAOA (report) | +0.027 | [−0.036, +0.093] | 3/8 | 0.680 | 0.674 | L=3 still absent |

Primary: matching QAOA’s parameter count by tying SNAP across layers does not produce a Nov3 energy edge. Mean ΔE is positive; CI is entirely above 0; 1/8. TIE is worse than matched 4-start QAOA.

Secondary: TIE is worse than untied JOINT. Extra per-layer SNAP degrees of freedom are not the L=3 failure mode; forcing SNAP_1=SNAP_2=SNAP_3 hurts.

JOINT−QAOA on this unread band is inconclusive (+0.027, CI covers 0, 3/8), same L=3 death as D / D18 / L3S8 C4. Do not revive candidacy. Do not patch GR.

24/24 fits ok. B* rejects 0. Charged 73728. Outside-cap 0. Recipe hash matched. Seeds 10560–67 unused. No more start-count / freeze-order / shell-unshare / two-stage. Stop after this matrix.
