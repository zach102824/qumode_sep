# L3S8 — FAIL. Eight equal starts do not recover L=3. GR unpatched.

Dev-only. N=15 L=3 Q=3072. Exclusive unread seeds **10400–10407** (not ≤10391, not R25 10344–59, not holdout/validation/final). Original shared-Uc Nov3. No pair-phase. No two-stage freeze-order. C4 is unmodified FT v4 `run_equal4` (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`). S8 is unmodified v4 `run_spread8` (T=Q/16−6=186, 8 starts, start 0 coherent center). QAOA p=3 uses the same 8-start `run_spread8` so the primary comparison is not 8-start Nov3 vs 4-start QAOA. Energy selects. F post hoc.

**Gate FAIL (both primary and secondary). Reject. Do not patch GR.**

| contrast | mean ΔE | CI | wins | mean F a | mean F b | result |
|---|---|---|---|---|---|---|
| S8 − QAOA (primary, both 8-start) | −0.007 | [−0.119, +0.091] | 4/8 | 0.611 | 0.599 | **FAIL** (CI covers 0) |
| S8 vs C4 (secondary) | mean E −11.636 vs −11.733 | — | — | 0.611 | 0.648 | **FAIL** (S8 worse) |
| C4 − QAOA (report; 4-start vs 8-start) | −0.103 | [−0.228, −0.002] | 6/8 | 0.648 | 0.599 | not a gate; start-count mismatch |

Primary: spreading the same Q over eight starts does not produce a Nov3 energy edge vs matched 8-start QAOA. Mean ΔE is near zero; CI covers 0; 4/8.

Secondary: S8 is worse than center-4. Extra starts cost T (186 vs 378). Under-exploration of four centers is not the L=3 failure mode; thinning each block hurts.

The C4−QAOA report line is **not** an L=3 adoption signal: QAOA here is 8-start T=186, C4 is 4-start T=378. Do not revive candidacy. Do not patch GR.

24/24 fits ok. B* rejects 0. Charged 73728. Outside-cap 0. Recipe hash matched. Seeds ≤10391 unused. No more two-stage freeze-order. Stop after this matrix.
