# L3SF — FAIL. Shell-first then Uc is worse than JOINT and QAOA. GR unpatched.

Dev-only. N=15 L=3 Q=3072. Exclusive unread seeds **10368–10375** (window 10360–10375; 10360–67 were L3TS; 10376+ unused). Original Nov3 shared-Uc + per-layer SNAP shells. No pair-phase, no per-cavity SNAP. Reverse of L3TS (which was joint-then-shells). JOINT and QAOA are unmodified FT v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`). Split frozen before fits: Q=1536 shell-only (Uc frozen at each center-4 x0, T=186, 4 starts, method_id 82); freeze shells; Q=1536 Uc-only (T=378, 2 blocks, method_id 83). Energy selects. F post hoc.

**Gate FAIL (both primary and secondary). Reject. Do not patch GR.**

| contrast | mean ΔE | CI | wins | mean F a | mean F b | result |
|---|---|---|---|---|---|---|
| SHELL_FIRST − QAOA (primary) | +0.142 | [+0.113, +0.173] | 0/8 | 0.667 | 0.709 | **FAIL** (CI entirely >0) |
| SHELL_FIRST vs JOINT (secondary) | mean E −12.565 vs −12.645 | — | — | 0.667 | 0.707 | **FAIL** (worse) |
| JOINT − QAOA (report) | +0.062 | [−0.013, +0.129] | 2/8 | 0.707 | 0.709 | not a gate |

Primary: SHELL_FIRST loses to matched QAOA p=3 on every seed. CI lies entirely above 0. This is not an inconclusive L=3 death; the reverse schedule is actively worse.

Secondary: worse than full-budget JOINT. Fitting shells with Uc pinned at the start, then moving only the 3-dim Uc, does not recover depth. Together with L3TS (joint-then-shells also FAIL), neither freeze order recovers L=3. Shared-Uc Nov3 vs QAOA p=L at L≥3 remains absent on this family of SPSA splits.

24/24 fits ok. B* rejects 0. Charged 73728. Outside-cap 0. Recipe hash matched. D18 10320–27, FT pair-phase 10328–35, R25 10344–59, L3TS 10360–67, holdout, validation, and final unused. No adoption, no GR patch, no CANDIDACY revival, no more center-4 L2 copies. Stop after this matrix.
