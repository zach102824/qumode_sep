# L3TS: N=15 L=3 two-stage SPSA vs JOINT vs QAOA p=3

JOINT/QAOA: unmodified v4 center-4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
TWO_STAGE: Q=1536 joint center-4 (T=186, 4 starts) then freeze Uc; Q=1536 shell-only (T=378, 2 blocks).
Seeds 10360–10367. N=15 L=3 Q=3072. Energy selects. F post hoc. No pair-phase. No GR.

Runs 24 ok 24 failures 0. Charged 73728. Outside cap 0. Rejected B* 0. Missing 0.

**Primary** TWO_STAGE−QAOA: +0.02855 med +0.05558 [-0.08605,+0.12363] tails [-0.31912,+0.20888] 3/8 FAIL
**Secondary** TWO_STAGE mean E -14.239520 vs JOINT -14.249466 FAIL
**Gate FAIL.** GR not patched.

mean F TWO_STAGE 0.7003 JOINT 0.7215 QAOA 0.6951
JOINT−QAOA (report): +0.01861 med +0.06177 [-0.09217,+0.11735] tails [-0.29661,+0.23902] 3/8

B* wall 9.617160081863403. Fit wall 86.47104573249817. Sum cell wall 326.7s. Sum cell CPU 326.4s. Peak RSS KiB 190692.
