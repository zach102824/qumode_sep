# L3SF: N=15 L=3 shell-first two-stage vs JOINT vs QAOA p=3

JOINT/QAOA: unmodified v4 center-4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
SHELL_FIRST: Q=1536 shell-only center-4 (T=186, 4 starts, Uc frozen at x0) then freeze shells; Q=1536 Uc-only (T=378, 2 blocks).
Seeds 10368–10375. N=15 L=3 Q=3072. Reverse of L3TS. Energy selects. F post hoc. No pair-phase. No GR.

Runs 24 ok 24 failures 0. Charged 73728. Outside cap 0. Rejected B* 0. Missing 0.

**Primary** SHELL_FIRST−QAOA: +0.14176 med +0.14182 [+0.11325,+0.17345] tails [+0.07947,+0.23107] 0/8 FAIL
**Secondary** SHELL_FIRST mean E -12.565432 vs JOINT -12.644853 FAIL
**Gate FAIL.** GR not patched.

mean F SHELL_FIRST 0.6671 JOINT 0.7073 QAOA 0.7085
JOINT−QAOA (report): +0.06234 med +0.10008 [-0.01273,+0.12948] tails [-0.12804,+0.17971] 2/8

B* wall 10.27722978591919. Fit wall 84.75284886360168. Sum cell wall 320.8s. Sum cell CPU 320.5s. Peak RSS KiB 190716.
