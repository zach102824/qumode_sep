# L3S8: N=15 L=3 eight equal starts vs center-4 vs QAOA p=3

C4: unmodified v4 `run_equal4` `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
S8: unmodified v4 `run_spread8` T=186, 8 starts. QAOA matches 8-start policy (not 4-start vs 8-start).
Seeds 10400–10407. N=15 L=3 Q=3072. Energy selects. F post hoc. No two-stage. No pair-phase. No GR.

Runs 24 ok 24 failures 0. Charged 73728. Outside cap 0. Rejected B* 0. Missing 0.

**Primary** S8−QAOA: -0.00670 med +0.03983 [-0.11921,+0.09073] tails [-0.31611,+0.14729] 4/8 FAIL
**Secondary** S8 mean E -11.636353 vs C4 -11.732694 FAIL
**Gate FAIL.** GR not patched.

mean F S8 0.6109 C4 0.6477 QAOA 0.5990
C4−QAOA (report): -0.10304 med -0.04803 [-0.22818,-0.00227] tails [-0.45089,+0.06275] 6/8
S8 start0 selected 1/8; QAOA start0 1/8.

B* wall 10.65723967552185. Fit wall 86.71579790115356. Sum cell wall 326.4s. Sum cell CPU 326.1s. Peak RSS KiB 193828.
