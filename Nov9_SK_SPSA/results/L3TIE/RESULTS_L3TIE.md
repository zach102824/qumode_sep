# L3TIE: N=15 L=3 layer-tied SNAP vs JOINT vs QAOA p=3

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Seeds 10552–10559. N=15 L=3 Q=3072 T=378. Four equal blocks, center-4. Energy selects. F post hoc.
TIE: one SNAP triple repeated L=3, SPSA dim 6. JOINT: per-layer SNAP, dim 12. QAOA: 4-start matched.
Not start-count / freeze-order / shell-unshare / two-stage. No GR.

Runs 24 ok 24 failures 0. Charged 73728. Outside cap 0. Rejected B* 0. Missing 0.

**Primary** TIE−QAOA: +0.13235 med +0.05348 [+0.01065,+0.31324] tails [-0.09751,+0.70880] 1/8 FAIL
**Secondary** TIE mean E -12.740694 vs JOINT -12.846230 FAIL
**Gate FAIL.** GR not patched.

mean F TIE 0.6475 JOINT 0.6804 QAOA 0.6737
JOINT−QAOA (report): +0.02681 med +0.03398 [-0.03635,+0.09235] tails [-0.09677,+0.19654] 3/8

B* wall 9.376810789108276. Fit wall 83.68330645561218. Sum cell wall 314.2s. Sum cell CPU 314.0s. Peak RSS KiB 190280.
