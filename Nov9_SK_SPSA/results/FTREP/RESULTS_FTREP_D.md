# FTREP D: N=15 center-4 depth curve L∈{2,3,4}

Same alloc.py `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Seeds 10264–10271. N=15 only. Q=1024·L. Four equal blocks, frozen gains, center-4. Energy selects. F post hoc.
Characterization, not adoption. GR not patched regardless of L=2.

Runs 48 ok 48 failures 0. Charged 147456. Outside cap 0. Rejected B* 0. Missing 0.

| L | Q | mean ΔE med [CI] tails wins | mean F Nov3 | mean F QAOA | label |
|---|---|---|---|---|---|
| 2 | 2048 | -0.07423 med -0.06494 [-0.14300,-0.00577] tails [-0.23323,+0.05909] 6/8 | 0.6607 | 0.6229 | **present** |
| 3 | 3072 | +0.05936 med +0.06357 [-0.03228,+0.14593] tails [-0.19803,+0.25154] 2/8 | 0.6924 | 0.7024 | **absent/inconclusive** |
| 4 | 4096 | +0.05589 med +0.08147 [-0.09067,+0.19474] tails [-0.26438,+0.34193] 3/8 | 0.6846 | 0.6905 | **absent/inconclusive** |

B* wall 9.865643978118896. Fit wall 166.68337655067444. Sum cell wall 626.4s. Sum cell CPU 625.9s. Peak RSS KiB 193044.
