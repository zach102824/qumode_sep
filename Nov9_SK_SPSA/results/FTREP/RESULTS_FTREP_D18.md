# FTREP D18: N=18 center-4 depth curve L∈{2,3} (+L=4 if run)

Same alloc.py `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Seeds 10320–10327. N=18. Q=1024·L. Four equal blocks, frozen gains, center-4. Energy selects. F post hoc.
Characterization, not adoption. GR not patched regardless of L=2.

Runs 48 ok 48 failures 0. Charged 147456. Outside cap 0. Rejected B* 0. Missing 0. L4 finished True.

Vs N15 (FTREP D): **still between L2 and L3**.

| L | Q | mean ΔE med [CI] tails wins | mean F Nov3 | mean F QAOA | label |
|---|---|---|---|---|---|
| 2 | 2048 | -0.05280 med -0.08641 [-0.10391,-0.00105] tails [-0.15479,+0.04971] 5/8 | 0.6572 | 0.6223 | **present** |
| 3 | 3072 | +0.00827 med +0.05304 [-0.10113,+0.09429] tails [-0.32796,+0.15201] 3/8 | 0.6712 | 0.6570 | **absent/inconclusive** |
| 4 | 4096 | +0.03406 med -0.01819 [-0.05064,+0.12783] tails [-0.12783,+0.25308] 4/8 | 0.7044 | 0.6997 | **absent/inconclusive** |

B* wall 106.59641981124878. Required fit wall 781.332857131958. L4 fit wall 908.3022012710571. Sum cell wall 6182.2s. Sum cell CPU 6176.4s. Peak RSS KiB 269568.
