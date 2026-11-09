# FTREP L3: FT v4 center-4 at N∈{12,15} L=3 Q=3072

Same alloc.py `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Seeds 10192–10199. Four equal blocks, T=378, frozen gains, center-4. Energy selects. F post hoc. No N18 L3. GR not patched.

Runs 32 ok 32 failures 0. Charged 98304. Outside cap 0. Rejected B* 0. Missing 0.

Gate L3-15 mean ΔE +0.06912 med +0.05463 [-0.06578,+0.21750] tails [-0.22450,+0.45345] 4/8: **fail**.
Gate L3-12 mean ΔE -0.00328 med +0.03677 [-0.11586,+0.07053] tails [-0.37024,+0.10942] 2/8: **fail**.
Decision: **FAIL_l3_inconclusive**.

## Nov3 − QAOA (center4 L=3)

| N | mean ΔE med [CI] tails wins(Nov3) | Δ(E−E0)/N | ΔF wins | start0 Nov3 |
|---|---|---|---|---|
| 12 | -0.00328 med +0.03677 [-0.11586,+0.07053] tails [-0.37024,+0.10942] 2/8 | -0.00027 med +0.00306 [-0.00987,+0.00585] tails [-0.03085,+0.00912] | +0.00809 med -0.00044 [-0.02501,+0.05707] tails [-0.04314,+0.15677] 4/8 | 4/8 |
| 15 | +0.06912 med +0.05463 [-0.06578,+0.21750] tails [-0.22450,+0.45345] 4/8 | +0.00461 med +0.00364 [-0.00439,+0.01430] tails [-0.01497,+0.03023] | -0.00252 med -0.00759 [-0.06871,+0.06167] tails [-0.16593,+0.12612] 4/8 | 3/8 |

B* wall 28.16217851638794. Fit wall 59.476770877838135. Sum cell wall 219.5s. Sum cell CPU 219.3s. Peak RSS KiB 188636.
