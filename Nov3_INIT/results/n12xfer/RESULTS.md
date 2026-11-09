# N=12 L=1 parameter concentration and transfer

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=12 L=1 Q=1024 T=122. Train 10664–10671, test 10672–10679. Mean and median frozen from train before test. Primary aggregator is mean. Same protocol as LE PR116, different N. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Transfer evals 32 ok 32. Charged 32800 (independent 32768 + transfer 32). Failures 0.

**Label TRANSFER_MEAN_NEGATIVE_CI.**

1 train independent Nov3−QAOA: -0.26291 med -0.22472 [-0.34255,-0.21227] tails [-0.52852,-0.19435] 8/8
2a test shared-mean Nov3−QAOA (primary): -1.09345 med -1.09691 [-1.19623,-0.99068] tails [-1.34226,-0.83275] 8/8
2b test shared-median Nov3−QAOA (descriptive): -2.10237 med -2.10370 [-2.36339,-1.86189] tails [-2.81321,-1.60616] 8/8
3 test ceiling independent Nov3−QAOA: -0.33252 med -0.29447 [-0.40434,-0.26941] tails [-0.53458,-0.23099] 8/8
4 transfer gap mean vs independent Nov3 (descriptive): +1.07746 med +0.95950 [+0.92753,+1.25613] tails [+0.75051,+1.57210] 0/8

Train Nov3 n=8; gamma mean -0.1765 std 0.2902 range [-0.3752,+0.3204]; theta_K mean -0.3796 std 0.0573 range [-0.4965,-0.3019]; theta_M mean -0.0154 std 0.0152 range [-0.0454,-0.0009]; s1 mean -0.4316 std 0.8265 range [-0.9301,+0.9666]; s2 mean -0.0003 std 0.1454 range [-0.3504,+0.1176]; s3 mean -0.0318 std 0.2588 range [-0.3410,+0.3817]
Train QAOA n=8; gamma mean +0.0157 std 0.3317 range [-0.3467,+0.3801]; beta mean +0.1952 std 0.5523 range [-0.4034,+1.1695]

Descriptive comparison to PR116 N=15 train stds (different seeds; not paired; n=8 caveat on both):

- θ_K std N=12 0.0573 vs N=15 0.0511
- θ_M std N=12 0.0152 vs N=15 0.0233
- s1 std N=12 0.8265 vs N=15 0.8058
- QAOA β std N=12 0.5523 vs N=15 0.4015

New energy queries 32800. Historical 3014656. Total 3047456.
Classical B* wall 10.7s.

Median is frozen but is not the primary gate. Do not switch primary to median. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. Exploratory; not adoption.

Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.
