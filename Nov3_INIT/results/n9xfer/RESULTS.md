# N=9 L=1 parameter concentration and transfer

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=9 L=1 Q=1024 T=122. Train 10696–10703, test 10704–10711. Mean and median frozen from train before test. Primary aggregator is mean. Same protocol as LE PR116 / PR118. Completes unsigned-mean ladder N9/N12/N15/N18. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Transfer evals 32 ok 32. Charged 32800 (independent 32768 + transfer 32). Failures 0.

**Label UNSTABLE.**

1 train independent Nov3−QAOA: -0.26125 med -0.26223 [-0.30708,-0.21253] tails [-0.36852,-0.14245] 8/8
2a test shared-mean Nov3−QAOA (primary): +1.22154 med +1.19275 [+1.07765,+1.38280] tails [+0.94429,+1.65738] 0/8
2b test shared-median Nov3−QAOA (descriptive): +1.38864 med +1.36536 [+1.24669,+1.54565] tails [+1.10920,+1.81960] 0/8
3 test ceiling independent Nov3−QAOA: -0.29997 med -0.29455 [-0.33858,-0.26114] tails [-0.39886,-0.20200] 8/8
4 transfer gap mean vs independent Nov3 (descriptive): +1.73268 med +1.67838 [+1.58742,+1.89521] tails [+1.47075,+2.19318] 0/8

Train Nov3 n=8; gamma mean +0.0197 std 0.3708 range [-0.3756,+0.4005]; theta_K mean -0.4509 std 0.0693 range [-0.5375,-0.3410]; theta_M mean +0.0015 std 0.0346 range [-0.0581,+0.0532]; s1 mean -0.0323 std 0.9605 range [-0.9676,+0.9122]; s2 mean +0.0371 std 0.0502 range [-0.0075,+0.1220]; s3 mean -0.0692 std 0.1833 range [-0.2991,+0.3023]
Train QAOA n=8; gamma mean -0.2500 std 0.2326 range [-0.4100,+0.3148]; beta mean +0.2967 std 0.2797 range [-0.3948,+0.4148]

Descriptive comparison to PR118 N=12 and PR116 N=15 train stds (different seeds; not paired; n=8 caveat):

- θ_K std N=9 0.0693 vs N=12 0.0573 vs N=15 0.0511
- θ_M std N=9 0.0346 vs N=12 0.0152 vs N=15 0.0233
- s1 std N=9 0.9605 vs N=12 0.8265 vs N=15 0.8058; train s1 n_neg 4/8 n_pos 4/8
- QAOA β std N=9 0.2797 vs N=12 0.5523 vs N=15 0.4015

New energy queries 32800. Historical 3047456. Total 3080256.
Classical B* wall 2.3s.

Unsigned-mean ladder (primary 2a, not this trial's selector): N=12 PASS ΔE −1.093; N=15 PASS ΔE −0.764; N=18 FAIL ΔE +1.826.
Median is frozen but is not the primary gate. Do not switch primary to median. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118 stands. Exploratory; not adoption.

Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.
