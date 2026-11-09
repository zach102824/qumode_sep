# N=12 L=1 two-template (bimodal cluster-mean) transfer

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=12 L=1 Q=1024 T=122. Train 10776–10783, test 10784–10791. Cluster means frozen from train before test. Primary is better-of-two min of frozen cluster templates. Not unsigned-mean copy. Not sign-aligned-only copy. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824 (independent 32768 + transfer 56). Failures 0.
Nov3 clusters: plus n=4 minus n=4 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}. QAOA clusters: plus n=5 minus n=3 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}.
Nov3 reduces to single cluster: False. QAOA reduces to single cluster: False.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**

Primary two-template Nov3−QAOA: -0.20236 med -0.21516 [-0.25248,-0.13402] tails [-0.28154,+0.01338] 7/8
1 train independent Nov3−QAOA: -0.31916 med -0.27138 [-0.44392,-0.23824] tails [-0.71057,-0.19011] 8/8
2 test ceiling independent Nov3−QAOA: -0.25940 med -0.25592 [-0.29112,-0.22902] tails [-0.33318,-0.20125] 8/8
3 two-template Nov3 vs independent test QAOA: -0.16062 med -0.19112 [-0.22922,-0.06185] tails [-0.27941,+0.15288] 7/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.09878 med +0.03345 [+0.01971,+0.21084] tails [+0.00343,+0.45787] 0/8
5a cluster+ Nov3 vs unsigned-mean QAOA: -1.69810 med -1.67092 [-1.98572,-1.44623] tails [-2.48889,-1.23344] 8/8
5b cluster+ Nov3 vs independent QAOA: -0.14584 med -0.19112 [-0.22577,-0.02206] tails [-0.27760,+0.25692] 7/8
5c cluster− Nov3 vs unsigned-mean QAOA: -1.70217 med -1.65659 [-1.98054,-1.46145] tails [-2.45864,-1.23755] 8/8
5d cluster− Nov3 vs independent QAOA: -0.14990 med -0.16921 [-0.22163,-0.05255] tails [-0.27941,+0.15288] 7/8
6a unsigned-mean Nov3 vs unsigned-mean QAOA (replication only): +0.13665 med +0.14102 [+0.11437,+0.15735] tails [+0.08883,+0.18371] 0/8
6b sign-aligned-mean Nov3 vs unsigned-mean QAOA (replication only): +0.07858 med +0.08083 [+0.06665,+0.08991] tails [+0.04994,+0.10714] 0/8

Train Nov3 unsigned n=8; gamma mean +0.0055 std 0.3498 range [-0.4622,+0.3465]; theta_K mean -0.4106 std 0.0730 range [-0.5518,-0.2980]; theta_M mean -0.0018 std 0.0251 range [-0.0311,+0.0385]; s1 mean -0.0165 std 0.9416 range [-0.9763,+0.8968]; s2 mean -0.0232 std 0.0667 range [-0.1476,+0.0664]; s3 mean +0.0477 std 0.2569 range [-0.1930,+0.4538]
Train Nov3 cluster+ n=4; gamma mean +0.3271 std 0.0141 range [+0.3127,+0.3465]; theta_K mean -0.3922 std 0.0279 range [-0.4295,-0.3619]; theta_M mean -0.0071 std 0.0233 range [-0.0303,+0.0244]; s1 mean +0.8633 std 0.0234 range [+0.8436,+0.8968]; s2 mean -0.0599 std 0.0662 range [-0.1476,+0.0133]; s3 mean +0.0869 std 0.2408 range [-0.1735,+0.4083]
Train Nov3 cluster− n=4; gamma mean -0.3161 std 0.0980 range [-0.4622,-0.2525]; theta_K mean -0.4290 std 0.1037 range [-0.5518,-0.2980]; theta_M mean +0.0034 std 0.0293 range [-0.0311,+0.0385]; s1 mean -0.8963 std 0.0665 range [-0.9763,-0.8141]; s2 mean +0.0135 std 0.0490 range [-0.0372,+0.0664]; s3 mean +0.0085 std 0.3031 range [-0.1930,+0.4538]
Train QAOA unsigned n=8; gamma mean -0.0433 std 0.3199 range [-0.2805,+0.4764]; beta mean +0.0886 std 0.4051 range [-0.4114,+0.3888]
Train QAOA cluster+ n=5; gamma mean -0.2698 std 0.0139 range [-0.2805,-0.2464]; beta mean +0.3821 std 0.0056 range [+0.3753,+0.3888]
Train QAOA cluster− n=3; gamma mean +0.3342 std 0.1259 range [+0.2370,+0.4764]; beta mean -0.4006 std 0.0094 range [-0.4114,-0.3949]

Train s1 n_neg 4/8 n_pos 4/8 n_zero 0/8 std 0.9416.
Train QAOA β n_neg 3/8 n_pos 5/8 std 0.4051.
Chosen Nov3 templates: {10784: 'minus', 10785: 'plus', 10786: 'minus', 10787: 'plus', 10788: 'plus', 10789: 'plus', 10790: 'minus', 10791: 'plus'}. Chosen QAOA templates: {10784: 'plus', 10785: 'plus', 10786: 'plus', 10787: 'minus', 10788: 'minus', 10789: 'minus', 10790: 'plus', 10791: 'minus'}.

New energy queries 32824. Historical 3113080. Total 3145904.
Classical B* wall 10.6s.

Unsigned-mean and sign-aligned-mean are secondary only; do not reopen the unsigned-mean ladder. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118 and PR120 stand. Exploratory; not adoption.

Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.
