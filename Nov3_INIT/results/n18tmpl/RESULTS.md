# N=18 L=1 two-template (bimodal cluster-mean) transfer

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=18 L=1 Q=1024 T=122. Train 10808–10815, test 10816–10823. Cluster means frozen from train before test. Primary is better-of-two min of frozen cluster templates. Not unsigned-mean copy. Not sign-aligned-only copy. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824 (independent 32768 + transfer 56). Failures 0.
Nov3 clusters: plus n=2 minus n=6 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}. QAOA clusters: plus n=3 minus n=5 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}.
Nov3 reduces to single cluster: False. QAOA reduces to single cluster: False.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**

Primary two-template Nov3−QAOA: -0.30171 med -0.32028 [-0.35295,-0.24681] tails [-0.42315,-0.18147] 8/8
1 train independent Nov3−QAOA: -0.40731 med -0.35200 [-0.50094,-0.33341] tails [-0.68585,-0.27895] 8/8
2 test ceiling independent Nov3−QAOA: -0.32215 med -0.32329 [-0.36366,-0.28394] tails [-0.43024,-0.24091] 8/8
3 two-template Nov3 vs independent test QAOA: -0.25074 med -0.28627 [-0.31014,-0.18902] tails [-0.34468,-0.10800] 8/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.07141 med +0.05586 [+0.02730,+0.11946] tails [+0.00425,+0.19623] 0/8
5a cluster+ Nov3 vs unsigned-mean QAOA: -2.12022 med -2.17584 [-2.30115,-1.94451] tails [-2.57121,-1.68907] 8/8
5b cluster+ Nov3 vs independent QAOA: -0.24306 med -0.28205 [-0.29819,-0.18576] tails [-0.33702,-0.10800] 8/8
5c cluster− Nov3 vs unsigned-mean QAOA: -2.11588 med -2.18217 [-2.31509,-1.92562] tails [-2.61158,-1.66943] 8/8
5d cluster− Nov3 vs independent QAOA: -0.23872 med -0.28627 [-0.30767,-0.16820] tails [-0.34468,-0.08837] 8/8
6a unsigned-mean Nov3 vs unsigned-mean QAOA (replication only): -0.97203 med -0.96839 [-1.01081,-0.92804] tails [-1.04174,-0.83743] 8/8
6b sign-aligned-mean Nov3 vs unsigned-mean QAOA (replication only): +2.93349 med +2.96011 [+2.72475,+3.13258] tails [+2.46563,+3.43929] 0/8

Train Nov3 unsigned n=8; gamma mean -0.1440 std 0.2630 range [-0.3823,+0.2795]; theta_K mean -0.3893 std 0.0529 range [-0.4670,-0.2976]; theta_M mean +0.0022 std 0.0223 range [-0.0286,+0.0310]; s1 mean -0.4565 std 0.8041 range [-0.9546,+0.8667]; s2 mean +0.0715 std 0.0822 range [-0.0997,+0.1961]; s3 mean -0.0461 std 0.2550 range [-0.5086,+0.2069]
Train Nov3 cluster+ n=2; gamma mean +0.2736 std 0.0082 range [+0.2678,+0.2795]; theta_K mean -0.3959 std 0.0182 range [-0.4088,-0.3830]; theta_M mean +0.0191 std 0.0065 range [+0.0144,+0.0237]; s1 mean +0.8450 std 0.0306 range [+0.8234,+0.8667]; s2 mean +0.0961 std 0.0159 range [+0.0848,+0.1073]; s3 mean -0.1823 std 0.4615 range [-0.5086,+0.1441]
Train Nov3 cluster− n=6; gamma mean -0.2832 std 0.0615 range [-0.3823,-0.2219]; theta_K mean -0.3871 std 0.0619 range [-0.4670,-0.2976]; theta_M mean -0.0034 std 0.0232 range [-0.0286,+0.0310]; s1 mean -0.8904 std 0.0400 range [-0.9546,-0.8382]; s2 mean +0.0634 std 0.0953 range [-0.0997,+0.1961]; s3 mean -0.0007 std 0.1963 range [-0.3148,+0.2069]
Train QAOA unsigned n=8; gamma mean +0.0536 std 0.2987 range [-0.4213,+0.3416]; beta mean -0.1013 std 0.4110 range [-0.4030,+0.4042]
Train QAOA cluster+ n=3; gamma mean -0.2974 std 0.1075 range [-0.4213,-0.2306]; beta mean +0.3950 std 0.0121 range [+0.3814,+0.4042]
Train QAOA cluster− n=5; gamma mean +0.2642 std 0.0501 range [+0.2122,+0.3416]; beta mean -0.3990 std 0.0043 range [-0.4030,-0.3919]

Train s1 n_neg 6/8 n_pos 2/8 n_zero 0/8 std 0.8041.
Train QAOA β n_neg 5/8 n_pos 3/8 std 0.4110.
Chosen Nov3 templates: {10816: 'minus', 10817: 'minus', 10818: 'plus', 10819: 'minus', 10820: 'minus', 10821: 'minus', 10822: 'plus', 10823: 'plus'}. Chosen QAOA templates: {10816: 'minus', 10817: 'minus', 10818: 'minus', 10819: 'minus', 10820: 'minus', 10821: 'plus', 10822: 'minus', 10823: 'minus'}.

New energy queries 32824. Historical 3145904. Total 3178728.
Classical B* wall 1842.7s.

Unsigned-mean and sign-aligned-mean are secondary only; do not reopen the unsigned-mean ladder. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118 and PR120 stand. Exploratory; not adoption.

Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.
