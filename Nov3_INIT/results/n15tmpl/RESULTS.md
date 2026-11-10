# N=15 L=1 two-template (bimodal cluster-mean) transfer

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=15 L=1 Q=1024 T=122. Train 10856–10863, test 10864–10871. Cluster means frozen from train before test. Primary is better-of-two min of frozen cluster templates. Not unsigned-mean copy. Not sign-aligned-only copy. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824 (independent 32768 + transfer 56). Failures 0.
Nov3 clusters: plus n=2 minus n=6 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}. QAOA clusters: plus n=5 minus n=3 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}.
Nov3 reduces to single cluster: False. QAOA reduces to single cluster: False.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**

Primary two-template Nov3−QAOA: -0.30108 med -0.32215 [-0.33391,-0.26548] tails [-0.35682,-0.20057] 8/8
1 train independent Nov3−QAOA: -0.39085 med -0.36150 [-0.47146,-0.31956] tails [-0.59036,-0.26962] 8/8
2 test ceiling independent Nov3−QAOA: -0.32304 med -0.33078 [-0.35510,-0.29334] tails [-0.41012,-0.26253] 8/8
3 two-template Nov3 vs independent test QAOA: -0.27391 med -0.31559 [-0.32223,-0.21717] tails [-0.34602,-0.13146] 8/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.04913 med +0.01073 [+0.00064,+0.11247] tails [-0.01041,+0.24460] 3/8
5a cluster+ Nov3 vs unsigned-mean QAOA: -1.81618 med -1.80303 [-1.95669,-1.68195] tails [-2.12950,-1.50072] 8/8
5b cluster+ Nov3 vs independent QAOA: -0.24490 med -0.29472 [-0.30962,-0.16742] tails [-0.34537,-0.06902] 8/8
5c cluster− Nov3 vs unsigned-mean QAOA: -1.84519 med -1.85389 [-1.97811,-1.71306] tails [-2.13015,-1.55458] 8/8
5d cluster− Nov3 vs independent QAOA: -0.27391 med -0.31559 [-0.32223,-0.21717] tails [-0.34602,-0.13146] 8/8
6a unsigned-mean Nov3 vs unsigned-mean QAOA (replication only): -0.71012 med -0.68530 [-0.76705,-0.66088] tails [-0.87504,-0.62326] 8/8
6b sign-aligned-mean Nov3 vs unsigned-mean QAOA (replication only): +2.54149 med +2.41945 [+2.34179,+2.80750] tails [+2.17602,+3.36733] 0/8

Train Nov3 unsigned n=8; gamma mean -0.1554 std 0.2964 range [-0.3590,+0.3288]; theta_K mean -0.3990 std 0.0753 range [-0.5322,-0.3325]; theta_M mean +0.0018 std 0.0263 range [-0.0387,+0.0366]; s1 mean -0.4250 std 0.8245 range [-0.9650,+0.9547]; s2 mean +0.0092 std 0.0481 range [-0.0761,+0.0755]; s3 mean -0.0009 std 0.1491 range [-0.3331,+0.1444]
Train Nov3 cluster+ n=2; gamma mean +0.3201 std 0.0123 range [+0.3114,+0.3288]; theta_K mean -0.4564 std 0.1072 range [-0.5322,-0.3806]; theta_M mean +0.0144 std 0.0064 range [+0.0099,+0.0189]; s1 mean +0.9082 std 0.0657 range [+0.8618,+0.9547]; s2 mean -0.0206 std 0.0785 range [-0.0761,+0.0349]; s3 mean -0.0943 std 0.3377 range [-0.3331,+0.1444]
Train Nov3 cluster− n=6; gamma mean -0.3139 std 0.0486 range [-0.3590,-0.2516]; theta_K mean -0.3798 std 0.0623 range [-0.4989,-0.3325]; theta_M mean -0.0023 std 0.0296 range [-0.0387,+0.0366]; s1 mean -0.8694 std 0.0537 range [-0.9650,-0.8021]; s2 mean +0.0191 std 0.0391 range [-0.0239,+0.0755]; s3 mean +0.0303 std 0.0606 range [-0.0395,+0.1106]
Train QAOA unsigned n=8; gamma mean -0.0962 std 0.3188 range [-0.4418,+0.3258]; beta mean +0.1007 std 0.4065 range [-0.3928,+0.4231]
Train QAOA cluster+ n=5; gamma mean -0.3219 std 0.0804 range [-0.4418,-0.2327]; beta mean +0.3950 std 0.0202 range [+0.3707,+0.4231]
Train QAOA cluster− n=3; gamma mean +0.2799 std 0.0577 range [+0.2151,+0.3258]; beta mean -0.3898 std 0.0051 range [-0.3928,-0.3839]

Train s1 n_neg 6/8 n_pos 2/8 n_zero 0/8 std 0.8245.
Train QAOA β n_neg 3/8 n_pos 5/8 std 0.4065.
Chosen Nov3 templates: {10864: 'minus', 10865: 'minus', 10866: 'minus', 10867: 'minus', 10868: 'minus', 10869: 'minus', 10870: 'minus', 10871: 'minus'}. Chosen QAOA templates: {10864: 'minus', 10865: 'minus', 10866: 'minus', 10867: 'minus', 10868: 'minus', 10869: 'minus', 10870: 'minus', 10871: 'minus'}.

New energy queries 32824. Historical 3178728. Total 3211552.
Classical B* wall 136.6s.

Unsigned-mean and sign-aligned-mean are secondary only; do not reopen the unsigned-mean ladder. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118 and PR120 stand. Exploratory; not adoption.

Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.
