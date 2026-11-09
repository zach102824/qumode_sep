# N=9 L=1 two-template (bimodal cluster-mean) transfer

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=9 L=1 Q=1024 T=122. Train 10744–10751, test 10752–10759. Cluster means frozen from train before test. Primary is better-of-two min of frozen cluster templates. Not unsigned-mean copy. Not sign-aligned-only copy. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824 (independent 32768 + transfer 56). Failures 0.
Nov3 clusters: plus n=3 minus n=5 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}. QAOA clusters: plus n=4 minus n=4 two_template={'n_nonempty': 2, 'templates': ['plus', 'minus'], 'reduces_to_single_cluster': False, 'both_empty': False, 'two_template': True}.
Nov3 reduces to single cluster: False. QAOA reduces to single cluster: False.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**

Primary two-template Nov3−QAOA: -0.25062 med -0.24884 [-0.28800,-0.21491] tails [-0.34680,-0.16989] 8/8
1 train independent Nov3−QAOA: -0.26712 med -0.25235 [-0.32876,-0.21207] tails [-0.41242,-0.16911] 8/8
2 test ceiling independent Nov3−QAOA: -0.22569 med -0.21831 [-0.26655,-0.18523] tails [-0.32456,-0.12941] 8/8
3 two-template Nov3 vs independent test QAOA: -0.21876 med -0.22109 [-0.25668,-0.18170] tails [-0.30955,-0.13525] 8/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.00694 med +0.00828 [-0.00326,+0.01787] tails [-0.01502,+0.03838] 3/8
5a cluster+ Nov3 vs unsigned-mean QAOA: -1.52035 med -1.43557 [-1.64785,-1.42197] tails [-1.91782,-1.38455] 8/8
5b cluster+ Nov3 vs independent QAOA: -0.21828 med -0.22109 [-0.25620,-0.18052] tails [-0.30955,-0.13525] 8/8
5c cluster− Nov3 vs unsigned-mean QAOA: -1.49156 med -1.42740 [-1.61162,-1.38995] tails [-1.84159,-1.31162] 8/8
5d cluster− Nov3 vs independent QAOA: -0.18950 med -0.17366 [-0.22912,-0.15056] tails [-0.27924,-0.10297] 8/8
6a unsigned-mean Nov3 vs unsigned-mean QAOA (replication only): -0.17207 med -0.16690 [-0.18924,-0.15757] tails [-0.22109,-0.14915] 8/8
6b sign-aligned-mean Nov3 vs unsigned-mean QAOA (replication only): +0.75289 med +0.76223 [+0.69328,+0.81509] tails [+0.59501,+0.91993] 0/8

Train Nov3 unsigned n=8; gamma mean -0.0888 std 0.3291 range [-0.3869,+0.3286]; theta_K mean -0.3987 std 0.0466 range [-0.4654,-0.3154]; theta_M mean +0.0021 std 0.0446 range [-0.0565,+0.0713]; s1 mean -0.1793 std 0.9148 range [-0.8654,+0.9555]; s2 mean -0.0466 std 0.0891 range [-0.2120,+0.0439]; s3 mean +0.1203 std 0.1895 range [-0.1642,+0.3120]
Train Nov3 cluster+ n=3; gamma mean +0.3063 std 0.0239 range [+0.2810,+0.3286]; theta_K mean -0.4173 std 0.0294 range [-0.4434,-0.3854]; theta_M mean +0.0046 std 0.0641 range [-0.0565,+0.0713]; s1 mean +0.9253 std 0.0273 range [+0.9025,+0.9555]; s2 mean -0.1283 std 0.0914 range [-0.2120,-0.0308]; s3 mean +0.2067 std 0.1083 range [+0.0957,+0.3120]
Train Nov3 cluster− n=5; gamma mean -0.3258 std 0.0441 range [-0.3869,-0.2623]; theta_K mean -0.3876 std 0.0544 range [-0.4654,-0.3154]; theta_M mean +0.0006 std 0.0377 range [-0.0527,+0.0413]; s1 mean -0.8420 std 0.0164 range [-0.8654,-0.8192]; s2 mean +0.0024 std 0.0412 range [-0.0613,+0.0439]; s3 mean +0.0685 std 0.2192 range [-0.1642,+0.3070]
Train QAOA unsigned n=8; gamma mean -0.0026 std 0.3054 range [-0.3253,+0.3098]; beta mean -0.0036 std 0.4149 range [-0.3934,+0.4309]
Train QAOA cluster+ n=4; gamma mean -0.2859 std 0.0503 range [-0.3253,-0.2190]; beta mean +0.3839 std 0.0349 range [+0.3465,+0.4309]
Train QAOA cluster− n=4; gamma mean +0.2807 std 0.0328 range [+0.2350,+0.3098]; beta mean -0.3912 std 0.0016 range [-0.3934,-0.3900]

Train s1 n_neg 5/8 n_pos 3/8 n_zero 0/8 std 0.9148.
Train QAOA β n_neg 4/8 n_pos 4/8 std 0.4149.
Chosen Nov3 templates: {10752: 'plus', 10753: 'plus', 10754: 'plus', 10755: 'plus', 10756: 'plus', 10757: 'plus', 10758: 'minus', 10759: 'plus'}. Chosen QAOA templates: {10752: 'minus', 10753: 'minus', 10754: 'plus', 10755: 'plus', 10756: 'minus', 10757: 'minus', 10758: 'plus', 10759: 'plus'}.

New energy queries 32824. Historical 3080256. Total 3113080.
Classical B* wall 2.3s.

Unsigned-mean and sign-aligned-mean are secondary only; do not reopen the unsigned-mean ladder. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118 and PR120 stand. Exploratory; not adoption.

Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.
