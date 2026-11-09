# N18TWO: N=18 L=1 two-template bimodal cluster-mean transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10728–10735, test 10736–10743. N=18 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Two Nov3 cluster means (sign(s1)) and two QAOA cluster means (sign(beta)) frozen from train before test.
Two-template energy = min of nonempty cluster-template energies. Unsigned/sign-aligned secondary only. No GR. Does not reopen PR119.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824. Rejected B* 0.
Nov3 templates 2 (n_pos=5 n_neg=3); QAOA templates 2 (n_pos=3 n_neg=5).

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**
primary two-template Nov3 − two-template QAOA: -0.39760 med -0.38372 [-0.44913,-0.34242] tails [-0.50046,-0.25699] 8/8
1 train independent Nov3−QAOA: -0.33004 med -0.32876 [-0.34742,-0.31403] tails [-0.36901,-0.30279] 8/8
2 test ceiling independent Nov3−QAOA: -0.41132 med -0.37768 [-0.50421,-0.33614] tails [-0.69123,-0.25350] 8/8
3 two-template Nov3 vs independent QAOA: -0.33881 med -0.35253 [-0.38953,-0.28818] tails [-0.46521,-0.21195] 8/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.07251 med +0.00196 [-0.00165,+0.19489] tails [-0.01246,+0.47928] 3/8
5a Nov3 cluster+ vs unsigned QAOA: -2.37095 med -2.45566 [-2.60344,-2.09775] tails [-2.87367,-1.63616] 8/8
5b Nov3 cluster− vs unsigned QAOA: -2.34492 med -2.42701 [-2.57140,-2.07879] tails [-2.77471,-1.62780] 8/8
5c Nov3 cluster+ vs independent QAOA: -0.33489 med -0.35253 [-0.37890,-0.28602] tails [-0.43385,-0.21195] 8/8
5d Nov3 cluster− vs independent QAOA: -0.30885 med -0.32388 [-0.37307,-0.24408] tails [-0.46521,-0.11300] 8/8
6a unsigned-mean Nov3 − unsigned QAOA (replication): -0.05908 med -0.05984 [-0.06671,-0.05080] tails [-0.07522,-0.04165] 8/8
6b sign-aligned Nov3 − unsigned QAOA (replication): -0.78453 med -0.75267 [-0.88373,-0.69509] tails [-1.04469,-0.58623] 8/8

Train Nov3 unsigned n=8; gamma mean +0.0716 std 0.2703 range [-0.2845,+0.3000]; theta_K mean -0.3837 std 0.0281 range [-0.4201,-0.3440]; theta_M mean -0.0073 std 0.0246 range [-0.0394,+0.0395]; s1 mean +0.2123 std 0.8973 range [-0.8956,+0.8910]; s2 mean -0.0029 std 0.0910 range [-0.0971,+0.1601]; s3 mean -0.1112 std 0.3584 range [-0.8171,+0.2681]
Train Nov3 sign-aligned n=8; gamma mean +0.0716 std 0.2703 range [-0.2845,+0.3000]; theta_K mean -0.3837 std 0.0281 range [-0.4201,-0.3440]; theta_M mean -0.0073 std 0.0246 range [-0.0394,+0.0395]; s1 mean +0.8653 std 0.0308 range [+0.8212,+0.8956]; s2 mean -0.0029 std 0.0910 range [-0.0971,+0.1601]; s3 mean -0.1112 std 0.3584 range [-0.8171,+0.2681]
Train Nov3 cluster+ n=5; gamma mean +0.2669 std 0.0191 range [+0.2516,+0.3000]; theta_K mean -0.3769 std 0.0232 range [-0.4025,-0.3440]; theta_M mean -0.0017 std 0.0285 range [-0.0394,+0.0395]; s1 mean +0.8620 std 0.0267 range [+0.8373,+0.8910]; s2 mean -0.0360 std 0.0623 range [-0.0971,+0.0581]; s3 mean -0.0473 std 0.2431 range [-0.4319,+0.2287]
Train Nov3 cluster- n=3; gamma mean -0.2538 std 0.0266 range [-0.2845,-0.2368]; theta_K mean -0.3948 std 0.0374 range [-0.4201,-0.3519]; theta_M mean -0.0166 std 0.0167 range [-0.0335,-0.0001]; s1 mean -0.8707 std 0.0428 range [-0.8956,-0.8212]; s2 mean +0.0522 std 0.1180 range [-0.0738,+0.1601]; s3 mean -0.2176 std 0.5515 range [-0.8171,+0.2681]
Train QAOA unsigned n=8; gamma mean +0.0569 std 0.2450 range [-0.2451,+0.2880]; beta mean -0.0988 std 0.4015 range [-0.3957,+0.3956]
Train QAOA cluster+ n=3; gamma mean -0.2374 std 0.0122 range [-0.2451,-0.2234]; beta mean +0.3860 std 0.0085 range [+0.3798,+0.3956]
Train QAOA cluster- n=5; gamma mean +0.2335 std 0.0318 range [+0.2131,+0.2880]; beta mean -0.3897 std 0.0049 range [-0.3957,-0.3830]

B* wall 196.47549486160278. Train fit wall 72.8027970790863. Test wall 75.77586364746094. Sum cell wall 531.3s. Peak RSS KiB 265608.
