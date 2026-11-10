# N9TWO: N=9 L=1 two-template bimodal cluster-mean transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10840–10847, test 10848–10855. N=9 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Two Nov3 cluster means (sign(s1)) and two QAOA cluster means (sign(beta)) frozen from train before test.
Two-template energy = min of nonempty cluster-template energies. Unsigned/sign-aligned secondary only. No GR. Does not reopen PR119.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824. Rejected B* 0.
Nov3 templates 2 (n_pos=2 n_neg=6); QAOA templates 2 (n_pos=5 n_neg=3).

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**
primary two-template Nov3 − two-template QAOA: -0.18261 med -0.22178 [-0.27003,-0.09588] tails [-0.38714,-0.00942] 8/8
1 train independent Nov3−QAOA: -0.22392 med -0.19745 [-0.27535,-0.18637] tails [-0.38530,-0.16858] 8/8
2 test ceiling independent Nov3−QAOA: -0.25754 med -0.26035 [-0.28321,-0.23080] tails [-0.31104,-0.19401] 8/8
3 two-template Nov3 vs independent QAOA: -0.13321 med -0.18462 [-0.21916,-0.04075] tails [-0.28067,+0.03035] 5/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.12433 med +0.04793 [+0.04625,+0.21021] tails [+0.00363,+0.31897] 0/8 (CI entirely >0: True; CI includes 0: False; descriptive only; does not switch primary)
5a Nov3 cluster+ vs unsigned QAOA: -1.36733 med -1.25384 [-1.58640,-1.21769] tails [-2.05175,-1.14154] 8/8
5b Nov3 cluster− vs unsigned QAOA: -1.43266 med -1.32622 [-1.63682,-1.30075] tails [-2.11725,-1.26220] 8/8
5c Nov3 cluster+ vs independent QAOA: -0.06471 med -0.17716 [-0.16386,+0.06553] tails [-0.21518,+0.16111] 5/8
5d Nov3 cluster− vs independent QAOA: -0.13004 med -0.17196 [-0.21570,-0.03977] tails [-0.28067,+0.03035] 5/8
6a unsigned-mean Nov3 − unsigned QAOA (replication): -0.67952 med -0.71869 [-0.78769,-0.58203] tails [-0.95684,-0.49948] 8/8
6b sign-aligned Nov3 − unsigned QAOA (replication): +1.76997 med +2.00938 [+1.45874,+2.04653] tails [+1.11853,+2.18581] 0/8

Train Nov3 unsigned n=8; gamma mean -0.1782 std 0.3550 range [-0.4639,+0.4172]; theta_K mean -0.4047 std 0.0402 range [-0.4663,-0.3508]; theta_M mean -0.0026 std 0.0465 range [-0.0460,+0.0933]; s1 mean -0.4589 std 0.8131 range [-0.9921,+0.8601]; s2 mean +0.0471 std 0.0720 range [-0.0257,+0.2004]; s3 mean -0.0534 std 0.1926 range [-0.4296,+0.1261]
Train Nov3 sign-aligned n=8; gamma mean -0.1782 std 0.3550 range [-0.4639,+0.4172]; theta_K mean -0.4047 std 0.0402 range [-0.4663,-0.3508]; theta_M mean -0.0026 std 0.0465 range [-0.0460,+0.0933]; s1 mean +0.8873 std 0.0447 range [+0.8536,+0.9921]; s2 mean +0.0471 std 0.0720 range [-0.0257,+0.2004]; s3 mean -0.0534 std 0.1926 range [-0.4296,+0.1261]
Train Nov3 cluster+ n=2; gamma mean +0.3920 std 0.0356 range [+0.3669,+0.4172]; theta_K mean -0.3879 std 0.0010 range [-0.3886,-0.3872]; theta_M mean -0.0369 std 0.0129 range [-0.0460,-0.0277]; s1 mean +0.8568 std 0.0046 range [+0.8536,+0.8601]; s2 mean +0.0039 std 0.0419 range [-0.0257,+0.0335]; s3 mean -0.2850 std 0.2044 range [-0.4296,-0.1405]
Train Nov3 cluster- n=6; gamma mean -0.3683 std 0.0527 range [-0.4639,-0.3169]; theta_K mean -0.4103 std 0.0460 range [-0.4663,-0.3508]; theta_M mean +0.0088 std 0.0487 range [-0.0420,+0.0933]; s1 mean -0.8974 std 0.0480 range [-0.9921,-0.8638]; s2 mean +0.0615 std 0.0769 range [-0.0054,+0.2004]; s3 mean +0.0239 std 0.1222 range [-0.2076,+0.1261]
Train QAOA unsigned n=8; gamma mean -0.0985 std 0.3608 range [-0.4498,+0.3670]; beta mean +0.0891 std 0.4071 range [-0.4234,+0.3885]
Train QAOA cluster+ n=5; gamma mean -0.3580 std 0.0535 range [-0.4498,-0.3191]; beta mean +0.3840 std 0.0053 range [+0.3775,+0.3885]
Train QAOA cluster- n=3; gamma mean +0.3340 std 0.0313 range [+0.3048,+0.3670]; beta mean -0.4024 std 0.0185 range [-0.4234,-0.3883]

B* wall 8.519668579101562. Train fit wall 0.686699390411377. Test wall 0.7284836769104004. Sum cell wall 1.7s. Peak RSS KiB 180764.
