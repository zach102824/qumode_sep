# N18ALIGN: N=18 L=1 sign-aligned and larger-cluster transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10680–10687, test 10688–10695. N=18 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Unsigned mean/median, sign-aligned mean (s1-only flip), and larger-cluster mean frozen from train before test.
Primary: sign-aligned-mean Nov3 − unsigned-mean QAOA. Not L2/L3. Not architecture. No GR. Not a PR119 test post-hoc.

Fits 32 ok 32. Transfer evals 32 ok 32. Charged 32800. Rejected B* 0.

**Label TRANSFER_MEAN_NEGATIVE_CI.**
1 train Nov3−QAOA: -0.31419 med -0.32459 [-0.34561,-0.27427] tails [-0.37366,-0.18857] 8/8
2 primary sign-aligned Nov3 − unsigned QAOA: -0.96926 med -0.94682 [-1.02708,-0.91588] tails [-1.11597,-0.83940] 8/8
2b larger-cluster Nov3 − unsigned QAOA: -2.62886 med -2.55847 [-2.84378,-2.49359] tails [-3.32599,-2.38663] 8/8
2c unsigned-mean Nov3 − unsigned QAOA (PR119 replication): -0.34887 med -0.34529 [-0.36315,-0.33478] tails [-0.37827,-0.31817] 8/8
3 test ceiling Nov3−QAOA: -0.35971 med -0.30213 [-0.47203,-0.27536] tails [-0.70860,-0.21133] 8/8
4 transfer gap sign-aligned vs independent Nov3: +1.77858 med +1.54936 [+1.48382,+2.23414] tails [+1.35041,+3.21719] (descriptive)

Train Nov3 unsigned n=8; gamma mean +0.0625 std 0.2640 range [-0.2648,+0.2805]; theta_K mean -0.3552 std 0.0566 range [-0.4324,-0.2584]; theta_M mean -0.0124 std 0.0290 range [-0.0441,+0.0465]; s1 mean +0.2262 std 0.8839 range [-0.8986,+0.9026]; s2 mean -0.0315 std 0.0964 range [-0.1376,+0.0890]; s3 mean +0.0941 std 0.2921 range [-0.3578,+0.4422]
Train Nov3 sign-aligned n=8; gamma mean +0.0625 std 0.2640 range [-0.2648,+0.2805]; theta_K mean -0.3552 std 0.0566 range [-0.4324,-0.2584]; theta_M mean -0.0124 std 0.0290 range [-0.0441,+0.0465]; s1 mean +0.8565 std 0.0357 range [+0.7905,+0.9026]; s2 mean -0.0315 std 0.0964 range [-0.1376,+0.0890]; s3 mean +0.0941 std 0.2921 range [-0.3578,+0.4422]
Train QAOA unsigned n=8; gamma mean +0.0057 std 0.2548 range [-0.2423,+0.2630]; beta mean -0.0010 std 0.4218 range [-0.3970,+0.4025]
s1 clusters n_pos=5 n_neg=3 chosen=s1>0 rule=strictly_larger n_cluster=5; n_flipped=3

B* wall 208.43230843544006. Train fit wall 75.2270085811615. Test wall 79.78961420059204. Sum cell wall 549.6s. Peak RSS KiB 265804.
