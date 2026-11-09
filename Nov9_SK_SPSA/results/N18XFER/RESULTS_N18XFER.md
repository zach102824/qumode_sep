# N18XFER: N=18 L=1 parameter concentration and transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10648–10655, test 10656–10663. N=18 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Mean and median frozen from train before test. Primary aggregator remains mean (not switched to median from PR116 test). Not L2/L3. Not architecture. No GR.

Fits 32 ok 32. Transfer evals 32 ok 32. Charged 32800. Rejected B* 0.

**Label UNSTABLE.**
1 train Nov3−QAOA: -0.35604 med -0.37311 [-0.40095,-0.30587] tails [-0.42501,-0.22954] 8/8
2a test mean Nov3−QAOA: +1.82618 med +1.80779 [+1.72854,+1.97027] tails [+1.65354,+2.26567] 0/8
2b test median Nov3−QAOA: +2.22550 med +2.19362 [+2.09108,+2.38379] tails [+1.91807,+2.69777] 0/8
3 test ceiling Nov3−QAOA: -0.30963 med -0.30641 [-0.34324,-0.27343] tails [-0.37496,-0.21130] 8/8
4 transfer gap mean vs independent Nov3: +2.57198 med +2.51086 [+2.40541,+2.75887] tails [+2.20902,+3.06271] (descriptive)

Train Nov3 n=8; gamma mean -0.0011 std 0.2853 range [-0.3272,+0.3012]; theta_K mean -0.3897 std 0.0532 range [-0.4419,-0.2767]; theta_M mean -0.0083 std 0.0163 range [-0.0236,+0.0199]; s1 mean -0.0055 std 0.9300 range [-0.8953,+0.9104]; s2 mean +0.0696 std 0.0901 range [-0.0717,+0.1992]; s3 mean -0.1505 std 0.2888 range [-0.6178,+0.1325]
Train QAOA n=8; gamma mean -0.1690 std 0.1962 range [-0.2910,+0.3127]; beta mean +0.2952 std 0.2777 range [-0.3919,+0.4023]
vs PR116 N=15 train std (descriptive, different seeds, not paired): nov3_gamma 0.2853 vs 0.2520; nov3_theta_K 0.0532 vs 0.0511; nov3_theta_M 0.0163 vs 0.0233; nov3_s1 0.9300 vs 0.8058; nov3_s2 0.0901 vs 0.1043; nov3_s3 0.2888 vs 0.2341; qaoa_gamma 0.1962 vs 0.2613; qaoa_beta 0.2777 vs 0.4015

B* wall 199.7056863307953. Train fit wall 76.68968605995178. Test wall 76.02319025993347. Sum cell wall 539.5s. Peak RSS KiB 265532.
