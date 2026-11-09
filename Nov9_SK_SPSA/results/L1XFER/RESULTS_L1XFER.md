# L1XFER: N=15 L=1 parameter concentration and transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10600–10607, test 10608–10615. L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Mean and median frozen from train before test. Not L2/L3. Not architecture. No GR.

Fits 32 ok 32. Transfer evals 32 ok 32. Charged 32800. Rejected B* 0.

**Label TRANSFER_MEAN_NEGATIVE_CI.**
1 train Nov3−QAOA: -0.29256 med -0.28934 [-0.32287,-0.26287] tails [-0.35767,-0.22570] 8/8
2a test mean Nov3−QAOA: -0.76351 med -0.75474 [-0.86313,-0.66344] tails [-0.94070,-0.57397] 8/8
2b test median Nov3−QAOA: -0.35047 med -0.31916 [-0.41353,-0.29560] tails [-0.53342,-0.26389] 8/8
3 test ceiling Nov3−QAOA: -0.30334 med -0.29360 [-0.36203,-0.25778] tails [-0.47492,-0.22320] 8/8
4 transfer gap mean vs independent Nov3: +1.19928 med +1.17448 [+1.05377,+1.35423] tails [+0.94936,+1.53965] (descriptive)

Train Nov3 n=8; gamma mean -0.1421 std 0.2520 range [-0.3347,+0.2633]; theta_K mean -0.3687 std 0.0511 range [-0.4410,-0.2941]; theta_M mean -0.0008 std 0.0233 range [-0.0337,+0.0373]; s1 mean -0.4066 std 0.8058 range [-0.8587,+0.9073]; s2 mean +0.0086 std 0.1043 range [-0.1495,+0.2169]; s3 mean -0.0265 std 0.2341 range [-0.3921,+0.1654]
Train QAOA n=8; gamma mean +0.0458 std 0.2613 range [-0.2766,+0.2873]; beta mean -0.0925 std 0.4015 range [-0.3939,+0.4037]

B* wall 18.610561847686768. Train fit wall 8.134418964385986. Test wall 8.277441024780273. Sum cell wall 55.4s. Peak RSS KiB 190736.
