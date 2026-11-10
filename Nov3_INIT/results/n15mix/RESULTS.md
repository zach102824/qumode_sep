# N=15 L=1 mixer-angle freeze

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=15 L=1 Q=1024 T=122. Train 10888–10895, test 10896–10903. Mixer angles θ_K, θ_M frozen from train mean before test. Test mixer-freeze is 4-D SPSA on (γ,s1,s2,s3). Not a two-template copy. Not unsigned-mean copy. Not architecture. No GR. Not L=2/L=3.

Fits 32 ok 32. Mixer-freeze fits 8 ok 8. Charged 40960 (independent 32768 + mix 8192). Failures 0.
Frozen mixer θ_K mean -0.444515 std 0.0469; θ_M mean -0.009452 std 0.0330.

**Label MIXER_FREEZE_NEGATIVE_CI.**

Primary mixer-freeze Nov3−QAOA: -0.31698 med -0.32223 [-0.35860,-0.27491] tails [-0.41640,-0.21905] 8/8
1 train independent Nov3−QAOA: -0.40558 med -0.36994 [-0.49550,-0.31664] tails [-0.61102,-0.24351] 8/8
2 test ceiling independent Nov3−QAOA: -0.31767 med -0.31598 [-0.35675,-0.27981] tails [-0.40571,-0.23622] 8/8
3 constraint gap mixer-freeze vs independent Nov3: +0.00069 med -0.00238 [-0.00975,+0.01191] tails [-0.02388,+0.02975] 5/8

New energy queries 40960. Historical 3211552. Total 3252512.
Classical B* wall 132.2s.

Exploratory; not adoption; does not replace per-instance SPSA on the full 6-vector. Two-template ladder is closed. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118, PR120, PR122, PR124, PR127, PR128 stand.

Fit workers 4, BLAS 1. Independent and mixer-freeze ledgers equal Q=1024.
