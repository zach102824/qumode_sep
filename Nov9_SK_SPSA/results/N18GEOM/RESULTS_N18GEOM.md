# N18GEOM: N=18 L=1 SK-geometry freeze (θ_K, θ_M) plus per-instance SPSA

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10872–10879, test 10880–10887. N=18 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
θ_K and θ_M frozen from train mean before test. Geom SPSA overwrites those coordinates in every energy call.
Not two-template. Not unsigned-mean primary. No GR. Does not reopen PR119.

Fits 32 ok 32. Charged 32768. Rejected B* 0.
Frozen θ_K=-0.4019013360241197 θ_M=-0.02521462618911758.

**Label GEOM_TRANSFER_NEGATIVE_CI.**
primary geom Nov3 − QAOA: -0.35585 med -0.33109 [-0.41559,-0.30173] tails [-0.49479,-0.27132] 8/8
1 test ceiling independent Nov3−QAOA: -0.34573 med -0.32169 [-0.41618,-0.28449] tails [-0.49194,-0.24124] 8/8
2 transfer gap geom Nov3 vs independent Nov3: -0.01012 med -0.00498 [-0.02021,-0.00106] tails [-0.03615,+0.00478] 6/8 (CI includes 0: False; entirely >0: False; descriptive only)

Train Nov3 unsigned n=8; gamma mean +0.1144 std 0.2292 range [-0.2915,+0.2709]; theta_K mean -0.4019 std 0.0350 range [-0.4430,-0.3247]; theta_M mean -0.0252 std 0.0665 range [-0.1640,+0.0676]; s1 mean +0.4329 std 0.8095 range [-0.8791,+0.9063]; s2 mean -0.0389 std 0.0913 range [-0.2265,+0.0465]; s3 mean -0.0672 std 0.2871 range [-0.5472,+0.2877]

B* wall 183.95586609840393. Train fit wall 41.77859377861023. Test wall 115.98914098739624. Sum cell wall 583.3s. Peak RSS KiB 265708.
