# Per-layer gamma, L=6 held out

This is not the one-gamma six-start recipe. P = 4L+2. L=6 has 26 parameters. One L=1 draw, then continuation. Energy only.

Executed evaluations 23942. Chains 48.

| L | n | median | worst | best | BOTH p6 | BOTH p11 | BOTH p12 |
|---|---|---|---|---|---|---|---|
| 1 | 48 | 0.7551 | 5.5045 | 0.7551 | 0/48 | 0/48 | 0/48 |
| 2 | 48 | 0.4300 | 5.2088 | 0.4300 | 0/48 | 0/48 | 0/48 |
| 3 | 48 | 0.2886 | 4.4842 | 0.2886 | 35/48 | 0/48 | 0/48 |
| 4 | 48 | 0.2254 | 4.4779 | 0.2254 | 36/48 | 0/48 | 0/48 |
| 5 | 48 | 0.1766 | 4.2951 | 0.1766 | 37/48 | 0/48 | 0/48 |
| 6 | 48 | 0.1465 | 3.9114 | 0.1465 | 42/48 | 35/48 | 35/48 |

L6 Wilson versus p11: [0.5900276538379887, 0.8343419643836709]. Versus p6: [0.7529927291394518, 0.9414302824959915]. Versus p12: [0.5900276538379887, 0.8343419643836709].
There is one L=1 start per chain, so the L=1 rate is the raw-start rate. It is not a six-start screen.
Dense replay: {'best': {'logged_E': -6.190602734356659, 'dense_E': -6.190602734356659, 'delta': 0.0, 'hash_match': True, 'F': 0.7338894539975509, 'error': 0.14648569912953313, 'grad': 4.8858310025761816e-05}, 'worst': {'logged_E': -2.4257066966236027, 'dense_E': -2.4257066966236023, 'delta': 4.440892098500626e-16, 'hash_match': True, 'F': 0.0038840920176577217, 'error': 3.9113817368625896, 'grad': 0.018058905991507494}}.

## Small transfer
N=6: 3633 evaluations. L6 BOTH p6 8/8, p11 3/8, p12 3/8. Median error 0.02851832760762507.
N=12: 3271 evaluations. L6 BOTH p6 7/8, p11 7/8, p12 7/8. Median error 0.054827011760987254.

The one-gamma equal-depth recipe is unchanged. N=15 and N=18 were not run.

Thirty-five of 48 L=6 chains beat p=11 on both metrics, and the same 35 beat p=12. The Wilson interval is [0.590, 0.834]. The median error is 0.146486 and the best fidelity at that energy is 0.733889. The upper quartile is 0.1845, which is above the p=11 error 0.170333, and the worst chain is error 3.911 with fidelity 0.00388. All 48 chains finished. There is no shortfall against the per-chain cap. Mean cost is 499 evaluations. The L=5 states along this same continuation have 22 parameters and 0 of 48 beat p=11. The win appears at the sixth layer.

Adopt per-layer gamma as the labeled family for matched-parameter pursuit. Do not replace the one-gamma six-start recipe on the equal-depth ledger. The N=6 transfer, 8 chains and 3633 evaluations, beats equal-depth p=6 on 8 of 8 and p=11 on 3 of 8. The N=12 transfer, 8 chains and 3271 evaluations, beats p=6, p=11, and p=12 on 7 of 8. Those are pilots, not 48-chain claims.
