# STATUS — ratio69_w9 L=16 seeds 0–5

Opus R1 Idea1. Unscaled shared γ, c1–c3; w6=1, w9=2; early-stop on held-out n9; Adam 600 → LBFGS 12.
Train ONLY n∈{6,9}. Eval CORE scale1000 B=1000, n∈{6,9,12,15}.
Gates locked: Hamming-shell SNAP + Nov1st X-sandwich. round_1/ and champion/ untouched.

scale1000 packs are gitignored and were not in the checkout. Regenerated with SCB=54e9 (`explore_OCT29.lib.sat_gen`, same protocol as the sealed n6 pack). sha256 prefixes match `Nov_3rd/DATA_MANIFEST.md`: n6 `d2e17f2c5e70`, n9 `49def928b148`, n12 `4c85d564b784`, n15 `c5279c0aa15d`. This is full scale1000 eval, not `eval_proxy`.

Success: R≤0.98 on every CORE n. Kill a seed if n9 R>0.93 or n15 R>0.96.

## ALL_CORE_BEAT

| seed | n6 Med | n6 R | n9 Med | n9 R | n12 Med | n12 R | n15 Med | n15 R | ALL_CORE_BEAT | KILL |
|------|--------|------|--------|------|---------|-------|---------|-------|---------------|------|
| 0 | 1.1568 | 0.9434 | 1.4069 | 0.8912 | 1.8246 | 0.8812 | 2.5840 | 0.8947 | True | no |
| 1 | 1.1448 | 0.9336 | 1.4103 | 0.8934 | 1.8498 | 0.8933 | 2.5956 | 0.8987 | True | no |
| 2 | 1.1476 | 0.9359 | 1.4114 | 0.8940 | 1.8326 | 0.8850 | 2.5657 | 0.8884 | True | no |
| 3 | 1.1455 | 0.9342 | 1.4132 | 0.8952 | 1.8314 | 0.8844 | 2.5686 | 0.8894 | True | no |
| 4 | 1.1451 | 0.9339 | 1.4058 | 0.8905 | 1.8389 | 0.8881 | 2.5963 | 0.8990 | True | no |
| 5 | 1.1496 | 0.9376 | 1.4220 | 0.9008 | 1.8623 | 0.8994 | 2.6382 | 0.9135 | True | no |

**ALL_CORE_BEAT = True for seeds 0, 1, 2, 3, 4, and 5.** No seed killed.

Per-seed tables: `eval_L16_seed{0..5}.md`. Params: `params_L16_seed{0..5}.pt`.

