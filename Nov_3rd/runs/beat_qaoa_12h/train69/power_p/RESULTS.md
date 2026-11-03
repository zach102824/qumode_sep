# train{6,9} Idea2 power_p — L=16 CORE

Train only n∈{6,9}. Eval CORE n∈{6,9,12,15} on scale1000 B=1000.
Script: `Nov_3rd/scripts/train_beat_train69.py --idea power_p --L 16 --threads 2`.
`a_n = (8/(n-1))^p` with a9=1. p is a free parameter, clipped to [-1, 2], prior `1e-2 p²`.
QAOA Med (paper-n6 L=16 ladder): n6 1.2261, n9 1.5787, n12 2.0707, n15 2.8881.
Verdict: R≤0.98 BEAT, R≤1.02 TIE, else UNDER.

## scale1000

The gitignored packs were absent. They were regenerated with SCB=54e9
(`explore_OCT29.lib.sat_gen.instances`, same schema as the existing n6 pack).
SHA256 prefixes match `Nov_3rd/DATA_MANIFEST.md` exactly
(n6 `d2e17f2c5e70`, n9 `49def928b148`, n12 `4c85d564b784`, n15 `c5279c0aa15d`).
Tables below are the real CORE B=1000 eval, not a proxy.

`explore_OCT29/lib/qaoa.py` was missing from the tree. The restored qaoa_ref
circuit reproduces the published L16 medians (n6 1.226144, n9 1.578650) before training.

## Learned p (init p0=0, seeds 0–2)

| seed | p | n6 Med | R | n9 Med | R | n12 Med | R | n15 Med | R |
|------|---|--------|---|--------|---|---------|---|---------|---|
| 0 | 0.0515 | 1.1555 | 0.9424 | 1.4056 | 0.8904 | 1.8175 | 0.8777 | 2.5488 | 0.8825 |
| 1 | 0.0520 | 1.1447 | 0.9336 | 1.4097 | 0.8930 | 1.8422 | 0.8896 | 2.5726 | 0.8908 |
| 2 | 0.0438 | 1.1465 | 0.9350 | 1.4106 | 0.8935 | 1.8290 | 0.8833 | 2.5562 | 0.8851 |

All three seeds are **ALL_CORE_BEAT**. Learned p sits near +0.05, inside (-1, 2), not on a clip bound.
Holdout checkpoint (median log-ratio on 50 held-out n9 instances): seed0 lbfgs9, seed1 lbfgs1, seed2 lbfgs0.

## p0 sweep (seed 0 each, `--tag p{val}`)

`--p0` is the initial value of trainable p. The script has no freeze flag, so p still moves.
`p0/` is the learned seed0 trajectory (`--p0 0 --seed 0`); it was copied, not retrained.

| tag | p init | p final | n6 Med | R | n9 Med | R | n12 Med | R | n15 Med | R |
|-----|--------|---------|--------|---|--------|---|---------|---|---------|---|
| p-0.5 | -0.5 | -0.0162 | 1.1559 | 0.9427 | 1.4074 | 0.8915 | 1.8278 | 0.8827 | 2.5821 | 0.8941 |
| p0 | 0 | 0.0515 | 1.1555 | 0.9424 | 1.4056 | 0.8904 | 1.8175 | 0.8777 | 2.5488 | 0.8825 |
| p0.5 | 0.5 | 0.1038 | 1.1524 | 0.9399 | 1.4145 | 0.8960 | 1.8431 | 0.8901 | 2.5881 | 0.8961 |
| p1 | 1 | 0.4101 | 1.1602 | 0.9462 | 1.4065 | 0.8909 | 1.8134 | 0.8758 | 2.5465 | 0.8817 |

All four are **ALL_CORE_BEAT**. Reported in full. Not selected on n12/n15.
None of the final p values sit on the clip bounds.

## Paths

Artifacts only under `Nov_3rd/runs/beat_qaoa_12h/train69/power_p/`.
`ratio69_w9` was not written.
