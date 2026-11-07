# Round 3 family N

Contiguous block-mean plus Neel selector on the frozen manifest. The file
`FROZEN_TARGET_R2.json` is the same byte string as round 2, SHA-256
`14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`.

```text
G = Hc + thetaK Khat + thetaM Mhat + thetaN Nhat
M = sum_{a<b} Jbar_ab Sa Sb
N = sum_a (Z0 Z1 + Z1 Z2 - 2 Z0 Z2)
```

Each hat is scaled by RMS(J)/RMS(feature) over every i<j slot, zeros included.
gamma and the three thetas are shared across layers. P = 3L+4.

```bash
python3 -m Nov_3rd.tfim_shared_gamma.round3_N.run_round3
```

N=12 depths 1..6 are the study. N=6 is an L=1 smoke test. This generator is
not a special case of round-2 S.
