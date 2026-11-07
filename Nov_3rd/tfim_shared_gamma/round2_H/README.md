# Round 2 family H

Shared-gamma homotopy for full-K family A on the frozen target `FROZEN_TARGET_R2.json`.

Legal ansatz, one gamma and one theta_K shared by every layer:

```text
G = Hc + theta_K * FK_normalized
|+> -> [exp(-i gamma G) -> shell(c1,c2,c3)_l ]^L
```

`FK_jk = sum_i J_ij J_ik` for `j != k`, then the full upper triangle (zeros included) is RMS-matched to Hc.

The homotopy minimizes `E + lambda * sum_l (gamma_l - mean)^2` and then projects onto one gamma. Those independent-gamma numbers are an initializer, not a legal candidate and not a certified ceiling.

```bash
cd Nov_3rd/tfim_shared_gamma/round2_H
OMP_NUM_THREADS=1 python3 test_smoke.py
OMP_NUM_THREADS=1 python3 run_round2.py
```

N=6 is a smoke test. The comparison is N=12, depths 1 through 6.
