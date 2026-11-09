# DEPTH C1: charged L1-continuation vs cold L

Exclusive unread development seeds 10036–10043. Ordinary IID SPSA. Energy-only selection. F post hoc.
Q = 1024·L inclusive of L1 prefix, continuation, calibration, candidates, and counted failures.

Runs 128 ok 128 failures 0. Charged 327680. Outside cap 0. Rejected instances 0. Missing 0.

Gate N15 L3 Nov3 mean ΔE (cont−cold) = +0.021958 CI [-0.027569, +0.060849] n=8: **reject**.

N3COND 772224 and LEXACT SPSA 2560 are not revised. This stream is DEPTH_C1.

## Continuation vs cold

| cell | Δ(E−E0)/N mean [CI] wins(cont) | ΔE mean [CI] | ΔF mean [CI] wins(cont) |
|---|---|---|---|
| N12 L2 Nov3 | +0.00240 [+0.00123,+0.00397] 0/8 | +0.0288 [+0.0148,+0.0477] | -0.0172 [-0.0328,-0.0046] 2/8 |
| N12 L2 QAOA | +0.01229 [+0.00582,+0.01902] 0/8 | +0.1475 [+0.0699,+0.2283] | -0.0769 [-0.1131,-0.0403] 0/8 |
| N12 L3 Nov3 | +0.00246 [-0.00321,+0.00771] 2/8 | +0.0295 [-0.0386,+0.0926] | -0.0385 [-0.0850,+0.0084] 2/8 |
| N12 L3 QAOA | +0.00777 [+0.00587,+0.00990] 0/8 | +0.0933 [+0.0705,+0.1188] | -0.0549 [-0.0695,-0.0413] 0/8 |
| N15 L2 Nov3 | +0.00077 [+0.00046,+0.00111] 0/8 | +0.0115 [+0.0070,+0.0167] | -0.0042 [-0.0078,-0.0005] 3/8 |
| N15 L2 QAOA | +0.01458 [+0.00778,+0.02129] 0/8 | +0.2187 [+0.1166,+0.3194] | -0.0869 [-0.1217,-0.0483] 1/8 |
| N15 L3 Nov3 | +0.00146 [-0.00184,+0.00406] 1/8 | +0.0220 [-0.0276,+0.0608] | -0.0267 [-0.0693,+0.0179] 2/8 |
| N15 L3 QAOA | +0.00970 [+0.00670,+0.01304] 0/8 | +0.1455 [+0.1005,+0.1957] | -0.0729 [-0.0943,-0.0523] 0/8 |

## Nov3 vs QAOA (per-seed energy-better of cold/continuation)

| cell | Δ(E−E0)/N mean [CI] Nov3 lower | ΔF mean [CI] Nov3 higher |
|---|---|---|
| N12 L2 | +0.00235 [+0.00024,+0.00468] 3/8 | +0.0179 [+0.0043,+0.0333] 6/8 |
| N12 L3 | +0.00356 [+0.00172,+0.00538] 1/8 | +0.0302 [+0.0071,+0.0512] 6/8 |
| N15 L2 | +0.00212 [+0.00035,+0.00371] 2/8 | +0.0105 [-0.0013,+0.0237] 4/8 |
| N15 L3 | +0.00460 [+0.00254,+0.00632] 1/8 | +0.0138 [-0.0107,+0.0435] 4/8 |
