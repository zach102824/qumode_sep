# Unread-seed N=15 and N=18 L=1 width panel

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. Exclusive seeds 10568–10575 at N=15 and 10576–10583 at N=18. Not L=2. Not pooled across N. Exploratory; no adoption; no GR. Compiler stream closed. L=2 holdout FAIL stands.

**WIDTH_L1_BOTH.**

New energy queries 32768. Historical 2883584. Total 2916352. Failures 0.
Classical B* wall 1006.2s.

## N=15 L=1 Q=1024

Nov3−QAOA mean ΔE -0.3325 CI [-0.437, -0.265] wins 8/8. CI<0 True.
mean E Nov3 -12.4792 QAOA -12.1467. mean (E−E0)/N Nov3 0.0445 QAOA 0.0667.
mean F Nov3 0.606 QAOA 0.476 (post hoc). start0 Nov3 3/8 QAOA 1/8.
seeds [10568, 10569, 10570, 10571, 10572, 10573, 10574, 10575].

## N=18 L=1 Q=1024

Nov3−QAOA mean ΔE -0.3852 CI [-0.453, -0.334] wins 8/8. CI<0 True.
mean E Nov3 -14.8828 QAOA -14.4976. mean (E−E0)/N Nov3 0.0421 QAOA 0.0635.
mean F Nov3 0.556 QAOA 0.424 (post hoc). start0 Nov3 2/8 QAOA 1/8.
seeds [10576, 10577, 10578, 10579, 10580, 10581, 10582, 10583].

v5 N=18 L=1 used development seeds 10018–10025 (confounded with N=15). This panel uses exclusive unread instances at each width.
Do not read WIDTH_L1_BOTH as undoing L=2 holdout FAIL or as a hardware claim.

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024.
