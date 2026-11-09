# N=15 L=3 doubled budget q=2048 Q=6144

Original shared-Uc Nov3 (12 params) vs QAOA p=3 (6 params). Frozen v4 center-4. Same Q both methods. Seeds 10616–10623. 10624–10631 unused. Not spread8/anneal/pair-shell/V(M). No GR. If FAIL, stop; no q=4096.

**L3_2XBUDGET_FAIL.**

New energy queries 98304. Historical 2916352. Total 3014656. Failures 0.
Classical B* wall 65.8s.

## Primary: Nov3−QAOA at Q=6144 (n=8)

mean ΔE +0.0358 CI [-0.017, +0.089] wins 2/8. CI<0 False.
mean E Nov3 -13.5433 QAOA -13.5791. mean (E−E0)/N Nov3 0.0332 QAOA 0.0309.
mean F Nov3 0.734 QAOA 0.725 (post hoc). start0 Nov3 4/8 QAOA 1/8.
T=762. seeds [10616, 10617, 10618, 10619, 10620, 10621, 10622, 10623].

## Secondary: historical L=3 q=1024 Q=3072 (different seeds, not paired)

- v7 10018–25: mean ΔE +0.0193 CI [-0.045, +0.075]
- spread8 equal4 10376–83: mean ΔE -0.0315 CI [-0.106, +0.043]
- anneal V4 10456–63: mean ΔE -0.0477 CI [-0.121, +0.022]

Not a paired test. Not an adoption. L=2 holdout FAIL stands. Compiler stream closed.

FAIL: stop further L=3 budget-increase retries. Do not climb to q=4096.

Fit workers 4, BLAS 1. Each completed ledger equals Q=6144.
