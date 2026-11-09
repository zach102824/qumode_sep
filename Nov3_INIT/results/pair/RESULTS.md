# N3INIT pair-shell probe N=15 L=2,3

In-family per-cavity SNAP shells. Paper V(M) not implemented. Seeds 10328–10335. Same Q=1024·L across arms. No GR patch. No candidacy file. Holdout FAIL stands.

**REJECT_PAIR_SHELL.**

New quantum queries 122880. Historical 2179072. Total research queries 2301952. Failures 0.
Classical B* wall 65.2s, CPU 65.1s, peak RSS 71836 KiB.

Param counts N=15: QAOA 2L, shared Nov3 3+3L, pair-cavity 3+15L. L=2: 4 / 9 / 33. L=3: 6 / 12 / 48.

## L=3 primary

pair vs QAOA mean ΔE +0.3061 CI [+0.127, +0.519] wins 1/8.
pair vs shared mean ΔE +0.3960 CI [+0.189, +0.621] pair mean E -12.0137 shared -12.4096 QAOA -12.3197.
mean F pair 0.576 shared 0.659 QAOA 0.602. start0 pair 8/8 shared 3/8 QAOA 0/8.

## L=2 sanity

pair vs QAOA mean ΔE +0.0049 CI [-0.078, +0.079] wins 3/8.
pair vs shared mean ΔE +0.1269 CI [+0.065, +0.202] pair mean E -12.2065 shared -12.3335 QAOA -12.2114.
mean F pair 0.567 shared 0.632 QAOA 0.568. start0 pair 8/8 shared 4/8 QAOA 1/8.

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024·L.
