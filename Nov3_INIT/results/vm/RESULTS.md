# N3INIT paper V(M) architecture N=15 L=2,3

Separate architecture arm. Paper V(M) p=L with shared Rx mixer. Not Nov3 shells. Seeds 10480–10487. Same Q=1024·L across arms. No GR patch. No candidacy file. Holdout FAIL stands.

**REJECT_VM_L3.**

New quantum queries 122880. Historical 2572288. Total research queries 2695168. Failures 0.
Classical B* wall 64.8s, CPU 64.7s, peak RSS 71844 KiB.

Param counts N=15: QAOA 2L, shared Nov3 3+3L, V(M) 106L. L=2: 4 / 9 / 212. L=3: 6 / 12 / 318.

## L=3 primary

V(M) vs QAOA mean ΔE +1.1038 CI [+0.835, +1.413] wins 0/8.
V(M) vs Nov3 mean ΔE +1.1127 CI [+0.838, +1.413] V(M) mean E -12.9971 Nov3 -14.1098 QAOA -14.1009.
mean F V(M) 0.482 Nov3 0.704 QAOA 0.677. start0 V(M) 8/8 Nov3 5/8 QAOA 2/8.

## L=2 sanity

V(M) vs QAOA mean ΔE +1.0693 CI [+0.848, +1.278] wins 0/8.
V(M) vs Nov3 mean ΔE +1.1449 CI [+0.940, +1.359] V(M) mean E -12.9434 Nov3 -14.0883 QAOA -14.0126.
mean F V(M) 0.472 Nov3 0.684 QAOA 0.638. start0 V(M) 8/8 Nov3 3/8 QAOA 3/8.

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024·L.
