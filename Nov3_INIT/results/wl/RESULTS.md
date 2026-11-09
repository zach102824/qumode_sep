# N3INIT non-ideal W_L leakage probe N=15 L=1

Frozen v4 center-4. NF=10. W_ε = G_{7,8}(ε=0.10) W_BinH. SNAP phase 0 on f≥8. Seeds 10528–10535. No GR patch. Not a hardware claim.

**WL_ROBUST_L1.**

New quantum queries 24576. Historical 2793472. Total research queries 2818048. Failures 0.
Classical B* wall 70.3s, CPU 70.3s, peak RSS 72008 KiB.

## Primary: non-ideal Nov3 vs QAOA (same W_ε)

Nov3−QAOA mean ΔE -0.3098 CI [-0.333, -0.283] wins 8/8. CI<0 True.
mean E Nov3 -11.0267 QAOA -10.7169. mean F Nov3 0.566 QAOA 0.457.
p_leak Nov3 2.354e-02 QAOA 2.348e-02. start0 Nov3 3/8 QAOA 0/8.

## Secondary: non-ideal Nov3 vs ideal Nov3

non-ideal−ideal mean ΔE +0.3068 CI [+0.273, +0.331] (positive means non-ideal hurts Nov3).
mean E non-ideal -11.0267 ideal -11.3335.
p_leak non-ideal 2.354e-02 ideal 4.774e-15.

## p_leak

ideal Nov3 4.774e-15 [4.663e-15, 4.857e-15].
non-ideal Nov3 2.354e-02 [2.270e-02, 2.460e-02].
non-ideal QAOA 2.348e-02 [2.270e-02, 2.437e-02].

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024. SNAP–D compile of W_L remains unmodelled. No hardware physical-edge claim.
