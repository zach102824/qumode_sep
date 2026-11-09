# N3INIT phenomenological G_{7,8}(0.10) sensitivity N=15 L=1

Frozen v4 center-4. NF=10. Illustrative \(W_\varepsilon=G_{7,8}(\varepsilon=0.10)\,W_{\rm BinH}\). Not a compiled-\(W_L\) model, not hardware-calibrated, not a hardware claim. SNAP phase 0 on f≥8. Seeds 10528–10535. No GR patch. No ε sweep.

**WL_ROBUST_L1** as a phenomenological Givens sensitivity label only (prespecified gate string preserved). See `ADDENDUM_AUDIT.md` and `Nov3_INIT/AUDIT_WL.md`.

New quantum queries 24576. Historical 2793472. Total research queries 2818048. Failures 0.
Classical B* wall 70.3s, CPU 70.3s, peak RSS 72008 KiB.

## Primary: G_{7,8}(0.10) Nov3 vs QAOA (same illustrative embedding)

Nov3−QAOA mean ΔE -0.3098 CI [-0.333, -0.283] wins 8/8. CI<0 True.
mean E Nov3 -11.0267 QAOA -10.7169. mean F Nov3 0.566 QAOA 0.457.
p_leak Nov3 2.354e-02 QAOA 2.348e-02. start0 Nov3 3/8 QAOA 0/8.

Selected E is unconditional ⟨H^ext⟩ (D4). Leakage is penalized, not postselected.

## Secondary: G_{7,8}(0.10) Nov3 vs ideal Nov3 only

non-ideal−ideal mean ΔE +0.3068 CI [+0.273, +0.331] (positive means this Givens residual hurts Nov3).
mean E non-ideal -11.0267 ideal -11.3335.
p_leak non-ideal 2.354e-02 ideal 4.774e-15.

This is Nov3 degradation only. There is no ideal QAOA arm on these seeds. Equal p_leak is not equal energy bias.

## p_leak

ideal Nov3 4.774e-15 [4.663e-15, 4.857e-15].
non-ideal Nov3 2.354e-02 [2.270e-02, 2.460e-02].
non-ideal QAOA 2.348e-02 [2.270e-02, 2.437e-02].

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024. No frozen SNAP–D compile of W_BinH exists; D5 unmodelled. No hardware physical-edge claim.
