# N3INIT Fock-cutoff validation N=15 L=1,2

Frozen v4 center-4. D0=NF=8, D0+2=NF=10, ideal W_L identity pad. Identity pad is algebraic invariance of H^{⊗3} ⊕ I, not a SNAP–D pulse-compile cutoff test. Compiled W_L not modelled. Seeds 10520–10527. No GR patch. See ADDENDUM_AUDIT.md.

**CUTOFF_ROBUST_L1.**

New quantum queries 98304. Historical 2695168. Total research queries 2793472. Failures 0.
Classical B* wall 68.8s, CPU 68.7s, peak RSS 71944 KiB.

Secondary |mean ΔE(D0+2)−mean ΔE(D0)| at L=1: 4.440892098500626e-16.
L=2 sign concordance (exploratory): True.

p_leak is 1−P(all cavities in Fock 0..7). At D0=8, p_highest is occupancy of code level 7, not pad leakage. At D0+2, p_highest is occupancy of Fock 9 (the pad edge). Compiled W_L is not modelled.

## L=1 D0=8 primary

Nov3 vs QAOA mean ΔE -0.3775 CI [-0.438, -0.331] wins 8/8. CI<0 True.
mean E Nov3 -11.0280 QAOA -10.6505. mean F Nov3 0.592 QAOA 0.455.
Nov3 p_leak 4.746e-15 mean n/cavity 3.5000 p_highest 4.669e-01. start0 Nov3 1/8 QAOA 0/8.

## L=1 D0+2=10 primary

Nov3 vs QAOA mean ΔE -0.3775 CI [-0.439, -0.330] wins 8/8. CI<0 True.
mean E Nov3 -11.0280 QAOA -10.6505. mean F Nov3 0.592 QAOA 0.455.
Nov3 p_leak 4.774e-15 mean n/cavity 3.5000 p_highest 4.843e-15. start0 Nov3 1/8 QAOA 0/8.

## L=2 D0=8 exploratory

Nov3 vs QAOA mean ΔE -0.1653 CI [-0.243, -0.081] wins 7/8. CI<0 True.
mean E Nov3 -11.1334 QAOA -10.9681. mean F Nov3 0.627 QAOA 0.546.
Nov3 p_leak 9.631e-15 mean n/cavity 3.5000 p_highest 4.627e-01. start0 Nov3 2/8 QAOA 3/8.

## L=2 D0+2=10 exploratory

Nov3 vs QAOA mean ΔE -0.1653 CI [-0.240, -0.076] wins 7/8. CI<0 True.
mean E Nov3 -11.1334 QAOA -10.9681. mean F Nov3 0.627 QAOA 0.546.
Nov3 p_leak 9.659e-15 mean n/cavity 3.5000 p_highest 9.631e-15. start0 Nov3 2/8 QAOA 3/8.

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024·L. Ideal W_L only; no hardware claim. Pulse-compile cutoff convergence remains unvalidated.
