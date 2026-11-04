# RESULTS — WT shared Walsh-degree tilt + power_p

**Verdict: kill**

Formula: \(U_c^X = W\exp(-i\sum_{w=1}^{4} \gamma_\ell \tau_w f_w) W\), \(\tau_1\equiv 1\), \(\tau_2,\tau_3,\tau_4\) shared across layers and \(n\). SNAP is the power_p cubic \(a(n)=(8/(n-1))^p\), unchanged. Graded \(H_c\) (Walsh degree), not a non-\(H_c\) gadget.

Train {6, 9} only (fresh250 packs, disjoint from scale1000). CORE {6, 9, 12, 15} B=1000.

## Gates

kill if every L=2 seed is worse than Walsh seed1 n15, or every L=16 seed has all |τ−1|<0.05, or every L=16 seed fails to grow the SNAP−QAOA gap; ship only if the best L=2 seed beats Walsh seed1 n15 and at least one L=16 seed moves τ, grows that gap, and beats both graded Fourier QAOA (same τ) and ungraded Fourier QAOA at n15; otherwise needs-edit

Reasons:
- every L=2 seed is worse than Walsh seed1 at n15

Gap without τ (ungraded Fourier QAOA Med − champ Med) at n15 = **-0.1937**.
Gap with τ = graded Fourier QAOA Med − WT Med, same τ. Larger means SNAP pulls further ahead of the matched qubit QAOA.

## Reading

Do not ship. The pre-registered L=2 gate fires: all 3 seeds are worse than Walsh seed1 at n15 (best WT Med 74.42 vs 72.80) and at n12 (best 28.53 vs 28.04). τ did move at L=2 (max |τ_w−1| = 0.08–0.10), so this is not the collapse kill.

The other two kill modes do not fire. At L=16, τ stays away from 1 on every seed (max |τ_w−1| = 0.22–0.26), and the SNAP−QAOA gap grows on every seed (gap without τ = −0.194; gap with τ = +0.048 to +0.072). All 3 L=16 seeds beat the power_p champ (2.579), graded Fourier QAOA with the same τ (about 2.427), and ungraded Fourier QAOA (2.385) at n15. Best L=16 n15 is seed 1 at 2.356. Sharing one tilt across layers gives up the shallow Walsh win.

## Champ recovery (L=16, τ=1)

Recovered = True (max |Med_τ=1 − Med_scalar| = 9.537e-07).

| n | scalar Med | τ=1 Med | |Δ| |
|---|------------|---------|-----|
| 6 | 1.1503 | 1.1503 | 2.384e-07 |
| 9 | 1.4183 | 1.4183 | 9.537e-07 |
| 12 | 1.8389 | 1.8389 | 5.364e-07 |
| 15 | 2.5791 | 2.5791 | 3.576e-07 |

Published champ n15 Med = 2.5791. This-run scalar n15 Med = 2.5791.

## L=2 vs Walsh seed1

Walsh seed1 (graded γ_w, smoke-trained) n12 Med = 28.0352, n15 Med = 72.7966 (knee R ≈ 0.9567 vs sealed QAOA).

| seed | n6 | n9 | n12 | n15 | n15 vs Walsh | τ | τ dev | collapsed | p |
|------|----|----|-----|-----|--------------|---|-------|-----------|---|
| 0 | 5.1059 | 12.3549 | 28.8100 | 74.9589 | worse | 1.000, 1.073, 1.088, 0.934 | 0.0877 | False | 0.052 |
| 1 | 5.1070 | 12.3126 | 28.8612 | 74.6786 | worse | 1.000, 1.060, 1.082, 0.935 | 0.0823 | False | 0.048 |
| 2 | 5.1193 | 12.3246 | 28.5320 | 74.4219 | worse | 1.000, 1.085, 1.105, 0.958 | 0.1049 | False | 0.044 |

## L=16 vs champ, ungraded Fourier QAOA, graded Fourier QAOA

Ungraded Fourier QAOA (published Q=6, re-evaluated here) n15 Med = **2.3853** (published 2.3853).
power_p champ n15 Med used for the gap = **2.5791**.

| seed | WT n15 | champ n15 | ungraded FQ n15 | graded FQ n15 | gap with τ | gap grew | τ dev | collapsed | beats graded | beats ungraded |
|------|--------|-----------|-----------------|---------------|------------|----------|-------|-----------|--------------|----------------|
| 0 | 2.3795 | 2.5791 | 2.3853 | 2.4279 | 0.0484 | True | 0.2552 | False | True | True |
| 1 | 2.3560 | 2.5791 | 2.3853 | 2.4280 | 0.0720 | True | 0.2208 | False | True | True |
| 2 | 2.3778 | 2.5791 | 2.3853 | 2.4261 | 0.0483 | True | 0.2331 | False | True | True |

### L=16 full CORE (WT)

| seed | n | Med | B90 | fresh QAOA Med | R |
|------|---|-----|-----|----------------|---|
| 0 | 6 | 1.1403 | 1.5495 | 1.2331 | 0.9248 |
| 0 | 9 | 1.3952 | 2.6931 | 1.5879 | 0.8787 |
| 0 | 12 | 1.7754 | 4.4411 | 2.0854 | 0.8514 |
| 0 | 15 | 2.3795 | 7.6040 | 2.9127 | 0.8170 |
| 1 | 6 | 1.1420 | 1.5601 | 1.2331 | 0.9261 |
| 1 | 9 | 1.3935 | 2.6867 | 1.5879 | 0.8776 |
| 1 | 12 | 1.7500 | 4.4582 | 2.0854 | 0.8392 |
| 1 | 15 | 2.3560 | 7.4527 | 2.9127 | 0.8089 |
| 2 | 6 | 1.1373 | 1.5596 | 1.2331 | 0.9223 |
| 2 | 9 | 1.3923 | 2.7325 | 1.5879 | 0.8768 |
| 2 | 12 | 1.7589 | 4.4780 | 2.0854 | 0.8435 |
| 2 | 15 | 2.3778 | 7.5711 | 2.9127 | 0.8164 |

### Graded Fourier QAOA CORE

| seed | n | Med | B90 |
|------|---|-----|-----|
| 0 | 6 | 1.1738 | 1.9143 |
| 0 | 9 | 1.4544 | 3.0926 |
| 0 | 12 | 1.8158 | 4.9004 |
| 0 | 15 | 2.4279 | 7.7978 |
| 1 | 6 | 1.1909 | 1.8256 |
| 1 | 9 | 1.4548 | 2.9395 |
| 1 | 12 | 1.8268 | 4.7214 |
| 1 | 15 | 2.4280 | 7.8530 |
| 2 | 6 | 1.1726 | 1.9332 |
| 2 | 9 | 1.4597 | 3.0823 |
| 2 | 12 | 1.8423 | 4.9391 |
| 2 | 15 | 2.4261 | 7.8905 |

## Protocol

- L=2: QAOA warm (sealed paper L=2 angles) + power_p (c2=c3=1e-4, p=0) + τ=1. Adam 600 × 1e-3, L-BFGS 12×8, batch 32, w6=1, w9=2, 20% n9 holdout. The warm point is scored first.
- L=16: warm start is the fresh250 power_p champ with τ=1. Same optimizer. Ratio loss uses the champ's stored fresh250 QAOA angles.
- Graded Fourier QAOA: Q=6 DST/DCT, frozen τ from that WT seed, warm-started from the published ungraded Fourier modes, Adam 600 × 0.05 + L-BFGS 12×8, same train packs.
- Ungraded Fourier QAOA number is a re-eval of the FM-WS Q=6 L=16 checkpoint, not a retrain.

_Finished 2026-11-04 09:57:13 +0000_
