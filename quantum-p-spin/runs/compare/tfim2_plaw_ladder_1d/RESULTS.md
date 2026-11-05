# 2-body power-law TFIM shared_gb ladder (open 1D chain)

## Hamiltonian

$$H = -\sum_{i<j} J_{ij} Z_i Z_j - \Gamma\sum_i X_i,\quad J_{ij}=\xi_{ij}\, r_{ij}^{-\alpha}/N$$

- $r_{ij}=|i-j|$ (open 1D chain; no wrap-around)
- $N=(\sum_{i<j} r^{-\alpha})/n$ (Kac; classical ferro $E=-n$)
- $\xi_{ij}=\pm 1$ Rademacher (signed)
- No glass baseline in this ladder

## Protocol

- `shared_gb`: one shared γ + β_ℓ (L+1 params), `φ_k=-βk` Hamming-shell SNAP
- cost `⟨H⟩`, LBFGS OUTER=40×max_iter=5, lr=0.25, init `+0.01`
- N=16 instances/cell; n∈[6, 9, 12]; L∈[2, 4, 8, 16]; α∈[0.25, 1.0, 1.5, 2.0]; Γ=0.5

## Median fidelity ladders (per α)

### α=0.25, Γ=0.5

| n \ L | 2 | 4 | 8 | 16 |
|--:|---:|---:|---:|---:|
| **6** | 0.8305 | 0.9574 | 0.9557 | 0.8194 |
| **9** | 0.7992 | 0.9160 | 0.9404 | 0.8039 |
| **12** | 0.8258 | 0.9207 | 0.9161 | 0.8079 |

### α=1, Γ=0.5

| n \ L | 2 | 4 | 8 | 16 |
|--:|---:|---:|---:|---:|
| **6** | 0.7903 | 0.9244 | 0.9384 | 0.6551 |
| **9** | 0.6948 | 0.8726 | 0.8231 | 0.6774 |
| **12** | 0.7293 | 0.8670 | 0.7939 | 0.7095 |

### α=1.5, Γ=0.5

| n \ L | 2 | 4 | 8 | 16 |
|--:|---:|---:|---:|---:|
| **6** | 0.7313 | 0.8924 | 0.8484 | 0.5509 |
| **9** | 0.6021 | 0.7827 | 0.6218 | 0.5048 |
| **12** | 0.5416 | 0.6755 | 0.6127 | 0.6141 |

### α=2, Γ=0.5

| n \ L | 2 | 4 | 8 | 16 |
|--:|---:|---:|---:|---:|
| **6** | 0.6275 | 0.8104 | 0.6830 | 0.4687 |
| **9** | 0.4813 | 0.6402 | 0.5839 | 0.4481 |
| **12** | 0.3711 | 0.5190 | 0.3529 | 0.3630 |

## Median half-cut entanglement S of GS

| n \ α | 0.25 | 1 | 1.5 | 2 |
|--:|---:|---:|---:|---:|
| **6** | 0.6715 | 0.6575 | 0.6527 | 0.6650 |
| **9** | 0.7598 | 0.6525 | 0.6465 | 0.6657 |
| **12** | 0.8525 | 0.6693 | 0.6184 | 0.6713 |

## Notes

- **Does L help?** Mildly up to L≈4–8, then **hurts** at L=16 under shared_gb + cold init (optimizer traps). Best cell often L=4.
- **Is n6 harder than n9/12?** **No** — opposite of 3-body: at α≥1.5, F falls with n (n6 easiest, n12 hardest).
- **Half-cut S:** stays ~0.65–0.85; grows with n mainly at α=0.25 (long-range); not the driver of the α=2 hardness trend.

## Example J matrix (n=6, α=1.5, seed=0)

Open 1D chain $r_{|i-j|}$, $N\approx 1.222$:

```
n=6 α=1.5 Γ=0.5 seed=0  open 1D chain r=|i-j|
N_norm = 1.221834  E0=-5.654189
J matrix:
[[ 0.      0.8184  0.2894  0.1575 -0.1023 -0.0732]
 [ 0.8184  0.     -0.8184 -0.2894 -0.1575 -0.1023]
 [ 0.2894 -0.8184  0.      0.8184  0.2894  0.1575]
 [ 0.1575 -0.2894  0.8184  0.      0.8184  0.2894]
 [-0.1023 -0.1575  0.2894  0.8184  0.      0.8184]
 [-0.0732 -0.1023  0.1575  0.2894  0.8184  0.    ]]

pairs (i,j), r, ξ, J:
  (0,1)  r=1  w=1.0000  ξ=+1  J=+0.8184
  (0,2)  r=2  w=0.3536  ξ=+1  J=+0.2894
  (0,3)  r=3  w=0.1925  ξ=+1  J=+0.1575
  (0,4)  r=4  w=0.1250  ξ=-1  J=-0.1023
  (0,5)  r=5  w=0.0894  ξ=-1  J=-0.0732
  (1,2)  r=1  w=1.0000  ξ=-1  J=-0.8184
  (1,3)  r=2  w=0.3536  ξ=-1  J=-0.2894
  (1,4)  r=3  w=0.1925  ξ=-1  J=-0.1575
  (1,5)  r=4  w=0.1250  ξ=-1  J=-0.1023
  (2,3)  r=1  w=1.0000  ξ=+1  J=+0.8184
  (2,4)  r=2  w=0.3536  ξ=+1  J=+0.2894
  (2,5)  r=3  w=0.1925  ξ=+1  J=+0.1575
  (3,4)  r=1  w=1.0000  ξ=+1  J=+0.8184
  (3,5)  r=2  w=0.3536  ξ=+1  J=+0.2894
  (4,5)  r=1  w=1.0000  ξ=+1  J=+0.8184
```

## Comparison to 3-body α=1.5 signed plaw (shared_gb, Γ=0.5)

| model | n6 L2 | n9 L2 | n12 L2 | L helps? | n trend |
|:------|------:|------:|-------:|:---------|:--------|
| 3-body ring | 0.451 | 0.906 | 0.979 | no (flat) | **easier** as n↑ (finite-size peak at n6) |
| 2-body 1D chain (this) | 0.731 | 0.602 | 0.542 | mild peak at L=4 then **worse** at L=16 | **harder** as n↑ (esp. α≥1.5) |

2-body open-chain power-law is the sensible control: hardness grows with n and with α (shorter range), half-cut S is O(1) and does not explode, and L=4 is typically best while deeper L overfits/traps under this init+LBFGS. The 3-body α=1.5 pattern (n6 hard, n≥9 easy, L flat) is **not** reproduced — that was a 3-body finite-size quirk, not generic TFIM hardness.


Generated 2026-11-05 22:08:12 +08
