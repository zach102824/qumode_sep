# idea2_cavity (graded H_c) L=2 seed=0 train{6,9}

U_c^X = W · exp(−i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W

E_in = Walsh strings inside one 3-bit cavity; E_x = strings touching ≥2 cavities.
Warm γ_in = γ_x = γ. Two classes only (no exact-span split). Motif: Hamming-shell + BinH X-sandwich.

| n | QAOA Med (same B) | Med | R | verdict | E_x frac | B |
|---|-------------------|-----|---|---------|----------|---|
| 6 | 5.0038 | 5.0038 | 1.0000 | TIE | 0.550 | 100 |
| 9 | 11.2179 | 11.2179 | 1.0000 | TIE | 0.623 | 100 |
| 12 | 33.2021 | 33.2021 | 1.0000 | TIE | 0.665 | 100 |
| 15 | 71.6819 | 71.6819 | 1.0000 | TIE | 0.671 | 100 |

**ALL_CORE_BEAT** (same-B R≤0.98) = False

Kill (R grows with n: n15>n12>n9)? **False** (R = [1.0, 1.0, 1.0, 1.0])

