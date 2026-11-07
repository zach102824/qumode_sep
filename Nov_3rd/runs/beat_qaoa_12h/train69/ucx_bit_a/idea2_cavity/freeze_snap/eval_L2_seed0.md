# idea2_cavity (graded H_c) L=2 seed=0 train{6,9}

U_c^X = W · exp(−i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W

E_in = Walsh strings inside one 3-bit cavity; E_x = strings touching ≥2 cavities.
Warm γ_in = γ_x = γ. Two classes only (no exact-span split). Motif: Hamming-shell + BinH X-sandwich.

| n | QAOA Med (same B) | Med | R | verdict | E_x frac | B |
|---|-------------------|-----|---|---------|----------|---|
| 6 | 5.0038 | 4.9718 | 0.9936 | TIE | 0.550 | 100 |
| 9 | 11.2179 | 11.1033 | 0.9898 | TIE | 0.623 | 100 |
| 12 | 33.2021 | 32.3150 | 0.9733 | BEAT | 0.665 | 100 |
| 15 | 71.6819 | 72.6339 | 1.0133 | TIE | 0.671 | 100 |

**ALL_CORE_BEAT** (same-B R≤0.98) = False

Kill (R grows with n: n15>n12>n9)? **False** (R = [0.9936, 0.9898, 0.9733, 1.0133])

## γ drift

- γ_in = [-5.386687278747559, -4.783361434936523]
- γ_x  = [-4.0351104736328125, -7.239455223083496]
- rel |γ_in−γ_x| / |γ_in| = 0.37440043687820435
- unused extra freedom (rel<0.02)? **False**

