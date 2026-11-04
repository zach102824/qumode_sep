# idea2_cavity (graded H_c) L=2 seed=0 train{6,9}

U_c^X = W · exp(−i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W

E_in = Walsh strings inside one 3-bit cavity; E_x = strings touching ≥2 cavities.
Warm γ_in = γ_x = γ. Two classes only (no exact-span split). Motif: Hamming-shell + BinH X-sandwich.

| n | QAOA Med (same B) | Med | R | verdict | E_x frac | B |
|---|-------------------|-----|---|---------|----------|---|
| 6 | 5.0038 | 4.6905 | 0.9374 | BEAT | 0.550 | 100 |
| 9 | 11.2179 | 10.9854 | 0.9793 | BEAT | 0.623 | 100 |
| 12 | 33.2021 | 32.2243 | 0.9705 | BEAT | 0.665 | 100 |
| 15 | 71.6819 | 73.7615 | 1.0290 | UNDER | 0.671 | 100 |

**ALL_CORE_BEAT** (same-B R≤0.98) = False

Kill (R grows with n: n15>n12>n9)? **False** (R = [0.9374, 0.9793, 0.9705, 1.029])

## γ drift

- γ_in = [-4.385346412658691, -6.700121879577637]
- γ_x  = [-4.070654392242432, -7.059878349304199]
- rel |γ_in−γ_x| / |γ_in| = 0.060840774327516556
- unused extra freedom (rel<0.02)? **False**

