# idea2_cavity (graded H_c) L=2 seed=1 train{6,9}

U_c^X = W · exp(−i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W

E_in = Walsh strings inside one 3-bit cavity; E_x = strings touching ≥2 cavities.
Warm γ_in = γ_x = γ. Two classes only (no exact-span split). Motif: Hamming-shell + BinH X-sandwich.

| n | QAOA Med (same B) | Med | R | verdict | E_x frac | B |
|---|-------------------|-----|---|---------|----------|---|
| 6 | 5.0038 | 4.8506 | 0.9694 | BEAT | 0.550 | 100 |
| 9 | 11.2179 | 11.0051 | 0.9810 | TIE | 0.623 | 100 |
| 12 | 33.2021 | 32.8081 | 0.9881 | TIE | 0.665 | 100 |
| 15 | 71.6819 | 72.4380 | 1.0105 | TIE | 0.671 | 100 |

**ALL_CORE_BEAT** (same-B R≤0.98) = False

Kill (R grows with n: n15>n12>n9)? **True** (R = [0.9694, 0.981, 0.9881, 1.0105])

## γ drift

- γ_in = [-4.338764667510986, -6.688036918640137]
- γ_x  = [-4.113119125366211, -7.0581440925598145]
- rel |γ_in−γ_x| / |γ_in| = 0.05402769520878792
- unused extra freedom (rel<0.02)? **False**

