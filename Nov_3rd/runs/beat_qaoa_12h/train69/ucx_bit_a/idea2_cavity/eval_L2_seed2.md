# idea2_cavity (graded H_c) L=2 seed=2 train{6,9}

U_c^X = W · exp(−i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W

E_in = Walsh strings inside one 3-bit cavity; E_x = strings touching ≥2 cavities.
Warm γ_in = γ_x = γ. Two classes only (no exact-span split). Motif: Hamming-shell + BinH X-sandwich.

| n | QAOA Med (same B) | Med | R | verdict | E_x frac | B |
|---|-------------------|-----|---|---------|----------|---|
| 6 | 5.0038 | 4.8326 | 0.9658 | BEAT | 0.550 | 100 |
| 9 | 11.2179 | 10.8213 | 0.9646 | BEAT | 0.623 | 100 |
| 12 | 33.2021 | 32.2314 | 0.9708 | BEAT | 0.665 | 100 |
| 15 | 71.6819 | 70.4755 | 0.9832 | TIE | 0.671 | 100 |

**ALL_CORE_BEAT** (same-B R≤0.98) = False

Kill (R grows with n: n15>n12>n9)? **True** (R = [0.9658, 0.9646, 0.9708, 0.9832])

## γ drift

- γ_in = [-4.271854400634766, -6.780664920806885]
- γ_x  = [-4.1328229904174805, -6.9610981941223145]
- rel |γ_in−γ_x| / |γ_in| = 0.02890424244105816
- unused extra freedom (rel<0.02)? **False**

