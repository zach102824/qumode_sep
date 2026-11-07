# Opus Idea 2 — cavity-locality-graded γ (graded H_c)

_L=2 seed=0 train{6,9} B_train=100 CORE B=100 smoke_

**Formula.** $U_c^X = W\cdot\exp(-i[\gamma_{\ell,\mathrm{in}} E_{\mathrm{in}} + \gamma_{\ell,x} E_x])\cdot W$

- $E_{\mathrm{in}}$: Walsh / X-strings inside one 3-bit cavity (plus DC).
- $E_x$: strings that touch ≥2 cavities. **Not** split by exact span 1/2/3/4.
- Warm start: $\gamma_{\mathrm{in}}=\gamma_x=\gamma_{\mathrm{QAOA}}$, Hamming-shell SNAP $\varphi_k=-\beta\cdot k$.
- Motif locked: Nov_3rd Hamming-shell + X-sandwich BinH. Label: **graded H_c** (not non-$H_c$).
- Gate: **BEAT** iff $R=\mathrm{Med}/\mathrm{Med}_{\mathrm{QAOA}}\le 0.98$ on CORE $\{6,9,12,15\}$.

## Med / R (same-B smoke)

| n | QAOA Med | Med | R | verdict | E_x frac |
|---|----------|-----|---|---------|----------|
| 6 | 5.0038 | 4.9718 | **0.9936** | TIE | 0.550 |
| 9 | 11.2179 | 11.1033 | **0.9898** | TIE | 0.623 |
| 12 | 33.2021 | 32.3150 | **0.9733** | BEAT | 0.665 |
| 15 | 71.6819 | 72.6339 | **1.0133** | TIE | 0.671 |

**ALL_CORE_BEAT** = False

## γ

| layer | γ_in | γ_x | |γ_in−γ_x| |
|-------|------|-----|-----------|
| 0 | -5.3867 | -4.0351 | 1.3516 |
| 1 | -4.7834 | -7.2395 | 2.4561 |

rel drift = 0.37440043687820435; unused (γ_in≈γ_x)? **False**

## Kill checks

- Extra freedom unused (γ collapse): False
- R steadily worse with n (n15>n12>n9): False
- L=2 near-miss bar (~0.986): n12 R=0.9733, n15 R=1.0133

