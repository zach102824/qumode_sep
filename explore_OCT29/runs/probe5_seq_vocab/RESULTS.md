# Opus probe 5 — W-atom seq vocab evolve

_Updated 2026-10-29 18:07:26 +0800 (Asia/Shanghai)_

Trials done: 13

## Promote gate (Fourier deepen)

- **q3_Ltrain6** n15 Med ≲ **7.7** @L12 / **5.99** @L20
- b_Med gate ≲ **1.240**
- Legacy probe2 n15≈30.9 (reject-only)

## Current seq champ (must beat Fourier to matter)

- motif: `fourier_proj_L16`
- b_Med=1.0798 b_B90=1.1319
- n9: 1.583/2.41
- n12: 2.356/3.772
- n15: 2.509/5.069
- vs Fourier L12 n15: champ=2.508528232574463 ceiling=7.7 BEATS/MATCHES

## Note

Promote only if n15 Med and b are within noise of q3_Ltrain6 (≲7.7 Med / b≲1.24). Mediocre projector mutants are rejected.
