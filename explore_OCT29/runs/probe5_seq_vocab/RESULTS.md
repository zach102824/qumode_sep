# Opus probe 5 — W-atom seq vocab evolve

_Updated 2026-10-29 17:50:41 +0800 (Asia/Shanghai)_

Trials done: 2

## Baselines to beat

- Probe2 projector L4 train{9,12}: n15 Med≈30.9, b≈1.41
- Probe3 Fourier L12: n15 Med≈9.67

## Current champion

- motif: `proj_cost_proj`
- b_Med=1.3931 b_B90=1.2543
- n9: 5.146/17.91
- n12: 14.26/24.6
- n15: 37.62/69.72
- vs Fourier L12 n15: champ=? ceiling=9.67

## Note

Promote on fitted b over {9,12,15}; reject if Med15 worse than probe2 projector unless b clearly better.
