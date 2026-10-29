# Opus probe 5 — W-atom seq vocab evolve

_Updated 2026-10-29 20:30 CST — STOPPED per Opus 20:25_

Trials done: ~197 then `STOP_PROBE5`.

## Why stopped

190+ trials produced **no new motif** (only SNAP/PROJ rearrangements). Arm A re-eval showed the champ's val n15 Med≈2.51 was selection-biased; **test n15 Med≈6.33 / B90≈24**. Further trials add winner's-curse noise. Top configs kept in `top5_for_reval.json` for val2 re-score later.

## Last champ (val tiny-B — retired yardstick)

- motif: `fourier_proj_L16`
- b_Med=1.0798 (retired)
- n9: 1.583/2.41 · n12: 2.356/3.772 · n15: 2.509/5.069
