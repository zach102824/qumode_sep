# WalkSAT R=50 confirm (vs paper R=20)

_Generated 2026-10-31 03:15:41 +0800 (Asia/Shanghai)_

Same packs: `data/scale1000` split=`scale1000`, max_inst=200, p=0.5.
Non-depth idea after L48 STOP. Units = flips/evals (not 1/p).

| n | R20 flips Med/B90 | R50 flips Med/B90 | R20 evals | R50 evals | ΔMed flips |
|---|-------------------|-------------------|-----------|-----------|------------|
| 9 | 12.5/40.1 | 11.5/34.5 | 51.0/161.4 | 47.0/139.2 | -8.0% |
| 12 | 26.5/65.9 | 26.2/60.5 | 107.0/264.6 | 106.0/243.2 | -0.9% |
| 15 | 43.8/128.1 | 42.5/120.1 | 176.0/513.6 | 171.0/481.4 | -2.9% |
| 18 | 71.5/203.6 | 67.8/210.9 | 287.0/815.4 | 272.0/844.8 | -5.2% |

**Verdict:** R=50 confirms R=20 order-of-magnitude; Med flips within ~few % (sampling noise). Paper table R=20 numbers remain fine.

