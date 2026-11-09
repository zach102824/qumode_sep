# FTREP D18 — N=18 center-4 death still between L=2 and L=3; GR unpatched

Characterization, not adoption. Same copied FT v4 center-4 recipe (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10320–10327**. N=18. L=2,3 required; L=4 ran after they finished (wall allowed, not truncated). Q=1024·L. Four equal SPSA blocks, frozen gains, center-4 starts. Energy selects. F post hoc. No schedule tweak. L=2 present on this band does not authorize a GR write. Local R23 10304–19, FT 10232–47, LE ≤10271, holdout 10144–51, validation, and final unused.

**Per-L labels (report only). Vs N15: death still between L2 and L3. Do not patch GR.**

| N | L | Q | mean ΔE | CI | wins | mean F Nov3 | mean F QAOA | label |
|---|---|---|---|---|---|---|---|---|
| 15 (D) | 2 | 2048 | −0.074 | [−0.143, −0.006] | 6/8 | 0.661 | 0.623 | present |
| 15 (D) | 3 | 3072 | +0.059 | [−0.032, +0.146] | 2/8 | 0.692 | 0.702 | absent/inconclusive |
| 15 (D) | 4 | 4096 | +0.056 | [−0.091, +0.195] | 3/8 | 0.685 | 0.691 | absent/inconclusive |
| 18 | 2 | 2048 | −0.053 | [−0.104, −0.001] | 5/8 | 0.657 | 0.622 | **present** |
| 18 | 3 | 3072 | +0.008 | [−0.101, +0.094] | 3/8 | 0.671 | 0.657 | **absent/inconclusive** |
| 18 | 4 | 4096 | +0.034 | [−0.051, +0.128] | 4/8 | 0.704 | 0.700 | **absent/inconclusive** |

N=18 L=2 is present but tighter than N=15 (CI upper bound −0.001). L=3 and L=4 CIs cover 0, as at N=15. The L2→L3 death persists at larger N; it is not an N=15-only cutoff. Fidelity tracks energy at L=2 and is mixed at L≥3.

48/48 fits ok (32 required + 16 L=4). B* rejects 0. Charged 147456. Outside-cap 0. Recipe hash matched. Prior streams unchanged. No adoption, no GR patch, no candidacy revival, no final. Stop after this matrix.
