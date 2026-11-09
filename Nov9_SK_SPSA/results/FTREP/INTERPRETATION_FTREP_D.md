# FTREP D — N=15 center-4 energy edge is present at L=2 and gone at L≥3; GR unpatched

Characterization, not adoption. Same copied FT v4 center-4 recipe (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on one exclusive unread band **10264–10271**. N=15 only. L∈{2,3,4}, Q=1024·L. Four equal SPSA blocks, frozen gains, center-4 starts. Energy selects. F post hoc. No schedule tweak. L=2 present on this band does not authorize a GR write. Holdout 10144–51, local R19 10184–91, FT N18×16 10232–47, validation, and final unused.

**Per-L labels (report only). Do not patch GR.**

| L | Q | mean ΔE | CI | wins | mean F Nov3 | mean F QAOA | label |
|---|---|---|---|---|---|---|---|
| 2 | 2048 | −0.074 | [−0.143, −0.006] | 6/8 | 0.661 | 0.623 | **present** |
| 3 | 3072 | +0.059 | [−0.032, +0.146] | 2/8 | 0.692 | 0.702 | **absent/inconclusive** |
| 4 | 4096 | +0.056 | [−0.091, +0.195] | 3/8 | 0.685 | 0.691 | **absent/inconclusive** |

The edge is present at L=2 on this fresh band (same qualitative picture as development 10152–59 / 10168–75) and is absent/inconclusive at L=3 and L=4 (same qualitative picture as FT L=3 on 10018–25, LE L=3 on 10192–99, and the holdout washout at L=2 on a different band). Fidelity follows energy at L=2 (Nov3 higher) and does not at L≥3. The death of the edge is between L=2 and L=3 under this frozen recipe, not a slow fade through L=4.

48/48 fits ok. B* rejects 0. Charged 147456. Outside-cap 0. Recipe hash matched. Prior streams unchanged. No adoption, no GR patch, no candidacy revival, no final. Stop after this matrix.
