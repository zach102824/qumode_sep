# FTREP H — FT L=2 holdout instability confirmed on LE; GR unpatched

FT PR79 `bd9effe3f` reported L=2 holdout FAIL on unread seeds **10144–10151**: H1 N15 Q=2048 mean ΔE −0.031 CI [−0.083, +0.014] 4/8, CI crosses 0. This stream is an independent LE replay of that exact band with the same copied v4 center-4 recipe (`alloc.py` sha256 `de595de592f33a1d` / `347ac23369db`). Intentional replay, not a collision. N=15 L=2 Q=2048 is Gate R. N=18 L=2 Q=2048 ran as the optional report (wall allowed). Energy selects. F post hoc. No schedule tweak. No candidacy revival. FT development 10018–25, local R19 10184–91, validation, and final unused.

**Gate R fails. Label: HOLDOUT_INSTABILITY CONFIRMED on LE. Do not patch GR.**

| cell | engine | mean ΔE | CI | Nov3 lower | pass |
|---|---|---|---|---|---|
| H1 / Gate R N15 Q=2048 | FT `bd9effe3f` | −0.031 | [−0.083, +0.014] | 4/8 | no |
| H1 / Gate R N15 Q=2048 | LE this stream | −0.031 | [−0.086, +0.017] | 4/8 | no |
| H2 N18 Q=2048 (report) | FT `bd9effe3f` | −0.065 | [−0.184, +0.045] | 5/8 | no |
| H2 N18 Q=2048 (report) | LE this stream | −0.065 | [−0.185, +0.043] | 5/8 | no |

Per-seed ΔE on both N=15 and N=18 matches the published FT table to displayed precision (LE−FT = 0.00000 on every seed). Selected starts match. The holdout washout is the same eight signed outcomes, not an FT-only fluke.

The L=2 center-4 edge that replicated on LE development bands (10152–59, 10168–75) does not survive this holdout band, on either engine. That is why GR stays unpatched: the remaining local R19 gate is still outstanding, and this replay confirms the published holdout failure rather than overturning it.

32/32 fits ok (16 N15 + 16 N18). B* rejects 0. Charged 65536. Outside-cap 0. Recipe hash matched. Prior streams unchanged. No candidacy revival, no GR patch, no final. Stop after this matrix.
