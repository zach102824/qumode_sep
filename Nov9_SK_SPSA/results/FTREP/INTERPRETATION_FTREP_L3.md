# FTREP L3 — center-4 does not recover an L=3 energy edge; GR unpatched

The frozen v4 center-4 recipe won at L=2 (PR #92 N12/N15, PR #93 N18). FT’s own L=3 on seeds 10018–25 was inconclusive. This stream asks the same copied allocator (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) whether that L=2 win returns at L=3 Q=3072 on a fresh band. Seeds **10192–10199**. N∈{12,15} only. Four equal SPSA blocks, T=378, frozen gains, center-4 starts. Energy selects. F post hoc. No schedule tweak. No N18 L3. FT holdout 10144–10151 and local R19 10184–10191 unused.

**Both gates fail. Label: FAIL / inconclusive. Do not patch GR. Do not open N18 L3.**

| gate | N | mean ΔE (Nov3−QAOA) | CI | Nov3 lower | pass |
|---|---|---|---|---|---|
| L3-15 | 15 | +0.069 | [−0.066, +0.217] | 4/8 | no |
| L3-12 | 12 | −0.003 | [−0.116, +0.071] | 2/8 | no |

N15 mean favors QAOA; the CI covers zero and a large positive tail. N12 mean is a wash (2/8 Nov3 energy wins) and the CI also covers zero. Fidelity does not move with a Nov3 edge (N15 ΔF −0.003, 4/8; N12 ΔF +0.008, 4/8). Start 0 was selected on 3/8 Nov3 fits at N15 and 4/8 at N12; it does not rescue the energy comparison.

This is the same qualitative L=3 picture FT reported on 10018–25, now on a disjoint LE band with the L=2-winning recipe held fixed. The L=2 center-4 edge does not automatically persist at L=3.

32/32 fits ok. B* rejects 0. Charged 98304. Outside-cap 0. Recipe hash matched FT PR79. Prior streams unchanged. No N18 L3, no GR patch, no holdout touch, no final. Stop after this matrix.
