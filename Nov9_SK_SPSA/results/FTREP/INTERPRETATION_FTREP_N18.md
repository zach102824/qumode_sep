# FTREP N18 — center-4 L=2 energy edge scales to N=18; GR unpatched

FT-L2 REPLICATED on LE (PR #92, seeds 10152–10159, N∈{12,15}). This stream is the prespecified **N=18 L=2 scaling check** of that same copied FT v4 recipe. `alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db` unchanged. Four equal SPSA blocks, T=250, Q=2048, frozen gains, center-4 starts. Energy selects. F post hoc. Seeds **10168–10175**. No schedule tweak. No L3. FT holdout 10144–10151 unused.

**Gate N18 passes. Label: N18-L2 SCALED. Do not patch GR.**

| gate | N | mean ΔE (Nov3−QAOA) | CI | Nov3 lower | pass |
|---|---|---|---|---|---|
| R2 (prior) | 12 | −0.107 | [−0.188, −0.030] | 6/8 | yes |
| R1 (prior) | 15 | −0.071 | [−0.136, −0.013] | 5/8 | yes |
| N18 | 18 | −0.109 | [−0.179, −0.036] | 6/8 | yes |

The N=18 mean and CI sit with the N=12 replicate, not below the N=15 floor. The edge did not fade. Two seeds (10172, 10174) still favor QAOA on energy; the bootstrap mean CI is entirely below 0 on the prespecified 8-seed band.

Start 0 (the coherent center) was selected on 5/8 Nov3 fits and 0/8 QAOA fits. Fidelity moved with energy (ΔF +0.055, CI [+0.029, +0.082], 7/8) and did not select.

16/16 fits ok. B* rejects 0. Charged 32768. Outside-cap 0. Recipe hash matched FT PR79. Prior streams unchanged. No L3, no GR patch, no holdout touch, no final. Stop after this matrix.
