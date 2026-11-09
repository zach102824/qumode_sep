# FTREP — FT v4 center-4 L=2 energy edge replicates on LE; GR unpatched

FT PR79 (`347ac23369db`, `Nov3_INIT/alloc.py` sha256 `de595de5…`) reported a Nov3 energy edge vs equal-layer QAOA at L=2 under frozen v4 **center-4** (four equal SPSA blocks, Q=qL). This stream copied that allocator byte-for-byte onto LE and re-ran it on unread seeds **10152–10159**, N∈{12,15}, L=2, Q=2048. Not a new schedule. Not S1JOINT. Not warm-start. FT holdout 10144–10151 unused.

**Both gates pass. Label: FT-L2 REPLICATED on LE. Do not patch GR.**

| gate | N | mean ΔE (Nov3−QAOA) | CI | Nov3 lower | pass |
|---|---|---|---|---|---|
| R1 | 15 | −0.071 | [−0.136, −0.013] | 5/8 | yes |
| R2 | 12 | −0.107 | [−0.188, −0.030] | 6/8 | yes |

FT’s own N15 L2 Q=2048 development number on seeds 10018–25 was mean ΔE −0.095, CI [−0.173, −0.011], 6/8. The LE replicate is the same sign, a CI entirely below 0, and a similar magnitude on a disjoint seed band.

Start 0 (the coherent center) was selected on 5/8 Nov3 wins at N15 and only 2/8 at N12. The N12 edge therefore is not only “start 0 was picked.” Fidelity moved with energy (N15 ΔF +0.042, 6/8) and did not select.

32/32 fits ok. B* rejects 0. Charged 65536. Outside-cap 0. Recipe hash matched FT PR79. Prior streams unchanged. No N18, no GR patch, no holdout touch. Stop after this matrix.
