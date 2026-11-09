# STARTS S1SH — Nov3 energy-wins at L=1, already loses at L=2; GR unpatched

S1Q showed the L=3 Nov3−QAOA energy gap is not mainly budget. This batch asked a distinct question: under frozen TQA-only S1, is that gap already present at **cold L=1 and L=2**? N=15, Q=qL, seeds **10080–10087**. Each L fitted independently from TQA. Not continuation. Not extra budget. Not S4/S8. Not local R10 / FT N18.

**Soft note: L=1 is a development energy-win candidate. L=2 is not. Do not patch GR. S1 remains a candidate.**

| L | Q | mean Δ(E−E0)/N | CI | Nov3 lower | energy-win candidate? |
|---|---|---|---|---|---|
| 1 | 1024 | −0.0207 | [−0.0246, −0.0175] | 8/8 | yes |
| 2 | 2048 | +0.0036 | [+0.0021, +0.0054] | 0/8 | no |

At L=1 Nov3 mean Eerror/N is 0.038 vs QAOA 0.059. ΔF +0.124, CI entirely above 0, 8/8. At L=2 the energy ranking has already reversed: Nov3 0.037 vs QAOA 0.034, ΔF ~0 (CI includes 0). Together with S1Q L=3 (~+0.002) and S1L6 L=6 (~+0.007), the fair energy loss is a **depth-onset after L=1**, not a missing-budget effect and not an L=1 architecture loss.

This does not reopen continuation (PR81). These are cold independent-L S1 fits. It does not adopt Nov3 into GR. L=1 energy-win is a development note only.

32/32 fits ok. B* rejects 0. Charged 49152. Outside-cap 0. Frozen hashes matched. Prior streams unchanged. Stop after this matrix.
