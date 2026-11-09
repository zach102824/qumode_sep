# STARTS S1Q — remaining Nov3−QAOA L3 energy gap is not mainly budget

S1 is frozen (PR83/84/85 TQA-only C0). S4/S8/continuation/conditioning/throughput stay closed. This batch asked whether extra inclusive budget at **N=15, L=3** closes the fair Nov3 vs QAOA energy gap. Seeds **10072–10079**, Q ∈ {3072, 6144, 12288}, K = 1531 / 3067 / 6139 via split_cap. Energy selects. F diagnostic only.

**No Q is a development energy-win candidate. GR not patched.** The preregistered soft note requires Nov3−QAOA mean Δ(E−E0)/N bootstrap 95% CI entirely below 0. None of the three budgets meet that.

| Q | mean Δ(E−E0)/N | median | CI | Nov3 lower | energy-win candidate? |
|---|---|---|---|---|---|
| 3072 | +0.00206 | +0.00364 | [−0.00165, +0.00491] | 1/8 | no |
| 6144 | +0.00158 | +0.00312 | [−0.00195, +0.00415] | 2/8 | no |
| 12288 | +0.00116 | +0.00268 | [−0.00209, +0.00343] | 1/8 | no |

Means stay positive (QAOA lower energy). Every CI includes 0, so this band also does not support a statistical QAOA claim; the ranking is 6–7/8 QAOA and a shrinking-but-still-positive mean. Quadrupling Q from 3072 to 12288 cuts the mean gap by about 0.0009/N — a small move relative to the remaining ~0.0012/N and to the CI width.

**Secondary: both methods improve with budget, 8/8.** Nov3 improves a bit more (Δ(E−E0)/N vs own Q=3072: −0.00077 at 6144, −0.00148 at 12288) than QAOA (−0.00029, −0.00058). That is why the gap shrinks slightly. Cell means: Nov3 0.0232 → 0.0224 → 0.0217; QAOA 0.0211 → 0.0208 → 0.0205.

F post hoc favors Nov3 at every Q (mean ΔF +0.033 to +0.041, CI entirely above 0). Energy remains the selection rule. Higher F does not reverse the energy ranking.

Resource gate: Q=3072 wall 56 s, Q=6144 114 s, estimated Q=12288 228 s < 1800 s, so 12288 ran (actual 227 s). No blocker.

48/48 fits ok. B* rejects 0. Charged 344064. Outside-cap 0. Frozen hashes matched. Prior streams unchanged. No S1 retune, no S4/S8, no continuation, no N18, no GR patch. Stop after this matrix.
