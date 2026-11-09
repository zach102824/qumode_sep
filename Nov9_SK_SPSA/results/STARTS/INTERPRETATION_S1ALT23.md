# STARTS S1ALT23 — S1+ALT holds at L=2, not at L=3; GR unpatched

PR88 showed block-ALT beats joint IID S1 at L=2. This batch stacked those two frozen levers (S1 TQA-only + exact PR88 ALT) at the onset depth **L=2** and one step deeper **L=3**, Q=1024·L, seeds **10096–10103**. No ALT/S1 retune. No extra budget. Energy selects. F diagnostic only.

**Gates: L=2 passes, L=3 does not. Do not patch GR.**

| L | Q | mean ΔE (S1ALT−S1JOINT) | CI | S1ALT energy wins | pass |
|---|---|---|---|---|---|
| 2 | 2048 | −0.036 | [−0.081, −0.011] | 8/8 | yes |
| 3 | 3072 | −0.006 | [−0.022, +0.010] | 4/8 | no |

L=2 replicates PR88 on a new seed band. At L=3 the mean still favors ALT but the CI includes 0 and only 4/8 seeds win. The stacked lever does not clear the preregistered “each L” gate.

QAOA S1ALT beat S1JOINT at both L (8/8). That is not the gate.

**Fair Nov3 vs QAOA under S1ALT is not an energy-win candidate at either L.**

| L | arm | Δ(E−E0)/N Nov3−QAOA mean [CI] | Nov3 lower | energy-win candidate? |
|---|---|---|---|---|
| 2 | S1JOINT | +0.00295 [+0.0009, +0.0049] | 2/8 | no |
| 2 | S1ALT | +0.00084 [−0.0014, +0.0028] | 3/8 | no |
| 3 | S1JOINT | +0.00263 [−0.0033, +0.0064] | 1/8 | no |
| 3 | S1ALT | +0.00265 [−0.0037, +0.0063] | 1/8 | no |

At L=2, ALT shrinks the Nov3–QAOA energy gap versus JOINT but does not reverse it. At L=3 the fair gap is essentially unchanged. No CI under S1ALT lies entirely below 0.

64/64 fits ok. B* rejects 0. Charged 163840. Outside-cap 0. Frozen hashes matched including alt_spsa.py. Prior streams unchanged. No S4/S8, no continuation, no extra budget, no N18, no GR patch. Stop after this matrix.
