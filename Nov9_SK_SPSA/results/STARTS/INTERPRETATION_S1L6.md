# STARTS S1L6 — S1 helps at N15 L=6; still not a GR patch

S1 vs S4 was already a candidate at L=3 (PR83) and confirmed on fresh-dev (PR84). This batch froze those same recipes (starts.py, C0 SPSA, TQA start-0; K via split_cap, not 2× the Q=3072 K) and fitted them at **N=15, L=6, Q=6144** on eight exclusive unread seeds 10064–10071. Hypothesis: TQA-only concentration helps where depth fade is worst. No S1 retune. S4 fitted and charged. Energy selects. F diagnostic only.

**Gate passes. Do not patch GR.** Nov3 mean(E_S1 − E_S4) bootstrap 95% CI is entirely below 0:

| cell | mean ΔE | median | CI | S1 energy wins |
|---|---|---|---|---|
| N15 L6 Nov3 | −0.087 | −0.120 | [−0.144, −0.014] | 7/8 |

Δ(E−E0)/N = −0.0058, CI [−0.0096, −0.0009]. QAOA S1 also beat S4 (mean ΔE −0.019, CI [−0.034, −0.006], 7/8). One Nov3 seed (10070) is a S1 loss; the mean CI still clears.

**Fair Nov3 vs QAOA is a different claim, and energy does not support Nov3 superiority.**

| arm | Δ(E−E0)/N Nov3−QAOA mean [CI] | Nov3 lower | ΔF mean [CI] | energy-superior? |
|---|---|---|---|---|
| S1 | +0.0074 [+0.0021, +0.0130] | 1/8 | −0.012 [−0.064, +0.041] | no |
| S4 | +0.0119 [+0.0078, +0.0156] | 0/8 | −0.062 [−0.108, −0.014] | no |

Under S1, Nov3 mean Eerror/N is 0.020 vs QAOA 0.013. The energy CI of Nov3−QAOA lies entirely **above** 0 on both arms. S1 shrinks the Nov3 vs QAOA energy gap relative to S4 but does not reverse it. Higher F under S1 vs S4 does not override energy selection.

32/32 fits ok. B* rejects 0 (all odd E1, resid < 1e-13). Charged 196608. Outside-cap 0. Frozen hashes matched. N3COND 772224, LEXACT 2560, DEPTH C1 327680, STARTS S8 98304, STARTS S1 98304, STARTS S1C 196608 unchanged. No S1 retune, no S8, no continuation, no N18, no GR/frozen-v1 patch. Stop after this matrix.
