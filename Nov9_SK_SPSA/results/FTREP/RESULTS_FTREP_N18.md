# FTREP N18: FT v4 center-4 at N=18 L=2 Q=2048

FT alloc.py sha256 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` from PR79 `347ac23369db`.
Exclusive unread seeds 10168–10175. N=18 L=2 Q=2048. Four equal SPSA blocks, frozen gains, center-4 starts.
Energy selects (init/last/Polyak/mid). F post hoc. GR not patched. FT holdout 10144–10151 unused. No L3.

Runs 16 ok 16 failures 0. Charged 32768. Outside cap 0. Rejected B* 0. Missing 0.

Gate N18 mean ΔE −0.10889 med −0.12720 [−0.17912,−0.03566] tails [−0.27588,+0.04643] 6/8: **N18-L2 SCALED**.

Δ(E−E0)/N −0.00605 med −0.00707 [−0.00989,−0.00221] tails [−0.01533,+0.00258].
ΔF +0.05489 med +0.05344 [+0.02920,+0.08165] tails [−0.01067,+0.11906] 7/8.
Nov3 start0 5/8. QAOA start0 0/8.

## Nov3 − QAOA (center4)

| N | mean ΔE med [CI] tails wins(Nov3) | Δ(E−E0)/N | ΔF wins | start0 Nov3 |
|---|---|---|---|---|
| 18 | −0.10889 med −0.12720 [−0.17912,−0.03566] tails [−0.27588,+0.04643] 6/8 | −0.00605 med −0.00707 [−0.00989,−0.00221] tails [−0.01533,+0.00258] | +0.05489 med +0.05344 [+0.02920,+0.08165] tails [−0.01067,+0.11906] 7/8 | 5/8 |

## Per seed (energy-selected)

| seed | B* | E_Nov3 | E_QAOA | ΔE | F_Nov3 | F_QAOA | start Nov3 | start QAOA |
|---|---|---|---|---|---|---|---|---|
| 10168 | 0.88779 | −18.56530 | −18.28942 | −0.27588 | 0.6527 | 0.5336 | 3 | 2 |
| 10169 | 0.72575 | −15.34509 | −15.28373 | −0.06136 | 0.6969 | 0.6550 | 0 | 3 |
| 10170 | 0.90785 | −18.80733 | −18.68809 | −0.11924 | 0.7013 | 0.6574 | 0 | 3 |
| 10171 | 1.24415 | −24.40808 | −24.27197 | −0.13610 | 0.6989 | 0.6359 | 0 | 1 |
| 10172 | 0.75143 | −16.29546 | −16.31604 | +0.02057 | 0.6335 | 0.6180 | 1 | 2 |
| 10173 | 0.49413 | −11.87954 | −11.74437 | −0.13517 | 0.5872 | 0.5065 | 0 | 3 |
| 10174 | 0.60530 | −13.94046 | −13.98688 | +0.04643 | 0.6262 | 0.6368 | 0 | 2 |
| 10175 | 0.84735 | −17.93949 | −17.72911 | −0.21038 | 0.7204 | 0.6346 | 1 | 3 |

## Wall / CPU

| quantity | value |
|---|---|
| B* wall (4 workers) | 98.8 s |
| fit wall (4 workers) | 258.1 s |
| sum of cell wall | 919.3 s |
| sum of cell CPU | 918.5 s |
| peak RSS | 265652 KiB |
