# Minimum classical lookups, one relabel round (R1), Gray code, fixed_low

10-trial screens (1 per instance), bar: success >= 0.9 AND mean p(GS) >= 0.9. Descend one knob at a time, keeping the cheapest passing value: sab (halve; floor 64, since sa_sample always spends 64 random probes, so sab below 64 changes nothing), then rawk 24, 12, then polt 1. Lookups = mean distinct strings per trial (total) and SA part. Screens are 10 trials, so a pass needs a 50-trial confirm before it is final.

### n=18

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n18_base_sab162_k48_p3 | 162 | 48 | 3 | 10 | 1.00 | 0.930 | 1053 | 503.9 | 162.0 | PASS |
| n18_sab81_k48_p3 | 81 | 48 | 3 | 10 | 0.90 | 0.836 | 1053 | 408.8 | 81.0 | fail |
| n18_sab162_k24_p3 | 162 | 24 | 3 | 10 | 1.00 | 0.930 | 1053 | 402.7 | 162.0 | PASS |
| n18_sab162_k12_p3 | 162 | 12 | 3 | 10 | 1.00 | 0.930 | 1053 | 343.6 | 162.0 | PASS |
| n18_sab162_k12_p1 | 162 | 12 | 1 | 10 | 1.00 | 0.930 | 1053 | 247.9 | 162.0 | PASS |

**n=18 cheapest passing: sab 162, rawk 12, polt 1**

### n=16

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
