# NON-DEPTH wave — Opus region-blocked; ran top-2 cheap

_Generated 2026-10-31 04:00 Asia/Shanghai_

## Opus

`claude-opus-5-5-high` (and variants) → **region blocked** (box+Mac CN).
Ideas ranked by agent from Zach idea-space + prior Opus greenlights:
`runs/opus_ideas/opus_reply_20261031_0321_nondepth.md`

## Top-2 CHEAP ran

### 1. Fair ArmC pad L16→L44
See `../armC_L44_pad/RESULTS.md`

| n | ArmC L44 | ArmC L16 | champ L44 |
|---|----------|----------|-----------|
| 9 | 1.162/1.754 | 1.441/3.314 | 1.051/1.473 |
| 12 | 1.353/3.291 | 1.991/6.252 | 1.113/2.263 |
| 15 | 1.728/6.399 | 2.921/10.959 | 1.233/4.243 |
| 18* | 2.160/20.294 | 4.573/27.810 | 1.409/9.645 |

\*n18 ArmC L44 **B=100**; else B=1000. Pad **improves** ArmC vs L16 on all n; champ still wins.
b_Med ArmC L44≈**1.073** vs champ **1.033** vs ArmC L16 **1.137**.

### 2. Kerr refresh @ L44 + paper mega-table
See `../kerr_paper_L44/` and refreshed `../PAPER_TABLE.md`.

- mean |γ|=4.20 → 6.06 µs/layer raw; fold→[0,π] → **2.18 µs/layer**; sum_L fold≈**96 µs**
- L44 abstract pulses/inst: n9 3916 … n18 7876

## Still Uc=exp(-iγ Hc). No L48+. Blind banned.

Next candidates (#3 mixture B90, #4 freedom@L44) deferred.
