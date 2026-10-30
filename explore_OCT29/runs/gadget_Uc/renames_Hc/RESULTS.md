# renames_Hc — depth pad wave status

_Generated 2026-10-31 03:14:44 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. No blind-cost.

## Current champ: **L44 pad**

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.051 | 1.473 |
| 12 | 1.113 | 2.263 |
| 15 | 1.233 | 4.243 |
| 18 | 1.409 | 9.645 |

b_Med≈**1.0333**, b_B90≈**1.2323**

## L48 pad — NO PROMOTE → STOP ladder

| n | L44 | L48 |
|---|-----|-----|
| 9 | 1.051/1.473 | 1.047/1.419 |
| 12 | 1.113/2.263 | 1.106/2.189 |
| 15 | 1.233/4.243 | 1.212/3.872 |
| 18 | 1.409/9.645 | 1.379/9.706 |

Gate ≤1.208 / ≤1.381: n15 **fail**, n18 ok. L52 skipped.

## NON-DEPTH wave (2026-10-31 ~04:00)

Opus 5.5 **region-blocked** (CN). Ran agent-ranked top-2 cheap:
1. Fair ArmC L44 pad — champ still wins; ArmC pad improves vs L16 (see `armC_L44_pad/`).
2. Kerr@L44 + paper mega-table finalized (`kerr_paper_L44/`, `PAPER_TABLE.md`).

No L48+ reopen. Still `Uc=exp(-iγ Hc)`.

## NON-DEPTH #3+#4 (2026-10-31 ~04:28) — CVaR PROMOTE

- #3a equal-shot mixture: Spearman max 0.997 → kill; no promote (best L44+L40 n18 B90 9.501 > thr 9.356).
- #3b CVaR_q80 @L44: **PROMOTE** → new champ Med/B90 n9–18 = 1.045/1.420, 1.109/2.163, 1.228/4.067, 1.412/9.145; b_Med≈1.0341.
- #4 freedom span_w @L44: NO PROMOTE (Med gains <2% gate).

Still `Uc=exp(-iγ Hc)`. No L48+. Blind banned.

## CVaR polish + jitter (2026-10-31 ~04:45)

- PAPER_TABLE refreshed with CVaR champ vs ArmC L16/L44 / QAOA / WalkSAT.
- LBFGS polish seeds 0–3: **NO PROMOTE**. Best n18 B90 seed3=8.966 vs thr 8.962 (≥2% gate miss by 0.04%).
- Joint γ+φ jitter: σ≤2% ok (n9 Med ×≲1.04); σ=5% hurts (×1.27). Paper robustness row.

Champ remains **L44 CVaR_q80**: 1.045/1.420, 1.109/2.163, 1.228/4.067, 1.412/9.145; b_Med≈1.034.

