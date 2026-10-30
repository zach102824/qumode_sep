# CVaR@L44 — LBFGS polish + joint jitter

_Generated 2026-10-31 04:45:48 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. No L48+. Blind banned.

Champ baseline: n9 1.045/1.420, n12 1.109/2.163, n15 1.228/4.067, n18 1.412/9.145
Promote gate: n18 B90 ≤ **8.962** (≥2%) AND Med ≤ 1.05×.

## PAPER_TABLE

Refreshed with current champ vs ArmC L16 / ArmC L44 / QAOA / WalkSAT (b_Med≈1.0341).

## LBFGS polish

| seed | val S | n9 | n12 | n15 | n18 | keep |
|------|-------|----|-----|-----|-----|------|
| seed0 | 0.383 | 1.045/1.414 | 1.108/2.148 | 1.229/4.073 | 1.409/9.084 | False |
| seed1 | 0.384 | — | — | — | — | None |
| seed2 | 0.390 | — | — | — | — | None |
| seed3 | 0.383 | 1.045/1.412 | 1.109/2.159 | 1.231/4.084 | 1.413/8.966 | False |

Winner: **seed0** promote=**False**
gates=None

## Joint γ+φ jitter (confirm)

| σ | n9 | n12 | n15 |
|---|----|-----|-----|
| 0.0 | 1.046/1.490 | 1.113/2.055 | 1.228/4.402 |
| 0.01 | 1.051/1.508 | 1.123/2.087 | 1.240/4.514 |
| 0.02 | 1.084/1.535 | 1.164/2.157 | 1.296/4.716 |
| 0.05 | 1.328/1.916 | 1.508/2.860 | 1.777/6.806 |

## Final champ

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.045 | 1.420 |
| 12 | 1.109 | 2.163 |
| 15 | 1.228 | 4.067 |
| 18 | 1.412 | 9.145 |

b_Med≈**1.0341**, b_B90≈**1.2304**
