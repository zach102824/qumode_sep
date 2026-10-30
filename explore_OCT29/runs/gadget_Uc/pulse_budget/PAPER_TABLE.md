# Paper table — scale1000 B=1000

_Generated 2026-10-31 00:05:20 +0800 (Asia/Shanghai)_

**Metric:** quantum columns = median / B90 of **1/p**; WalkSAT = median / B90 **flips** (different unit).

Honesty: champ = soft-n15 L32 projector-Uc ≡ exp(-iγ Hc); offline 26 SNAP masks Fid=1; two-cav needs joint SNAP.

| n | champ Med/B90 | ArmC L16 | QAOA L16 | WalkSAT flips |
|---|---------------|----------|----------|---------------|
| 9 | **1.100/1.759** | 1.441/3.314 | 2.242/5.534 | 13.0/37.1 |
| 12 | **1.231/2.919** | 1.991/6.252 | 3.263/10.868 | 26.5/69.6 |
| 15 | **1.483/5.847** | 2.921/10.959 | 5.099/18.647 | 44.8/123.6 |
| 18 | **1.884/13.704** | 4.573/27.810 | 7.894/34.070 | 75.5/210.7 |

## b fits (9→18)

- champ b_Med≈**1.0619**, b_B90≈**1.2567**
- ArmC b_Med≈**1.1368**
- QAOA b_Med≈**1.1511**
- WalkSAT flips b_Med≈**1.2133**

## Ablation (L16 n15 Med) — no blind revive

| arm | n15 Med |
|-----|---------|
| projector | 2.754 |
| armC | 2.921 |
| trunc_Hc | 192.176 |
| mixer_only | 16384.0 |
