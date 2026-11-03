# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 22:46 Asia/Shanghai_

BEAT = R≤0.98. Gates locked. Idea-gen R1=Opus; R2+=Grok 4.6.

## Box L=2 (shallow) — no ALL_CORE_BEAT yet

### ratio69_w9 L=2 seed0
| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 5.326 | 5.100 | **0.958** | BEAT |
| 9 | 12.642 | 12.339 | **0.976** | BEAT |
| 12 | 29.606 | 28.985 | **0.979** | BEAT |
| 15 | 76.088 | 75.976 | 0.999 | TIE |

### trust69 L=2 seed0 ρ=0.1
| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 5.326 | 5.184 | **0.973** | BEAT |
| 9 | 12.642 | 12.597 | 0.997 | TIE |
| 12 | 29.606 | 29.343 | 0.991 | TIE |
| 15 | 76.088 | 75.562 | 0.993 | TIE |

## Cloud L=16 — RUNNING
- Idea1: https://cursor.com/agents/bc-7fa7da80-0efd-4d0f-b392-95c2d9fb5866 branch `cursor/train69-idea1-ratio69-w9-l16-66bb`
- Idea2: https://cursor.com/agents/bc-883dadcf-dbf5-4b78-9195-ab7d5d48272f branch `cursor/train69-idea2-power-p-l16-3c46`
