# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 23:02 Asia/Shanghai_

BEAT = R≤0.98. Gates locked. Idea-gen R1=Opus; R2+=Grok 4.6.

## L=16 BOX — **ALL_CORE_BEAT = True** (honest scale1000 B=1000)

Train ONLY {6,9}. Eval CORE {6,9,12,15}.

### ratio69_w9 L=16 seed0 (box)

| n | QAOA Med | Med | R | QAOA B90 | B90 | verdict |
|---|----------|-----|---|----------|-----|---------|
| 6 | 1.226 | 1.157 | **0.943** | 2.300 | 1.686 | BEAT |
| 9 | 1.579 | 1.407 | **0.891** | 3.901 | 3.239 | BEAT |
| 12 | 2.071 | 1.825 | **0.881** | 7.027 | 5.821 | BEAT |
| 15 | 2.888 | 2.584 | **0.895** | 11.423 | 9.694 | BEAT |

**ALL_CORE_BEAT** = True

### ratio69_w9 L=16 seed1 (box)

| n | QAOA Med | Med | R | QAOA B90 | B90 | verdict |
|---|----------|-----|---|----------|-----|---------|
| 6 | 1.226 | 1.145 | **0.934** | 2.300 | 1.734 | BEAT |
| 9 | 1.579 | 1.410 | **0.893** | 3.901 | 3.305 | BEAT |
| 12 | 2.071 | 1.850 | **0.893** | 7.027 | 6.010 | BEAT |
| 15 | 2.888 | 2.596 | **0.899** | 11.423 | 10.036 | BEAT |

**ALL_CORE_BEAT** = True

### power_p L=16 seed0 (box) — best n15 so far

| n | QAOA Med | Med | R | QAOA B90 | B90 | verdict |
|---|----------|-----|---|----------|-----|---------|
| 6 | 1.226 | 1.155 | **0.942** | 2.300 | 1.685 | BEAT |
| 9 | 1.579 | 1.406 | **0.890** | 3.901 | 3.227 | BEAT |
| 12 | 2.071 | 1.817 | **0.878** | 7.027 | 5.796 | BEAT |
| 15 | 2.888 | 2.549 | **0.883** | 11.423 | 9.527 | BEAT |

**ALL_CORE_BEAT** = True

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

## Cloud L=16 — still RUNNING (no branch push yet; box already has ALL_CORE_BEAT)

- Idea1 ratio69_w9: https://cursor.com/agents/bc-7fa7da80-0efd-4d0f-b392-95c2d9fb5866 `cursor/train69-idea1-ratio69-w9-l16-66bb`

- Idea2 power_p: https://cursor.com/agents/bc-883dadcf-dbf5-4b78-9195-ab7d5d48272f `cursor/train69-idea2-power-p-l16-3c46`

Conversation stalled at setup; box results are authoritative for R1 L16.

