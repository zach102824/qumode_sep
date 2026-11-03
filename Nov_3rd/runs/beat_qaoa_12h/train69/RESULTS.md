# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 23:16 Asia/Shanghai_

BEAT = R≤0.98. Gates locked. Idea-gen R1=Opus; R2+=Grok 4.6.

## L=16 — **ALL_CORE_BEAT = True** (ratio69_w9 champion)

Train ONLY {6,9}. Eval CORE scale1000 B=1000. Box seed0 **bit-identical** to cloud `cursor/train69-idea1-ratio69-w9-l16-66bb` (SCB=54e9 packs sha-matched).

### ratio69_w9 L=16 seeds 0–5 (cloud+box)

| seed | n6 R | n9 R | n12 R | n15 R | ALL |
|------|------|------|-------|-------|-----|
| 0 | **0.9434** | **0.8912** | **0.8812** | **0.8947** | True |
| 1 | **0.9336** | **0.8934** | **0.8933** | **0.8987** | True |
| 2 | **0.9359** | **0.8940** | **0.8850** | **0.8884** | True |
| 3 | **0.9342** | **0.8952** | **0.8844** | **0.8894** | True |
| 4 | **0.9339** | **0.8905** | **0.8881** | **0.8990** | True |
| 5 | **0.9376** | **0.9008** | **0.8994** | **0.9135** | True |

**Champion:** seed2 (best n15 R=0.8884). Params: `ratio69_w9/params_L16_seed2.pt` → `champion/params_L16_train69_ratio69_w9.pt`.

### Champion seed2 raw Med/B90

| n | QAOA Med | Med | R | QAOA B90 | B90 |
|---|----------|-----|---|----------|-----|
| 6 | 1.2261 | 1.1476 | **0.9359** | 2.300 | 1.727 |
| 9 | 1.5787 | 1.4114 | **0.8940** | 3.901 | 3.284 |
| 12 | 2.0707 | 1.8326 | **0.8850** | 7.027 | 6.032 |
| 15 | 2.8881 | 2.5657 | **0.8884** | 11.423 | 10.059 |

### power_p L=16 (box; cloud Idea2 still RUNNING)

| seed | n6 R | n9 R | n12 R | n15 R | ALL |
|------|------|------|-------|-------|-----|
| 0 | 0.9424 | 0.8904 | 0.8777 | 0.8825 | True |
| 1 | 0.9336 | 0.8930 | 0.8896 | 0.8908 | True |

## L=2 — no ALL_CORE_BEAT yet (Grok R2 running)

Grok verdict: L=2 ALL_CORE **likely capacity** under train{6,9}; Round-2 = lucky-n15 hunt.

### Early R2

| idea | seed | n6 | n9 | n12 | n15 | ALL |
|------|------|----|----|-----|-----|-----|
| n9med_trust ρ0.05 | 0 | 0.982 | 1.003 | 0.996 | 0.988 | False |
| shrink_p p=0.15 | 0 | 0.954 | 0.974 | 0.981 | 1.003 | False |
| shrink_p p=0.15 | 1 | 0.953 | 0.976 | 0.982 | 1.003 | False |
| shrink_p p=0.30 | 0 | 0.958 | 0.979 | 0.982 | 1.001 | False |
| shrink_p p=0.30 | 1 | 0.951 | 0.976 | 0.980 | 1.007 | False |
| ratio69_w9 R1 | 0 | 0.958 | 0.976 | 0.979 | 0.999 | False |
| trust69 ρ0.1 R1 | 0 | 0.973 | 0.996 | 0.991 | 0.993 | False |

Best L=2 n15 so far under train{6,9}: **n9med s0 R=0.988** (still TIE). Continue seeds 1–7 + shrink_p + alpha_ray.

## Cloud

- Idea1 ratio69_w9 L16: **FINISHED** https://cursor.com/agents/bc-7fa7da80-0efd-4d0f-b392-95c2d9fb5866 branch `cursor/train69-idea1-ratio69-w9-l16-66bb`

- Idea2 power_p L16: still RUNNING https://cursor.com/agents/bc-883dadcf-dbf5-4b78-9195-ab7d5d48272f

