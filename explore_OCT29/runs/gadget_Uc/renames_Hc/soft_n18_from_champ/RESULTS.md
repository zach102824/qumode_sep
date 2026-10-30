# Soft-n18 FROM locked LBFGS seed0 (renames_Hc fix)

_Generated 2026-10-31 02:32:50 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. **No blind-cost.**

## CRITICAL: prior soft-n18 INVALID

Earlier `renames_Hc` soft-n18 table (n9–18 ≈ 2.24/5.20 … **9.80/38.64**) was trained from a
**bad short retrain seed**, not locked LBFGS seed0. **Do not cite for promote.**

## Locked champ (verified scale1000)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

Gate: n18 Med≤98% locked **or** B90≤98% locked; AND n9/12/15 Med ≤105% locked.

## Arms (scale1000 B=1000)

| arm | n9 | n12 | n15 | n18 | keep |
|-----|----|-----|-----|-----|------|
| t9121518 | 1.107/1.751 | 1.220/2.753 | 1.440/5.483 | 1.772/11.249 | False |
  - n18 Med 1.772 (thr 1.717) B90 11.249 (thr 11.126) → no
  - n9 Med 1.107 ≤ 1.147 → ok
  - n12 Med 1.220 ≤ 1.257 → ok
  - n15 Med 1.440 ≤ 1.489 → ok
| t9121518_w1 | 1.098/1.719 | 1.197/2.779 | 1.395/5.460 | 1.709/11.107 | True |
  - n18 Med 1.709 (thr 1.717) B90 11.107 (thr 11.126) → ok
  - n9 Med 1.098 ≤ 1.147 → ok
  - n12 Med 1.197 ≤ 1.257 → ok
  - n15 Med 1.395 ≤ 1.489 → ok

**Winner:** `t9121518_w1`

Fair compare: zero-shot honesty = locked was trained without n18; soft-n18 uses n18 val
(B=16 typical). scale1000 remains holdout.

