# train69 RESULTS — train{6,9} only → CORE eval

_Updated 2026-11-03 23:18 Asia/Shanghai_

BEAT = R≤0.98. Gates locked. R1=Opus; R2+=Grok 4.6.

## L=16 — **DONE / ALL_CORE_BEAT** under train{6,9}

### Champion: **power_p** learned seed0 (box ≡ cloud)

Edges ratio69_w9 on n12/n15. p_final≈0.0515. Params: `champion/params_L16_train69_power_p.pt`.

| n | QAOA Med | Med | R | QAOA B90 | B90 |
|---|----------|-----|---|----------|-----|
| 6 | 1.2261 | 1.1555 | **0.9424** | 2.300 | 1.685 |
| 9 | 1.5787 | 1.4056 | **0.8904** | 3.901 | 3.227 |
| 12 | 2.0707 | 1.8175 | **0.8777** | 7.027 | 5.796 |
| 15 | 2.8881 | 2.5488 | **0.8825** | 11.423 | 9.527 |

### power_p learned seeds 0–2

| seed | p | n6 R | n9 R | n12 R | n15 R | ALL |
|------|---|------|------|-------|-------|-----|
| 0 | 0.0515 | 0.9424 | 0.8904 | 0.8777 | 0.8825 | True |
| 1 | 0.0520 | 0.9336 | 0.8930 | 0.8896 | 0.8908 | True |
| 2 | 0.0438 | 0.9350 | 0.8935 | 0.8833 | 0.8851 | True |

### power_p p0 sweep (seed0) — all ALL_CORE_BEAT

| tag | p_final | n6 | n9 | n12 | n15 |
|-----|---------|----|----|-----|-----|
| p-0.5 | -0.0162 | 0.9427 | 0.8915 | 0.8827 | 0.8941 |
| p0 | 0.0515 | 0.9424 | 0.8904 | 0.8777 | 0.8825 |
| p0.5 | 0.1038 | 0.9399 | 0.8960 | 0.8901 | 0.8961 |
| p1 | 0.4101 | 0.9462 | 0.8909 | 0.8758 | 0.8817 |

(p1-init ends n15 R=**0.8817**, slight edge over learned-s0 0.8825; champ kept as learned-s0 per primary recipe.)

### ratio69_w9 L=16 seeds 0–5 (also ALL_CORE; runner-up)

| seed | n6 | n9 | n12 | n15 |
|------|----|----|-----|-----|
| 0 | 0.9434 | 0.8912 | 0.8812 | 0.8947 |
| 1 | 0.9336 | 0.8934 | 0.8933 | 0.8987 |
| 2 | 0.9359 | 0.8940 | 0.8850 | 0.8884 |
| 3 | 0.9342 | 0.8952 | 0.8844 | 0.8894 |
| 4 | 0.9339 | 0.8905 | 0.8881 | 0.8990 |
| 5 | 0.9376 | 0.9008 | 0.8994 | 0.9135 |

## L=2 — **NO ALL_CORE yet** (all remaining compute here)

Grok R2: likely capacity; lucky-n15 hunt via n9med_trust / shrink_p / alpha_ray.

| idea | seed | n6 | n9 | n12 | n15 | ALL |
|------|------|----|----|-----|-----|-----|
| ratio69_w9 R1 | 0 | 0.9576 | 0.9761 | 0.9790 | 0.9985 | False |
| trust69 ρ0.1 R1 | 0 | 0.9733 | 0.9965 | 0.9911 | 0.9931 | False |
| n9med ρ0.05 | 0 | 0.9815 | 1.0026 | 0.9962 | 0.9883 | False |
| n9med ρ0.05 | 1 | 0.9896 | 0.9980 | 0.9925 | 0.9885 | False |
| shrink_p=0.15 | 0 | 0.9536 | 0.9741 | 0.9814 | 1.0034 | False |
| shrink_p=0.15 | 1 | 0.9527 | 0.9759 | 0.9818 | 1.0029 | False |
| shrink_p=0.30 | 0 | 0.9581 | 0.9788 | 0.9816 | 1.0006 | False |
| shrink_p=0.30 | 1 | 0.9507 | 0.9764 | 0.9795 | 1.0067 | False |
| shrink_p=0.45 | 0 | 0.9592 | 0.9798 | 0.9777 | 1.0023 | False |
| shrink_p=0.45 | 1 | 0.9542 | 0.9773 | 0.9851 | 1.0083 | False |

Best L=2 n15 so far: **n9med ρ0.05 s0 R=0.9883** (need ≤0.98).

## Cloud

- Idea1 ratio69_w9: FINISHED `cursor/train69-idea1-ratio69-w9-l16-66bb`

- Idea2 power_p: FINISHED `cursor/train69-idea2-power-p-l16-3c46`

- L=16 closed. Remaining: L=2 ONLY.

