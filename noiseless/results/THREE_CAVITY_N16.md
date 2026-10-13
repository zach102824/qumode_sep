# 3-cavity / 3-transmon device (n=14/16/18 R2 confirms done; n=20 baseline failed, larger-resource screen running)

Device: cavities A, B, C, each with its own transmon d, e, f (1 data bit per transmon). Split: 3 transmon bits +
Gray-code cavity bits 5, 4, 4 (code levels 32, 16, 16; Fock truncation 48, 32, 32; state 393,216). Layer: rotation
on each transmon, ECD on each cavity with its own transmon (own displacement), then joint-parity A-B and B-C (chain;
`--ring` adds A-C). Algorithm identical to the 2-cavity best: SA layout pass (sab 128 lookups), explore-mask sa
fixed_low, xeta128, xL3 xs35, rawk48 polt3, XOR relabel over all three cavity codes (guess at Fock 0,0,0), R=2,
r=150, L=4, lr 0.05. Evals counted identically. Code: `noiseless/relayout_3cav.py`, runner `--device 3cav --ks 5,4,4`.
Tests: `noiseless/tests/test_relayout_3cav.py` (8 pass), including exact reduction to the 2-cavity code when C has
0 bits (state, eval, and a full explore+SA+relabel trial match), and the energy diagonal for all 2^16 strings.

Timing (1 core): one cost eval 0.078 s at L=3, 0.100 s at L=4 (2-cavity: 0.015 / 0.022 s, about 5x faster).
One full trial at E=4 R=2: about 126 s (2-cavity E=12 R=2: about 66 s).

## Step 0: 2-cavity n=16, SA budget (E12 R2 rawk48 polt3 xeta128 xL3 xs35), 50 trials

| sab | success | mean p(GS) | evals | lookups | per-run find rate |
|---|---|---|---|---|---|
| 128 | 0.90 | 0.893 | 3158 | 844 | not recorded (older run) |
| 256 | 0.80 | 0.794 | 3158 | 903 | 0.197 (6 of 50 trials miss in all runs) |
| 512 | 0.82 | 0.814 | 3158 | 1154 | 0.205 (9 of 50 trials miss in all runs) |

A larger SA budget does not help; sab 128 kept for 3 cavities.

## 3-cavity screen (10 trials, 1 per instance, R2)

| E | success | mean p(GS) | evals | lookups | leakage max | per-run find rate |
|---|---|---|---|---|---|---|
| 4 | 1.00 | 0.975 | 1454 | 423 | 4e-13 | 0.90 |
| 8 | 1.00 | 0.976 | 2306 | 584 | 3e-21 | 0.88 |
| 12 | 1.00 | 0.981 | 3158 | 738 | 8e-19 | 0.84 |

## n=16 confirm: 3 cavities, E=4, 50 trials (5 per instance)

| device | E | success | mean p(GS) | evals | lookups | leakage max | find rate | all-miss trials |
|---|---|---|---|---|---|---|---|---|
| 3 cavities (5,4,4) | 4 | 1.00 | 0.976 | 1454 | 435 | 4e-13 | 0.84 | 0 |
| 2 cavities (7,7) best | 12 | 0.91 | 0.903 | 3158 | about 850 | 1e-10 | about 0.2 | 6 to 9 of 50 |

Worst instance mean p(GS) 0.970. 3 cavities need 2.2x fewer evals, with higher quality, and land on the n=8..14
2-cavity trend (about 1420 to 1500 evals predicted at n=16). The win comes from the per-run find rate (0.84 vs about
0.2): with smaller cavities each explore run nearly always gets the answer into its top-48 states.

Find rate = fraction of explore runs whose top-48 states come within Hamming 1 of the ground state.

## n=18 (5,5,5; Fock 48,48,48) and n=20 (6,6,5; Fock 80,80,48)

Eval time (1 core): n=18 0.18 s (L=3), 0.23 s (L=4); n=20 0.68 s / 0.91 s. sab 162 / 200. Screen E=4,6,8
(10 trials), then a min-resource descent (lower E, then lower r/R/sab) and 50-trial confirm of the cheapest
setting with success and mean p(GS) both at least 0.9. Old screen queue `logs/3cav/q_3cav_n18n20.sh` was stopped; min-resource work finished in
`logs/3cav/q_3cav_resume_n18n16n20.sh` (n=20 baseline failed). n=18 R2 confirm finished 17:51 CST; R1 confirms wait on the n=20 larger-resource queue.

### n=18 screen (5,5,5), 10 trials, 1 per instance, R2, sab 162

| E | success | mean p(GS) | evals | lookups | leakage max | per-run find rate |
|---|---|---|---|---|---|---|
| 4 | 1.00 | 0.978 | 1454 | 521 | 6e-31 | 0.45 (18 of 40 explore runs) |
| 6 | 0.90 | 0.879 | 1880 | 614 | 0.099 | 0.42 (25 of 60 explore runs) |
| 8 | 0.86 (7/10, killed) | 0.832 | 2306 | 706 | 0.189 | 0.41 (partial) |

Worst instance mean p(GS) at E=4: 0.947. Evals at E=4 match n=16 (1454); only lookups grew (435 to 521).
E=6 clears success 0.9 but mean p(GS) 0.879 falls short of 0.9, so the cheapest passing screen stays E=4.
E=8 was stopped mid-run when the min-resource queue took over (7 of 10 trials done).

### n=18 min-resource descent (10-trial screens)

Goal: cheapest setting with success and mean p(GS) both at least 0.9. Start from E=4 r=150 R=2 sab=162.

| setting | success | mean p(GS) | evals | lookups | pass? |
|---|---|---|---|---|---|
| E=4 r=150 R=2 sab=162 (screen above) | 1.00 | 0.978 | 1454 | 521 | yes |
| E=3 r=150 R=2 sab=162 | 0.90 | 0.878 | 1241 | 476 | no (mean p(GS) < 0.9) |
| E=4 r=100 R=2 sab=162 | 1.00 | 0.941 | 1254 | 519 | yes (worst instance 0.864) |
| E=4 r=75 R=2 sab=162 | 1.00 | 0.895 | 1154 | 518 | no (mean p(GS) < 0.9, worst instance 0.730) |
| E=4 r=100 R=1 sab=162 | 1.00 | 0.930 | 1053 | 504 | yes (worst instance 0.854) |
| E=4 r=100 R=1 sab=64 | 1.00 | 0.930 | 1053 | 397 | yes (worst instance 0.854), cheapest |

E=3 fails the mean p(GS) bar, so explore stays at E=4. r=100 passes and r=75 does not, so refine stays at
r=100. One relabel round (R=1) passes at 1053 evals, and cutting the SA budget from 162 to 64 lookups gives the
same quality (same trial outcomes) with 397 lookups instead of 504. Cheapest n=18 setting: E=4 r=100 R=1 sab=64,
1.00 / 0.930 at 1053 evals and 397 lookups (10-trial screen). Its 50-trial confirm started 15:06 CST
(queue `logs/3cav/q_3cav_resume_n18n16n20.sh`, which then runs the n=16 descent and n=20).

| setting | trials | success | mean p(GS) | evals | lookups | worst instance | pass? |
|---|---|---|---|---|---|---|---|
| E=4 r=100 R=1 sab=64 (confirm) | 50 | 0.96 | 0.879 | 1053 | 399 | 0.695 | no (mean p(GS) < 0.9) |

The 50-trial confirm (finished 15:26 CST) misses the bar: 48 of 50 trials succeed (2 misses, p(GS) 0.001, on
instances 8 and 9), mean p(GS) 0.879 (0.915 over the 48 successful trials), median 0.920. With one relabel round there
is no second round to recover these. So the R=1 sab=64 setting is not confirmed. The cheapest n=18 setting that has
passed a screen is E=4 r=100 R=2 sab=162 (1.00 / 0.941 at 1254 evals, 519 lookups). Its 50-trial confirm finished
17:51 CST (queue `logs/3cav/q_3cav_n18confirm2.sh`):

| setting | trials | success | mean p(GS) | evals | lookups | SA lookups | worst instance | pass? |
|---|---|---|---|---|---|---|---|---|
| E=4 r=100 R=2 sab=162 (confirm) | 50 | 1.00 | 0.938 | 1254 | 510 | 162 | 0.883 | yes |

Confirmed cheapest n=18: E=4 r=100 R=2 sab=162, 1.00 / 0.938 at 1254 evals, 510 total lookups (162 SA).
Round-0 polished/raw hit 0.54 / 0.12; mean wall about 581 s per trial on 8 cores.

## n=14 on 3 cavities (4,4,3; Fock 32,32,24), min-resource descent

Same recipe, sab start 98, 10-trial screens, then a 50-trial confirm (5 per instance) of the cheapest pass.

| setting | trials | success | mean p(GS) | evals | lookups | worst instance | pass? |
|---|---|---|---|---|---|---|---|
| E=4 r=100 R=2 sab=98 | 10 | 1.00 | 0.951 | 1254 | 368 | 0.924 | yes |
| E=3 r=100 R=2 sab=98 | 10 | 1.00 | 0.936 | 1041 | 327 | 0.907 | yes |
| E=2 r=100 R=2 sab=98 | 10 | 1.00 | 0.936 | 828 | 289 | 0.893 | yes |
| E=2 r=75 R=2 sab=98 | 10 | 1.00 | 0.836 | 728 | 288 | 0.525 | no |
| E=2 r=100 R=1 sab=98 | 10 | 1.00 | 0.891 | 627 | 274 | 0.711 | no |
| E=2 r=100 R=2 sab=64 | 10 | 1.00 | 0.936 | 828 | 254 | 0.893 | yes, cheapest |
| E=2 r=100 R=2 sab=64 (confirm) | 50 | 1.00 | 0.932 | 828 | 257 | 0.915 | yes |

n=14 on 3 cavities needs 828 evals, fewer than the 2-cavity best (907), so the table now uses 3 cavities at n=14.

## n=16 on 3 cavities (5,4,4), min-resource descent

Start from the confirmed E=4 r=150 R=2 sab=128 (1.00 / 0.976 at 1454 evals, 435 lookups, 50 trials).

| setting | trials | success | mean p(GS) | evals | lookups | worst instance | pass? |
|---|---|---|---|---|---|---|---|
| E=4 r=100 R=2 sab=128 | 10 | 1.00 | 0.934 | 1254 | 423 | 0.871 | yes |
| E=4 r=75 R=2 sab=128 | 10 | 1.00 | 0.874 | 1154 | 424 | 0.726 | no (mean p(GS) < 0.9) |
| E=4 r=100 R=2 sab=128 (confirm) | 50 | 1.00 | 0.934 | 1254 | 435 | 0.885 | yes, cheapest |

r=75 fails, so the queue stopped the descent there; R=1, sab=64 and E=3 were not tried at n=16. Confirm finished
15:47 CST. Cheapest n=16: E=4 r=100 R=2 sab=128, 1.00 / 0.934 at 1254 evals, 435 lookups (50 trials).

Scaling fit on the cheapest 0.9+ evals so far: n=8 403, n=10 443, n=12 503 on 2 cavities; n=14 828, n=16 1254,
n=18 1254 (all 50-trial confirms) on 3 cavities. Log-linear fit: about 1.16 per added variable over n=8 to 16,
about 1.14 over n=8 to 18. n=20 (6,6,5) baseline finished 16:38 CST (failed); larger-resource screen started
17:51 CST (`q_3cav_n20big.sh`, first try E=6 r=150 R=2 sab=200 xs=35).

### n=20 baseline (6,6,5), 10-trial screen, E=4 r=150 R=2 sab=200

| setting | trials | success | mean p(GS) | evals | lookups | SA lookups | worst instance | pass? |
|---|---|---|---|---|---|---|---|---|
| E=4 r=150 R=2 sab=200 | 10 | 0.80 | 0.783 | 1454 | 590 | 200 | 0.0001 | no |

Baseline fails both bars (need success and mean p(GS) at least 0.9). Two of ten trials miss completely (instances 5 and 7, p(GS) about 0.0001 to 0.0003); the eight hits sit near 0.96 to 0.99, so mean p(GS) given success is 0.979. Round-0 polished/raw hit both 0. The queue stopped the descent and skipped the 50-trial confirm. n=20 needs more resources (higher E, or a larger cavity/Fock budget, or another algorithmic change) before it can join the scaling table.

### n=20 larger-resource step 1 (6,6,5), 10-trial screen, E=6 r=150 R=2 sab=200 xs=35 (finished 20:28 CST)

| setting | trials | success | mean p(GS) | evals | lookups | SA lookups | worst instance | pass? |
|---|---|---|---|---|---|---|---|---|
| E=6 r=150 R=2 sab=200 xs=35, after relabel round 1 | 10 | 0.90 | 0.874 | 1579 | n/a | 200 | 0.0000 | no |
| E=6 r=150 R=2 sab=200 xs=35, after relabel round 2 | 10 | 0.90 | 0.880 | 1880 | 686 | 200 | 0.0003 | no |

Success reaches 0.90 but mean p(GS) stays under 0.9, so this step fails. Instance 5 (missed in the baseline) is now found; instance 7 still misses: all six explore runs proposed wrong guesses (best guess energy 2, not the ground state), so relabel refines a wrong string. The nine hits sit at 0.97 to 0.99 (mean p(GS) given success 0.978). Wall time about 39 min per trial on 8 cores (2 h for the screen).

Next n=20 steps (sab=400 at E=6, then E=8, xs=50, r=200) are queued in `q_3cav_n20cont.sh`, gated behind the lookup-cutting screens (`q_3cav_minlookup.sh`, running since 20:28 CST) and the one-relabel-round 50-trial confirms for n=14/16/18 (`q_3cav_r1confirm.sh`).

## n=20 explore-only layout-noise test, instance 5 (2026-10-13 14:52-15:16 CST)

Strict n=20 recipe (E=8, xL3 xs35, K=40, polt1, SA 200 raw, ks 6,6,5), relabel round skipped (QM_EXPLORE_ONLY=1), instance 5 only (QM_INSTS=5), same seed as kbit_fixed. Each job: 1704 explore evals, 680 lookups. next_guess and fix-up hit are cumulative (best of the pool so far), so once GS is found later rows stay at GS. Queue logs/3cav/q_n20_noise_explore.sh, outdir relayout_runs/n20_noise_explore.

Summary: 0.15 -> E=1 (HD 4), miss; 0.3 -> GS found at run 2; 0.5 -> E=1 (HD 2), miss; 1.0 -> GS at run 5; 2.0 -> GS at run 6.

### inst 5 sanoise 0.15  GS 10111111100011010101  SA best is GS: False  evals 1704  lookups 680
| run | perm differs from run 1 | center differs | next_guess | E | HD to GS | p_gs | fix-up hit GS |
|---|---|---|---|---|---|---|---|
| 1 | no | no | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 2 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0008 | no |
| 3 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 4 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0002 | no |
| 5 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0002 | no |
| 6 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 7 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 8 | yes | yes | 10001001100011010101 | 1 | 4 | 0.0000 | no |
any run hit GS: False; final guess E 1, HD 4

### inst 5 sanoise 0.3  GS 10111111100011010101  SA best is GS: False  evals 1704  lookups 680
| run | perm differs from run 1 | center differs | next_guess | E | HD to GS | p_gs | fix-up hit GS |
|---|---|---|---|---|---|---|---|
| 1 | no | no | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 2 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0001 | yes |
| 3 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
| 4 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
| 5 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
| 6 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0004 | yes |
| 7 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
| 8 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
any run hit GS: True; final guess E 0, HD 0

### inst 5 sanoise 0.5  GS 10111111100011010101  SA best is GS: False  evals 1704  lookups 680
| run | perm differs from run 1 | center differs | next_guess | E | HD to GS | p_gs | fix-up hit GS |
|---|---|---|---|---|---|---|---|
| 1 | no | no | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 2 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0006 | no |
| 3 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0000 | no |
| 4 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0000 | no |
| 5 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0001 | no |
| 6 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0004 | no |
| 7 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0000 | no |
| 8 | yes | yes | 10111001100011010101 | 1 | 2 | 0.0000 | no |
any run hit GS: False; final guess E 1, HD 2

### inst 5 sanoise 1  GS 10111111100011010101  SA best is GS: False  evals 1704  lookups 680
| run | perm differs from run 1 | center differs | next_guess | E | HD to GS | p_gs | fix-up hit GS |
|---|---|---|---|---|---|---|---|
| 1 | no | no | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 2 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 3 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 4 | yes | yes | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 5 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0004 | yes |
| 6 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0014 | yes |
| 7 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
| 8 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0003 | yes |
any run hit GS: True; final guess E 0, HD 0

### inst 5 sanoise 2  GS 10111111100011010101  SA best is GS: False  evals 1704  lookups 680
| run | perm differs from run 1 | center differs | next_guess | E | HD to GS | p_gs | fix-up hit GS |
|---|---|---|---|---|---|---|---|
| 1 | no | no | 00111111110001110101 | 3 | 4 | 0.0000 | no |
| 2 | yes | yes | 10101111100011010101 | 1 | 1 | 0.0002 | no |
| 3 | yes | yes | 10001001100011010101 | 1 | 4 | 0.0000 | no |
| 4 | yes | yes | 01101101010000010001 | 1 | 9 | 0.0000 | no |
| 5 | yes | yes | 01101101010000010001 | 1 | 9 | 0.0000 | no |
| 6 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0012 | yes |
| 7 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
| 8 | yes | yes | 10111111100011010101 | 0 | 0 | 0.0000 | yes |
any run hit GS: True; final guess E 0, HD 0

## n=20 full screens, r150 + layout noise (2026-10-13 15:21–16:45 CST)

Strict recipe: E=8, xL3 xs35, K=40, polt1, SA 200 raw (`QM_SA_COUNT=raw`), R=1, r=150, ks 6,6,5.
Every trial: **2005 evals**, **740 lookups** (`n_lookups_fixed`; distinct strings vary ~554–611).
Queue `logs/3cav/q_n20_noise_r150.sh` → `relayout_runs/n20_noise_r150/`; instances 8–9 for sanoise 0.3 finished in `q_n20_r150_nz03_extra.sh` → `n20_noise_r150_extra/` (main nz0.3 killed after 8 lines to avoid duplicating 8–9). Noise 1.0 screen killed after 4 trials (partial).

| sanoise | trials | success | mean p(GS) | evals | lookups | pass? | notes |
|---|---|---|---|---|---|---|---|
| 0.3 | 10 (inst 0–9) | **1.00** | **0.957** | 2005 | 740 | **yes** | worst inst 4 at 0.926; inst 5 (prior miss) 0.963 |
| 1.0 | 4 (inst 0–3, killed) | 0.75 | 0.720 | 2005 | 740 | partial | miss inst 3; hits match nz0.3 p(GS) |

### sanoise 0.3 per instance (combined main + extra)

| inst | success | p(GS) | explore runs with next_guess / polished = GS |
|---|---|---|---|
| 0 | yes | 0.955 | 6, 7 |
| 1 | yes | 0.968 | 5, 6, 7 |
| 2 | yes | 0.957 | 0–7 (all) |
| 3 | yes | 0.967 | 7 |
| 4 | yes | 0.926 | 1–7 |
| 5 | yes | 0.963 | 1–7 |
| 6 | yes | 0.966 | 0–7 (all) |
| 7 | yes | 0.960 | 6, 7 |
| 8 | yes | 0.945 | 1–7 |
| 9 | yes | 0.963 | 1–7 |

No misses. SA best was never the ground state; the ground string entered via explore fix-up, then the r150 relabel round raised p(GS) to 0.93–0.97.

### sanoise 1.0 partial (inst 0–3 only)

| inst | success | p(GS) |
|---|---|---|
| 0 | yes | 0.955 |
| 1 | yes | 0.968 |
| 2 | yes | 0.957 |
| 3 | **no** | 0.0004 |

**Miss inst 3:** ground `00001011001000111101`. Final most_likely `10001110000111010010` (HD 12). Final / BSF next_guess `11100000101010001000` at energy 1, HD 12. Selected round 3 (explore). No explore run had `polished_is_ground` or `next_guess_is_ground` (all eight `run_guess_ground` false); runs 1–7 stuck on the same energy-1 guess.

**Takeaway:** sanoise 0.3 + r150 clears both bars at n=20 (1.00 / 0.957). Prefer 0.3 over 1.0 for the recipe; 1.0 still collapses on some instances.
