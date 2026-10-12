# 3-cavity / 3-transmon device at n=16 (n=16 complete; n=18/20 running)

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
(10 trials), then 50 trials at the cheapest E clearing 0.9 on both. Queue `logs/3cav/q_3cav_n18n20.sh` (running).
