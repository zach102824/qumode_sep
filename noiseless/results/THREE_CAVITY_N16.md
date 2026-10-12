# 3-cavity / 3-transmon device at n=16 (PARTIAL — runs in progress)

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
| 512 | running | | | | |

## 3-cavity screen (10 trials, 1 per instance, R2)

| E | success | mean p(GS) | evals | lookups | leakage max | per-run find rate |
|---|---|---|---|---|---|---|
| 4 | 1.00 | 0.975 | 1454 | 423 | 4e-13 | 0.90 |
| 8 | running | | | | | |
| 12 | queued | | | | | |

Find rate = fraction of explore runs whose top-48 states come within Hamming 1 of the ground state.
Next: 50 trials at E=4 (queued, starts when the screen finishes), compared with 2-cavity 0.91 / 0.903 at 3158 evals.
