# Improvement search on the full ladder n = 8, 10, 12, 14, 16 (wmaxsat, Gray code)

Goal: one recipe family with about the same performance at every n (success and mean p(GS) about ≥ 0.9, preferably about 0.95+), with evals growing competitively (target about 1.1–1.2^n or better). Classical SA lookups of about n^2 are fine.

This file supersedes `IMPROVE_N12_N14.md` for the ladder view; that file keeps the early n=12/14 idea history.

## Common recipe family (current winners)

- Hardware / encoding: k-bit ECD chip, Gray cavity code, wmaxsat Hamiltonians (10 instances per n).
- Explore then relabel: L=4, s=50, r=150, usually R=1 relabel round, lr=0.05, xeta=16 (xeta=32 sometimes helps), reta=2, 8 workers.
- Classical SA layout: `--explore-mask sa --samode fixed_low`, sab ≈ n^2/2, sanoise 0.15 default.
- Energy-ranked readout: `--rawk 24 --polt 2` (top-24 probable states, Hamming-1 fix-up of the 2 lowest-energy ones).
- Cheap explore knobs that help: `--xL 2` or `3`, `--xs 35` (shallower / fewer SPSA steps per explore growth).
- Dropped ideas: thr (threshold relabel cost), xstop (early-stop explore), xwarm (warm-start explore params), lmut (cold layout mutation). All hurt or did nothing on 50-trial confirms.

Evals per trial ≈ (explore cost) × E + (relabel cost) × R. Lookups = distinct classical energy lookups including SA + readout pool + fix-up.

## Best confirmed so far (50 trials unless noted)

| n | E | setting | success | mean p(GS) | evals | lookups (approx) |
|---|---|---|---|---|---|---|
| 8 | — | not yet run on this stack | — | — | — | — |
| 10 | — | not yet run on this stack | — | — | — | — |
| 12 | 2 | sab72 rawk24 xL2 R1 | 0.92 | 0.910 | 705 | about 162 |
| 12 | 2 | sab72 rawk24 xL3 xs35 R1 | 0.92 | 0.912 | 727 | about 160 |
| 12 | 3 | sab72 rawk24 xL2 R1 | 0.98 | 0.965 | 907 | about 170 |
| 12 | 2 | sab72 rawk24 R1 (no xL/xs) | 0.92 | 0.910 | 1109 | about 146 |
| 14 | 7 | sab98 rawk24 xL3 xs35 R1 | 0.90 | 0.887 | 1792 | about 300 |
| 14 | 6 | sab98 rawk24 xs35 R1 | 0.94 / 0.90 (rep) | 0.930 / 0.888 | 2005 | about 302 |
| 14 | 6 | sab98 rawk24 xL3 R1 xeta32 | 0.96 | 0.947 | 2119 | about 289 |
| 14 | 6 | sab98 rawk24 xL3 R1 | 0.94 | 0.927 | 2119 | about 290 |
| 14 | 7 | sab98 rawk24 xs35 R1 | 1.00 / 0.92 (rep) | 0.989 / 0.910 | 2289 | about 324 |
| 14 | 7 | sab98 rawk24 xs35 (R2) | 1.00 | 0.993 | 2590 | about 320 |
| 16 | — | queued (E=8..20 screen) | — | — | — | — |

Older baselines for context: n=12 E5 random layout 0.92 / 0.914 at 2622 evals; n=14 E17 random layout 0.94 / 0.933 at 7470 evals. Current n=14 ~0.9 points are about 70% fewer evals than that baseline.

## Scaling table (matched ~0.9 quality)

Using the cheapest confirmed points with success ≥ 0.9 and mean p(GS) ≥ 0.9 where available:

| n | cheapest ~0.9 point | evals | ratio to previous | b per unit n |
|---|---|---|---|---|
| 8 | pending | — | — | — |
| 10 | pending | — | — | — |
| 12 | E2 xL2 R1 rawk24 | 705 | — | — |
| 14 | E6 xs35 R1 rawk24 (0.94/0.930; rep 0.90/0.888) | 2005 | 2.84× over 2 steps | about 1.69 |
| 14 alt | E6 xL3 R1 xeta32 (0.96/0.947) | 2119 | 3.01× over 2 steps | about 1.73 |
| 16 | pending | — | — | — |

If we accept the borderline n14 E7 xL3 xs35 at 1792 (success 0.90, mean p(GS) 0.887 just under 0.9), the step from n12@705 is about 1.59 per unit n. Still above the 1.1–1.2 target. Filling n=8, 10, 16 is the next step so the fit is trustworthy.

Fitted form evals ≈ A · b^n over n=12..14 alone is noisy; the full ladder will decide.

## Recent idea outcomes (queues 10–19)

Wins
- SA-chosen layout (sab ≈ n^2/2): modest gain, especially at n=14.
- rawk 16→24 + polt 2: large gain; lets E drop a lot.
- R=1: enough once the guess is usually right after explore.
- xs35 and xL2/xL3: cut explore cost without much quality loss.
- xeta32 at n14 E6 xL3: 0.96 / 0.947 at 2119 (better than xeta16 at same evals).

Losses / neutral
- xwarm, lmut: clearly bad (layout diversity / growth stages matter).
- thr, xstop, polt3, rawk8: no gain or worse.
- xagree early-stop: mixed / often cuts quality when it saves evals.
- xeta8, xbeta5: hurt; xbeta1.5 neutral.

## Status (Oct 10, 2026, about 23:20 Shanghai)

Best per n
- n=8, 10, 16: not yet measured on this stack (queued).
- n=12: 0.92 / 0.910 at 705 evals (E2 xL2 R1 rawk24 sab72).
- n=14: cheapest solid ~0.9 is 0.94 / 0.930 at 2005 evals (E6 xs35 R1 rawk24); higher quality 0.96 / 0.947 at 2119 (xeta32).

Running (detached, chained; do not kill)
- logs/improve/queue19.sh (finishing n14/n12 explore-quality knobs: xkick, reta, fewer longer explores).
- logs/improve/queue20.sh (done): n12/n14 xeta/xL/xs knob sweeps (no n=8/10 points; those moved to queue22/23).
- logs/improve/queue21.sh (waits for queue20): push n14 under ~2000 with xeta32 / xL2 stacks; start n=16 E=8..16 screens; n12 E1 mirrors.
- logs/improve/queue22.sh (waits for queue21): denser n16 E scan (6..20), n8/n10 quality bumps, more n14 under-2000 tries.
- logs/improve/watchdog.sh: every 30 min, if no queue/runner is alive, starts an emergency auto queue. Stop by creating logs/improve/STOP.

Next ideas
1. Confirm cheapest ≥0.9 at n=8 and n=10 with the same family; put them in the scaling table.
2. Find cheapest E at n=16 for ≥0.9 (grow E / rawk / sab with n; try xeta32 and rawk32).
3. Push n=14 under ~2000 while keeping mean p(GS) ≥ 0.9 (xeta32 + xs35 + maybe rawk32).
4. Once all five n have a ~0.9 point, fit b^n and decide whether to grow rawk with n, add a second relabel only when guess_hit is low, or try a classical backbone XOR that SA almost-fixes.
5. Keep several queues deep so compute never idles until Zach says stop.
- logs/improve/queue23.sh (waits for queue22, queued 23:55): n=8 E1..3 (xL2/xL3/rawk16/xeta32), n=10 E1..3, more n=14 under-2000 tries with rawk32.
