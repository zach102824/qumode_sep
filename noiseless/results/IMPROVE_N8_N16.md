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
| 8 | 1 | sab32 rawk24 xL2 xs35 R1 xeta128 | 1.00 | 0.987 | 443 | about 40 |
| 8 | 1 | sab32 rawk24 xL2 xs35 R1 (xeta16) | 0.98 | 0.967 | 443 | about 40 |
| 10 | 1 | sab50 rawk24 xL2 xs35 R1 xeta128 | 0.96 | 0.951 | 443 | about 60 |
| 10 | 1 | sab50 rawk24 xL2 R1 xeta128 | 1.00 | 0.989 | 503 | about 60 |
| 10 | 2 | sab50 rawk24 xL2 xs35 R1 (xeta16) | 1.00 | 0.990 | 585 | about 70 |
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
| 12 | 1 | sab72 rawk24 xL2 R1 xeta128 | 0.94 | 0.925 | 503 | about 130 |
| 12 | 1 | sab72 rawk24 xL3 R1 xeta128 | 0.98 | 0.964 | 604 | about 130 |
| 12 | 2 | sab72 rawk24 xL3 xs35 R1 xeta128 | 1.00 | 0.989 | 727 | about 160 |
| 14 | 3 | sab98 rawk24 xL3 R1 xeta256 | 0.92 | 0.911 | 1210 | about 260 |
| 14 | 4 | sab98 rawk24 xL2 R1 xeta128 | 0.92 | 0.909 | 1109 | about 280 |
| 14 | 4 | sab98 rawk24 xL3 R1 xeta256 | 0.96 | 0.947 | 1513 | about 280 |
| 14 | 4 | sab98 rawk32 xL3 R1 xeta128 | 0.98 | 0.967 | 1513 | about 290 |
| 14 | 5 | sab98 rawk24 xL3 R1 xeta256 | 0.98 | 0.968 | 1816 | about 300 |
| 16 | 8 | sab128 rawk24 xs35 R1 xeta16 (full-depth explore) | 0.58 | 0.572 | 2573 | about 330 |
| 16 | 8 | sab128 rawk24 xL3 xs35 R1 xeta128 | 0.68 | 0.653 | 2005 | about 330 |
| 16 | 12 | sab128 rawk24 xL3 xs35 R1 xeta128 | 0.84 | 0.818 | 2857 | about 350 |

Older baselines for context: n=12 E5 random layout 0.92 / 0.914 at 2622 evals; n=14 E17 random layout 0.94 / 0.933 at 7470 evals. Current n=14 ~0.9 points are about 70% fewer evals than that baseline.

## Scaling table (matched ~0.9 quality), updated Oct 11 about 01:20 Shanghai

The big new lever (queue20, previously unrecorded here) is a much larger explore eta multiplier: xeta128/256 instead of 16/32. It cut the explore count E needed at n=12 and n=14 a lot (n=14: E6 -> E3/E4).

Cheapest confirmed points with success >= 0.9 and mean p(GS) >= 0.9:

| n | cheapest ~0.9 point | success / mean p(GS) | evals |
|---|---|---|---|
| 8 | E1 xL2 xs35 R1 xeta128 | 1.00 / 0.987 | 443 |
| 10 | E1 xL2 xs35 R1 xeta128 | 0.96 / 0.951 | 443 |
| 12 | E1 xL2 R1 xeta128 | 0.94 / 0.925 | 503 |
| 14 | E4 xL2 R1 xeta128 | 0.92 / 0.909 | 1109 |
| 14 alt | E3 xL3 R1 xeta256 | 0.92 / 0.911 | 1210 |
| 16 | not yet >= 0.9 | best so far 0.84 / 0.818 (E12) | 2857 |

Fit of evals = A * b^n using the n=8..14 points (443 at n=8, 1109 at n=14): b is about 1.165 per unit n over n=8..14. Piecewise: n=8..12 is almost flat (443 -> 503, b about 1.03 per unit n), and the whole growth sits in n=12 -> 14 (503 -> 1109, b about 1.48 per unit n) because one explore run is enough up to n=12 but not at n=14. With n=16 still below 0.9 at 2857 evals the full-ladder fit is not final; if n=16 lands near 3000 evals the 8..16 fit would be b about 1.27 per unit n.

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


## Status (Oct 11, 2026, about 01:25 Shanghai)

Confirmed since the Oct 10 23:20 status
- n=8, 10 measured on the full stack: both reach about 0.96-1.00 success at 443-503 evals (E=1, xL2).
- n=12 E1 xeta128 xL2: 0.94 / 0.925 at 503 evals; E2 xL3 xs35 xeta128: 1.00 / 0.989 at 727.
- n=14: xeta128/256 reaches 0.92 / 0.909 at 1109 evals (E4 xL2) and 0.92 / 0.911 at 1210 (E3 xL3 xeta256); 0.96 / 0.947 at 1513; 0.98 / 0.967 at 1513 with rawk32.
- n=16 first points: E8 xeta16 full-depth explore 0.58 / 0.572 at 2573; E8 xL3 xs35 xeta128 0.68 / 0.653 at 2005; E12 same stack 0.84 / 0.818 at 2857. Still below 0.9, E needs to go to about 16 or the explore quality needs another lever.
- Cost note: xL2/xs35 explore is cheap (about 4 s per trial at n=14), but at n=16 full-depth explore is about 55 s per trial.

Running / queued (all detached; do not kill; STOP file stops the watchdog only)
- queue21/22/23 chain (xeta16/32 n=16 E scan, n8/10 fills, n14 rawk32): still alive but now time-shared. queue25/26 temporarily SIGSTOP any xeta 16/32 job and SIGCONT it when they finish, so the 8 cores are never oversubscribed.
- queue24 (done): n=8 / n=10 xeta16 points.
- queue25 (running): n=8/10/12 xeta128 points, n=14 xeta128/256 sweep, n=16 xL3 xs35 xeta128 E12/8/6/10 and xeta256, rawk32, sab192, nz40 variants.
- queue26 (waits for queue25): n=16 E12 knob screen (rawk48 polt3, xeta256/512, sab192, nz40, R2), n=16 E16, n=14 rawk32/xeta512 tries, n=12 rawk32/xeta512.

## Update (Oct 11, 2026, about 01:50 Shanghai)
- n=16 E6 xL3 xs35 xeta128: 0.58 / 0.538 at 1579 evals (final round).
- n=16 E10 xL3 full xs xeta128: 0.76 / 0.749 at 3331 evals.
- Together with E12 xs35 (0.84 / 0.818 at 2857), n=16 is still below 0.9; queue25 continues (xeta256 E8 running), queue26 (E12 knob screen, E16, n14/n12 pushes) is queued behind it. Compute is fully busy (two n=16 jobs on 8 cores).

## Update (Oct 11, 2026, about 02:15 Shanghai)
- n=16 E8 xL3 full xs xeta256: 0.74 / 0.719 at 2725 evals (final round). Better than E8 xeta16 (0.58 at 2573) but still below 0.9 and not cheaper than E12 xs35 (0.84 / 0.818 at 2857).
- Compute fully busy: queue25 running n=16 E8 xL2 xs35 xeta128 (5 trials/worker batch), queue26 queued behind it (E12 knob screen, E16, n14/n12 pushes); queue21/22/23 time-shared and watchdog alive. No new confirmed ladder-wide result yet; n=16 remains the open point.

## Update (Oct 11, 2026, about 02:55 Shanghai)
- n=16 E8 xL2 xs35 xeta128 (rawk24): 0.52 / 0.500 at 1437 evals (final round). Cheaper but clearly below 0.9.
- n=16 E8 xL3 xs35 xeta128 rawk32: 0.70 / 0.672 at 2005 evals. Same as the rawk24 point (0.68 / 0.653 at 2005), so rawk32 gives little at n=16.
- n=16 stays the open point (best 0.84 / 0.818 at 2857 for E12). Still running: n16 E8 sab192 (queue25), then queue26 (E12 knob screen, E16, n14/n12 pushes). Compute busy; watchdog alive.

## Update (Oct 11, 2026, about 03:25 Shanghai)
- n=16 E8 xL3 xs35 xeta128 sab192 (rawk24): 0.62 / 0.613 at 2005 evals (final round). No better than sab128 (0.68 / 0.653 at 2005), so a larger annealing budget does not help n=16.
- n=16 remains the open point (best 0.84 / 0.818 at 2857 for E12). Running: nz40 variant (queue25), then queue26 (E12 knob screen, E16, n14/n12 pushes). Compute fully busy (load about 7.5 on 8 cores); queues and watchdog alive. No new ladder-wide result.

## Update (Oct 11, 2026, about 04:25 Shanghai)
- n=16 E8 xL3 xs35 xeta128 nz40 (rawk24): 0.68 / 0.652 at 2005 evals (final round). Identical to the nz0.15 baseline (0.68 / 0.653), so higher annealing noise does not help n=16.
- Knob screens so far at n=16 E8 (rawk32, sab192, nz40, xL2) are all flat or worse; the only lever that has moved n=16 is E (E12: 0.84 / 0.818 at 2857).
- Running: n=16 E12 rawk48 polt3 (45 of 50 trials done), then the rest of queue26 (E16, n14/n12 pushes). Compute fully busy (load about 8.6), watchdog alive. No new ladder-wide result.

## Update (Oct 11, 2026, about 04:55 Shanghai)
- n=16 E12 xL3 xs35 xeta128 rawk48 polt3: 0.90 / 0.878 at 2857 evals (final round; guess_hit 0.90, worst-H mean p(GS) 0.677). Same eval count as rawk24 polt2 (0.84 / 0.818), so a wider top-k readout (48 candidates, 3 fix-up passes) is the first knob that moves n=16 at fixed cost. First n=16 point with success at 0.90; mean p(GS) just under 0.9.
- Running: n=16 E12 xeta256 rawk24 (about 45 of 50 trials done at last check), then rest of queue26 (E16, n14/n12 pushes). Next: try rawk48 polt3 on E8/E10 (cheaper) and rawk64, and apply rawk48 polt3 at n=14 and 12. Compute fully busy, watchdog alive.

## Update (Oct 11, 2026, about 05:50 Shanghai)
- n=16 E12 xL3 xs35 xeta256 rawk24 polt2: 0.82 / 0.811 at 2857 evals (final round; guess_hit 0.82). Same as xeta128 rawk24 (0.84 / 0.818), so xeta256 does not help; rawk48 polt3 (0.90 / 0.878) remains the best n=16 point.
- Running: n=16 E12 xL3 full-xs xeta256 (queue26), then queue27 (n=16 E10/E12/E8/E14 and n=14/n=12 with rawk48 polt3, rawk64 polt4). Compute fully busy (load about 8), watchdog alive. No new ladder-wide result.

## Update (Oct 11, 2026, about 06:45 Shanghai)
- n=16 E12 xL3 xs35 xeta512 rawk24 polt2: 0.82 / 0.799 at 2857 evals (final round; worst-H mean p(GS) 0.593). Same as xeta128 and xeta256 (0.84 / 0.818 and 0.82 / 0.811), so the explore-eta knob is flat at n=16; rawk48 polt3 (0.90 / 0.878) is still the best n=16 point.
- Running: n=16 E12 sab192 (queue26), then queue27 (rawk48 polt3 on E8/E10/E14, n=14/n=12 with rawk48, rawk64 polt4). Compute fully busy (load about 14 on 8 cores, briefly oversubscribed), watchdog alive. No new ladder-wide result.
