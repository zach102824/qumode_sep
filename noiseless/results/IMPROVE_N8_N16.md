# Improvement search on the ladder n = 8, 10, 12, 14, 16 (and screens at n = 18, 20) (wmaxsat, Gray code)

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
| 16 | 12 | sab128 rawk48 polt3 xL3 xs35 R1 xeta128 | 0.90 | 0.878 | 2857 | about 400 |
| 16 | 12 | sab128 rawk24 xL3 xs35 R2 xeta128 | 0.84 | 0.834 | 3158 | about 350 |
| 16 | 16 | sab128 rawk24 xL3 xs35 R1 xeta128 | 0.86 | 0.848 | 3709 | about 370 |

Older baselines for context: n=12 E5 random layout 0.92 / 0.914 at 2622 evals; n=14 E17 random layout 0.94 / 0.933 at 7470 evals. Current n=14 ~0.9 points are about 70% fewer evals than that baseline.

## Scaling table (matched >= 0.9 quality), updated Oct 11 about 12:00 Shanghai

Levers that built this table: SA-chosen layout, top-k readout with Hamming-1 fix-up (rawk/polt), R=1, cheap explore (xL2/xL3, xs35), and a large explore eta multiplier (xeta128/256).

Cheapest confirmed points with success >= 0.9 and mean p(GS) >= 0.9 (50 trials each):

| n | cheapest point | success / mean p(GS) | evals |
|---|---|---|---|
| 8 | E1 sab32 rawk48 polt3 xL2 xs35 R1 xeta128 | 1.00 / 0.987 | 443 |
| 10 | E1 sab50 rawk48 polt3 xL2 xs35 R1 xeta128 | 1.00 / 0.990 | 443 |
| 12 | E1 sab72 rawk48 polt3 xL2 R1 xeta128 | 0.98 / 0.967 | 503 |
| 14 | E4 sab98 rawk48 polt3 xL2 R1 xeta128 | 1.00 / 0.988 | 1109 |
| 14 alt | E3 sab98 rawk48 polt3 xL3 R1 xeta256 | 0.96 / 0.944 | 1210 |
| 16 | E12 sab128 rawk64 polt4 xL3 xs35 R2 xeta128 (new, one 50-trial run; confirm queued) | 0.94 / 0.933 | 3158 |
| 16 alt | E16 sab128 rawk48 polt3 xL3 xs35 R1 xeta128 | 0.92 / 0.907 | 3709 |
| 16 alt (mean just under 0.9) | E12 sab128 rawk64 polt4 xL3 xs35 R1 xeta128 | 0.92 / 0.898 | 2857 |

Note: n=12 E1 xs35 with rawk48 polt3 (443 evals) is too weak: 0.84 / 0.806. n=12 needs the xL2 run without xs35 (503 evals).

Fit evals = A * b^n (least squares on log evals, best >= 0.9 point per n):
- Strict set with the new n=16 point (443, 443, 503, 1109, 3158 at n=8,10,12,14,16): b = 1.27 per unit n (A about 44). Previous strict set with 3709: b = 1.29 (A about 38).
- With the n=16 alt point (2857 evals, mean p(GS) 0.898): b = 1.26.
- n=8..14 alone: b = 1.15. Two-point 8 to 16: 1.30 (strict) or 1.26 (alt).
- Piecewise: n=8..12 is almost flat; n=12 -> 14 is a factor 2.2 and n=14 -> 16 is a factor 3.3 (strict) or 2.6 (alt). The n=16 step is where the target 1.1-1.2^n is missed.


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

## Update (Oct 11, 2026, about 07:20 Shanghai)
- n=16 E12 xL3 xs35 xeta128 sab192 rawk24 polt2: success 0.78, mean p(GS) 0.771, evals 2857 (final round; guess_hit 0.80). Worse than sab128 baseline 0.84 / 0.818 at the same evals, so a larger annealing budget does not help at E12 either (same story as E8 sab192).
- n=16 E12 xL3 xs35 xeta128 nz40 rawk24 polt2: success 0.78, mean p(GS) 0.755, evals 2857 (final round; guess_hit 0.80). Flat/worse vs nz0.15 baseline 0.84 / 0.818; higher SA noise does not help at E12.
- Also logged for completeness: n=16 E12 xL3 full-xs (no xs35) xeta256 rawk24: success 0.84, mean p(GS) 0.817, evals 3937. Same quality as xs35 xeta128 at higher cost.
- Best n=16 remains rawk48 polt3: success 0.90, mean p(GS) 0.878 at 2857. Mean p(GS) still short of 0.9; success already at 0.90.
- Running: queue26 on n=16 E12 R2 xeta128 rawk24 (5/50 at last check), then E16 xeta128, E16 rawk32 xeta256, then cheap n14/n12 knobs, then queue26.done. queue27 waits on that (rawk48 polt3 on E10/E8/E14, rawk64 polt4 at E12, n14/n12 rawk48). queue28 waits on queue27.done (E16/E14 rawk48, rawk48+xeta256, R2 rawk48, E11/E13 denser, xL2 vs xL3). queue21/22/23 alive but SIGSTOPped xeta16 fleet until queue26 cleanup CONT. Watchdog alive; no STOP file.

## Update (Oct 11, 2026, about 07:50 Shanghai)
- n=16 E12 xL3 xs35 xeta128 rawk24 R2: success 0.84, mean p(GS) 0.834, evals 3158 (final round; guess_hit 0.86, worst-H mean p(GS) about 0.598). Loss/neutral vs R1 rawk24 (0.84 / 0.818 at 2857) and clearly below rawk48 polt3 (0.90 / 0.878 at 2857); a second relabel round does not beat wider readout at this E.
- n=16 E16 xL3 xs35 xeta128 rawk24 R1: success 0.86, mean p(GS) 0.848, evals 3709 (final round; guess_hit 0.86, worst-H mean p(GS) about 0.596). Slightly above E12 rawk24 (0.84 / 0.818 at 2857) but still below 0.9 and below E12 rawk48 polt3 at lower cost. Growing E alone is not enough.
- Best n=16 remains E12 sab128 rawk48 polt3 xL3 xs35 R1 xeta128: success 0.90, mean p(GS) 0.878 at 2857. Mean p(GS) still short of 0.9.
- Running: queue26 on n=16 E16 rawk32 xeta256 (just started after E16 rawk24), then cheap n14/n12 knobs, then queue26.done. queue27 waits (rawk48 polt3 on E10/E8/E14, rawk64 polt4 at E12, n14/n12 rawk48). queue28 waits on queue27.done (E16/E14 rawk48, rawk48+xeta256, R2 rawk48, denser E). queue29 waits on queue28.done (more mean-p(GS) levers). queue21/22/23 still alive (SIGSTOPped xeta16 fleet while high-priority runs). Watchdog alive; no STOP file. Compute busy (about 8 active workers). No new ladder-wide ≥0.9 result; no scaling-fit change yet.



## Update (Oct 11, 2026, about 12:00 Shanghai): catch-up for 07:50 to 12:00

First confirmed n=16 points with success and mean p(GS) both at or above 0.9 (success / mean p(GS) / evals, 50 trials, final round):
- n=16 E16 sab128 rawk32 xL3 xs35 R1 xeta256: 0.92 / 0.907 / 3709.
- n=16 E16 sab128 rawk48 polt3 xL3 xs35 R1 xeta128: 0.92 / 0.907 / 3709.
So n=16 does now reach the matched quality, but only at about 3700 evals (b about 1.29 over n=8..16).

Other n=16 results since 07:50 (rawk48 polt3 xL3 xs35 R1 unless noted):
- E8 xeta128: 0.74 / 0.712 / 2005.
- E10 xeta128: 0.80 / 0.773 / 2431.
- E12 xeta128 (earlier): 0.90 / 0.878 / 2857.
- E12 xeta256: 0.90 / 0.878 / 2857 (xeta flat).
- E12 R2 xeta128: 0.90 / 0.893 / 3158 (second relabel round raises mean p(GS) slightly, not success).
- E12 rawk64 polt4 xeta128: 0.92 / 0.898 / 2857 (best n=16 per eval so far; mean p(GS) 0.898, just short of 0.9).
- E14 xeta128: 0.88 / 0.871 / 3283. E14 xeta256 running.
- E16 xeta128: 0.92 / 0.907 / 3709.
Reading: at n=16 success is about 0.9 for E >= 12 with wide readout, E growth beyond 12 does not help much (E12 0.90, E14 0.88, E16 0.92: statistical noise of about 0.04 with 50 trials). Deeper readout (rawk64 polt4) is the only knob that moved mean p(GS) at E12 (0.878 to 0.898).

n=14 (rawk48 polt3 stack), 50 trials:
- E4 xL2 R1 xeta128: 1.00 / 0.988 / 1109 (new best n=14; rawk24 gave 0.92 / 0.909 at the same evals, rawk32 0.98 / 0.968).
- E3 xL3 R1 xeta256: 0.96 / 0.944 / 1210 (rawk24: 0.92 / 0.911; rawk32: 0.94 / 0.930).
- Weaker: rawk32 E3 xL2 xeta256: 0.80 / 0.777 / 907; rawk24 E4 xL2 xeta256: 0.82 / 0.810 / 1109; rawk24 E3 xL3 xeta512: 0.90 / 0.885 / 1210.

n=12: E1 xL2 R1 xeta128 rawk48 polt3: 0.98 / 0.967 / 503 (rawk32: 0.96 / 0.947; rawk24 xeta512: 0.92 / 0.909).
n=8: E1 xL2 xs35 xeta128 rawk48 polt3: 1.00 / 0.987 / 443; xL2 no xs35: 1.00 / 0.988 / 503.
n=10: E1 xL2 xs35 xeta128 rawk48 polt3: 1.00 / 0.990 / 443; xL2 no xs35: 1.00 / 0.989 / 503.

Compute / queues (Oct 11 about 12:00): queue28 running (n=16 E14 xeta256 now, then E12 xL2, E11, E13, E9, n14/n12 rawk48 fills). queue29 waits on queue28.done (n=16 E16 xeta256, rawk64 polt4 E12 xeta256, R2 E14, E15, xeta512, E18, n14 xeta256). queue42 (new) waits on queue29.done: n=16 rawk64 polt4 at E14, E10, E11, E16, R2, xeta256, plus rawk96 polt5. queue41 (new, 1 worker, cheap): n=8/10/12/14 rawk48 polt3 confirms and n=14 E2-E4 under about 1100 evals. queue21/22/23 stay SIGSTOPped (time-shared) until the high-priority queues finish. Watchdog alive; no STOP file.

## Consistent recipe (Oct 11, 2026)

One recipe for every n (wmaxsat, gray): top-K energy-ranked readout with K = 3n and 3 Hamming-1 fix-up passes (polt 3), xeta 128, 35 steps per explore layer (xs 35), explore depth xL 2 for n up to 14 and xL 3 at n=16, one relabel round (R=1, r=150, L=4), SA layout with about n²/2 lookups (sab 36, 50, 72, 98, 128). 50 trials each (10 instances x 5 starts).

| n | E | K | xL | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 24 | 2 | 1.00 | 0.987 | 443 | 101 |
| 10 | 1 | 30 | 2 | 0.98 | 0.970 | 443 | 136 |
| 12 | 1 | 36 | 2 | 0.78 | 0.761 | 443 | 175 |
| 14 | 4 | 42 | 2 | 0.74 | 0.720 | 869 | 368 |
| 16 | 12 | 48 | 3 | 0.90 | 0.878 | 2857 | 834 |
| 16 | 16 | 48 | 3 | 0.92 | 0.907 | 3709 | 1017 |

Fitted evals ~ b^n over n=8..16 (least squares on log evals): b = 1.25 with n=16 at E=12, b = 1.28 with n=16 at E=16.

Caveat: n=12 and n=14 miss the 0.9 bar with this recipe, so the fitted b is not at matched quality. Compared with the earlier ladder (n=12 E=1 0.98 / 0.967 at 503 evals, n=14 E=4 1.00 / 0.988 at 1109 evals, both xL 2 with 50 steps per explore layer and K=48), the drop comes mainly from xs 35 at xL 2 (cheaper explore runs, 443 vs 503 evals at n=12) and partly from the smaller K. A consistent recipe that holds 0.9 at every n likely needs 50 steps per explore layer for n up to 14, or a larger E at n=12 and 14.

Runs: logs/improve/cons_n8_E1_sab36_rawk24_xL2, cons_n10_E1_sab50_rawk30_xL2, cons_n12_E1_sab72_rawk36_xL2, cons_n14_E4_sab98_rawk42_xL2; n=16 rows reuse n16_E12 and n16_E16 _sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 (identical settings).

### Consistent recipe, xs 50 for n up to 14 (Oct 11, 2026)

Same as above but 50 steps per explore layer (xs 50) at xL 2 for n=8..14. n=16 rows unchanged (xL 3, xs 35). 50 trials each.

| n | E | K | xL | xs | success | mean p(GS) | evals | lookups |
|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 24 | 2 | 50 | 1.00 | 0.988 | 503 | 100 |
| 10 | 1 | 30 | 2 | 50 | 1.00 | 0.989 | 503 | 135 |
| 12 | 1 | 36 | 2 | 50 | 0.98 | 0.967 | 503 | 171 |
| 14 | 4 | 42 | 2 | 50 | 1.00 | 0.988 | 1109 | 360 |
| 16 | 12 | 48 | 3 | 35 | 0.90 | 0.878 | 2857 | 834 |
| 16 | 16 | 48 | 3 | 35 | 0.92 | 0.907 | 3709 | 1017 |

K=48 check at n=12 and 14 gave identical success / mean p(GS) / evals (0.98 / 0.967 / 503 and 1.00 / 0.988 / 1109) with more lookups (192, 387), so K = 3n is enough there.

Fitted evals ~ b^n (least squares on log evals): n=8..16 b = 1.24 (n=16 E=12) or b = 1.27 (n=16 E=16, the only n=16 row with both metrics at 0.9 or above). n=8..14 alone b = 1.13. Every n now has success at 0.9 or above; mean p(GS) is at 0.9 or above everywhere except n=16 E=12 (0.878). The n=14 to 16 step is the remaining cost driver.

Runs: logs/improve/consxs50_*; queue logs/improve/consistent_xs50_20261011_120723.sh.


## Update (Oct 11, 2026, about 12:30 Shanghai): post-12:00 harvest

New finals since the 12:00 catch-up (50 trials, final round; absolute values):

n=16 (rawk48 polt3 xL3 xs35 R1 unless noted)
- E11 xeta128: success 0.80, mean p(GS) 0.792, evals 2644. Below 0.9; denser E between 10 and 12 does not fill the gap.
- E12 xL2 xs35 xeta128: success 0.72, mean p(GS) 0.697, evals 2005. xL2 clearly hurts at n=16 vs xL3 at the same E (0.90 / 0.878 at 2857).
- E14 xeta256: success 0.90, mean p(GS) 0.890, evals 3283. Success at 0.9; mean p(GS) still short of 0.9. Cheaper than the strict E16 point (3709) but not a new matched ≥0.9 row.
- E13 xeta128: still running at harvest (~30/50); skipped for the table.

n=14 (rawk48 polt3 R1)
- E2 xL3 xeta128: success 0.82, mean p(GS) 0.808, evals 907. Misses 0.9.
- E3 xL2 xeta128: success 0.88, mean p(GS) 0.869, evals 907. Misses 0.9.
- E3 xL2 xeta256: success 0.90, mean p(GS) 0.875, evals 907. Success hits 0.9; mean p(GS) short. Cheapest success-only near-miss under 1100, but not a matched ≥0.9 point.
- E3 xL2 xs35 xeta128: success 0.70, mean p(GS) 0.680, evals 727. xs35 at xL2 is too weak at n=14.
- E4 xL2 xs35 xeta128: success 0.78, mean p(GS) 0.760, evals 869. Same story; keep xs50 (no xs35) for the n=14 cheapest ≥0.9 row.

consxs50 confirms (already in the consistent-recipe table): n8/10/12/14 with xs50 match the 12:00 ladder; K=48 vs K=3n identical at n=12 and 14.

Cheapest matched ≥0.9 table: no change.
- Still n8 E1 443; n10 E1 443; n12 E1 503; n14 E4 1109; n16 E16 3709 (strict).
- Fit b unchanged: about 1.29 (strict) / 1.26 (n16 alt 2857 mean 0.898). n=8..14 alone about 1.13–1.15. Bottleneck remains n=14→16.
- Near-misses that do not rewrite the table: n16 E14 xeta256 (0.90 / 0.890 / 3283), n14 E3 xL2 xeta256 (0.90 / 0.875 / 907), n16 E12 rawk64 (0.92 / 0.898 / 2857).

Compute / queues (about 12:30)
- queue28: running n16 E13 (~30/50), then E9, n14 E4 xL3, n14 E5 xL2, n12 E1 xL3.
- queue29 waits on queue28.done; queue42 waits on queue29.done (rawk64/96 denser E, R2, xeta256).
- queue41 done (n8/10/12/14 cheap fills and n14 E2–E4 under ~1100).
- queue43 waits on queue42.done; queue44 waits on queue43.done (deeper n16 mean-p(GS) / under-3000 tries, n14 under-1100 mean-p pushes).
- queue21/22/23 alive but SIGSTOPped xeta16/32 fleet; watchdog alive; no STOP file.

Next ideas
1. Push n16 mean p(GS) to ≥0.9 under ~3000: rawk64/96 at E12–14, R2 with wide readout, xs50 (full explore steps) at E12, denser E13/E15 with rawk64.
2. n14 under ~1100 with mean ≥0.9: E3 xL3 (no xs35) already 0.96 / 0.944 at 1210; try E3 with xs50 + rawk48 / rawk64; E4 cheaper only if xs35 can be rescued (so far no).
3. If rawk96 or xs50 at E12 clears mean 0.9 at 2857, rewrite the strict n16 row and refit b.


## Update (Oct 11, 2026, about 12:50 Shanghai): post-12:30 harvest

New finals since the 12:30 section (50 trials, final round; absolute values):

n=16 (rawk48 polt3 xL3 xs35 R1 xeta128)
- E13: success 0.88, mean p(GS) 0.871, evals 3070 (guess_hit 0.88). Below 0.9. Denser E between 12 and 14 does not fill the gap; do not change the cheapest ≥0.9 table.
- E9: success 0.78, mean p(GS) 0.759, evals 2218 (guess_hit 0.80). Below 0.9; cheaper E remains too weak at this readout.

queue28 tail fills (absolute values; none rewrite the cheapest ≥0.9 table)
- n14 E4 xL3 rawk48 (no xs35): success 0.98, mean p(GS) 0.967, evals 1513. Strong quality but above the 1109 cheapest n14 row.
- n14 E5 xL2 rawk48: success 1.00, mean p(GS) 0.986, evals 1311. Also above 1109.
- n12 E1 xL3 rawk48: success 0.98, mean p(GS) 0.964, evals 604. Above the 503 cheapest n12 row.

Cheapest matched ≥0.9 table: no change.
- Still n8 E1 443; n10 E1 443; n12 E1 503; n14 E4 1109; n16 E16 3709 (strict).
- Fit b unchanged: about 1.29 (strict) / 1.26 (n16 alt 2857 mean 0.898). n=8..14 alone about 1.13–1.15. Bottleneck remains n=14→16.

Compute / queues (about 12:51)
- queue28 done (E9 + n14/n12 fills).
- queue29 now running n16 E16 xeta256 rawk48 (first job after handoff); then rawk64/R2/E15/xeta512/E18 and n14 fills per queue29.sh.
- Waiting: queue42 → queue43 → queue44 → queue45 (new; waits on queue44.done).
- queue45 jobs: n16 E12 xs50 rawk48; E14 xs50 rawk64/96; E13 rawk96; E12 R2 xs50 rawk64; E15 rawk96; n14 E3 xL3 xs50 rawk64 and E3 xL3 rawk64 xeta256.
- queue41 done. Old xeta16/32 fleet SIGSTOPped again by queue29 stopwatch. Watchdog alive; no STOP file.

Next ideas (unchanged priority)
1. Push n16 mean p(GS) to ≥0.9 under ~3000 via queued rawk64/96, xs50, R2+wide readout (queue42–45).
2. n14 under ~1100 mean ≥0.9 via E3 xs50 rawk48/64 (queue43–45).
3. Rewrite cheapest ≥0.9 / refit b only if a new matched point lands.


## Update (Oct 11, 2026, about 13:20 Shanghai): queue29 first finals

n=16 (xL3 xs35 R1, 50 trials, final round; absolute values)
- E16 rawk48 polt3 xeta256: success 0.94, mean p(GS) 0.926, evals 3709 (guess_hit 0.94). Same cost as the strict E16 row (xeta128 gave 0.92 / 0.907), so quality is better but the cheapest matched n=16 point is unchanged.
- E12 rawk64 polt4 xeta256: success 0.90, mean p(GS) 0.878, evals 2857. Worse than xeta128 at the same settings (0.92 / 0.898); larger xeta does not help at E12.

Cheapest matched ≥0.9 table: no change (n8 443, n10 443, n12 503, n14 1109, n16 3709). Fit b unchanged.

Compute: queue29 now on n16 E14 R2 xeta128, then E15, E12 xeta512, E18, n14 xeta256 fills; queue42 to 45 wait behind it. Old xeta16/32 fleet SIGSTOPped; watchdog alive; no STOP file.


## Update (Oct 11, 2026, about 13:45 Shanghai): queue29 harvest (E14 R2, E15, E8 diverse)

n=16 (rawk48 polt3 xL3 xs35 xeta128, 50 trials, final round; absolute values)
- E14 R2: success 0.88, mean p(GS) 0.873, evals 3584. Second relabel round does not lift quality over R1 (E14 R1 xeta256 was 0.90 / 0.890 at 3283); not worth the extra evals.
- E15 R1: success 0.90, mean p(GS) 0.890, evals 3496. Still just under 0.9 mean; E16 (3709, 0.92 / 0.907) stays the strict n=16 row.
- E8 diverse-layout explore R1: success 0.66, evals 2005. Too weak.

Cheapest matched >=0.9 table: no change (n8 443, n10 443, n12 503, n14 1109, n16 3709). Fit b unchanged (about 1.29 strict, 1.26 alt).

Compute: busy (queue29 on n16 E12 xeta512, queue46 running, queue42 to 45 waiting; xeta16/32 fleet SIGSTOPped; watchdog alive; no STOP file).

## Find-rate diagnostic (Oct 11, 2026, about 07:50 Rome)
Per-explore-run find rate = fraction of explore runs whose own fixed-up top-K contains the GS (50 trials, rawk48 polt3):
- n=12 E1 xL2 xs50: 0.98 (trial hit 0.98).
- n=14 E4 xL2 xs50: 0.69 (trial hit 1.00); xL3: 0.80; xs35: 0.54.
- n=16 xL3 xs35: 0.44 (E8), 0.54 (E12), 0.62 (E16); trial hit 0.70 / 0.86 / 0.92. xL2 at E12: 0.41.
Runs inside one trial are strongly correlated: at n=16 E16, 4 of 50 trials find the GS in 0 of 16 runs, while most trials find it in 10 or more. So adding E has diminishing returns; the misses are trials where every explore run is stuck. Per-instance find rates at n=16 are uniform (0.49-0.85), so it is not one bad instance.
- New samode suffix `_div` (SA centers forced at least n/4 Hamming apart): n=16 E8 xL3 xs35 gives 0.66 / 0.650 at 2005, worse than baseline 0.74 / 0.712. Dropped.
- queue46 (priority, runs alongside queue29): fewer deeper explore runs at n=16 (E4-8, xL3/xL4, xs50/80, R1/R2).

## n=16: adaptive early stop (idea 3) and independent SA per explore run (idea 2), queue47 (in progress, 2026-10-11)

New opt-in flags (defaults unchanged, tests in tests/test_explore_adaptive_saind.py):
- `--xconsec N`: each explore run's own guess = best Hamming-1 fix-up of its polt lowest-energy top-rawk states; stop exploring once the last N runs give the same guess energy (--explore = Emax).
- `--saind B`: every explore run gets its own SA pass of B lookups (own seed) for its layout and centre.

Base: n=16 wmaxsat gray, sab128, L=4 s=50 r=150 R=1 lr=0.05, xeta128 reta2, xL3, rawk48 polt3, 50 trials.
Queue: logs/improve/queue47_xconsec_saind.sh (detached with setsid), summary: `python noiseless/summ_q47.py`.
find = fraction of explore runs whose own fix-up guess is the GS (xconsec arms) or whose raw top-48 is within Hamming 1 of the GS (other arms).
Results so far (mean evals):

| arm | success | mean p(GS) | mean evals | lookups | mean runs | find |
|---|---|---|---|---|---|---|
| E16 xc2 xs35 | 0.66 | 0.640 | 1967 | 825 | 7.8 | 0.14 |
| E16 xc3 xs35 | 0.84 | 0.827 | 3257 | 1293 | 13.9 | 0.15 |
| E8 si128 xs35 | 0.80 | 0.787 | 2005 | 1525 | 8.0 | 0.19 |

## queue47 interim update (Oct 11, 2026, about 16:30 Shanghai)
n=16 early-stop (xconsec) and independent SA per explore run (si128 = saind 128 lookups per run); absolute values, 50 trials unless noted:
- E16 xc2 xs35 si128: success 0.76, mean p(GS) 0.742, evals 1907, lookups 1599.
- E16 xc3 xs35 si128: success 0.92, mean p(GS) 0.908, evals 3326, lookups 2915. Matches the strict n=16 row (0.92 / 0.907 at 3709) with about 10% fewer evals, but classical lookups are far above n^2 (256).
- E16 xc2 xs50: 0.64 / 0.626 at 2404 evals. E16 xc3 xs50 (34 of 50 trials so far): 0.88 / 0.872 at 4596 evals; xs50 is not better than xs35.
Takeaway: independent SA per run helps early stop (xc3: 0.84 to 0.92 success), so the miss mode is partly shared layout. Next: cheaper per-run SA (si32/si64) with xc3, and si128 with E12 cap. Compute: busy (queue47/48 and queue43-45 running; watchdog alive).
Queued queue49 (si64/si32 xc3, E12 si128, independent-seed confirm of xc3 si128) behind queue48.

## queue47 finals (Oct 11, 2026, about 16:55 Shanghai)
n=16 (50 trials, absolute values):
- E16 xc3 xs50 (no per-run SA): success 0.90, mean p(GS) 0.888, evals 4434. Worse than xs35.
- E8 si128 xs50: success 0.82, mean p(GS) 0.812, evals 2725 (xs35: 0.80 / 0.787 at 2005).
Cheapest matched >=0.9 table unchanged (n8 443, n10 443, n12 503, n14 1109, n16 3709). Compute busy (queue43 E12 xs50, queue47/48/49 running; watchdog alive).


## Update (Oct 11, 2026, about 17:30 Shanghai): harvest, new n=16 matched point, queue50

New confirmed n=16 results (50 trials each, final round, absolute values; base: sab128 xL3 xs35 xeta128 reta2, gray, wmaxsat; lookups in brackets):
- E12 R2 rawk64 polt4: success 0.94, mean p(GS) 0.933, evals 3158 (lookups 1074). Meaningful new result: cheapest matched n=16 point so far (was 3709 with 0.92 / 0.907), about 15% fewer evals with better quality. This run finished at 15:05 and was not in the table before. Single 50-trial run; independent-seed confirm is queued first in queue50.
- E15 R1 rawk64 polt4: 0.92 / 0.909, 3496 evals (1245). Also matched (mean at or above 0.9), cheaper than 3709 but above 3158.
- E13 R1 rawk64 polt4: 0.90 / 0.891, 3070 evals (1123). E11 R1: 0.82 / 0.812, 2644. E10 R1: 0.82 / 0.792, 2431.
- E12 R1 rawk96 polt5: 0.92 / 0.898, 2857 evals (1476 lookups). Same evals as the 0.898 alt, no gain from wider readout.
- E12 R1 xeta512 rawk48: 0.88 / 0.858, 2857 (830). E18 R1 rawk48: 0.94 / 0.926, 4135 (1109).
- Fewer, deeper explore runs: E6 xs50 rawk48: 0.68 / 0.669 at 2119 (551); E6 xs50 with diverse SA: 0.72 / 0.701; E4 xL4 xs50: 0.62 / 0.598 at 1917 (455); E4 R2 xs80: 0.62 / 0.615 at 2534; E8 xL4 xs35: 0.78 / 0.771 at 2573 (652); E5 xL4 xs50 diverse: 0.68 / 0.672 at 2321. None competitive.
- Early stop (xconsec) with xs50: E16 xc2 si128: 0.78 / 0.762, 2343 evals, 1438 lookups. E16 xc3 si128 xs50 is running (2 of 50 done at last look).
- E12 R1 xs50 rawk64 polt4 (running, 32 of 50 done): 0.84 / 0.834 at 3937 evals (1056 lookups). xs50 is worse and more expensive than xs35.

n=14 (50 trials): E4 xL3 xeta256 rawk48 polt3: 1.00 / 0.982, 1513 evals (382 lookups). Does not beat the 1109 row (E4 xL2 xeta128: 1.00 / 0.988).

Cheapest matched >=0.9 table (updated):
- n8 443 (1.00 / 0.987), n10 443 (1.00 / 0.990), n12 503 (0.98 / 0.967), n14 1109 (1.00 / 0.988), n16 3158 (0.94 / 0.933; lookups 1074).
- Fit evals = A b^n: strict b = 1.27 (A about 44), was 1.29. With n=16 at 3496: 1.29; at 2857 (alt, mean 0.898): 1.26. n=8..14 alone about 1.15. n=14 to 16 still dominates (1109 to 3158). n=16 lookups about 1074, still about 4 times n^2 (256); the cheapest-lookup matched n=16 point remains xc3 si128 at 2915 lookups (not cheaper), so lookups near n^2 is not yet reached with matched quality (E16 xc3 xs50 no per-run SA: 0.90 / 0.888 at 1263 lookups, just under).

Pending / queued (all detached; compute busy, load about 16):
- Running: queue43 (E12 xs50 finishing, then E14 rawk96, E12 R2 rawk96, n14 E3/E4 rawk48/64), queue47 (xc3 si128 xs50 last arm).
- Waiting chain: queue44 -> queue45 -> queue48 (off50 confirms n8/n10/n12/n14, xeta256 n16) -> queue49 (si64/si32 xc3, E12 si128, independent-seed xc3 si128).
- New queue50 (starts now, 4 workers, runs alongside): n16 E12 R2 rawk64 independent-seed confirm (trial offset 50), then R2 at E10, E11, E9; E12 R2 with xc3 si64; E12 R2 xeta256; n14 E3 R2 and E2 R2 xeta256; n8 / n10 E1 xs25. Watchdog alive; no STOP file.


## Update (Oct 11, 2026, 12:50 CEST): independent-seed confirm of n=16 E12 R2, n=14 fills

Absolute values, 50 trials each, final round, gray, wmaxsat.
- n=16 E12 R2 rawk64 polt4 xL3 xs35 xeta128 (trial offset 50, independent seed): success 0.88, mean p(GS) 0.874, evals 3158. First 50 trials gave 0.94 / 0.933, so the pooled 100 trials are 0.91 / 0.903 at 3158 evals. The 3158 point is matched only marginally (mean about 0.90), not a clear win over 3709 (0.92 / 0.907). The rawk96 polt5 run with the same seed gives the identical 0.94 / 0.933, so wider readout changes nothing at R2.
- n=16 E12 si64 xs35 early-stop arm: 0.80 / 0.791 at 2857 evals. Not matched.
- n=14 E4 xL2 rawk64 polt4 xeta128: 1.00 / 0.988 at 1109 (same as the 1109 row). n=14 E3 xL2 rawk64 xeta256: 0.92 / 0.895 at 907. n=14 E3 rawk48 xs50: 0.88 / 0.869 at 907. So n=14 under 1000 evals reaches about 0.9 but not strictly matched.

Cheapest matched table (pooled for n=16): n8 443, n10 443, n12 503, n14 1109, n16 3158 (pooled 0.91 / 0.903; single-run 0.94 / 0.933). Fit b about 1.27 (strict, with pooled n=16 slightly weaker); n=8..14 alone about 1.15. n=14 to 16 still dominates.

Running (all detached; load about 12, watchdog alive, no STOP file): queue50 (n16 E10 R2 25 of 50, then E11 R2, n14 E3 R2, E12 R2 xc3 si64, n8/n10 E1 xs25, E9 R2, n14 E2 R2 xeta256, E12 R2 xeta256), queue44 (n16 E17 R1 5 of 50, then E12 rawk128, E14 R2, E16, cheap ladder confirms), queue45/48/49 waiting chain.


## Update (Oct 11, 2026, about 19:55 Shanghai): harvest of queue44 / queue50 results, no change to the matched table

Absolute values, gray, wmaxsat, base sab128 xL3 xs35 xeta128 reta2 (n=14: sab98 xL2), 50 trials each unless noted; lookups in brackets. Numbers taken from the per-run jsonl files in noiseless/results/relayout_runs/kbit (untracked, gitignored); progress logs are in logs/improve/.

n=16 final:
- E17 R1 rawk64 polt4: success 0.94, mean p(GS) 0.930, evals 3922 (1363).
- E18 R1 rawk64 polt4: 0.94 / 0.926, evals 4135 (1424).
- E12 R1 rawk128 polt5: 0.92 / 0.898, evals 2857 (1858 lookups). Same as the rawk96 polt5 and rawk64 values at E12 R1; a wider readout only adds lookups.
- E10 R2 rawk64 polt4: 0.86 / 0.854, evals 2732 (949).
- E11 R2 rawk64 polt4: 0.88 / 0.874, evals 2945 (1010).
- E12 R2 rawk64 polt4, pooled over two independent 50-trial seeds (100 trials): 0.91 / 0.903, evals 3158 (1071). Unchanged from the previous update.

n=14 final:
- E3 R2 rawk48 polt3 xL2 xeta128: 0.90 / 0.893, evals 1208 (349). Worse than E3 R1 and than the 1109 row.

Partial (not final, do not use yet):
- n=16 E14 R2 rawk64 polt4: 16 of 50 trials done, 1.00 / 0.994 at 3584 evals (1213). Too few trials; cost is above 3158 anyway.
- n=16 E12 R2 rawk64 polt4 with xc3 si64: 16 of 50 trials done, 0.75 / 0.743 at 3118 evals (2074). Looks unmatched and not cheaper.

Cheapest matched table (mean p(GS) at or above 0.9), unchanged:
- n=8: 443 evals (1.00 / 0.987)
- n=10: 443 evals (1.00 / 0.990)
- n=12: 503 evals (0.98 / 0.967)
- n=14: 1109 evals (1.00 / 0.988; 474 lookups)
- n=16: 3158 evals (pooled 0.91 / 0.903; single first seed 0.94 / 0.933; 1071 lookups). Matched only marginally.
Cheapest n=16 points at or above 0.92 success and mean at or above 0.9 remain: E12 R2 at 3158, E15 R1 at 3496 (0.92 / 0.909), strict old row 3709 (0.92 / 0.907). E13 R1 at 3070 gives 0.90 / 0.891 (mean just under 0.9).

Refit evals = A b^n (least squares on log evals, n = 8, 10, 12, 14, 16):
- with n=16 at 3158 (pooled): b = 1.274, A = 44.2 (same as before).
- with n=16 at 3496: b = 1.287, A = 39.9.
- with n=16 at 3709: b = 1.295, A = 37.6.
- n = 8 to 14 only: b = 1.155.
Fit does not change: n=14 to 16 (1109 to 3158) still dominates. n=16 lookups (1071) are about 4 times n^2 (256); n=14 lookups (474) are about 2.4 times n^2 (196).

Conclusion: no new matched n=16 point below 3158. Cheaper R2 points (E10: 2732, E11: 2945) fall to mean p(GS) 0.854 and 0.874. No n=14 point under 1000 evals reaches mean 0.9 with success 0.9 or better beyond the earlier 907-eval arms (0.92 / 0.895 at E3 xeta256).

Compute / queue state (8 cores, load about 15, watchdog alive, no STOP file): queue44 running n16 E14 R2 (16 of 50 done), then E16 rawk96 polt5, n12 E1 xL3, n14 E3 xL3, n8 E1 confirms; queue50 running n16 E12 R2 xc3 si64 (16 of 50), then n8 / n10 E1 xs25, n16 E9 R2, n14 E2 R2 xeta256, n16 E12 R2 xeta256; waiting chain queue45 then queue48 (n8, n10, n12, n14 independent-seed confirms with trial offset 50, n16 xeta256 E13 to E16) then queue49 (si64/si32 xc3, E12 si128, off50 xc3 si128). No new jobs launched because the machine is fully loaded.


## Update (Oct 11, 2026, 15:55 CEST): harvest of queue44 / queue50 finals, n=8 and n=10 now 403 evals, fit b = 1.29

Absolute values, gray, wmaxsat, base sab128 xL3 xs35 xeta128 reta2 (n=14: sab98; n=8, n=10: xL2), 50 trials each (all final) unless noted; lookups in brackets. Numbers taken from the per-run jsonl files in noiseless/results/relayout_runs/kbit (untracked, gitignored); progress logs are in logs/improve/.

n=16 final:
- E14 R2 rawk64 polt4: success 0.92, mean p(GS) 0.913, evals 3584 (1196). Matched but above 3158 evals. The 16-trial partial (1.00 / 0.994) was a lucky start.
- E16 R1 rawk96 polt5: 0.94 / 0.926, evals 3709 (1831 lookups). Same evals as the old 3709 row with better quality, no gain in cost.
- E12 R2 rawk64 polt4 with xc3 si64: 0.80 / 0.793, evals 2975 (1969). Not matched.

n=14 final:
- E3 R1 xL3 xs50 rawk48 polt3: 0.88 / 0.871, evals 1210 (332). Not matched, not cheaper than 1109.

n=12 final:
- E1 R1 xL3 xs50 rawk48 polt3: 0.98 / 0.964, evals 604 (191). Matched but above the 503 row.

n=10 final:
- E1 R1 xL2 xs25 rawk48 polt3: 0.96 / 0.928, evals 403 (166 lookups). New cheapest matched n=10 point (was 443 with 1.00 / 0.990).

n=8 final:
- E1 R1 xL2 xs25 rawk48 polt3: 1.00 / 0.988, evals 403 (126 lookups). New cheapest matched n=8 point (was 443).
- E1 R1 xL2 xs35 rawk24 polt3 with sab32 (confirm of the 443 row): 1.00 / 0.987, evals 443 (101 lookups). Reproduces the existing row exactly.

Partial (not final, do not use yet):
- n=16 E9 R2 rawk64 polt4: 43 of 50 trials done, 0.81 / 0.808 at 2519 evals (893). Looks unmatched.
- n=16 E12 R1 xs50 rawk48 polt3 (queue45): 20 of 50 trials done, 0.85 / 0.839 at 3937 evals (838). Looks worse and more expensive than the xs35 runs.

Cheapest matched table (success and mean p(GS) both at or above 0.9), refreshed:
- n=8: 403 evals (1.00 / 0.988; 126 lookups)
- n=10: 403 evals (0.96 / 0.928; 166 lookups). Only 0.96 success, so the 443 row (1.00 / 0.990) remains the safer alternative.
- n=12: 503 evals (0.98 / 0.967)
- n=14: 1109 evals (1.00 / 0.988; 474 lookups)
- n=16: 3158 evals (pooled over two seeds 0.91 / 0.903; first seed 0.94 / 0.933; 1071 lookups). Matched only marginally.

Refit evals = A b^n (least squares on log evals, n = 8, 10, 12, 14, 16):
- table above (403, 403, 503, 1109, 3158): b = 1.292, A = 35.9.
- with n=16 at 3496 (E15 R1, 0.92 / 0.909): b = 1.306, A = 32.4.
- with n=16 at 3709: b = 1.313, A = 30.5.
- with the old 443 rows for n=8 and n=10 (3158 at n=16): b = 1.274, A = 44.2 (previous fit).
- n = 8 to 14 only: b = 1.177 (was 1.155; the cheaper small-n points steepen the fit slightly).
The fit gets slightly steeper (1.29 versus 1.27) only because n=8 and n=10 got cheaper, not because n=16 got worse. n=14 to 16 (1109 to 3158) still dominates. Nothing at n=16 below 3158 evals reaches mean 0.9 yet.

Compute / queue state (8 cores, load about 17, watchdog alive, no STOP file): queue44 finished. Running: queue50 (n=16 E9 R2 at 43 of 50, then n14 E2 R2 xeta256, n16 E12 R2 xeta256), queue45 (n=16 E12 R1 xs50 rawk48, 20 of 50). Waiting: queue48 (independent-seed confirms n8, n10, n12, n14 with trial offset 50, n16 xeta256 E13 to E16) and queue49 (si64/si32 xc3, E12 si128, off50 xc3 si128). No new jobs launched because the machine is fully loaded.

## Update (2026-10-11 16:25 CEST): n=16 hunt continues; n=18 and n=20 instances ready; queue51 queued

Absolute values, gray, wmaxsat. Numbers from jsonl / logs/improve summary JSONs.

### Fresh n=16 finals (50 trials)
- E9 R2 rawk64 polt4 xL3 xs35 xeta128: success 0.84, mean p(GS) 0.834, evals 2519 (lookups about 893). Under 2500 but not matched (mean well below 0.9).
- E12 R1 xs50 rawk48 polt3 xL3 xeta128 (queue45): success 0.90, mean p(GS) 0.890, evals 3937. Not matched, more expensive than xs35.
- n=14 E2 R2 xL3 xeta256 rawk48: success 0.86, mean p(GS) 0.854, evals 1208. Not matched.

### Cheapest matched table (unchanged)
- n=8: 403 evals (1.00 / 0.988)
- n=10: 403 evals (0.96 / 0.928); safer alt 443 (1.00 / 0.990)
- n=12: 503 evals (0.98 / 0.967)
- n=14: 1109 evals (1.00 / 0.988)
- n=16: 3158 evals (pooled 0.91 / 0.903). Nothing under about 2500 reaches mean 0.9 yet. Closest under-2500 miss: E9 R2 at 2519 with 0.84 / 0.834.

Fit with (403, 403, 503, 1109, 3158): b about 1.29. n=8 to 14 alone about 1.18. n=14 to 16 still dominates.

### n=18 and n=20 Hamiltonians
Generated 10 unique-GS wmaxsat instances each under `Hamiltonians/wmaxsat/` with the same rules (m=round(9.93*n), weights 1-5, unique ground state):
- n=18: m=179, 10/10 done (4 with E_gs=0).
- n=20: m=199, 10/10 done (9 with E_gs=0).
`Hamiltonians/wmaxsat_gen.py` now covers n in (8,10,12,14,16,18,20).

### Compute / queues (detached setsid/nohup; no STOP file)
- Running: queue50 (last arm: n16 E12 R2 xeta256, 8 of 50 at last look, early 1.00 / 0.992 at 3158 — too few trials), queue45 (n16 E14 xs50 rawk64 after finishing E12 xs50).
- Waiting chain: queue48 (off50 confirms + n16 xeta256 E13-E16) then queue49 (si32/si64 xc3).
- New: `logs/improve/queue51_n16cheap_n18n20.sh` waits for queue50 (+ soft-wait queue45), then (1) n=16 under-2500 hunt with xc3+saind, R2 denser E, colder xeta256, readout variants; (2) n=18 screens with K=54=3n, polt3, xeta128, xs50, xL3, R=1, sab162, grow E=4..24 at 10 then 20 trials; (3) n=20 screens with K=60, sab200, E=8..32 at 10 then 20 trials. Watchdog alive.

No new matched n=16 point under 3158. Exploration continues non-stop.


## Update (Oct 11, 2026, 16:25 CEST): harvest of queue50 / queue45 finals; matched table and fit unchanged

Absolute values, gray, wmaxsat, base sab128 xL3 xs35 xeta128 reta2 (n=14: sab98 xL3), 50 trials each (all final) unless noted; lookups in brackets. Numbers taken from the per-run jsonl files in noiseless/results/relayout_runs/kbit (untracked, gitignored); progress logs are in logs/improve/.

n=16 final:
- E9 R2 rawk64 polt4: success 0.84, mean p(GS) 0.834, evals 2519 (887). Not matched (below the E10/E11 R2 misses already logged).
- E12 R1 xs50 rawk48 polt3: 0.90 / 0.890, evals 3937 (831). Mean just under 0.9 and more expensive than the xs35 E12 R1 row at 2857; xs50 does not help here.

n=14 final:
- E2 R2 xL3 xeta256 rawk48 polt3: 0.86 / 0.854, evals 1208 (301). Not matched; R2 at E2 does not reach mean 0.9.

Partial (not final, do not use yet):
- n=16 E12 R2 rawk64 polt4 xeta256 (queue50 last job): 8 of 50 trials done, 1.00 / 0.992 at 3158 evals (1069). Same eval cost as the xeta128 E12 R2 row; quality looks strong so far but too few trials.
- n=16 E14 R1 xs50 rawk64 polt4 (queue45): just started (0 of 50).

Cheapest matched table (success and mean p(GS) both at or above 0.9), unchanged:
- n=8: 403 evals (1.00 / 0.988; 126 lookups)
- n=10: 403 evals (0.96 / 0.928; 166 lookups). Safer alt still 443 (1.00 / 0.990).
- n=12: 503 evals (0.98 / 0.967)
- n=14: 1109 evals (1.00 / 0.988; 474 lookups)
- n=16: 3158 evals (pooled 0.91 / 0.903; first seed 0.94 / 0.933; 1071 lookups). Matched only marginally.

Near-miss reminder still open for a sub-1100 n=14 point: E3 xL2 xeta256 rawk64 polt4 at 907 evals scored 0.92 / 0.895 (mean just under 0.9). queue51 will push rawk/polt and independent-seed confirms at that cost.

Refit evals = A b^n (least squares on log evals, n = 8, 10, 12, 14, 16):
- table above (403, 403, 503, 1109, 3158): b = 1.292, A = 35.9 (unchanged).
- n = 8 to 14 only: b = 1.177 (unchanged).
No new cheapest matched point; fit unchanged. n=14 to 16 still dominates.

Compute / queue state (8 cores, load about 11, watchdog alive, no STOP file): Running: queue45 (n16 E14 xs50 rawk64 after finishing E12 xs50), queue50 (last job n16 E12 R2 xeta256 at 8 of 50). Waiting: queue48 (after queue45.done; n16 xeta256 E13–E16 + independent-seed confirms), queue49 (after queue48; xc3 si variants). New: queue51_n16cheap_n18n20 waits for queue50.done (then soft-waits queue45) — n16 under-2500 hunt plus n18/n20 screens. queue51 waits for queue49.done — n14 sub-1100 near-miss pushes (E3 xL2 xeta256 rawk96/80/off50/xeta512, E2 R1 xeta256), E12 R2 xeta256 off50 confirm, E14 R2 xeta256, and n8/n10/n12 cheap confirms.


## Update (Oct 11, 2026, 17:30 CEST): harvest E12 R2 xeta256 + E14 xs50 + E13 rawk96; matched table unchanged

Absolute values, gray, wmaxsat, base sab128 xL3 xs35 reta2, 50 trials each (all final) unless noted; lookups in brackets. Numbers from jsonl in noiseless/results/relayout_runs/kbit (gitignored) and summary footers in logs/improve/.

### n=16 finals
- E12 R2 rawk64 polt4 xL3 xs35 xeta256 (queue50 last arm): success 0.92, mean p(GS) 0.913, evals 3158 (1071). Matched at the same eval cost as the current cheapest n=16 row. Quality is a clean single-seed confirm (better than the pooled xeta128 0.91 / 0.903 at the same 3158). Does not beat 3158 on cost.
- E14 R1 xs50 rawk64 polt4 xL3 xeta128 (queue45): 0.90 / 0.891, evals 4543 (1171). Not matched (mean under 0.9) and more expensive than xs35; log as miss. Same story as the earlier E12 xs50 rawk48 miss (0.90 / 0.890 at 3937).
- E13 R1 rawk96 polt5 xL3 xs35 xeta128 (queue45): 0.90 / 0.891, evals 3070 (1565). Mean just under 0.9, so not matched. Same quality as the earlier E13 rawk64 row at 3070 (0.90 / 0.891); wider readout adds lookups without lifting mean over 0.9. Cost is under 3158 but fails the mean bar.

### Cheapest matched table (unchanged)
- n=8: 403 evals (1.00 / 0.988; 126 lookups)
- n=10: 403 evals (0.96 / 0.928; 166 lookups). Safer alt still 443 (1.00 / 0.990).
- n=12: 503 evals (0.98 / 0.967)
- n=14: 1109 evals (1.00 / 0.988; 474 lookups)
- n=16: 3158 evals (pooled xeta128 0.91 / 0.903; first xeta128 seed 0.94 / 0.933; new xeta256 seed 0.92 / 0.913; 1071 lookups). Still the cheapest matched n=16 point; nothing under 3158 reaches mean 0.9.

### Fit (unchanged)
evals = A b^n on (403, 403, 503, 1109, 3158): b = 1.292, A = 35.9. n=8 to 14 alone: b = 1.177. n=14 to 16 still dominates. No new cheapest matched point at any n, so fit b is unchanged.

### Compute / queue state (8 cores, load about 8.6, watchdog alive, no STOP file)
- queue50 finished (wrote queue50.done at 17:19 CEST).
- Running: queue45 on n16 E12 R2 xs50 rawk64 (after E13); then E14 xs50 rawk96, E15 rawk96, n14 E3 xs50 / xeta256. Workers=8.
- Waiting (alive, do not kill): queue48 (after queue45.done; off50 confirms + n16 xeta256 E13-E16), queue49 (after queue48; si32/si64 xc3), queue51_n16cheap_n18n20 (after queue50.done, soft-waiting queue45 up to 120 min; n16 under-2500 + n18/n20 screens), queue51 (after queue49; n14 sub-1100), queue52 (after both queue51*.done; more n16 under-3000 + n14 + n18/n20).
- New: queue53 waits for queue52.done — denser n16 under-3158 mean>=0.9 tries (R2 E11-E13, xeta256/512, rawk80-128, careful large-saind early-stop), more n14 sub-1100 near-miss seeds, and cheap ladder off100 confirms.

No message to Zach: no new cheapest >=0.9 point and no fit change.
