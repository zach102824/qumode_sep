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
| 16 | E16 sab128 rawk48 polt3 xL3 xs35 R1 xeta128 (also rawk32 xeta256: same) | 0.92 / 0.907 | 3709 |
| 16 alt (mean just under 0.9) | E12 sab128 rawk64 polt4 xL3 xs35 R1 xeta128 | 0.92 / 0.898 | 2857 |

Note: n=12 E1 xs35 with rawk48 polt3 (443 evals) is too weak: 0.84 / 0.806. n=12 needs the xL2 run without xs35 (503 evals).

Fit evals = A * b^n (least squares on log evals, best >= 0.9 point per n):
- Strict set (443, 443, 503, 1109, 3709 at n=8,10,12,14,16): b = 1.29 per unit n (A about 38).
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
