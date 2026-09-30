# SPSA step scan at n = 16 (family F1)

Date: 2026-09-30 (Asia/Shanghai). Instances: `Hamiltonians/four_sat_scaling/n16` (20 instances,
m/n 2.69–3.88). Arms: `ry0` (RY-only product, 16 params, exact enumerated Gibbs cost) and `hea1`
(HEA L=1 on the 2×8 snake lattice, 32 params). 25 trials/instance.

Only the number of SPSA steps changes (`noiseless/run_scaling.py --steps S`, tags `steps400/600/800`;
200-step rows are the main sweep, tag `main`). Seeds are
`20260917 + 1e6·inst + 1e5·arm + 1e3·n + trial`, independent of steps, so x0 and the SPSA Δ stream
are identical to the 200-step sweep. No SPSA constant depends on the step count: a = 0.2·sqrt(37/n_params)
(ry0 0.304, hea1 0.215), c = 0.15, A = 10 (fixed, *not* rescaled to ~10% of steps), α = 0.602, γ = 0.101,
η refresh every 5 steps. Hence an S-step run is exactly the 200-step run continued: its first 200
iterations are identical. Cost evaluations per trial = 2S + 1 (401 / 801 / 1201 / 1601).

| arm | SPSA steps | trials | success | mean p(GS) | median p(GS) | mean <H> | inst with >=1 success | s/trial |
|---|---|---|---|---|---|---|---|---|
| ry0 | 200 | 500 | 0.278 | 0.0827 | 4.91e-06 | 1.927 | 20/20 | 0.4 |
| ry0 | 400 | 500 | 0.388 | 0.2590 | 2.54e-05 | 1.361 | 20/20 | 1.2 |
| ry0 | 600 | 500 | 0.452 | 0.3574 | 0.000233 | 1.123 | 20/20 | 1.7 |
| ry0 | 800 | 500 | 0.498 | 0.4184 | 0.0226 | 1.002 | 20/20 | 2.5 |
| hea1 | 200 | 500 | 0.104 | 0.0362 | 7.36e-09 | 1.839 | 17/20 | 3.2 |
| hea1 | 400 | 500 | 0.242 | 0.1040 | 1.21e-05 | 1.742 | 19/20 | 6.7 |
| hea1 | 600 | 500 | 0.346 | 0.1810 | 8.99e-05 | 1.601 | 20/20 | 9.8 |
| hea1 | 800 | 500 | 0.418 | 0.2548 | 0.000516 | 1.467 | 20/20 | 13.1 |

s/trial measured with 8 concurrent workers. Wall time for the scan (400+600+800, both arms):
36.5 min (09:35–10:12 Asia/Shanghai; 8.2 / 12.0 / 16.3 min per step count).

**Conclusion.** More SPSA steps help both arms substantially and monotonically at n = 16:
RY-only success 0.28 → 0.50 and mean p(GS) 0.08 → 0.42 from 200 to 800 steps; HEA L=1 success
0.10 → 0.42 and mean p(GS) 0.04 → 0.25, with every instance now solved at least once by both.
Neither arm has plateaued at 800 steps, and HEA L=1 gains more in relative terms but stays below
RY-only at every step count. The 200-step budget of the scaling sweep therefore understates what
both ansätze can reach at n ≥ 16; note the larger budgets also raise the matched classical budget
(WalkSAT at n = 16 already reaches 0.69 success at 4010 evaluations).

Reproduce: `noiseless/run_scaling_step_scan_n16.sh`, then `python noiseless/analyze_step_scan.py`
(writes `noiseless/results/scaling_step_scan_n16_summary.json`).
