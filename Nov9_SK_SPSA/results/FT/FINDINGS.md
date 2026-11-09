# N9-FT pilot reading

Approved simulation definitions (Zach, 18:41 Asia/Shanghai, 2026-11-09): SNAP phase 0 above level 7, Uz, per-term `|J_ij|` and `+b` per leaked qubit, ideal logical readout. Not a hardware-feasibility claim. Fidelity is unconditioned. U0 was run as a sensitivity. Validation `11000..11031` and final `20000..20999` were not read.

Held-out seeds `10000..10007`, N=6 and N=9, b=1 and 1.5, L=1, 2, 3. Transfer training seeds `10008..10015` at N=9 only. `NF_sim=16` (worst-case ladder versus NF=48: max `|ΔE|` about `4e-11`). Postpass of every selected cap-1024 Nov9 state: 384/384 within `1e-8` of NF=48. α perturbation scale was set to 0.5 before any pilot fit so a unit SPSA step in α changes `p_leak` by at most 0.01. SPSA only. One shared γ and one coefficient per feature.

The local stream's N=18 Nov3rd point (seed 10000, b=1, L=1, F_+ about 0.21317 to F about 0.61289 at 1024 queries, no QAOA pair) was not reproduced here.

## What the paired numbers say

Cold QAOA at cap 1024 reaches median F 0.93–0.999. Cold Nov3rd is in the same range at L=1 and a few points lower at L=3. Cold Nov9, with or without R1, with or without the K-sign prior, under Uz or U0, and in the strict-logical D8 diagnostic, stays at the product-state fidelity: median η is about 0 and median energy is about `-bN`. Paired against cold QAOA, Nov9 wins 0 of 8 seeds in every N/b/L cell at caps 128, 256, and 1024. At N=9, b=1, L=1, cap 1024 the median ΔF is about -0.20 (Nov9 median F 0.737 versus QAOA 0.932).

R1 is not redundant (upper-triangle residual versus span(J,K,M) is 0.21 at N=6 seed 10000 and 0.67 at N=9 seed 10000), but it does not move median F by 0.01. The K prior does not beat θ=0 on the best-of-4 selection. U0 versus Uz median `|ΔF|` is far below 0.01, and median `p_leak` on the selected cold states is below 0.001, so this pilot is not extension-dependent and not leakage-limited. The strict-logical diagnostic lands on the same product-state energy, so the gap is in the real-α circuit plus this SPSA search, not in the off-code penalty.

Transfer does not fix it. QAOA trained for 4096 instance queries and finetuned for 256 is close to cold QAOA. The same pipeline on Nov9 often leaves the product state toward higher energy and much lower F (N=9, b=1, L=1, cap 256: finetuned KM median F about 0.26 versus cold KM 0.74 and finetuned QAOA 0.93). Amortized training is 4096/16 = 256 queries per served instance plus the per-instance budget, except three L=3 training runs that aborted one or two starts on norm drift and recorded 2928, 3304, and 3752 counted queries. Those aborts are kept. The cause checked on a replay is a restricted-displacement norm of about `1-1.8e-9`, which crosses the protocol's `1e-9` failure line; the selected states themselves still match NF=48.

N=12 and N=15 were not optimized. One-query timing at NF=16 was about 7.1 ms at N=12 (about 7.3 s per 1024 queries) and about 116 ms at N=15 (about 119 s per 1024 queries, spectrum not computed). That is the feasibility note for a later round.

Sum of recorded cell and training runtimes is about 321 seconds. Artifacts: `results/FT/SUMMARY.md`, `summary.json`, `per_seed.csv`, `cutoff.json`, `postpass.json`, `conformance.json`, `MANIFEST.json`, `train/`, `cells/`.
