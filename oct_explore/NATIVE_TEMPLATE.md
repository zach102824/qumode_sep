# Templated native cavity-chain ansatz (Zach plan 2026-10-19) — running summary

Method (own terms, QAOA is only the baseline): all data in cavities (Gray code, <=5 bits/cavity), chain of neighbour buses.
Per layer: compiled clause phases (SNAP per cavity + neighbour bus tables, computed from the formula; terms spanning
non-neighbour / 3+ cavities least-squares projected) -> calibrated one-bit-flip mixer per cavity (short native SNAP+D
sandwich fitted offline to prod_j exp(-i b X_j/2) on the cavity's Gray digits). Learned: strengths g_l, b_l, optional GNN
clause weights. Code: `v3/native/` (mixcal.py, calib.py, compiler.py, assign.py, nsim.py, fixed.py, evalnative.py, gnnw.py).
Results: `results/v3/native/`, logs `logs/v3/native/`. Size families (Zach 17:34): 16-level train n=8+12 test 16;
32-level train n=10+15 test 20.

## M1 mixer calibration (unitary fidelity on valid levels, outer SNAPs merged into compiled SNAPs, leak penalised)
| levels | k=2 | 3 | 4 | 6 | 8 | 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 (K3) Fmin over b in [0,pi] | 0.13 | 0.34 | 0.67 | 0.83 | 0.9987 | | | | | |
| 16 (K4) | | | 0.36 | | | 0.954 | 0.998 | 0.9998 | | |
| 32 (K5) NF56 | 0.018 | 0.067 | 0.17 | 0.22 | 0.46 | | 0.78 | | 0.951 | 0.989* |
(*K5 at NF=56 exploits truncation: F at NF+40 drops to 0.953 -> refitting k32/k40 at NF=80, running.)
- Needs ~N displacements for a full-unitary fit (~16-20 at 16 levels, >=32 at 32 levels). Prep (flat state): 3 pairs >= 0.99998.
- Interpolating fitted params between beta grid points fails (F~0): branches not smooth -> use per-beta fits (warm from grid), cheap.
- Sensitivity (n=6, QAOA-angle strengths): per-mixer F~0.89 (k=4) -> 5.17/10.74; F~0.98-0.99 (k=6) -> 3.62/7.30;
  F~0.998 (k=8) -> 3.37/6.69; ideal 3.31/6.72. Need F >~ 0.995.
- 16-level family, k=16 (F 0.9988-0.9993 at learned b) / k=20 (0.9999): native == ideal within noise (table below).

## M2 compiler
Walsh/FWHT projection (exact for 2 cavities). Sanity: n=6 all w=1 ideal mixer = qubit QAOA L4 exactly (3.31/6.72, loss 1.2188).
Residual variance fraction (100 test formulas): block / min-cut assignment
n=12 0.28/0.18, n=14 0.25/0.16, n=15 0.27/0.17, n=16 0.41/0.28, n=18 0.39/0.28, n=20 0.39/0.28 (n=8,10: 0).

## M3 fixed strengths (L4, all clause weights 1), 16-level family, test median/B90 of 1/p
| version | n=8 | n=12 | n=16 |
|---|---|---|---|
| qubit-QAOA L4 baseline (trained 8+12) | 5.36/10.07 | 12.97/31.76 | 34.82/98.20 |
| same angles, all-to-all phases + native mixer k20 | 5.36/10.15 | 12.77/31.96 | 35.06/101.79 |
| chain compiled, ideal mixer | 5.40/10.31 | 37.14/407.2 | 394.4/20998 |
| chain compiled, native mixer k16 | 5.42/10.27 | 38.10/429.1 | 398.0/27541 |
| chain compiled, native mixer k20 | 5.41/10.32 | 37.10/404.6 | 393.1/20833 |
| chain + min-cut assignment, ideal | 5.36/10.36 | 25.03/176.7 | 203.0/5684 |
| random | 128/256 | 2048/4096 | 32768/65536 |
Diagnostics (eval of chain angles, ideal mixer): gamma rescale 0.3-1.0 does not help; any-two-cavity connectivity
("pairs", e.g. chain + cavity SWAP network) n=12 21.2/69.6, n=16 86.8/1477 (still far from all-to-all 36/114 -> 3-cavity
terms matter); product-measure projection with ORACLE marginals biased 0.3 to a solution: n=16 25/52 (diagnostic only);
polarity-heuristic marginals: n=16 246-270 median, tail unchanged.
=> The dominant loss is the chain connectivity of the clause phases, not the mixer.

## Running / next (19:20 CST)
- F32_gray_chain, F32_qaoa (fixed.py, 32-level family, n=10+15 -> 10/15/20), mixcal NF80 K5 k32/k40.
- gnnw.py (GNN clause weights warm from S_g4_snapF_s1 + learned g/b): G16_chain_mc_w, then G16_chain_w, G16_full_w.
- Next ideas: learned per-variable marginal head for product-measure projection; ancilla/transmon-mediated 3-cavity
  phases (hardware question); per-layer-different assignments are impossible (bits fixed) unless cavity SWAPs are native.

## Software escape hatch closed (19:20 CST)
Algebra: conjugating a diagonal by the Gray X-mixer never enlarges its Z-support. So any depth of
(neighbour-pair diagonal + X mixer) stays inside the span of local + neighbour-pair terms. Numeric check
at n=6 as 2/2/2 (resid 0.33): full-cost L4 2.77/5.63 vs chain-only L4 5.72/29.2. No software-only way to
synthesize 3-cavity diagonal phases from the chain gate set.

## Milestone 4 in progress (19:24 CST)
- G16_chain_w (clause weights + learned g/b, block assign, warm S_g4 + x0 F16_gray_chain): step 250 val n=12
  45.1/249.5 (from 70.8/725); ~0.22 s/step, ETA ~20:30 for 3000 steps + test.
- G16_chain_mc_w: still building n=12 mincut pool (~3300/4500 perms); then trains.
- G16_full_w queued behind G16_chain_w (q_g16b.sh).
- Bus-off (G16_local_w), bias head (B16_chain_mc), random-assign (G16_chain_rand) queued in q_next.sh.
- gnn_bias.py: GNN hv -> soft p_v, product-measure projection; smoke OK (compile 10/28 s at n=8/12).
- F32: 4/5 starts each; mixcal NF80 K5 k32/k40 pass0 mid-grid.

## Milestone 4 results (2026-10-19 20:02 CST) — GNN clause weights + bias head
All L4, 16-level family train n=8+12, test median/B90 of 1/p. Ideal mixer.

| version | n=8 | n=12 | n=16 |
|---|---|---|---|
| QAOA L4 (full-cost baseline) | 5.36/10.07 | 12.97/31.76 | 34.82/98.20 |
| G16_full_w (GNN w + g/b, all-to-all phases) | **4.76/8.81** | **12.31/26.31** | **31.19/74.14** |
| fixed chain (block) | 5.40/10.31 | 37.14/407 | 394/20998 |
| G16_chain_w (GNN w + g/b, block) | 5.39/9.65 | 34.76/302 | 408/13422 |
| fixed chain + mincut | 5.36/10.36 | 25.03/177 | 203/5684 |
| G16_chain_mc_w (GNN w + g/b, mincut) | 5.02/9.67 | 22.20/128 | 193/4059 |
| **B16_chain_mc (GNN w + learned p_v projection bias, mincut)** | **4.86/9.29** | **16.55/76.2** | **113/6341** |
| random | 128/256 | 2048/4096 | 32768/65536 |

Notes: G16_full_w beats QAOA on both median and B90 at every size (network helps when the cost is fully representable).
Bias head pv_acc vs a solution ~0.66–0.69; median at n=16 improves 203→113 vs fixed mincut, but B90 still worse than QAOA by ~60×.
Clause weights alone (no bias) only modestly beat fixed strengths.

## Milestone 3b — 32-level family fixed strengths (train n=10+15)
| version | n=10 | n=15 | n=20 |
|---|---|---|---|
| F32_qaoa (full) | 8.44/18.87 | 23.12/62.53 | 84.46/226.7 |
| F32_gray_chain | 8.48/20.25 | 116/1047 | 1678/119790 |
| random | — | 10923/32768 | 349525/1M |

Same pattern: 2-cavity sizes match QAOA; chain residual kills n≥15. Official n=20 QAOA L4 was 84.9/238.1 — F32_qaoa (trained 10+15) matches (84.5/227).

## Still running (20:03)
- mixcal NF80 K5 k32/k40; G16_local_w (bus-off). Bias 32-level family next.

## Bus-off (G16_local_w) — DONE
n=8 24.0/119, n=12 235/3143, n=16 2567/23032 (vs chain+GNN 35/302 / 408/13422). Bus contributes a lot.

## Ensemble of assignments (classical best-of) — BIG B90 win
B16_chain_mc checkpoint, ideal mixer, mincut + 8 random perms, keep best p per formula:
| | n=12 | n=16 |
|---|---|---|
| single mincut (B16) | 16.6/76.2 | 113/6341 |
| best-of-9 | **10.70/30.72** | **53.79/257.3** |
| QAOA L4 | 12.97/31.76 | 34.82/98.20 |

n=12 best-of-9 beats QAOA on median and B90. n=16 median 54 / B90 257 — past the ~500 B90 gate; calibrated-mixer ens eval launched.

## Random-assign train (G16_chain_rand)
n=8 ? / n=12 42.7/230 / n=16 519/10257 — worse than block (no bias). Assignment randomization alone without bias does not help.

## Running (20:12)
B32_chain_mc building n=15 mincut pool (~600/1500, shared with preperm); mixcal NF80 in sweeps; native ens eval of B16.
