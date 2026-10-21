# Templated native cavity-chain ansatz (Zach plan 2026-10-19) — running summary

> **STRICT-CONTROL RULE (2026-10-20 14:30 UTC+8, Zach):** a result counts only if the training size is strictly smaller than the validation/test size (16-level: train n=8 -> val n=12 -> test n=16; 32-level: train n=10 -> val n=15 -> test n=20). Every earlier number from training on n=10+15 (or n=8+12) with validation/early stopping on n=15 (or n=12) is **not strictly controlled (train/val overlap)**. That includes F32_*, B32_*, B20_*, G16_*, B16_*, the RR4/RR8 runs, and the OpenEvolve net_smoke/net_long runs (which trained on n=10+15). Treat those as indicative only.


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

## Calibrated mixer + ensemble (20:18 CST) — native ≈ ideal
B16_chain_mc, k=16 SNAP+D pairs, mean betas, ensR8:
n=12 **10.49/30.86** (ideal ens 10.70/30.72), n=16 **54.60/260.45** (ideal 53.79/257).
Mixer F ≥ 0.9988. Native transfer confirmed at 16 levels.

## B32_chain_mc in training (20:25)
pool n=15 built (12 min mincut). step 750 val n=15 44.4/253 (from 197/5131); pv rising 0.19→0.50.
ETA ~20:40 for 3k steps + test n=10/15/20.

## B32_chain_mc DONE (20:38–20:48 CST) vs F32_qaoa
| version | n=10 | n=15 | n=20 |
|---|---|---|---|
| F32_qaoa | 8.44/18.87 | 23.12/62.53 | **84.46/226.7** |
| F32 fixed chain | 8.48/20.25 | 116/1047 | 1678/119790 |
| B32 single mincut+bias | **8.07/17.29** | 35.58/302 | 330/9835 |
| B32 best-of-9 ens | — | **20.10/80.93** | **110.6/764** |

n=15 ens beats QAOA median (20.1 < 23.1), B90 close (81 vs 63).
n=20 ens median 111 (~1.3× QAOA), B90 764 (~3.4×). vs fixed chain: 15× median, 157× B90 improvement.
Native 32-level ens pending NF80. R=16 ens queued for B90.

## Ensemble R scaling (21:05 CST)
| | n=20 median/B90 |
|---|---|
| QAOA L4 | 84.5/226.7 |
| B32 single | 330/9835 |
| best-of-9 (R=8) | 110.6/764 |
| best-of-17 (R=16) | **107.0/626** |

More random perms: small median gain, B90 764→626. Diminishing returns — need smarter search / better model.
NF80 k=32: Fmin **0.9909** (F@NF+40 same), leakmax 3.5e-3. Native ens eval launched.

## R=16 ensemble (21:05 CST)
| | n=16 (B16) | n=20 (B32) |
|---|---|---|
| QAOA L4 | 34.8/98.2 | 84.5/226.7 |
| ens R=8 | 53.8/257 | 110.6/764 |
| ens R=16 | **37.9/249** | **107.0/626** |

n=16 median nearly matches QAOA (37.9 vs 34.8); B90 still 2.5×. Random ens saturates — SA variants + B90-retrain + native k32 queued.
NF80 k=32 Fmin 0.9909 (usable). k=40 still sweeping.

## NF80 complete
| k | Fmin (grid) | F@NF+40 | leakmax |
|---|---|---|---|
| 32 | 0.9909 | 0.9909 | 3.5e-3 |
| 40 | **0.9983** | 0.9982 | 1.3e-3 |

Per-beta fit of B32 strengths at k=32: F 0.995–0.996 (better than worst-grid). Native ens still running.
## Still running (21:17)
- nav_B32_ens8 (calibrated k=32, n=20 best-of-9)
- ens_B32_SA (8 SA restarts + 4 random)
- B32_b90 retrain queued (tailw=2, ES on B90)

## Native + SA + B90-retrain (21:49–22:02 CST)
| version | n=20 med/B90 |
|---|---|
| QAOA L4 | **84.5/226.7** |
| B32 single | 330/9835 |
| B32_b90 single (tailw=2, ES on B90) | 309/8034 |
| ens R8 (B32) | 110.6/764 |
| ens R16 (B32) | 107.0/626 |
| ens SA8+4rand | 105.5/855 (B90 worse) |
| **ens R16 (B32_b90)** | **95.5/645** |
| native ens R8 k=32 (F~0.995) | 120.1/849 |

Residual vs difficulty: corr(log 1/p, resid_mincut)=0.21; hard-tail mean resid 0.297 vs easy 0.283.
=> B90 gap is NOT mostly "unrepresentable clauses" — software headroom remains (bias/schedule/depth/assignment).
Next: train at n=20 (31e9 pool), R=32, L8.

### 2026-10-19 22:55 (UTC+8): best-of-33 at n=20 and a fairness caveat
- B32_b90 assignment ensemble at n=20: best-of-17 95.5/645, **best-of-33 70.3/443** (median/B90 of 1/p). QAOA L4 is 84.5/227. Median now beats it; B90 is still 2x short.
- **Caveat:** "best-of-R" uses the largest p over the assignments, and you can only know which assignment has the largest p if you already know the solution. So it is an oracle number. The fair cost is round-robin over all R assignments, which stops at the first verified solution: 1/mean_r(p_r). The `mean` column gives this: R33 **337/3393** at n=20. A fair ensemble needs a formula-only selector.
- New: `ensemble_eval2.py` saves per-assignment p plus formula-only selector features (uniform and bias-weighted projection residual). Smoke test at n=15: these features are only weakly correlated with p.
- B32_resw (`--resw 1`, residual-weighted loss, + tail 2, ES b90): single-assignment n=15 35.6/323 (B32_b90: 39.1/251); n=20 309/7545 (B32_b90: 309/8034). The median improves but the tail does not.

## FAIR-METRIC RULE (2026-10-19 23:00 UTC+8). Read this before any number above
Best-of-R over assignments is an **oracle**: to pick the assignment with the largest p, you have to know the solution. It is **not** a valid comparison with QAOA. From now on only these count:
(a) a single assignment (min-cut); (b) round-robin over R assignments, stopping at the first verified solution, giving 1/mean_r(p_r); (c) the assignment picked by a formula-only selector.
Every "ens"/"best-of" figure earlier in this file is an oracle number and does NOT beat QAOA.

Fair numbers (median/B90 of 1/p; QAOA L4: n=12 13.0/31.8, n=15 23.1/62.5, n=16 34.8/98, n=20 84.5/227):
| model | n | single min-cut | round-robin R=8 | round-robin R=16 | round-robin R=32 | (oracle best-of, for reference) |
|---|---|---|---|---|---|---|
| B16_chain_mc | 12 | 16.6/76 | 19.9/86 | | | 10.7/31 (R8) |
| B16_chain_mc | 16 | 113/6341 | 139/1304 | 125/1603 | | 38/249 (R16) |
| B32_chain_mc | 15 | 35.6/302 | 43.5/341 | | | 20.1/81 (R8) |
| B32_chain_mc | 20 | 330/9835 | 363/3190 | 380/3163 | | 107/626 (R16) |
| B32_b90 | 15 | 39.1/251 | | 40.1/248 | | 17.9/67 (R16) |
| B32_b90 | 20 | 309/8034 | | 363/2969 | 337/3393 | 70/443 (R32) |
| G16_full_w (all-to-all, NOT native) | 16 | 31.2/74 | | | | |
Conclusion: on the fair metric the native chain is still about 3x behind QAOA on the median and 15-60x behind on B90 at n=16 and n=20. Round-robin swaps median for tail and does not close the gap.

### Selector test, n=15 (B32_b90, min-cut + 8 random assignments; fair)
single min-cut 39.1/251; round-robin over all 43.8/285; oracle 20.8/78; QAOA 23.1/62.5.
- uniform-residual selector: Spearman correlation with -log p inside each formula is 0.18; top-1 39.1/251, top-2 round-robin 39.4/223.
- bias-weighted-residual selector: correlation 0.30; top-1 35.4/268, **top-2 round-robin 32.2/190** (best fair number at n=15).
- Even a perfect selector (the oracle, 20.8/78) loses to QAOA on B90 at n=15. Selection alone cannot win; single-assignment quality has to improve.

### 2026-10-19 23:50 UTC+8: B16_Hb_resw, pairs upper bound, L8, residual diagnosis
Fair single-assignment only:
| model | n=12 | n=16 | notes |
|---|---|---|---|
| B16_chain_mc | 16.6/76 | 113/6341 | prior best chain |
| **B16_Hb_resw** (Hb=256, resw=1, tail=2) | **18.2/83** | **121/4487** | B90 ~1.4× better; median slightly worse. Far from 10×. |
| **B16_pairs** (proj=pairs, any 2 cavities) | **16.1/51** | **55.2/376** | ~12× better B90 vs chain; still ~1.6×/3.8× behind QAOA 34.8/98 |
| QAOA L4 | 13.0/32 | 34.8/98 | |

Residual variance (block assign, median over 100): n=16 chain 0.41, pairs 0.17, full 0; n=20 chain 0.40, pairs 0.16.
Software escape hatch closed: L=1..8 of (chain-diag + bit-flip mixers) cannot match a 3-cavity phase on |+> (probe loss stuck at idle 0.708). Deeper L does not create non-neighbor diagonal support without new connectivity.
B20_train2 still running (val B90 stuck ~6k). ens2 n=20 ~80/100. B16_L8 training.

### 2026-10-20 00:20 UTC+8: landed (fair only)
| model | n=12 | n=15 | n=16 | n=20 |
|---|---|---|---|---|
| B16_Hb_resw | 18.2/83 | | 121/4487 | |
| B16_L8 (L8, Hb128, resw, tail2) | 13.7/83 | | 95.2/3932 | |
| B16_pairs (any-2-cavity projection) | 16.1/51 | | 55.2/376 | |
| B20_train2 (trained on n=15+20) | | 38.0/291 | | 312/8049 |
| B32_b90 single (ref) | | 39.1/251 | | 309/8034 |
| selector top-2 round-robin (bias-resid), B32_b90 | | 30.7/251 | | 290/4455 |
| QAOA L4 | 13.0/32 | 23.1/62.5 | 34.8/98 | 84.5/227 |
Selector at n=20: Spearman correlation 0.22 (bias residual) and 0.15 (uniform). It confirms the n=15 result: weak, and capped by the oracle (95.5/645 > 227).
Nothing chain-native moved B90 by 10×. L8 helps ~1.15×; training directly at n=20 helps 0×. Only the connectivity change (pairs) gave ~12× at n=16.
Running: B32_pairs and B32_full (n=10+15 train, test 15/20) for n=20 connectivity numbers.

### n=20 connectivity ladder (fair, single min-cut assignment; same training recipe)
- chain (native today): n=10 8.3/18; n=15 39.1/251; n=20 309.0/8034
- any 2 cavities: n=15 25.3/115; n=20 115.7/810
- full / multi-cavity phases: n=15 20.2/57; n=20 77.3/195
- QAOA L4: n=15 23.1/62.5; n=20 84.5/227

**Recommendation:** software paths are used up on the chain (selector, residual/tail loss, wider bias, L8, training at n=20, theory check that depth cannot synthesize 3-cavity phases). Beating QAOA fairly at n=20 needs applying cost phases across 3+ cavities (the full-projection model beats QAOA: 77/195 vs 84.5/227). Any-pair connectivity alone gets about 10× on B90 but still loses (116/810).

## 2026-10-20 13:40 UTC+8: gate lock, round-robin training (idea #1), OpenEvolve on the network only
- **Gate lock:** `v3/openevolve/net/frozen_hashes.json` holds SHA-256 hashes of compiler.py, nsim.py, calib.py, assign.py, sim.py, frozen_circuit.py and evaluator_net.py. They are checked before and after every candidate. Candidates may only import numpy/jax/math/etc. and get a restricted API (data / batch / differentiable p). Output shapes are checked: gamma, beta (L), w (m, L), pv (n). `frozen_circuit.py` reproduces gnn_bias.forward and the numpy evaluator exactly.
- **Idea #1 (gnn_bias `--rr R`, loss -log mean_r p_r + tail term), fair test numbers:**
  - B32_RR4_t2 (early stopping on B90; it chose step 0, so this equals B32_b90): n=15 single 39.1/251, rr4 37.7/228; n=20 single 309/8034, rr4 303/3916.
  - B32_RR4_S (early stopping on S): n=15 single 37.4/247, **rr4 36.6/188**; n=20 pending.
  - QAOA L4: n=15 23.1/62.5, n=20 84.5/227.
- **OpenEvolve network smoke (10 iterations, composer-2.5 + gpt-5.3-codex):** 5/10 candidates valid. Best is still the initial net: held-out n=15 rr4 49.3/225, single 39.9/283. Failures were NaN outputs and np-on-tracer; the prompt now warns about both. A 120-iteration long run has been launched (runs/net_long) with an n=20 check of the best program every 10 iterations (n20_watch.sh).


## 2026-10-20 14:30 UTC+8: strictly controlled restart
> **STRICT-CONTROL RULE (2026-10-20 14:30 UTC+8, Zach):** a result counts only if the training size is strictly smaller than the validation/test size (16-level: train n=8 -> val n=12 -> test n=16; 32-level: train n=10 -> val n=15 -> test n=20). Every earlier number from training on n=10+15 (or n=8+12) with validation/early stopping on n=15 (or n=12) is **not strictly controlled (train/val overlap)**. That includes F32_*, B32_*, B20_*, G16_*, B16_*, the RR4/RR8 runs, and the OpenEvolve net_smoke/net_long runs (which trained on n=10+15). Treat those as indicative only.

Launched: OpenEvolve net16_n8 (train 8, val 12 with rr4 over 100 formulas, check 16 every 10 iterations, cores 5-6) and net32_n10 (train 10, val 15, check 20, cores 3-4). Both are seeded from the initial network; the gamma/beta init comes from the *other* family's fixed schedule so no tested size was ever seen. Hand runs: A16_n8_RR4 (train 8, early stopping on n=12, test 12/16) and B32_n10_RR4 (train 10, early stopping on n=15, test 15/20). B32_RR8 was killed (not strictly controlled).

## 2026-10-20 14:35 UTC+8: SPLIT REVISED (Zach). This supersedes the 14:30 rule
16-level: train n=8+12, select/test ONLY on held-out n=16. 32-level: train n=10+15, select/test ONLY on held-out n=20. No separate validation size.
Model/candidate selection uses a fixed held-out SELECTION set (seed base 33e9). Final numbers come from a DIFFERENT held-out TEST set (base 32e9, the standard 100 formulas).
Earlier runs that early-stopped on n=15 / n=12 (a training size) are **not strictly controlled (train/val overlap)**.
Launched: OpenEvolve net16 (selection 60 n=16 formulas, rr4; test-set check every 10 iterations; cores 5-6) and net32 (selection 30 n=20 formulas, rr4; cores 3-4). Hand runs: A16_RR4 (train 8+12, early stopping on 60 n=16 selection formulas, test n=16) and B32_RR4 (train 10+15, early stopping on 30 n=20 selection formulas, test n=20).

### Core budget (Zach 2026-10-20 14:45 UTC+8): calculation jobs capped at **6 cores** total.
Map: A16_RR4 cores 0-1; B32_RR4 core 2; OpenEvolve net32 cores 3-4; OpenEvolve net16 core 5. Adapter is idle-wait only.
Measured OE iteration times so far: net16 ~3–5.5 min/candidate (selection 60×n=16 rr4); net32 first candidates OOM at 3.5GB — mem cap raised to 4.5GB.

### First fair held-out n=16 (A16_RR4 hand, train 8+12, early-stop on n=16 selection base 33e9, TEST base 32e9)
- single min-cut: **103.1 / 7363**
- fair round-robin R=4: **130.3 / 2647**
- QAOA L4 n=16: **34.8 / 98**
- Selection (60 formulas) looked much better (~88/736 at best S), so B90 on the official test is still dominated by a hard tail. Not close to QAOA.
- Core map (cap 8): B32_RR4 0-2; OE net32 3-5; OE net16 6-7. OE parents are parent+worker trees, not duplicates.

### Memory policy (2026-10-20 15:15 UTC+8)
The box has 16 GB, and about 8 GB is held by other services, so my jobs get about 5 GB with at least 3.2 GB (20%) kept free.
- Every hand run uses `gq_bias.sh` with an RSS watchdog (MEMMB; it kills only that job). OpenEvolve children have their own RSS cap (OE_MEM_MB).
- Paused OpenEvolve net32 (n=20 children peak about 4–5 GB) until RAM frees up. It restarts with a lean config after B32_RR4 finishes.
- Running: B32_RR4 (cores 0-1, about 2.8 GB), A16_full teacher (core 2, about 0.6 GB), OpenEvolve net16 (cores 6-7, child about 1 GB).
- Implemented #4 distillation in gnn_bias: `--teacher <full-proj json> --kd λ --kd_mode dist|marg`. Loss = -log p (+ tail term) + λ·KL(teacher readout ‖ student readout). The teacher is A16_full (proj=full, train 8+12, selection on n=16).

### Strict-split fair held-out TEST results (train 8+12 → test 16; train 10+15 → test 20; separate selection set)
| run | idea | n | single | round-robin R=4 | QAOA L4 |
|---|---|---|---|---|---|
| A16_RR4 | #1 round-robin loss | 16 | 103.1/7363 | 130.3/2647 | 34.8/98.2 |
| A16_full (teacher, global cost, NOT native) | — | 16 | 33.8/79.1 | — | 34.8/98.2 |
| KD16_dist | #4 distill (KL readout, λ=1, init=teacher) | 16 | 140.9/4436 | — | 34.8/98.2 |
| B32_RR4 | #1 round-robin loss | 20 | 314.3/11260 | 264.3/4639 | 84.5/226.7 |
Notes: KD16_dist trades median for B90 (4436 vs 7363) but is still ~45× short of QAOA on B90. KD16_marg is running. Other services hold ~8 GB of RAM, so I launch only when ≥3.2 GB is free.

### Oct 20 16:20 (UTC+8) strict-split TEST n=16 (train 8+12; QAOA L4 34.82/98.20)
| Run | Idea | single med/B90 | rr4 med/B90 |
|---|---|---|---|
| KD16_marg | #4 KD on per-bit marginals | 131.98/4371.6 | – |
| COR16 | #3 SNAP correction (mean-embedding table) | 96.68/7468.5 | 117.35/3052.3 |
Goal now: make the chain-applied cost act like the global cost (global teacher 33.8/79). OpenEvolve net16c: optional additive
dS (SNAP) / dB (neighbour-bus) tables, same gate sequence; teacher via api.p_from(glob=True). Hand run CB16: --corr2 SNAP+bus pattern tables.
| TR16 | #5 two-round feedback, 200 shots split | 89.38/5540.1 | 87.34/2851.9 |
| CB16 | SNAP+bus signed-pattern correction (--corr2 0.01) | 87.24/7328.2 | 108.81/4986.1 |
OpenEvolve: from 16:59 (UTC+8) LLM = claude-opus-5-5-high only (run net16o, phase-correction goal; net16c stopped, DB kept).
| JC16 | SNAP+bus tables trained jointly from F16_gray_chain_mc angles, tailw 1, global-p aux 0.5 | 102.28/9965.4 | 119.06/4779.7 |
| TC16 | #5 two-round + clamp 3 most-confident vars, 200 shots | 88.09/5294.1 | 85.15/3130.8 |
QAOA L4 refs: F16_qaoa fitted on n=8+12, F32_qaoa on n=10+15 (same train sizes as the networks).
| FP16 | free SNAP+bus tables REPLACE projected cost (L2 0.01), from F16_gray_chain_mc angles | 4911.9/48821 | 4841.9/49588 |
OpenEvolve net16o ckpt_10 OFFICIAL TEST n=16 (TEB): single 111.7/7991 | rr4 113.7/3685 (selection was 85/1024).
| AH16 | formula-only assign_head selector over R=4 (warm key-wise from A16_RR4) | 99.01/6645.6 | 134.34/2614.0 | sel 99.01/6645.6 |
OpenEvolve net16o official TEST n=16 (TEB): ckpt_10 single 111.7/7991 rr 113.7/3685; ckpt_20 single 111.0/8166 rr 111.3/3957.
| HY16 | hybrid: projected cost + residual SNAP/bus on compiler support (non-zero init, L2 0.01), warm A16_RR4 | 108.27/7153 | 107.22/4196 |
OpenEvolve net32o official TEST n=20 (chunked checker, verified = frozen score_set to 7e-7; QAOA L4 84.46/226.7):
ckpt_10 single 284.2/8939 | rr4 251.3/4930 ; ckpt_20 single 282.1/8807 | rr4 241.9/3414 (selection n=15 rr 27.6/104.2).
net16r ckpt_10 (2026-10-21 13:56 UTC+8) fair TEST n=16 rr 83.1/4943.9 single 78.5/12676.2 pv_acc 0.72 | selection 61.0/490.2 | stall 0 (QAOA L4 34.82/98.20)
net16r ckpt_20 (2026-10-21 15:02 UTC+8) fair TEST n=16 rr 94.3/6215.7 single 80.9/15707.0 pv_acc 0.72 | selection 53.9/543.4 | stall 1 (QAOA L4 34.82/98.20)

## net16r v2 (native framing), from 2026-10-21 15:10 UTC+8
Framing (Zach, FRAMING_NATIVE.md): do NOT imitate QAOA's global cost; use the hardware-native control space (arbitrary per-Fock-level SNAP dS / neighbour-bus dB phases on multi-bit cavities; only the final bit string counts).
net16r v1 (archived runs/net16r_v1, 20 iterations, 5 islands loss/architecture/features/curriculum/placement): fair TEST n=16 ckpt_10 rr 83.1/4943.9, ckpt_20 rr 94.3/6215.7 (selection 61.0/490.2 and 53.9/543.4).
v2 islands = Opus path families (OPUS_PATHS_1500.md), starting directions only (Opus may add/combine/retire families; wildcard every 3rd idea; Opus picks new families at stagnation):
1 conditional pair messages, 2 clause-local optimistic tables, 3 rank level shaping, 4 role-specialised layers, 5 cavity-local Grover (capped candidate set); hardness add-on shared.
Seeds = v1 ckpt_20 best + zero-initialised native-family gates (start at parent function); loss 0.5 mean + 0.5 median + CVaR-10% of log(1/p); LAM_GLOB 0.
Stage 1 s1 = -(ln med + 2 ln B90)/3 on 30 held-out n=12; static guard rejects full-assignment enumeration / argmin / >1 pair table at once; pv_acc>0.85 -> -20 kept.
net16r ckpt_10 (2026-10-21 17:02 UTC+8) fair TEST n=16 rr 83.9/2607.0 single 87.1/8869.9 pv_acc 0.72 | selection 60.9/511.4 | stall 0 (QAOA L4 34.82/98.20)
net16r ckpt_20 (2026-10-21 17:54 UTC+8) fair TEST n=16 rr 93.7/3128.8 single 102.6/9069.6 pv_acc 0.72 | selection 60.9/463.7 | stall 1 (QAOA L4 34.82/98.20)
net16r ckpt_50 (2026-10-21 18:59 UTC+8) fair TEST n=16 rr 93.4/4045.5 single 88.1/10691.4 pv_acc 0.72 | selection 60.9/463.7 | stall 2 (QAOA L4 34.82/98.20)
