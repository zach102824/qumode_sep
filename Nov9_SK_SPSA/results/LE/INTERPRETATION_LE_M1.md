# Experiment A (N9-LE) milestone 1: interpretation

**Status.** Development only (dev seeds 10000..10007; 8 instances per cell). Off-code physics follows the approved Nov9 simulation benchmark: D2 SNAP phase 0 above level 7, D3 Uz (U0 as a sensitivity arm), D4 per-term \(H^{\rm ext}\), D5 ideal logical readout. These are exactly the definitions frozen in `PREREG_LE_M1.json` before the approval, when they were still labeled "provisional". This is **not** a hardware-feasibility result. F is always unconditioned (no postselection). Optimization uses exact-energy SPSA only, with no shots. With 8 samples there is no target claim. Validation and final seeds were not touched.

Tables: `RESULTS_M1.md` (every cell, both caps), `summary_M1.json`. Plots: `medF_vs_L.png`, `leak_vs_NF.png`, `paired_F_scatter.png`. Per-run checkpoints: `runs/N{6,9,12}/*.json`.

## 1. Headline

Every Nov9th arm stays at the \(|+\rangle\) baseline. This holds for true D (Uz or U0), with and without \(s_3\), for the alternating, same-sign and frame starts, and for the strict-logical D8 diagnostic. Median \(\eta=(F-F_+)/(1-F_+)\) at cap 1024 lies in [−0.04, +0.002]. QAOA p = L reaches η = 0.75–0.99. Nov9th wins 0/8 against QAOA on F and on E in every cell. N=12 (cutoff 16 by the preregistered rule; 608 runs; cutoff deviation ≤ 4.2e-13) shows the same pattern.

| N, b = 1, cap 1024 | \(F_+\) | QAOA L1/L2/L3 | LE_alt L1/L2/L3 | B_alt (strict D8) L1/L2/L3 |
|---|---|---|---|---|
| 6 | .8995 | .9918/.9974/.9991 | .8994/.8988/.8979 | .8988/.8987/.8973 |
| 9 | .7363 | .9319/.9757/.9890 | .7367/.7359/.7341 | .7363/.7356/.7286 |
| 12 (cutoff 16) | .6673 | .9162/.9681/.9798 | .6672/.6668/.6657 | .6656/.6664/.6640 |

**Scope note (user, Nov 9 19:05 Shanghai).** New optimization experiments are now restricted to the maximum-GS-entanglement-entropy field \(B^*\). These b = 1/1.5 results are kept as a finished reference. No further b = 1/1.5 runs are scheduled. All 1824 LE runs finished before the scope change, so nothing was interrupted.

The b = 1.5 cells look the same (for example N9: QAOA .9875/.9968/.9983 against LE ≈ .89, with \(F_+\) = .8911). At cap 128 the Nov9th arms sit slightly *below* \(F_+\) (η −0.02 to −0.76), because unconverged SPSA moves off \(|+\rangle\). QAOA is already at η ≈ 0.75–0.95 at cap 128. The cost curve is therefore flat for Nov9th and nearly saturated for QAOA by 128 queries.

## 2. Leakage is not the limiter

| check | result |
|---|---|
| strict D8 vs true D (K2 falsifier) | B_alt is equally null, so **K2 is triggered**: the limit is the ansatz or landscape, not leakage. K1 is not triggered. |
| median \(p_{\rm leak}\) at cap 1024 | 1e-5 to 1e-3. The maximum is 1.4e-2 (LE_same, L3, cap 128). |
| cutoff ladder 16/20/24/32/48 | max \(|\Delta E|\), \(|\Delta F|\), \(|\Delta p_{\rm leak}|\) against NF = 48 ≤ 1.7e-13 for every NF ≥ 16 (N6, N9; N12 in RESULTS_M1.md). Tolerance 1e-8. |
| Uz vs U0 at identical parameters | median \(|\Delta F|\) ≤ 1.7e-6. Not extension-dependent. |
| \(s_3\) off (K4) | no worse than LE_alt. Moot, because there is no gain to lose. |
| alternating vs same-sign start (K3) | LE_alt stays alternating (16/16 at L2). LE_same is slightly worse at L3, with more leakage. Moot for the same reason. |

## 3. Why: diagnosis of the \(|+\rangle\) basin and mixing (diagnostics only, no optimization)

This is not a vanishing-gradient claim. FT's central differences at its cold start (γ = 0.25, α ≠ 0) show slopes of several units, and the downhill direction leads back to the product state. That agrees with what follows: \(|+\rangle\) is a positive-curvature local minimum (plus a small linear α term of 0.04–0.32 and zero modes), and the cold start lies in its basin. QAOA's \(|+\rangle\) is a saddle instead.

The scripts are `hessian_at_plus.py`, `curvature_escape.py` and `mixing_diag.py`. Results are in the matching JSON files in this directory.

1. **Exact conjugation symmetry.** With all SNAP phases s = 0, every Nov9th gate is real (\(|+\rangle\), real D, real cost matrix) except \(e^{-i\gamma A}\). So \(\psi(-\gamma)=\psi(\gamma)^*\), and \(E(\gamma)=E(-\gamma)\) for **any** real α, any θ, any L, true or strict D. Checked numerically: max \(|E(\gamma)-E(-\gamma)|\) = 0.0 exactly over 192 random s = 0 points; the median is 0.18 at random s ≠ 0. Consequences:
   - \(\partial_\gamma E=0\) and \(\partial_\gamma\partial_\alpha E=0\) on the whole s = 0 plane.
   - From a cold start (s ≈ 0), the cost couples to the mixer only at third order (γ·s·α).
   - The argument does not depend on the encoding, the grouping, the K/M features or the off-code extension. It therefore also covers the GR (encoding/assignment) and FT (feature/prior) nulls, not just LE.
2. **Mixing curvature.** In the (γ, α) block at \(|+\rangle\), a cost-coupled escape needs \(H_{\gamma\alpha}^2>H_{\gamma\gamma}H_{\alpha\alpha}\). Medians over 8 seeds and both b, strict D8:

   | point | \(H_{\gamma\gamma}\) | \(H_{\gamma\alpha}\) | \(H_{\alpha\alpha}\) | 2×2 min eig |
   |---|---|---|---|---|
   | QAOA L1, N9 | 36.7 | 29.1 | **0** | −16.2 |
   | real α L1 at \(|+\rangle\), N9 | 36.7 | **0** | 25.9 | +25.9 |
   | real α L2 frame point (\(s_{1,1}=\pi/2\), \(s_{2,1}=-\pi/2\)), N9 | 146.8 | −4.5 | 60.0 | +50.9 |
   | imaginary-axis \(D(i\alpha)\) L1, N9 (**not authorized**) | 36.7 | −40.4 | 35.7 | −5.9 |

   \(|+\rangle\) is an eigenstate of QAOA's mixer (\(H_{\alpha\alpha}=0\)), so any cross term descends. It is also invariant under Nov3rd's \(WSW\) for every S, which is the same mechanism and explains why Nov3rd escapes. A displacement moves \(|+\rangle\) and raises the energy (\(H_{\alpha\alpha}\) ≈ 17–60), so it needs a cross term several times larger than \(H_{\gamma\gamma}\) and \(H_{\alpha\alpha}\) allow. The π/2 frame start makes 5 of the 7 Fock edges imaginary (edges 1–2 and 5–6 have Δpopcount = 0 and cannot be phased by a popcount SNAP), but its cross term is about 10× too small.
3. **Full Hessian at the start** (`curvature_escape.json`, medians):
   - real α at \(|+\rangle\): min eigenvalue 0.000;
   - frame point: −0.006 to −0.27, but the eigenvector's γ component is about 0.003, and the value is unchanged when γ is frozen. It is a cavity-local (product-state) direction, not an interacting escape;
   - imaginary α: −0.7 to −5.3, cost-coupled (γ component 0.1–0.36; at L=1 the curvature turns non-negative when γ is frozen);
   - QAOA: −7.6 to −41.
4. **Sampling.** The best of 20000 uniform samples in the bounded box is always above \(E_+=-bN\) (`landscape_audit.json`, N6).

**What this does and does not show.** It shows that \(|+\rangle\) is an attracting, positive-curvature point for real-α Nov9th, for reasons that are exact and leakage-independent. Cold starts near it descend back into it. That is enough to explain the three-stream null (LE, FT, GR) from cold starts. It does **not** prove an expressivity ceiling. A good basin farther out (large |s|, or α beyond D9's 0.30) is not excluded. Uniform sampling in 7–13 dimensions is weak evidence. FT's warm-transfer result (F ≈ .26, worse than cold) is consistent with there being no instance-agnostic basin near the start. It is not evidence of a ceiling either.

## 4. Pre-registered addendum (M1b), written after seeing M1 and before fitting

`PREREG_LE_M1b.json`:

- **LE_frame** init variant: the same family, with the π/2 frame offset on s. It is null (median η −0.011 to −0.039 at cap 1024 for N6/N9) and slightly worse at cap 128.
- **N12** at cutoff 16, by the preregistered rule. The N=9 deviation at cutoff 16 is 9.4e-14.

## 5. Accounting, bugs, compute

- **Queries.** Every run used exactly 1024 or 128 counted energy queries: calibration plus 2K SPSA plus 2 candidates per start, 4 starts. Selection uses counted candidates only. Post-selection diagnostics (F, cutoff ladder, U0 cross-evaluation) use the exact GS and are evaluation-only. They are never used for selection and are not counted. There are no uncounted initial energy calls; the gap GR reported does not occur here.
- **Readout.** Exact-energy mode. The D5 ideal readout and the unmodelled \(W_L\) cost were not exercised; in shot mode each X-setting shot costs C ideal \(W_L\). Shot mode was not run.
- **Bug fixed before any reported number.** `Path.with_suffix` on "…_b1.5" made the b = 1.5 instance files collide with b = 1 (wrong GS, F and E0). The optimizations themselves had the correct energy. The fix uses explicit `_b{10b:02d}` names, checks b on load and asserts \(E_0\le-bN\). All runs and instances were wiped and rerun. The prereg is unchanged.
- **Throughput incident.** M1b was first launched without `OMP_NUM_THREADS=1`. BLAS thread oversubscription slowed it about 100×. The job was restarted single-threaded and resumed from checkpoints; results are unaffected, because the runs are deterministic.
- **Norm.** D is built in nf + 32 levels and then restricted, so no contraction occurs at the cutoff. Max \(|\lVert\psi\rVert-1|\) at the fit cutoff is 3.4e-15 (N12, cutoff 16), and 1.2e-14 over the whole ladder. There are no aborts and no renormalization, unlike FT's original NF16 restriction.
- Failures: 0. Nothing has been left running in the background.
