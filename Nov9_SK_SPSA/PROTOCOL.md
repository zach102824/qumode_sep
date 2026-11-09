# Nov9_SK_SPSA shared protocol (draft v0, consultation, not frozen)

Status: consultation draft written 2026-11-09. No experiment in this folder has been run by the consultation stream. Numbers quoted from the local 8-core stream (`/workspace/Nov9_SK_SPSA_local`) are as reported by that stream and were not reproduced here. The final panel (seeds 20000..20999) has not been touched by anyone.

Names: **Nov3rd** = original circuit. **Nov9th** = new primary displacement-middle circuit. **QAOA** = standard p = L baseline.

---

## 1. Target

\[
H=\sum_{i<j}J_{ij}Z_iZ_j-b\sum_iX_i,\qquad J_{ij}\sim\mathcal N(0,1/N)\ \text{i.i.d. for } i<j,\quad b\in\{1,1.5\}.
\]

- Qubit convention (same as `Nov_8th/spin_glass_r1/run_sg.py` on PR 72 branch, read only, `7b2444f0`): site \(i\) is bit \(i\) of the integer index, LSB = site 0, \(Z|0\rangle=+|0\rangle\).
- \(J\) generation (compatible with the local stream): `rng = np.random.default_rng(seed)`; `u = rng.normal(0.0, 1.0/np.sqrt(N), size=N*(N-1)//2)`; fill \(J_{ij}=J_{ji}=u_t\) in row-major order \((0,1),(0,2),\dots,(N-2,N-1)\); diagonal 0. The same \(J\) is used for both \(b\) values and for every method.
- Seed panels per N: **dev** 10000..10031, **validation** 11000..11031, **final** 20000..20999 (blind until a recipe is frozen by commit hash). Local initial bounded slice: N=6, dev 10000..10007.
- Ground truth (evaluation only, never visible to the optimizer): exact lowest eigenpair by dense `eigh` (N ≤ 12) or Lanczos with residual \(\|H g-E_0 g\|\le 10^{-10}\) (N = 15, 18). Store \(E_0\), \(E_1\), the lowest energy in the odd global-flip sector \(E_0^{\rm odd}\), \(\lambda_{\max}\) (N ≤ 12), the GS vector (npz for N ≤ 12, sha256 + regeneration recipe above that), and `J_upper` with its float64-LE sha256.
- Perron–Frobenius: for \(b>0\) the GS is unique, strictly positive in the Z basis, and even under \(\Pi=\prod_iX_i\). Store \(F_+=|\langle g|+\rangle^{\otimes N}|^2\) for every instance as the trivial baseline.

## 2. Circuits (exact definitions)

### 2.1 Common cost unitary (shared across layers)

\[
U_{cZ}=\exp\Big[-i\sum_{i<j}A_{ij}Z_iZ_j\Big],\quad A=\gamma\Big[J+\sum_f\theta_f\hat F_f\Big].
\]

One \(\gamma\), one \(\theta_f\) per feature, all shared by every layer. Feature hat: \(\hat F=F\cdot\mathrm{RMS}(\mathrm{upper}\,J)/\mathrm{RMS}(\mathrm{upper}\,F)\) (`hat()` in `Nov_3rd/tfim_shared_gamma/round4_N/model.py`, PR 72 branch). Baseline features:

- \(K=\mathrm{sym}(\mathrm{offdiag}(J@J))\) (`pair_K`). Identity: \(\sum_ih_i(z)^2=\mathrm{const}+2\sum_{j<k}K_{jk}z_jz_k\), where \(h_i=\sum_jJ_{ij}z_j\). So K is the squared-local-field operator.
- \(M\) = every inter-cavity block of \(J\) replaced by its mean, with intra-cavity entries set to 0 (`block_coeff(J, grouping)`). This is **not** the paper's \(M_{nm}\) in \(V(M)\). To avoid the name clash, the paper's matrix is called \(M^{\rm GCS}\) below.

### 2.2 Nov9th (primary, gate order is fixed)

\[
|+\rangle^{\otimes N}\to\big[U_{cZ}\to D(\alpha_\ell)^{\otimes C}\to S_\ell^{\otimes C}\big]_{\ell=1}^{L}
\]

- \(D(\alpha)=\exp(\alpha a^\dagger-\alpha^*a)\) with **real** \(\alpha_\ell\), one per layer, the same on every cavity. It is the true oscillator displacement. It must be built as the matrix exponential in a large Fock space (NF_big ≥ NF_sim + 32) and then restricted, or directly in NF_sim with a convergence proof (§4). An exponential of the 8×8 cropped \(a\) is a different gate (§2.5).
- Bare shellSNAP per cavity of 3 logical qubits: \(S_\ell=\exp[-i(s_{1\ell}\sum n_r+s_{2\ell}\sum_{r<r'}n_rn_{r'}+s_{3\ell}n_1n_2n_3)]\), where \(n_r\) is logical bit \(r\) of the cavity. It is shared across cavities. On popcount \(p\), the phase is \(s_1p+s_2\binom p2+s_3[p=3]\). This is an invertible linear reparametrization of the 3 Walsh shell controls `shell_weights` \(t_1,t_2,t_3\) used in Nov 8 (conformance test T6).
- The first \(U_{cZ}\) acts on \(|+\rangle\). There is no prep gate and no final basis change.
- Control count: \(1+n_\theta+L+3L\). With the KM pair this is \(3+4L\), so 7, 11, 15 for L = 1, 2, 3.

### 2.3 Nov3rd (secondary baseline, unchanged)

\(|0\rangle\to[U_c^X\to S_\ell]^L\to W\), with \(U_c^X=W\,e^{-i\gamma E}W\) and \(W=H^{\otimes3}\) on Fock 0..7 (`Nov_3rd/ansatz.py` L99–150, `build_W_BinH` in `explore_OCT29/scripts/binH_vs_grayH.py` L19–26). It must satisfy the exact rewrite \(|+\rangle\to[U_{cZ}\to WSW]^L\) (local stream reports a difference below 3e-16). Nov3rd lives in a strictly 8-level model: `NF = 8` in `Nov_3rd/settings.py` L29, so it has no leakage by construction. Control count is \(3+3L\) with KM.

### 2.4 QAOA (fair baseline)

\(|+\rangle\to[e^{-i\gamma_\ell H_{ZZ}}\to e^{-i\beta_\ell\sum X}]^{p}\), with p = L and 2L independent angles. It gets the same SPSA, the same budget, the same number of hyperparameter configurations tuned on dev, and the same transfer opportunity when an idea uses transfer. It uses the plain qubit model (no leakage). Equal L does not mean equal compiled cost; see the cost table in §7.

### 2.5 Hardware models (keep them separate in every table)

| label | displacement | off-code space | where used |
|---|---|---|---|
| **leaky** (primary) | true \(D(\alpha)\) on the oscillator | levels ≥ 8 exist; §3 extension | Nov9th headline |
| **strict-logical** "Nov9th-B" | \(D_8(\alpha)=\exp[\alpha(Pa^\dagger P-PaP)]\), \(P\) = projector on levels 0..7 | none | diagnostic only. It is physical only for an 8-level qudit or a blockaded cavity. It must never be called a displacement. |
| qubit | none | none | QAOA |
| 8-level ideal | ideal \(W\) | none | Nov3rd |

## 3. Logical-to-Fock encoding, off-code extension, energy observation

### 3.1 Encoding (source-based)

- **Binary Fock j = 0..7** is the Nov3rd encoding. Bit r of Fock level f is `(f >> r) & 1` (`build_grid_to_x_binary`, `binH_vs_grayH.py` L29–48). The override `GadgetAnsatz` sets `grid_to_x` to binary (`explore_OCT29/scripts/gadget_Uc_explore.py` L145–154). `Nov_3rd/PLAN.md` L12 says "Binary" and L33 says "BinH". `Nov_3rd/settings.py` lists "Gray" in `FORBIDDEN` (L96–106). The SNAP popcount is taken over Fock k < 8 (`probe6_spectral_mixer.py` L92–93), and the phase is zero for levels above 7 (`_snap_theta`, L100–105).
- **Gray** \(g=f\oplus(f\gg1)\) (`explore_OCT29/lib/gray_grid.py`) is the historical Oct-29 multi-cavity architecture. It is the only repository forward pass that already uses a true displacement: eigendecomposition of \(i(a^\dagger-a)\) at nf = 24 in `explore_OCT29/lib/ansatz.py` L79–97, with `default_nf = 24` and `default_nf_chk = 40` in `packing.py` L30–37. It is a historical comparison, not Nov3rd.
- **Nov9th default: binary**, so that \(S\), \(W\), and the encoding match Nov3rd. Gray is allowed only as the labeled variant "Nov9th-Gray" (Idea 2).
- Grouping default: interleaved \([a,a+m,a+2m]\), \(m=N/3\), as in Nov 8 KM (`packing()` in `run_sg.py`). Local bit r of cavity a is site `cav[r]`. Other groupings or bit-role orders count as classical preprocessing and are charged (§7).

What the binary map implies for \(D\) (analytic, workers must confirm in T9):

- Within 0..7, \(a^\dagger-a\) has edges \(f\leftrightarrow f+1\) with weight \(\sqrt{f+1}\). The edges 0–1, 2–3, 4–5, 6–7 flip bit 0 only. The edges 1–2 and 5–6 flip bits 0 and 1 together. The edge 3–4 flips all three bits. The edge 7–8 leaves the code.
- Real \(\alpha\) makes \(D\) a real orthogonal matrix. Its logical generator is a sum of controlled real Y-type rotations, so it rotates \(|+\rangle\) toward Z-polarization.
- Level 7 = |111⟩ is the only popcount-3 level, so \(s_3\) is an independent phase on exactly the level that leaks. That is a leakage-interference control (Idea 1). In Gray encoding, level 7 = 100 and shares a shell with levels 1 and 3, so this control does not exist.

### 3.2 Off-code gate extension

Recommended defaults are marked ★. Each is a user decision (DECISIONS.md D2–D3); workers may launch with the defaults and must label runs with them.

- **SNAP on f ≥ 8:** ★ phase 0 (`_snap_theta` pad-0 convention). This adds no control.
- **\(U_{cZ}\) on leaked components:**
  - ★ **Uz** (zero-extended logical Z): \(\tilde Z_i=\sum_{f<8}z_i(f)|f\rangle\langle f|\) on its cavity, and 0 on levels ≥ 8. A pair term that touches a leaked cavity acts as the identity, and the other terms still act. This is the natural extension of pairwise conditional-phase compilation.
  - **U0** (legacy code convention, sensitivity arm): if any cavity is leaked, the whole product component gets no phase. This follows `psi[sl] * phase` on the computational slice in `Nov_3rd/ansatz.py` L143–145 and `explore_OCT29/lib/ansatz.py` L171–175. The local "identity off-code" diagnostic presumably used this. The parent should confirm.
  - Leaked amplitude can re-enter through a later \(D\), so the choice matters for L ≥ 2. A headline result needs median |ΔF| between Uz and U0 ≤ 0.01. Otherwise report it as extension-dependent.

### 3.3 Energy observable on the full Fock space (anti-exploitation)

Energy is obtained only from projective cavity readout, never from a renormalized logical state. Use the per-term penalized extension ★:

\[
H^{\rm ext}=\sum_{i<j}J_{ij}\,\mathcal Z_{ij}-b\sum_iX_i^{L}P^{L}_{c(i)}+b\sum_i\big(1-P^L_{c(i)}\big),
\]

- \(\mathcal Z_{ij}=z_iz_j\) if both cavities are in the code, and \(|J_{ij}|\) if either is leaked.
- \(P^L_c\) projects cavity c onto levels 0..7. \(X^L_i\) is logical X on the code block.

Claim (proof in DECISIONS.md D4): \(H^{\rm ext}\) is block diagonal over leak patterns. Every block's minimum is ≥ \(E_0\), with equality only for no leak plus the GS. So leakage can never lower the measured energy.

Alternative (more conservative, higher variance): global \(E_{\rm pen}=\sum_{i<j}|J_{ij}|+bN\) for any leaked shot. Negative control (diagnostic only, never used for selection): **H0**, where leaked outcomes score 0. It exists to demonstrate exploitation.

Measurement accounting per energy query:

1. **Z setting:** photon-number-resolved readout of every cavity. f < 8 gives the 3 bits; f ≥ 8 is a per-cavity leak flag. Simulation model ★: ideal number-resolved projective readout with a "≥ 8" bin. Stress diagnostic: bit-wise ancilla readout of f mod 16 (f ≥ 16 aliases into the code space). Report the aliased population.
2. **X setting:** logical readout unitary \(W_L=\) `build_W_BinH` on 0..7, identity on ≥ 8, per cavity, then the same number readout. In binary encoding \(W_L\) is exactly Nov3rd's \(W\). This is the same idealized primitive Nov3rd already assumes \(2L+1\) times per circuit. The X setting adds one \(W\) per cavity per X-shot; count it. The product of all X outcomes gives the parity \(\langle\Pi\rangle\) at no extra cost.
3. Two settings per query. A query with S shots per setting costs 2S shots plus C·S \(W_L\) applications. The \(W_L\) compile (SNAP–D sequence) and its own leakage are **not modelled**. No hardware-feasibility claim may be made until they are (D5).

### 3.4 Fidelity and leakage (evaluation only)

- \(F=|\langle g|P^L\psi\rangle|^2\), unconditioned, with \(P^L\) projecting all cavities onto the code and no renormalization.
- \(p_{\rm leak}=1-\|P^L\psi\|^2\).
- A postselected fidelity \(F_{\rm ps}=F/(1-p_{\rm leak})\) may be printed only in a column labeled "postselected (not physical)".
- Also report the parity \(\langle\Pi\rangle\), \(E/E_0\), and the residual \((E-E_0)/|E_0|\).
- F is never a stopping rule, a selector, or a hyperparameter target.

## 4. Cutoff convergence (per idea, before any dev table)

- Bound \(|\alpha_\ell|\le\alpha_{\max}=0.30\) via \(\alpha=\alpha_{\max}\tanh(u)\).
- NF ladder {16, 20, 24, 32, 48}. Choose the smallest NF_sim such that, at the selected parameters of every dev run, the state norm outside NF_sim and the change in E, F, \(p_{\rm leak}\) against NF = 48 are all ≤ 1e-8.
- Freeze NF_sim before validation. The local stream reports a difference below 2e-16 between 32 and 48 at fixed controls; repeat the check at optimized controls.
- First-order leakage estimate, a hypothesis to check against local numbers: the level-7 amplitude of \(|+\rangle\) per cavity is \(1/\sqrt8\), so one layer leaks about \(\alpha^2\) per cavity, and the total is about \(C\alpha^2\). With aligned signs over L layers it is about \(C(\sum_\ell\alpha_\ell)^2\). This is consistent with the local report of about 2.4% at L = 1 and about 20.4% at L = 3 with α = 0.1 if those numbers were at N = 6, which the report does not state. Consequence: at fixed α, unconditioned F loses about \(C\alpha^2\) per layer, which grows linearly in N. This is the central scaling risk.

## 5. Optimizer: SPSA only

No BFGS, no natural gradient, no autograd gradients inside the optimizer.

Per run (one method × instance × b × L), the cap is **1024 energy queries, inclusive**. Every call to the energy function counts. Allocation:

| item | queries |
|---|---|
| 4 independent starts × [8 calibration (4 two-sided estimates) + 123 iterations × 2 + 2 final candidates] | 4 × 256 = 1024 |
| (allowed alternative, if preregistered identically for every method: 2 starts × [16 + 247 × 2 + 2]) | 2 × 512 = 1024 |
| monitoring | 0 counted. Off-budget diagnostics are allowed in simulation and must be labeled "oracle diagnostic, not selectable". |

The local audit spent 340 of 1024 queries on monitoring. Recommend ≤ 64 if any in-budget monitoring is used.

Schedule (Spall):

- \(a_k=a/(k+1+A)^{0.602}\), \(c_k=c/(k+1)^{0.101}\), \(A=25\).
- Rademacher \(\Delta\). The perturbation on coordinate j is \(c_k\sigma_j\Delta_j\), with fixed group scales σ: γ 0.10, θ 0.25, α 0.02 (in α units, applied through u), s 0.15.
- Calibration: \(a\) is set so the median initial step is 0.5σ per coordinate, using the 8 calibration estimates. Per-step coordinate clip at 2σ.
- Candidates: the last iterate and the Polyak average of the last 64 iterates. Evaluate both (2 queries) and keep the lower energy. Return the better of the 2 starts by these evaluated energies.

Finite-shot mode:

- Final candidate selection must use **fresh shots** not used by the SPSA steps, inside the cap (count them).
- Default rule, adopted from the local N = 12 development pilot: keep a shortlist of 8 iterates (spread over the run plus the Polyak averages), then re-rank them once by an independent noisy energy at equal total shots per run. Keep the 1024-query cap: shortlist re-evaluations count as queries. This beat the original monitor in 36/48 F cells and moved the median bias from −0.315 to −0.013, with one exception (Nov3rd b = 1, L = 3). Re-verify it per idea on dev. Also report the re-evaluated exact energy of the selected point to measure winner's-curse bias. The local stream saw up to −0.50 at N = 9.
- \(c\) must satisfy: median \(|E(x+c\Delta)-E(x-c\Delta)|\ge3\hat\sigma_E\) at calibration. If it fails, multiply c by 2, up to two times, inside the budget.

Hyperparameter fairness: each method may try at most 4 (σ, target-step) configurations, selected on dev only, with identical rules for QAOA.

Failures: NaN, norm drift > 1e-9, or cutoff loss > 1e-8 is logged and scored as F = F(x0), E = E(x0). Report the count.

RNG: `np.random.default_rng(np.random.SeedSequence([seed, N, round(10*b), L, method_id, start_id, stream]))`, with stream 0 for SPSA Δ, 1 for shots, 2 for random init. method_id is fixed in `protocol_draft.json`.

Budget curves: also record the selectable result at internal caps 64, 128, 256, 512 by running separate capped runs (a shorter schedule, same rule). They are cheap at N ≤ 12. "Fast" means F at small cap, not only at 1024. The local report has QAOA above 0.9 by 64 queries at N = 12, b = 1.

## 6. Stages and gates

| stage | content | seeds | notes |
|---|---|---|---|
| S0 | conformance tests T1–T14 (below) + classical ED diagnostics: \(F_+\), \(E_0\), gap, \(\Delta_{\rm odd}=E_0^{\rm odd}-E_0\) | dev + val, N 6–15 | no optimization |
| S1 | exact-energy SPSA screen | dev 10000..10031, N 6, 9, 12, b ∈ {1, 1.5}, L ∈ {1, 2, 3} | local slice N6 dev 10000..10007 first |
| S2 | N = 15 dev (32) plus budget curves at N = 12 | dev | NF frozen |
| S3 | shot mode, S ∈ {128, 512, 2048} per setting, fresh-shot selection | dev N 9, 12 | ranks the ideas |
| FREEZE | recipe, NF_sim, extensions, hyperparameters → PREREG.json sha256 committed | – | |
| S4 | validation, exact + S = 512 | 11000..11031, N 6, 9, 12, 15 | at most one re-freeze, logged. A second failure is reported as negative. |
| S5 | final, exact + shot subset (first 100 final seeds at S = 512, preregistered) | 20000..20999, N 6, 9, 12, then 15, 18 if N = 12 passes | blind |

Small N is not evidence. The local \(F_+\) medians are 0.899 / 0.736 / 0.667 for b = 1 at N = 6 / 9 / 12, and 0.951 / 0.891 / 0.857 for b = 1.5. Gates are therefore relative and evaluated at N ≥ 12, with b = 1 as the discriminating field.

- **Absolute goal (user):** median F ≥ 0.8 and ≥ 80% of instances with F ≥ 0.7, at one fixed L ≤ 3, for every N in {12, 15, 18}, both b, exact mode. Also report it in shot mode.
- **Non-triviality:** median deficit closure \(\eta=(F-F_+)/(1-F_+)\ge0.5\) at N = 12 and 15, b = 1.
- **Scaling:** median F(N = 15) − median F(N = 12) ≥ −0.03 at the same L and budget, with the per-run query cap held at 1024.
- **QAOA comparison:** paired per-instance ΔF against QAOA p = L, with median, win-rate, and a bootstrap 95% CI, at caps 128 and 1024 in both exact and shot mode. Never report a threshold "victory" without the paired QAOA column and \(F_+\).

## 7. Accounting table (every result row)

- Controls: Nov9th \(1+n_\theta+4L\), Nov3rd \(1+n_\theta+3L\), QAOA \(2L\).
- Classical preprocessing:
  - K: \(O(N^3)\).
  - M: \(O(N^2)\).
  - Low-rank R: \(O(N^3)\) eigendecomposition.
  - Grouping/roles rule: \(O(N^3)\).
  - ED appears only in evaluation.
- Compiled per layer:
  - Nov9th: \(\binom N2\) ZZ terms (3C intra-cavity, the rest cross-cavity) + C displacements + C SNAPs.
  - QAOA: \(\binom N2\) ZZ + N single-qubit X.
  - Nov3rd: \(\binom N2\) ZZ + C SNAPs + 2C ideal \(W\).
  - Readout per query: Nov9th 2 settings with C \(W_L\) per X-shot. QAOA 2 settings with single-qubit H.
- Queries: calibration / iterations / candidates / restarts / monitoring / continuation / failures, separately. Shots = 2·S·queries.

## 8. Conformance tests (each worker implements independently; tolerances 1e-12 unless stated)

- T1: \(J\) bytes and sha256 regenerate from the seed.
- T2: GS residual.
- T3: Nov3rd ≡ \(|+\rangle[U_{cZ}\,WSW]^L\).
- T4: \(D\) at NF = 48 against analytic coherent state, Poisson weights, and Laguerre matrix elements (local stream: passed below 4e-16).
- T5: \(D(\alpha)D(-\alpha)=I\) to NF-convergence.
- T6: the s ↔ t shell map.
- T7: \([U_{cZ},S]=0\) on the full space under Uz and U0.
- T8: \(H^{\rm ext}\) lower bound \(\min\mathrm{spec}\ge E_0\) by exhaustive pattern check at N = 6, NF = 10.
- T9: \(P(a^\dagger-a)P\) edge table of §3.1.
- T10: \(\alpha=0\Rightarrow\langle H_{ZZ}\rangle=0\) and \(p_{\rm leak}=0\).
- T11: at L = 1 with real α, \(\langle H_{ZZ}\rangle(\gamma)=\langle H_{ZZ}\rangle(-\gamma)\) (complex-conjugation symmetry).
- T12: H0 negative control shows a lower energy than \(H^{\rm ext}\) at some leaky point (proves the penalty is active).
- T13: query counter equals 1024 at exit.
- T14: QAOA p = L dense energy agrees with an independent implementation.

## 9. Structural facts to keep in view (analytic; workers verify by T7, T11)

- \(U_{cZ}\) and bare \(S\) are both diagonal in the Fock product basis, so they commute. The circuit is \(S_L D\,(U_{cZ}S_{L-1})D\cdots(U_{cZ}S_1)D\,U_{cZ}|+\rangle\). The only mixer is \(D\).
- The final \(S_L\) cannot change Z-basis probabilities. It changes only the X-setting energy and F. Because the GS is real and positive, its job is to straighten phases, and it has only 3 shared knobs to do it.
- At L = 1 with real α, the Z-distribution is even in γ. Cost information enters \(\langle H_{ZZ}\rangle\) only at \(O(\gamma^2)\), whereas QAOA's mixer gives \(O(\gamma\beta)\). For L ≥ 2 the \(S_1\) phases restore first-order coupling. Prediction: Nov9th L = 1 is weak. Complex α would remove this limit but is not authorized (D7).
- Parity \(\Pi\): QAOA and Nov3rd commute with \(\Pi\) (ZZ, X-strings, \(|+\rangle\) even). Nov9th does not, because \(D\) breaks \(f\leftrightarrow7-f\) (edge weights \(\sqrt{f+1}\) are not mirror-symmetric). \(S\) also breaks it unless \(s_2=-s_1\) and \(s_3=0\). Energy-only optimization can therefore land on even/odd mixtures with near-GS energy but F ≲ 0.5 when \(\Delta_{\rm odd}\) is small. Report \(\langle\Pi\rangle\) and \(\Delta_{\rm odd}\) per instance. Expect this to hurt mostly at b = 1.
- Nesting: appending a layer with α = 0, s = 0 is **not** the identity for Nov9th, because the shared \(U_{cZ}\) is applied again. QAOA nests exactly. L-continuation for Nov9th is initialization only.
