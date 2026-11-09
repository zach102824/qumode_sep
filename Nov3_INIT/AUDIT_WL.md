# Audit of PR106 / PR107 (no new fits)

Classical interpretation and conformance audit. Zero optimization queries. Ledger stays **2818048**. No GR, validation, or final changes. No ε sweep. No invented SNAP–D sequence.

**Verdict on the numbers:** keep them. **Verdict on the label:** PR107 is a single phenomenological \(G_{7,8}(0.10)\) sensitivity test. It is **not** compiled-\(W_L\) validation and **not** a hardware claim.

| # | Question | Verdict |
|---|---|---|
| 1 | Provenance of \(G_{7,8}\) and \(\varepsilon=0.10\) | **FAIL** as specified physical error model / hardware-calibrated / compiled-\(W\). **PASS** as an illustrative one-knob. |
| 2 | \(\varepsilon=0\) native Nov3 and native QAOA vs frozen v4 | **PASS** (L=1,2,3; rewrite and hardware; center and off-center). |
| 3 | \(H^{\rm ext}\) on leaked states | **PASS**: unconditional energy, no postselection. Leakage is penalized, not favored. |
| 4 | PR107 secondary as equal Nov3 vs QAOA degradation | **FAIL wording** (corrected below). B−A is Nov3-only. Equal \(p_{\rm leak}\) is not equal energy bias. |
| 5 | Frozen native SNAP–D compile of \(W_{\rm BinH}\) | **ABSENT**. Hardware / cutoff *pulse-compile* convergence **unvalidated**. PR106 identity pad is algebraic invariance. |

Machine record: `results/wl/AUDIT.json`. Replay of frozen selected \(x\) matches saved \(E\) on all 24 cells.

## 1. Provenance of \(G_{7,8}(0.10)\)

**Specified in protocol:** PROTOCOL §3.1 states that in binary Fock the unique \(a^\dagger-a\) edge that leaves the code is **7–8**, and that real \(\alpha\) makes a displacement a real orthogonal matrix. That is a statement about Nov9 *displacement*, not about compiling \(W_L=H^{\otimes 3}\).

**Not specified:** a Givens residual after ideal \(W_{\rm BinH}\). \(\varepsilon=0.10\) does not appear in PROTOCOL, DECISIONS, Nov3rd notes, or arxiv-2204.029 machinery in this repo. \(\sin^2\varepsilon\approx 0.00997\) was chosen as “about 1% \(|7\rangle\to|8\rangle\) per \(W\), large enough to see.”

Classification: **illustrative phenomenological knob**, not a hardware-calibrated error model, not a SNAP–D compile, not D2 periodic SNAP. D5 still says compiled \(W_L\) is unmodelled.

No further \(\varepsilon\) sweeps.

## 2. \(\varepsilon=0\) identity vs frozen v4 (classical, not in ledger)

Frozen v4 is the logical rewrite \(|+\rangle\to[U_{cZ}\to WSW]^L\) with QAOA \(|+\rangle\to[e^{-i\gamma H_{ZZ}}R_x(\beta)]^p\).

Conventions checked:

- Parameter order Nov3: \((\gamma,\theta_K,\theta_M)\) then per layer \((s_1,s_2,s_3)\).
- Parameter order QAOA: per layer \((\gamma,\beta)\). Mixer SNAP \(s_1=-2\beta\). Matches \(R_x(\beta)=\exp(-i\beta X)\) (`apply_rx_all`), not the \(\beta/2\) convention.
- \(W=H^{\otimes 3}\oplus I\) is real symmetric and \(W^2=I\), so \(W=W^\dagger=W^{-1}\) at \(\varepsilon=0\). Apply-\(W\) vs apply-\(W^\dagger\) coincide.
- Rewrite starts at \(|+\rangle\) and has no extra final \(W\). Hardware starts at \(|0\rangle\), uses \([W\,U_{cZ}\,W\to S]^L\), then a **final** \(W\). Equivalent at \(\varepsilon=0\) because \(W|0\rangle=|+\rangle\) and \(W^2=I\).
- SNAP: \(\exp(-i\,\mathrm{snap\_level\_phase})\), phase 0 on \(f\ge 8\).

N=6 seed 10018, NF=10, 24 checks (Nov3 and QAOA × L=1,2,3 × rewrite/hardware × center/off-center): max \(|\Delta E|\sim 10^{-14}\), state fidelity \(1-O(10^{-15})\), \(p_{\rm leak}\) numerical zero. **PASS.**

## 3. \(H^{\rm ext}\) on leaked states

D4 / `metrics_from_fock`: selection energy is unconditional \(\langle H^{\rm ext}\rangle\) on the full Fock vector.

\[
H^{\rm ext}=\sum_{i<j}J_{ij}\mathcal Z_{ij}-b\sum_i X_i^L P^L_{c(i)}+b\sum_i(1-P^L_{c(i)})
\]

with \(\mathcal Z_{ij}=z_iz_j\) if both cavities are in the code and \(|J_{ij}|\) if either leaked. Fidelity is unconditioned \(|\langle g|P^L\psi\rangle|^2\). No code renormalization. Postselected \(F_{\rm ps}=F/p_{\rm code}\) is diagnostic only and was not a selector.

On PR107 frozen selected states (oracle replay, 0 ledger queries):

| arm | mean \(p_{\rm code}\) | mean \(E_{\rm uncond}\) (selected) | mean \(E_{\rm cond}\) (code, not selected) | mean leak-block |
|---|---|---|---|---|
| A ideal Nov3 | \(1-5\times10^{-15}\) | −11.3335 | −11.3335 | \(\sim0\) |
| B \(G_{7,8}(0.10)\) Nov3 | 0.9765 | −11.0267 | −11.3246 | +0.0306 |
| C same-\(W_\varepsilon\) QAOA | 0.9765 | −10.7169 | −11.0098 | +0.0342 |

Leakage **raises** selected energy relative to the code-conditional value. It does not favor leaked population. Previous \(E\) columns stay unconditional \(H^{\rm ext}\); do not reclassify them as postselected.

## 4. Secondary wording (corrected)

B−A mean \(\Delta E=+0.3068\) measures **Nov3 degradation under \(G_{7,8}(0.10)\)** versus ideal Nov3 on the same seeds.

It does **not** establish equal degradation versus QAOA. There is no ideal-QAOA arm on 10528–35. Equal \(p_{\rm leak}\) (\(\approx 2.35\%\) on B and C) is not equal energy bias: leak-block means differ (+0.031 vs +0.034) and code-conditional energies differ (−11.325 vs −11.010). Primary B vs C remains a same-embedding comparison of two leaky circuits, not a proof that the mixer is unbiased.

## 5. SNAP–D compile of \(W_{\rm BinH}\)

**No frozen native SNAP–displacement compilation of the full 8×8 BinH \(W\) on \(\mathrm{NF}\ge 10\) exists.**

Near misses (not substitutes):

- `explore_OCT29/scripts/gadget_Uc_grape_library.py` `fit_snap_D_to_target`: cropped \(a\) on 8 levels, used for \(U_X\), not \(W_{\rm BinH}\). D8 is not a displacement (PROTOCOL §2.5).
- `explore_OCT29/lib/drivers.py` `compile_flat_prep`: fits \(|0\rangle\to|+\rangle_8\) only, not the 8×8 gate.
- `Nov9_SK_SPSA/experiments/FT/sim.py` L12–13: ideal \(W_L\) SNAP–D compile and its leakage are not modelled.
- D5 / PROTOCOL §3.3: blocker for any hardware-feasibility claim.

PR106 identity pad \(H^{\otimes 3}\oplus I\) never couples to \(f\ge 8\), so \(p_{\rm leak}\sim 0\) and \(\Delta E\) is invariant by algebra. That is **not** a cutoff-convergence test of a physical pulse compilation. Hardware / compiled-cutoff convergence remains **unvalidated**.

## Off limits (held)

No new SPSA. No new seeds. No \(\varepsilon\) sweep. No invented pulse sequence. No GR / validation / final edits. Historical cell files and `summary.json` numbers untouched.
