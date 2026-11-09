# Native SNAP + true-D compilation of \(W_{\rm BinH}\) (feasibility, not SK)

Bounded development. **No SK energy fits.** Historical SK ledger stays **2818048**. No GR, validation, final, Nov9/SYK, ε sweep, or BFGS/Adam/autograd. PR106 remains algebraic pad invariance. PR107 remains one illustrative Givens sensitivity test.

This asks whether the **full code isometry** \(W_{\rm BinH}=H^{\otimes 3}\) on inputs \(|0\rangle,\ldots,|7\rangle\) can be approximated by repository-justified **SNAP + true displacement** primitives, with leakage into the pad measured on the unnormalized full Fock vector.

It is **not** flat-state prep (`compile_flat_prep`: \(|0\rangle\to|+\rangle_8\) only). It is **not** \(U_X\) compilation (`fit_snap_D_to_target`: cropped \(a\), nf=8, Adam). It is **not** a hardware claim even if a candidate appears.

## Justified primitives (no extras)

| object | source | used here |
|---|---|---|
| True \(D(\alpha)=\exp(\alpha a^\dagger-\alpha^*a)\) | PROTOCOL §2.2, `displacement_matrix(..., kind='restricted')`: exp in NF+32, then restrict | yes |
| Displacement axis | PROTOCOL D7 / Nov9: **real** \(\alpha\). Complex \(\alpha\) is "not now". | **real only** |
| \(\lvert\alpha\rvert\) bound | `compile_flat_prep` `amax=1.8` (W-compile path). Nov9 D9 \(\alpha_{\max}=0.30\) is a variational-layer leak bound, not this compile. | \(\alpha=1.8\tanh u\) |
| SNAP | Hardware SNAP + `compile_flat_prep` / `fit_snap_D_to_target`: independent phases on Fock 0..7 | 8 phases per SNAP |
| SNAP on \(f\ge 8\) | D2 / `compile_flat_prep` pad-0 | **forced 0** |
| Pair order | `compile_flat_prep.apply_W`: \(D\) then SNAP, repeated | \(K\) pairs \((D,S)\) |
| Optimizer | this stream: SPSA only | no Adam/BFGS |

Not assumed (would be extra controls; if the budget fails they are **remaining limitations**, not silent additions):

- Complex \(\alpha\) (used by `compile_flat_prep` for one column; D7 forbids adding it here).
- SNAP phases on the pad.
- Cropped \(D_8=\exp[\alpha P(a^\dagger-a)P]\) (PROTOCOL §2.5: not a displacement).
- Shell-polynomial SNAP (that is the Nov3 mixer, not the \(W\) primitive).

## Target and phase convention

Let \(T\) be the \(\mathrm{NF}\times 8\) isometry: \(T[:8,:]=H^{\otimes 3}\), pad rows 0. For a compiled map, \(V\) has columns \(U|k\rangle\) in the **full** NF space (no crop, no renormalization, no postselection).

**One global phase** for the whole isometry:

\[
\phi=\arg\operatorname{tr}(T^\dagger V),\qquad V_\phi=e^{-i\phi}V.
\]

Per-column phases are **not** free (that would not be \(W_{\rm BinH}\)). Superposition tests use the same \(\phi\).

## Pulse depth and cap (one setting, not a sweep)

- \(K=3\) pairs. Same depth `compile_flat_prep` used for a *single* column. Full 8-column isometry is harder; this is a feasibility budget, not an optimal-depth search.
- Fit cutoff \(\mathrm{NF}_{\rm opt}=16\) (PROTOCOL ladder, pad exists).
- Replay cutoff \(\mathrm{NF}_{\rm replay}=32\), **same frozen** \((u,\theta)\), no reoptimization.
- Parameter count: \(K\times(1+8)=27\). Layout per pair \(k\): \(u_k\), then \(\theta_{k,0..7}\). \(\alpha_k=1.8\tanh u_k\).
- SPSA: frozen v4 Spall \(c=0.1\), \(A=25\), \(\alpha=0.602\), \(\gamma=0.101\), 4 starts, 4 cal pairs, clip \(2\sigma\). \(\sigma_u=0.15\), \(\sigma_\theta=0.25\).
- Inclusive compiler cap \(Q=2048\). \(T=Q/8-6=250\) iterations per start. `iterations_equal4` ledger must equal 2048.
- Start 0: all zeros (identity compile). Starts 1–3: \(u\sim U(-1,1)\), \(\theta\sim U(-\pi,\pi)\).
- Compiler RNG `SeedSequence([33001, NF_opt, K, start_id, stream])`. Not an SK instance seed. Not validation/final.

## Objective (compiler, not SK energy)

SPSA minimizes mean column MSE after the global phase:

\[
L=\frac{1}{8}\lVert V_\phi-T\rVert_F^2.
\]

Lower is better. Not an SK query. Counted as **compiler evaluations** only.

Reported (not selectors except \(L\)):

- Average / worst code-input infidelity \(1-\lvert\langle T_k\vert V_{\phi,k}\rangle\rvert^2\) (full vector, unnormalized).
- Mean / worst \(p_{\rm leak}=1-\lVert P_{\rm code}U|k\rangle\rVert^2\).
- Hilbert–Schmidt overlap \(F_{\mathrm{HS}}=\lvert\operatorname{tr}(T^\dagger V)\rvert^2/64\).
- Superpositions: code \(\lvert+\rangle\), \(\lvert W|0\rangle\) check, one complex random code state (seed 33001, unit). Same \(\phi\).
- Column norms \(\lVert U|k\rangle\rVert\) (cutoff leak out of NF).

## Success (prespecified)

**WCOMP_CANDIDATE** if all hold at \(\mathrm{NF}=16\) and replay at 32 does not move \(F_{\mathrm{avg}}\) by more than \(10^{-3}\):

- \(F_{\mathrm{avg}}\ge 0.99\), \(F_{\mathrm{worst}}\ge 0.95\)
- mean \(p_{\rm leak}\le 0.01\), worst \(p_{\rm leak}\le 0.05\)
- \(F_{\mathrm{HS}}\ge 0.99\)

These bars are "drop-in substitute for ideal \(W_L\)" quality, not "beats identity". Else **WCOMP_FAIL_BUDGET**. Do not add complex \(\alpha\), pad SNAP, or extra pairs after seeing the number. If the failure mode is clearly the refused extra control, say so as a **missing primitive**, still under FAIL_BUDGET.

Replay \(\lvert\Delta F_{\mathrm{avg}}\rvert>10^{-3}\) → fail (truncation not converged), even if NF=16 looked good.

## Accounting

Compiler evaluations: 2048 inclusive if the SPSA finishes. SK energy queries: **0**. Historical SK total **2818048**.

## Off limits

SK fits, new SK seeds, GR, validation, final, Nov9/SYK, BFGS/Adam, ε sweep, cropped-\(a\) \(D_8\), hardware claim from a numerical compile, open-ended depth/hyperparameter sweep.
