# Γ = 0.8 staggered Baxter–Wu — delivery

No new search. Numbers are the stored ladders on branch A `fbaac8375a9159678c0ece3e54bde60bbf62d5f1` and this branch’s 4×4 angles, checked against branch B `8c060490b2b6a3bc6df77612b41080ab5b44efc6` (`pspin_nov6/bw_independent_audit/REPORT_round4.md`). A later N=18 commit was not on A’s branch at the last fetch. 4×3 is the same N=12 Hamiltonian as 3×4 and is not a fifth point.

## Hamiltonian

Unscaled. Γ = 0.8 is the same number at every size.

\[
H = -\sum_{\triangle} J_{\triangle}\, Z_i Z_j Z_k - \Gamma \sum_i X_i,
\qquad J_{\blacktriangle}=+1,\quad J_{\blacktriangledown}=-1.
\]

That is the operator branch B rebuilt and matched to the published term lists:

\[
H = -\sum_{\blacktriangle} ZZZ + \sum_{\blacktriangledown} ZZZ - \Gamma \sum_i X_i.
\]

Triangular torus, periods \(L_x, L_y\), periodic. Site \((x,y)\) has index \(x+L_x y\). Qubit \(q\) is bit \(q\). Faces:

- up \(\blacktriangle\): \((x,y),\ (x+1,y),\ (x+1,y+1)\)
- down \(\blacktriangledown\): \((x,y),\ (x,y+1),\ (x+1,y+1)\)

The ground state is the lowest eigenvector of this full \(2^N\) operator. No symmetry sector was projected out. Cost eigenvalues are multiples of 8, so γ has period π/4 and π/8 is not a global phase. Each β has period π.

A 3-coloring exists on 3×6 and not on 3×4, 3×5, or 4×4. On 3×6 the three sublattice magnetizations are 0 on the ground state and on these circuits, so there is no \(m_A\) curve.

## Ansatz

Initial state \(\lvert+\rangle^{\otimes N}\). Mixer \(H_B=\sum_i X_i\). Cost \(H_C\) is the diagonal part above. Layer \(\ell=1\) acts first, cost then mixer:

\[
\lvert\psi_L\rangle
= e^{-i\beta_L H_B} e^{-i\gamma_L H_C}
\cdots
e^{-i\beta_1 H_B} e^{-i\gamma_1 H_C}
\lvert+\rangle^{\otimes N}.
\]

Primary schedule: independent \((\gamma_\ell,\beta_\ell)\), objective \(\langle H\rangle\). A shared γ (one γ, independent β) is a diagnostic and is not the curve in the figures. Minimizing energy does not force the fidelity, or the observable errors, to be monotone. Nothing in the figures is smoothed.

Branch B’s replay uses this order. Every stored Γ=0.8 schedule they reran agrees to about \(10^{-15}\) in fidelity.

## Table

\(F_{1,\mathrm{grid}}\) is the best overlap on the stated L=1 period grid before local refinement. \(F_{1\max}\) is the best-found value after that grid plus a local search. Branch B reproduced \(F_{1\max}\) and states that this is not a proof no higher peak exists. \(F_{\mathrm{deep}}(E)\) is the overlap at the lowest energy in the snapshot. \(F_{\mathrm{deep}}(F)\) is the highest overlap among those reported states, written only when it is a different state. The product upper is branch B’s Schmidt bound at Γ=0.8 (minimum largest eigenvalue over the named cut and 16 random balanced cuts). \(w\) is the ground-state probability on the classical cost minima. \(F_{\mathrm{flat}}\) is the overlap with the equal superposition of those minima.

| N | geometry | \(F_{1,\mathrm{grid}}\) | \(F_{1\max}\) | \(F_{\mathrm{deep}}(E)\) | \(F_{\mathrm{deep}}(F)\) | product upper | gap | \(w\) | \(F_{\mathrm{flat}}\) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 3×4 | 0.269 | 0.2695 | 0.895 at L=6 | same state | 0.150 | 0.0422 | 0.837 | 0.835 |
| 15 | 3×5 | 0.453 | 0.460 | 0.976 at L=6 | same state | 0.197 | 0.227 | 0.796 | 0.788 |
| 16 | 4×4 | 0.408 | 0.413 | 0.973 at L=5 | same state | 0.196 | 0.240 | 0.803 | 0.802 |
| 18 | 3×6 | 0.090 | 0.095 | 0.471 at L=5 | 0.524 | 0.055 | 0.0256 | 0.786 | 0.786 |

N=18 is three verified states, not one number:

| state | F | energy |
|---|---:|---:|
| blind 8-restart ladder, L=5 | 0.420 | −23.412 |
| 3×4 L=5 angles, transferred | 0.524 | −23.485 |
| energy polish of that transfer | 0.471 | −23.596 |

The polish has the lowest energy. The transfer has the highest fidelity. The blind state is neither. Branch B’s pass/fail line scored the blind state and kept the other two separate. All three clear \(F\ge 0.4\) and \(F\ge 2 F_{1\max}\) on these digits. A later search can still move them; the deep N=18 value is a lower bound on the energy optimization, not a certified minimum of the landscape.

On N=12 and N=18 the Schmidt upper (0.150 and 0.055) is far below the deep fidelities. The same is true on N=15 (0.197 vs 0.976) and N=16 (0.196 vs 0.973). Those bounds are not an exhaustive cut minimum.

## Caveats

- The L=1 maxima are grid-checked best-found values, not a formal certificate. On 4×4 a coarser 32×16 grid stopped at 0.356; a finer grid reached 0.413.
- N=18’s deep energy result is a lower bound from the searches that were stored. Branch A had not published a later N=18 file when this report was written.
- No size-scaling claim. N=15 lying above N=12 on the \(1/N\) plot is a fact about these four circuits. There is no fit.
- The width-3 tori are strips. Calling them quasi-one-dimensional is the right finite-size description of 3×4, 3×5, and 3×6.
- 4×4 reaches \(F=0.973\) by L=5 from \(F_{1\max}=0.413\), so the depth gain is not confined to those strips. It is still one 16-site torus.
- No hardness claim, no speedup, no quantum advantage, and no thermodynamic or critical-point claim.
- At N=18 the best energy and the best fidelity are different states.
- On 3×5 the site-averaged \(|\Delta\langle X\rangle|\) rises from 0.0092 at L=5 to 0.0104 at L=6 while the fidelity rises from 0.974 to 0.976. The curve is not smoothed.

## Figures

- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.pdf`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.pdf`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.pdf`

Fig. 1 is fidelity against depth. Solid points are the lowest energy found at that depth. On N=18 the open marker is the transferred state and the cross is the blind L=5 state. Fig. 2 has no line. Fig. 3 is the stored-angle replay: relative energy error, mean up-triangle \(|\Delta\langle ZZZ\rangle|\), and site-averaged \(|\Delta\langle X\rangle|\). Down-triangle errors equal the up-triangle errors on these states.
