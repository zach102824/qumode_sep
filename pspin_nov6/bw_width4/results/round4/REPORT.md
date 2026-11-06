# Γ = 0.8 staggered Baxter–Wu

Final compilation. No new search. Unscaled Γ = 0.8 at every size. 4×3 is the same N = 12 Hamiltonian as 3×4.

## Hamiltonian

\[
H = -\sum_{\triangle} J_{\triangle} Z_i Z_j Z_k - \Gamma \sum_i X_i,
\qquad J_{\blacktriangle}=+1,\quad J_{\blacktriangledown}=-1.
\]

Equivalently \(H = -\sum_{\blacktriangle} ZZZ + \sum_{\blacktriangledown} ZZZ - \Gamma \sum_i X_i\). Triangular torus, index \(x + L_x y\), qubit \(q\) = bit \(q\). Up face \((x,y),(x+1,y),(x+1,y+1)\). Down face \((x,y),(x,y+1),(x+1,y+1)\). Branch B’s geometry hash for 3×6 matches `72597a804a743f44797ebd217af83bb3afc8de72be62ae971cd3ef6fa2d63be2`. The target is the full-space ground state. γ period π/4. β period π.

## Ansatz

\(|+\rangle^{\otimes N}\). \(H_B = \sum_i X_i\). Layer 1 first, cost then mixer, independent \((\gamma_\ell,\beta_\ell)\):

\[
|\psi_L\rangle = e^{-i\beta_L H_B} e^{-i\gamma_L H_C}\cdots e^{-i\beta_1 H_B} e^{-i\gamma_1 H_C}|+\rangle^{\otimes N}.
\]

Shared-γ is a diagnostic only. The plotted points are energy-selected. Energy minimization does not force the fidelity to rise. Nothing is smoothed.

## Table

\(F_{1\max}\) is the best value found after a period grid and a local search. It is not a certificate. For N = 18 that value is 0.095229. \(F_{\mathrm{deep}}\) is the fidelity of the energy-selected state at the largest depth that was run. On N = 18 that state is also the highest fidelity found in the final budget.

| N | geometry | grid / \(F_{1\max}\) | energy-selected deep \(F\) | product upper | gap | \(w\) | \(F_{\mathrm{flat}}\) |
|---:|---|---|---|---:|---:|---:|---:|
| 12 | 3×4 | 0.269 / 0.2695 | 0.895 at L=6 | 0.150 | 0.0422 | 0.837 | 0.835 |
| 15 | 3×5 | 0.453 / 0.460 | 0.976 at L=6 | 0.197 | 0.227 | 0.796 | 0.788 |
| 16 | 4×4 | 0.408 / 0.413 | 0.973 at L=5 | 0.196 | 0.240 | 0.803 | 0.802 |
| 18 | 3×6 | 0.090 / 0.095229 | 0.836 at L=6 | 0.055 | 0.0256 | 0.786 | 0.786 |

N = 18 energy-selected states from branch A `ce5b838a`, replayed by branch B `fe3d45a5` (`n18_g08_round4_replay.json`, verdict PASS; \(|\Delta E|\le 3\times 10^{-13}\), \(|\Delta F|\le 2\times 10^{-14}\); geometry and state hashes match):

| L | energy | fidelity |
|---:|---:|---:|
| 5 | −23.91728827275738 | 0.610420656879744 |
| 6 | −24.702062791530587 | 0.835805168290513 |

Ground energy \(E_0 = -25.559302\). Both circuit energies lie above \(E_0\). They are variational upper bounds on the ground energy. The fidelities are the overlaps of these located states. They are not a certificate that no better schedule exists. At L = 5 and L = 6 the best fidelity in that budget is this same state. At L = 1 the energy-selected overlap is 0.0786 and the fidelity search reached 0.095229.

Reapplying the stored angles reproduces those fidelities to about \(10^{-14}\) and the published triangle expectations. Errors against the ground state: at L = 5, \((E-E_0)/|E_0| = 0.0642\), mean \(|\Delta\langle ZZZ\rangle_{\blacktriangle}| = 0.126\), site-averaged \(|\Delta\langle X\rangle| = 0.200\). At L = 6 those are 0.0335, 0.0667, and 0.107. Down-triangle errors equal the up-triangle errors. On 3×6 the sublattice magnetizations are 0, so no \(m_A\) curve is drawn. 3×4, 3×5, and 4×4 have no 3-coloring.

Earlier N = 18 L = 5 states, kept as history and not used as the curve:

| state | F | energy |
|---|---:|---:|
| blind search | 0.420 | −23.412 |
| 3×4 angles transferred | 0.524 | −23.485 |
| energy polish of that transfer | 0.471 | −23.596 |

## Caveats

- \(F_{1\max}\) is grid-checked and not certified. On 4×4 a coarser grid stopped at 0.356.
- No fit and no \(L/N\) or size-scaling claim. N = 15 above N = 12 is these four circuits.
- The width-3 tori are quasi-1D strips. 4×4 reaches high fidelity and is still one 16-site torus.
- No hardness claim, no speedup, and no criticality claim.
- Product uppers are branch B’s Schmidt bounds over the named cut and 16 random cuts, not every cut.

## Figures

- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.png`
