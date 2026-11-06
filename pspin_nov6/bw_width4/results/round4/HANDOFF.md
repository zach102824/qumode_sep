# Handoff — staggered Baxter–Wu at Γ = 0.8

Compilation only. No new QAOA. Snapshot of branch A `cursor/sparse-p3-qaoa-round1-13b4` at `fbaac8375a9159678c0ece3e54bde60bbf62d5f1`. N=18 on that branch was still described there as not fully settled; a later commit was not present when this file was written. Branch B `cursor/collective-p3-family-a-f461` at `c352e0f55bd9a4c2e89484f4dfeb63a1445435e2` has not published a completed replay of these Γ=0.8 deep ladders.

## Hamiltonian

Unscaled. Same operator at every size. Γ = 0.8 is not rescaled by a gap or by N.

\[
H = -\sum_{\triangle} J_{\triangle} Z_i Z_j Z_k - \Gamma \sum_i X_i
\]

with \(J_\blacktriangle = +1\) and \(J_\blacktriangledown = -1\). That is the same operator as

\[
H = -\sum_\blacktriangle ZZZ + \sum_\blacktriangledown ZZZ - \Gamma \sum_i X_i.
\]

Triangular torus, periods \(L_x, L_y\), site \((x,y)\) has index \(x + L_x y\), and qubit \(q\) is bit \(q\) (bit 0 = site 0). Faces, periodic:

- up: \(\{(x,y),(x+1,y),(x+1,y+1)\}\)
- down: \(\{(x,y),(x,y+1),(x+1,y+1)\}\)

Checked against the published 3×3 list and, for 4×3 versus 3×4, by an explicit lattice map and a full-spectrum difference of \(4\times 10^{-14}\). 4×3 is not a second N=12.

The ground state is the lowest eigenvector of this full \(2^N\) operator. No coloring sector was projected out. 4×4 and 4×3 are not 3-colorable, so there are no sublattice magnetizations. On 3×3 and 3×6 a 3-coloring does exist; it was not used to restrict the target.

Cost eigenvalues are multiples of 8, so \(\gamma \to \gamma+\pi/4\) is a global phase on the cost unitary and \(\pi/8\) is not. Each \(\beta\) has period \(\pi\) because it multiplies \(\sum_i X_i\), and \(\pi/2\) is not a global phase. Schedules below are raw. They were not smoothed. A fold into one period is not part of the plotted curves.

## Ansatz

Initial state \(\lvert+\rangle^{\otimes N}\).

Primary circuit, layer \(\ell=1\) first:

\[
\lvert\psi\rangle = \prod_{\ell=1}^{L} e^{-i\beta_\ell \sum_i X_i}\, e^{-i\gamma_\ell H_C}\lvert+\rangle,
\]

independent \((\gamma_\ell,\beta_\ell)\). The objective is \(\langle H\rangle\), not fidelity. Minimizing energy does not force \(F\) to rise. Two plateaus in the width-4 run keep the shallower circuit.

Shared-\(\gamma\) (one \(\gamma\), independent \(\beta_\ell\)) is a diagnostic. It is lower than the independent schedule at the deep point of the Γ=0.8 strips, except 4×4 where it is closer.

Pagano et al., arXiv:1906.02700, Suppl. I.1.2, as implemented on branch A and in the width-4 run: L=1 grid over one \(\gamma\) period and one \(\beta\) period, then local minimization; L=2 seed \((\gamma,\gamma+0.2)\), \((\beta,\beta-0.2)\); later depths resample the previous raw angles in normalized time (linear, then a natural cubic spline) and optimize locally; zero-pads and random restarts are extra seeds, not part of that supplement. L=1 fidelity maximization is a separate search. Its overlap is not the energy-selected \(F\).

## Provisional table at Γ = 0.8

\(F_{1,\mathrm{grid}}\) is the best overlap on the stated L=1 period grid. \(F_{1,\mathrm{loc}}\) is after a local fidelity search. \(F_{\mathrm{deep}}(E)\) is the overlap at the lowest energy reported for that size. For N=18 that is the polish, not the blind ladder and not the transferred state.

| N | geometry | \(F_{1,\mathrm{grid}}\) | \(F_{1,\mathrm{loc}}\) | \(F_{1E}\) | \(F_{\mathrm{deep}}(E)\) | depth | manifold weight | flat-cat overlap | product search (lower bound) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 3×4 | 0.269 | 0.269 | 0.236 | 0.895 | 6 | 0.837 | 0.835 | 0.070 |
| 15 | 3×5 | 0.453 | 0.460 | 0.426 | 0.976 | 6 | 0.796 | 0.788 | 0.127 |
| 16 | 4×4 | 0.408 | 0.413 | 0.385 | 0.973 | 5 | 0.803 | not stored | 0.115 |
| 18 | 3×6 | 0.090 | 0.095 | 0.079 | 0.471 | 5 | 0.786 | 0.786 | 0.013 |

Headline the parent asked to keep until a later N=18 file replaces it: N=12, 0.269 → F(L=6) 0.895; N=15, 0.453 → F(L=6) 0.976; N=16, 0.413 → F(L=5) 0.973; N=18, 0.09 → F(L=5) 0.471 (best-energy polish).

N=18, three different objects, all from branch A at this snapshot:

- Blind 8-restart energy search, L=5: \(F=0.420\), energy \(-23.412\).
- 3×4 L=5 angles transferred, no reoptimization: \(F=0.524\). Higher overlap, not the energy minimum.
- Energy polish of that transferred seed: \(F=0.471\), energy \(-23.596\). This is the lowest energy of the three, so it is the energy-selected deep number. It is not the highest fidelity.

A Schmidt upper bound on product overlap was computed by branch B at other fields (about 0.11–0.14 at 3×4 Γ=0.7, 3×5 Γ=0.6, 3×6 Γ=1.0). It was not computed at Γ=0.8 in the files read. The product column above is a local nonnegative search. On these four ground states it did not beat \(\lvert\langle +\vert\mathrm{GS}\rangle\rvert^2\). It is a lower bound on the best product overlap, not an upper bound. The flat-cat column is \(\lvert\langle\mathrm{uniform\ classical\ ground\ superposition}\vert\mathrm{GS}\rangle\rvert^2\). Where it matches the manifold weight, the ground-state mass on those bitstrings is coherent. For 4×4 only the probability mass was stored (0.803), not that coherent overlap.

## Width

4×4 at this one field reaches \(F=0.973\) by L=5 from an L=1 fidelity-max of 0.413, with triangle errors falling from 0.225 at L=1 to 0.012 at L=5. That shows the depth gain is not confined to the width-3 strips. It is still one 16-site torus. The ground state at Γ=0.8 still has classical-manifold weight 0.80. Nothing here is a thermodynamic transition, a quantum advantage, or an \(L/N\) law. N=15 lying above N=12 on the \(1/N\) plot is a fact about these four circuits, not a fit.

## Branch B

Not asserted as a completed common-field replay. The audit report replays older stored anchors to about \(10^{-14}\) in fidelity and checks L=1 at Γ=0.8 (3×4 fidelity-max 0.270 against the CSV 0.2695). It states that no deep search was run at 3×4 Γ=0.8 or at 3×6 Γ=0.8. The Γ=0.8 L=6 numbers and the N=18 polish are therefore still waiting on that replay.

## Files

Each path is one file.

- `/opt/cursor/artifacts/pure_p4/round4/HANDOFF.md`
- `/opt/cursor/artifacts/pure_p4/round4/summary_gamma08.csv`
- `/opt/cursor/artifacts/pure_p4/round4/provenance.json`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.pdf`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.pdf`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.pdf`

Fig. 3 plots \((E-E_0)/|E_0|\) for all four geometries. Per-depth \(\lvert\Delta\langle ZZZ\rangle_\blacktriangle\rvert\) and connected-ZZ error exist only for 4×4. \(\langle X\rangle\) error was not stored and is absent from the figure.
