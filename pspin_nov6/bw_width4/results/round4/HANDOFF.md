# Handoff — staggered Baxter–Wu at Γ = 0.8

The user-facing report is `REPORT.md` in this directory. This file is the same delivery.

Compilation only. No new QAOA. Branch A `fbaac8375a9159678c0ece3e54bde60bbf62d5f1` had not moved at the last fetch. Branch B `8c060490b2b6a3bc6df77612b41080ab5b44efc6` has published the Γ=0.8 replay (`REPORT_round4.md` on that branch): stored schedules agree to about \(10^{-15}\) in fidelity, and the faces match.

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

The table, the three N=18 states, the Schmidt upper bounds at Γ=0.8, and the caveat list are in `REPORT.md`. Branch B’s round-4 file is the replay: it is no longer pending. The product upper bounds at this field are 0.150 (3×4), 0.197 (3×5), 0.196 (4×4), and 0.055 (3×6).

## Files

- `/opt/cursor/artifacts/pure_p4/round4/REPORT.md`
- `/opt/cursor/artifacts/pure_p4/round4/HANDOFF.md`
- `/opt/cursor/artifacts/pure_p4/round4/summary_gamma08.csv`
- `/opt/cursor/artifacts/pure_p4/round4/provenance.json`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig1_F_vs_L_gamma08.pdf`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig2_F_vs_invN_gamma08.pdf`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.png`
- `/opt/cursor/artifacts/pure_p4/round4/figures/fig3_errors_vs_L_gamma08.pdf`
