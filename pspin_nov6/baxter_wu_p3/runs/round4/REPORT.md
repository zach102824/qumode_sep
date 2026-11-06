# Round 4 report — N=18, Gamma=0.8

Hamiltonian, unscaled, 3×6 torus only:

`H = -sum_▲ ZZZ + sum_▼ ZZZ - 0.8 * sum X`

Full-space ground state. Primary ansatz is independent (γ_ℓ, β_ℓ) with the energy objective. Shared-γ is only a side diagnostic.

## Stop

- L=5: stall, 32 of 48 starts. The second block of 16 did not improve the best energy (relative gain 0).
- L=6: stall, 32 of 48 starts. Same rule.

Relative gain is (E_before − E_after) / |E_before|. The tolerance is 1e-4.

The budget polish used 16 L-BFGS iterations and left gradient norms 0.443 (L=5) and 3.27 (L=6). A continuation polish of those two winners only (40 iterations) moved them further. The ladders below use the continued points. The budget rows, including every seed fidelity, stay in `all_runs.csv`.

## One-line cross statement

At L=5 and at L=6 the best-energy state and the best-fidelity state in this budget are the same state (the polished N=15 transfer). They do not cross.

A different seed does cross inside itself, and that comparison is kept. The raw 3×4 L=5 angles have F_seed = 0.5236 at E = −23.4849. Energy polishing that seed reaches F = 0.4712 at E = −23.5960. Lower energy, lower fidelity. That point is not the budget winner.

## Best-energy ladder

| L | E | F | source |
|---:|---:|---:|---|
| 1 | −19.619610 | 0.0786 | round 3 energy search |
| 2 | −22.146569 | 0.2240 | round 3 energy search |
| 3 | −22.504551 | 0.2743 | round 3 energy search |
| 4 | −22.504551 | 0.2743 | round 3 energy search |
| 5 | −23.917288 | 0.6104 | round 4, N=15 transfer, extra polish |
| 6 | −24.702063 | 0.8358 | round 4, N=15 transfer, extra polish |

## Best-fidelity ladder

| L | E | F | source |
|---:|---:|---:|---|
| 1 | −18.462670 | 0.0952 | round 3 L=1 fidelity search |
| 2 | −22.146569 | 0.2240 | best F among round-3 energy basins |
| 3 | −22.504551 | 0.2743 | best F among round-3 energy basins |
| 4 | −22.504551 | 0.2743 | best F among round-3 energy basins |
| 5 | −23.917288 | 0.6104 | same state as best energy |
| 6 | −24.702063 | 0.8358 | same state as best energy |

L=1 is the only depth where the fidelity search and the energy search disagree (F 0.0952 at a higher energy versus F 0.0786). From L=2 through L=6 the best fidelity found is the best-energy state.

## What changed relative to the incomplete search

Round 3’s blind L=5 result on this state was F = 0.420 at about E = −23.42. The raw 3×4 transfer was F = 0.524. This budget’s best L=5 point is F = 0.610 at E = −23.917. The best L=6 point is F = 0.836 at E = −24.702. Both came from polishing the N=15, Gamma 0.8 angles. No random start and neither differential-evolution start beat those seeds. The last 16 starts at each depth did not move the incumbent energy.

Ground state of this field: E0 = −25.559302, gap = 0.025644, Lanczos residual 8.5e-14. The L=6 energy is still about 0.86 above E0.

## Convergence label

After the continuation polish, the gradient norms are 9.6e-3 (L=5) and 1.1e-2 (L=6). The stall rule fired, and no later start beat the incumbent, but the gradient is not at 1e-4. Treat F = 0.610 and F = 0.836 as the fidelities of the best energies located here, and as lower bounds on the fidelity of a fully stationary energy minimum only in the weak sense that a tighter local solve could still move them by a small amount. They are not a certificate of the global minimum.

## Shared-gamma diagnostic

Not the primary ansatz.

- L=5: E = −19.770, F = 0.0865.
- L=6: E = −21.601, F = 0.1952.

Both are far below the independent-angle energies.

Angle files: `angles_best_energy.json`, `angles_best_fidelity.json`. Every start is in `all_runs.csv` with `start_id`, `seed_type`, `L`, `E`, `F`, `grad_norm`, plus `E_seed` and `F_seed`.
