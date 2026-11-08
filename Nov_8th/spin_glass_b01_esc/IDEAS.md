# Historical B=0.1 round: repulsion escape from the energy trap

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, the quarter-pi restarts, and the classical popcount initialization stay as recorded. This round does not take a union of packings, does not add π/4 or π/2 to a shell, and does not build shells from cavity popcounts.

## Why this change

On the later slices the shared-gamma search often stops in a state whose energy is already below QAOA while its overlap with the ground space is worse. Energy-only selection then keeps that state. N12 k96 in the classical-popcount round is the extreme: F_sub 0.000 and energy 0.314 lower. The quarter-pi hop was a fixed shell displacement of that kind of vector, and one selected state was the hop itself with F_sub 0.004. Dropping the standard search, as the popcount seeds did, instead left N=9 above QAOA in energy.

The penalty gradient of overlap with a fixed state is zero when the circuit state is that state, because the circuit is unitary. A new local search has to step off the trapped vector before a repulsion term can move it. The step used here does not touch the shell angles.

## What is fit

Fresh instances k=106..117. One interleaved shared-gamma KM circuit, the same rule at every N. The round-4 start list is unchanged, including the nodata to K to KM nesting that the standard search uses. QAOA keeps that same list and does not receive the escape. Equal depth is the claim. Equal search is not claimed. Equal parameter count is reported and is not the claim.

After the standard starts at a depth, three kicks leave the energy-best KM vector and a repulsion term λ|⟨φ|ψ⟩|², with λ = 1 and φ that trapped state, is minimized. The kicks negate the two feature angles, swap those two angles, or add N(0, 1) to the three head angles only. Shells are copied. Each repulsion run is then polished with pure energy. Selection is the lowest true energy. Fidelity is not an objective and is not a tie break.
