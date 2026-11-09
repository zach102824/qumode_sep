# N3COND B3 reading

Read-dev seeds 10010–10017. Not a holdout. Parent pid 8895. Clock 8.3 s. 128/128 fits ok. Summed worker time 32.9 s. Queries inside Q: 147456, of which 6656 are axial calibration, 139776 perturbations, 1024 candidates. Outside-cap queries: 0. Fidelity was not used to select or to gate.

## Ledger, restated

C1’s 425984 queries were the fit cap only. The scale that defined C1 was 832 axial queries at each of N=9 and N=12, outside that cap. The unused simultaneous probes are 512. B3 does not move any of those into Q.

## What B3 tested

One hypothesis, frozen before the fits: at each start, inside Q, shrink coordinates steeper than the median axial slope, with λ in [0.5, 1], and set the scalar gain from those slopes. M0 spends the same axial queries and sets λ=1. QAOA p=L uses the same rule. The floor was not scanned. On the gate cell, 34% of M2 coordinates sit on the floor 0.5 and 50% stay at 1. None exceed 1.

## Preregistered energy gate

N9, L=3, q=1024, Nov3, M2−M0, eight read-dev seeds. Median ΔE −0.340, mean −0.508, 95% CI [−0.981, −0.083], 6/8 lower energy. The CI lies below 0, so the gate as written passes. Post hoc, not gated: median ΔF +0.159, mean +0.164, CI [+0.044, +0.289], 6/8. Marginal medians: energy −5.684 (M0) and −5.876 (M2); fidelity 0.538 and 0.708; min F 0.242 and 0.560.

The same cell for QAOA, same allocation rule: median ΔE −0.007, mean −0.028, CI [−0.057, −0.006]. The matched control moves a little in the same direction. Nov3’s paired gap is larger. K is 371 for Nov3 and 377 for QAOA, because Nov3 spends 24 calibration queries per start and QAOA spends 12.

## This is not a win over scalar SPSA

Historical C0 on the same eight seeds, same cell, median energy −6.145 and median F 0.821. M2’s median energy is −5.876 and median F is 0.708. M2 beats M0 and still loses to C0. At L=3, q=128 the medians are C0 −5.774, M2 −5.739, M0 −4.633. At L=1, q=1024 they are C0 −5.686, M2 −5.652, M0 −5.438, and the M2−M0 energy CI includes 0.

The reason M0 is weak: its gain is fit to the median axial slope. Flat shell coordinates pull that median down, so the gain is about 10–50× the gain C0 fits to its simultaneous probes (for seed 10014, L=3, q=1024: C0 gains 2.5–6.3, M0 gains 38–371). The update then uses simultaneous probes and rides the clip. M2’s lower energy against M0 is a smaller clipped step on the steep coordinates. It does not repair that gain.

L=1 does not pass the energy CI test. q=128 L=1 median ΔE is +0.038.

## Gates that stay shut

C1 stays rejected. N15 and N18 stay off. No fresh-holdout claim. No adoption into the GR recipe. No second floor in this batch. A positive budget claim against scalar SPSA is not supported.
