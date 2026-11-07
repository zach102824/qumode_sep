# Round 4 V — independent sublattice + block-mean replay

Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Packing `4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717`. E0 = -8.6132024589.

The stored L5 and L6 states were reproduced by a new logical simulator, and the physical packing matched that logical state. Reproducing a stored vector is not a fresh multi-start result.

| L | replay E | source E | dE | replay F | source F | dF | overlap | amplitude residual |
|---|---|---|---|---|---|---|---|---|
| 5 | -8.2948613956 | -8.2948613956 | 1.31e-13 | 0.60137306 | 0.60137306 | 8.66e-15 | 0.99999999999999 | 4.58e-15 |
| 6 | -8.3214828470 | -8.3214828470 | 7.11e-15 | 0.60984146 | 0.60984146 | 4.44e-16 | 1.00000000000000 | 1.56e-15 |

Source tradeoff on these same states: QAOA p12 energy error 0.2878 and F 0.5147, against Rfull L6 energy error 0.2917 and F 0.6098. The 0.004 energy edge is a difference between two states, not an uncertainty on either energy. A new energy-selected QAOA state removes the L6 Pareto lead only when its error is at most 0.292 and its fidelity is at least 0.610.

| method | L | P | E | error | F | Var(H) | median E | IQR E | median F | starts | nfev | err≤0.292 | F≥0.610 | both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qaoa_fresh | 9 | 18 | -8.222874 | 0.3903 | 0.4120 | 0.5340 | -8.0502 | 5.0355 | 0.3031 | 256/256 | 76799 | 0 | 0 | 0 |
| qaoa_fresh | 10 | 20 | -8.263750 | 0.3495 | 0.4523 | 0.5303 | -8.0788 | 4.6521 | 0.3195 | 256/256 | 79751 | 0 | 0 | 0 |
| qaoa_fresh | 11 | 22 | -8.298216 | 0.3150 | 0.4849 | 0.4552 | -8.1192 | 3.2541 | 0.3459 | 256/256 | 84980 | 0 | 0 | 0 |
| qaoa_fresh | 12 | 24 | -8.325408 | 0.2878 | 0.5147 | 0.4211 | -8.0654 | 5.3966 | 0.3153 | 256/256 | 89272 | 2 | 0 | 0 |
| Rfull_fresh | 1 | 6 | -7.678785 | 0.9344 | 0.2375 | 1.5700 | -2.5937 | 6.3642 | 0.0053 | 12/12 | 584 | 0 | 0 | 0 |
| Rfull_fresh | 2 | 9 | -8.067337 | 0.5459 | 0.4280 | 1.2302 | -7.6318 | 2.0247 | 0.3487 | 14/14 | 1751 | 0 | 0 | 0 |
| Rfull_fresh | 3 | 12 | -8.111146 | 0.5021 | 0.4331 | 0.7738 | -7.9084 | 1.8087 | 0.4242 | 14/14 | 2907 | 0 | 0 | 0 |
| Rfull_fresh | 4 | 15 | -8.236940 | 0.3763 | 0.6049 | 0.7235 | -8.0375 | 5.3882 | 0.4227 | 15/15 | 3901 | 0 | 0 | 0 |
| Rfull_fresh | 5 | 18 | -8.294861 | 0.3183 | 0.6014 | 0.4524 | -7.1272 | 7.2693 | 0.2451 | 15/15 | 5329 | 0 | 1 | 0 |
| Rfull_fresh | 6 | 21 | -8.321483 | 0.2917 | 0.6098 | 0.3905 | -1.0044 | 7.3797 | 0.0034 | 15/15 | 4168 | 4 | 0 | 0 |
| Rfull_extra | 4 |  | -8.236940 | 0.3763 | 0.6049 | 0.7235 | -8.1582 | 1.7040 | 0.5074 | 48/48 | 13445 | 0 | 0 | 0 |
| Rfull_extra | 5 |  | -8.294861 | 0.3183 | 0.6014 | 0.4524 | -8.2442 | 1.4856 | 0.5507 | 48/48 | 17049 | 0 | 0 | 0 |
| Rfull_extra | 6 |  | -8.321483 | 0.2917 | 0.6098 | 0.3905 | -8.2710 | 1.4357 | 0.5560 | 48/48 | 18238 | 16 | 2 | 0 |
| control_stride4 | 6 | 21 | -8.320803 | 0.2924 | 0.8580 | 1.2474 | -0.9837 | 0.2221 | 0.0034 | 6/6 | 1556 | 0 | 1 | 0 |
| control_random0 | 6 | 21 | -8.120801 | 0.4924 | 0.3671 | 0.6752 | -0.6793 | 0.0851 | 0.0005 | 6/6 | 1309 | 0 | 0 | 0 |
| control_random1 | 6 | 21 | -8.114042 | 0.4992 | 0.3586 | 0.6721 | -0.6854 | 5.6319 | 0.0014 | 6/6 | 2151 | 0 | 0 | 0 |

## QAOA inclusive budget

- p=9: nfev 76799, source nfev 11518, ratio 6.67, 4x target 46072, reached_4x=True, best tag `fourier_34`.
- p=10: nfev 79751, source nfev 16383, ratio 4.87, 4x target 65532, reached_4x=True, best tag `fourier_38`.
- p=11: nfev 84980, source nfev 16962, ratio 5.01, 4x target 67848, reached_4x=True, best tag `fourier_19`.
- p=12: nfev 89272, source nfev 16740, ratio 5.33, 4x target 66960, reached_4x=True, best tag `fourier_7`.

## What was checked before any search

- Block-mean projection: intra-cavity M coefficients are 0, and each 3×3 block equals the mean of J on that block.
- Norms from the raw J: rK = 2.7121073849, rM = 0.8036186699.
- Physical packing circuit matches the logical state with overlap 1. The opposite within-cavity bit map does not: overlap 0.927 at L5 and 0.902 at L6. The matching direction is first-listed site as the low bit.
- Reversing the site order inside a cavity leaves the logical energy unchanged, because the shell generators are symmetric.
- A joint site relabel of J, the packing, and the state preserves energy and fidelity (differences 0 and 1e-16 on the L6 state).
- Adding π/2 to c1, c2, or c3 was a global phase on the tested N=12 point. Negating gamma and the shell, and leaving theta_K and theta_M fixed, produces the complex conjugate. Equal energy is not the same ray.
- Replay overlap is 1−8e-15 at L5 and 1−2e-15 at L6. Energy and fidelity agree with the stored cells inside 1e-10. The stored npz hashes equal the published state hashes. The recomputed amplitudes differ from those bytes by about 1e-15, so the recomputed sha256 is different.

## What the fresh searches support

Counts in the table are over every finished start. The E and F columns are the energy-selected start only. No start was initialized at the stored angles.

QAOA used 256 starts at each of p=9,10,11,12 (64, then 192 more). The eval counts are 6.67×, 4.87×, 5.01×, and 5.33× the source ledgers, all above the 4× target. Every start finished. Zero starts, at any of these depths, have energy error ≤ 0.292 and fidelity ≥ 0.610 on the same state. At p=12, two starts reach error 0.2878 with F=0.5147, and none reach F≥0.610. The p=12 best-so-far curve meets the published p=12 energy within the first few dozen starts and does not go below it.

The p=9 and p=10 energy-selected states lie on the published ray (overlap 1). The p=11 and p=12 energy-selected states lie on the time-reversed ray (overlap with the conjugate is 1, overlap with the published vector is about 0.65). Same energies and fidelities, different states. The published tradeoff is unchanged: p=12 error 0.2878 and F 0.5147 against Rfull L6 error 0.2917 and F 0.6098. The 0.004 energy gap is the difference between those two states.

The fresh Rfull L6 best is also the time-reversed ray of the stored L6 state (overlap with the conjugate 1−4e-13), found by the linear bootstrap. Four of the 15 ascending-arm starts reach that energy, including one random restart. The median of that arm is −1.004 and the IQR is 7.38. That is the same basin showing up, not a tight multi-start distribution, and it is not a robustness claim built from the stored replay. The extra 48 starts at L6, half of them σ=0.1 perturbations of this best, do not improve the energy. Their median is −8.271. Sixteen of 48 have error ≤ 0.292. Two wide restarts have F≥0.610, at errors 0.315 and 0.471. Zero starts have both. L5 is the same pattern: the energy-selected state matches the stored L5 scalars on the conjugate ray, while one non-selected restart has F=0.627 at error 0.321.

Stride-4 plus M, six random starts at L6, produced one state at E=−8.320803 (error 0.2924, F=0.858, Var(H)=1.247). The other five energies are between −1.25 and −0.93. The median is −0.984. This state misses the 0.292 error bar by 4e-4 and does not improve on the Rfull L6 energy. It is one start in a six-start control, not a new lead. The two frozen random packings, also six starts, stay near error 0.49 with F about 0.36.

Older gates 0.390/0.315 and fidelities 0.412/0.485 are the previous matched-depth baselines. The L6 comparison that matters here is the p=12 state above. Nothing in this run removes that state's Pareto position, and nothing here is a seal.

## Notes

- Replay reproduced the stored L5 and L6 states. That match is not a fresh robustness result.
- Fresh QAOA and Rfull starts were drawn from new seeds and were not initialized at the stored angles.
- Wall time 13.6 min. The second QAOA wave ran because more than 40 minutes remained after the 64-start wave.

Figures: `seed_energy_cdf.png` (the orange curve is the 15-start ascending arm, not the extra 48), `best_so_far.png`, `pareto_energy_fidelity.png`.
Old baseline gates 0.390 (L5) and 0.315 (L6), and fidelities 0.412 and 0.485, are the earlier matched-depth numbers. They are not a substitute for the p12 comparison above.
