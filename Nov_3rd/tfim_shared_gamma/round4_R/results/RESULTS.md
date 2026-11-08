# Round 4 R — initialization of the same sublattice + block-mean generator

No new generator. G = Hc + thetaK Khat + thetaM Mhat, P = 3L+3, sublattice packing `4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717`, manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Three arms were frozen in `protocol.json` before this round's training evaluations. Gamma and theta are shared and are not wrapped. New shells are not padded with zeros.

Stored-angle replay of round-3 L6: energy error 0.291720 (stored 0.291720), fidelity 0.609841, state overlap 1.000000.

## Interpolation

Knots are layer centers x_l = (l-0.5)/L. Interior interpolation is linear below 3 source layers and a natural cubic otherwise. Outside the source knots the value is the nearest endpoint. Shell coefficients are unwrapped with period pi after a numerical global-phase check. Gamma and theta are copied, not unwrapped.

## Results

| L | arm | finished | best E−E0 | F | median | IQR | energy successes | joint successes | marginal nfev | inclusive nfev | basins | conjugate-only pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | plain | 12/12 | 0.9344 | 0.2375 | 0.9344 | 0.6397 | — | — | 575 | 575 | 6 | 14 |
| 1 | transfer | 12/12 | 0.9344 | 0.2375 | 0.9344 | 0.0000 | — | — | 540 | 1484 | 2 | 20 |
| 1 | chebyshev | 12/12 | 0.9344 | 0.2375 | 0.9344 | 0.6995 | — | — | 719 | 719 | 7 | 11 |
| 2 | plain | 12/12 | 0.5459 | 0.4280 | 0.5533 | 0.5953 | — | — | 1248 | 1248 | 6 | 12 |
| 2 | transfer | 12/12 | 0.5459 | 0.4280 | 0.5459 | 0.0037 | — | — | 1008 | 3651 | 1 | 0 |
| 2 | chebyshev | 12/12 | 0.5608 | 0.4442 | 0.8438 | 0.6097 | — | — | 1808 | 1808 | 7 | 10 |
| 3 | plain | 12/12 | 0.4848 | 0.5799 | 0.5601 | 3.8620 | — | — | 2353 | 2353 | 11 | 2 |
| 3 | transfer | 12/12 | 0.4848 | 0.5799 | 0.5021 | 0.0553 | — | — | 2432 | 8546 | 6 | 1 |
| 3 | chebyshev | 12/12 | 0.4848 | 0.5799 | 0.8181 | 0.8014 | — | — | 4326 | 4326 | 10 | 1 |
| 4 | plain | 24/24 | 0.3763 | 0.6049 | 0.5528 | 2.1306 | 1 | 1 | 8900 | 8900 | 17 | 7 |
| 4 | transfer | 24/24 | 0.3763 | 0.6049 | 0.4500 | 0.0737 | 11 | 11 | 6886 | 16723 | 6 | 0 |
| 4 | chebyshev | 24/24 | 0.3763 | 0.6049 | 0.7287 | 7.3336 | 3 | 3 | 13747 | 13747 | 22 | 4 |
| 5 | plain | 24/24 | 0.3183 | 0.6014 | 0.6301 | 7.2500 | 4 | 4 | 9974 | 9974 | 22 | 2 |
| 5 | transfer | 24/24 | 0.3183 | 0.6014 | 0.3183 | 0.0000 | 21 | 21 | 10544 | 24567 | 4 | 0 |
| 5 | chebyshev | 24/24 | 0.3183 | 0.6014 | 0.5369 | 2.2941 | 7 | 7 | 18728 | 18728 | 18 | 10 |
| 6 | plain | 24/24 | 0.2963 | 0.6080 | 1.0599 | 7.2572 | 1 | 1 | 10442 | 10442 | 21 | 6 |
| 6 | transfer | 24/24 | 0.2917 | 0.6098 | 0.2917 | 0.0093 | 21 | 21 | 11065 | 20902 | 6 | 36 |
| 6 | chebyshev | 24/24 | 0.2917 | 0.6098 | 0.4728 | 7.3280 | 5 | 5 | 18937 | 18937 | 20 | 6 |

L1–L3 have no preregistered threshold, so those success cells are blank. Energy success at L4–L6 uses the frozen thresholds: L4 error ≤ 0.435669, L5 ≤ 0.390, L6 ≤ 0.315. 
Joint success also requires fidelity at least 0.3728, 0.412, and 0.485 on that same start. 
A zero success count is censored: evals per success is not a number.

The L6 transfer seed is the resampled stored L5 schedule. Before optimization it differs from the stored L6 vector by 1.71 in the largest coordinate. After optimization the state overlaps the stored L6 state by 1−10⁻¹². The angle vector is not the same: layers 2–6 differ by +π/2 on c1 and −π/2 on c2. That shift is not the π global-phase equivalence checked for a single coefficient. The fresh plain L6 best is a different state, overlap 0.77 with the stored state, error 0.296. Its median, 1.06, stays above the threshold. So fresh restarts did not occupy the stored basin; transfer from the neighboring stored schedule did, including 9 of 12 starts at σ=0.1.

L4 plain, transfer, and Chebyshev share a best energy 0.376 and fidelity 0.605. That is better than the round-3 stored L4 state (0.450 / 0.463) and, on that one start, better than QAOA p=8 (0.436 / 0.373) on both metrics. The plain median is 0.553 and the transfer median is 0.450, so the median does not beat the QAOA best energy. L5 transfer's median equals its best, 0.318, which is below QAOA p=9. That arm is seeded from stored L4 and stored L6, so the median is not a fresh discovery.

Basins use direct overlap above 0.99. L6 transfer has 36 pairs whose complex-conjugate overlap exceeds 0.99 while the direct overlap does not. Those pairs stay in separate basins.

## Screening

The finite-budget screen asks whether transfer or Chebyshev raises energy-success rate by 0.25 and cuts inclusive evals per energy success to at most half of plain, at L5 and L6. Missing the screen is not a proof that the basin is a single point, and a numerical miss is not a ceiling.
L5 transfer: energy p_success 0.875 versus plain 0.167 (difference 0.708); inclusive evals/success 1169.9 versus plain 2493.5 (ratio 0.469); screen met.
L5 chebyshev: energy p_success 0.292 versus plain 0.167 (difference 0.125); inclusive evals/success 2675.4 versus plain 2493.5 (ratio 1.073); screen not met.
L6 transfer: energy p_success 0.875 versus plain 0.042 (difference 0.833); inclusive evals/success 995.3 versus plain 10442.0 (ratio 0.095); screen met.
L6 chebyshev: energy p_success 0.208 versus plain 0.042 (difference 0.167); inclusive evals/success 3787.4 versus plain 10442.0 (ratio 0.363); screen not met.
L6 transfer starts with sigma 0.1 that reached error ≤ 0.315: 9 of 12. Zero would mean this budget did not show a neighborhood, not that the basin is only the stored point.

## Ledgers and the stored optimum

Plain and Chebyshev inclusive counts equal their marginal counts. Transfer adds the round-3 A_sub_M evaluations from L=1 through the highest neighbor used. Seeding L=5 from stored L=6, or any neighbor from an ascending round-3 chain, is not an independent fresh discovery. The stored L=6 point itself is not copied onto an L=6 start.

Round-3 QAOA after its own rescue remains the comparison: p=9 error 0.3903 F 0.412, p=11 error 0.3150 F 0.485, p=12 error 0.2878 F 0.515. The p=12 energy edge of about 0.004 over the stored block-mean state is a different state, not a noise bar on that state's energy. p=12 does not have both error ≤ 0.292 and F ≥ 0.610, so it does not remove the fidelity side of that comparison. Equal-layer results and parameter-matched results stay separate. P=21 versus p=11's 22 parameters is a rounded match, not an equality of optima.

## Provenance

- wall 346.0124840839999 s
- protocol sha256 `efa896e559d0cf750a8aa398f649db7c1a4242a1f4c9f6bca199ec82ed47b8f5`
- partial: False
