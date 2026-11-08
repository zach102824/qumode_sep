# Equal-depth diagnosis, before the held-out recipe

Parent counts were recomputed from the raw jsonl. Censored rows are excluded. Tau is 1e-9. Each candidate is compared with the lowest-energy equal-depth QAOA row, and the fidelity of that same row. The one censored row in each file is a QAOA extra at p=9, not a candidate. All 48 candidates are present at every depth.

N=6 BOTH is 45, 46, 46, 46, 46, 46 out of 48. N=9 BOTH is 27, 27, 27, 27, 36, 35. N=9 energy wins match those BOTH counts. N=9 fidelity wins are 41, 28, 27, 41, 41, 40. The N=9 failure split, worse-energy/better-fidelity versus worse on both, is 14/7, 1/20, 0/21, 14/7, 5/7, 5/8. There is no better-energy/worse-fidelity case.

On N=9 the 27 L=1 BOTH chains are still BOTH at every later depth. The 7 L=1 losses on both metrics are still losses at L=6. Eight of the 14 L=1 worse-energy/better-fidelity chains become BOTH by L=6. L=1 has two states at error 0.7551, with overlaps above 0.99 in groups of 17 and 10, and a worse state at error 1.5401 in groups of 5 and 9. Every L=1 gradient norm is below 1e-6, so those points are stationary. The L=6 winners share one thetaK and one thetaM; gamma appears with both signs. Several L=6 losers are also stationary, including chains whose error is about 4.3. Extra L-BFGS from those saved points, and three perturbations of width 0.2, did not reach the winner. That is incomplete evidence about expressivity: it only says this search did not leave the basin.

A separate screen of six fresh L=1 starts, seeds `[9109, 9, 1, k, s]`, on eight failed chain ids all selected the energy -5.581943, which is the 0.7551 basin. Those seeds are burned. They are not part of the held-out trial. The screen spent 2202 optimizer evaluations and the continuation probes spent 2541. Both are charged here and are not held-out trials.

The frozen remedy is therefore an energy-only L=1 screen of six starts, then one Pagano continuation. It does not change the field, the packing, the generator, or the per-layer controls. Fidelity is not used to choose the start.
