# Historical B=0.1 quarter-pi shell restarts

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The `|J|`-max confirmation, the Fiedler confirmation, and the energy-only union stay as they were recorded.

A shift of π/2 on any shell channel was committed first and was not fit. On development index k=4 it leaves the prepared state unchanged, because the flip-symmetric support changes every shell weight in steps of 4. The rule that was fit adds π/4.

## Idea

Both circuits are the Nov 3 shared-gamma family. One uses the `|J|`-max triples. The other uses the Fiedler triples. The same four restarts are used at every N. After the standard starts at a depth, take that packing's lowest-energy standard vector and launch four more L-BFGS runs: add π/4 to shell channel 0, the same for channel 1, the same for channel 2, and negate the shared gamma while adding π/4 to channel 0. The claimed state is the lower energy of the two packings. Equal energy keeps the `|J|`-max state. Fidelity is not used to choose.

## Protocol

Frozen in `dc1c95b0` before the fits. `PREREG.json` SHA-256 `8d537cbed32496f6a412ffbdfdeb19e2f3fe625f6b4100350524f3b229a9f7cc`. Instances k=80..91. Two repeats for each packing. QAOA keeps the round-4 start list and repeats 0 and 1. Equal search is not claimed.

A size passes only when the union has more wins than losses against equal-depth QAOA at L=6 and the bootstrap 95% interval for the median fidelity difference lies above 0. The round is confirmed only if N=6, 9, and 12 all pass.

## Result

Overall: **partial**. The quarter-pi restarts do not close the hole.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM_union | 6 | 11 (11 / 0) | 1 | 0 | +0.122 [+0.047, +0.358] |
| KM_union | 9 | 6 (6 / 0) | 3 | 3 | +0.073 [−0.057, +0.191] |
| KM_union | 12 | 7 (6 / 1) | 2 | 3 | +0.087 [−0.025, +0.230] |

N=6 passes. N=9 and N=12 do not, because each interval includes 0. The losses are:

- N6 k90, fidelity only, Fiedler, F_sub 0.099 against 0.445, energy 0.082 lower. The other packing had F_sub 0.046.
- N9 k84, joint, `|J|`-max, F_sub 0.115 against 0.241, energy 0.205 higher.
- N9 k88, fidelity only, `|J|`-max, F_sub 0.511 against 0.604, energy 0.199 lower. The Fiedler state on that instance had F_sub 0.605. Energy kept the lower-fidelity packing.
- N9 k90, fidelity only, `|J|`-max, and the selected state is itself the gamma-and-shell-0 quarter-pi start. F_sub 0.004 against 0.109, energy 0.478 lower.
- N12 k86, fidelity only, `|J|`-max, F_sub 0.172 against 0.337, energy 0.748 lower.
- N12 k87, joint, Fiedler, F_sub 0.112 against 0.219, energy 0.573 higher.

The one fidelity-only win is N12 k83, energy 0.149 higher. Of 36 instances the lower energy was the `|J|`-max circuit 19 times and the Fiedler circuit 17 times. The L=6 union state is itself a quarter-pi start on 6 of 36 instances, and 18 of 36 trace their root to a quarter-pi start. Mean L=6 F_sub, union / `|J|`-max / Fiedler / QAOA: N=6 0.876 / 0.788 / 0.855 / 0.669; N=9 0.462 / 0.368 / 0.471 / 0.351; N=12 0.345 / 0.325 / 0.291 / 0.251.

On this sample neither component passes N=9 or N=12. The `|J|`-max component at N=9 has 4 wins and 5 losses. Those columns are not the claim.

Equal parameter count, union at L=3 against QAOA at L=6, has median dF +0.051, −0.075, and −0.089. The N=12 interval lies below 0. More losses than wins at N=9 and N=12.

## Cost

Each component has 3+3L parameters, 21 at L=6. The union keeps one of those vectors. QAOA has 2L parameters, 12 at L=6. QAOA did not receive the extra starts, so this is not an equal-search comparison.

Optimizer wall 2422 s of 4200. Starts 46944, of which 3456 are the quarter-pi restarts. Logged energy evaluations 4739971. L=1 grid evaluations 32256. One hundred ninety-four starts reached 2000 evaluations, all at N=6 and almost all at L=5 or L=6.

Independent replay of 1296 KM_pack, KM_fid, and QAOA states rebuilt both partitions and the ground space without the runner. The largest relative energy error is 8.9e-13 and the largest F_sub error is 5.3e-13.

These are local optima under this budget, on these 12 instances at each N.
