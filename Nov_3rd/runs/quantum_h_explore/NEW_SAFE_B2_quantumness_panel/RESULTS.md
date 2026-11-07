# NEW_SAFE B=2 quantumness panel

**Verdict: PASS.** At locked NEW_SAFE, every \(n\in\{6,9,12,15,18\}\) has median cavity \(\Lambda\le 0.80\) and ground-state Z-peak \(< 1/2\).

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75\sum_i X_i
\]

\(\mu=0\). No intra-cavity XX. Cavity \(c\) owns qubits \([3c, 3c+3)\), qubit 0 = MSB. This file does not retrain SNAP and does not load \(n=21\).

## Definitions

- **Cavity \(\Lambda\).** \(\Lambda_c=\langle(\sum_{i\in c} X_i/3)^2\rangle\) on the X histogram of the true ground state. The table is the median over cavities and instances. \(\Lambda\to 1\) is one X-magnetization sector. A Z-basis product state scores \(\Lambda=1/3\). The pass line is 0.80.
- **Product gap.** \((E_{\mathrm{prod}}-E_0)/|E_0|\) for the best real XZ product state. This is the PR #51 / PR #59 quantity.
- **Product gap/\(n\).** \((E_{\mathrm{prod}}-E_0)/n\), the same energies per qubit.
- **Z-peak.** \(\max_x|\langle x|\psi_0\rangle|^2\), the largest computational-basis weight. A classical bitstring scores 1. The Z-peak column is the median over instances. The max column is the worst instance in that pool.
- **\(R_{\mathrm{Fourier}}\).** L=2, B=2 quadratic Hamming shell, \(\mathrm{Med}\varepsilon_{B=2}/\mathrm{Med}\varepsilon_{\mathrm{Fourier}}\), quoted from PR #59. Not recomputed here.

## Gate

**PASS** when every listed \(n\) has median \(\Lambda\le 0.80\) and Z-peak \(< 1/2\).

**needs_edit** when any \(n\ge 12\) has median \(\Lambda>0.80\) or Z-peak \(\ge 0.5\).

The gap columns are the referee table. They are not an extra bar on this panel.

## Panel

Ground states are the stored PR #59 diagonals of this \(H\) (`seal_NEW_SAFE_B2_n18/diag_n*.npz`). \(n=12/15/18\) are the seal pools (B = 128 / 64 / 32). \(n=6/9\) are the prior full-pack probe, train \([0:200)\) plus holdout \([200:250)\), B = 250. That is the same diagonalization as the PR #50 ground-state table. No new `eigh` or ARPACK.

| n | B | median \(\Lambda\) | product gap | gap/\(n\) | Z-peak | max Z-peak | L=2 \(R_{\mathrm{Fourier}}\) |
|--:|--:|-------------------:|------------:|----------:|-------:|-----------:|-------------------------:|
| 6 | 250 | 0.7612 | 0.1278 | 0.0412 | 0.2022 | 0.3275 | 0.6061 |
| 9 | 250 | 0.7555 | 0.1308 | 0.0415 | 0.0916 | 0.1903 | 0.6248 |
| 12 | 128 | 0.7477 | 0.1265 | 0.0400 | 0.0379 | 0.0822 | 0.6197 |
| 15 | 64 | 0.7489 | 0.1294 | 0.0410 | 0.0148 | 0.0402 | 0.5979 |
| 18 | 32 | 0.7487 | 0.1336 | 0.0427 | 0.0064 | 0.0161 | 0.5828 |

\(R\) at n = 12, 15, 18 is the PR #59 L=2 B=2 seal (Med\(\varepsilon\) 0.4261 / 0.5043 / 0.6298 over Fourier 0.6876 / 0.8434 / 1.0807). \(R\) at n = 6 and n = 9 is the same run's holdout, B = 50 (Med\(\varepsilon\) 0.2170 / 0.2950 over Fourier 0.3580 / 0.4722). Those two rows are the checkpoint distribution, not a freeze.

Fraction of cavity-instance pairs with \(\Lambda>0.80\): 0.262 / 0.247 / 0.230 / 0.269 / 0.203 at n = 6 / 9 / 12 / 15 / 18. The bar is the median. Every instance has Z-peak \(< 1/2\) (fraction \(\ge 0.5\) is 0). The largest weight in the panel is 0.3275, at n = 6.

The relative product gap is at least 0.0386 on every instance (n = 6). The #51 line of 0.03 on that relative gap holds on the whole panel. Median gap/\(n\) sits near 0.041. The per-instance floor of gap/\(n\) is lower (n = 12 minimum 0.0160), so 0.03 on the per-qubit gap is a statement about the median.

## Quote check

n = 12 / 15 / 18 median \(\Lambda\), product gap, and median Z-peak match PR #59 and PR #51 at four decimals. Deltas are 0. PR #46 cell C printed n = 12 and n = 18 as 0.748 / 0.749, 0.127 / 0.134, and 0.0379 / 0.0064. n = 6 / 9 median \(\Lambda\) and median Z-peak match PR #50 at four decimals (0.7612 / 0.2022 and 0.7555 / 0.0916). Product gap and gap/\(n\) at n = 6 and n = 9 were not in that table. They are medians of the stored \(E_{\mathrm{prod}}\) and \(E_0\).

PR #51 L=2 B=4 \(R_{\mathrm{Fourier}}\) is 0.6209 / 0.6016 / 0.5880. The column above is the B=2 quote. \(\Delta R\) against that B=4 row is \(-0.0012\) / \(-0.0037\) / \(-0.0052\).

The seal band is B \(\in\{32,64,128\}\). The n = 6 and n = 9 probe is the full pack, B = 250. The first 128 training instances, inside that band, stay on the same side of both gates:

| n | B | median \(\Lambda\) | product gap | gap/\(n\) | Z-peak | max Z-peak |
|--:|--:|-------------------:|------------:|----------:|-------:|-----------:|
| 6 | 128 | 0.7622 | 0.1249 | 0.0401 | 0.2058 | 0.3144 |
| 9 | 128 | 0.7545 | 0.1303 | 0.0417 | 0.0891 | 0.1602 |

## Reading

The call is **PASS**.

Median cavity \(\Lambda\) is 0.7612 / 0.7555 / 0.7477 / 0.7489 / 0.7487. The largest of those is 0.7612, under 0.80. Z-peak falls with \(n\): 0.2022 / 0.0916 / 0.0379 / 0.0148 / 0.0064, and the worst instance in each pool is under \(1/2\). The ground state is not a classical bitstring at any listed size. L=2 B=2 \(R_{\mathrm{Fourier}}\) on the sealed sizes is 0.6197 / 0.5979 / 0.5828.

## Non-claims

SNAP was not retrained. The \(R\) column cites PR #59. \(n=21\) was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written. L = 4 and B = 4 stay in PR #51 and PR #59. This panel does not replace those seals.
