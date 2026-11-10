# NOV3_QAOA_QL_v1

Owner assignment t3328u, 20:23 Asia/Shanghai, 2026-11-09. One cloud campaign on this branch. Machine-readable contract: `prereg.json` (hashed before the first fit and stored with every result).

## Question

On the per-instance entropy field \(B^*\), how does original Nov3rd compare with QAOA at \(p=L\), and how does that comparison move when the inclusive query budget is \(Q=qL\) with \(q\) halved from 1024?

\(B^*\) is the refined maximizer of the ground-state entropy across the fixed cut. It is not an assumed thermodynamic critical field.

## Ansatz (frozen)

Nov3rd is the original circuit. One \(\gamma\), one \(\theta_K\), and one \(\theta_M\) multiply the same cost diagonal in every layer. Each layer has three polynomial shell parameters \(s_1 p + s_2 C(p,2) + s_3[p=3]\). Packing is the interleaved triples \([a, a+N/3, a+2N/3]\). There is no per-layer \(\gamma\).

QAOA has \(2L\) independent angles and \(p=L\) only: \(\exp(-i\gamma_\ell H_{ZZ})\exp(-i\beta_\ell\sum_i X_i)\).

Both methods use SPSA. BFGS, natural gradient, and any ground-state or fidelity oracle inside the loop are out. The selector minimizes energy. Squared fidelity is computed on that selected state afterward.

Each \((N,\mathrm{seed},L,q)\) cell is a cold start. Cross-depth warm start is not used, so transfer queries are zero. The separate local depth-transfer run is not merged into these numbers.

## Field

\[
H=\sum_{i<j} J_{ij} Z_i Z_j - B^*\sum_i X_i
\]

\(J_{ij}\) are i.i.d. normal with variance \(1/N\), drawn from `numpy` Generator(seed), upper triangle row-major. The same \(J\) and the same \(B^*\) are reused for every method, depth, and budget.

The cut is sites \(0..\lfloor N/2\rfloor-1\) with site 0 the least significant bit. Entropy is in nats. The primary scan is 60 points on \([0.05,3]\), then a bounded scalar refine. A maximum on the boundary extends the window; it does not jump to a more convenient interior grid point. A seed whose maximizer is unresolved, or whose ground state is not a positive even state with a passing residual, is a prep failure and stays in the failure list.

Odd-sector gaps can be tiny. They are recorded and are not treated as the even-sector gap.

## Budget

\(Q=qL\) counts every initial state, calibration pair, iteration pair, restart, and selection evaluation. Monitoring is zero.

Four starts share \(Q\) equally. A start is runnable when its share is at least 16 and the share fits 1 initial + 8 calibration + an even number of iteration evaluations (at least two iterations) + 3 selection evaluations (last iterate, mean of the last \(\min(64,T)\) iterates, midpoint iterate). The initial state competes in the energy selection.

Candidate \(q\): 1024, 512, 256, 128, 64, 32, 16. Cells below that accounting floor are not given a smaller invented ledger. \(q=8\) never fits \(L\le 6\).

The halving curve is measured down to the floor. A small \(q\) is not skipped because it looks degraded.

Classical field preparation, throughput timings, and one post-selection energy recomputation sit outside \(Q\) and are labeled as such. Fidelity is not an energy query.

Gains, \(\sigma\), and the four-start law are the N3X_matched_v1 values. At \(L=1\), \(q=1024\), the total \(Q=1024\) and the split match that recipe. The campaign replays N=6 seed 10000 against the stored R2 result and stops if the energy or the gains disagree.

The SPSA seed does not include \(q\). Halving the budget repeats the same four trajectories and stops earlier; it does not draw a new start. Depth \(L\) is part of the seed, so a deeper circuit is not a prefix of a shallower one.

## Seeds

Owner t3334u at 20:34 Asia/Shanghai. This cloud does not read another worker's seeds.

| block | seeds | what is run |
|---|---|---|
| this cloud, dev | 10000–10009 | all \(N\), all \(L=1..6\), every runnable \(q\), both methods |
| LE conditioning | 10010–10017 | not read |
| FT init/restart | 10018–10025 | not read |
| local SPSA schedules | 10026–10031 | not read |
| final | 20000–20999 | full runnable grid, both methods, 1000 seeds, after this \(N\)'s freeze |
| validation | 11000–11031 | not opened |

`NOV3_QAOA_QL_v1` is frozen when this protocol is committed, before any fit. Final seeds for an \(N\) open only after that \(N\)'s ten dev seeds are on disk and `FREEZE_n{N}.json` copies this grid with the protocol hash and `retuned: false`. Dev outcomes do not change \(\sigma\), starts, or which \(q\) values exist. A later idea is a new recipe, not an edit of this panel. No seed is dropped because its fidelity is low.

Order is N=6, then 9, then 12, then 15, then 18. This campaign does not wait on LE, FT, or local. N=15 and N=18 energy timings are recorded before those large panels.

## What “minimal q” means

For a reported cell, \(q_\star(\tau)\) is the smallest runnable \(q\) such that every runnable \(q'\ge q\) is within \(\tau\) in mean fidelity of the \(q=1024\) reference of the same \(N\), \(L\), and method, on the paired seeds actually present. \(\tau\in\{0.01,0.02,0.05\}\). The 95% interval of the paired difference is reported beside the point estimate. If that interval is wider than \(\tau\), the tolerance is not resolved at that \(n\).

\(q=1024\) is the top of this grid, not a converged optimizer. There is no larger \(q\) in v1, so every minimum is relative to the \(q=1024\) cells that were actually run. \(q_\star\) does not remove final cells.

## Delivery

`PROGRESS.csv` and `SUMMARY.md` carry, for each stage, \(N\), \(L\), and \(q\): actual \(n\), mean fidelity of each method with a 95% interval, paired mean \(\Delta F\) with its interval, fidelity and energy win fractions, fractions with \(F\ge 0.7\) and \(F\ge 0.8\), mean energy error \(E-E_0\), mean charged queries, and mean wall time. Failures stay in the failure files. \(n=1000\) is claimed only when 1000 paired ok rows exist for that cell.

## External local R4, not this benchmark

Local charged L1→L3 transfer finished 64 runs and 163840 queries. The embedding check passed at \(5.49\times 10^{-15}\) and the L1 energy floor was retained. Transfer underperformed cold energy on 1/16 of the runs and on 0/8 at 4096. Old two-seed means at 4096, L3, cold/transfer Nov3 were N9 0.7745/0.7522, N12 0.7588/0.7416, N15 0.7763/0.6613. Matched cold QAOA means were N9 0.7731, N12 0.7748, N15 0.7413. On fresh N15 seeds 10008 and 10009 at frozen 4096, cold Nov3/QAOA mean fidelity was 0.8111/0.7830 with wins on 1/2 of the seeds for both energy and fidelity, and Nov3 energy error 0.3128 against QAOA 0.2816. Transfer means were 0.7853/0.7907. Those two-seed means do not promote a transfer recipe. This v1 panel is cold at each depth. Seeds 10008 and 10009 are inside this cloud's dev range and are run here under v1 only. The local numbers are not merged.

## Out of scope

Nov9 displacement, SYK, AF, N21, the R3-A ablation arms, depth-transfer warm starts, other workers' dev seeds, validation seeds, extra clouds, and any edit to the frozen N3X runner or the earlier result files.
