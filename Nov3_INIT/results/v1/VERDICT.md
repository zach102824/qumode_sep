# N3INIT v1 verdict

Original Nov3, four workers, 0 failures, 384 cells. Every ledger equals `Q = q L` (212992 quantum queries on the train half and 212992 on the test half). Classical `B*` preparation for the 16 instances took 6.7s and is not in those totals. Sum of cell times 271s. Fidelity did not select anything. This is not the frozen 1000-seed benchmark.

`halve8to2` was adopted on the train half at `q=128` and `q=512`, and not at `q=1024`. The same pooled rule on the untouched test half agrees at `q=128` (mean Nov3 energy change −0.152, gap vs QAOA +0.121 versus +0.203) and does not agree at `q=512` (Nov3 energy still slightly lower, but the gap vs QAOA gets worse, −0.086 versus −0.112). `q=1024` stays with `equal4` on both halves.

That `q=128` mean is not uniform across cells. On the test half at `N=9`, `L=1`, halving made Nov3 worse (0/4 energy wins). At `N=12`, `L=1`, it made Nov3 better (3/4). `n=4` confidence intervals at low budget and at `L=3` often include 0.

The stable comparison is `L=1` at `q=512` and `q=1024`: Nov3 beats QAOA on energy in 4/4 seeds on both halves and both arms. Mean paired `ΔE` is about −0.20 to −0.29 and the intervals exclude 0. Mean paired `ΔF` is about +0.10 to +0.13. QAOA at `L=1` often reaches the same energy by `q=128` (one train seed sits at −8.68822 for every arm and budget, including the sign-flipped angle pair). Nov3 still moves between `q=128` and `q=512`.

`L=3` does not show that edge. Energy wins are mixed, and giving two finalists a longer SPSA often helps QAOA as much as Nov3. Both methods escape `| + ⟩` in every cell (`E < E_+ − 0.001 N`). The product state is not where these runs stop.

`halve8to2` bundled a new center, correlated draws, and the 8-to-2 split. v2 separates the center. It is not an L1-to-L3 transfer.
