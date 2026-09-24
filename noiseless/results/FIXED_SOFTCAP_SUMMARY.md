# Fixed soft-cap λ=2 vs λ=3 at 200 SPSA steps

Updated Asia/Shanghai: 2026-09-24 11:36 CST (run artifacts created 03:36 UTC).

Both fixed fleets used the same 500-trial setup: `ck_pi4`, four ECD layers,
20 unique-ground-state `four_sat` Hamiltonians, 25 trials per Hamiltonian,
200 SPSA steps, seed `20260917`, and 7 workers. The cost was

\[
L = \mathrm{Gibbs} + \lambda \sum_i \max(|\beta_i|-2.1,0)^2,
\]

with no L1 term and no `--adapt-lambda`. Both runs completed with 500/500
successful jobs and zero failures.

## Results

The cut is `(1.84 - mean|β|) / 1.84`; positive is an improvement over the
requested 1.84 reference.

| arm | λ | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | \|β\| cut vs 1.84 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2 | 2 | 0.834 (417/500) | 0.1352 | 1.558 | 1.569 | 2.165 | 15.3% |
| F3 | 3 | 0.810 (405/500) | 0.1315 | 1.551 | 1.559 | 2.123 | 15.7% |
| B0 Gibbs reference | 0 | 0.940 | 0.1620 | 1.844 | 1.862 | 3.378 | -0.2% |

B0 is the existing 200-step Gibbs-only reference from
`ADAPT_LAMBDA_SUMMARY.md`, not re-run here.

## Conclusion

At 200 steps, **λ=2 is better overall for Gibbs performance**: it improves
success by 2.4 percentage points and mean p(GS) by 0.0036 versus λ=3. **λ=3
is better only for tighter amplitudes**, reducing mean\|β\| by 0.007 and
mean trial-max\|β\| by 0.042, for an additional 0.4 percentage-point cut.
Neither fixed arm matches B0 success, but both provide about a 15% amplitude
cut versus 1.84.

Artifacts:

- `fleet_fixed_lam2_bmax2p1_steps200_20260924T033612Z_summary.json`
- `fleet_fixed_lam3_bmax2p1_steps200_20260924T033612Z_summary.json`
