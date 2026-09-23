# β-aware fleet summary (ck_pi4)

Updated UTC: 2026-09-23T09:50:04Z

Baseline (`fleet_phase_bakeoff` ck_pi4 L*=4): success=0.94, mean|β|≈1.84.

## Results

| tag | λ1 | λ3 | β_max | L* | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | % over β_max | \|β\| cut vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fleet_beta_aware_A0_ckpi4` | 0 | 0 | — | 4 | 0.940 | 0.1620 | 1.844 | 1.862 | 3.378 | 0.0% | -0.2% |
| `fleet_beta_aware_A1_l1_0p005` | 0.005 | 0 | — | 4 | 0.938 | 0.1622 | 1.831 | 1.842 | 3.342 | 0.0% | 0.5% |
| `fleet_beta_aware_A1_l1_0p01` | 0.01 | 0 | — | 4 | 0.944 | 0.1627 | 1.821 | 1.838 | 3.337 | 0.0% | 1.0% |
| `fleet_beta_aware_A1_l1_0p02` | 0.02 | 0 | — | 4 | 0.930 | 0.1651 | 1.799 | 1.824 | 3.312 | 0.0% | 2.3% |
| `fleet_beta_aware_A1x_l1_0p04` | 0.04 | 0 | — | 4 | 0.912 | 0.1639 | 1.777 | 1.798 | 3.271 | 0.0% | 3.4% |
| `fleet_beta_aware_A1_l1_0p05` | 0.05 | 0 | — | 4 | 0.910 | 0.1638 | 1.757 | 1.777 | 3.258 | 0.0% | 4.5% |
| `fleet_beta_aware_A1x_l1_0p06` | 0.06 | 0 | — | 4 | 0.906 | 0.1618 | 1.727 | 1.743 | 3.242 | 0.0% | 6.2% |
| `fleet_beta_aware_A1x_l1_0p07` | 0.07 | 0 | — | 4 | 0.900 | 0.1626 | 1.717 | 1.729 | 3.237 | 0.0% | 6.7% |
| `fleet_beta_aware_A1x_l1_0p08` | 0.08 | 0 | — | 4 | 0.890 | 0.1604 | 1.694 | 1.702 | 3.205 | 0.0% | 7.9% |
| `fleet_beta_aware_A1x_l1_0p09` | 0.09 | 0 | — | 4 | 0.868 | 0.1592 | 1.682 | 1.694 | 3.187 | 0.0% | 8.6% |
| `fleet_beta_aware_A1_l1_0p1` | 0.1 | 0 | — | 4 | 0.866 | 0.1586 | 1.664 | 1.679 | 3.186 | 0.0% | 9.6% |
| `fleet_beta_aware_A1x_l1_0p12` | 0.12 | 0 | — | 4 | 0.862 | 0.1597 | 1.633 | 1.641 | 3.140 | 0.0% | 11.3% |
| `fleet_beta_aware_A1_l1_0p2` | 0.2 | 0 | — | 4 | 0.774 | 0.1520 | 1.496 | 1.490 | 3.029 | 0.0% | 18.7% |

## Notes

- Phase-1 A0/A1 complete: pure L1 never hit success≥0.90 AND |β| cut≥15%.
- Phase-2: finer λ1, soft-cap A2, longer SPSA, L*=3 follow-ups.
- fine λ1=0.04: success=0.912, mean|β|=1.777, cut=3.4%.
- fine λ1=0.06: success=0.906, mean|β|=1.727, cut=6.2%.
- fine λ1=0.07: success=0.900, mean|β|=1.717, cut=6.7%.
- fine λ1=0.08: success=0.890, mean|β|=1.694, cut=7.9%.
- fine λ1=0.09: success=0.868, mean|β|=1.682, cut=8.6%.
- fine λ1=0.12: success=0.862, mean|β|=1.633, cut=11.3%.

## How to toggle

Defaults (`--lambda1 0 --lambda3 0`, no `--beta-max`) preserve Gibbs-only cost.
Enable β terms via CLI on `python -m noiseless.run_u_sweep`.

