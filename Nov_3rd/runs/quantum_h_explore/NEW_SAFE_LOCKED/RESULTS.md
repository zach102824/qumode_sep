# NEW_SAFE locked — paper claim

**Status: LOCKED.** λ = 0.75, μ = 0. Shallow depth L = 2 only. Evidence is closed; this file does not re-seal n = 15 or n = 18, and it does not load n = 21.

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i
\]

μ = 0, so there is no intra-cavity XX term. The ansatz is Walsh Hamming-shell SNAP (Walsh degrees 1..4, then one shared shell on popcount 0..3 inside every 3-qubit cavity). The baseline is Fourier QAOA on the same \(H\), same L-BFGS budget. Readout is the Z histogram (unsat) plus the X histogram (\(\langle X_i\rangle\)).

## Claim

Trained only on \(n\in\{6,9\}\) and frozen, Walsh Hamming-shell SNAP at L = 2 beats Fourier QAOA on this Hamiltonian through n = 18, and the ground state stays under the cavity line Λ ≤ 0.80. On the five-seed seal the excess-energy ratio \(R = \mathrm{Med}\varepsilon_{\mathrm{SNAP}} / \mathrm{Med}\varepsilon_{\mathrm{Fourier}}\) is 0.6209 at n = 12 (B = 128), 0.6016 at n = 15 (B = 64), and 0.5880 at n = 18 (B = 32), with SNAP below Fourier on every paired instance (640/640, 320/320, 160/160). Median cavity Λ of the true ground state is 0.7477 / 0.7489 / 0.7487 at those three sizes. The locked sentence is that shallow L = 2 transfer. L = 4 is a soft lead near 0.87–0.89 and is not part of the claim.

Medε is the median over seeds of each seed's median-over-instances excess \(\langle H\rangle - E_0\). Λ is the median, over cavities and instances, of \(\langle(\sum_{i\in c} X_i / 3)^2\rangle\) on the ground state. Cavity \(c\) owns qubits \([3c, 3c+3)\), qubit 0 = MSB. Train indices are \([0:200)\) on n = 6 and n = 9; the checkpoint is the n = 9 holdout \([200:250)\). L-BFGS is 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64. Paper init β = 0.01, γ = γ_w = −0.01, φ_k = −β k.

## Frozen sizes (L = 2)

Source: PR #51, `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_lam075_mu0_n18/RESULTS.md`. Five seeds. This is the size table for the paper.

| n | B | role | Medε SNAP | Medε Fourier | R | median Λ |
|--:|--:|:-----|------------:|---------------:|------:|---------:|
| 12 | 128 | freeze | 0.4271 | 0.6879 | 0.6209 | 0.7477 |
| 15 | 64 | probe | 0.5066 | 0.8421 | 0.6016 | 0.7489 |
| 18 | 32 | probe | 0.6352 | 1.0803 | 0.5880 | 0.7487 |

Steps of R are −0.0194 (n = 12 → 15) and −0.0136 (n = 15 → 18). 5/5 seeds have n = 12 R ≤ 0.98. Product gap \((E_{\mathrm{prod}} - E_0)/|E_0|\) is 0.1265 / 0.1294 / 0.1336, all ≥ 0.03 on every instance. Median Z-peak of the ground state is 0.0379 / 0.0148 / 0.0064. The fraction of cavity-instance pairs with Λ > 0.80 is 0.230 / 0.269 / 0.203; the pass line is the median, and no n = 18 cavity is at or above 0.90.

The same n = 12 row on the bakeoff stream (PR #46 cell C and PR #50) is Medε SNAP 0.4264, Fourier 0.6876, R 0.6201, Λ 0.7477. Absolute gaps versus the seal are 0.0007, 0.0003, 0.0008, and 0.0000.

## Depth sweep R(L) at frozen n = 12

Source: PR #50, `Nov_3rd/runs/quantum_h_explore/NEW_SAFE_L_sweep/RESULTS.md`. Same packs and budget as the bakeoff. Λ = 0.7477 at every depth (it is a property of the ground state). A depth is a win at R ≤ 0.85, soft in (0.85, 0.98), and a loss above 1.02.

| L | seeds | Medε SNAP | Medε Fourier | R | paired SNAP < Fourier | call |
|--:|------:|----------:|-------------:|------:|----------------------:|:-----|
| 1 | 5 | 0.9762 | 1.1177 | 0.8734 | 639/640 | soft |
| 2 | 5 | 0.4264 | 0.6876 | 0.6201 | 640/640 | win |
| 4 | 5 | 0.2755 | 0.3172 | 0.8687 | 581/640 | soft |
| 8 | 3 | 0.2255 | 0.1173 | 1.9232 | 0/384 | loss |
| 16 | 2 | 0.1726 | 0.0797 | 2.1641 | 0/256 | loss |

L = 2 is also a win on the holdout (n = 6 R = 0.6135, n = 9 R = 0.6307). Fourier at L = 8 (Medε 0.1173) is already below SNAP at L = 16 (Medε 0.1726). L = 16 is a two-seed check.

## Disordered λ

Source: PR #49, `Nov_3rd/runs/quantum_h_explore/disordered_lambda_NEW_SAFE/RESULTS.md`. One site pattern, NumPy seed 20261105, \(\lambda_i = 0.75\,(1 + \delta u_i)\) with \(u_i\sim\mathrm{Unif}[-1,1]\), μ = 0. Freeze is still n = 12, B = 128, five seeds. δ = 0 is the bakeoff row, not a remeasurement.

| δ | Medε SNAP | Medε Fourier | L = 2 R | n = 12 median Λ | seeds R ≤ 0.85 | call |
|--:|----------:|-------------:|--------:|----------------:|:---------------|:-----|
| 0 | 0.4264 | 0.6876 | 0.6201 | 0.7477 | 5/5 | PASS |
| 0.15 | 0.4151 | 0.6772 | 0.6130 | 0.7586 | 5/5 | PASS |
| 0.30 | 0.4128 | 0.6694 | 0.6167 | 0.7610 | 5/5 | PASS |

Both disordered strengths keep L = 2 R ≤ 0.85 and median Λ ≤ 0.80. The closest Λ on that run is the n = 6 pack at δ = 0.30 (0.7956). Median Z-peak at n = 12 is 0.0334 and 0.0312. L = 4 at δ = 0.30 moves to R = 0.9699 and is not the gate.

## Old SAFE, Λ contrast

Source: PR #41, `Nov_3rd/runs/quantum_h_explore/n18_shallow_lambda_seal/RESULTS.md`. Old SAFE is λ = 0.75, μ = −0.50,

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75\sum_i X_i - 0.50\sum_c\sum_{i<j\in c} X_i X_j,
\]

scored against HVA. It is the Λ-kill appendix. NEW_SAFE Λ is the PR #51 column.

| n | NEW_SAFE Λ (μ = 0) | old SAFE Λ (μ = −0.50) |
|--:|-------------------:|-----------------------:|
| 12 | 0.7477 | 0.941 |
| 15 | 0.7489 | 0.942 |
| 18 | 0.7487 | 0.943 |

Old SAFE n = 18 median Λ = 0.943 is above the 0.85 kill line used in that seal (fraction of cavities above 0.85 is 0.995). The shallow ratios there still favor SNAP (n = 18 L = 2 R_HVA = 0.2958, Medε SNAP / HVA = 0.1336 / 0.4519). Those ratios are a different baseline and a different Hamiltonian. They stay in the appendix because the ground state has collapsed inside each cavity.

## Non-claims

**L ≥ 8.** On the depth sweep, L = 8 and L = 16 lose to Fourier (R = 1.9232 and 2.1641). Fourier is lower on every paired n = 12 instance. A deep SNAP advantage is not claimed. L = 4 remains a soft lead (seal R = 0.8881 / 0.8684 / 0.8756 at n = 12 / 15 / 18; sweep R = 0.8687) and is outside the locked sentence.

**μ ≪ 0 and XX locking.** The locked Hamiltonian has μ = 0. The measured μ = −0.50 point drives median Λ to 0.941–0.943 (PR #41): each cavity sits near a single X-magnetization sector, Λ → 1, which is an effective-spin collapse even though the Z-peak stays small (n = 18 median 0.0002). That regime is not a second lock. Bakeoff cell B (λ = 0.50, μ = −0.15, HVA baseline, L = 2 R = 0.635, Λ12 = 0.741) cleared Λ ≤ 0.80 and lost the rank to μ = 0 on L = 2 R (PR #46). It is not NEW_SAFE.

**Power-law family.** PR #42 (`Nov_3rd/runs/quantum_h_explore/powerlaw_tfim_breadth/RESULTS.md`) is a different Hamiltonian, long-range ZZ plus a field, and is needs_edit. The best frozen n = 12 ratio there is R_Fourier = 0.937 at λ = 0.50, μ = −0.50, L = 4 (SNAP Medε 0.675 / Fourier 0.720), with every seed above 0.90. It is not frozen and is not part of NEW_SAFE.

## Agreement at n = 12, L = 2

The overlap of PR #46, PR #50, and PR #51 is the locked cell at frozen n = 12. All four rows sit inside 0.01 of each other.

| source | Medε SNAP | Medε Fourier | R | Λ |
|:-------|----------:|-------------:|------:|------:|
| #46 bakeoff cell C | 0.4264 | 0.6876 | 0.6201 | 0.7477 |
| #50 depth sweep, L = 2 | 0.4264 | 0.6876 | 0.6201 | 0.7477 |
| #51 n = 18 seal | 0.4271 | 0.6879 | 0.6209 | 0.7477 |
| spot check, seeds 0–2 median | 0.4264 | 0.6876 | 0.6201 | 0.7477 |

The seal stream matches the TF-only anchor in PR #45 (n = 12 Λ = 0.7477, gap = 0.1265, L = 2 R = 0.6209, L = 4 R = 0.8881, all Δ = 0). The bakeoff and the depth sweep share one batch stream. The two streams differ by at most 0.0008 in R.

## Spot check (this dossier)

The PR #50 trainer (`Nov_3rd/scripts/run_new_safe_L_sweep.py` on `cursor/new-safe-l-sweep-445d`) was rerun for L = 2, seeds 0, 1, and 2 only. Same packs (`beat_qaoa_12h/train_packs`), same budget, λ = 0.75, μ = 0. Ground states were recomputed (dense eigh at n = 6 and n = 9, ARPACK at n = 12). Seeds 3 and 4, L ≠ 2, n = 15, and n = 18 were not rerun. n = 21 was not loaded.

Recomputed median Λ is 0.7612 (n = 6, B = 250), 0.7555 (n = 9, B = 250), and 0.7477 (n = 12, B = 128), matching the PR #50 ground-state table at four decimals.

| seed | Medε SNAP | Medε Fourier | R | SNAP best round |
|-----:|----------:|-------------:|------:|----------------:|
| 0 | 0.4264 | 0.6876 | 0.6202 | 35 |
| 1 | 0.4243 | 0.6878 | 0.6169 | 32 |
| 2 | 0.4264 | 0.6876 | 0.6201 | 31 |

Each seed matches the PR #50 L = 2 seed table at the printed four decimals, including the SNAP checkpoint round. The three-seed median equals the published five-seed median (R = 0.6201, Λ = 0.7477) because that median seed sits inside {0, 1, 2}. Against PR #51 the same median differs by 0.0007 in Medε SNAP, 0.0003 in Medε Fourier, and 0.0008 in R.

## Source index

| PR | what it locks | result directory |
|---:|:--------------|:-----------------|
| #45 | TF-only λ scan; n = 12 anchor at λ = 0.75 | `Nov_3rd/runs/quantum_h_explore/tf_only_shallow_Lambda/` |
| #46 | bakeoff; NEW_SAFE = cell C | `Nov_3rd/runs/quantum_h_explore/bakeoff_Lambda_safe_top3/` |
| #49 | disordered λ at μ = 0 | `Nov_3rd/runs/quantum_h_explore/disordered_lambda_NEW_SAFE/` |
| #50 | R(L) at frozen n = 12 | `Nov_3rd/runs/quantum_h_explore/NEW_SAFE_L_sweep/` |
| #51 | five-seed seal through n = 18 | `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_lam075_mu0_n18/` |
| #41 | old SAFE (0.75, −0.50) Λ-kill appendix | `Nov_3rd/runs/quantum_h_explore/n18_shallow_lambda_seal/` |
| #42 | power-law family, needs_edit, not this claim | `Nov_3rd/runs/quantum_h_explore/powerlaw_tfim_breadth/` |

Those directories live on the PR branches above. This branch adds only this dossier. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.
