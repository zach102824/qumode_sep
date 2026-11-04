# NEW_SAFE locked — paper claim

**Status: LOCKED.** λ = 0.75, μ = 0. Shallow depth L = 2 only. The locked SNAP is B = 2, \(\varphi_k = c_1 k + c_2 k^2\). The seal through n = 18 is PR #59. This update does not retrain, and it does not load n = 21.

**Round 16:** paper lock prefers B = 2. The seal table is PR #59 (R = 0.6197 / 0.5979 / 0.5828 at n = 12 / 15 / 18, Λ ≈ 0.75; Medε at or under the PR #51 B = 4 reference). Disordered λ is PR #60 (δ = 0.15 / 0.30, R = 0.6118 / 0.6156, Λ = 0.7586 / 0.7610, matching #49). B = 4 / #51 stays the equivalent reference. PR #58 already recorded Round 15 (B = 2 compression + sparse-p3 KILL).

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i
\]

μ = 0, so there is no intra-cavity XX term. The locked ansatz is Walsh degrees 1..4 on the cost pulse, then one shared cavity SNAP with two coefficients, \(\varphi_k = c_1 k + c_2 k^2\) on popcount \(k = 0..3\) (B = 2). That phase is the hardware quote for the L = 2 claim. The four-phase Hamming shell (B = 4, free \(\varphi_0..\varphi_3\)) and the full eight-phase cavity match it at L = 2; PR #51 is the B = 4 reference seal. The baseline is Fourier QAOA on the same \(H\), same L-BFGS budget. Readout is the Z histogram (unsat) plus the X histogram (\(\langle X_i\rangle\)).

## Claim

Trained only on \(n\in\{6,9\}\) and frozen, the B = 2 SNAP at L = 2 beats Fourier QAOA on this Hamiltonian through n = 18, and the ground state stays under the cavity line Λ ≤ 0.80. On the five-seed seal (PR #59) the excess-energy ratio \(R = \mathrm{Med}\varepsilon_{\mathrm{B=2}} / \mathrm{Med}\varepsilon_{\mathrm{Fourier}}\) is 0.6197 at n = 12 (128 instances), 0.5979 at n = 15 (64 instances), and 0.5828 at n = 18 (32 instances), with B = 2 below Fourier on every paired instance (640/640, 320/320, 160/160). Median cavity Λ of the true ground state is 0.7477 / 0.7489 / 0.7487 at those three sizes. The locked sentence is that shallow L = 2 transfer on \(\varphi_k = c_1 k + c_2 k^2\).

The B = 4 shell (PR #51) is the equivalent reference. Medε ratios of B = 2 to B = 4 are 0.998 / 0.995 / 0.991, and ΔR against 0.6209 / 0.6016 / 0.5880 is −0.0012 / −0.0037 / −0.0052. Medε stays at or under the #51 column at every sealed size. L = 4 on the locked B = 2 arm is a win (PR #59, R = 0.6174 / 0.6252 / 0.6067) and sits outside the locked sentence. The B = 4 reference stays soft at that depth (PR #51, R = 0.8881 / 0.8684 / 0.8756).

Medε is the median over seeds of each seed's median-over-instances excess \(\langle H\rangle - E_0\). Λ is the median, over cavities and instances, of \(\langle(\sum_{i\in c} X_i / 3)^2\rangle\) on the ground state. Cavity \(c\) owns qubits \([3c, 3c+3)\), qubit 0 = MSB. Train indices are \([0:200)\) on n = 6 and n = 9; the checkpoint is the n = 9 holdout \([200:250)\). L-BFGS is 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64. Paper init β = 0.01, γ = γ_w = −0.01, φ_k = −β k, which sits in the B = 2 span.

## Frozen sizes (L = 2)

Source: PR #59, `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_B2_n18/RESULTS.md`. Five seeds. B = 2, \(\varphi_k = c_1 k + c_2 k^2\). This is the size table for the paper. The n = 12 row reproduces PR #57 at four decimals (Medε 0.4261 / 0.6876, R = 0.6197, Λ = 0.7477, all deltas 0).

| n | instances | role | Medε B = 2 | Medε Fourier | R | median Λ |
|--:|----------:|:-----|-----------:|-------------:|------:|---------:|
| 12 | 128 | freeze | 0.4261 | 0.6876 | 0.6197 | 0.7477 |
| 15 | 64 | probe | 0.5043 | 0.8434 | 0.5979 | 0.7489 |
| 18 | 32 | probe | 0.6298 | 1.0807 | 0.5828 | 0.7487 |

Steps of R are −0.0218 (n = 12 → 15) and −0.0151 (n = 15 → 18). 5/5 seeds have n = 12 R ≤ 0.85. Median product gap \((E_{\mathrm{prod}} - E_0)/|E_0|\) is 0.1265 / 0.1294 / 0.1336. PR #51 records that gap at or above 0.03 on every instance of these pools; the medians, which PR #59 reproduces, are those three numbers. Median Z-peak of the ground state is 0.0379 / 0.0148 / 0.0064. The fraction of cavity-instance pairs with Λ > 0.80 is 0.230 / 0.269 / 0.203; the pass line is the median, and the n = 18 maximum Λ is 0.8748. Those ground-state columns match PR #51, because Λ does not depend on the SNAP budget.

Equivalent reference, PR #51 B = 4 (free \(\varphi_0..\varphi_3\)), `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_lam075_mu0_n18/RESULTS.md`:

| n | Medε #51 B = 4 | Medε ratio | R #51 | ΔR |
|--:|---------------:|-----------:|------:|----:|
| 12 | 0.4271 | 0.998 | 0.6209 | −0.0012 |
| 15 | 0.5066 | 0.995 | 0.6016 | −0.0037 |
| 18 | 0.6352 | 0.991 | 0.5880 | −0.0052 |

The ratio stays under 1 and does not grow with n. The same n = 12 row on the bakeoff stream (PR #46 cell C and PR #50, B = 4) is Medε SNAP 0.4264, Fourier 0.6876, R 0.6201, Λ 0.7477. Absolute gaps versus the B = 2 seal are 0.0003, 0.0000, 0.0004, and 0.0000. Absolute gaps of that bakeoff row versus the B = 4 seal are 0.0007, 0.0003, 0.0008, and 0.0000.

## Depth sweep R(L) at frozen n = 12

Source: PR #50, `Nov_3rd/runs/quantum_h_explore/NEW_SAFE_L_sweep/RESULTS.md`. Same packs and budget as the bakeoff. This sweep trains the four-phase Hamming shell (B = 4), the equivalent reference at L = 2. Λ = 0.7477 at every depth (it is a property of the ground state). A depth is a win at R ≤ 0.85, soft in (0.85, 0.98), and a loss above 1.02. L ≥ 8 was not rerun on B = 2.

| L | seeds | Medε SNAP | Medε Fourier | R | paired SNAP < Fourier | call |
|--:|------:|----------:|-------------:|------:|----------------------:|:-----|
| 1 | 5 | 0.9762 | 1.1177 | 0.8734 | 639/640 | soft |
| 2 | 5 | 0.4264 | 0.6876 | 0.6201 | 640/640 | win |
| 4 | 5 | 0.2755 | 0.3172 | 0.8687 | 581/640 | soft |
| 8 | 3 | 0.2255 | 0.1173 | 1.9232 | 0/384 | loss |
| 16 | 2 | 0.1726 | 0.0797 | 2.1641 | 0/256 | loss |

L = 2 is also a win on the holdout (n = 6 R = 0.6135, n = 9 R = 0.6307). The locked B = 2 seal at this depth is R = 0.6197 (PR #59), 0.0004 under the B = 4 row above, with holdout R = 0.6061 at n = 6 and 0.6248 at n = 9. Fourier at L = 8 (Medε 0.1173) is already below SNAP at L = 16 (Medε 0.1726). L = 16 is a two-seed check.

## Disordered λ

Source: PR #60, `Nov_3rd/runs/quantum_h_explore/disordered_lambda_NEW_SAFE_B2/RESULTS.md`. One site pattern, NumPy seed 20261105, \(\lambda_i = 0.75\,(1 + \delta u_i)\) with \(u_i\sim\mathrm{Unif}[-1,1]\), μ = 0. The SNAP arm is B = 2, \(\varphi_k = c_1 k + c_2 k^2\). Freeze is still n = 12, 128 instances, five seeds. δ = 0 is the clean B = 2 row from PR #57 / #59, not a remeasurement inside #60. The shell column is the four-phase reference: the bakeoff B = 4 row at δ = 0, and PR #49 on this same disorder at δ = 0.15 and 0.30.

| δ | Medε B = 2 | Medε Fourier | L = 2 R | n = 12 median Λ | seeds R ≤ 0.85 | shell R | call |
|--:|-----------:|-------------:|--------:|----------------:|:---------------|--------:|:-----|
| 0 | 0.4261 | 0.6876 | 0.6197 | 0.7477 | 5/5 | 0.6201 | PASS |
| 0.15 | 0.4143 | 0.6772 | 0.6118 | 0.7586 | 5/5 | 0.6130 | PASS |
| 0.30 | 0.4121 | 0.6694 | 0.6156 | 0.7610 | 5/5 | 0.6167 | PASS |

Both disordered strengths keep L = 2 R ≤ 0.85 and median Λ ≤ 0.80. Fourier and Λ match PR #49 at four decimals (Λ = 0.7586 / 0.7610). B = 2 Medε sits within 0.001 of the #49 shell (0.4143 versus 0.4151, 0.4121 versus 0.4128). The closest Λ on that run is the n = 6 pack at δ = 0.30 (0.7956). Median Z-peak at n = 12 is 0.0334 and 0.0312. L = 4 on B = 2 stays a win (R = 0.6032 / 0.6398, 5/5 seeds) and is outside the gate. The #49 shell was softer there (R = 0.8249 / 0.9699).

## Old SAFE, Λ contrast

Source: PR #41, `Nov_3rd/runs/quantum_h_explore/n18_shallow_lambda_seal/RESULTS.md`. Old SAFE is λ = 0.75, μ = −0.50,

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75\sum_i X_i - 0.50\sum_c\sum_{i<j\in c} X_i X_j,
\]

scored against HVA. It is the Λ-kill appendix. NEW_SAFE Λ is the ground-state column shared by PR #59 and PR #51.

| n | NEW_SAFE Λ (μ = 0) | old SAFE Λ (μ = −0.50) |
|--:|-------------------:|-----------------------:|
| 12 | 0.7477 | 0.941 |
| 15 | 0.7489 | 0.942 |
| 18 | 0.7487 | 0.943 |

Old SAFE n = 18 median Λ = 0.943 is above the 0.85 kill line used in that seal (fraction of cavities above 0.85 is 0.995). The shallow ratios there still favor SNAP (n = 18 L = 2 R_HVA = 0.2958, Medε SNAP / HVA = 0.1336 / 0.4519). Those ratios are a different baseline and a different Hamiltonian. They stay in the appendix because the ground state has collapsed inside each cavity.

## Non-claims

**L ≥ 8.** On the B = 4 depth sweep, L = 8 and L = 16 lose to Fourier (R = 1.9232 and 2.1641). Fourier is lower on every paired n = 12 instance. A deep SNAP advantage is not claimed. B = 2 was not trained at L ≥ 8. L = 4 stays outside the locked sentence. On the locked B = 2 arm it is a win through n = 18 (PR #59, R = 0.6174 / 0.6252 / 0.6067; PR #57 n = 12 full-cavity R = 0.6399). The B = 4 reference stays soft there (PR #51 R = 0.8881 / 0.8684 / 0.8756; sweep R = 0.8687).

**μ ≪ 0 and XX locking.** The locked Hamiltonian has μ = 0. The measured μ = −0.50 point drives median Λ to 0.941–0.943 (PR #41): each cavity sits near a single X-magnetization sector, Λ → 1, which is an effective-spin collapse even though the Z-peak stays small (n = 18 median 0.0002). That regime is not a second lock. Bakeoff cell B (λ = 0.50, μ = −0.15, HVA baseline, L = 2 R = 0.635, Λ12 = 0.741) cleared Λ ≤ 0.80 and lost the rank to μ = 0 on L = 2 R (PR #46). It is not NEW_SAFE.

**Power-law family.** PR #42 (`Nov_3rd/runs/quantum_h_explore/powerlaw_tfim_breadth/RESULTS.md`) is a different Hamiltonian, long-range ZZ plus a field, and is needs_edit. The best frozen n = 12 ratio there is R_Fourier = 0.937 at λ = 0.50, μ = −0.50, L = 4 (SNAP Medε 0.675 / Fourier 0.720), with every seed above 0.90. It is not frozen and is not part of NEW_SAFE.

## Agreement at n = 12, L = 2

The overlap of PR #46, PR #50, PR #51, and the locked B = 2 seal (PR #57, reproduced by PR #59) is the locked cell at frozen n = 12. All five rows sit inside 0.01 of each other.

| source | arm | Medε SNAP | Medε Fourier | R | Λ |
|:-------|:----|----------:|-------------:|------:|------:|
| #46 bakeoff cell C | B = 4 | 0.4264 | 0.6876 | 0.6201 | 0.7477 |
| #50 depth sweep, L = 2 | B = 4 | 0.4264 | 0.6876 | 0.6201 | 0.7477 |
| #51 n = 18 seal | B = 4 | 0.4271 | 0.6879 | 0.6209 | 0.7477 |
| #57 and #59 | B = 2 | 0.4261 | 0.6876 | 0.6197 | 0.7477 |
| spot check, seeds 0–2 median | B = 4 | 0.4264 | 0.6876 | 0.6201 | 0.7477 |

The B = 4 seal stream matches the TF-only anchor in PR #45 (n = 12 Λ = 0.7477, gap = 0.1265, L = 2 R = 0.6209, L = 4 R = 0.8881, all Δ = 0). The bakeoff and the depth sweep share one batch stream. Those two B = 4 streams differ by at most 0.0008 in R. PR #59 matches PR #57 at n = 12 with all deltas 0. The B = 2 seal and the B = 4 reference differ by 0.0010 in Medε, 0.0003 in Fourier Medε, and 0.0012 in R.

## Spot check (this dossier)

This check reruns the PR #50 trainer, the B = 4 reference arm. The locked B = 2 n = 12 row is the PR #59 reproduction of PR #57 (all deltas 0) and was not rerun here.

The PR #50 trainer (`Nov_3rd/scripts/run_new_safe_L_sweep.py` on `cursor/new-safe-l-sweep-445d`) was rerun for L = 2, seeds 0, 1, and 2 only. Same packs (`beat_qaoa_12h/train_packs`), same budget, λ = 0.75, μ = 0. Ground states were recomputed (dense eigh at n = 6 and n = 9, ARPACK at n = 12). Seeds 3 and 4, L ≠ 2, n = 15, and n = 18 were not rerun. n = 21 was not loaded.

Recomputed median Λ is 0.7612 (n = 6, 250 instances), 0.7555 (n = 9, 250 instances), and 0.7477 (n = 12, 128 instances), matching the PR #50 ground-state table at four decimals.

| seed | Medε SNAP | Medε Fourier | R | SNAP best round |
|-----:|----------:|-------------:|------:|----------------:|
| 0 | 0.4264 | 0.6876 | 0.6202 | 35 |
| 1 | 0.4243 | 0.6878 | 0.6169 | 32 |
| 2 | 0.4264 | 0.6876 | 0.6201 | 31 |

Each seed matches the PR #50 L = 2 seed table at the printed four decimals, including the SNAP checkpoint round. The three-seed median equals the published B = 4 five-seed median (R = 0.6201, Λ = 0.7477) because that median seed sits inside {0, 1, 2}. Against PR #51 the same median differs by 0.0007 in Medε SNAP, 0.0003 in Medε Fourier, and 0.0008 in R. Against the locked B = 2 seal the same median differs by 0.0003 in Medε, 0.0000 in Fourier Medε, and 0.0004 in R.

## Round 14

**Status at Round 14:** paper lock unchanged (then the B = 4 seal); breadth still open (sparse p3 in flight elsewhere).

Round 14 recorded two outcomes and left the lock tables on the B = 4 seal. Round 16 later retargets the frozen-size and disorder tables to B = 2. n = 21 was not loaded. Nothing in this append was retrained.

### Cost-only ablation — PASS (PR #53)

Source: PR #53, `Nov_3rd/runs/quantum_h_explore/cost_only_ablation_NEW_SAFE/RESULTS.md`. Separate protocol from the seal: Adam then L-BFGS, checkpoint chosen on a disjoint n ∈ {6, 9} val slice, frozen primary n = 12 equal to indices [0:64] of `train_n12.npz`.

The motif is two gates. The cost layer is the BinH conjugate of a diagonal Walsh pulse,

\[
U_c^X = W \exp\Bigl(-i\sum_{w=1}^{4}\gamma_{\ell,w} E_w\Bigr) W, \qquad W = H^{\otimes 3}
\]

per cavity. The mixer is the shared Hamming-shell SNAP on popcount \(k = 0..3\), \(\varphi_k = c_1 k + c_2 k^2 + c_3 k^3\), and a final \(W\) reads bitstrings. \(U_c^X\) is that cost pulse. The shell is the SNAP. Equal Walsh weights with \(c_2 = c_3 = 0\) and \(c_1 = -\beta\) reproduce the repo mixer \(\exp(-i\beta X/2)\).

\(U_c^X\) alone freezes the shell at 0. Every diagonal waveform then collapses to a phase on \(|+\rangle^{\otimes n}\), and the lowest energy in that family is \(|+\rangle\) itself (\(\langle E_{\mathrm{unsat}}\rangle = \mathrm{mean}(E)\), \(\sum_i\langle X_i\rangle = n\)). SNAP-only freezes \(\gamma = 0\). Each layer returns the register to \(|0\rangle\) before the Z-diagonal shell, the shell contributes a global phase, and the closing \(W\) prepares \(|+\rangle\). On primary n = 12 both arms sit on that analytic row: Medε = 2.3598 (≈ 2.36). Trained cost-only seeds land there; the summed waveform has standard deviation about \(10^{-5}\).

The full motif is required. Val-selected L = 2 on primary n = 12 gives Medε (Walsh SNAP + \(U_c^X\)) / cost-only / SNAP-only / Fourier = 0.4279 / 2.3598 / 2.3598 / 0.7290. Key ratios: \(r_B \approx 0.18\) (0.181) and \(r_D \approx 0.59\) (0.587), with \(r_C = 0.181\). PASS is every ratio ≤ 0.85. The same protocol at L = 4 also passes (\(r_B = 0.077\), \(r_D = 0.565\)) and stays outside the locked sentence. At the time of Round 14 the paper quote was the B = 4 seal, R = 0.6209 / 0.6016 / 0.5880 at n = 12 / 15 / 18. Round 16 moves that quote to the B = 2 seal (PR #59, R = 0.6197 / 0.5979 / 0.5828). The ablation Fourier Medε 0.7290 is the 64-instance slice. The B = 4 seal Fourier Medε is 0.6879 on 128 instances; the locked B = 2 seal Fourier Medε is 0.6876. The QAOA-dual checkpoint inside the full motif matches Fourier. The remaining gap is Walsh grading of \(U_c^X\) together with the nonlinear shell. A diagonal GRAPE schedule of \(H_c\) is the cost-only family, whose best state is \(|+\rangle\).

### Signed 3-local — KILL (PR #54)

Source: PR #54, `Nov_3rd/runs/quantum_h_explore/signed_3local_tf_breadth/RESULTS.md`. Shortlist family D, μ = 0, a separate Hamiltonian:

\[
H = \sum_c J_c Z_i Z_j Z_k - \lambda \sum_i X_i, \qquad J_c\in\{+1,-1\}.
\]

Cavity Λ in that smoke is the largest eigenvalue of the 3-qubit reduced ground state (a cavity-product state has Λ = 1). The NEW_SAFE column above is a different quantity, \(\langle(\sum_{i\in c} X_i/3)^2\rangle\). At λ ∈ {0.50, 0.75, 1.00} this reduced-state Λ is ≈ 1.0 (n = 12 medians 1.000 / 1.000 / 1.000) and the product gap per qubit is ~0 (n = 12 medians 0.0001 / 0.0003 / 0.0009). The quantumness gate asked for median Λ ≤ 0.80 and gap/n ≥ 0.03 at n = 6, 9, and 12. Every scored λ misses both. L = 2 Walsh/Fourier R at n = 12 is 1.013 / 1.012 / 1.010, all above 0.98. KILL. Signed 3-local is not a breadth family.

Power-law (PR #42) is already soft. The non-claims section records it as needs_edit: frozen n = 12 R_Fourier = 0.937 at λ = 0.50, μ = −0.50, L = 4, every seed above 0.90. It stays out of the lock.

## Round 15

**Status at Round 15:** paper lock unchanged (still the B = 4 seal); B=2 compression OK; breadth (sparse p3) closed as KILL; power-law (#42) still soft only.

Round 15 recorded two outcomes and left the lock tables on the B = 4 seal. n = 21 was not loaded. Nothing in that append was retrained. The quote at the end of Round 15 was λ = 0.75, μ = 0, L = 2, with the B = 4 seal ratios R = 0.6209 / 0.6016 / 0.5880 at n = 12 / 15 / 18. Round 16 moves the paper quote to B = 2. PR #58 is this record.

### Walsh B-ablation — PASS (PR #57)

Source: PR #57, `Nov_3rd/runs/quantum_h_explore/NEW_SAFE_B_ablation/RESULTS.md`. Same NEW_SAFE Hamiltonian, packs, and L-BFGS budget as the depth sweep. B counts free SNAP phase coefficients of the shared cavity mixer. Every SNAP arm keeps four Walsh weights on degrees 1..4. Fourier is the matched baseline. The hardware-relevant SNAP for the L = 2 claim is B = 2 (two coefficients). B = 4 and the full cavity are equivalent to it at L = 2.

| arm | phases per layer | span |
|:----|-----------------:|:-----|
| B = 2 | 2 | \(\varphi_k = c_1 k + c_2 k^2\) on popcount \(k = 0..3\). Paper init \(\varphi_k = -\beta k\) sits in this span. |
| B = 4 | 4 | free \(\varphi_0..\varphi_3\). This is the four-phase Hamming shell. It reproduces PR #50 and is the PR #51 reference. |
| full | 8 | one phase per Fock state of the 3-bit cavity. |

Frozen n = 12, five seeds. Median Λ of the ground state is 0.7477 at every arm.

| L | arm | Medε | Medε Fourier | R | Medε / Medε(full) | paired SNAP < Fourier | call |
|--:|:----|-----:|-------------:|------:|------------------:|----------------------:|:-----|
| 2 | B = 2 | 0.4261 | 0.6876 | 0.6197 | 1.000 | 640/640 | win |
| 2 | B = 4 | 0.4264 | 0.6876 | 0.6201 | 1.001 | 640/640 | win |
| 2 | full | 0.4261 | 0.6876 | 0.6196 | 1.000 | 640/640 | win |
| 4 | B = 2 | 0.1958 | 0.3172 | 0.6174 | 0.965 | 640/640 | win |
| 4 | B = 4 | 0.2755 | 0.3172 | 0.8687 | 1.358 | 581/640 | soft |
| 4 | full | 0.2030 | 0.3172 | 0.6399 | 1.000 | 640/640 | win |

At L = 2, B = 2 matches the full shell (R ≈ 0.620, Medε 1.000× full) and is 0.999× the B = 4 shell. 5/5 seeds have R ≤ 0.85. B = 4 reproduces PR #50 at four decimals (Medε 0.4264 / 0.6876, R = 0.6201, Λ = 0.7477, all deltas 0). PASS is full R ≤ 0.85, Medε(B = 2) ≤ 1.3 × Medε(full), and B = 2 itself at R ≤ 0.85. All three hold.

L = 4 is soft for B = 4 only (R = 0.8687, 2/5 seeds ≤ 0.85). B = 2 and full are wins there. That depth stays outside the locked sentence. n = 15 and n = 18 were not resealed in #57. PR #59 later seals those sizes at B = 2.

### Sparse p = 3 TFIM — KILL (PR #56)

Source: PR #56, `Nov_3rd/runs/quantum_h_explore/sparse_p3_tfim_breadth/RESULTS.md`. Shortlist family B, μ = 0, a separate Hamiltonian:

\[
H = -\sum_{(i,j,k)\in T}\frac{J_{ijk}}{s}\, Z_i Z_j Z_k - \lambda \sum_i X_i,
\qquad s=\sqrt{|T|/n},\quad J_{ijk}\in\{+1,-1\}.
\]

Cavity Λ in that smoke is the largest eigenvalue of the 3-qubit reduced ground state. The NEW_SAFE column above is \(\langle(\sum_{i\in c} X_i/3)^2\rangle\). No scored λ ∈ {0.50, 1.00, 1.50, 2.00} clears the cavity line (median Λ ≤ 0.80 and product gap/n ≥ 0.03 at n = 6, 9, and 12). The Λ curve bottoms at λ = 1. Scored medians there are 0.829 / 0.823 / 0.927 at n = 6 / 9 / 12. The floor quoted from that smoke is min Λ ≈ 0.829 at λ = 1 (the n = 6 cell, and the same 0.829 on the 8-instance diagnostic). The lowest full-val cell is 0.823 at n = 9, above 0.80. λ = 1.50 and 2.00 open a product gap and still sit above Λ = 0.87.

L = 2 Walsh stays on the Fourier mixer. Median-seed R at n = 12 is 0.999 / 0.999 / 1.021 / 1.029 for λ = 0.50 / 1.00 / 1.50 / 2.00, and train loss moves by at most 0.001 after the Fourier warm start. Walsh ≈ Fourier. KILL. Sparse p = 3 is not a breadth family.

Power-law (PR #42) is still soft only. The non-claims section records it as needs_edit: frozen n = 12 R_Fourier = 0.937 at λ = 0.50, μ = −0.50, L = 4, every seed above 0.90. It stays out of the lock.

## Round 16

**Status:** paper lock prefers B = 2, \(\varphi_k = c_1 k + c_2 k^2\). The frozen-size table is the PR #59 seal. The disordered-λ table is PR #60. B = 4 / PR #51 stays the equivalent reference. No retrain. n = 21 was not loaded.

PR #58 already recorded Round 15 (B = 2 compression at frozen n = 12, and the sparse-p3 KILL). Round 16 promotes that compression to the paper quote once the same two coefficients hold through n = 18 and under the disorder of #49. The depth-sweep losses at L ≥ 8, the old-SAFE Λ appendix, and the Round 14 / 15 breadth kills stay as recorded. Result trees from #59 and #60 are not copied onto this branch.

### B = 2 seal through n = 18 — PASS (PR #59)

Source: PR #59, `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_B2_n18/RESULTS.md`. Five seeds. Train \(n\in\{6,9\}\). Freeze n = 12 (128 instances), probe n = 15 (64) and n = 18 (32). L = 2 is the gate. Phases are \(\varphi_k = c_1 k + c_2 k^2\) on popcount, shared across cavities. The n = 12 row matches PR #57 at four decimals (Medε 0.4261 / 0.6876, R = 0.6197, Λ = 0.7477, all Δ = 0). Ground-state Λ matches the PR #51 table at every size. Holdout R is 0.6061 at n = 6 and 0.6248 at n = 9.

| n | Medε B = 2 | Medε #51 B = 4 | ratio | R | R #51 | ΔR | median Λ |
|--:|-----------:|---------------:|------:|------:|------:|----:|---------:|
| 12 | 0.4261 | 0.4271 | 0.998 | 0.6197 | 0.6209 | −0.0012 | 0.7477 |
| 15 | 0.5043 | 0.5066 | 0.995 | 0.5979 | 0.6016 | −0.0037 | 0.7489 |
| 18 | 0.6298 | 0.6352 | 0.991 | 0.5828 | 0.5880 | −0.0052 | 0.7487 |

PASS bars, all held. n = 12 L = 2 R = 0.6197 ≤ 0.85, and 5/5 seeds are at or under 0.85. Paired B = 2 is below Fourier on 640/640, 320/320, and 160/160 instances. At each of n = 12, n = 15, and n = 18, Medε is within 1.05× of the PR #51 L = 2 Medε (ratios 0.998 / 0.995 / 0.991) and |ΔR| ≤ 0.02 against 0.6209 / 0.6016 / 0.5880. Medε stays at or under the #51 B = 4 column, and the ratio does not grow with n. Median Λ is 0.7477 / 0.7489 / 0.7487, all ≤ 0.80.

L = 4, same budget, is a win and sits outside the locked sentence. R is 0.6174 / 0.6252 / 0.6067. Medε is 0.695 / 0.721 / 0.694 times the PR #51 L = 4 shell. The n = 12 L = 4 row matches PR #57 (Medε 0.1958, R = 0.6174, Δ = 0).

### Disordered λ at B = 2 — PASS (PR #60)

Source: PR #60, `Nov_3rd/runs/quantum_h_explore/disordered_lambda_NEW_SAFE_B2/RESULTS.md`. Same Hamiltonian and protocol as #49 (seed 20261105, δ ∈ {0.15, 0.30}, μ = 0), with the mixer replaced by \(\varphi_k = c_1 k + c_2 k^2\). Fourier is matched. Λ is the ground state and does not depend on B.

| δ | L = 2 Medε B = 2 | L = 2 Medε Fourier | L = 2 R | n = 12 median Λ | #49 shell R | 5/5 | call |
|--:|-----------------:|-------------------:|--------:|----------------:|------------:|:----|:-----|
| 0.15 | 0.4143 | 0.6772 | 0.6118 | 0.7586 | 0.6130 | 5/5 | PASS |
| 0.30 | 0.4121 | 0.6694 | 0.6156 | 0.7610 | 0.6167 | 5/5 | PASS |

Both pass bars hold: L = 2 R ≤ 0.85 and median Λ ≤ 0.80. Fourier reproduces the #49 numbers at four decimals, and the Λ table matches #49 (0.7586 / 0.7610). B = 2 Medε sits within 0.001 of the shell (0.4143 versus 0.4151, 0.4121 versus 0.4128). The closest Λ is still the n = 6 pack at δ = 0.30 (0.7956). L = 4 is outside the gate. B = 2 stays a win there (R = 0.6032 / 0.6398, 5/5), while the #49 shell was softer (0.8249 / 0.9699).

## Source index

| PR | what it locks | result directory |
|---:|:--------------|:-----------------|
| #45 | TF-only λ scan; n = 12 anchor at λ = 0.75 | `Nov_3rd/runs/quantum_h_explore/tf_only_shallow_Lambda/` |
| #46 | bakeoff; NEW_SAFE = cell C | `Nov_3rd/runs/quantum_h_explore/bakeoff_Lambda_safe_top3/` |
| #49 | disordered λ, B = 4 shell; equivalent reference for #60 | `Nov_3rd/runs/quantum_h_explore/disordered_lambda_NEW_SAFE/` |
| #50 | R(L) at frozen n = 12 on the B = 4 shell; L ≥ 8 losses | `Nov_3rd/runs/quantum_h_explore/NEW_SAFE_L_sweep/` |
| #51 | five-seed B = 4 seal through n = 18; equivalent reference for the locked B = 2 seal | `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_lam075_mu0_n18/` |
| #41 | old SAFE (0.75, −0.50) Λ-kill appendix | `Nov_3rd/runs/quantum_h_explore/n18_shallow_lambda_seal/` |
| #42 | power-law family, needs_edit, not this claim | `Nov_3rd/runs/quantum_h_explore/powerlaw_tfim_breadth/` |
| #53 | cost-only ablation, PASS; full motif required; lock tables unchanged | `Nov_3rd/runs/quantum_h_explore/cost_only_ablation_NEW_SAFE/` |
| #54 | signed 3-local breadth, KILL; not a breadth family | `Nov_3rd/runs/quantum_h_explore/signed_3local_tf_breadth/` |
| #56 | sparse p=3 TFIM breadth, KILL; no λ clears cavity Λ (min Λ ≈ 0.829 at λ = 1); Walsh ≈ Fourier | `Nov_3rd/runs/quantum_h_explore/sparse_p3_tfim_breadth/` |
| #57 | Walsh B-ablation, PASS at n = 12; B = 2 matches full (R = 0.6197, Medε 1.000× full). Round 16 promotes that arm | `Nov_3rd/runs/quantum_h_explore/NEW_SAFE_B_ablation/` |
| #59 | locked five-seed B = 2 seal through n = 18, PASS; R = 0.6197 / 0.5979 / 0.5828, Λ ≈ 0.75; Medε ≤ #51 | `Nov_3rd/runs/quantum_h_explore/seal_NEW_SAFE_B2_n18/` |
| #60 | disordered λ at B = 2, PASS; δ = 0.15 / 0.30, R = 0.6118 / 0.6156, Λ matches #49 | `Nov_3rd/runs/quantum_h_explore/disordered_lambda_NEW_SAFE_B2/` |

Those directories live on the PR branches above. This branch adds only this dossier. Round 14 appends #53 and #54; Round 15 appends #56 and #57; Round 16 retargets the paper quote to the B = 2 seal (#59) and the B = 2 disorder table (#60), and keeps #51 / #49 as the equivalent reference. None of these rounds copies those result trees. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written. n = 21 was not loaded, and nothing was retrained.
