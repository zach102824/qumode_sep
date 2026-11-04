# L=16 SNAP rescue — SAFE point

**Overall: needs_edit.**

Deep qubit HVA wins; paper = shallow SNAP. No pre-specified SNAP arm put frozen-n12 Med ε at or below the matched L=16 HVA.

Same Hamiltonian and the same freeze rule as the PR #36 fairness baseline. Train n∈{6,9} only. n12 and n15 never enter the loss. n21 was not read or written.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j
\]

SAFE point **λ=0.75, μ=−0.50**. ε = ⟨H⟩ − E_0. **Med ε** is the median over seeds of each seed's median-over-instances ε. Readout is the Z histogram plus the X histogram. No Y.

## What was trained

PR #36 at the seal budget (40 × max_iter 8) had L=16 n12 Med ε HVA 0.0144 and Walsh SNAP 0.0328 (R=2.2751). One extra 40 rounds on seed 0 narrowed that seed to R=1.49 and still left HVA ahead. This run is the rescue: a longer matched budget, a shallow warm start, and power_p.

| arm | family | L=16 L-BFGS rounds | starts from |
|---|---|---:|---|
| `hva_long` | hva | 120 | PR36 L=16 HVA checkpoint, then 120 L-BFGS rounds |
| `walsh_long` | snap | 120 | PR36 L=16 Walsh SNAP checkpoint, then 120 L-BFGS rounds |
| `power_p` | snap | 160 | scalar γ + a_n polynomial shell, 160 rounds from the paper init |
| `warm_pad` | snap | 120 | L=4 Walsh SNAP, identity on the other 12 layers, then 120 rounds |
| `warm_tile` | snap | 120 | L=4 Walsh block repeated ×4, then 120 rounds |

power_p uses φ_k = c1·k + a_n (c2·k² + c3·k³) with a_n = (8/(n−1))^p (a_9 = 1) and one scalar γ per layer. c2 = c3 = p = 0 at init, so the paper init is the linear shell, the same state as Fourier. No penalty on c2, c3, or p.

`warm_pad` copies the seed's own PR #36 L=4 Walsh angles into layers 0..3 and sets layers 4..15 to γ=0, φ=0. At that init the circuit is exactly the L=4 SNAP. `warm_tile` repeats the same 4-layer block four times.

HVA angles map into the Walsh shell exactly: equal γ_w reproduce U_C(γ), and φ_k = −β^X k + (β^XX / 2) ev_XX(k) with ev_XX = (3, −1, −1, 3) reproduces U_XX U_X, because both factors are popcount-diagonal on a 3-qubit cavity and therefore commute. The shell also has the XXX direction (it can split popcount 1 from popcount 2, which XX cannot). The rescue does not start SNAP from that embedded HVA point. Doing so would copy the baseline into the SNAP column. The map is only an expressivity check.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train [0:200) on n=6 and n=9. Holdout [200:250) selects the checkpoint by n9 Med ε. Freeze n12 first 128, n15 first 64.
- Optimizer, unchanged from `Nov_3rd.settings`: L-BFGS lr 0.25, max_iter 8, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = 0.5 (mean ε/n on n6 + mean ε/n on n9). No Adam.
- Round r draws its batch from `default_rng((seed, r, 16))`. The same (seed, r) hits every arm, and it does not replay the seal's 40 batches.
- Seeds {0,1,2}. The reported Med ε is the median of the three seeds.
- E0 values are the PR #36 caches (same instances, same λ, μ).
- n21 phase-2 ladder was not read or written.

## Gate

- **PASS** — some pre-specified SNAP arm has n12 Med ε ≤ matched `hva_long`. Three seeds. R ≤ 0.98 is recorded separately and is not required for PASS.
- **needs_edit** — every SNAP arm is still above `hva_long` at n12. Deep qubit HVA wins; paper = shallow SNAP.
- The seal's 0.85 / KILL labels are not reused. This round asks only whether SNAP can get back in front of HVA at L=16.

## Frozen n12

| arm | Medε n6 | Medε n9 | Medε n12 | Medε n15 | R vs hva_long | ≤0.98 | verdict |
|---|---:|---:|---:|---:|---:|:---:|---|
| seal HVA (40 rnd, not retrained) | 0.00636 | 0.00882 | 0.01440 | 0.01730 |  |  | reference |
| seal Walsh (40 rnd, not retrained) | 0.01976 | 0.02281 | 0.03276 | 0.03723 | 2.2751 vs seal | no | KILL in PR #36 |
| `hva_long` | 0.00247 | 0.00374 | 0.00798 | 0.01093 | 1 |  | matched baseline |
| `walsh_long` | 0.00444 | 0.00503 | 0.00803 | 0.00924 | 1.0059 | no | needs_edit |
| `power_p` | 0.01022 | 0.01281 | 0.01882 | 0.02156 | 2.3586 | no | needs_edit |
| `warm_pad` | 0.01092 | 0.01237 | 0.01947 | 0.02350 | 2.4404 | no | needs_edit |
| `warm_tile` | 0.00304 | 0.00350 | 0.02834 | 0.05139 | 3.5522 | no | needs_edit |

R vs hva_long uses this run's HVA, not the 40-round seal. The seal row is the unmatched shorter budget.

## Seeds

| arm | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |
|---|---:|---:|---:|---:|---:|---:|
| `hva_long` | 0 | 106 | 0.00376 | 0.00798 | 0.01093 | 100 |
| `hva_long` | 1 | 119 | 0.00373 | 0.00774 | 0.01060 | 99 |
| `hva_long` | 2 | 117 | 0.00374 | 0.00835 | 0.01175 | 96 |
| `walsh_long` | 0 | 116 | 0.00542 | 0.00861 | 0.00994 | 71 |
| `walsh_long` | 1 | 118 | 0.00486 | 0.00775 | 0.00897 | 70 |
| `walsh_long` | 2 | 117 | 0.00503 | 0.00803 | 0.00924 | 71 |
| `power_p` | 0 | 159 | 0.01242 | 0.01849 | 0.02156 | 49 |
| `power_p` | 1 | 159 | 0.01281 | 0.01882 | 0.02173 | 50 |
| `power_p` | 2 | 158 | 0.01299 | 0.01900 | 0.02156 | 50 |
| `warm_pad` | 0 | 118 | 0.01210 | 0.01923 | 0.02346 | 75 |
| `warm_pad` | 1 | 119 | 0.01237 | 0.01947 | 0.02350 | 75 |
| `warm_pad` | 2 | 117 | 0.01328 | 0.02148 | 0.02644 | 73 |
| `warm_tile` | 0 | 116 | 0.00350 | 0.02834 | 0.05139 | 72 |
| `warm_tile` | 1 | 119 | 0.00370 | 0.02785 | 0.05040 | 74 |
| `warm_tile` | 2 | 119 | 0.00345 | 0.03965 | 0.07116 | 73 |

n12 seed values (seed 0, 1, 2):

- `hva_long`: 0.00798, 0.00774, 0.00835
- `walsh_long`: 0.00861, 0.00775, 0.00803
- `power_p`: 0.01849, 0.01882, 0.01900
- `warm_pad`: 0.01923, 0.01947, 0.02148
- `warm_tile`: 0.02834, 0.02785, 0.03965

## Did the extra budget flatten?

- `hva_long`: s0 r79→r119 0.00409→0.00389 (end lower), best r106; s1 r79→r119 0.00476→0.00373 (end lower), best r119; s2 r79→r119 0.00432→0.00396 (end lower), best r117
- `walsh_long`: s0 r79→r119 0.00718→0.00562 (end lower), best r116; s1 r79→r119 0.00661→0.00486 (end lower), best r118; s2 r79→r119 0.00638→0.00531 (end lower), best r117
- `power_p`: s0 r119→r159 0.01502→0.01242 (end lower), best r159; s1 r119→r159 0.01454→0.01281 (end lower), best r159; s2 r119→r159 0.01528→0.01301 (end lower), best r158
- `warm_pad`: s0 r79→r119 0.01570→0.01214 (end lower), best r118; s1 r79→r119 0.01770→0.01237 (end lower), best r119; s2 r79→r119 0.01839→0.01340 (end lower), best r117
- `warm_tile`: s0 r79→r119 0.00418→0.00351 (end lower), best r116; s1 r79→r119 0.00462→0.00370 (end lower), best r119; s2 r79→r119 0.00435→0.00345 (end lower), best r119

## Where the arms started

n12 Med ε of the init, before the new rounds (median over seeds when stored):

- `hva_long`: init n9 Medε 0.0088, init n12 Medε 0.0144
- `walsh_long`: init n9 Medε 0.0228, init n12 Medε 0.0328
- `power_p`: init n9 Medε 0.8044, init n12 Medε 1.1001
- `warm_pad`: init n9 Medε 0.0285, init n12 Medε 0.0394
- `warm_tile`: init n9 Medε 1.7296, init n12 Medε 2.5608

## Did the extra generators move?

- `walsh_long` mean RMS of φ after removing a+bk (0 = X-only mixer): 0.10351
- `warm_pad` mean RMS of φ after removing a+bk (0 = X-only mixer): 0.11078
- `warm_tile` mean RMS of φ after removing a+bk (0 = X-only mixer): 0.23731
- `hva_long` mean |β^XX|: 0.14420
- `power_p` p per seed (0.020, 0.016, 0.020), mean |c2|=0.0977, mean |c3|=0.0293

## Reading

The closest SNAP arm is `walsh_long`. Frozen n12 Med ε is 0.00803 against matched HVA 0.00798 (R=1.0059, gap 4.689e-05). R is above 1, so the arm also misses R≤0.98.

Walsh n12 seeds span 0.00775–0.00861; HVA spans 0.00774–0.00835. The ranges overlap. The gate uses the median, and that median is on the HVA side.

The same Walsh checkpoints versus the same HVA: n6 R=1.799, n9 R=1.347, n12 R=1.006, n15 R=0.845 (n15 Med ε Walsh 0.00924, HVA 0.01093). HVA leads on the trained sizes and on the n12 gate. Walsh leads at n15. The assigned PASS rule is n12.

The last iterate, which is not the checkpoint rule, has n12 medians Walsh 0.00794 and HVA 0.00779 (R=1.0184). The median stays on the HVA side.

power_p, warm_pad, and warm_tile remain near R≈2.4–3.6. power_p keeps p near 0, so a_n stays near 1, and its n9 holdout is still falling at round 159 without getting close to HVA. warm_pad starts exactly at the L=4 SNAP solution and improves, and after 120 rounds it is still about twice the matched HVA. warm_tile drives n9 down to the HVA range and does not transfer (n12 and n15 stay several times higher).

The Hamming shell contains the HVA mixer, and equal Walsh weights contain the HVA cost. `embed_check.json` copies the seal HVA seed-0 angles into that shell and reproduces its frozen-n12 Med ε. The L-BFGS trajectories from the paper init, from the seal SNAP checkpoint, and from the L=4 warm start get Walsh within a percent of the matched HVA at n12 and do not put the median under it. Under this rescue the n12 HVA lead remains.

**Deep qubit HVA wins; paper = shallow SNAP.** Lead with L=2 and L=4, where PR #36 measured n12 R=0.328 and R=0.536 for Walsh SNAP against this HVA. The L=16 n12 median does not support a SNAP win on this Hamiltonian.

Per-seed JSON: `{arm}_seed{s}.json` in this folder. `angles` is the restored n9-holdout checkpoint. `final_angles` is the last iterate.
