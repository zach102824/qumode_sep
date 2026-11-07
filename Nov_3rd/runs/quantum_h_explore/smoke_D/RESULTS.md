# RESULTS — quantum H explore, family D smoke

Signed 3-local Ising plus a transverse field, on the locked Nov_3rd gate sequence.
Metric is the median excess energy per qubit ε = (⟨H⟩ − E₀) / n. Smaller is better.
BEAT means Med ε(power_p) ≤ 0.98 × Med ε(Fourier QAOA) on that (n, L).

**Verdict: KILL**

KILL. No λ on the grid passes the quantumness gate on both n, and the best-scoring λ (3.0) also fails the train and holdout checks (n=6 failed [prod/bit 0.113]; n=9 failed [prod/bit 0.054]). At that λ the ground state is not a bitstring (peak about 0.2, and (E_bit−E₀)/n is O(1)), but a product state already captures most of the drop below the best bitstring: the prod/bit ratio is under the 0.15 bar (about 0.11 at n=6 and 0.05 at n=9 on holdout). The clause packing is dense (m≈10n), so this signed 3-spin glass is mean-field. Classical frustration is real (about 36–40% of triples unhappy; classical degeneracy typically 2) and still does not produce an entangled ground state. Do not open a train{6,9} ladder on family D. If a transverse-field smoke continues, family A’s unsat diagonal has the larger product-state residual at moderate λ.

## Hamiltonian

```
H = Σ_c J_c Z_i Z_j Z_k  −  λ Σ_i X_i ,   λ > 0
```

- Support (i, j, k): the first three literals of each scale1000 clause (`cls_v`), dropping the 4th sorted variable.
- J_c = product of the three kept `cls_s` signs, so J_c ∈ {±1}. Redraw was not needed.
- Z|0⟩ = +|0⟩, Z|1⟩ = −|1⟩, variable 0 is the MSB. This is the E1 sign convention (field lowers |+⟩).
- λ = **3.0** is the pilot λ with the largest product-state gap (indices 192:224), then checked on train 0:64 and holdout 64:192. It is the diagnostic point, not a passing gate.
- Packs: n=6 `/agent/repos/qumode_sep/explore_OCT29/data/scale1000/data_n6.npz`; n=9 `sat_gen SCB=54000000000 prefix 224`.
- Gate sequence unchanged: |0⟩ → [W exp(−iγ H_c) W → Hamming-shell SNAP]^L → W, W = H^{⊗3} per cavity.
- The diagonal pulse is the physical H_c (a constant shift would be a global phase). γ is in the same units as the QAOA cost angle.
- ⟨H⟩ uses only the Z histogram of H_c and the n single-qubit ⟨X_i⟩. No YY or XX correlators.
- Train {6, 9} jointly (equal mean-ε weight). QAOA is per-layer (β, γ), paper init β=0.01, γ=−0.01, mixer exp(−i β X/2), L-BFGS. power_p warms from the trained QAOA angles (γ ← γ, φ_k ← −β k, then c2, c3, p).
- Seeds {0, 1, 2}. Seed 0 is the paper point; seeds 1 and 2 add a small Gaussian kick. The reported seed is the one with the best holdout average of Med ε(n=6) and Med ε(n=9). Within a seed, a polish that loses to its warm start on the **train** objective is discarded.
- phase2 n21, `settings.py`, `qaoa_angles/`, and train69 ladder outputs were not written.
- `opus_reply_20261104_2150_hamiltonian_families.md` is not in the repo. Family D is taken from `SHORTLIST.md` (same 2026-11-04 ~21:50 pass) plus the signed E1 field in the task statement.

## Quantumness gate and λ

Pre-registered pass, on one split, both required for n=6 and n=9:

- median GS bitstring peak < 0.5, and at least 80% of instances below that peak
- median (E_bit − E₀)/n ≥ 0.02 (no computational-basis state is the ground state)
- median (E_prod − E₀)/n ≥ 0.01
- median (E_prod − E₀)/(E_bit − E₀) ≥ 0.15 (product states miss at least 15% of the drop below the best bitstring)

E_prod is a variational upper bound: multi-start L-BFGS on Bloch angles θ∈[0, π] (so ⟨X⟩≥0, matching −λX), coordinate-descent polish, and the explicit |+⟩ and best-bitstring product states.

Pilot scan (family D):

| λ | n | med peak | frac peak<0.5 | med (E_bit−E₀)/n | med (E_prod−E₀)/n | med prod/bit | pass |
|---|---|----------|---------------|------------------|-------------------|--------------|------|
| 0.00 | 6 | 1.000 | 0.00 | 0.0000 | 0.0000 | nan | no |
| 0.00 | 9 | 1.000 | 0.00 | 0.0000 | 0.0000 | nan | no |
| 0.25 | 6 | 0.498 | 1.00 | 0.0457 | 0.0000 | 0.000 | no |
| 0.25 | 9 | 0.497 | 1.00 | 0.0325 | 0.0000 | 0.000 | no |
| 0.50 | 6 | 0.493 | 1.00 | 0.0998 | 0.0001 | 0.001 | no |
| 0.50 | 9 | 0.488 | 1.00 | 0.0745 | 0.0001 | 0.001 | no |
| 0.75 | 6 | 0.484 | 1.00 | 0.1628 | 0.0004 | 0.002 | no |
| 0.75 | 9 | 0.474 | 1.00 | 0.1257 | 0.0003 | 0.003 | no |
| 1.00 | 6 | 0.468 | 1.00 | 0.2355 | 0.0013 | 0.005 | no |
| 1.00 | 9 | 0.455 | 1.00 | 0.1860 | 0.0010 | 0.005 | no |
| 1.50 | 6 | 0.413 | 1.00 | 0.4158 | 0.0083 | 0.020 | no |
| 1.50 | 9 | 0.410 | 1.00 | 0.3328 | 0.0037 | 0.011 | no |
| 2.00 | 6 | 0.327 | 1.00 | 0.6567 | 0.0401 | 0.062 | no |
| 2.00 | 9 | 0.357 | 1.00 | 0.5097 | 0.0110 | 0.022 | no |
| 3.00 | 6 | 0.119 | 1.00 | 1.3313 | 0.1986 | 0.146 | no |
| 3.00 | 9 | 0.172 | 1.00 | 1.0348 | 0.0695 | 0.066 | no |

Pilot quantumness (both n): **FAIL**. Confirm on train: **FAIL**. Confirm on holdout: **FAIL**.

Holdout confirmation at the chosen λ (family D):

| n | med peak | med (E_bit−E₀)/n | med (E_prod−E₀)/n | med prod/bit | med ⟨X⟩/n | med (E₁−E₀)/n |
|---|----------|------------------|-------------------|--------------|-----------|----------------|
| 6 | 0.190 | 1.1517 | 0.1469 | 0.113 | 0.732 | 0.4155 |
| 9 | 0.198 | 0.9825 | 0.0521 | 0.054 | 0.562 | 0.2102 |

## Frustration vs family A

Family A on the same formulas is transverse Max-E, H = diag(unsat) − λ Σ X, same field sign. The packs are satisfiable, so A’s classical ground energy is 0 and the λ=0 ground state is a bitstring (a product state). Any quantumness in A is produced only by the transverse field, on top of an unfrustrated satisfying assignment (or a small degenerate set of them).

Family D is a different diagonal: each kept triple is a signed ZZZ, and a bitstring scores −1 on a term only when the three spins match that sign. Random signs on an overcomplete triple set leave a finite fraction of terms unhappy at the best bitstring. That residual is classical frustration. It is not the same object as Max-E unsat, which is driven to zero by the pack filter.

| n | split | D median unhappy-term fraction | D median classical degeneracy of H_c | D frac unique classical GS | A median nsol |
|---|-------|--------------------------------|------------------------|----------------------------|---------------|
| 6 | pilot | 0.400 | 2.0 | 0.00 | 2.0 |
| 6 | hold | 0.383 | 2.0 | 0.00 | 2.0 |
| 9 | pilot | 0.360 | 2.0 | 0.00 | 1.0 |
| 9 | hold | 0.360 | 2.0 | 0.00 | 2.0 |

Family A quantumness on the **same pilot and λ grid** (not a SNAP train; ED only):

| λ | n | med peak | med (E_bit−E₀)/n | med (E_prod−E₀)/n | med prod/bit | pass |
|---|---|----------|------------------|-------------------|--------------|------|
| 0.00 | 6 | 1.000 | 0.0000 | 0.0000 | nan | no |
| 0.00 | 9 | 1.000 | 0.0000 | 0.0000 | nan | no |
| 0.25 | 6 | 0.759 | 0.0337 | 0.0018 | 0.047 | no |
| 0.25 | 9 | 0.785 | 0.0357 | 0.0014 | 0.044 | no |
| 0.50 | 6 | 0.385 | 0.1392 | 0.0185 | 0.121 | no |
| 0.50 | 9 | 0.297 | 0.1446 | 0.0157 | 0.100 | no |
| 0.75 | 6 | 0.205 | 0.3125 | 0.0446 | 0.144 | no |
| 0.75 | 9 | 0.093 | 0.3163 | 0.0418 | 0.134 | no |
| 1.00 | 6 | 0.121 | 0.5222 | 0.0554 | 0.106 | no |
| 1.00 | 9 | 0.038 | 0.5211 | 0.0524 | 0.101 | no |
| 1.50 | 6 | 0.061 | 0.9768 | 0.0484 | 0.049 | no |
| 1.50 | 9 | 0.015 | 0.9758 | 0.0454 | 0.047 | no |
| 2.00 | 6 | 0.045 | 1.4532 | 0.0408 | 0.028 | no |
| 2.00 | 9 | 0.009 | 1.4543 | 0.0381 | 0.026 | no |
| 3.00 | 6 | 0.032 | 2.4294 | 0.0285 | 0.012 | no |
| 3.00 | 9 | 0.006 | 2.4319 | 0.0275 | 0.011 | no |

At the λ chosen for D (n=6: A peak 0.032, gap_prod 0.0285, pass=no; D peak 0.119, gap_prod 0.1986, pass=no; n=9: A peak 0.006, gap_prod 0.0275, pass=no; D peak 0.172, gap_prod 0.0695, pass=no). λ values where **both** n pass the same quantumness gate: A none, D none. A’s classical minimum is an unfrustrated satisfying assignment, so λ=0 is a bitstring. Moderate λ (around 0.75–1) is where A’s product-state residual is largest (pilot med prod/bit about 0.14 at n=6 and 0.13 at n=9), and the GS peak is already far below 1/2. D’s signs do create classical frustration — roughly 36–40% of the triples are unhappy at the best bitstring, and the classical ground level is typically twofold — but the transverse field resolves that degeneracy into a spin-coherent product state. Across the grid, D’s prod/bit ratio stays near 0 until λ=3, and even there n=9 is only about 0.07. Matched λ by λ, A’s product gap is larger than D’s for every λ in [0.5, 2]. Signing does not make a more quantum target than unsigned unsat. Family A was not circuit-trained here; this is the ED comparison the shortlist asked for before preferring A.

## Walsh-degree

Mean Walsh power fraction by weight (0, 1, 2, 3, 4, ≥5), averaged over 16 train instances of each n, then the two n’s: 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000.
Weight-3 fraction 1.000000. H_c is a sum of weight-3 characters, so a Walsh-degree γ (degrees 1..4) spans the same diagonal pulses as a single scalar γ. Walsh was not trained as a separate method; it is not an extra ansatz on this H.

## Smoke vs Fourier QAOA

Train B=64, holdout B=128, λ=3.0. R = Med ε(power_p) / Med ε(QAOA) at the jointly chosen seed. Product-state Med ε is the ED variational baseline on the same holdout (not a circuit).

| L | seed picked | Med ε QAOA n6 | Med ε QAOA n9 | Med ε power_p n6 | Med ε power_p n9 | R n6 | R n9 | Med ε product n6 | Med ε product n9 |
|---|-------------|----------------|---------------|------------------|------------------|------|------|------------------|------------------|
| 2 | 1 | 0.1691 | 0.3028 | 0.1260 | 0.2473 | 0.745 | 0.817 | 0.1469 | 0.0521 |
| 16 | 0 | 0.1327 | 0.2470 | 0.0715 | 0.1809 | 0.539 | 0.732 | 0.1469 | 0.0521 |

Per seed (holdout Med ε):

| L | seed | QAOA n6 | QAOA n9 | power_p n6 | power_p n9 | R n6 | R n9 | snap_off n6 | snap_off n9 |
|---|------|---------|---------|------------|------------|------|------|-------------|-------------|
| 2 | 0 | 0.1782 | 0.3194 | 0.1311 | 0.2471 | 0.736 | 0.774 | 0.6013 | 0.7603 |
| 2 | 1 | 0.1691 | 0.3028 | 0.1260 | 0.2473 | 0.745 | 0.817 | — | — |
| 2 | 2 | 0.1666 | 0.3007 | 0.1533 | 0.2791 | 0.920 | 0.928 | — | — |
| 16 | 0 | 0.1327 | 0.2470 | 0.0715 | 0.1809 | 0.539 | 0.732 | 0.6013 | 0.7603 |
| 16 | 1 | 0.1343 | 0.2554 | 0.0779 | 0.2103 | 0.580 | 0.823 | — | — |
| 16 | 2 | 0.1001 | 0.2203 | 0.0742 | 0.1871 | 0.741 | 0.849 | — | — |

snap_off is φ=0 (shell identity) with only γ trained, seed 0. It is the cost-pulse control: GRAPE of exp(−iγ H_c) inside the BinH sandwiches, without a nontrivial Hamming shell.

Circuit metric: 4/4 chosen-seed (n, L) cells have R≤0.98, and 6/6 seed×L runs beat on both n. L=2 (seed 1): R=0.745/0.817 (n6/n9), Med ε power_p 0.126/0.247 vs QAOA 0.169/0.303 vs product 0.147/0.052 L=16 (seed 0): R=0.539/0.732 (n6/n9), Med ε power_p 0.072/0.181 vs QAOA 0.133/0.247 vs product 0.147/0.052. φ=0 (cost pulse only, seed 0) sits far above the shell — Med ε about 0.60/0.76 versus power_p about 0.13/0.25 at L=2 — so GRAPE of exp(−i H_c) does not prepare the low-energy state by itself. The Hamming shell is doing work. Read that against the product column. At n=9 the product-state Med ε (about 0.05) is below every circuit, so mean-field beats the motif on the size transfer has to survive. At n=6, L=16, power_p Med ε is below the product upper bound found by the Bloch search. QAOA and power_p used the same L-BFGS budget (smoke depth, not a fully converged train).

## Verdict rule

Locked before the smoke numbers were read off the optimizer:

- **PASS** — quantumness confirms on train and on holdout, and the chosen seed has R≤0.98 on both n at both L∈{2, 16}.
- **KILL** — no λ on the grid confirms quantumness, or every (n, L) cell has R>1.02 (power_p does not beat QAOA anywhere).
- **needs_edit** — quantumness holds but the BEAT pattern is partial (one L, one n, or only one fragile seed), or power_p loses to the product-state baseline while still beating a weak QAOA.

Applied verdict: **KILL**.

## Repro

```
python Nov_3rd/scripts/smoke_quantum_h_D.py
```

Wall clock 245s. Self-check includes a probability match against `Nov3rdAnsatz.forward_nov3rd` at n=6 and n=9.
