# Disordered λ at shallow L

**Overall: PASS.** Gate is δ=0.15, frozen n12. L=2 → PASS (R_HVA=0.3185). L=4 → PASS (R_HVA=0.5211).

**Stress δ=0.30 (reported, does not relabel the δ=0.15 gate): PASS.** L=2 → PASS (R_HVA=0.3224). L=4 → PASS (R_HVA=0.5744).

Hypothesis: the shallow SNAP win at uniform λ may be Hamming-shell symmetry matching a permutation-symmetric field. This run breaks that match. The gates are the same three ansatzes as the uniform SAFE-point run.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - \sum_i \lambda_i X_i - 0.50 \sum_c \sum_{i<j\in c} X_i X_j
\]

\[
\lambda_i = 0.75\,(1+\delta_i),\qquad \delta_i = \delta\, u_i,\qquad u_i\sim\mathrm{Unif}[-1,1]
\]

One pattern u (NumPy seed 20261105), shared by both strengths and by every n (qubit i uses u_i, so the n=6 pattern is the prefix of n=12). δ_i is Unif[−δ, δ]. XX stays the uniform cavity Rank-2 term with coefficient −0.50. All λ_i stay positive, so |+⟩ is still the ground state of the off-diagonal part.

Readout is the Z histogram (unsat) plus the X histogram. A site-dependent field is \(\sum_i \lambda_i\langle X_i\rangle\) from that same X histogram. No Y.

## Ansatz (unchanged)

| method | angles / layer | unitary |
|---|---|---|
| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), same β on every qubit |
| HVA QAOA | 3: γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\,U_X(\beta^X)\,U_C(\gamma)\). β^XX=0 is Fourier. |
| SNAP Walsh + Hamming shell | 8: γ_{w=1..4}, φ_{k=0..3} | Walsh degree-1..4 cost, then one shared shell on every cavity |

No per-qubit angles were added. Disorder is only in H. Cost family is still \(C_n=\mathrm{unsat}\cdot n/m\). Paper init β=β^X=0.01, β^XX=0, γ=γ_w=-0.01, φ_k=−β k, so the three methods start on the same state.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n9 holdout [200:250). Freeze n12, first 128 instances. n15 is not in this gate.
- L-BFGS budget matched to `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. No Adam.
- Seeds {0,1,2} share the paper init; the seed only reshuffles batches.
- Each seed restores the round with the best n9-holdout Med ε.
- The gate is the **median over seeds** of frozen-n12 Med ε.
- n21 phase-2 ladder was not read or written.

## Uniform reference (not remeasured)

δ=0 numbers are the SAFE-point HVA fairness run (`hva_fairness_safe`, same packs, budget, and gates). They are the shell-symmetric baseline this disorder is trying to break.

| L | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/HVA |
|---|---:|---:|---:|---:|
| 2 | 0.2667 | 0.2608 | 0.0855 | 0.3280 |
| 4 | 0.0771 | 0.0736 | 0.0394 | 0.5356 |

## λ patterns

### δ=0.15 (d0p15)

n=12 λ = 0.8090, 0.8535, 0.8490, 0.7594, 0.7723, 0.6785, 0.6494, 0.6893, 0.8033, 0.8295, 0.7300, 0.8291.

| cavity | λ | range |
|---|---|---:|
| 0 | 0.8090, 0.8535, 0.8490 | 0.0445 |
| 1 | 0.7594, 0.7723, 0.6785 | 0.0938 |
| 2 | 0.6494, 0.6893, 0.8033 | 0.1538 |
| 3 | 0.8295, 0.7300, 0.8291 | 0.0994 |

## δ=0.15  L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1323 | 0.0924 | 0.0424 | 0.3207 | 0.4592 |
| 9 | holdout (select) | 0.1879 | 0.1567 | 0.0603 | 0.3212 | 0.3851 |
| 12 | freeze | 0.2585 | 0.2597 | 0.0827 | 0.3200 | 0.3185 |

**δ=0.15 L=2 n12 gate: PASS.** R_HVA=0.3185 (SNAP 0.0827 / HVA 0.2597), R_Fourier=0.3200.

| method | seed | best round | n9 hold Medε | n12 Medε | wall s | mean\|β^XX\| or shell-nl |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 6 | 0.1879 | 0.2585 | 7 | — |
| fourier | 1 | 34 | 0.1878 | 0.2582 | 6 | — |
| fourier | 2 | 17 | 0.1879 | 0.2587 | 6 | — |
| hva | 0 | 18 | 0.1567 | 0.2603 | 7 | 0.2732 |
| hva | 1 | 6 | 0.1567 | 0.2582 | 7 | 0.3020 |
| hva | 2 | 29 | 0.1564 | 0.2597 | 7 | 0.2708 |
| snap | 0 | 38 | 0.0603 | 0.0829 | 6 | 0.3721 |
| snap | 1 | 11 | 0.0602 | 0.0824 | 5 | 0.4367 |
| snap | 2 | 17 | 0.0605 | 0.0827 | 5 | 0.4310 |

n12 seed range (min–max Medε): fourier 0.2582–0.2587, hva 0.2582–0.2603, snap 0.0824–0.0829.

Versus uniform δ=0 at this L, n12 R_HVA was 0.3280 (SNAP 0.0855 / HVA 0.2608).

## δ=0.15  L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0415 | 0.0206 | 0.0226 | 0.5452 | 1.0978 |
| 9 | holdout (select) | 0.0573 | 0.0352 | 0.0297 | 0.5181 | 0.8444 |
| 12 | freeze | 0.0759 | 0.0775 | 0.0404 | 0.5319 | 0.5211 |

**δ=0.15 L=4 n12 gate: PASS.** R_HVA=0.5211 (SNAP 0.0404 / HVA 0.0775), R_Fourier=0.5319.

| method | seed | best round | n9 hold Medε | n12 Medε | wall s | mean\|β^XX\| or shell-nl |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 13 | 0.0574 | 0.0759 | 9 | — |
| fourier | 1 | 39 | 0.0573 | 0.0759 | 8 | — |
| fourier | 2 | 21 | 0.0573 | 0.0750 | 8 | — |
| hva | 0 | 38 | 0.0351 | 0.0776 | 11 | 0.3163 |
| hva | 1 | 39 | 0.0352 | 0.0775 | 10 | 0.3158 |
| hva | 2 | 34 | 0.0354 | 0.0751 | 10 | 0.3165 |
| snap | 0 | 39 | 0.0294 | 0.0401 | 8 | 0.3732 |
| snap | 1 | 39 | 0.0316 | 0.0423 | 8 | 0.3511 |
| snap | 2 | 38 | 0.0297 | 0.0404 | 8 | 0.3606 |

n12 seed range (min–max Medε): fourier 0.0750–0.0759, hva 0.0751–0.0776, snap 0.0401–0.0423.

Versus uniform δ=0 at this L, n12 R_HVA was 0.5356 (SNAP 0.0394 / HVA 0.0736).

### δ=0.30 (d0p30)

n=12 λ = 0.8680, 0.9570, 0.9480, 0.7687, 0.7947, 0.6071, 0.5489, 0.6285, 0.8565, 0.9090, 0.7101, 0.9081.

| cavity | λ | range |
|---|---|---:|
| 0 | 0.8680, 0.9570, 0.9480 | 0.0889 |
| 1 | 0.7687, 0.7947, 0.6071 | 0.1876 |
| 2 | 0.5489, 0.6285, 0.8565 | 0.3077 |
| 3 | 0.9090, 0.7101, 0.9081 | 0.1989 |

## δ=0.30  L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1310 | 0.0903 | 0.0399 | 0.3048 | 0.4421 |
| 9 | holdout (select) | 0.1874 | 0.1551 | 0.0592 | 0.3157 | 0.3814 |
| 12 | freeze | 0.2530 | 0.2577 | 0.0831 | 0.3284 | 0.3224 |

**δ=0.30 L=2 n12 gate: PASS.** R_HVA=0.3224 (SNAP 0.0831 / HVA 0.2577), R_Fourier=0.3284.

| method | seed | best round | n9 hold Medε | n12 Medε | wall s | mean\|β^XX\| or shell-nl |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 6 | 0.1874 | 0.2527 | 6 | — |
| fourier | 1 | 17 | 0.1873 | 0.2530 | 6 | — |
| fourier | 2 | 17 | 0.1874 | 0.2535 | 6 | — |
| hva | 0 | 18 | 0.1551 | 0.2587 | 6 | 0.2725 |
| hva | 1 | 9 | 0.1552 | 0.2568 | 6 | 0.2858 |
| hva | 2 | 32 | 0.1548 | 0.2577 | 6 | 0.2708 |
| snap | 0 | 38 | 0.0592 | 0.0831 | 5 | 0.3736 |
| snap | 1 | 39 | 0.0588 | 0.0828 | 6 | 0.3590 |
| snap | 2 | 37 | 0.0593 | 0.0839 | 5 | 0.3826 |

n12 seed range (min–max Medε): fourier 0.2527–0.2535, hva 0.2568–0.2587, snap 0.0828–0.0839.

Versus uniform δ=0 at this L, n12 R_HVA was 0.3280 (SNAP 0.0855 / HVA 0.2608).

## δ=0.30  L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0388 | 0.0209 | 0.0223 | 0.5764 | 1.0676 |
| 9 | holdout (select) | 0.0576 | 0.0371 | 0.0311 | 0.5408 | 0.8405 |
| 12 | freeze | 0.0772 | 0.0728 | 0.0418 | 0.5412 | 0.5744 |

**δ=0.30 L=4 n12 gate: PASS.** R_HVA=0.5744 (SNAP 0.0418 / HVA 0.0728), R_Fourier=0.5412.

| method | seed | best round | n9 hold Medε | n12 Medε | wall s | mean\|β^XX\| or shell-nl |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 38 | 0.0577 | 0.0773 | 9 | — |
| fourier | 1 | 19 | 0.0576 | 0.0770 | 8 | — |
| fourier | 2 | 32 | 0.0575 | 0.0772 | 8 | — |
| hva | 0 | 31 | 0.0370 | 0.0709 | 10 | 0.3141 |
| hva | 1 | 39 | 0.0371 | 0.0785 | 10 | 0.3148 |
| hva | 2 | 36 | 0.0373 | 0.0728 | 10 | 0.3148 |
| snap | 0 | 39 | 0.0320 | 0.0421 | 8 | 0.3619 |
| snap | 1 | 39 | 0.0311 | 0.0418 | 8 | 0.3529 |
| snap | 2 | 39 | 0.0280 | 0.0388 | 8 | 0.3884 |

n12 seed range (min–max Medε): fourier 0.0770–0.0773, hva 0.0709–0.0785, snap 0.0388–0.0421.

Versus uniform δ=0 at this L, n12 R_HVA was 0.5356 (SNAP 0.0394 / HVA 0.0736).

## Reading

Disorder of this size does not remove the shallow SNAP lead. Frozen-n12 R_HVA stays near the uniform-λ value at both strengths.

| δ | L=2 R_HVA | L=4 R_HVA | L=2 gate | L=4 gate |
|---|---:|---:|---|---|
| 0 (prior) | 0.3280 | 0.5356 | PASS | PASS |
| 0.15 | 0.3185 | 0.5211 | PASS | PASS |
| 0.30 | 0.3224 | 0.5744 | PASS | PASS |

Every disordered cell is PASS (R_HVA ≤ 0.85). Seed ranges do not overlap: at each completed cell the highest SNAP n12 Medε is still below the lowest HVA n12 Medε.

The field inside a cavity is not a small perturbation. At δ=0.30, cavity 2 has λ = 0.5489, 0.6285, 0.8565 (range 0.3077 on a mean near 0.75). That breaks 3-bit shell symmetry, and n12 R_HVA at L=2 is 0.322 versus 0.328 with a uniform field.

HVA used the XX mixer (mean |β^XX| is about 0.27 at L=2 and 0.32 at L=4). SNAP shells left the linear X-only line (shell-nl about 0.35–0.44; zero would be φ_k = a + b k). The lead is not a stuck β^XX=0 init, and it is not SNAP sitting on the product X mixer.

At L=4 the n=6 holdout has R_HVA above 1 (HVA slightly ahead on that slice), and the n=9 holdout sits near 0.84. The uniform run already had this tilt (n6 R_HVA=1.016, n9=0.807, n12=0.536). Disorder does not create it. The freeze-n12 gate, the transfer number, stays a SNAP win.

The shell-symmetry hypothesis does not account for the L=2/4 SNAP win. needs_edit and KILL do not trigger at δ=0.15, and the δ=0.30 stress is PASS as well.


## How to read the gate

- **PASS** — at δ=0.15, R_HVA = Medε_SNAP / Medε_HVA ≤ 0.85 on frozen n12. The shallow SNAP lead survives a 15% site disorder.
- **needs_edit** — at δ=0.15 the lead shrinks past 0.85 and HVA is not ahead. The win vanished.
- **KILL** — at δ=0.15, Medε_HVA < Medε_SNAP on a shallow depth. HVA dominates once the field is disordered.
- δ=0.30 is the stronger break of the same pattern. It is reported beside the gate and does not flip a δ=0.15 PASS into a KILL.
- Overall uses every shallow depth that was run at δ=0.15. Any KILL there wins; else any needs_edit; else PASS.

Per-seed JSON: `{d0p15,d0p30}_L{L}_{method}_seed{s}.json`. Angles are the restored n9-holdout checkpoint. `shell-nl` is the mean over layers of the std of consecutive φ differences (0 means the shell is exactly linear, φ_k = a + b k, the pure X mixer). `mean|β^XX|` is the mean absolute XX angle.
