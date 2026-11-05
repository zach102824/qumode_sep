# HVA fairness baseline — SAFE point

**Overall: KILL.** L=2 → PASS (R_HVA=0.3280). L=4 → PASS (R_HVA=0.5356). L=8 → KILL (R_HVA=1.3720). L=16 → KILL (R_HVA=2.2751).

Same Hamiltonian for every column. Train n∈{6,9} only; n12 and n15 are frozen.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j
\]

SAFE point **λ=0.75, μ=−0.50**: both signs are the minus signs in that formula (μ is the coefficient of ΣXX, already negative). Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. ε = ⟨H⟩ − E_0, with E_0 the exact ground energy. **Med ε** below is the median over seeds of each seed's median-over-instances ε.

Readout is the Z histogram (unsat) plus the X histogram (⟨X_i⟩ and ⟨X_i X_j⟩; XX is diagonal in the X basis). No Y.

## Ansatz (per layer, applied to |+⟩⊗n as cost then mixer)

| method | angles / layer | unitary |
|---|---|---|
| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), \(U_X=\bigotimes\exp(-i\beta X/2)\) |
| **HVA QAOA** | 3: γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\,U_X(\beta^X)\,U_C(\gamma)\). β^XX=0 is Fourier. |
| SNAP Walsh + Hamming shell | 8: γ_{w=1..4}, φ_{k=0..3} | Walsh degree-1..4 cost, then \(M=H^{\otimes 3}\mathrm{diag}(e^{-i\phi_{\mathrm{pc}}})H^{\otimes 3}\) on every cavity |

Cost generator \(U_C=\exp(-i\gamma C_n)\) with \(C_n=\mathrm{unsat}\cdot n/m\), \(m=\mathrm{round}(9.93 n)\), the same family as \(\exp(-i\gamma\,\mathrm{diag}(\mathrm{unsat}))\). 4-SAT unsat is Walsh degree ≤4, so equal γ_w reproduces scalar γ·Cn up to a global phase. Linear shell \(\phi_k=-\beta k\) reproduces \(U_X(\beta)\). Free shells also generate within-cavity XXX; HVA's mixer stops at XX. That extra shell term, plus independent γ_w, is the champ — the question is whether giving Fourier the missing XX term removes the win.

Paper init β=β^X=0.01, β^XX=0, γ=γ_w=-0.01, φ_k=−β k. On these instances the three inits are the same state.

## Protocol

- Packs: `beat_qaoa_12h/train_packs` (satisfiable 4-SAT, 250 each). Train indices [0:200) on n=6 and n=9. Holdout [200:250) checkpoints on n9 Med ε. Freeze n12 first 128, n15 first 64.
- L-BFGS budget matched to `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round (seal pattern). Loss = mean ε/n on the batch. No Adam.
- Seeds {0,1,2} share the paper init; the seed only changes the L-BFGS batches.
- Each seed restores the round with the best n9-holdout Med ε (init allowed).
- Gate uses the **median over seeds** of frozen-n12 Med ε, not the lucky seed.
- n21 phase-2 ladder was not read or written.
- Opus brief `opus_reply_20261105_0314_h_explore_round8_refine.md` was not in the tree.

## L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1369 | 0.0952 | 0.0465 | 0.3399 | 0.4887 |
| 9 | holdout (select) | 0.1905 | 0.1600 | 0.0588 | 0.3087 | 0.3676 |
| 12 | freeze | 0.2667 | 0.2608 | 0.0855 | 0.3208 | 0.3280 |
| 15 | freeze | 0.3331 | 0.3504 | 0.1047 | 0.3142 | 0.2987 |

**L=2 n12 gate: PASS.** R_HVA=0.3280 (SNAP 0.0855 / HVA 0.2608), R_Fourier=0.3208.

Seed Med ε (n12 freeze), and the n9 holdout used to pick the checkpoint:

| method | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 6 | 0.1905 | 0.2667 | 0.3329 | 6 |
| fourier | 1 | 34 | 0.1904 | 0.2667 | 0.3331 | 5 |
| fourier | 2 | 7 | 0.1905 | 0.2664 | 0.3332 | 5 |
| hva | 0 | 19 | 0.1604 | 0.2632 | 0.3523 | 6 |
| hva | 1 | 9 | 0.1600 | 0.2608 | 0.3504 | 6 |
| hva | 2 | 6 | 0.1589 | 0.2555 | 0.3353 | 6 |
| snap | 0 | 36 | 0.0588 | 0.0851 | 0.1040 | 5 |
| snap | 1 | 28 | 0.0589 | 0.0858 | 0.1048 | 5 |
| snap | 2 | 32 | 0.0588 | 0.0855 | 0.1047 | 5 |

n12 seed range (min–max Medε): fourier 0.2664–0.2667, hva 0.2555–0.2632, snap 0.0851–0.0858.

Paper-init Medε (identical ansatz state; one shared eval): n6=0.6249, n9=0.8260, n12=1.1290, n15=1.4020.

## L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0452 | 0.0229 | 0.0232 | 0.5142 | 1.0164 |
| 9 | holdout (select) | 0.0587 | 0.0354 | 0.0285 | 0.4863 | 0.8071 |
| 12 | freeze | 0.0771 | 0.0736 | 0.0394 | 0.5112 | 0.5356 |
| 15 | freeze | 0.0983 | 0.1110 | 0.0450 | 0.4574 | 0.4053 |

**L=4 n12 gate: PASS.** R_HVA=0.5356 (SNAP 0.0394 / HVA 0.0736), R_Fourier=0.5112.

Seed Med ε (n12 freeze), and the n9 holdout used to pick the checkpoint:

| method | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 18 | 0.0587 | 0.0771 | 0.0983 | 8 |
| fourier | 1 | 21 | 0.0586 | 0.0774 | 0.0987 | 8 |
| fourier | 2 | 21 | 0.0588 | 0.0760 | 0.0968 | 8 |
| hva | 0 | 29 | 0.0354 | 0.0718 | 0.1061 | 10 |
| hva | 1 | 33 | 0.0354 | 0.0764 | 0.1149 | 9 |
| hva | 2 | 34 | 0.0352 | 0.0736 | 0.1110 | 9 |
| snap | 0 | 38 | 0.0285 | 0.0394 | 0.0450 | 8 |
| snap | 1 | 39 | 0.0298 | 0.0408 | 0.0463 | 7 |
| snap | 2 | 39 | 0.0255 | 0.0370 | 0.0426 | 7 |

n12 seed range (min–max Medε): fourier 0.0760–0.0774, hva 0.0718–0.0764, snap 0.0370–0.0408.

Paper-init Medε (identical ansatz state; one shared eval): n6=0.6241, n9=0.8248, n12=1.1273, n15=1.4000.

## L=8

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0201 | 0.0107 | 0.0235 | 1.1663 | 2.2004 |
| 9 | holdout (select) | 0.0242 | 0.0164 | 0.0289 | 1.1955 | 1.7687 |
| 12 | freeze | 0.0302 | 0.0281 | 0.0385 | 1.2741 | 1.3720 |
| 15 | freeze | 0.0364 | 0.0384 | 0.0450 | 1.2380 | 1.1744 |

**L=8 n12 gate: KILL.** R_HVA=1.3720 (SNAP 0.0385 / HVA 0.0281), R_Fourier=1.2741.

Seed Med ε (n12 freeze), and the n9 holdout used to pick the checkpoint:

| method | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 39 | 0.0248 | 0.0319 | 0.0374 | 14 |
| fourier | 1 | 35 | 0.0239 | 0.0295 | 0.0353 | 13 |
| fourier | 2 | 39 | 0.0242 | 0.0302 | 0.0364 | 13 |
| hva | 0 | 39 | 0.0160 | 0.0295 | 0.0407 | 16 |
| hva | 1 | 38 | 0.0171 | 0.0278 | 0.0367 | 16 |
| hva | 2 | 39 | 0.0164 | 0.0281 | 0.0384 | 16 |
| snap | 0 | 39 | 0.0289 | 0.0385 | 0.0450 | 12 |
| snap | 1 | 39 | 0.0326 | 0.0443 | 0.0496 | 12 |
| snap | 2 | 39 | 0.0274 | 0.0371 | 0.0427 | 12 |

n12 seed range (min–max Medε): fourier 0.0295–0.0319, hva 0.0278–0.0295, snap 0.0371–0.0443.

Paper-init Medε (identical ansatz state; one shared eval): n6=0.6216, n9=0.8205, n12=1.1214, n15=1.3932.

## L=16

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0184 | 0.0064 | 0.0198 | 1.0723 | 3.1051 |
| 9 | holdout (select) | 0.0218 | 0.0088 | 0.0228 | 1.0483 | 2.5870 |
| 12 | freeze | 0.0273 | 0.0144 | 0.0328 | 1.2001 | 2.2751 |
| 15 | freeze | 0.0323 | 0.0173 | 0.0372 | 1.1511 | 2.1522 |

**L=16 n12 gate: KILL.** R_HVA=2.2751 (SNAP 0.0328 / HVA 0.0144), R_Fourier=1.2001.

Seed Med ε (n12 freeze), and the n9 holdout used to pick the checkpoint:

| method | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |
|---|---:|---:|---:|---:|---:|---:|
| fourier | 0 | 38 | 0.0216 | 0.0270 | 0.0321 | 23 |
| fourier | 1 | 38 | 0.0219 | 0.0273 | 0.0323 | 22 |
| fourier | 2 | 39 | 0.0218 | 0.0273 | 0.0324 | 23 |
| hva | 0 | 39 | 0.0085 | 0.0134 | 0.0164 | 32 |
| hva | 1 | 39 | 0.0104 | 0.0166 | 0.0204 | 32 |
| hva | 2 | 39 | 0.0088 | 0.0144 | 0.0173 | 33 |
| snap | 0 | 39 | 0.0241 | 0.0339 | 0.0399 | 22 |
| snap | 1 | 39 | 0.0225 | 0.0323 | 0.0367 | 23 |
| snap | 2 | 39 | 0.0228 | 0.0328 | 0.0372 | 23 |

n12 seed range (min–max Medε): fourier 0.0270–0.0273, hva 0.0134–0.0166, snap 0.0323–0.0339.

Paper-init Medε (identical ansatz state; one shared eval): n6=0.6120, n9=0.8044, n12=1.1001, n15=1.3677.

## How to read the gate

- **PASS** — R_HVA = Medε_SNAP / Medε_HVA ≤ 0.85 at frozen n12. SNAP stays clearly ahead of a mixer that can express ΣX and ΣXX.
- **needs_edit** — R_HVA > 0.85 and HVA does not beat SNAP. The SNAP lead shrinks once Fourier is allowed an XX mixer.
- **KILL** — Medε_HVA < Medε_SNAP at n12. The fair mixer wins.
- Overall is KILL if any reported depth is KILL, else needs_edit if any depth is needs_edit, else PASS.

## Reading

The n12 gate depends on depth, at one L-BFGS budget. PASS at L=2, 4. KILL at L=8, 16.
Where HVA wins, Fourier also undercuts SNAP at this budget, and HVA is ahead of Fourier. The XX angle is doing real work, and by L=8 it has erased the shallow SNAP lead.

HVA did use the new mixer (L=2 mean|β^XX|=0.285, L=4 mean|β^XX|=0.317, L=8 mean|β^XX|=0.189, L=16 mean|β^XX|=0.110). The shallow n12 near-tie with Fourier is not a stuck β^XX=0 init.

At L=2 and L=4 the XX term helps the n6/n9 holdout more than it helps frozen n12, and at n15 HVA is slightly worse than Fourier. SNAP shells leave the linear φ_k=−βk line (the X-only mixer), and the γ_w do not collapse together.

At L=8 and L=16 the seal budget still has every method's best n9-holdout checkpoint on the last rounds, so 40×8 is not a flat minimum. A continuation of L=16 seed 0 for another 40 rounds (same batch size and lr, fresh batches) is in `extra40_L16_seed0.json` when present.
That continuation, n12 Medε: fourier n12=0.0237, hva n12=0.0099, snap n12=0.0148.
R_HVA after 80 rounds on that one seed is 1.4921. Doubling the budget narrows the deep-L gap and lets SNAP pass Fourier; it does not put SNAP back in front of HVA.

Gate used here is excess energy above the exact ground state, not the classical 1/p(success) ratio. A previously quoted freeze-n12 R≈0.15 vs Fourier was not reproduced at this budget (measured R vs Fourier is the table column). Overall label: KILL.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint.
