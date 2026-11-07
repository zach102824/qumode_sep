# NEW_SAFE λ=0.75, μ=0 shallow seal — Walsh SNAP vs Fourier through n=18

**Verdict: PASS.** All four PASS bars hold.

Paper-lock from bakeoff #46, cell C: λ=0.75, μ=0, H = diag(unsat) − 0.75 ΣX. That bakeoff reported L=2 R_Fourier=0.620 (5/5), Λ12=0.748, Λ18=0.749, with L=4 a soft lead near 0.87. The TF-only seal (PR #45) at this λ, on the same batch stream as this run, had n=12 Λ=0.7477, gap=0.1265, L=2 R=0.6209, L=4 R=0.8881. Old SAFE (λ=0.75, μ=−0.50) is Λ-KILL appendix-only: median Λ≈0.94 (PR #41). The question here is whether the NEW_SAFE lead survives the PR #47 five-seed protocol through n=15 and n=18 without the cavity lock drifting up.

## Gate

Med ε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier. A seed BEATs when its own n=12 L=2 R_Fourier ≤ 0.98. Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. The reported Λ is the median over cavities and instances. The product gap is (E_prod − E_0)/|E_0| for the best real XZ product state.

- **PASS** when all four hold: (1) n=12 L=2 R_Fourier ≤ 0.85 and at least 4/5 seeds BEAT; (2) median Λ ≤ 0.80 at n=12 and at n=18; (3) median product gap ≥ 0.03 at n=12 and at n=18; (4) L=2 R_Fourier rises by at most 0.05 on each +3-qubit step n=12→15 and n=15→18.
- **needs_edit** when nothing kills the point and at least one PASS bar is missed. Λ drifting above 0.80 (and still under 0.90) is this case.
- **KILL** when n=18 median Λ ≥ 0.90, or when L=2 R_Fourier > 0.98 at n=12, n=15, or n=18.
- **L=4** is reported and is not a PASS bar. A ratio above 0.85 is noted as soft (bakeoff #46 n=12 L=4 was about 0.87; the TF-only stream was 0.888).

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i
\]

μ = 0, so there is no intra-cavity XX term. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. The circuit phases Cn = unsat · n/m. ⟨H⟩ uses the raw unsat count from the Z histogram and ⟨X_i⟩ from the X histogram. No Y, and no XX correlator in the energy. The mixer is BinH Hamming-shell SNAP with a Walsh-graded cost. Fourier is the two-angle QAOA. HVA is not trained: β^XX = 0 is Fourier.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128. Probe n=15 first 64.
- n=18: 32 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.
- L-BFGS matches `Nov_3rd.settings` and the PR #47 shallow seal: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε. The batch stream matches the Round11 TF-only trainer at λ=0.75.
- Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. Fourier and SNAP start on the same state.
- L≥8 was not trained. n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written. Old SAFE (0.75, −0.50) was not re-run.

## Ground state: Λ and product gap

Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has collapsed to an effective spin. A Z-basis product state scores Λ=1/3. The pass line is 0.80 and the kill line is 0.90 at n=18. Old SAFE μ=−0.50 sat near 0.94.

| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | fraction Λ≥0.90 | median product gap | min gap | fraction gap≥0.03 | median Z-peak |
|--:|:-----|--:|--------:|------:|------:|----------------:|----------------:|-------------------:|--------:|------------------:|-------------:|
| 12 | freeze (gate) | 128 | 0.7477 | 0.5498 | 0.9117 | 0.230 | 0.004 | 0.1265 | 0.0450 | 1.000 | 0.0379 |
| 15 | probe | 64 | 0.7489 | 0.5058 | 0.9142 | 0.269 | 0.003 | 0.1294 | 0.0669 | 1.000 | 0.0148 |
| 18 | probe (gate) | 32 | 0.7487 | 0.5687 | 0.8748 | 0.203 | 0.000 | 0.1336 | 0.0786 | 1.000 | 0.0064 |

n=12 median Λ = **0.7477**. n=15 median Λ = **0.7489**. n=18 median Λ = **0.7487**. n=12 median product gap = **0.1265** (n=15 0.1294, n=18 0.1336). The n=12 Z-basis peak of the ground state is 0.0379 (n=15 0.0148, n=18 0.0064). Max ground-state residual is 4.928e-09.

## L=2

| n | role | Medε Fourier | Medε SNAP | R_Fourier |
|--:|:-----|-------------:|----------:|----------:|
| 6 | holdout | 0.3579 | 0.2187 | 0.6110 |
| 9 | holdout (select) | 0.4732 | 0.2980 | 0.6298 |
| 12 | freeze (gate) | 0.6879 | 0.4271 | 0.6209 |
| 15 | probe | 0.8421 | 0.5066 | 0.6016 |
| 18 | probe | 1.0803 | 0.6352 | 0.5880 |

**L=2 n12 R_Fourier = 0.6209** (SNAP 0.4271 / Fourier 0.6879). Seeds BEAT (n12 R≤0.98): 5/5. Probe R is 0.6016 at n=15 and 0.5880 at n=18. Steps of R: n12→n15 -0.0194, n15→n18 -0.0136.

Paired instances with SNAP ε below Fourier: n12 640/640, n15 320/320, n18 160/160. Median shell RMS (distance of φ_k from the best line a+bk) = 0.1104.

| seed | n12 SNAP | n12 Fourier | n12 R | BEAT | paired n12 | n15 R | n18 R | shell RMS | SNAP best round | wall s |
|---:|---:|---:|---:|:----:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4264 | 0.6877 | 0.6201 | yes | 128/128 | 0.6036 | 0.5936 | 0.1300 | 36 | 3 |
| 1 | 0.4265 | 0.6879 | 0.6201 | yes | 128/128 | 0.6018 | 0.5880 | 0.1104 | 20 | 4 |
| 2 | 0.4271 | 0.6879 | 0.6209 | yes | 128/128 | 0.5986 | 0.5855 | 0.0783 | 38 | 4 |
| 3 | 0.4275 | 0.6881 | 0.6213 | yes | 128/128 | 0.6060 | 0.5919 | 0.1393 | 36 | 4 |
| 4 | 0.4273 | 0.6878 | 0.6214 | yes | 128/128 | 0.5994 | 0.5871 | 0.1042 | 37 | 4 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.860, SNAP 0.856. At n18: Fourier 0.861, SNAP 0.855. The gate uses the ground-state Λ, not these.

## L=4

| n | role | Medε Fourier | Medε SNAP | R_Fourier |
|--:|:-----|-------------:|----------:|----------:|
| 6 | holdout | 0.1452 | 0.1389 | 0.9563 |
| 9 | holdout (select) | 0.2257 | 0.2043 | 0.9053 |
| 12 | freeze | 0.3175 | 0.2819 | 0.8881 |
| 15 | probe | 0.3886 | 0.3375 | 0.8684 |
| 18 | probe | 0.4871 | 0.4265 | 0.8756 |

**L=4 n12 R_Fourier = 0.8881** (SNAP 0.2819 / Fourier 0.3175). Seeds BEAT (n12 R≤0.98): 4/5. Probe R is 0.8684 at n=15 and 0.8756 at n=18. Steps of R: n12→n15 -0.0197, n15→n18 +0.0072.

Paired instances with SNAP ε below Fourier: n12 565/640, n15 298/320, n18 146/160. Median shell RMS (distance of φ_k from the best line a+bk) = 0.2263.

| seed | n12 SNAP | n12 Fourier | n12 R | BEAT | paired n12 | n15 R | n18 R | shell RMS | SNAP best round | wall s |
|---:|---:|---:|---:|:----:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2722 | 0.3145 | 0.8655 | yes | 128/128 | 0.8523 | 0.8481 | 0.2381 | 39 | 5 |
| 1 | 0.2923 | 0.3187 | 0.9169 | yes | 110/128 | 0.9034 | 0.9126 | 0.2175 | 39 | 5 |
| 2 | 0.2431 | 0.3176 | 0.7654 | yes | 128/128 | 0.7658 | 0.7583 | 0.2402 | 39 | 5 |
| 3 | 0.3158 | 0.3165 | 0.9977 | no | 73/128 | 0.9707 | 0.9742 | 0.2037 | 39 | 5 |
| 4 | 0.2819 | 0.3175 | 0.8881 | yes | 126/128 | 0.8687 | 0.8756 | 0.2263 | 39 | 5 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n12: Fourier 0.826, SNAP 0.846. At n18: Fourier 0.822, SNAP 0.846. The gate uses the ground-state Λ, not these.

L=4 note: n12 L=4 R_Fourier=0.8881 is above 0.85, reported as a soft lead and not required for PASS.

## Reading

The call is **PASS**. All four PASS bars hold.

At the n=12 freeze, L=2 R_Fourier=0.6209 (SNAP 0.4271, Fourier 0.6879) with 5/5 seeds at or under the 0.98 BEAT line. The same ratio is 0.6016 at n=15 and 0.5880 at n=18. The +3-qubit steps are -0.0194 and -0.0136 (bar ≤ +0.05 on each step).

Ground-state median Λ is 0.7477 / 0.7489 / 0.7487 at n=12 / 15 / 18. The fraction of cavities above 0.80 is 0.230 / 0.269 / 0.203. The fraction at or above 0.90 is 0.004 / 0.003 / 0.000. The product gap is 0.1265 / 0.1294 / 0.1336 (bar ≥ 0.03 at n=12 and n=18). Z-basis peaks are 0.0379 / 0.0148 / 0.0064.

Λ equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩, computed from the joint X histogram of the ground state. Values near 1 mean the intra-cavity space has collapsed to an effective spin, which is what killed old SAFE μ=−0.50 (Λ≈0.941). The TF-only n=12 anchor on this batch stream was Λ=0.7477, gap=0.1265, Z-peak=0.0379, L=2 R=0.6209, L=4 R=0.8881. Bakeoff #46 reported Λ12=0.748, Λ18=0.749, L=2 R=0.620, L=4 R≈0.869. The n=12 freeze reproduces that anchor (ΔΛ=0.0000, Δgap=0.0000, ΔL=2 R=0.0000). n=18 median Λ is 0.7487 (-0.0003 versus the bakeoff #46 probe of 0.749). n=12 L=4 R is 0.8881 against the TF-only anchor 0.8881 and the bakeoff prior 0.869.

L=4 R_Fourier is 0.8881 / 0.8684 / 0.8756 at n=12 / 15 / 18. n12 L=4 R_Fourier=0.8881 is above 0.85, reported as a soft lead and not required for PASS. The soft tail is seed 3 n12/n15/n18 R=0.9977/0.9707/0.9742.

L=2 shell RMS median is 0.1104; L=4 shell RMS median is 0.2263. A small L=2 shell RMS means the Hamming shell stays close to a linear φ_k = a + bk, so part of the L=2 lead over Fourier is the Walsh-graded cost (four γ weights per layer against Fourier's one).

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n=9-holdout checkpoint. Ground-state Λ, the product gap, and the Z-peak are in `diag_n{n}_{role}.npz`.

