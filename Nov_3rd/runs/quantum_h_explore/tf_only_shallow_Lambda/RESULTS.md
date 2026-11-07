# Transverse-field only — shallow SNAP vs Fourier (μ = 0)

**Verdict: PASS.**

No cavity XX. The Hamiltonian is H = diag(unsat) − λ Σ_i X_i with λ ∈ {0.50, 0.75, 1.00}. Walsh Hamming-shell SNAP is compared with Fourier QAOA. HVA is not trained: β^XX = 0 prepares the Fourier state, and there is no XX term in H for a separate mixer to match. SAFE (PR #41, λ=0.75, μ=−0.50) had n=12 median cavity Λ = 0.941.

## Gate

Med ε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier. Λ, the product gap, and the Z-peak are medians on the frozen n=12 pool (B=128). Λ pools cavities and instances of the true ground state's X histogram. The product gap is (E_prod − E_0)/|E_0|. A classical ground state has median Z-peak ≥ 0.5.

- **PASS** if some λ has median Λ ≤ 0.80, product gap ≥ 0.03, Z-peak < 0.5, and n=12 R_Fourier ≤ 0.85 at L=2 or at L=4.
- **needs_edit** if Λ, the gap, and the Z-peak meet that bar and the better n=12 depth has R_Fourier in (0.85, 0.98]. Also the residual: a λ that misses PASS and is not a KILL.
- **KILL** a λ when median Λ > 0.85, or the ground state is classical, or R_Fourier > 0.98 at both L=2 and L=4 on the n=6 and n=9 holdouts. The family is KILL only when every λ is KILL. One PASS λ is a family PASS.

## Hamiltonian and protocol

Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. The circuit phases Cn = unsat · n/m. ⟨H⟩ uses the raw unsat count from the Z histogram and ⟨X_i⟩ from the X histogram. No Y, and no XX correlator in the energy. L-BFGS matches `Nov_3rd.settings` and the shallow seals: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε.

Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Holdout [200:250). Freeze n=12 first 128. Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. L≥8 was not trained. n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Ground state

| λ | n | role | B | median Λ | min Λ | max Λ | frac Λ>0.85 | median gap | min gap | median Z-peak | frac peak≥0.5 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 6 | train | 200 | 0.5771 | 0.4308 | 0.8725 | 0.005 | 0.1152 | 0.0392 | 0.3995 | 0.270 |
| 0.50 | 6 | hold | 50 | 0.5799 | 0.4576 | 0.8225 | 0.000 | 0.1118 | 0.0338 | 0.3996 | 0.200 |
| 0.50 | 9 | train | 200 | 0.5586 | 0.4090 | 0.9206 | 0.005 | 0.1094 | 0.0360 | 0.2741 | 0.015 |
| 0.50 | 9 | hold | 50 | 0.5587 | 0.4192 | 0.7729 | 0.000 | 0.0912 | 0.0541 | 0.3097 | 0.040 |
| 0.50 | 12 | freeze | 128 | 0.5585 | 0.4037 | 0.8593 | 0.002 | 0.1102 | 0.0413 | 0.1569 | 0.008 |
| 0.75 | 6 | train | 200 | 0.7612 | 0.5741 | 0.9417 | 0.072 | 0.1283 | 0.0386 | 0.2022 | 0.000 |
| 0.75 | 6 | hold | 50 | 0.7618 | 0.5883 | 0.9055 | 0.050 | 0.1273 | 0.0464 | 0.2011 | 0.000 |
| 0.75 | 9 | train | 200 | 0.7562 | 0.5223 | 0.9343 | 0.060 | 0.1308 | 0.0450 | 0.0891 | 0.000 |
| 0.75 | 9 | hold | 50 | 0.7530 | 0.5862 | 0.8996 | 0.067 | 0.1279 | 0.0558 | 0.1027 | 0.000 |
| 0.75 | 12 | freeze | 128 | 0.7477 | 0.5498 | 0.9117 | 0.061 | 0.1265 | 0.0450 | 0.0379 | 0.000 |
| 1.00 | 6 | train | 200 | 0.8548 | 0.6920 | 0.9571 | 0.527 | 0.1016 | 0.0331 | 0.1183 | 0.000 |
| 1.00 | 6 | hold | 50 | 0.8559 | 0.6947 | 0.9487 | 0.550 | 0.0972 | 0.0404 | 0.1155 | 0.000 |
| 1.00 | 9 | train | 200 | 0.8536 | 0.6475 | 0.9543 | 0.528 | 0.1013 | 0.0427 | 0.0379 | 0.000 |
| 1.00 | 9 | hold | 50 | 0.8567 | 0.7098 | 0.9378 | 0.567 | 0.1033 | 0.0506 | 0.0417 | 0.000 |
| 1.00 | 12 | freeze | 128 | 0.8514 | 0.6836 | 0.9345 | 0.514 | 0.0989 | 0.0507 | 0.0129 | 0.000 |

The decision row is n=12 freeze. n=6 and n=9 are the same ensemble, reported so a size accident is visible.

## Ratios

| λ | L | n | role | Medε Fourier | Medε SNAP | R_Fourier | paired SNAP<Fourier |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.50 | 2 | 6 | holdout | 0.6046 | 0.4441 | 0.7345 | 243/250 |
| 0.50 | 2 | 9 | holdout (select) | 0.8908 | 0.6892 | 0.7736 | 245/250 |
| 0.50 | 2 | 12 | freeze | 1.1686 | 0.8779 | 0.7512 | 638/640 |
| 0.50 | 4 | 6 | holdout | 0.3192 | 0.2956 | 0.9258 | 182/250 |
| 0.50 | 4 | 9 | holdout (select) | 0.5022 | 0.4736 | 0.9430 | 205/250 |
| 0.50 | 4 | 12 | freeze | 0.6462 | 0.5839 | 0.9037 | 541/640 |
| 0.75 | 2 | 6 | holdout | 0.3579 | 0.2187 | 0.6110 | 250/250 |
| 0.75 | 2 | 9 | holdout (select) | 0.4732 | 0.2980 | 0.6298 | 250/250 |
| 0.75 | 2 | 12 | freeze | 0.6879 | 0.4271 | 0.6209 | 640/640 |
| 0.75 | 4 | 6 | holdout | 0.1452 | 0.1389 | 0.9563 | 161/250 |
| 0.75 | 4 | 9 | holdout (select) | 0.2257 | 0.2043 | 0.9053 | 216/250 |
| 0.75 | 4 | 12 | freeze | 0.3175 | 0.2819 | 0.8881 | 565/640 |
| 1.00 | 2 | 6 | holdout | 0.2426 | 0.1198 | 0.4938 | 250/250 |
| 1.00 | 2 | 9 | holdout (select) | 0.3238 | 0.1528 | 0.4720 | 250/250 |
| 1.00 | 2 | 12 | freeze | 0.4740 | 0.2332 | 0.4920 | 640/640 |
| 1.00 | 4 | 6 | holdout | 0.0789 | 0.0657 | 0.8333 | 233/250 |
| 1.00 | 4 | 9 | holdout (select) | 0.1174 | 0.0890 | 0.7575 | 246/250 |
| 1.00 | 4 | 12 | freeze | 0.1817 | 0.1301 | 0.7161 | 634/640 |

| λ | call | detail | n12 Λ | n12 gap | n12 Z-peak | n12 R L=2 | n12 R L=4 | shell RMS L=2 | shell RMS L=4 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | PASS | n12 R≤0.85 with Λ≤0.80, gap≥0.03, and a non-classical GS | 0.5585 | 0.1102 | 0.1569 | 0.7512 | 0.9037 | 0.0452 | 0.2157 |
| 0.75 | PASS | n12 R≤0.85 with Λ≤0.80, gap≥0.03, and a non-classical GS | 0.7477 | 0.1265 | 0.0379 | 0.6209 | 0.8881 | 0.1104 | 0.2263 |
| 1.00 | KILL | median Λ 0.8514 > 0.85 | 0.8514 | 0.0989 | 0.0129 | 0.4920 | 0.7161 | 0.1454 | 0.2486 |

### Seeds at n=12

| λ | L | seed | SNAP | Fourier | R | shell RMS | prep Λ SNAP | prep Λ Fourier | best round SNAP | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 2 | 0 | 0.8769 | 1.1688 | 0.7503 | 0.0845 | 0.7907 | 0.7813 | 39 | 4 |
| 0.50 | 2 | 1 | 0.8767 | 1.1686 | 0.7502 | 0.0452 | 0.7862 | 0.7839 | 34 | 4 |
| 0.50 | 2 | 2 | 0.8779 | 1.1686 | 0.7512 | 0.0447 | 0.7878 | 0.7789 | 38 | 4 |
| 0.50 | 2 | 3 | 0.8802 | 1.1677 | 0.7538 | 0.0447 | 0.7874 | 0.7788 | 39 | 4 |
| 0.50 | 2 | 4 | 0.8789 | 1.1686 | 0.7521 | 0.0533 | 0.7861 | 0.7829 | 24 | 4 |
| 0.50 | 4 | 0 | 0.5797 | 0.6447 | 0.8992 | 0.2157 | 0.7615 | 0.7315 | 39 | 5 |
| 0.50 | 4 | 1 | 0.5936 | 0.6475 | 0.9168 | 0.2194 | 0.7642 | 0.7264 | 39 | 5 |
| 0.50 | 4 | 2 | 0.5402 | 0.6461 | 0.8360 | 0.1821 | 0.7570 | 0.7298 | 38 | 5 |
| 0.50 | 4 | 3 | 0.5839 | 0.6462 | 0.9037 | 0.2158 | 0.7573 | 0.7234 | 39 | 5 |
| 0.50 | 4 | 4 | 0.6317 | 0.6473 | 0.9760 | 0.2051 | 0.7663 | 0.7264 | 39 | 5 |
| 0.75 | 2 | 0 | 0.4264 | 0.6877 | 0.6201 | 0.1300 | 0.8591 | 0.8596 | 36 | 4 |
| 0.75 | 2 | 1 | 0.4265 | 0.6879 | 0.6201 | 0.1104 | 0.8563 | 0.8599 | 20 | 4 |
| 0.75 | 2 | 2 | 0.4271 | 0.6879 | 0.6209 | 0.0783 | 0.8546 | 0.8595 | 38 | 4 |
| 0.75 | 2 | 3 | 0.4275 | 0.6881 | 0.6213 | 0.1393 | 0.8589 | 0.8619 | 36 | 4 |
| 0.75 | 2 | 4 | 0.4273 | 0.6878 | 0.6214 | 0.1042 | 0.8547 | 0.8579 | 37 | 4 |
| 0.75 | 4 | 0 | 0.2722 | 0.3145 | 0.8655 | 0.2381 | 0.8430 | 0.8264 | 39 | 5 |
| 0.75 | 4 | 1 | 0.2923 | 0.3187 | 0.9169 | 0.2175 | 0.8477 | 0.8255 | 39 | 5 |
| 0.75 | 4 | 2 | 0.2431 | 0.3176 | 0.7654 | 0.2402 | 0.8382 | 0.8255 | 39 | 5 |
| 0.75 | 4 | 3 | 0.3158 | 0.3165 | 0.9977 | 0.2037 | 0.8458 | 0.8283 | 39 | 5 |
| 0.75 | 4 | 4 | 0.2819 | 0.3175 | 0.8881 | 0.2263 | 0.8468 | 0.8266 | 39 | 5 |
| 1.00 | 2 | 0 | 0.2346 | 0.4740 | 0.4949 | 0.1866 | 0.8991 | 0.9065 | 24 | 4 |
| 1.00 | 2 | 1 | 0.2332 | 0.4721 | 0.4940 | 0.1000 | 0.8992 | 0.9057 | 32 | 4 |
| 1.00 | 2 | 2 | 0.2328 | 0.4750 | 0.4901 | 0.1551 | 0.8990 | 0.9056 | 27 | 4 |
| 1.00 | 2 | 3 | 0.2323 | 0.4739 | 0.4902 | 0.0845 | 0.8987 | 0.9065 | 37 | 4 |
| 1.00 | 2 | 4 | 0.2343 | 0.4747 | 0.4936 | 0.1454 | 0.8991 | 0.9052 | 31 | 4 |
| 1.00 | 4 | 0 | 0.1053 | 0.1816 | 0.5798 | 0.2615 | 0.8876 | 0.8864 | 39 | 5 |
| 1.00 | 4 | 1 | 0.1301 | 0.1818 | 0.7159 | 0.2486 | 0.8911 | 0.8864 | 39 | 5 |
| 1.00 | 4 | 2 | 0.1228 | 0.1821 | 0.6743 | 0.2567 | 0.8884 | 0.8853 | 38 | 5 |
| 1.00 | 4 | 3 | 0.1492 | 0.1811 | 0.8239 | 0.2240 | 0.8958 | 0.8871 | 39 | 5 |
| 1.00 | 4 | 4 | 0.1370 | 0.1817 | 0.7540 | 0.2473 | 0.8928 | 0.8857 | 39 | 5 |

Prepared-state Λ is the X histogram of the circuit state. The gate uses the ground-state Λ above, not this column.

## Reading

The call is **PASS**. Dropping the cavity XX term cuts the ground-state cavity lock from the SAFE n=12 median Λ of 0.941 down to 0.559 / 0.748 / 0.851 at λ = 0.50 / 0.75 / 1.00.

On the frozen n=12 pool the product gap stays near 0.10 (0.110, 0.127, 0.099), and every instance on that pool is at least 0.041 above the best XZ product state. The median Z-peak falls with the field: 0.157, 0.038, 0.013, all below 0.5.

λ=0.50 and λ=0.75 are PASS. Λ is under 0.80 (0.559 and 0.748) and the gap is above 0.03. L=2 beats Fourier, R=0.751 and R=0.621, with every seed under 0.76. L=4 at those two fields is only a soft lead (R=0.904 and R=0.888). The PASS line accepts either depth, so L=2 carries them.

λ=1.00 is the KILL cell. Median Λ=0.851 is just above 0.85, and 51% of cavity-instance pairs sit above 0.85. The Z-peak is 0.013, so the state is spread in Z. The ratio at this field would have cleared the PASS line by itself (L=2 R=0.492, L=4 R=0.716; SNAP is below Fourier on every paired L=2 instance). The transverse field alone is enough to push Λ over 0.85 once λ=1.

At λ=0.50, 27% of the n=6 training instances have a Z-peak ≥ 0.5. On the n=12 freeze that fraction is 0.8%, and the median peak is 0.157.

The L=2 Hamming shell is only mildly bent off φ_k = a + bk (median RMS 0.045 at λ=0.50 and 0.110 at λ=0.75). Part of the lead over Fourier is the Walsh-graded cost: four γ weights per layer against Fourier's one. L=4 shells bend more (RMS 0.216 and 0.226 at those two fields). Prepared-state Λ sits above the ground-state Λ, near 0.78–0.90. The gate uses the ground-state number.

Per-seed JSON is `L{L}_{method}_s{seed}_l{λ}.json` in this folder. Angles are the restored n=9-holdout checkpoint. Ground-state Λ, the product gap, and the Z-peak are in `diag_n{n}_{role}_l{λ}.npz`.
