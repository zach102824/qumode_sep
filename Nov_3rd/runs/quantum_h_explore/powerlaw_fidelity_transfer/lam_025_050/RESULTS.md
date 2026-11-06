# Power-law TFIM fidelity transfer — λ ∈ {0.25, 0.50}

**Verdict: KILL.**

This slice trains by ground-state fidelity, not by excess energy. Sibling runs own λ ∈ {0.75, 1.0, 1.5}. μ = 0, α = 1.5, J_ij ∈ {±1}.

## Bar

L* is the smallest L in {1,2,4,8,12,16} whose median-over-seeds of the median-over-instances fidelity is at least 0.95 on **train and holdout**, at **both** n=6 and n=9. F = |⟨ψ_GS|ψ(θ)⟩|².

PASS if some λ in this slice has an L* and the frozen n=12 fidelity is ≳ 0.70 with drop from min(train n6, train n9) ≲ 0.25. needs_edit if some λ hits that train bar and the freeze then falls through it. KILL if neither λ hits the train bar at any L ≤ 16. Fourier is a reference column. Beating it is not the bar. n=15 is reported beside n=12; the written bar is n=12.

## Hamiltonian

$$
H = \sum_{i<j} \frac{J_{ij}}{|i-j|^{1.5}} Z_i Z_j - \lambda \sum_i X_i
$$

Open cavity chain, same builder as `Nov_3rd/powerlaw_tfim.py` (PR #42). Qubit 0 is the MSB. Cavity c owns sites [3c, 3c+3). Bonds that cross a cavity stay in the sum. Circuit cost is exp(−i γ H_ZZ / n); ⟨H⟩ and E0 use the un-normalized H. H_ZZ is pure Walsh degree 2, so SNAP trains γ_2 only.

Two SNAP shells, both with γ_2 only. **B=2** is φ_k = c₁ k + c₂ k². **shell** is the PR #42 Hamming shell, free φ_k for k=0..3. Paper init is c₁=−β, c₂=0 (and φ_k=−β k), γ_2=−0.01, β=0.01. Fourier uses that same linear mixer (standard QAOA X rotation) and a scalar γ. At init all three states match; `library_check` asserts the B=2 and graded shells against Fourier. B=2 contains the Fourier mixer (c₂=0). The full shell contains B=2.

Mean cross-cavity share of Σ|K_ij| over 8 instances: n6=0.358, n9=0.456, n12=0.503.

Prior energy smoke (PR #42), best cell λ=0.50 μ=−0.50 L=4: SNAP Medε 0.209 at n=9 hold to 0.675 at n=12. That cell is not this Hamiltonian (it has an XX driver). On this μ=0, λ=0.50 cell the same energy smoke had n=9 SNAP Medε 0.544 and n=12 Medε 1.661, and the product gap failed the quantumness floor (n=9 hold peak 0.313, product gap 0.004). Those numbers are energy, not fidelity.

## Setup

| knob | value |
|---|---|
| λ | 0.25, 0.50 |
| μ, α | 0, 1.5 |
| ansatz | SNAP B=2 (γ_2, c1, c2), full Hamming shell (γ_2, φ_k), Fourier (β, γ) |
| L | 1, 2, 4, 8, 12, 16, stop a λ once SNAP hits L* |
| seeds | 0, 1, 2 |
| pools | train B=64 indices 0..63; holdout B=32 indices 64..95; freeze n=12 and n=15 B=32 indices 0..31 |
| loss | ½ (mean infidelity n6 + mean infidelity n9), minibatch 32+32 |
| optimizer | L-BFGS 40×8, lr 0.25, grad clip 5, paper init, single start. Stop early only after min4 ≥ 0.97 with 5 rounds of no improvement. +20 rounds if the best min4 lands in [0.90, 0.95) |
| checkpoint | maximize min(train6, hold6, train9, hold9) median F |
| instance seed | 1500003017 + n·100003 + index |
| n≤12 GS | exact (`ground_state`: dense eigh for n≤9, eigsh for n=12) |
| n=15 GS | exact eigsh, same helper, k=1, which=SA, start \|+⟩. Not approximate. |

## Phase

Diagnostics are the n=6 and n=9 **holdout** (B=32), the same index split as PR #42. Peak is max_z |ψ(z)|². IPR is Σ_z p(z)². Cat weight is p(z*)+p(flip z*), the global spin flip. q_EA = (1/n) Σ_i ⟨Z_i⟩². It is near zero in the even ground state because ⟨Z_i⟩ cancels between z* and its flip, so it is not the order parameter on this H. ⟨X⟩ is (1/n) Σ_i ⟨X_i⟩. Gap is E1−E0 from the dense spectrum. Product gap is (E_prod − E0)/|E0|.

| λ | n | peak | cat | IPR | q_EA | ⟨X⟩ | E1−E0 | product gap | E0 | #42 quantumness | reading |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.25 | 6 | 0.475 | 0.949 | 0.451 | 0.000 | 0.171 | 0.0004 | 0.0003 | -5.423 | fail (peak<0.5 on 100%, gap≥0.03 on 0%) | Z2-even cat (weight on z* and its global flip; ⟨Z⟩ cancels); weak ⟨X⟩; product state nearly matches E0; small many-body gap |
| 0.25 | 9 | 0.449 | 0.898 | 0.405 | 0.000 | 0.179 | 0.0000 | 0.0004 | -8.687 | fail (peak<0.5 on 100%, gap≥0.03 on 0%) | Z2-even cat (weight on z* and its global flip; ⟨Z⟩ cancels); weak ⟨X⟩; product state nearly matches E0; small many-body gap |
| 0.50 | 6 | 0.383 | 0.766 | 0.303 | 0.000 | 0.368 | 0.0208 | 0.0058 | -5.851 | fail (peak<0.5 on 100%, gap≥0.03 on 12%) | Z2-even cat (weight on z* and its global flip; ⟨Z⟩ cancels); intermediate ⟨X⟩; product state nearly matches E0; small many-body gap |
| 0.50 | 9 | 0.313 | 0.626 | 0.205 | 0.000 | 0.351 | 0.0031 | 0.0043 | -9.340 | fail (peak<0.5 on 100%, gap≥0.03 on 3%) | Z2-even cat (weight on z* and its global flip; ⟨Z⟩ cancels); intermediate ⟨X⟩; product state nearly matches E0; small many-body gap |

On n=9 holdout, λ=0.25 has peak 0.449, cat weight 0.898, ⟨X⟩ 0.179, gap 0.0000, product gap 0.0004. λ=0.50 has peak 0.313, cat weight 0.626, ⟨X⟩ 0.351, gap 0.0031, product gap 0.0043. Both sit on the ordered side of this α=1.5 glass: the ground state is the even cat of a classical bitstring, λ=0.25 more concentrated than λ=0.50, and a product state already matches E0. That product state has fidelity at most about half a cat, which is why an energy residual can be small while fidelity stays well below 0.95. These two λ are not a fitted critical point.

## Ladder

Each entry is the median over seeds 0, 1, 2 of that seed's median instance fidelity. min4 is the minimum of train6, hold6, train9, hold9. A dash means that depth was not trained because a smaller L* already cleared the bar for that λ.

| λ | L | B=2 train6 | hold6 | train9 | hold9 | B=2 min4 | B=2 hit | shell min4 | Fourier min4 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.25 | 1 | 0.3785 | 0.3801 | 0.1466 | 0.1466 | 0.1466 | no | 0.1502 | 0.1394 |
| 0.25 | 2 | 0.5781 | 0.5880 | 0.2911 | 0.2997 | 0.2911 | no | 0.3202 | 0.2616 |
| 0.25 | 4 | 0.7865 | 0.7652 | 0.4634 | 0.4968 | 0.4634 | no | 0.5541 | 0.3937 |
| 0.25 | 8 | 0.8363 | 0.8172 | 0.5153 | 0.5503 | 0.5153 | no | 0.7862 | 0.6474 |
| 0.25 | 12 | 0.8559 | 0.8288 | 0.5488 | 0.5816 | 0.5488 | no | 0.8413 | 0.7442 |
| 0.25 | 16 | 0.8626 | 0.8311 | 0.5491 | 0.5797 | 0.5491 | no | 0.8672 | 0.7741 |
| 0.50 | 1 | 0.5605 | 0.5526 | 0.3142 | 0.3146 | 0.3142 | no | 0.3167 | 0.3034 |
| 0.50 | 2 | 0.7581 | 0.7444 | 0.4743 | 0.4788 | 0.4743 | no | 0.4953 | 0.4548 |
| 0.50 | 4 | 0.8873 | 0.8674 | 0.6691 | 0.6665 | 0.6665 | no | 0.7097 | 0.6533 |
| 0.50 | 8 | 0.9212 | 0.9065 | 0.7139 | 0.7088 | 0.7088 | no | 0.8645 | 0.8232 |
| 0.50 | 12 | 0.9186 | 0.8998 | 0.7230 | 0.7168 | 0.7168 | no | 0.9344 | 0.8749 |
| 0.50 | 16 | 0.9295 | 0.9074 | 0.7310 | 0.7219 | 0.7219 | no | 0.9471 | 0.8793 |

## Freeze

Angles are the train checkpoint at the reported L. n=12 and n=15 are new instances (indices 0..31 at that n), not a retrain. Medε = median(⟨H⟩ − E0). Energies are the median over seeds of each seed's median instance value.

| λ | L | role | method | F train | F hold | F n12 | drop12 | F n15 | drop15 | Medε n9 hold | Medε n12 | Medε n15 | med ⟨H⟩ n12 | med E0 n12 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 16 | best, below 95% | snap | 0.5491 | 0.5797 | 0.2274 | 0.3217 | 0.0693 | 0.4798 | 0.6336 | 1.8037 | 3.6699 | -9.554 | -11.440 |
| 0.25 | 16 | best, below 95% | shell | 0.8672 | 0.8888 | 0.5300 | 0.3372 | 0.2926 | 0.5746 | 0.1380 | 0.5016 | 1.2154 | -10.981 | -11.440 |
| 0.25 | 16 | best, below 95% | fourier | 0.7741 | 0.8054 | 0.4288 | 0.3454 | 0.2037 | 0.5705 | 0.2102 | 0.7128 | 1.6963 | -10.869 | -11.440 |
| 0.50 | 16 | best, below 95% | snap | 0.7310 | 0.7219 | 0.3912 | 0.3398 | 0.1378 | 0.5932 | 0.3841 | 1.3504 | 2.7871 | -10.834 | -12.299 |
| 0.50 | 16 | best, below 95% | shell | 0.9478 | 0.9471 | 0.6969 | 0.2509 | 0.3183 | 0.6295 | 0.0774 | 0.5012 | 1.4301 | -11.829 | -12.299 |
| 0.50 | 16 | best, below 95% | fourier | 0.8793 | 0.8844 | 0.5639 | 0.3154 | 0.2516 | 0.6277 | 0.1381 | 0.7240 | 1.8958 | -11.531 | -12.299 |

### Per seed at the reported SNAP depth

| λ | method | L | seed | train6 | hold6 | train9 | hold9 | n12 | n15 | best round | wall s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | snap | 16 | 0 | 0.8552 | 0.8311 | 0.5374 | 0.5650 | 0.2198 | 0.0664 | 39 | 12 |
| 0.25 | snap | 16 | 1 | 0.8626 | 0.8299 | 0.5516 | 0.5844 | 0.2370 | 0.0732 | 38 | 12 |
| 0.25 | snap | 16 | 2 | 0.8800 | 0.8579 | 0.5491 | 0.5797 | 0.2274 | 0.0693 | 39 | 12 |
| 0.25 | shell | 16 | 0 | 0.9760 | 0.9694 | 0.8672 | 0.8888 | 0.5326 | 0.2926 | 39 | 26 |
| 0.25 | shell | 16 | 1 | 0.9756 | 0.9666 | 0.8614 | 0.8909 | 0.5269 | 0.2859 | 39 | 27 |
| 0.25 | shell | 16 | 2 | 0.9745 | 0.9656 | 0.8690 | 0.8865 | 0.5300 | 0.2978 | 38 | 27 |
| 0.50 | snap | 16 | 0 | 0.9370 | 0.9152 | 0.7402 | 0.7352 | 0.4035 | 0.1378 | 38 | 12 |
| 0.50 | snap | 16 | 1 | 0.9253 | 0.8979 | 0.7310 | 0.7219 | 0.3912 | 0.1399 | 38 | 12 |
| 0.50 | snap | 16 | 2 | 0.9295 | 0.9074 | 0.7213 | 0.7130 | 0.3798 | 0.1337 | 38 | 12 |
| 0.50 | shell | 16 | 0 | 0.9793 | 0.9782 | 0.9464 | 0.9469 | 0.6969 | 0.3153 | 94 | 68 |
| 0.50 | shell | 16 | 1 | 0.9786 | 0.9779 | 0.9478 | 0.9480 | 0.6961 | 0.3183 | 86 | 67 |
| 0.50 | shell | 16 | 2 | 0.9784 | 0.9771 | 0.9485 | 0.9471 | 0.6973 | 0.3187 | 86 | 68 |

## Call by λ

- **λ=0.25: no95.**
  - snap: cannot hit 95% at train. Best L=16 with min(train n6, train n9) F=0.5491. Freeze n12 F=0.2274, n15 F=0.0693.
  - shell: cannot hit 95% at train. Best L=16 with min(train n6, train n9) F=0.8672. Freeze n12 F=0.5300, n15 F=0.2926.
- **λ=0.50: no95.**
  - snap: cannot hit 95% at train. Best L=16 with min(train n6, train n9) F=0.7310. Freeze n12 F=0.3912, n15 F=0.1378.
  - shell: cannot hit 95% at train. Best L=16 with min(train n6, train n9) F=0.9478. Freeze n12 F=0.6969, n15 F=0.3183.

Slice call: **KILL**.

## Reading

λ=0.25: snap peaks at train F=0.5491 on L=16; shell peaks at train F=0.8672 on L=16. λ=0.50: snap peaks at train F=0.7310 on L=16; shell peaks at train F=0.9478 on L=16.

The only near miss is the full Hamming shell at λ=0.50, L=16. It took the automatic +20 rounds and then 40 more from that checkpoint (last logged round 99). The restored checkpoint is not the last round. Median train min F=0.9478, hold min F=0.9471, n12 F=0.6969, n15 F=0.3183. n=6 is already above 0.95; n=9 is the miss. That is still short of 0.95, so λ=0.50 has no L*. The frozen n=12 fidelity is already under 0.70, and n=15 falls further, so clearing the train bar with a longer run would still be a transfer miss on the written n=12 rule.

B=2 contains the Fourier mixer at c₂=0, but from the paper init it settles at a lower fidelity than Fourier. The full Hamming shell beats Fourier at every depth in this slice and is the ansatz the KILL refers to. Neither shell reaches a train L*.

JSON per (λ, L, method, seed) is `L{L}_{method}_s{seed}_l{λ}.json` in this folder. Angles are the restored checkpoint. Phase rows are `phase_l0.25.json` and `phase_l0.50.json`. `settings.py` is unchanged. n=21 was not loaded.

