# Binary 3-bit cavity, exactly 2 SNAP + 2 displacements

Encoding `|n⟩ = |b2 b1 b0⟩` for n=0..7, b0 = LSB. SNAP phases free on 0..7 and **fixed at 0 for n≥8**. Displacements are the exponential of the NF-truncated generator. Headline amplitude cap `|α|≤1.5` (repo cavity ansatz soft-clips at 1.8). Sensitivity at 0.8 and 2.2.

Only alternating words spend the whole budget. `D(a)D(b)` is one displacement up to a global phase, and two SNAPs in a row are one SNAP.

| order | physical time (first → last) | written product |
|---|---|---|
| SDSD | S1, D1, S2, D2 | U = D2 S2 D1 S1 |
| DSDS | D1, S1, D2, S2 | U = S2 D2 S1 D1 |

DSDS ends on a SNAP, which cannot change a photon-number measurement, so its readout uses only D,S,D. SDSD can absorb a uniform Z rotation on the measured bit into S1 (X↔Y) without an extra pulse. An output diagonal does not move the measured axis at all.

**F_proc** = `|Tr(U_tgt† M)|²/64` with `M_ij=⟨i|U|j⟩` on levels 0..7 (mixcal subspace process fidelity, global phase free). Leakage lowers it. **F_avg** is the Haar state fidelity with leakage as failure.

**eps_X** = max of the two operator norms `‖E± − (I±X_b)/2‖_op`. E± only count computational levels with the chosen bit; n≥8 is a third outcome (no postselection). This is the worst-case binary probability error over every state in the 8-level space, including spectator entanglement. eps_X on the X eigenstates alone is recorded and is not the claim. Classical bin-label flip is `eps_X_basis`.

`H = Ry(+π/2) Z` on the encoded bit. Both measure the X basis (`Ry(+π/2)` yields −X, a label flip). They are not the same gate. On SDSD the extra Z folds into S1; on DSDS it would be an extra pulse.

Numbers below are at the **largest validation truncation**, same physical controls as the NF used in the fit. These are best-of-multistart values, not upper bounds.

## Table

| bit | order | F_proc(H) | F_avg(H) | F_proc(Ry) | eps_X basis | eps_X signed | leak worst state | retain worst | α from Z (deg) | single-bit residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lsb | SDSD | 0.9154 | 0.9242 | 0.9154 | 0.2652 | 0.2652 | 0.1961 | 0.8039 | 78.5 | 0.4943 |
| lsb | DSDS | 0.9154 | 0.9242 | — | 0.3269 | 0.3269 | 0.2273 | 0.7727 | 76.6 | 0.6210 |
| mid | SDSD | 0.6293 | 0.6677 | 0.6293 | 0.5285 | 0.9795 | 0.4501 | 0.5499 | 51.5 | 0.7049 |
| mid | DSDS | 0.6293 | 0.6677 | — | 0.9944 | 0.9944 | 0.9915 | 0.0085 | 88.0 | 0.8247 |
| msb | SDSD | 0.6066 | 0.6477 | 0.6066 | 0.6866 | 0.6866 | 0.3410 | 0.6590 | 32.8 | 0.9443 |
| msb | DSDS | 0.6066 | 0.6477 | — | 0.8614 | 0.8614 | 0.7136 | 0.2864 | 46.1 | 0.8121 |

F_proc(Ry) is the fit that targeted Ry, not the Ry fidelity of the H fit. eps is the better of the Frobenius measurement fit and the operator-norm fit (`meas_op`), validated at the largest NF. It is not inferred from the gate fit. Doing nothing (identity) measures Z and has eps_X = 1/√2 ≈ 0.707 with zero leakage; a larger eps is not a useful X readout.

## LSB candidate

Best verified LSB readout is SDSD `meas_op` (physical S1, D1, S2, D2; written U = D2 S2 D1 S1), same controls at NF=80. eps_X=0.2652 (basis 0.2652), leak worst/mean 0.1961/0.0252, retain worst 0.8039, |α|=(0.336, 0.245). X-eigenstate-only error=0.1466, which is smaller than eps_X; the operator norm is the claim. Loss best/median 0.6122/1.0694 over 52 starts, and polish perturbations returned to the same loss. Truncation drift of eps_X is 1.4e-15.

Pulled-back observable's single-bit Bloch vector is n=(0.899, 0.003, 0.182), angle from Z 78.5°. Residual op-norm after removing that single-bit part and the identity offset: 0.494. That residual is spectator-correlated Pauli weight, so this is not a clean encoded-qubit axis. Folding an extra Z rotation into S1 (free on SDSD) does not remove it: the azimuth is already ~0.

Largest Pauli coefficients of Z_pull (|c|≥0.02):

`IIX`=0.899, `IIZ`=0.182, `IYZ`=-0.163, `IXY`=0.115, `IZZ`=0.106, `YXZ`=-0.090, `XYZ`=0.085, `XIZ`=0.084, `ZZZ`=0.078, `YYY`=0.076, `ZXY`=-0.069, `ZIZ`=0.059, `XXY`=0.057, `ZYZ`=0.054, `IXZ`=0.034, `XXZ`=-0.032.

Spectator blocks of Z_pull, (n_X, n_Y, n_Z):

- spec 0 levels [0, 1]: (0.892, 0.002, 0.425), radius 0.988
- spec 1 levels [2, 3]: (0.952, 0.001, 0.057), radius 0.954
- spec 2 levels [4, 5]: (0.901, 0.010, 0.151), radius 0.913
- spec 3 levels [6, 7]: (0.852, -0.003, 0.095), radius 0.857

Worst state for the larger outcome error: p+=0.384 (ideal 0.649), p-=0.575 (ideal 0.351), p_leak=0.040.

## Notes

- Guard SNAP phases are identically 0. Controlling n≥8 would be a different model (mixcal’s internal SNAPs do that) and was not used.
- No terminal SNAP was added outside the two. F_proc_extra_output_SNAP in the JSON is the trace overlap if one were allowed; it is not a gate fidelity claim.
- Identity is an exact Z measurement (α=0, eps_X=1/√2≈0.707). A free-angle fit is reported only through the achieved Bloch angle and the residual to a single-bit axis, not by declaring a tiny loss against Z.
- Prior Gray-mixer mixcal table (free outer SNAPs, internal SNAP on all levels, unbounded α, target ∏ exp(−iβ X/2) in Gray code) is a different problem. K3 Fmin over β was quoted in `oct_explore/NATIVE_TEMPLATE.md` as k=2: 0.13, k=6: 0.83, k=8: 0.9987. The K3 JSON is not in the tree. This study does not reproduce that fit.
- MSB H at `|α|≤1.5` saturates both displacements (`|α|=1.5`). Raising the cap to 2.2 moves the best-found F_proc(H) only from 0.607 to 0.641, with `|α|≈1.90` still inside 2.2. The cap is weakly active for MSB and inactive for LSB (best `|α|≈0.18` for the gate, `≈0.31` for the readout).
- MSB X-readout at `|α|≤2.2` has eps_X=0.689 versus 0.690 at `|α|≤1.5`. Opening the cap does not fix MSB.
- Practical recommendation: do not use 2 SNAP+2 D as an X-basis readout. LSB is a partial, spectator-correlated rotation (signed eps_X≈0.265, residual ≈0.49), not a 0.01 X measurement. Middle's unfolded operator-norm fit has signed eps_X≈0.544. The table's 0.53 basis entry is that fit after a free S1 Z-rotation plus a classical label flip; the signed error of the flipped representative is ≈0.98, and the residual to a single-bit axis stays ≈0.70 with worst-case leakage ≈0.45. MSB stays at ≈0.69. Coherent LSB H reaches F_proc≈0.915 (F_avg≈0.924) and is a different pulse from the best readout; middle/MSB H are F_proc≈0.63/0.61 (0.64 for MSB if `|α|≤2.2`). None of these is a 0.99 gate.

