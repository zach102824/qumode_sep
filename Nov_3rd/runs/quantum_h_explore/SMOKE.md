# SMOKE — quantum H explore protocol

_Asia/Singapore 2026-11-04 ~21:50_

**Do not launch from this doc until Zach says GO.** No cloud agents. Do not touch
n21 phase2 ladder (`PID ~143971`, `run_train69_ladder_fresh250_phase2.py --pass n21`).

## One-liner
`n∈{6,9} × L∈{2,16} × λ-grid × {power_p SNAP, Walsh-degree SNAP, Fourier QAOA}` on family **A** first; exact \(E_0\) via dense eigh; score Med \(\langle H\rangle/n\) + gap-to-\(E_0\).

## Scope
| knob | value |
|------|-------|
| families | **A** first; if time **D**; **B** only if A inconclusive |
| \(n\) | `{6, 9}` only |
| \(L\) | `{2, 16}` |
| methods | (1) Hamming-shell SNAP **power_p** (2) SNAP **Walsh-degree** γ (3) Fourier QAOA fair same-\(L\) |
| B | train `B_tr=64` (or 128 if cheap); holdout `B_ho=128` from scale1000 |
| seeds | train `{0,1,2}`; report best-by-hold Med |
| λ (A) | `{0.0, 0.25, 0.5, 1.0, 2.0}` |
| λ (D) | `{0.25, 0.5, 1.0, 2.0}` + signs from `cls_s` or ±1 redraw seed=0 |
| device | box CPU/GPU already used by Nov_3rd; **low priority** vs n21 |

## Hamiltonian construction (family A)
```text
E[b, x]  = unsat count from pack  # shape (B, 2^n), uint8/float
H_c      = diag(E[b])             # optional: center/scale as cost_bitstring does
H        = H_c + λ * sum_i X_i    # dense (2^n × 2^n) at n6/n9
E_0, ψ_0 = eigh(H)               # lowest eigenvalue / vector per instance
```
- Packs: `Nov_3rd/data/scale1000/data_n{6,9}.npz` keys `scale1000_E`, optional `scale1000_cls_*`.
- Reuse `Nov3rdAnsatz.cost_Cn` / `cost_bitstring` for the **diagonal pulse** only.
- Field term: implement as sparse Kronecker sum of \(X\) (or dense at n≤9).

## Expectation from Z-hist + X-hist
\[
\langle H\rangle = \sum_z p_Z(z)\, H_c(z) + \lambda\sum_{i=1}^n \langle X_i\rangle,
\qquad \langle X_i\rangle = 2p_{X_i}(+)\!-\!1.
\]
Circuit: same forward as Nov_3rd → measure in Z (Fock/bitstring) for \(p_Z\); for each \(i\), basis-change \(H_i\) then measure for \(p_{X_i}\). **No YY.**

## Exact oracle (n6/n9)
- Dense `numpy.linalg.eigh` / `torch.linalg.eigh` on Hermitian \(H\) (64×64 / 512×512).
- Record per instance: \(E_0\), \(\max_x|\langle x|\psi_0\rangle|^2\) (bitstring peak), \(\sum_i\langle X_i\rangle_{\psi_0}\).
- **Quantum check:** peak bitstring prob \(< 0.5\) (or clearly \(<1\)) at chosen λ; else λ too small.

## Train / eval protocol
1. **Build** holdout table once: for each (n, λ, instance): \(E_0\), classical \(E_{\min}(H_c)\).
2. **QAOA angles:** train Fourier QAOA (β,γ per layer) w.r.t. \(\langle H\rangle\) on \(B_tr\), init paper `(β,γ)=(0.01,−0.01)`, L-BFGS; **separate** from classical Max-E `qaoa_angles/` (do not overwrite).
3. **SNAP warm:** dual map \(\gamma\leftarrow\gamma_{\mathrm{QAOA}}\), \(\phi_j\leftarrow -\beta j\); then polish power_p and/or Walsh-degree on same \(B_tr\) objective \(\langle H\rangle\).
4. **Eval** on \(B_ho\): Med / mean \(\langle H\rangle\), \(\langle H\rangle/n\), gap \(\langle H\rangle-E_0\), R = Med_SNAP / Med_QAOA.
5. **Gate:** BEAT if R≤0.98 on both n6 and n9 at that λ; TIE (0.98,1.02]; UNDER >1.02.
6. Log JSON+md under `runs/quantum_h_explore/smoke_A/` (create when launching).

## Budget / stop
- Cap wall ~**30–60 min** box time total for A×{L=2,16}×λ grid (skip λ if GS classical).
- Stop early if λ=0.5 and λ=1.0 both UNDER for all methods → try **D** signs, else report A weak.
- **Never** start n12+ or L>16 in smoke.
- **Never** `kill` / renice n21 ladder.

## Code touch list (when GO — not now)
| action | path |
|--------|------|
| NEW smoke script | `Nov_3rd/scripts/smoke_quantum_h_A.py` (suggested) |
| NEW out dir | `Nov_3rd/runs/quantum_h_explore/smoke_A/` |
| REUSE | `ansatz.py`, `data_io.py`, sealed/scale1000 packs |
| DO NOT WRITE | `qaoa_angles/L*/`, `train69_ladder_fresh250/`, `settings.py` locks |

## Pass / fail summary template
```text
family=A  n=6/9  L=2/16  λ=?  method=power_p|walsh|qaoa
Med_H, Med_H/n, gap, R, bitstring_peak_GS, verdict=BEAT|TIE|UNDER|KILL
```

## Status
- Protocol drafted only. **No smoke process started.**
- Ladder n21 still running as of doc write.
