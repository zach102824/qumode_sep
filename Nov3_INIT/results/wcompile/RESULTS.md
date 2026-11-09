# W_BinH SNAP + true-D compile feasibility

Real α, K=3 (D then SNAP), SNAP pad 0, true D from NF+32. SPSA Q=2048 compiler evals. SK ledger unchanged at 2818048. Not flat prep. Not U_X. Not cropped D8. Not a hardware claim.

**WCOMP_FAIL_BUDGET.**

Selected start 0 (polyak64). Compiler queries 2048. Wall 0.4s. SK queries 0.

## NF=16 (optimized)

F_avg 0.1169 F_worst 0.0166 F_HS 0.0841 L 0.8220.
p_leak mean 6.723e-01 worst 7.403e-01. plus_fid 0.0070 random_fid 0.0815.
mean column norm 0.402000 (no renormalization).

Identity compile (start 0 init) had F_avg=0.125, F_HS=0, L=2.000. SPSA lowered L by driving real α to the tanh rails (1.762, 1.768, 1.799) and leaking: mean p_leak 67%, mean ‖U|k⟩‖=0.40 so the restricted D is far from unitary on NF=16. Basis infidelities stay O(1). Superposition/phase checks fail (plus_fid 0.007).

## NF=32 frozen replay (no reopt)

F_avg 0.1120 F_worst 0.0161 F_HS 0.0800. ΔF_avg 4.884e-03. replay_ok False.
p_leak mean 6.738e-01 worst 7.402e-01.

Truncation is not converged at the 10^{-3} bar, and the pulse is not a W_BinH isometry at either cutoff.

Global phase: one overall arg(tr(T† V)). Per-column phases are not free. Complex α and pad SNAP were not added.

Drop-in bars (all required): F_avg≥0.99, F_worst≥0.95, mean p_leak≤0.01, worst p_leak≤0.05, F_HS≥0.99, replay |ΔF_avg|≤1e-3.

## Resource accounting

| ledger | count |
|---|---|
| SK energy queries this task | 0 |
| historical SK total | 2818048 |
| compiler evaluations | 2048 inclusive (init 4, cal 32, iter 2000, candidate 12) |
| pulse depth | K=3 pairs, 27 parameters |
| fit cutoff | NF=16 |
| replay cutoff | NF=32, frozen (u,θ), no reopt |

Pulse artifact: `results/wcompile/pulse.json`.

## Remaining limitations (not swept)

- Complex α exists in `compile_flat_prep` for **one** column and is D7 "not now" for this stream. It was **not** added. Failure at real α does not prove complex α would succeed.
- SNAP phases on f≥8 stay 0 (D2). Not added.
- Depth K and NF_opt were one preregistered setting, not a search.
- Cropped D8 was not used (not a displacement).
- A numerical compile, even a future candidate, is not a hardware claim.
- SK energy fits were not launched. PR106 pad invariance and PR107 G_{7,8}(0.10) sensitivity labels unchanged.

Fit workers 1, BLAS 1.
