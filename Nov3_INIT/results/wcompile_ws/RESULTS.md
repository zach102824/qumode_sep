# Corrected-workspace W_BinH compile K=3 Q=2048

Common workspace 96, frozen replay 128. Real α, SNAP pad 0. Coherent L selector. Identity incumbent measured. SK ledger 2818048. Historical PR109 2048 compiler evals recorded separately. Not a hardware claim.

**WCOMP_WS_FAIL_BUDGET.**

Selected start 1 (last) L=1.1291 vs identity L=2.0000 (beats identity True). Compiler queries 2048. Wall 1.8s. SK queries 0.

## NF=96 fit

L 1.1291 F_HS 0.1896 F_avg 0.2253 F_worst 0.0866 (F_avg not selected).
p_code 0.6389 p_pad 0.3611 p_lost 4.025e-16 (sum 1.000000).
plus_fid 0.3309 random_fid 0.1138.
α = [-1.4473423307261888, -0.8974615669603153, 1.5338103629613484].

## NF=128 frozen replay

L 1.1291 F_HS 0.1896 ΔL 8.882e-16.
p_code 0.6389 p_pad 0.3611 p_lost 1.568e-15.
full-column overlap 1.000000 (worst 1.000000) code-block fro 3.185e-14.

PR109 remains history under repeated P_16 D P_16. This trial does not claim impossibility if FAIL_BUDGET. No extra K/α/SNAP after a miss.
