# N9-FT

Development screen for Idea 3 under the simulation definitions Zach approved at 18:41 Asia/Shanghai on 2026-11-09.

Approved model:

- SNAP phase is 0 on Fock levels above 7.
- A cost term that touches a leaked cavity is off. Every other term still acts (Uz).
- Leakage energy is per term: `|J_ij|` on a touched pair and `+b` on each leaked qubit.
- Readout in this simulator is ideal and logical. The compiled cost of `W_L` is not modelled.

This approval is not a hardware-feasibility claim. U0 (the whole component unphased if any cavity leaks) is still run as a sensitivity. Fidelity is unconditioned: no leakage renormalization and no postselection. Validation seeds `11000..11031` and final seeds `20000..20999` are not read.

The circuit is fixed: `|+> -> [U_cZ -> D(alpha_l) -> bare shell SNAP_l]^L`. One `gamma` and every feature coefficient are shared by all layers. `alpha` is real and shared across cavities. There are three shell controls per layer, also shared. Displacement is the true oscillator operator, built in `NF+32` and restricted. `NF=8` strict-logical `D8` is a diagnostic and is not called a displacement.

QAOA is standard `p=L` with independent angles. Nov3rd is the secondary baseline `|0> -> [U_c^X -> S]^L -> W`, checked against `|+> -> [U_cZ -> W S W]^L`. Both use SPSA only. The objective is energy. The ground state is used only when reporting fidelity.

Features are RMS-matched `J`, `K`, and `M`, plus `R1 = offdiag(lambda_min v_min v_min^T)` of `J`. `R1` is not an eigenfeature of `H`. The K-sign prior `theta_K^raw = b^2/2` is an initialization only. Transfer trains on dev seeds `10008..10015` and is tested on `10000..10007`. Training queries are counted and amortized.

The local stream reported one preliminary Nov3rd point with no QAOA pair: N=18, seed 10000, b=1, L=1, F_+ about 0.21317 to F about 0.61289 at 1024 queries. That point is not reproduced here.

```bash
OMP_NUM_THREADS=1 python -m Nov9_SK_SPSA.experiments.FT.conformance
OMP_NUM_THREADS=1 python -m Nov9_SK_SPSA.experiments.FT.run_ft --phase all
```
