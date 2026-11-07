# RESULTS — family A transverse-field Max-E3LIN2

**Verdict: PASS**

## Hamiltonian

Opus sign, rank-1 transverse-field Max-E3LIN2:

\[
H = \sum_{(a,b,c)\in E} J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i
\]

Packs store 4-SAT unsat tables, not 3-local `cls_s`. Each instance keeps cross-cavity triples taken from its 4-clauses (variables not all in one 3-qubit cavity). `J=±1` is the product of those three literal signs, first occurrence wins. Greedy selection caps degree at 3 so `m=n` when the geometry allows it. Train seeds are the existing `beat_qaoa_12h/train_packs` rows; regenerated unsat `E` matches the pack. Holdout seeds use the scale1000 base `54_000_000_000` (`regen_scale1000.py`); the scale1000 npz is not in this checkout. Classical `qaoa_angles/` and `train69_ladder_fresh250/` were not written.

Gate sequence is the locked Nov_3rd motif: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W` with BinH `W=H^{⊗3}`. `U_c^X = W exp(-iγ H_c) W`. The field is not inside `U_c`. ⟨H⟩ = ⟨H_c⟩ from the Z-histogram (after the final W) minus λ Σ ⟨X_i⟩ from the X-histogram (final W omitted), ⟨X_i⟩ = 2 p(+)-1.

Walsh-degree γ is a null knob here: `H_c` is pure degree 3, so the degree-1/2/4 Walsh pieces are identically zero and graded γ collapses to scalar γ. It was not trained.

## Exact diagonalization

Smoke λ = **0.5**. Entropy-peak λ_c: n6=0.25, n9=0.25. Fidelity-susceptibility peak: n6=0.25, n9=0.5.

Quantumness on train+holdout at the smoke λ (median gate):

- n=6: med peak=0.109, med product gap=0.073, fraction with product gap≥2%=0.69, med cavity entropy=1.711 bits, pass=yes.
- n=9: med peak=0.087, med product gap=0.035, fraction with product gap≥2%=0.66, med cavity entropy=1.940 bits, pass=yes.

Gate: median bitstring peak of the ground state < 0.5 and median product-state gap (E_prod−E_0)/|E_0| ≥ 2%. Product states are XZ-plane, ⟨Z⟩=cos θ, ⟨X⟩=sin θ.

| n | λ | med peak | med prod gap | med S (bits) | med χ | med E0/n | large-λ e | pass |
|--:|--:|---------:|-------------:|-------------:|------:|---------:|----------:|:----:|
| 6 | 0.25 | 0.156 | 0.021 | 1.910 | 1.447 | -0.702 | -0.917 | yes |
| 6 | 0.50 | 0.109 | 0.073 | 1.711 | 1.110 | -0.807 | -0.833 | yes |
| 6 | 0.75 | 0.074 | 0.126 | 1.328 | 0.510 | -0.968 | -0.972 | yes |
| 6 | 1.00 | 0.054 | 0.143 | 0.942 | 0.197 | -1.166 | -1.167 | yes |
| 6 | 1.25 | 0.043 | 0.096 | 0.722 | 0.082 | -1.383 | -1.383 | yes |
| 6 | 1.50 | 0.037 | 0.069 | 0.545 | 0.039 | -1.611 | -1.611 | yes |
| 9 | 0.25 | 0.115 | 0.009 | 2.509 | 1.360 | -0.803 | -0.917 | no |
| 9 | 0.50 | 0.087 | 0.036 | 2.228 | 1.559 | -0.878 | -0.833 | yes |
| 9 | 0.75 | 0.052 | 0.078 | 1.776 | 1.490 | -1.007 | -0.972 | yes |
| 9 | 1.00 | 0.027 | 0.116 | 1.265 | 0.741 | -1.184 | -1.167 | yes |
| 9 | 1.25 | 0.016 | 0.101 | 0.903 | 0.255 | -1.392 | -1.383 | yes |
| 9 | 1.50 | 0.011 | 0.072 | 0.659 | 0.098 | -1.616 | -1.611 | yes |

## Train

Fourier QAOA: Zhou sine modes, q=L (full schedule, 2*L coefficients), mixer is the linear Hamming shell φ_j=−β j, i.e. the QAOA point `exp(-i β X/2)`.
power_p SNAP: φ_k = c1 k + c2 k² + c3 k³ on each cavity (4L parameters) plus scalar γ. Trained per n, so the cross-n exponent p is idle (c2, c3 already absorb a per-n scale). Fourier init is the Boulebnane–Montanaro point (β,γ)=(0.01,−0.01) plus N(0,0.01²) seed jitter. power_p is warm-started from that seed's trained Fourier dual (γ, c1=−β, c2=c3=0) and polished; the warm checkpoint is kept if polish does not improve holdout median ε. Optimizer is L-BFGS.

B_train=32, B_hold=48, seeds=[0, 1, 2], L=[2, 16].

| n | L | method | params | Med ε | seed std | Med f | R | verdict |
|--:|--:|:-------|------:|------:|---------:|------:|--:|:--------|
| 6 | 2 | fourier_qaoa | 4 | 0.0440 | 0.0000 | 0.857 |  |  |
| 6 | 2 | power_p | 8 | 0.0440 | 0.0000 | 0.857 | 1.000 | TIE |
| 6 | 16 | fourier_qaoa | 32 | 0.0096 | 0.0009 | 0.969 |  |  |
| 6 | 16 | power_p | 64 | 0.0086 | 0.0006 | 0.972 | 0.889 | BEAT |
| 9 | 2 | fourier_qaoa | 4 | 0.0970 | 0.0000 | 0.734 |  |  |
| 9 | 2 | power_p | 8 | 0.0969 | 0.0000 | 0.734 | 0.999 | TIE |
| 9 | 16 | fourier_qaoa | 32 | 0.0127 | 0.0022 | 0.965 |  |  |
| 9 | 16 | power_p | 64 | 0.0098 | 0.0011 | 0.972 | 0.770 | BEAT |

ε = (⟨H⟩ − E_0)/n on holdout, median over instances, then median over seeds. R = Med ε_SNAP / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better.

Per-seed holdout Med ε:

- `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0096, 0.0084, 0.0102
- `n6_L16_power_p` seeds [0, 1, 2]: 0.0086, 0.0084, 0.0095
- `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0439, 0.0440, 0.0440
- `n6_L2_power_p` seeds [0, 1, 2]: 0.0439, 0.0440, 0.0440
- `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0096, 0.0140, 0.0127
- `n9_L16_power_p` seeds [0, 1, 2]: 0.0079, 0.0099, 0.0098
- `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0969, 0.0970, 0.0970
- `n9_L2_power_p` seeds [0, 1, 2]: 0.0969, 0.0969, 0.0969

## Kill checks (L=16)

- Within one seed-std on both n: no
- Advantage shrinks n6→n9: no (adv n6=0.00107 n9=0.00292 noise=0.00177)
- Both methods f>0.99 on both n: no
- Fired: none

L=16 margin: n6 R=0.889 (ε 0.0086 vs 0.0096, gap 0.0011, seed std 0.0009); n9 R=0.770 (ε 0.0098 vs 0.0127, gap 0.0029, seed std 0.0022).

## Self-check

Fock twin vs `Nov3rdAnsatz` max |Δp| Z=2.07e-08, X=4.60e-07. QAOA-point vs ideal qubit mixer max |Δp|=4.44e-10. X-histogram vs Pauli ⟨X⟩ max |Δ|=5.00e-15.

## Notes

- n=6: extracted m median 6 (min 6, max 6), degree ≤ 3. Cap is 3, so a few n=9 instances stop at m=8 instead of raising the degree.
- n=9: extracted m median 9 (min 8, max 9), degree ≤ 3. Cap is 3, so a few n=9 instances stop at m=8 instead of raising the degree.
- power_p has 4L parameters and Fourier QAOA has 2L (q=L). power_p is polished from the trained Fourier dual, so the comparison asks whether c2 and c3 shells beat the mixer they contain. It is not a matched-parameter claim.
- lambda_c is the grid point with the highest median cavity-0 entanglement. The smoke lambda is the grid point nearest 0.9 lambda_c that still passes the quantumness gate on both n when one exists.
- Holdout checkpoint selection uses the holdout median residual (protocol best-by-hold). The same rule is applied to both methods.

Wall clock 107.2s. Branch `cursor/quantum-h-smoke-a-7099`.
