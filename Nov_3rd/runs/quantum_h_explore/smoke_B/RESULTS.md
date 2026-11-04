# RESULTS — quantum H explore, arm B (sparse p=3 TFIM)

_Finished 2026-11-04 14:54:21. Verdict: **needs_edit**._

## Setup

Locked gates: BinH `U_c^X = W exp(-i γ H_c) W` with `W = H^{⊗3}`, then bare Hamming-shell SNAP. `power_p` shell `φ_k = c1 k + a_n c2 k² + a_n c3 k³`, `a_n = (8/(n-1))^p`. Fourier QAOA is ideal-qubit QAOA at the same L: per-layer `(β, γ)` from paper init `(0.01, −0.01)`, cost pulse on the **same** normalized `Cn(H_c)`, mixer `exp(-i β X/2)`.

Hamiltonian (stoquastic sign, `| + ⟩^{⊗n}` lowers the field):

\[ H = H_c - \lambda \sum_i X_i, \quad H_c = -s^{-1}\sum_{(i,j,k)\in T} J_{ijk} Z_i Z_j Z_k, \quad s=\sqrt{|T|/n},\; J=\pm 1. \]

Readout is Z-histogram of `H_c` plus single-qubit `⟨X_i⟩`. No YY or XX correlators. `ε = (⟨H⟩ − E_0) / n`. BEAT means holdout median `ε` of power_p ≤ `0.98 ×` the same-split Fourier QAOA median, on **both** n=6 and n=9.

Instances: satisfiable Max-E 4-SAT supports from `sat_gen` base `54000000000` (scale1000 protocol), first `B_tr=16` train / next `B_ho=32` holdout. λ chosen on the **train** ED split only.

Chosen **λ = 1.0** (max min(ε_class, ε_plus) inside the quantum window).

The file `opus_reply_20261104_2150_hamiltonian_families.md` was not in the repository. Family locks are from `PLAN.md` / `SHORTLIST.md` / `SMOKE.md` (same 2026-11-04 ~21:50 note). Rank2 intra-cavity XX/XXX is not trained: XX correlators fail the Z+X readout lock, so it is not a fallback path.

## Why GRAPE of e^{-i H_c} is the right pulse and not the answer

`H_c` is diagonal in the computational basis, so `exp(-i γ H_c)` is the Fock diagonal inside `U_c^X` — the pulse GRAPE synthesizes. `⟨H_c⟩` on `| + ⟩` is 0 (each ZZZ averages to 0). Any diagonal unitary leaves the Z histogram of `| + ⟩` uniform, and `|⟨X_i⟩| ≤ 1`, so

\[ \langle + | e^{i\gamma H_c} H e^{-i\gamma H_c} | + \rangle \ge -\lambda n = \langle H\rangle_{|+\rangle}. \]

A diagonal pulse alone cannot undercut `| + ⟩`. The GS sits below both `| + ⟩` and the best bitstring whenever the table shows `ε_plus > 0` and `ε_class > 0`. Hamming-shell SNAP (or the QAOA X mixer) is what moves Z-population. That is the GRAPE justification: the cost layer is exactly `e^{-i H_c}`, and it does not solve `H`.

## ED λ grid (train split, reuse supports)

| n | λ | med peak | frac peak<0.5 | med ε_class | med ε_plus | med gap | quantum |
|---|---|----------|---------------|-------------|------------|---------|---------|
| 6 | 0.00 | 1.000 | 0.00 | 0.000 | 0.962 | 0.5774 |  |
| 6 | 0.15 | 0.981 | 0.00 | 0.005 | 0.818 | 0.5700 |  |
| 6 | 0.30 | 0.930 | 0.00 | 0.021 | 0.687 | 0.5500 |  |
| 6 | 0.50 | 0.833 | 0.06 | 0.058 | 0.528 | 0.5098 |  |
| 6 | 0.70 | 0.705 | 0.25 | 0.117 | 0.389 | 0.4852 |  |
| 6 | 1.00 | 0.422 | 0.88 | 0.262 | 0.230 | 0.6663 | yes |
| 6 | 1.50 | 0.101 | 1.00 | 0.661 | 0.125 | 1.5177 | yes |
| 6 | 2.00 | 0.055 | 1.00 | 1.126 | 0.089 | 2.7355 | yes |
| 6 | 3.00 | 0.033 | 1.00 | 2.095 | 0.057 | 5.0776 | yes |
| 6 | 4.00 | 0.027 | 1.00 | 3.080 | 0.042 | 7.2876 | yes |
| 9 | 0.00 | 1.000 | 0.00 | 0.000 | 0.990 | 0.7913 |  |
| 9 | 0.15 | 0.987 | 0.06 | 0.005 | 0.845 | 0.7826 |  |
| 9 | 0.30 | 0.949 | 0.06 | 0.020 | 0.712 | 0.7493 |  |
| 9 | 0.50 | 0.860 | 0.06 | 0.055 | 0.549 | 0.6968 |  |
| 9 | 0.70 | 0.706 | 0.12 | 0.111 | 0.404 | 0.6295 |  |
| 9 | 1.00 | 0.402 | 0.62 | 0.245 | 0.232 | 0.4921 | yes |
| 9 | 1.50 | 0.035 | 1.00 | 0.633 | 0.123 | 1.4554 | yes |
| 9 | 2.00 | 0.013 | 1.00 | 1.097 | 0.088 | 2.7683 | yes |
| 9 | 3.00 | 0.006 | 1.00 | 2.067 | 0.057 | 5.1220 | yes |
| 9 | 4.00 | 0.004 | 1.00 | 3.052 | 0.042 | 7.3173 | yes |

Quantum window (both n): `[1.0, 1.5, 2.0, 3.0, 4.0]`.

At λ=1.0: med `(E_class−E0)/n` = 0.2536, med `(E_+−E0)/n` = 0.2309, med bitstring peak = 0.412. GRAPE-alone gap is open: **True**.

## Supports: reuse vs regenerate

| n | density | med \|T\| | med Jaccard(reuse, regen) | med sign agreement on ∩ |
|---|---------|-----------|-----------------------------|--------------------------|
| 6 | full | 18.0 | 1.000 | 0.444 |
| 6 | sparse | 12.0 | 0.500 | 0.472 |
| 9 | full | 57.0 | 0.517 | 0.523 |
| 9 | sparse | 18.0 | 0.125 | 0.500 |

Full-hypergraph median Jaccard(reuse, regen) is n=6 1.00, n=9 0.52. Sparse cap |T|=2n Jaccard is n=6 0.50, n=9 0.12. n=6 has only 18 cross-cavity triples and m=60 clauses, so the uncapped reuse set is the complete cross-cavity hypergraph and regenerating T only rerolls signs. n=9 has 81 cross-cavity triples and the load-balanced clause extraction does not fill all of them, so regen changes the support. The |T|=2n rows are the matched-cardinality comparison where supports differ at both sizes. Intra-cavity triples are not used.

## Holdout smoke

Medians of `ε` on the holdout (B=32). R = Med ε(power_p) / Med ε(Fourier QAOA). QAOA is trained **per n**. `joint` is the train{6,9} gate. `transfer` fits n=6 only and deploys at n=9 with `a_n`. `per_n` is a capacity check (separate shell at each n), not the gate.

| graph | density | L | n | train | Med ε SNAP | Med ε QAOA | R | verdict |
|-------|---------|---|---|-------|------------|------------|---|---------|
| reuse | full | 2 | 6 | joint | 0.0600 | 0.0598 | 1.0031 | TIE |
| reuse | full | 2 | 9 | joint | 0.0586 | 0.0584 | 1.0037 | TIE |
| reuse | full | 2 | 6 | transfer | 0.0590 | 0.0598 | 0.9873 | TIE |
| reuse | full | 2 | 9 | transfer | 0.0741 | 0.0584 | 1.2686 | UNDER |
| regen | full | 2 | 6 | joint | 0.0567 | 0.0570 | 0.9947 | TIE |
| regen | full | 2 | 9 | joint | 0.0590 | 0.0588 | 1.0041 | TIE |
| regen | full | 2 | 6 | transfer | 0.0561 | 0.0570 | 0.9852 | TIE |
| regen | full | 2 | 9 | transfer | 0.0761 | 0.0588 | 1.2953 | UNDER |
| reuse | full | 2 | 6 | per_n | 0.0589 | 0.0598 | 0.9862 | TIE |
| reuse | full | 2 | 9 | per_n | 0.0584 | 0.0584 | 0.9998 | TIE |
| reuse | full | 16 | 6 | joint | 0.0186 | 0.0280 | 0.6645 | BEAT |
| reuse | full | 16 | 9 | joint | 0.0282 | 0.0272 | 1.0348 | UNDER |
| reuse | full | 16 | 6 | transfer | 0.0219 | 0.0280 | 0.7843 | BEAT |
| reuse | full | 16 | 9 | transfer | 0.0480 | 0.0272 | 1.7624 | UNDER |
| regen | full | 16 | 6 | joint | 0.0191 | 0.0263 | 0.7278 | BEAT |
| regen | full | 16 | 9 | joint | 0.0314 | 0.0340 | 0.9255 | BEAT |
| regen | full | 16 | 6 | transfer | 0.0232 | 0.0263 | 0.8824 | BEAT |
| regen | full | 16 | 9 | transfer | 0.0382 | 0.0340 | 1.1250 | UNDER |
| reuse | full | 16 | 6 | per_n | 0.0200 | 0.0280 | 0.7159 | BEAT |
| reuse | full | 16 | 9 | per_n | 0.0229 | 0.0272 | 0.8409 | BEAT |
| reuse | sparse | 2 | 6 | joint | 0.0442 | 0.0438 | 1.0094 | TIE |
| reuse | sparse | 2 | 9 | joint | 0.0457 | 0.0455 | 1.0034 | TIE |
| reuse | sparse | 2 | 6 | transfer | 0.0415 | 0.0438 | 0.9465 | BEAT |
| reuse | sparse | 2 | 9 | transfer | 0.0659 | 0.0455 | 1.4470 | UNDER |
| regen | sparse | 2 | 6 | joint | 0.0480 | 0.0478 | 1.0043 | TIE |
| regen | sparse | 2 | 9 | joint | 0.0529 | 0.0526 | 1.0062 | TIE |
| regen | sparse | 2 | 6 | transfer | 0.0477 | 0.0478 | 0.9976 | TIE |
| regen | sparse | 2 | 9 | transfer | 0.0613 | 0.0526 | 1.1666 | UNDER |

### Alternate λ = 1.5 (L=2 only, because λ* was not a dual-n BEAT)

| graph | L | n | Med ε SNAP | Med ε QAOA | R | verdict |
|-------|---|---|------------|------------|---|---------|
| reuse | 2 | 6 | 0.0119 | 0.0119 | 1.0007 | TIE |
| reuse | 2 | 9 | 0.0098 | 0.0096 | 1.0127 | TIE |
| regen | 2 | 6 | 0.0107 | 0.0108 | 0.9878 | TIE |
| regen | 2 | 9 | 0.0099 | 0.0099 | 1.0051 | TIE |

### Walsh-degree γ (optional, reuse, L=2)

Graded `γ_{ℓ,w}` for Walsh degrees 1..4 of the same `H_c` diagonal, shell still power_p. Holdout ε matches the scalar-γ joint row. Layer-rms of the four degree weights is about (1.88, 1.88, 1.80, 1.88): the optimum stays on a common weight, i.e. the GRAPE-exact scalar pulse. Walsh is not a separate win.

| n | Med ε Walsh | Med ε QAOA | R |
|---|-------------|------------|---|
| 6 | 0.0600 | 0.0598 | 1.0031 |
| 9 | 0.0586 | 0.0584 | 1.0037 |

## Does regenerating T hurt transfer?

Regenerating T does **not** hurt n=6→n=9 transfer relative to reusing Max-E supports. Every transfer deployment is UNDER (R from 1.13 to 1.76): angles fit at n=6 do not carry. Where the two graphs differ, regen is the smaller transfer/joint gap (L=16 full: 1.22 vs reuse 1.70; sparse L=2: 1.16 vs 1.44). L=2 full is a 2% toss-up (1.29 vs 1.26). Reuse is not a better transfer geometry at these sizes. At n=6 the uncapped supports are identical (Jaccard 1), so that row only rerolls signs; n=9 full and both sparse rows actually change T.

- L=2 full: n=9 transfer R reuse=1.269 (ε=0.0741) vs regen=1.295 (ε=0.0761). Joint R reuse=1.004, regen=1.004. Transfer/joint ε ratio reuse=1.264, regen=1.290.
- L=16 full: n=9 transfer R reuse=1.762 (ε=0.0480) vs regen=1.125 (ε=0.0382). Joint R reuse=1.035, regen=0.926. Transfer/joint ε ratio reuse=1.703, regen=1.216.
- L=2 sparse: n=9 transfer R reuse=1.447 (ε=0.0659) vs regen=1.167 (ε=0.0613). Joint R reuse=1.003, regen=1.006. Transfer/joint ε ratio reuse=1.442, regen=1.159.

## Verdict

**needs_edit.** Quantumness and the GRAPE gap both pass at λ=1 (and on the holdout split): median bitstring peak 0.41, and both the best bitstring and `| + ⟩` sit ~0.23/n above E0. A diagonal pulse on `| + ⟩` cannot close that gap.

The train{6,9} power_p gate does not BEAT Fourier QAOA on both sizes at both depths.

- L=2 joint is a converged tie on reuse (R=1.003 at n=6, R=1.004 at n=9). The shell does not leave the QAOA dual. The same tie shows up for regen, for the sparse cap, for Walsh, and at the alternate λ=1.5.
- L=16 joint reuse beats n=6 (R=0.665) and misses n=9 (R=1.035, just past the 1.02 tie band). The L=16 train mean was still falling at the round cap, so that 3% is not a fully polished minimum. It is enough to refuse PASS: L=2 is already a tie, and the dual-n rule fails.
- Per-n shells at L=16 do beat both sizes (R=0.716 and R=0.841). The circuit has the capacity. The joint `a_n` loss is what gives n=6 its win and lets n=9 slip over the line. Regen joint at L=16 clears both n (R=0.728, 0.926), so the miss is not "clause supports are required," and it is not a dead Hamiltonian.

Do not freeze this recipe onto the ladder. The edit, if any, is the shared {6,9} objective (per-n shells, or a heavier n=9 weight), not a new gate and not Rank2. Intra-cavity XX/XXX stays closed: it needs XX correlators and fails the Z+X readout lock, and arm B did not clearly win the dual-n gate.

## Repro

```bash
python Nov_3rd/scripts/smoke_quantum_h_B.py --b-tr 16 --b-ho 32 --rounds-l2 18 --rounds-l16 14 --max-iter 6 --wall-fit 240 --threads 4
```

Wall 250.6s. Device `cpu`. Script `Nov_3rd/scripts/smoke_quantum_h_B.py`.
Train69 ladder, phase2 n21, `settings.py`, and `qaoa_angles/` were not written.
