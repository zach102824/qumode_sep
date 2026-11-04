# LN-LA (Opus Idea 2)

Load-normalized graded H_c times a load-adaptive cavity SNAP, on the locked
Nov_3rd motif: Hamming-shell SNAP and U_c^X = W D W with W = H^{⊗3}.

The cost is the usual normalized unsat count. Its Walsh coefficients are
reweighted by the cavity load of the instance. Every weight is positive.
Call this **graded H_c**, not a non-H_c gadget.

## Formulas

Cavity j owns variables `3j, 3j+1, 3j+2` (var 0 is the MSB of the bitstring index).

- ρ_j = number of clause incidences on those three bits (a 4-literal clause
  adds one per variable it contains).
- r_j = ρ_j / mean_j ρ_j.
- For a Walsh mode S, cav(S) is the set of cavities that own at least one bit of S.

```
γ_S = γ_ℓ · Π_{j ∈ cav(S)} r_j^(−α / |cav(S)|)
```

The empty product (S = 0) is 1, so the constant mode is unchanged.

SNAP stays the frozen power_p shell, scaled per cavity:

```
φ_k = c_{ℓ,1} k + a_n (c_{ℓ,2} k² + c_{ℓ,3} k³)
a_n = (8 / (n − 1))^p
φ_k^(j) = r_j^β · φ_k
```

α = β = 0 multiplies every Walsh mode by 1 and every shell by 1, so the
circuit is the frozen champ (`forward_nov3rd`).

Optional tie: β = κ α, with κ checked on the same grid
(κ ∈ {−1, −0.5, 0, 0.5, 1}).

## What is frozen

γ, c1, c2, c3, and p are not trained. The smoke is a 9×9 grid of (α, β) on
[−1, 1].

| L | Checkpoint |
|---|------------|
| 16 | `runs/beat_qaoa_12h/train69_ladder_fresh250/power_p/params_L16_seed0.pt` |
| 2 | `runs/beat_qaoa_12h/train69/power_p/pfix0.15/params_L2_seed0.pt` |

The fresh250 ladder never trained L=2 (its depths are 1, 4, …, 32). The L=2
file is the best-holdout power_p checkpoint that does exist: p fixed at 0.15,
trained on the older train{6,9} packs with weights w6=0.25, w9=1.

## Smoke

```
python Nov_3rd/scripts/train69_lnla_grid.py --threads 4
```

- Train grid: fresh250 packs, first 100 instances of n=6 and n=9.
  Score = (Med_6 + 2 Med_9) / 3. No optimizer.
- CORE: new satisfiable packs, base `73_000_000_000`, B=1000,
  n ∈ {6, 9, 12, 15}. Seeds are disjoint from train and from scale1000.
  The decisive Medians are only these packs.
- QAOA: frozen fresh250 L=16 angles, re-evaluated on the fresh CORE.
  L=2 QAOA is trained with the ladder recipe (the ladder has no L=2 QAOA).
  The historical n=6-only L=2 angles are reported beside that column.

Kill if the selected cell has both |α| and |β| < 0.05, or the n=15 Med gain
versus the champ on the fresh CORE is under 0.5%. Ship only if that bar is
cleared and Med is below same-L QAOA on every CORE n.

Code: `Nov_3rd/lnla.py`. Checks: `python Nov_3rd/tests/test_lnla.py`.
Results: `Nov_3rd/runs/beat_qaoa_12h/train69/lnla/RESULTS.md`.

## Smoke result

**Kill.** On fresh CORE B=1000 the selected cells lose to the frozen champ at n=15.

| L | selected (α, β) | train-score gain | fresh n15 Med gain vs champ |
|---|---|---|---|
| 2 | (−0.25, 1.00) | +0.51% | −3.07% (75.42 → 77.74) |
| 16 | (1.00, 0.00) | +3.10% | −0.80% (2.613 → 2.634) |

The |α|,|β| < 0.05 collapse kill does not fire. The n15 gain kill does.
Tied β=κα is worse on n15 at both depths. The L=16 train gain did not transfer
off the B=100 slice. Details, the 9×9 tables, and the heatmaps are in the run folder.
