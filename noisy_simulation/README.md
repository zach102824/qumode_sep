# Noisy simulation

This folder is for noisy simulations of the qumode_sep ground-state finder. The target circuit is in `circuit.json`. Noise models and run scripts will be added here later.

## Target circuit (noiseless reference)

The best low-cost noiseless setting at n = 8 on the weighted MAX-4-SAT set (`Hamiltonians/wmaxsat/`). It was run on 10 instances with 5 starts each (50 trials).

| stage | guess correct | success | mean p(GS) | cumulative evals |
|---|---|---|---|---|
| explore | 0.80 | 0.80 | 0.150 | 164 |
| relabel 1 | 0.94 | 0.90 | 0.849 | 465 |
| relabel 2 | 0.98 | 0.96 | 0.957 | 766 |

Each trial uses about 11 distinct classical energy lookups.

## Hardware

- 2 data transmons (d, e) and 2 cavities (A, B).
- Bits x1 and x2 go on the transmons. Bits x3 to x5 go on cavity A and x6 to x8 on cavity B. The default layout is shown here; explore runs may permute it.
- Each cavity stores its 3 bits as a Gray-coded Fock level, so neighbouring Fock levels differ by one bit.
- Each cavity is simulated with 24 Fock levels (code levels + 16).
- The bus gate is the joint-parity (jp) gate. It is ideal in the noiseless simulation.

## Algorithm

1. **Explore (E = 1).** Grow an ECD circuit from L = 1 to 4 with 20 SPSA-Adam steps per layer (lr 0.05) on a cold Gibbs cost (eta x16). Read the most likely bitstring, then do a Hamming-1 fix-up: check the n + 1 states at distance 0 or 1 and keep the lowest-energy one as the guess.
2. **Relabel (R = 2 rounds).** XOR the cavity bits so the guess sits at Fock vacuum. Run a fresh L = 4 circuit for 150 steps from a small-beta start (Gibbs eta x2), then do a Hamming-1 fix-up. Keep the best round by cost.

Success means the most likely bitstring of the final state is the ground state. Guess correct means the classical guess after the fix-up is the ground state.

## Reproduce (noiseless)

```
python noiseless/run_relayout_kbit.py --n 8 --ham-set wmaxsat --code gray --L 4 --xL 4 --s 20 --explore 1 --explore-mask perm --topk 1 --xeta 16 --reta 2 --lr 0.05 --R 2 --r 150 --trials 5
```
