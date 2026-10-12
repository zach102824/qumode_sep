# Noisy simulation

This folder holds noisy simulations of the qumode_sep ground-state finder. The circuit to simulate is in [`input_circuit/`](input_circuit/); to change the target, replace that folder.

The simulator evolves the density matrix of the real device (data transmons d and e, coupler q, cavities A and B). ECD blocks are ideal unitaries followed by exact Lindblad-derived Kraus channels. Each JP gate runs as its physical pulse/wait sequence, with the coupler included in the master equation. [`NOISE.md`](NOISE.md) gives the noise strengths, where they come from (Dutta *et al.* Appendix A, the Eickbusch device), the Kraus forms, the accuracy against a full `mesolve`, and the error budget.

## Files

| file | contents |
|---|---|
| `device.py` | `Device` hardware parameters (T1,cav, T1,q, T2, χ, τ_ECD); preset `EICKBUSCH` |
| `kraus.py` | photon-loss and transmon T1/T2 Kraus channels, helpers that act on one mode of ρ |
| `circuit.py` | noisy layer: ideal ECD block, idle noise, JP gate (`fast` precomputed channel or `direct` with q in ρ) |
| `simulator.py` | `NoisyXorKbitSim`, a drop-in for `noiseless.relayout_kbit.XorKbitSim` |
| `run_noisy_relayout.py` | explore + relabel SPSA run of the input circuit on the noisy device |
| `tests/test_noise.py` | channel, machinery and end-to-end checks (QuTiP `mesolve` as ground truth) |

## Run

```
python -m pytest -q noisy_simulation/tests                     # ~30 s
python noisy_simulation/run_noisy_relayout.py --trials 5      # input circuit, device noise
python noisy_simulation/run_noisy_relayout.py --trials 5 --noise-scale 0.5
python noisy_simulation/run_noisy_relayout.py --max-h 1 --trials 1 --s 2 --r 4 --nf 12   # smoke
```

Defaults come from `input_circuit/circuit.json`: n = 8, weighted MAX-4-SAT, Gray code, nf = 24, L = 4, E = 1 explore with s = 20, R = 2 relabel rounds of r = 150, lr 0.05, eta ×16 / ×2. Seeds, checkpointing (one JSON line per trial, finished trials skipped on restart) and the summary are the same as in `noiseless/run_relayout_kbit.py`. Results go to `noisy_simulation/results/` (gitignored).

`--noise-scale 0` (or `--preset noiseless`) reproduces the noiseless optimizer trial for trial on the same seeds. The tests check this.

## Cost

At nf = 24 one cost evaluation takes about 2.7 s on one core (ρ is 2304 × 2304, 85 MB). A full trial makes about 770 evaluations, so it takes about 35 min. The JP channel is built once (about 12 s) and shared by all workers through a memory-mapped file. With 8 GB RAM, use `--workers 4` (the default). The 50-trial campaign (10 instances × 5 starts) then takes about 7 to 8 hours. `--nf 12` is a quick screening mode at about 0.15 s per evaluation.
