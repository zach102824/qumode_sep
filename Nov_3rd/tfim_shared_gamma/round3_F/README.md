# Round 3 family F

Tied polynomial of `Hc` on the frozen manifest:

```text
G = Hc + theta_K (FK + Q4) / rK
  = Hc + theta_K / (2 rK) (Hc^2 - c0 I)
```

`rK = RMS(FK) / RMS(J)`. One shared gamma and one shared theta_K. Shell controls stay per layer. The 495 four-body supports are part of the cost ledger.

`freeQ4` trains `theta_Q` separately and is not `f(Hc)`.

```bash
cd Nov_3rd/tfim_shared_gamma/round3_F
OMP_NUM_THREADS=1 python3 test_algebra.py
OMP_NUM_THREADS=1 python3 run_round3.py
```
