# N9-GR encoding-route pilot

This folder is the GR worker only. It does not edit the shared consultation protocol.

Zach approved D2–D5 at 18:41 Asia/Shanghai on 2026-11-09 as simulation assumptions:

- SNAP phase is 0 on Fock levels above 7.
- A `UcZ` term that touches a leaked cavity is off. Other terms still act. The global U0 rule remains a sensitivity arm.
- Energy adds `|Jij|` on each leaked coupling and `+b` on each leaked qubit.
- Readout in this simulation is ideal and logical.

The compiled `W_L` and its leakage are not modelled. Nothing here is a hardware-feasibility claim, a shot-noise claim, a freeze, or a validation/final result.

The circuit is fixed: `|+\rangle → [UcZ → true D(α_ℓ) → bare shell SNAP]^L`, one shared `UcZ`, real `α`, SPSA only, QAOA at `p = L`.

## Run

```bash
python3 conformance.py
python3 cutoff.py
python3 run_pilot.py
python3 summarize.py
```

The pilot is dev seeds 10000–10007 at N = 6 and 9 only. Seeds 11000–11031 and 20000–20999 are refused. Those b = 1 and b = 1.5 runs are the reference. Zach’s 19:05 note keeps the next comparison on the per-instance maximum-entropy field only.

`bstar_harness.py` accepts a manifest of B* values and builds Nov3rd, true-displacement Nov9th, and QAOA p = L job records. `python3 bstar_harness.py launch` is refused. No new field sweep is started from here. See `HARNESS.md`.
