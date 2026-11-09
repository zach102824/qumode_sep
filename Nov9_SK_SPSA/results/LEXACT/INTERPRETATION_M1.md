# LEXACT M1 — candidate kernel, not a GR adoption

Conditioning C1/B3/B4 stays closed. This batch is exact-simulator throughput for original Nov3 vs QAOA p=L.

The candidate is in-place numba FWHT, in-place X mixers, and an energy pass that does not fancy-index-copy the state. Types stay float64/complex128. Packing, shared γ/θ, full shells, and the C0 SPSA recipe are unchanged. QAOA p=L uses the same kernel policy.

## What the numbers are

A speedup is present on every timed cell. It is modest at the sizes that matter: **2.86× at N12 L6 Nov3, 2.27× at N15 L1 Nov3, 3.60× at N18 L1 Nov3**. QAOA is in the same band (1.8–3.5× for N12–N18). Compile/cache is 0.11 s and is not in the steady times. N18 B* field prep is 49 s of classical Lanczos, separate from Q. Peak RSS 262 MB on one process.

469 reference checks on N6/9/12 L1–6 passed before the traces. SPSA traces of 128 inclusive queries match to ≤1e-13 in energy and pick the same start/candidate. Fidelity is post hoc.

N15 seed 10032 needed k=8 Lanczos to label E1. k=2 had missed the odd excited state. B* itself (entropy max, even positive GS, residual 5e-14) was already valid. After the repair, E1 is odd and gap_even > 0, same as the N3COND checks. N18 seed 10032 was already odd-E1.

## What this is not

It is not a GR recipe change and not a claim that the 1000-seed panel is done. Dividing the reported ~2200 h by 3.6 would be a misuse of a fixed-field probe that is not a B* completion estimate.

C1/B3/B4 remain unadopted. No further gain rescue. Final 20000–20999 and validation 11000–11031 unread. 10018–25 and 10026–31 were not used.

## Bottleneck for the next report

Energy evaluation at N18 is now ~23 ms (Nov3 L1) on one core. A matched Q=qL L=6 fit is still minutes, and a 1000-instance N18 grid is still a many-hour job even at 3–4×. The remaining in-scope lever is **more exact-kernel efficiency at N18** (SIMD/in-place numpy FWHT fused with the shared phase, or more than one worker on independent instances), not another SPSA scale. This kernel is a versioned candidate only.
