# N3COND ledger audit (before any further fit)

Historical C0/C1 files are unchanged. This audit counts them. It does not relabel them.

## What Q=qL contained

Every C0 and C1 fit spent exactly Q=qL energy queries. Across 384 fits that is **425984**. The categories inside those fits are:

| category | queries | inside Q=qL |
|---|---|---|
| per-start calibration | 12288 | yes |
| perturbations | 410624 | yes |
| candidates | 3072 | yes |
| monitoring | 0 | — |
| total | 425984 | yes |

12288 / 384 = 32 calibration queries per fit = 8 per start × 4 starts. C0 used those 8 to set the scalar gain. C1 executed the same 8, counted them, and then discarded the gain. Every C1 fit has `gain_frozen: true` and `a_used` equal to one of 8 values frozen earlier (one per N, L, circuit).

## What Q=qL did not contain

| block | queries | inside Q=qL | used by C1 |
|---|---|---|---|
| axial scale, N9, dev seeds 10010–10013 | 832 | no | yes, froze λ and a0, including QAOA |
| axial scale, N12, same dev seeds | 832 | no | yes, same |
| simultaneous probes, N9 only | 512 | no | no. λ is identically 1. File kept. |
| outside-cap total | 2176 | no | |
| grand total incurred | 428160 | | |

832 = 192 (N L1 Nov3) + 64 (N L1 QAOA) + 384 (N L3 Nov3) + 192 (N L3 QAOA). Same split at N12. The simultaneous file is 4 cells × 128.

## Reconciliation

C1’s per-cell Q=qL was the **fit cap only**. It was inclusive of the per-start calibration queries, the perturbations, and the two candidates. It was **not** inclusive of the scale/gain training that defined C1.

That training is the axial block, 832 queries at each of N=9 and N=12, outside the fit cap. QAOA C1 received its own axial freeze inside those 832. C0 received no outside-cap queries. The control was matched on the fit cap and on the shape of the extra block, and the extra block was still extra.

The 512 simultaneous probes are unused. They count on the global research ledger. They are not moved into Q after the fact.

Do not describe the 425984 as a cap that included C1’s scale training. The negative N9 conclusion stands: on the fit cap, with extra outside-cap training that C0 did not get, C1 still failed the preregistered holdout energy-and-fidelity rule.

One uncharged diagnostic state rebuild per fit is not in the 428160. Ground-state preparation is classical and not in it either.

## B3 addendum (separate, after this audit)

B3 spent 147456 energy queries. All of them are inside its own Q=qL, including 6656 axial calibration queries. Outside-cap queries for B3: 0. These 147456 are not part of the 425984 and do not revise it. Stream total incurred energy queries: 428160 + 147456 = 575616.
