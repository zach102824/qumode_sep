# Nov9_SK_SPSA: consultation round 1 (no experiments run here)

Isolated area for the quantum SK Nov9th / Nov3rd / QAOA comparison, using SPSA only.

- `PROTOCOL.md`: the shared protocol for three independent workers. It covers:
  - exact circuits
  - binary Fock encoding with source citations
  - off-code extension and the anti-exploitation energy \(H^{\rm ext}\)
  - measurement accounting
  - cutoff convergence
  - the SPSA budget
  - seeds and stages
  - gates
  - conformance tests T1–T14
- `IDEAS.md`: three ideas (N9-LE leak echo, N9-GR encoding route, N9-FT features + transfer), with mechanics, initialization, SPSA schedule, ablations, and falsification. Paper findings are kept separate from hypotheses.
- `DECISIONS.md`: physical-definition choices D1–D9 that need the user, recommended defaults, the \(H^{\rm ext}\) proof, and what is safe to run now. D2–D5 were approved as simulation definitions on Nov 9 (approval record at the top).
- `experiments/LE/`, `results/LE/`: experiment A (leak echo) milestone 1, with b = 1/1.5 reference results. Interpretation: `results/LE/INTERPRETATION_LE_M1.md`.
- `NEXT_IDEAS_R2.md`: round-2 ideas, restricted to the \(B^*\) field, built from the LE, FT, GR and local results.
- `protocol_draft.json`: machine-readable draft (not frozen).

Numbers from the local 8-core stream are quoted as reported. Nothing here touches existing data, jobs, AF, or PR 72. The final panel 20000..20999 is unused.
