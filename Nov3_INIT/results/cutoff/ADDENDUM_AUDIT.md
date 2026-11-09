# Addendum: PR106 identity pad is algebraic, not a pulse-compile cutoff test

Numerical results in `summary.json` and the cell files are unchanged. CUTOFF_ROBUST_L1 still stands as **ideal-\(W_L\) pad robustness**. Ledger contribution of this probe is already in the historical 2793472.

Ideal \(W_L=H^{\otimes 3}\oplus I\) never couples Fock \(0..7\) to the pad, so \(p_{\rm leak}\) is numerical zero and L=1 \(\Delta E\) is invariant when NF goes 8→10. That is an algebraic identity of the embedding, **not** cutoff convergence of a physical SNAP–D compilation of \(W_{\rm BinH}\).

No frozen native SNAP–D compile of the full 8×8 BinH \(W\) on \(\mathrm{NF}\ge 10\) exists in-repo (D5). Hardware / compiled-cutoff convergence remains **unvalidated**. QAOA in this probe was a qubit circuit, not a Fock embedding.

No GR. See `Nov3_INIT/AUDIT_WL.md`.
