# New ≤2-param ideas (post CVaR floor ~1.235×)

_Updated 2026-11-03 00:38 Asia/Shanghai_

CVaR q∈{0.96..0.99} + hiLBFGS + many seeds all pile up at **n18 B90 ≈ 14.0–14.2**
(bottleneck **~1.235×** vs sealed 11.353). More CVaR re-seeds are unlikely to move it.

## Ideas (wired into `snap_easy_kerr_polish.py`, queued after hiLeak in watch)

1. **gauss_project warm** — start from sealed-beating 3p gauss (`gp_cvar08_from_gauss_s0`),
   expand θ_k then project to few_m M=2 (θ[:2]) or kerr (LS χ,ω). Keeps the γ that already
   dual-beats; only compresses SNAP.
2. **shared SNAP** — same 2 SNAP params across all L (γ still per-L). Stronger inductive bias;
   may help n18 B90 generalization vs per-L overfit.
3. **softplus loss** — unused in ≤2p polish so far (only cvar / log_inv).
4. **shell warm long** — non-polish init (escape the 1.235 basin).

Arms: `fewm2_gproj_*`, `kerr_gproj_*`, `fewm2_softplus_n12n15`,
`fewm2_shared_cvar097_n12n15`, `kerr_shared_gproj_cvar097`, `fewm2_shell_cvar097_n12n15`.

Still honor: finish scheduled slow / cvar975 / hiLeak first, then these.
