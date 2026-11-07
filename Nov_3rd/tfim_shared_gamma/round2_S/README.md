# Round 2 family S

Frozen manifest `FROZEN_TARGET_R2.json`, SHA-256
`14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`.
The decimal strings in that file are the couplings and fields. They are not
recomputed from a power law and they are not re-selected.

```text
G = Hc_inter + thetaIn * Hc_intra + thetaK * sK * FK_inter
```

`sK` matches the RMS of `FK_inter` to the RMS of `J` over every `i<j` slot,
zeros included. `thetaIn = 1` is inter-K A, not the original full-K A.
`gamma`, `thetaIn`, and `thetaK` are each one number shared by every layer.

```bash
python3 -m Nov_3rd.tfim_shared_gamma.round2_S.run_round2
```

N=12 depths 1..6 are the study. N=6 is a smoke test. Standard QAOA is also
run at L+1 and at depth `ceil((3L+3)/2)`. `same_G_qaoa` is a diagnostic, not
a ceiling and not standard QAOA.
