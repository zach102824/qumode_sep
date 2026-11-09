# Coupled-gap reproduction

The runnable script is `reproduce_author_gap.py`. It executes the author
antiferromagnetic cells and the author spin-glass parity loop. The spin-glass
`J` arrays are the ones written in the notebooks; in the gap notebook they are
commented, and in the MTE notebook the 0.001 matrix is active.

Environment: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1.
Code pin: tag `2.0`, commit `f4c329fa69dcf98b7be32374c145584209e21f0e`.

## Table 1 gaps

The quantity reproduced is `Δ`, the minimum coupled gap in Table 1. The
columns `1 - E/E_gs` are modulated-time-evolution energy errors. Those
optimizations were not rerun.

| Case | What was run | B | Ordinary `E1-E0` | Coupled gap | Reference |
|---|---|---|---|---|---|
| AFM, 8 sites, `J=1/\|i-j\|` | author even-even notebook, step `1e-4`, `B∈[0.1,2]` | 0.3803 | 0.0285492566894483 | 0.45778532405044103 | Table 1: `Δ=0.458` at `Bc=0.38` |
| AFM at the printed `Bc` | same matrices | 0.38 | 0.028397306232534447 | 0.45778596870079546 | same table cell |
| Spin glass `t_f≈493` | author full-space spin-parity loop, 99000 fields | 0.2153 | 0.003442964009986227 | 0.008846979172492553 | Table 1: `Δ=0.009` at `Bc=0.215`. Source comment: `E_c=0.008849508983285403` |
| Spin glass `t_f≈11980` | same 99000-field loop | 0.1 | 3.7901023137720813e-06 | 0.0009558548042960524 | Table 1: `Δ=0.001` at `Bc=0.1` |

The antiferromagnetic reduced basis has dimension 72, as Appendix B says.
Its gap agrees with a separate full-space diagonalization restricted to the
ground state's spin and spatial parities, to about `1e-15`. The ordinary gap
at the same field is 0.02855. Those are not the same number. At `B=0.1` the
ordinary gap is `8.5e-7` and the coupled gap is 0.723.

The `t_f≈493` loop matches the author file
`data/fig_5/energy_spec_tf_493.csv` on all 99000 rows, maximum absolute
residual `2.13e-13`. The notebook comment `0.008849508983285403` is
`2.5e-6` above that minimum. The table rounds the minimum to 0.009 and the
field to 0.215.

The `t_f≈11980` loop matches `energy_spec_tf_11980.csv` on all 99000 rows,
maximum absolute residual `1.99e-13`. The minimum is the first grid point
`B=0.1`, so the quoted `Δ` depends on the path's lower endpoint. The ordinary
gap there is `3.79e-6`, about 250 times smaller than the coupled gap.

The supplemental file `data/supplemental material/energy_spec.csv` does not
match the antiferromagnetic notebook. On the overlapping fields the maximum
absolute residual is 0.180. Its own minimum is 0.331 at `B=0.424`. That file
is not the Table 1 number, and this run does not call it reproduced.

## Command

```
git clone https://github.com/zach102824/Modulated-time-evolution-MTE-and-QAOA.git
git -C Modulated-time-evolution-MTE-and-QAOA checkout f4c329fa69dcf98b7be32374c145584209e21f0e
python3 Nov_8th/coupled_gap_r1/reproduce_author_gap.py
```

The script expects the clone at `/tmp/mte` or pass `--repo`. The 99000-field
spin-glass loops took 1645 s for the `t_f≈493` instance on this machine.
