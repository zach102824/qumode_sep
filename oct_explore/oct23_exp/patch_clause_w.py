with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "r") as f:
    code = f.read()

# Let's check what we need to add:
# 1. In prepare_split: when need_graph is True and a.clause_w is True, compute V_grid (B, m, D_reg, D_reg) float32 or uint8
# Wait: for n=6, (10000, 60, 8, 8) uint8 is 38 MB! For n=8, (10000, 79, 16, 16) uint8 is 202 MB!
# And in each chunk of B_ch=1000 or 2000, V_grid_f = V_grid.astype(jnp.float32) * nm_scale.
# Then cost_phase_rel(ll) = jnp.einsum("bm,bmij->bij", g_b[:, ll, None] + w_b[:, :, ll], V_grid_f),
# and we subtract max_{i,j} cost_phase(ll) so spare levels get 0 phase and valid levels get (cost_phase - max)!
