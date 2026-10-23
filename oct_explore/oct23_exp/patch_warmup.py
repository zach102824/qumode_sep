with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_step = """            v_b, g_b = vg(x_cur, G_sub, D_tr["Cn_rel"][idx], D_tr["Succ"][idx], D_tr["Viol"][idx])
            lr_t = a.lr * (0.08 + 0.92 * 0.5 * (1.0 + np.cos(np.pi * st / a.steps)))
            x_cur, mom, vel = adam_step(x_cur, mom, vel, g_b, st, lr_t)"""

new_step = """            v_b, g_b = vg(x_cur, G_sub, D_tr["Cn_rel"][idx], D_tr["Succ"][idx], D_tr["Viol"][idx])
            gnorm = jnp.sqrt(jnp.sum(g_b * g_b) + 1e-12)
            g_b = g_b * jnp.minimum(1.0, 3.0 / gnorm)
            warm = min(1.0, st / 200.0)
            lr_t = a.lr * warm * (0.08 + 0.92 * 0.5 * (1.0 + np.cos(np.pi * st / a.steps)))
            x_cur, mom, vel = adam_step(x_cur, mom, vel, g_b, st, lr_t)"""

assert old_step in code
code = code.replace(old_step, new_step)
with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Patched linear warmup + grad norm clipping into gnn_5block.py!")
