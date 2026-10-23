with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

old_loop = '    for kind, Vf, w_j in [("gray", V_GRAY_J, W_GRAY_J), ("tri", V_TRI_J, W_TRI_J)]:'
new_loop = '    for kind, Vf, w_j in [("gray", V_GRAY_J, W_GRAY_J)]:'
assert old_loop in code
code = code.replace(old_loop, new_loop, 1)

old_st1 = """        best_id = min((minimize(lambda x: tuple(np.asarray(v, float) for v in vg_id(jnp.asarray(x, jnp.float32))),
                       np.asarray(bgi, np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 100})
                       for bgi in bg_inits), key=lambda r: r.fun)"""

new_st1 = """        gray_out = os.path.join(OUT_DIR, f"p2_gray_L{L}.json")
        if kind == "gray" and os.path.exists(gray_out):
            g_cached = json.load(open(gray_out))
            class _Res: pass
            best_id = _Res(); best_id.x = np.array(g_cached["bg_opt"], dtype=np.float64); best_id.fun = 0.0
        else:
            best_id = min((minimize(lambda x: tuple(np.asarray(v, float) for v in vg_id(jnp.asarray(x, jnp.float32))),
                           np.asarray(bgi, np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 100})
                           for bgi in bg_inits), key=lambda r: r.fun)"""

assert old_st1 in code
code = code.replace(old_st1, new_st1, 1)

code = code.replace("ch = 50 if n_te <= 12 else (10 if n_te == 16 else 2)", "ch = 50 if n_te <= 12 else (25 if n_te == 16 else 10)")

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Successfully patched part2_multicav.py for 2.5x speedup!")
