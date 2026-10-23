with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

# 1. Free training arrays and caches before test evaluation in run_gnn_pipeline()
old_save_th = """    th_best = jnp.asarray(best_val["th"], jnp.float32)
    np.savez_compressed(OUT.replace(".json", "_theta.npz"), theta=np.asarray(th_best, np.float32))

    test_res = {}"""

new_save_th = """    th_best = jnp.asarray(best_val["th"], jnp.float32)
    np.savez_compressed(OUT.replace(".json", "_theta.npz"), theta=np.asarray(th_best, np.float32))
    del G8_tr, G12_tr, G12_va, Cn8_tr, S8_tr, Cn12_tr, S12_tr, Cn12_va, S12_va, E8_tr, E12_tr, E12_va
    import gc; gc.collect()

    test_res = {}"""

assert old_save_th in code
code = code.replace(old_save_th, new_save_th, 1)

# 2. If _theta.npz already exists and OUT does not exist, skip training and jump straight to test evaluation!
old_for_st = """    m0, b0, s0 = check_val12(th0, 0)
    log(f"Start Part2 Pure GNN L={L} pairs={PAIRS} arch={a.arch} H={H} cw={a.clause_w} params={th0.size} | step 0 val12={m0:.2f}/{b0:.2f} S={s0:.4f}")"""

new_for_st = """    th_file = OUT.replace(".json", "_theta.npz")
    if os.path.exists(th_file):
        th_loaded = np.load(th_file)["theta"].astype(np.float32)
        m0, b0, s0 = check_val12(th_loaded, a.steps)
        log(f"Loaded saved {th_file} for {a.tag} | val12={m0:.2f}/{b0:.2f} S={s0:.4f}, skipping training!")
        a.steps = 0
    else:
        m0, b0, s0 = check_val12(th0, 0)
        log(f"Start Part2 Pure GNN L={L} pairs={PAIRS} arch={a.arch} H={H} cw={a.clause_w} params={th0.size} | step 0 val12={m0:.2f}/{b0:.2f} S={s0:.4f}")"""

assert old_for_st in code
code = code.replace(old_for_st, new_for_st, 1)

# 3. In cost_Ew, compute clause violations in chunks of 40 clauses so (2^20, m, 4) never allocates 890 MB at once!
old_cost_ew = """            def cost_Ew(w_eff):
                # w_eff: (B, m) -> Ew: (B, 2^n)
                # viol_b[x,c] = prod_k (xb[x, VP[b,c,k]] == BB[b,c,k])
                def one_b(vp_b, bb_b, w_b):
                    # vp_b, bb_b: (m, K); w_b: (m,)
                    lit = xb[:, vp_b]                      # (2^n, m, K)
                    viol = jnp.prod((lit == bb_b[None]).astype(jnp.float32), axis=-1)  # (2^n, m)
                    return viol @ w_b                       # (2^n,)
                return nm * jax.vmap(one_b)(VP, BB, w_eff)  # (B, 2^n)"""

new_cost_ew = """            def cost_Ew(w_eff):
                def one_b(vp_b, bb_b, w_b_vec):
                    m_tot = vp_b.shape[0]
                    acc = jnp.zeros(xb.shape[0], jnp.float32)
                    for c0 in range(0, m_tot, 40):
                        c1 = min(c0 + 40, m_tot)
                        lit = xb[:, vp_b[c0:c1]]
                        viol = jnp.prod((lit == bb_b[None, c0:c1]).astype(jnp.float32), axis=-1)
                        acc = acc + viol @ w_b_vec[c0:c1]
                    return acc
                return nm * jax.vmap(one_b)(VP, BB, w_eff)"""

assert old_cost_ew in code
code = code.replace(old_cost_ew, new_cost_ew, 1)

# 4. Also use ch=1 at n=20 in GNN test evaluation for minimal peak memory
code = code.replace("ch = 50 if n_te <= 12 else (10 if n_te == 16 else 2)", "ch = 50 if n_te <= 12 else (10 if n_te == 16 else 1)")

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Patched memory-safe test evaluation and theta.npz resume into part2_multicav.py!")
