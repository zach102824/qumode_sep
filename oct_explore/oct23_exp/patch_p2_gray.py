with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

old_arg = 'ap.add_argument("--mode", default="shared", choices=["shared", "gnn"])'
new_arg = 'ap.add_argument("--mode", default="shared", choices=["shared", "gnn"])\nap.add_argument("--mixer", default="sd", choices=["sd", "gray"])'
assert old_arg in code
code = code.replace(old_arg, new_arg, 1)

old_st1 = """        bg_opt = best_id.x.reshape(L, 2)
        p_id_va = np.asarray(sim_id(jnp.asarray(best_id.x, jnp.float32), Cn12_va, S12_va, 12))
        log(f"  Stage 1 {kind}: loss={best_id.fun:.4f} | ideal val12 med/B90 = {np.median(1/p_id_va):.2f}/{np.quantile(1/p_id_va, 0.9):.2f}")"""

new_st1 = """        bg_opt = best_id.x.reshape(L, 2)
        p_id_va = np.asarray(sim_id(jnp.asarray(best_id.x, jnp.float32), Cn12_va, S12_va, 12))
        inv_id_va = 1.0 / np.maximum(p_id_va, 1e-12)
        med_id_va, b90_id_va = float(np.median(inv_id_va)), float(np.quantile(inv_id_va, 0.9))
        log(f"  Stage 1 {kind}: loss={best_id.fun:.4f} | ideal val12 med/B90 = {med_id_va:.2f}/{b90_id_va:.2f}")
        gray_out = os.path.join(OUT_DIR, f"p2_gray_L{L}.json")
        if kind == "gray" and (a.mixer == "gray" or not os.path.exists(gray_out)):
            bg_j = jnp.asarray(best_id.x, jnp.float32)
            bg_z = jnp.asarray(np.stack([bg_opt[:, 0], np.zeros(L)], 1).reshape(-1), jnp.float32)
            te_g = {}
            for n_te in [8, 12, 16, 20]:
                E_te, _, _, _ = get_instances(n_te, 100, TEB, "test")
                m_te = int(round(9.93 * n_te))
                ch = 50 if n_te <= 12 else (10 if n_te == 16 else 2)
                ev_g = jax.jit(lambda bg_v, c_v, s_v, nv=n_te: sim_id(bg_v, c_v, s_v, nv))
                ps_g, pz_g = [], []
                for i in range(0, 100, ch):
                    Cn_b = jnp.asarray(E_te[i:i+ch].astype(np.float32) * (n_te / float(m_te)))
                    S_b = jnp.asarray((E_te[i:i+ch] == 0).astype(np.float32))
                    ps_g.append(np.asarray(ev_g(bg_j, Cn_b, S_b)))
                    pz_g.append(np.asarray(ev_g(bg_z, Cn_b, S_b)))
                p_all_g = np.concatenate(ps_g); pz_all_g = np.concatenate(pz_g)
                lk_g = np.full_like(p_all_g, 1.0 - float(jnp.sum(jnp.abs(V0_LOG)**2)) ** (n_te // 4))
                te_g[str(n_te)] = eval_metrics(p_all_g, lk_g, pz_all_g, (E_te[:, 0] == 0).astype(float), n_te)
                te_g[str(n_te)]["p"] = p_all_g.tolist()
            sc_g = fit_scaling(te_g)
            g_doc = {
                "args": {**vars(a), "tag": f"p2_gray_L{L}", "mixer": "gray"},
                "val12": {"median_inv_p": med_id_va, "B90_inv_p": b90_id_va, "S": 0.5*(np.log(med_id_va)+np.log(b90_id_va))},
                "test": te_g, "scaling": sc_g, "bg_opt": best_id.x.tolist(), "sec": time.time() - t0
            }
            json.dump(g_doc, open(gray_out, "w"))
            log(f"  Saved {gray_out}: n=8..20 med = {[round(te_g[str(n)]['median_inv_p'], 2) for n in [8,12,16,20]]}, b_med={sc_g['b_median']:.4f}^n, b_B90={sc_g['b_B90']:.4f}^n")
            if a.mixer == "gray":
                json.dump(g_doc, open(OUT, "w"))
                return"""

assert old_st1 in code
code = code.replace(old_st1, new_st1, 1)

# Also support --mixer gray in run_gnn_pipeline()
old_gnn_init = """    assert a.init_shared and os.path.exists(a.init_shared), f"init_shared {a.init_shared} required for GNN"
    Js = json.load(open(a.init_shared))
    xs = jnp.asarray(Js["x_best"], jnp.float32)
    A0, TH0, SOUT0, g0 = unpack_shared(xs)
    p0["A0"] = A0; p0["TH0"] = TH0; p0["SOUT0"] = SOUT0; p0["g0"] = g0"""

new_gnn_init = """    assert a.init_shared and os.path.exists(a.init_shared), f"init_shared {a.init_shared} required for GNN"
    Js = json.load(open(a.init_shared))
    if a.mixer == "gray":
        bg_init = np.array(Js["bg_opt"], np.float32).reshape(L, 2)
        A0 = jnp.zeros((L, PAIRS, 2), jnp.float32).at[:, 0, 0].set(bg_init[:, 0])
        TH0 = jnp.zeros((L, PAIRS, NU), jnp.float32)
        SOUT0 = jnp.zeros((L, NU), jnp.float32)
        g0 = jnp.asarray(bg_init[:, 1], jnp.float32)
    else:
        xs = jnp.asarray(Js["x_best"], jnp.float32)
        A0, TH0, SOUT0, g0 = unpack_shared(xs)
    p0["A0"] = A0; p0["TH0"] = TH0; p0["SOUT0"] = SOUT0; p0["g0"] = g0"""

assert old_gnn_init in code
code = code.replace(old_gnn_init, new_gnn_init, 1)

old_gnn_loop = """            for c in range(C_cav):
                M_log_b = layer_log_batch(A_b[:, c, l], TH_b[:, c, l], SOUT_b[:, c, l])
                psi = apply_cav_batch(M_log_b, psi, c, C_cav)"""

new_gnn_loop = """            for c in range(C_cav):
                if a.mixer == "gray":
                    b_cl = A_b[:, c, l, 0, 0]  # (B,)
                    th_cl = TH_b[:, c, l, 0]   # (B, 16) computational SNAP on 0..15
                    sout_cl = SOUT_b[:, c, l]  # (B, 16)
                    e_w = jnp.exp(-0.5j * b_cl[:, None] * W_GRAY_J[None, :])
                    U_f = jnp.einsum("ik,bk,jk->bij", V_GRAY_J, e_w, V_GRAY_J.conj())
                    Vp = jnp.exp(-1j * sout_cl)[:, :, None] * U_f * jnp.exp(-1j * th_cl)[:, None, :]
                    M_log_b = Vp[:, FOCK_OF_LOG, :][:, :, FOCK_OF_LOG]
                else:
                    M_log_b = layer_log_batch(A_b[:, c, l], TH_b[:, c, l], SOUT_b[:, c, l])
                psi = apply_cav_batch(M_log_b, psi, c, C_cav)"""

assert old_gnn_loop in code
code = code.replace(old_gnn_loop, new_gnn_loop, 1)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Patched --mixer gray and automatic p2_gray_L{L}.json saving into part2_multicav.py!")
