with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

# 1. Replace Python loop in sim_gnn_n with jax.lax.scan for 10x faster JIT compilation at L=12, 15
old_gnn_loop = """        for l in range(L):
            if not zero_oracle:
                if a.clause_w:
                    w_eff = dg[:, l][:, None] + w_b[:, :, l]  # (B, m)
                    Ew = cost_Ew(w_eff)
                    psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape))
                else:
                    psi = psi * jnp.exp(-1j * dg[:, l].reshape((B,) + (1,) * C_cav) * Cn_sh)
            for c in range(C_cav):
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

new_gnn_loop = """        # Use jax.lax.scan over L layers so XLA compiles 1 layer body instead of unrolling L*C_cav layers
        A_scan = jnp.transpose(A_b, (2, 0, 1, 3, 4))     # (L, B, C, PAIRS, 2)
        TH_scan = jnp.transpose(TH_b, (2, 0, 1, 3, 4))   # (L, B, C, PAIRS, NU)
        SOUT_scan = jnp.transpose(SOUT_b, (2, 0, 1, 3))  # (L, B, C, NU)
        dg_scan = jnp.transpose(dg, (1, 0))              # (L, B)
        wb_scan = jnp.transpose(w_b, (2, 0, 1))          # (L, B, m_or_1)

        def layer_body(psi_c, xs_l):
            A_l, TH_l, SOUT_l, dg_l, wb_l = xs_l
            if not zero_oracle:
                if a.clause_w:
                    w_eff = dg_l[:, None] + wb_l
                    Ew = cost_Ew(w_eff)
                    psi_c = psi_c * jnp.exp(-1j * Ew.reshape(psi_c.shape))
                else:
                    psi_c = psi_c * jnp.exp(-1j * dg_l.reshape((B,) + (1,) * C_cav) * Cn_sh)
            for c in range(C_cav):
                if a.mixer == "gray":
                    b_cl = A_l[:, c, 0, 0]
                    th_cl = TH_l[:, c, 0]
                    sout_cl = SOUT_l[:, c]
                    e_w = jnp.exp(-0.5j * b_cl[:, None] * W_GRAY_J[None, :])
                    U_f = jnp.einsum("ik,bk,jk->bij", V_GRAY_J, e_w, V_GRAY_J.conj())
                    Vp = jnp.exp(-1j * sout_cl)[:, :, None] * U_f * jnp.exp(-1j * th_cl)[:, None, :]
                    M_log_b = Vp[:, FOCK_OF_LOG, :][:, :, FOCK_OF_LOG]
                else:
                    M_log_b = layer_log_batch(A_l[:, c], TH_l[:, c], SOUT_l[:, c])
                psi_c = apply_cav_batch(M_log_b, psi_c, c, C_cav)
            return psi_c, None

        psi, _ = jax.lax.scan(layer_body, psi, (A_scan, TH_scan, SOUT_scan, dg_scan, wb_scan))"""

assert old_gnn_loop in code, "old_gnn_loop not found"
code = code.replace(old_gnn_loop, new_gnn_loop, 1)

# 2. Also use jax.lax.scan in sim_shared_n for fast shared N_train scaling
old_sh_loop = """    for l in range(L):
        psi = psi * jnp.exp(-1j * g[l] * Cn_sh)
        M_log = layer_log_shared(A[l], TH[l], SOUT[l])
        for c in range(C_cav):
            psi = apply_cav_shared(M_log, psi, c, C_cav)"""

new_sh_loop = """    def sh_body(psi_c, xs_l):
        A_l, TH_l, SOUT_l, g_l = xs_l
        psi_c = psi_c * jnp.exp(-1j * g_l * Cn_sh)
        M_log = layer_log_shared(A_l, TH_l, SOUT_l)
        for c in range(C_cav):
            psi_c = apply_cav_shared(M_log, psi_c, c, C_cav)
        return psi_c, None
    psi, _ = jax.lax.scan(sh_body, psi, (A, TH, SOUT, g))"""

assert old_sh_loop in code, "old_sh_loop not found"
code = code.replace(old_sh_loop, new_sh_loop, 1)

# 3. In Stage 1 gray, only use p2_gray_L{L}.json cache if ntrain8 == 2000; if ntrain8 != 2000, optimize on full a.ntrain8 and save p2_gray_L{L}_ntr{a.ntrain8}.json
old_st1_loss = """        vg_id = jax.jit(jax.value_and_grad(
            lambda bg: 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn8_tr[:1000], S8_tr[:1000], 8) + 1e-12))
                     + 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn12_tr[:500], S12_tr[:500], 12) + 1e-12))
        ))"""

new_st1_loss = """        vg_id = jax.jit(jax.value_and_grad(
            lambda bg: 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn8_tr, S8_tr, 8) + 1e-12))
                     + 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn12_tr, S12_tr, 12) + 1e-12))
        ))"""
assert old_st1_loss in code
code = code.replace(old_st1_loss, new_st1_loss, 1)

old_gray_cache = """        gray_out = os.path.join(OUT_DIR, f"p2_gray_L{L}.json")
        if kind == "gray" and os.path.exists(gray_out):"""

new_gray_cache = """        gray_out = os.path.join(OUT_DIR, f"p2_gray_L{L}.json" if a.ntrain8 == 2000 else f"p2_gray_L{L}_ntr{a.ntrain8}.json")
        if kind == "gray" and os.path.exists(gray_out) and not a.grow_from:"""

assert old_gray_cache in code
code = code.replace(old_gray_cache, new_gray_cache, 1)

# 4. Support --grow_from in run_shared_pipeline() so N_train scaling warm-starts from the converged x_best checkpoint!
old_cands = """    # Stage 1: Fit 2L-parameter (beta_l, gamma_l) schedule for both "gray" and "tri" target mixers
    log(f"Stage 1: Optimizing 2L={2*L} (beta, gamma) schedules for Gray and Tri mixers...")
    cands_x = []"""

new_cands = """    # Stage 1: Fit 2L-parameter (beta_l, gamma_l) schedule for both "gray" and "tri" target mixers
    cands_x = []
    if a.grow_from and os.path.exists(a.grow_from) and a.mixer != "gray":
        Jg = json.load(open(a.grow_from))
        cands_x.append((Jg.get("winner_init", "warm"), np.array(Jg["x_best"], dtype=np.float32)))
        log(f"Warm-starting Stage 3 directly from {a.grow_from}")
    else:
        log(f"Stage 1: Optimizing 2L={2*L} (beta, gamma) schedules for Gray and Tri mixers...")"""

assert old_cands in code
code = code.replace(old_cands, new_cands, 1)

old_for_kind = '    for kind, Vf, w_j in [("gray", V_GRAY_J, W_GRAY_J), ("tri", V_TRI_J, W_TRI_J)]:'
new_for_kind = '    for kind, Vf, w_j in ([] if cands_x else ([("gray", V_GRAY_J, W_GRAY_J)] if a.mixer == "gray" else [("gray", V_GRAY_J, W_GRAY_J), ("tri", V_TRI_J, W_TRI_J)])):'
assert old_for_kind in code
code = code.replace(old_for_kind, new_for_kind, 1)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Successfully patched jax.lax.scan and --grow_from warm-start into part2_multicav.py!")
