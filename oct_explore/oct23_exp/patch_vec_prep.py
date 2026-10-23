with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_loop = """    for b in range(B):
        order, flip = build_perm_and_flip(pos[b], neg[b], h_bp[b], a.relabel)
        x_old = np.zeros_like(GRID_TO_X)
        for s in range(n):
            bit_s = ((GRID_TO_X >> (n - 1 - s)) & 1) ^ int(flip[s])
            x_old |= (bit_s << (n - 1 - int(order[s])))
        E_grid = Es[b][x_old].astype(np.float32)
        c_grid = E_grid * nm_scale
        Cn_rel[b] = c_grid - np.max(c_grid)
        Succ[b] = (E_grid == 0).astype(np.float32)

        inv_order = np.empty(n, dtype=np.int64)
        inv_order[order] = np.arange(n)
        for c_i in range(m_clauses):
            v0, v1, v2, v3 = [int(inv_order[int(cls_v[b, c_i, k_i])]) for k_i in range(4)]
            s0, s1, s2, s3 = [int(cls_s[b, c_i, k_i]) * (1 if flip[int(inv_order[int(cls_v[b, c_i, k_i])])] == 0 else -1) for k_i in range(4)]
            for s_new, sign_new in ((v0, s0), (v1, s1), (v2, s2), (v3, s3)):
                if sign_new > 0: Ip[b, c_i, s_new] = 1.0
                else: In[b, c_i, s_new] = 1.0
            if a.clause_w:
                # Clause is violated iff all 4 literals are false
                viol_c = (
                    (slot_bits[v0] == (0 if s0 > 0 else 1)) &
                    (slot_bits[v1] == (0 if s1 > 0 else 1)) &
                    (slot_bits[v2] == (0 if s2 > 0 else 1)) &
                    (slot_bits[v3] == (0 if s3 > 0 else 1))
                )
                Viol[b, c_i] = viol_c
        sgn = 1.0 - 2.0 * flip
        fv[b, :, 0] = 1.0
        fv[b, :, 1] = Ip[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 2] = In[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 3] = BP_DATA[f"{split_name}_h_bp"][b, order] * sgn
        fv[b, :, 4] = BP_DATA[f"{split_name}_h_dec"][b, order] * sgn
        fv[b, :, 5] = BP_DATA[f"{split_name}_h_bp4"][1, b, order] * sgn
        fv[b, :, 6] = BP_DATA[f"{split_name}_h_bp4"][2, b, order] * sgn
        deg[b] = np.maximum(Ip[b].sum(0) + In[b].sum(0), 1.0)"""

new_loop = """    orders = np.zeros((B, n), dtype=np.int64)
    flips = np.zeros((B, n), dtype=np.int64)
    inv_orders = np.zeros((B, n), dtype=np.int64)
    for b in range(B):
        order, flip = build_perm_and_flip(pos[b], neg[b], h_bp[b], a.relabel)
        orders[b] = order; flips[b] = flip
        inv_orders[b, order] = np.arange(n)
        x_old = np.zeros_like(GRID_TO_X)
        for s in range(n):
            bit_s = ((GRID_TO_X >> (n - 1 - s)) & 1) ^ int(flip[s])
            x_old |= (bit_s << (n - 1 - int(order[s])))
        E_grid = Es[b][x_old].astype(np.float32)
        c_grid = E_grid * nm_scale
        Cn_rel[b] = c_grid - np.max(c_grid)
        Succ[b] = (E_grid == 0).astype(np.float32)

    b_idx = np.arange(B)[:, None, None]
    c_idx = np.arange(m_clauses)[None, :, None]
    v_new = inv_orders[b_idx, cls_v]  # (B, m, 4)
    s_new = cls_s * (1 - 2 * flips[b_idx, v_new])  # (B, m, 4)
    np.add.at(Ip, (b_idx, c_idx, v_new), (s_new > 0).astype(np.float32))
    np.add.at(In, (b_idx, c_idx, v_new), (s_new < 0).astype(np.float32))
    if a.clause_w:
        need_val = (s_new < 0).astype(np.uint8)[..., None]  # (B, m, 4, 1): literal is false when bit == (s_new < 0)
        Viol[:] = (
            (slot_bits[v_new[:, :, 0]] == need_val[:, :, 0]) &
            (slot_bits[v_new[:, :, 1]] == need_val[:, :, 1]) &
            (slot_bits[v_new[:, :, 2]] == need_val[:, :, 2]) &
            (slot_bits[v_new[:, :, 3]] == need_val[:, :, 3])
        )
    b_v = np.arange(B)[:, None]
    sgn = (1.0 - 2.0 * flips).astype(np.float32)
    fv[:, :, 0] = 1.0
    fv[:, :, 1] = Ip.sum(1) / (2.0 * m_clauses / n)
    fv[:, :, 2] = In.sum(1) / (2.0 * m_clauses / n)
    fv[:, :, 3] = BP_DATA[f"{split_name}_h_bp"][:B][b_v, orders] * sgn
    fv[:, :, 4] = BP_DATA[f"{split_name}_h_dec"][:B][b_v, orders] * sgn
    fv[:, :, 5] = BP_DATA[f"{split_name}_h_bp4"][1, :B][b_v, orders] * sgn
    fv[:, :, 6] = BP_DATA[f"{split_name}_h_bp4"][2, :B][b_v, orders] * sgn
    deg[:] = np.maximum(Ip.sum(1) + In.sum(1), 1.0)"""

assert old_loop in code, "old_loop not found"
code = code.replace(old_loop, new_loop)
with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Vectorized prepare_split in gnn_5block.py!")
