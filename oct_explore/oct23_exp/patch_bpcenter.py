for fname in ["part1_2cav.py", "gnn_5block.py"]:
    p = f"/workspace/qumode_sep/oct_explore/oct23_exp/{fname}"
    with open(p, "r") as f:
        code = f.read()
    old_blk = '    if mode in ("bppol", "bpdec"):'
    new_blk = '''    if mode == "bpcenter":
        K_c = n // 2
        fc = (1 << (K_c - 1)) - 1  # f_c = 3 for n=6 (K=3), f_c = 7 for n=8 (K=4)
        g_center = gray(fc)
        center_bits = [(g_center >> j) & 1 for j in range(K_c)]
        df_per_j = [abs([i for i in range(1 << K_c) if gray(i) == (g_center ^ (1 << j))][0] - fc) for j in range(K_c)]
        j_by_speed = np.argsort(df_per_j, kind="stable")
        slots_c = [K_c * c + int(j) for j in j_by_speed for c in range(C)]
        var_by_conf = np.argsort(np.abs(h_bp), kind="stable")
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_c[rank]] = v_orig
        pref = (h_bp >= 0).astype(np.int64)[order]
        target_bits = np.array(center_bits * C, dtype=np.int64)
        return order, pref ^ target_bits
    if mode in ("bppol", "bpdec"):'''
    assert old_blk in code, f"old_blk not in {fname}"
    code = code.replace(old_blk, new_blk, 1)
    code = code.replace('choices=["none", "bias", "biaspol", "bppol", "bpdec"', 'choices=["none", "bias", "biaspol", "bppol", "bpdec", "bpcenter"')
    code = code.replace('rmode_use == "bpdec"', 'rmode_use in ("bpdec", "bpcenter")')
    code = code.replace('a.relabel == "bpdec"', 'a.relabel in ("bpdec", "bpcenter")')
    with open(p, "w") as f:
        f.write(code)
    print("Patched bpcenter into", fname)
