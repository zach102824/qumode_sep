with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "r") as f:
    code = f.read()

old_block = """    if mode == "rr4":
        # 4 deterministic compile-time modes: r=0 bppol, r=1 biaspol, r=2 bppol with cavity swap/rank shift, r=3 damped-BP temp 2
        if r_idx == 0:
            return build_perm_and_flip(pos, neg, h_bp, "bppol", 0)
        elif r_idx == 1:
            return build_perm_and_flip(pos, neg, h_bp, "biaspol", 0)
        elif r_idx == 2:
            score = np.abs(h_bp + 0.3 * (pos - neg))
            var_by_conf = np.argsort(score, kind="stable")
            slots_alt = [K_total * c + j for j in range(K_total) for c in [1, 0]]
            order = np.empty(n, dtype=np.int64)
            for rank, v_orig in enumerate(var_by_conf):
                order[slots_alt[rank]] = v_orig
            # Flip lowest-confidence variable in addition to BP pref
            pref = (h_bp >= 0).astype(np.int64)
            pref[var_by_conf[0]] ^= 1
            return order, pref[order]
        else:
            score = np.abs(0.5 * h_bp + (pos - neg))
            var_by_conf = np.argsort(-score, kind="stable")  # reversed slot assignment
            order = np.empty(n, dtype=np.int64)
            for rank, v_orig in enumerate(var_by_conf):
                order[slots_by_speed[rank]] = v_orig
            pref = ((0.5 * h_bp + (pos - neg)) >= 0).astype(np.int64)
            return order, pref[order]"""

new_block = """    if mode == "rr4":
        # h_bp passed in is already h_bp4[r_idx, b] from the r_idx-th decimated BP branch!
        return build_perm_and_flip(pos, neg, h_bp, "bppol", 0)"""

assert old_block in code, "old_block not found"
code = code.replace(old_block, new_block)

old_prep = '    h_bp = BP_DATA[f"{split_name}_h_bp"]'
new_prep = '    h_bp = BP_DATA[f"{split_name}_h_bp4"][r_idx] if f"{split_name}_h_bp4" in BP_DATA else BP_DATA[f"{split_name}_h_bp"]'
assert old_prep in code, "old_prep not found"
code = code.replace(old_prep, new_prep)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "w") as f:
    f.write(code)
print("Patched part1_2cav.py successfully")
