# 1. Fix gnn_5block.py
with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

# Add --pure_gnn flag
old_arg = 'ap.add_argument("--clause_w", action="store_true")'
new_arg = 'ap.add_argument("--clause_w", action="store_true")\nap.add_argument("--pure_gnn", action="store_true", help="Use only raw factor-graph degrees in fv[:,:,0:3] (fv[:,:,3:7]=0, no BP/decimation features)")'
assert old_arg in code
code = code.replace(old_arg, new_arg, 1)

# Fix line 99 tuple bug
assert 'if mode == "bpcenter", "mpw4":' in code
code = code.replace('if mode == "bpcenter", "mpw4":', 'if mode == "bpcenter":')

# Fix h_bp selection for mpw4
old_h = 'h_bp = BP_DATA[f"{split_name}_h_mpw4"] if a.relabel == "mpw4" else (BP_DATA[f"{split_name}_h_dec"] if a.relabel in ("bpdec", "bpcenter", "mpw4") else BP_DATA[f"{split_name}_h_bp"])'
new_h = 'h_bp = BP_DATA[f"{split_name}_h_mpw4"] if (a.relabel == "mpw4" and f"{split_name}_h_mpw4" in BP_DATA) else (BP_DATA[f"{split_name}_h_dec"] if a.relabel in ("bpdec", "bpcenter", "mpw4") else BP_DATA[f"{split_name}_h_bp"])'
if old_h in code:
    code = code.replace(old_h, new_h)

# Cleanly separate Pure GNN (relabel in none/bias/biaspol or --pure_gnn) from BP-assisted GNN in fv[:,:,3:7]
old_fv = """    fv[:, :, 0] = 1.0
    fv[:, :, 1] = Ip.sum(1) / (2.0 * m_clauses / n)
    fv[:, :, 2] = In.sum(1) / (2.0 * m_clauses / n)
    fv[:, :, 3] = BP_DATA[f"{split_name}_h_bp"][:B][b_v, orders] * sgn
    fv[:, :, 4] = BP_DATA[f"{split_name}_h_dec"][:B][b_v, orders] * sgn
    fv[:, :, 5] = BP_DATA[f"{split_name}_h_bp4"][1, :B][b_v, orders] * sgn
    fv[:, :, 6] = BP_DATA[f"{split_name}_h_bp4"][2, :B][b_v, orders] * sgn"""

new_fv = """    fv[:, :, 0] = 1.0
    fv[:, :, 1] = Ip.sum(1) / (2.0 * m_clauses / n)
    fv[:, :, 2] = In.sum(1) / (2.0 * m_clauses / n)
    if not a.pure_gnn and a.relabel not in ("none", "bias", "biaspol"):
        fv[:, :, 3] = BP_DATA[f"{split_name}_h_bp"][:B][b_v, orders] * sgn
        fv[:, :, 4] = BP_DATA[f"{split_name}_h_dec"][:B][b_v, orders] * sgn
        fv[:, :, 5] = BP_DATA[f"{split_name}_h_bp4"][1, :B][b_v, orders] * sgn
        fv[:, :, 6] = BP_DATA[f"{split_name}_h_bp4"][2, :B][b_v, orders] * sgn"""

assert old_fv in code, "old_fv not found in gnn_5block.py"
code = code.replace(old_fv, new_fv)

# Also only use Wbit BP-channel init when BP features are present
old_wbit = """    wbit_np = np.array(W(H + 7, FM, 1.0), np.float32, copy=True)
    for m_i in range(4):
        wbit_np[:, m_i] *= 0.1
        wbit_np[H + 3 + m_i, m_i] = 6.0
        wbit_np[:, 4 + m_i] *= 0.2
        wbit_np[H + 3 + m_i, 4 + m_i] = 2.0"""

new_wbit = """    wbit_np = np.array(W(H + 7, FM, 1.0), np.float32, copy=True)
    if not a.pure_gnn and a.relabel not in ("none", "bias", "biaspol"):
        for m_i in range(4):
            wbit_np[:, m_i] *= 0.1
            wbit_np[H + 3 + m_i, m_i] = 6.0
            wbit_np[:, 4 + m_i] *= 0.2
            wbit_np[H + 3 + m_i, 4 + m_i] = 2.0"""

assert old_wbit in code, "old_wbit not found in gnn_5block.py"
code = code.replace(old_wbit, new_wbit)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Fixed gnn_5block.py successfully!")

# 2. Fix part1_2cav.py
with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "r") as f:
    pcode = f.read()

assert 'if mode == "bpcenter", "mpw4":' in pcode
pcode = pcode.replace('if mode == "bpcenter", "mpw4":', 'if mode == "bpcenter":')

old_pfv = 'fv[b, :, 3] = h_bp[b, order] * (1.0 - 2.0 * flip)'
new_pfv = 'fv[b, :, 3] = 0.0 if rmode_use in ("none", "bias", "biaspol") else (h_bp[b, order] * (1.0 - 2.0 * flip))'
assert old_pfv in pcode
pcode = pcode.replace(old_pfv, new_pfv)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "w") as f:
    f.write(pcode)
print("Fixed part1_2cav.py successfully!")
