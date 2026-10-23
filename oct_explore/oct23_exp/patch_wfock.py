with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_init = '    p["Wo"] = jnp.zeros((FM, NB_ACT))'
new_init = '    p["Wo"] = jnp.zeros((FM, NB_ACT))\n    p["Wfock"] = jnp.zeros((K * H, NB_ACT * NU))'
assert old_init in code
code = code.replace(old_init, new_init)

old_fwd = '    TH_b = p["TH0"][None] + ph_free + ph_lin'
new_fwd = '    ph_fock = (hd.reshape(B, C, K * a.H) @ p["Wfock"]).reshape(B, C, NB_ACT, NU)\n    TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock'
assert old_fwd in code
code = code.replace(old_fwd, new_fwd)

old_wd = 'keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo") + (("Wc",) if a.clause_w else ())'
new_wd = 'keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo", "Wfock") + (("Wc",) if a.clause_w else ())'
assert old_wd in code
code = code.replace(old_wd, new_wd)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Added direct per-Fock-level computational SNAP head Wfock to gnn_5block.py!")
