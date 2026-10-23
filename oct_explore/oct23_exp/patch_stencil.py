with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_init = '    p["Wprod"] = jnp.zeros((FM, NB_ACT))'
new_init = '    p["Wprod"] = jnp.zeros((9, FM, NB_ACT))\n    p["Wphi_a"] = jnp.zeros((NU * FM, NB_ACT * 2))'
assert old_init in code
code = code.replace(old_init, new_init)

old_fwd = """    # Multilinear K-bit Fock-Product SNAP head: exact addressability of any Fock level f in 0..2^K-1
    P_bit = jax.nn.sigmoid(hd_aug @ p["Wbit"] + p["bbit"])  # (B, C, K, FM)
    M_jf = P_bit[:, :, :, None, :] * BT[None, None, :, :, None] + (1.0 - P_bit[:, :, :, None, :]) * (1.0 - BT[None, None, :, :, None])
    Phi = jnp.prod(M_jf, axis=2)  # (B, C, NU, FM)
    ph_prod = jnp.transpose(Phi @ p["Wprod"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)

    TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock + ph_prod"""

new_fwd = """    # Multilinear K-bit Fock-Product SNAP head + 9-tap Fock-space constructive-interference stencil
    P_bit = jax.nn.sigmoid(hd_aug @ p["Wbit"] + p["bbit"])  # (B, C, K, FM)
    M_jf = P_bit[:, :, :, None, :] * BT[None, None, :, :, None] + (1.0 - P_bit[:, :, :, None, :]) * (1.0 - BT[None, None, :, :, None])
    Phi = jnp.prod(M_jf, axis=2)  # (B, C, NU, FM)
    Phi_pad = jnp.pad(Phi, ((0, 0), (0, 0), (4, 4), (0, 0)))
    Phi_st = jnp.stack([Phi_pad[:, :, d:d+NU, :] for d in range(9)], axis=2)  # (B, C, 9, NU, FM)
    ph_prod = jnp.einsum("bcdfm,dml->bclf", Phi_st, p["Wprod"])  # (B, C, NB_ACT, NU)
    dA_phi = (Phi.reshape(B, C, NU * 32) @ p["Wphi_a"]).reshape(B, C, NB_ACT, 2)
    A_b = A_b + dA_phi

    TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock + ph_prod"""

assert old_fwd in code
code = code.replace(old_fwd, new_fwd)

old_wd = 'keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo", "Wfock", "Wprod") + (("Wc",) if a.clause_w else ())'
new_wd = 'keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo", "Wfock", "Wprod", "Wphi_a") + (("Wc",) if a.clause_w else ())'
assert old_wd in code
code = code.replace(old_wd, new_wd)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Successfully patched 9-tap Fock stencil + Wphi_a into gnn_5block.py!")
