with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_init = """    p["Wo"] = jnp.zeros((FM, NB_ACT))
    p["Wfock"] = jnp.zeros((K * H, NB_ACT * NU))"""

new_init = """    p["Wo"] = jnp.zeros((FM, NB_ACT))
    p["Mcav1"] = W(K * (H + 7), H, 1.0)
    p["bcav1"] = jnp.zeros(H)
    p["Wfock"] = jnp.zeros((H, NB_ACT * NU))
    p["Wbit"] = W(H + 7, FM, 1.5)
    p["bbit"] = jnp.zeros(FM)
    p["Wprod"] = jnp.zeros((FM, NB_ACT))"""

assert old_init in code, "old_init not found"
code = code.replace(old_init, new_init)

old_fwd = """    hd = hv[:, CV]  # (B, C, K, H)
    dA_cav = (hd.mean(2) @ p["Wa"]).reshape(B, C, NB_ACT, 2)
    A_b = p["A0"][None] + dA_glob + dA_cav

    U0 = p["u0"] + hd @ p["P0"]
    U1 = p["u1"] + hd @ p["P1"]
    z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT)) / K
    ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)
    ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT)
    ph_fock = (hd.reshape(B, C, K * a.H) @ p["Wfock"]).reshape(B, C, NB_ACT, NU)
    TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock"""

new_fwd = """    hd = hv[:, CV]  # (B, C, K, H)
    hd_aug = jnp.concatenate([hd, G["fv"][:, CV]], axis=-1)  # (B, C, K, H + 7)
    hcav_h = jnp.tanh(hd_aug.reshape(B, C, K * (a.H + 7)) @ p["Mcav1"] + p["bcav1"])  # (B, C, H)
    dA_cav = ((hd.mean(2) + hcav_h) @ p["Wa"]).reshape(B, C, NB_ACT, 2)
    A_b = p["A0"][None] + dA_glob + dA_cav

    U0 = p["u0"] + hd @ p["P0"]
    U1 = p["u1"] + hd @ p["P1"]
    z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT)) / K
    ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)
    ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT)
    ph_fock = (hcav_h @ p["Wfock"]).reshape(B, C, NB_ACT, NU)

    # Multilinear K-bit Fock-Product SNAP head: exact addressability of any Fock level f in 0..2^K-1
    P_bit = jax.nn.sigmoid(hd_aug @ p["Wbit"] + p["bbit"])  # (B, C, K, FM)
    M_jf = P_bit[:, :, :, None, :] * BT[None, None, :, :, None] + (1.0 - P_bit[:, :, :, None, :]) * (1.0 - BT[None, None, :, :, None])
    Phi = jnp.prod(M_jf, axis=2)  # (B, C, NU, FM)
    ph_prod = jnp.transpose(Phi @ p["Wprod"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)

    TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock + ph_prod"""

assert old_fwd in code, "old_fwd not found"
code = code.replace(old_fwd, new_fwd)

old_loss = """        if a.p_cap > 0:
            cap_nl = -np.log(a.p_cap)
            nl_eff = jnp.where(nl < cap_nl, cap_nl + 0.08 * (nl - cap_nl), nl)
        else:
            nl_eff = nl
        out = jnp.mean(nl_eff)
        if a.tailw > 0:
            k = max(1, int(round(0.15 * nl.shape[0])))
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])"""

new_loss = """        if a.p_cap > 0:
            cap_nl = -np.log(a.p_cap)
            nl_eff = cap_nl + jax.nn.softplus(4.0 * (nl - cap_nl)) / 4.0
        else:
            nl_eff = nl
        out = jnp.mean(nl_eff)
        if a.tailw > 0:
            k = max(1, int(round(0.16 * nl.shape[0])))
            top_nl = jax.lax.top_k(nl, k)[0]
            out = out + a.tailw * jnp.mean(top_nl) + 0.12 * a.tailw * jnp.mean(jnp.exp(jnp.minimum(top_nl, 3.5)))"""

assert old_loss in code, "old_loss not found"
code = code.replace(old_loss, new_loss)

old_wd = 'keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo", "Wfock") + (("Wc",) if a.clause_w else ())'
new_wd = 'keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo", "Wfock", "Wprod") + (("Wc",) if a.clause_w else ())'
assert old_wd in code, "old_wd not found"
code = code.replace(old_wd, new_wd)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Successfully patched multilinear K-bit Fock-Product SNAP head + Cavity MLP + tail-focused softplus loss into gnn_5block.py!")
