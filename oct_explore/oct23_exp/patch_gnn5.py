with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_arg = 'ap.add_argument("--tailw", type=float, default=0.5)'
new_arg = 'ap.add_argument("--tailw", type=float, default=0.5)\nap.add_argument("--p_cap", type=float, default=0.0, help=">0: downweight gradient once p_succ > p_cap so optimizer focuses on B90 tail")'
assert old_arg in code
code = code.replace(old_arg, new_arg)

old_loss = """        nl = -jnp.log(p_s + 1e-12)
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])"""

new_loss = """        nl = -jnp.log(p_s + 1e-12)
        if a.p_cap > 0:
            cap_nl = -np.log(a.p_cap)
            nl_eff = jnp.where(nl < cap_nl, cap_nl + 0.08 * (nl - cap_nl), nl)
        else:
            nl_eff = nl
        out = jnp.mean(nl_eff)
        if a.tailw > 0:
            k = max(1, int(round(0.15 * nl.shape[0])))
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])"""

assert old_loss in code
code = code.replace(old_loss, new_loss)

old_fv = """        fv[b, :, 0] = 1.0
        fv[b, :, 1] = Ip[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 2] = In[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 3] = h_bp[b, order] * (1.0 - 2.0 * flip)
        deg[b] = np.maximum(Ip[b].sum(0) + In[b].sum(0), 1.0)"""

new_fv = """        sgn = 1.0 - 2.0 * flip
        fv[b, :, 0] = 1.0
        fv[b, :, 1] = Ip[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 2] = In[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 3] = BP_DATA[f"{split_name}_h_bp"][b, order] * sgn
        fv[b, :, 4] = BP_DATA[f"{split_name}_h_dec"][b, order] * sgn
        fv[b, :, 5] = BP_DATA[f"{split_name}_h_bp4"][1, b, order] * sgn
        fv[b, :, 6] = BP_DATA[f"{split_name}_h_bp4"][2, b, order] * sgn
        deg[b] = np.maximum(Ip[b].sum(0) + In[b].sum(0), 1.0)"""

assert old_fv in code
code = code.replace(old_fv, new_fv)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Successfully patched gnn_5block.py!")
