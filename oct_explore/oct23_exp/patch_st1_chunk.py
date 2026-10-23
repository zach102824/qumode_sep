with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

old_st1_sim = """        def sim_id(bg, Cn, S, n):
            C_cav = n // 4; B = Cn.shape[0]
            psi = jnp.broadcast_to(make_psi0_log(C_cav), (B,) + (NU,) * C_cav)
            Cn_sh = Cn.reshape(psi.shape)
            for l in range(L):
                b, g = bg[2*l], bg[2*l+1]
                psi = psi * jnp.exp(-1j * g * Cn_sh)
                U_f = (Vf * jnp.exp(-0.5j * b * w_j)[None, :]) @ Vf.conj().T
                U_l = U_f[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]
                for c in range(C_cav):
                    psi = apply_cav_shared(U_l, psi, c, C_cav)
            pr = jnp.abs(psi.reshape(B, -1)) ** 2
            return jnp.sum(pr * S, axis=1)
        vg_id = jax.jit(jax.value_and_grad(
            lambda bg: 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn8_tr, S8_tr, 8) + 1e-12))
                     + 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn12_tr, S12_tr, 12) + 1e-12))
        ))"""

new_st1_sim = """        def sim_id(bg, Cn, S, n):
            C_cav = n // 4; B = Cn.shape[0]
            psi = jnp.broadcast_to(make_psi0_log(C_cav), (B,) + (NU,) * C_cav)
            Cn_sh = Cn.reshape(psi.shape)
            bg_mat = bg.reshape(L, 2)
            def id_body(psi_c, bg_l):
                b, g = bg_l[0], bg_l[1]
                psi_c = psi_c * jnp.exp(-1j * g * Cn_sh)
                U_f = (Vf * jnp.exp(-0.5j * b * w_j)[None, :]) @ Vf.conj().T
                U_l = U_f[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]
                for c in range(C_cav):
                    psi_c = apply_cav_shared(U_l, psi_c, c, C_cav)
                return psi_c, None
            psi, _ = jax.lax.scan(id_body, psi, bg_mat)
            pr = jnp.abs(psi.reshape(B, -1)) ** 2
            return jnp.sum(pr * S, axis=1)
        vg_id8  = jax.jit(jax.value_and_grad(lambda bg, c, s: jnp.mean(-jnp.log(sim_id(bg, c, s, 8) + 1e-12))))
        vg_id12 = jax.jit(jax.value_and_grad(lambda bg, c, s: jnp.mean(-jnp.log(sim_id(bg, c, s, 12) + 1e-12))))
        def vg_id(bg):
            tot_v = 0.0; tot_g = np.zeros(2 * L, dtype=np.float64)
            for i in range(0, a.ntrain8, 1000):
                w = (min(i + 1000, a.ntrain8) - i) / float(a.ntrain8)
                v, g = vg_id8(bg, Cn8_tr[i:i+1000], S8_tr[i:i+1000])
                tot_v += 0.5 * float(v) * w; tot_g += 0.5 * np.asarray(g, np.float64) * w
            for i in range(0, a.ntrain12, 500):
                w = (min(i + 500, a.ntrain12) - i) / float(a.ntrain12)
                v, g = vg_id12(bg, Cn12_tr[i:i+500], S12_tr[i:i+500])
                tot_v += 0.5 * float(v) * w; tot_g += 0.5 * np.asarray(g, np.float64) * w
            return tot_v, tot_g"""

assert old_st1_sim in code
code = code.replace(old_st1_sim, new_st1_sim, 1)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Patched chunked lax.scan Stage 1 into part2_multicav.py!")
