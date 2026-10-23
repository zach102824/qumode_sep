with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "r") as f:
    code = f.read()

old_arg = 'ap.add_argument("--tailw", type=float, default=0.25, help="CVaR top-10% tail weight in loss")'
new_arg = 'ap.add_argument("--tailw", type=float, default=0.25, help="CVaR top-10% tail weight in loss")\nap.add_argument("--obj", default="logp", choices=["logp", "invp", "s_direct"])'
assert old_arg in code
code = code.replace(old_arg, new_arg)

old_loss = """def loss_shared(x, Cn_rel, Succ):
    base = unpack_shared(x)
    p_succ, p_leak = SIM_MAIN(base, Cn_rel, Succ)
    nl = -jnp.log(p_succ + 1e-12)
    out = jnp.mean(nl)
    if a.tailw > 0:
        k = max(1, nl.shape[0] // 10)
        out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
    if a.leakpen > 0:
        out = out + a.leakpen * jnp.mean(p_leak)
    return out"""

new_loss = """def loss_shared(x, Cn_rel, Succ):
    base = unpack_shared(x)
    p_succ, p_leak = SIM_MAIN(base, Cn_rel, Succ)
    if a.obj == "invp":
        inv = 1.0 / jnp.maximum(p_succ, 1e-4)
        k = max(1, inv.shape[0] // 10)
        out = jnp.mean(inv) + a.tailw * jnp.mean(jax.lax.top_k(inv, k)[0])
    elif a.obj == "s_direct":
        nl = -jnp.log(p_succ + 1e-12)
        s_nl = jnp.sort(nl)
        B = nl.shape[0]
        # Smooth quantile band around 50% (40%..60%) and 90% (85%..95%)
        med_band = jnp.mean(s_nl[int(0.40 * B):max(int(0.40 * B) + 1, int(0.60 * B))])
        b90_band = jnp.mean(s_nl[int(0.85 * B):max(int(0.85 * B) + 1, int(0.95 * B))])
        out = 0.5 * (med_band + b90_band) + 0.2 * jnp.mean(nl)
    else:
        nl = -jnp.log(p_succ + 1e-12)
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
    if a.leakpen > 0:
        out = out + a.leakpen * jnp.mean(p_leak)
    return out"""

assert old_loss in code
code = code.replace(old_loss, new_loss)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "w") as f:
    f.write(code)
print("Added --obj invp and s_direct")
