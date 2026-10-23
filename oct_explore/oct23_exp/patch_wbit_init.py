with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_wbit = """    p["Wbit"] = W(H + 7, FM, 1.5)
    p["bbit"] = jnp.zeros(FM)"""

new_wbit = """    wbit_np = np.asarray(W(H + 7, FM, 1.0), np.float32)
    for m_i in range(4):
        wbit_np[:, m_i] *= 0.1
        wbit_np[H + 3 + m_i, m_i] = 6.0
        wbit_np[:, 4 + m_i] *= 0.2
        wbit_np[H + 3 + m_i, 4 + m_i] = 2.0
    p["Wbit"] = jnp.asarray(wbit_np)
    p["bbit"] = jnp.zeros(FM)"""

assert old_wbit in code
code = code.replace(old_wbit, new_wbit)
with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Patched Wbit branch-indicator initialization into gnn_5block.py!")
