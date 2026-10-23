with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "r") as f:
    code = f.read()

old_init = """        if K_PREP == 0:
            pr1 = PREPS[f"sd_K{K}_k1"]
            A0[:, 0, :] = np.array(pr1["alphas"][0])[None, :]
            TH0[:, 0, :] = np.array(pr1["thetas"][0])[None, :]"""

new_init = """        if K_PREP == 0:
            if a.relabel in ("bppol", "bpdec", "biaspol"):
                A0[:, 0, 0] = 0.85; A0[:, 0, 1] = 0.0
            else:
                pr1 = PREPS[f"sd_K{K}_k1"]
                A0[:, 0, :] = np.array(pr1["alphas"][0])[None, :]
                TH0[:, 0, :] = np.array(pr1["thetas"][0])[None, :]"""

assert old_init in code
code = code.replace(old_init, new_init)
with open("/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py", "w") as f:
    f.write(code)
print("Patched gnn_5block.py 0p5x1 init")
