with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "r") as f:
    code = f.read()

old_init = """        if K_PREP == 0:
            pr1 = PREPS[f"sd_K{K_cav}_k1"]
            A[:, 0, 0] = pr1["alphas"][0][0] * rng.uniform(0.85, 1.15, size=(nc,))
            A[:, 0, 1] = pr1["alphas"][0][1] * rng.uniform(0.85, 1.15, size=(nc,))
            TH = rng.uniform(-0.3, 0.3, (nc, NB_ACT, NU))
            TH[:, 0, :] = np.array(pr1["thetas"][0])[None, :] + rng.normal(0, 0.15, (nc, NU))"""

new_init = """        if K_PREP == 0:
            pr1 = PREPS[f"sd_K{K_cav}_k1"]
            if a.relabel in ("bppol", "bpdec", "biaspol", "rr4"):
                # Solutions are concentrated in low Fock levels f=0..4; initialize Block 0 with moderate displacement
                r0 = rng.uniform(0.5, 1.3, size=(nc,))
                p0 = rng.uniform(0, 2 * np.pi, size=(nc,))
                A[:, 0, 0] = r0 * np.cos(p0)
                A[:, 0, 1] = r0 * np.sin(p0)
                TH = rng.normal(0, 0.3, (nc, NB_ACT, NU))
            else:
                A[:, 0, 0] = pr1["alphas"][0][0] * rng.uniform(0.85, 1.15, size=(nc,))
                A[:, 0, 1] = pr1["alphas"][0][1] * rng.uniform(0.85, 1.15, size=(nc,))
                TH = rng.uniform(-0.3, 0.3, (nc, NB_ACT, NU))
                TH[:, 0, :] = np.array(pr1["thetas"][0])[None, :] + rng.normal(0, 0.15, (nc, NU))"""

assert old_init in code
code = code.replace(old_init, new_init)
with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "w") as f:
    f.write(code)
print("Patched 0p5x1 initialization in part1_2cav.py")
