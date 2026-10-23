with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

old_loop = '    for kind, Vf, w_j in [("gray", V_GRAY_J, W_GRAY_J)]:'
new_loop = '    for kind, Vf, w_j in [("gray", V_GRAY_J, W_GRAY_J), ("tri", V_TRI_J, W_TRI_J)]:'
assert old_loop in code
code = code.replace(old_loop, new_loop, 1)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Restored tri alongside gray in part2_multicav.py!")
