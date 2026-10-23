with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "r") as f:
    code = f.read()

old = '            g_doc = {\n                "args": {**vars(a), "tag": f"p2_gray_L{L}", "mixer": "gray"},'
new = '            g_tag = f"p2_gray_L{L}" if a.ntrain8 == 2000 else f"p2_gray_L{L}_ntr{a.ntrain8}"\n            g_doc = {\n                "args": {**vars(a), "tag": g_tag, "mixer": "gray"},'
assert old in code
code = code.replace(old, new, 1)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py", "w") as f:
    f.write(code)
print("Patched g_tag in part2_multicav.py!")
