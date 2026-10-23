with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "r") as f:
    code = f.read()

code = code.replace(
    'choices=["none", "bias", "biaspol", "bppol", "rr4"]',
    'choices=["none", "bias", "biaspol", "bppol", "bpdec", "rr4"]'
)

code = code.replace(
    '    if mode == "bppol":',
    '    if mode in ("bppol", "bpdec"):'
)

old_line = '    h_bp = BP_DATA[f"{split_name}_h_bp4"][r_idx] if f"{split_name}_h_bp4" in BP_DATA else BP_DATA[f"{split_name}_h_bp"]'
new_line = '    h_bp = BP_DATA[f"{split_name}_h_bp4"][r_idx] if rmode_use == "rr4" else (BP_DATA[f"{split_name}_h_dec"] if rmode_use == "bpdec" else BP_DATA[f"{split_name}_h_bp"])'
assert old_line in code, "old_line not found"
code = code.replace(old_line, new_line)

code = code.replace(
    'train_rmode = "bppol" if a.relabel == "rr4" else a.relabel',
    'train_rmode = "bpdec" if a.relabel == "rr4" else a.relabel'
)
code = code.replace(
    'relabel_mode=("bppol" if a.relabel == "rr4" else a.relabel)',
    'relabel_mode=("bpdec" if a.relabel == "rr4" else a.relabel)'
)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py", "w") as f:
    f.write(code)
print("Patched bpdec into part1_2cav.py")
