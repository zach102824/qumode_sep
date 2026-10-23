with open("/workspace/qumode_sep/oct_explore/oct23_exp/run_wave3_queue.py", "r") as f:
    c = f.read()

old_jobs = """    ("p2_gnn_L12_k2_mp64_nocw",     f"--tag p2_gnn_L12_k2_mp64_nocw --mode gnn --L 12 --pairs 2 --kprep 2 --arch mp64 --steps 800 --init_shared {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_gnn_L15_k3_mp64_nocw",     f"--tag p2_gnn_L15_k3_mp64_nocw --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --steps 800 --init_shared {RESDIR}/p2_shared_L15_k3.json"),
]"""

new_jobs = """    ("p2_gnn_L12_k2_mp64_cw",       f"--tag p2_gnn_L12_k2_mp64_cw   --mode gnn --L 12 --pairs 2 --kprep 2 --arch mp64 --clause_w --steps 800 --init_shared {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_gnn_L12_k2_mp64_nocw",     f"--tag p2_gnn_L12_k2_mp64_nocw --mode gnn --L 12 --pairs 2 --kprep 2 --arch mp64 --steps 800 --init_shared {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_gnn_L15_k3_mp64_cw",       f"--tag p2_gnn_L15_k3_mp64_cw   --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --clause_w --steps 800 --init_shared {RESDIR}/p2_shared_L15_k3.json"),
    ("p2_gnn_L15_k3_mp64_nocw",     f"--tag p2_gnn_L15_k3_mp64_nocw --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --steps 800 --init_shared {RESDIR}/p2_shared_L15_k3.json"),
]"""

assert old_jobs in c
c = c.replace(old_jobs, new_jobs, 1)

with open("/workspace/qumode_sep/oct_explore/oct23_exp/run_wave3_queue.py", "w") as f:
    f.write(c)
print("Updated run_wave3_queue.py!")
