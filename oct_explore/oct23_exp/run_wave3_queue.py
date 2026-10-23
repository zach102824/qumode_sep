import os, time, subprocess

SCRIPT = "/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py"
REP    = "/workspace/qumode_sep/oct_explore/oct23_exp/build_part2_report.py"
LOGDIR = "/workspace/qumode_sep/oct_explore/logs/oct23_exp_part2"
RESDIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp_part2"

JOBS = [
    ("p2_gray_L4",                  f"--tag p2_gray_L4         --mode shared --mixer gray --L 4  --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"),
    ("p2_gray_L6",                  f"--tag p2_gray_L6         --mode shared --mixer gray --L 6  --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"),
    ("p2_gray_L8",                  f"--tag p2_gray_L8         --mode shared --mixer gray --L 8  --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"),
    ("p2_gray_L10",                 f"--tag p2_gray_L10        --mode shared --mixer gray --L 10 --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"),
    ("p2_gray_L15_ntr500",          f"--tag p2_gray_L15_ntr500   --mode shared --mixer gray --L 15 --pairs 1 --kprep 2 --ntrain8 500   --ntrain12 200"),
    ("p2_gray_L15_ntr5000",         f"--tag p2_gray_L15_ntr5000  --mode shared --mixer gray --L 15 --pairs 1 --kprep 2 --ntrain8 5000  --ntrain12 1000"),
    ("p2_gray_L15_ntr10000",        f"--tag p2_gray_L15_ntr10000 --mode shared --mixer gray --L 15 --pairs 1 --kprep 2 --ntrain8 10000 --ntrain12 1000"),
    ("p2_gnn_gray_L12_mp64_nocw",   f"--tag p2_gnn_gray_L12_mp64_nocw --mode gnn --mixer gray --L 12 --pairs 1 --kprep 2 --arch mp64 --steps 600 --init_shared {RESDIR}/p2_gray_L12.json"),
    ("p2_gnn_gray_L15_mp64_nocw",   f"--tag p2_gnn_gray_L15_mp64_nocw --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch mp64 --steps 600 --init_shared {RESDIR}/p2_gray_L15.json"),
    ("p2_shared_L12_k2_ntr500",     f"--tag p2_shared_L12_k2_ntr500   --mode shared --L 12 --pairs 2 --kprep 2 --ntrain8 500   --ntrain12 200  --maxiter 60 --grow_from {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_shared_L12_k2_ntr5000",    f"--tag p2_shared_L12_k2_ntr5000  --mode shared --L 12 --pairs 2 --kprep 2 --ntrain8 5000  --ntrain12 1000 --maxiter 60 --grow_from {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_shared_L12_k2_ntr10000",   f"--tag p2_shared_L12_k2_ntr10000 --mode shared --L 12 --pairs 2 --kprep 2 --ntrain8 10000 --ntrain12 1000 --maxiter 60 --grow_from {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_shared_L15_k3_ntr500",     f"--tag p2_shared_L15_k3_ntr500   --mode shared --L 15 --pairs 3 --kprep 3 --ntrain8 500   --ntrain12 200  --maxiter 60 --grow_from {RESDIR}/p2_shared_L15_k3.json"),
    ("p2_shared_L15_k3_ntr5000",    f"--tag p2_shared_L15_k3_ntr5000  --mode shared --L 15 --pairs 3 --kprep 3 --ntrain8 5000  --ntrain12 1000 --maxiter 60 --grow_from {RESDIR}/p2_shared_L15_k3.json"),
    ("p2_shared_L15_k3_ntr10000",   f"--tag p2_shared_L15_k3_ntr10000 --mode shared --L 15 --pairs 3 --kprep 3 --ntrain8 10000 --ntrain12 1000 --maxiter 60 --grow_from {RESDIR}/p2_shared_L15_k3.json"),
    ("p2_gnn_L12_k2_mp64_nocw",     f"--tag p2_gnn_L12_k2_mp64_nocw --mode gnn --L 12 --pairs 2 --kprep 2 --arch mp64 --steps 800 --init_shared {RESDIR}/p2_shared_L12_k2.json"),
    ("p2_gnn_L15_k3_mp64_nocw",     f"--tag p2_gnn_L15_k3_mp64_nocw --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --steps 800 --init_shared {RESDIR}/p2_shared_L15_k3.json"),
]

env = os.environ.copy()
env.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_NUM_INTRA_OP_THREADS": "1",
    "JAX_NUM_INTER_OP_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1",
})

def count_running():
    out = subprocess.check_output("ps aux | grep 'python3 /workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py' | grep -v grep | grep -v 'sh -c' | wc -l", shell=True).decode().strip()
    return int(out)

def is_tag_running(tag):
    out = subprocess.check_output(f"ps aux | grep 'part2_multicav.py' | grep ' {tag} ' | grep -v grep | wc -l", shell=True).decode().strip()
    return int(out) > 0

for tag, args in JOBS:
    out_json = os.path.join(RESDIR, f"{tag}.json")
    if os.path.exists(out_json) or is_tag_running(tag):
        print(f"Skipping {tag} (exists or running)", flush=True)
        continue
    while count_running() >= 6:
        time.sleep(5)
    print(f"Launching {tag} ...", flush=True)
    logf = open(os.path.join(LOGDIR, f"{tag}.log"), "w")
    subprocess.Popen(f"taskset -c 0-5 python3 {SCRIPT} {args} && python3 {REP}", shell=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
    time.sleep(3)

while count_running() > 0:
    time.sleep(10)
subprocess.run(f"python3 {REP}", shell=True, env=env)
print("ALL WAVE 3 JOBS COMPLETE!", flush=True)
