"""Audit a single JSON file in results/oct23_exp/:
Re-evaluate its saved parameters using the current code, check max|p_repro - p_saved|,
and compute presolve_00_frac, zero_oracle_med/B90, and unsolved_med/B90."""
import os, sys, json, shutil
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

RES_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"
ARCH_DIR = os.path.join(RES_DIR, "stale_archive")
os.makedirs(ARCH_DIR, exist_ok=True)

tag = sys.argv[1]
json_path = os.path.join(RES_DIR, f"{tag}.json")
if not os.path.exists(json_path):
    sys.exit(0)

J = json.load(open(json_path))
a_dict = J.get("args", {})
is_gnn5 = ("opt" in a_dict) or tag.startswith("gnn5_")

if not is_gnn5:
    argv = ["part1_2cav.py", "--tag", tag, "--n", str(a_dict["n"]), "--sched", a_dict["sched"],
            "--order", a_dict["order"], "--gate", a_dict.get("gate", "sd"), "--relabel", a_dict.get("relabel", "none")]
    if a_dict.get("asym", False):
        argv.append("--asym")
    sys.argv = argv
    sys.path.insert(0, "/workspace/qumode_sep/oct_explore/oct23_exp")
    import part1_2cav as P1
    x_best = jnp.asarray(J["x_best"], jnp.float32)
    D_te = P1.prepare_split("test")
    p_te, lk_te = P1.EVAL_MAIN(x_best, D_te["Cn_rel"], D_te["Succ"])
    p_chk, lk_chk = P1.EVAL_CHK(x_best, D_te["Cn_rel"], D_te["Succ"])
    p_saved = np.array(J["p_test"], float)
    max_diff = float(np.max(np.abs(np.asarray(p_te, float) - p_saved)))

    # Zero-oracle evaluation (g = 0)
    unp = P1.unpack_shared(x_best)
    unp_zero = dict(unp)
    unp_zero["g"] = jnp.zeros_like(unp["g"])
    p_zero, _ = P1.SIM_CHK(unp_zero, D_te["Cn_rel"], D_te["Succ"])
    p_zero = np.asarray(p_zero, float)
    Succ_np = np.asarray(D_te["Succ"], float)
else:
    npz_path = os.path.join(RES_DIR, f"{tag}_theta.npz")
    if not os.path.exists(npz_path):
        print(f"STALE (missing _theta.npz): {tag} -> archiving")
        shutil.move(json_path, os.path.join(ARCH_DIR, f"{tag}.json"))
        sys.exit(0)
    rel_use = a_dict.get("relabel", "none")
    if rel_use == "bpfeat_canon":
        rel_use = "none"
    argv = ["gnn_5block.py", "--tag", tag, "--n", str(a_dict["n"]), "--sched", a_dict["sched"],
            "--order", a_dict["order"], "--relabel", rel_use,
            "--arch", a_dict.get("arch", "mp"), "--H", str(a_dict.get("H", 128)), "--rounds", str(a_dict.get("rounds", 4))]
    if a_dict.get("clause_w", False):
        argv.append("--clause_w")
    if a_dict.get("pure_gnn", False):
        argv.append("--pure_gnn")
    sys.argv = argv
    sys.path.insert(0, "/workspace/qumode_sep/oct_explore/oct23_exp")
    import gnn_5block as G5
    th = jnp.asarray(np.load(npz_path)["theta"], jnp.float32)
    rng = np.random.default_rng(777 + int(a_dict.get("seed", 0)))
    p0 = G5.init_net(rng)
    th0, unr = ravel_pytree(p0)
    if th.size != th0.size:
        print(f"STALE (param size {th.size} != {th0.size}): {tag} -> archiving")
        shutil.move(json_path, os.path.join(ARCH_DIR, f"{tag}.json"))
        shutil.move(npz_path, os.path.join(ARCH_DIR, f"{tag}_theta.npz"))
        sys.exit(0)
    D_te = G5.prepare_split("test")
    A_b, TH_b, dg, w_b = G5.forward_net(unr(th), D_te["G"])
    p_te, lk_te = G5.SIM_MAIN(A_b, TH_b, dg, w_b, D_te["Cn_rel"], D_te["Succ"], D_te["Viol"])
    p_chk, lk_chk = G5.SIM_CHK(A_b, TH_b, dg, w_b, D_te["Cn_rel"], D_te["Succ"], D_te["Viol"])
    p_saved = np.array(J["p_test"], float)
    max_diff = float(np.max(np.abs(np.asarray(p_te, float) - p_saved)))

    p_zero, _ = G5.SIM_CHK(A_b, TH_b, jnp.zeros_like(dg), jnp.zeros_like(w_b), D_te["Cn_rel"], D_te["Succ"], D_te["Viol"])
    p_zero = np.asarray(p_zero, float)
    Succ_np = np.asarray(D_te["Succ"], float)

if max_diff > 1e-4:
    print(f"STALE (max_diff={max_diff:.3e} > 1e-4): {tag} -> archiving")
    shutil.move(json_path, os.path.join(ARCH_DIR, f"{tag}.json"))
    npz_path = os.path.join(RES_DIR, f"{tag}_theta.npz")
    if os.path.exists(npz_path):
        shutil.move(npz_path, os.path.join(ARCH_DIR, f"{tag}_theta.npz"))
    sys.exit(0)

p_chk_np = np.asarray(p_chk, float)
inv_chk = 1.0 / np.maximum(p_chk_np, 1e-12)
inv_zero = 1.0 / np.maximum(p_zero, 1e-12)
presolve_mask = (Succ_np[:, 0, 0] > 0)
unsolved_mask = ~presolve_mask
unsolved_inv = inv_chk[unsolved_mask]

J["audit"] = {
    "max_repro_diff": max_diff,
    "presolve_00_frac": float(np.mean(presolve_mask)),
    "zero_oracle_med": float(np.median(inv_zero)),
    "zero_oracle_B90": float(np.quantile(inv_zero, 0.9)),
    "unsolved_count": int(np.sum(unsolved_mask)),
    "unsolved_med": float(np.median(unsolved_inv)) if len(unsolved_inv) > 0 else 0.0,
    "unsolved_B90": float(np.quantile(unsolved_inv, 0.9)) if len(unsolved_inv) > 0 else 0.0,
}
json.dump(J, open(json_path, "w"))
print(f"OK {tag:30s}: max_diff={max_diff:.2e} | p00={J['audit']['presolve_00_frac']*100:4.0f}% | zero_oracle={J['audit']['zero_oracle_med']:5.2f}/{J['audit']['zero_oracle_B90']:6.2f} | unsolved({J['audit']['unsolved_count']:2d})={J['audit']['unsolved_med']:6.2f}/{J['audit']['unsolved_B90']:6.2f}")
