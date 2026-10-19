"""Layer-1 structure of a GNN-SNAP model (Oct 19, vacuum-start test).
usage: analyze_vac.py <tag> <run_i> [ns...]  -> per n: mean over instances of layer-1 per-digit mixer angles (beta_0+e, mod 2pi folded to [0,pi]),
block-0 transmon theta (folded), layer-1 SNAP phase rms (global phase removed), and overlap of |psi| after layer 1 with the flat state:
F_mag = (sum_x |psi_x| / sqrt(2^n))^2,  F = |<flat|psi>|^2 (flat = uniform real amplitudes on valid strings), plus pad leakage."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
tag, ri = sys.argv[1], int(sys.argv[2]); NS = [int(x) for x in sys.argv[3:]] or [8, 20]
src = json.load(open(f"results/v3/{tag}.json")); a0 = src["args"]
argv = ["--tag", "_lib_an", "--L", str(a0["L"]), "--snap_form", a0["snap_form"], "--feat", a0["feat"], "--psi0", a0.get("psi0", "uniform")]
argv += [f"--{k}" for k in ("clause_w", "var_mix") if a0.get(k)]
path = os.path.join(HERE, "gnn_snap.py"); code = open(path).read(); code = code[: code.index("# ---------- main ----------")]
sys.argv = [path] + argv; G = {"__name__": "lib", "__file__": path}; exec(compile(code, path, "exec"), G)
V, sim, jnp, L = G["V"], G["sim"], G["jnp"], G["L"]
p, b = G["unr"](jnp.asarray(src["runs"][ri]["theta"], jnp.float32))
fold = lambda x: np.abs((np.asarray(x) + np.pi) % (2 * np.pi) - np.pi)   # angle -> [0, pi] (rotation strength)
out = {"tag": tag, "run": ri, "psi0": a0.get("psi0", "uniform")}
for n in NS:
    lay = V.Layout(n, a0["code"])
    N = 20 if n <= 10 else (10 if n < 20 else 6); base = G["TRB"] if n == 8 else G["TEB"]; insts = sim.instances(n, N, base)
    Gr, cls = G["batch_graph"](n, insts, lay); bv, w, e, hv = G["net"](p, b, Gr); loc, mix, g = G["circuit_args"](lay, bv, e)
    PH = G["snap_args"](lay, p, hv, bv.shape[0]); loc, mix, g = np.asarray(loc), np.asarray(mix), np.asarray(g)
    nm = n / insts[0]["m"]; B = len(insts)
    ph0 = np.stack([nm * V.weighted_energy(n, cls[i], g[i, 0] + (0 if w is None else np.asarray(w)[i, :, 0])) for i in range(B)])
    psi = jnp.broadcast_to(G["psi0_of"](lay), (B,) + tuple(lay.shape))
    psi = G["local_snap"](lay, psi, jnp.asarray(loc[:, 0]), [P[:, 0] for P in PH])
    psi_loc = np.asarray(psi)
    pnc = jnp.asarray(psi_loc)
    for c in range(lay.C): pnc = lay.bapply(lay.mixU(c, jnp.asarray(mix[:, 0, c, : lay.ks[c]])), pnc, 1 + lay.C + c)
    pnc = np.asarray(pnc).reshape(B, -1)
    psi = psi * jnp.exp(-1j * lay.to_D(jnp.asarray(ph0))).astype(V.C64)
    for c in range(lay.C): psi = lay.bapply(lay.mixU(c, jnp.asarray(mix[:, 0, c, : lay.ks[c]])), psi, 1 + lay.C + c)
    psi = np.asarray(psi).reshape(B, -1); valid = lay.idxD < (1 << n); flat = 1 / np.sqrt(1 << n)
    Fmag = (np.abs(psi[:, valid]).sum(1) * flat) ** 2; F = np.abs(psi[:, valid].sum(1) * flat) ** 2; leak = (np.abs(psi[:, ~valid]) ** 2).sum(1)
    r = {"N": B, "F_mag_after_L1": [float(Fmag.mean()), float(Fmag.min())], "F_after_L1": float(F.mean()), "pad_leak": float(leak.mean()),
         "F_mag_L1_without_cost": float(((np.abs(pnc[:, valid]).sum(1) * flat) ** 2).mean())}
    for c, k in enumerate(lay.ks):
        snp = np.asarray(PH[c])[:, 0, : 1 << k]; snp = snp - snp.mean(-1, keepdims=True)
        r[f"reg{c}"] = {"k": k, "mix_L1_digit_folded": [round(float(v), 3) for v in fold(mix[:, 0, c, :k]).mean(0)],
                        "mix_L1_digit_raw": [round(float(v), 3) for v in mix[:, 0, c, :k].mean(0)],
                        "tr_theta_blk0_folded": round(float(fold(loc[:, 0, c, 2]).mean()), 3), "tr_theta_blk0_raw": round(float(loc[:, 0, c, 2].mean()), 3),
                        "tr_phi_blk0": round(float(loc[:, 0, c, 3].mean()), 3), "snap_L1_rms": round(float(np.sqrt((snp ** 2).mean())), 3)}
    r["g_L1"] = float(g[:, 0].mean()); out[str(n)] = r; print(json.dumps({n: r}), flush=True)
os.makedirs("results/v3/analysis", exist_ok=True)
json.dump(out, open(f"results/v3/analysis/L1_{tag}_r{ri}.json", "w"), indent=1)
