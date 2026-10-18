"""Analysis scaffolding for Phase-1 architecture scan (16:09 CST 2026-10-18).

For each finished gnn_snap tag, dumps:
  1. SNAP phase / clause-weight correlations vs clause count, var degree, clause signs
  2. Transfer check: output magnitude at n=8 vs n=14 vs n=20 (explode/collapse?)
  3. B90-tail instance traits (nsol, mean/max degree) on the TEB n=20 test set
  4. Feature-ablation hooks (plan only unless --ablate)

Usage:
  python3.13 -u v3/analyze_arch.py --tags S_g4_snapF_f N_g4_attn ... [--nprobe 40]
Writes results/v3/analyze_<tag>.json and appends a short markdown block to stdout.
"""
from __future__ import annotations
import argparse, json, os, sys, time, fcntl
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--nprobe", type=int, default=40, help="instances per n for phase/weight probes")
ap.add_argument("--ns", type=int, nargs="*", default=[8, 14, 20])
ap.add_argument("--ablate", action="store_true", help="run feature ablation (expensive; deferred by default)")
r = ap.parse_args()
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)

STRUCT_SKIP = ("smoke", "tag", "warm", "warm_ids", "init_from", "ninit", "ntrain", "maxiter",
               "val", "val_count", "test", "seed", "chunk")


def load_lib(a0):
    argv = []
    for k, v in a0.items():
        if k in STRUCT_SKIP: continue
        if isinstance(v, bool):
            if v: argv.append(f"--{k}")
        elif isinstance(v, list):
            if v: argv += [f"--{k}"] + [str(x) for x in v]
        elif v is None or v == "": continue
        else: argv += [f"--{k}", str(v)]
    if "arch" not in a0: argv += ["--arch", "mp"]
    argv += ["--tag", "_lib_analyze_arch"]  # same fix as reselect_arch 21:58: gnn_snap requires --tag (dummy; lib part only defines save())
    path = os.path.join(HERE, "gnn_snap.py")
    code = open(path).read(); code = code[: code.index("# ---------- main ----------")]
    sys.argv = [path] + argv
    G = {"__name__": "gnn_snap_lib", "__file__": path}
    exec(compile(code, path, "exec"), G)
    return G


def instance_traits(n, ins, lay, V):
    """Per-instance: nsol, m, mean_deg, max_deg, frac_pos_lits."""
    out = []
    for i in ins:
        m, cl = V.clauses(n, i)
        deg = np.zeros(n)
        npos = 0; nlit = 0
        for v, s in cl:
            for vv, ss in zip(v, s):
                deg[vv] += 1; nlit += 1; npos += int(ss > 0)
        out.append({"seed": i["seed"], "nsol": int(i["nsol"]), "m": m,
                    "mean_deg": float(deg.mean()), "max_deg": float(deg.max()),
                    "frac_pos": float(npos / max(nlit, 1))})
    return out


def probe(tag):
    d = json.load(open(f"results/v3/{tag}.json"))
    if "winner" not in d:
        log(f"{tag}: no winner yet, skip"); return None
    a0 = d["args"]; G = load_lib(a0)
    unr, eval_n, net, snap_args, batch_graph, circuit_args = (
        G["unr"], G["eval_n"], G["net"], G["snap_args"], G["batch_graph"], G["circuit_args"])
    V, jnp, sim, L, TEB = G["V"], G["jnp"], G["sim"], G["L"], G["TEB"]
    wid = d["winner"]; rr = d["runs"][wid]
    p, b = unr(jnp.asarray(rr["theta"], jnp.float32))
    report = {"tag": tag, "arch": a0.get("arch", "mp"), "winner": wid, "loss": rr["loss"],
              "transfer": {}, "corr": {}, "tail": {}, "ablate_plan": [
                  "zero fv pos/neg degree channels",
                  "shuffle clause signs (Ip/In swap random subset)",
                  "permute variable ids",
                  "drop clause_w head / drop var_mix head / drop SNAP head (one at a time)",
              ]}
    # --- transfer + correlations on probe sets ---
    for n in r.ns:
        ins = sim.instances(n, r.nprobe, 38_000_000_000 + n)  # probe base, not val/test
        lay = V.Layout(n, a0.get("code", "gray"))
        Gbat, cls = batch_graph(n, ins, lay)
        bv, w, e, hv = net(p, b, Gbat)
        PH = snap_args(lay, p, hv, bv.shape[0])
        # SNAP phase magnitude per instance (rms over used Fock, layer 0)
        phase_rms = []
        if PH is not None:
            for c in range(lay.C):
                ph = np.asarray(PH[c])[:, :, : 1 << lay.ks[c]]  # (B,L,2^k)
                ph = ph - ph.mean(-1, keepdims=True)
                phase_rms.append(np.sqrt((ph ** 2).mean(axis=(1, 2))))  # (B,)
            phase_rms = np.mean(phase_rms, 0)
        else:
            phase_rms = np.zeros(len(ins))
        w_rms = np.zeros(len(ins)) if w is None else np.sqrt(np.asarray(w).reshape(len(ins), -1) ** 2).mean(1)
        traits = instance_traits(n, ins, lay, V)
        degs = np.array([t["mean_deg"] for t in traits])
        nsols = np.array([t["nsol"] for t in traits])
        ms = np.array([t["m"] for t in traits])
        def corr(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
            return float(np.corrcoef(x, y)[0, 1])
        report["transfer"][str(n)] = {
            "phase_rms_mean": float(phase_rms.mean()), "phase_rms_std": float(phase_rms.std()),
            "phase_rms_max": float(phase_rms.max()),
            "w_rms_mean": float(w_rms.mean()), "w_rms_max": float(w_rms.max()),
        }
        report["corr"][str(n)] = {
            "phase_vs_mean_deg": corr(phase_rms, degs),
            "phase_vs_nsol": corr(phase_rms, nsols),
            "phase_vs_m": corr(phase_rms, ms),
            "w_vs_mean_deg": corr(w_rms, degs),
            "w_vs_nsol": corr(w_rms, nsols),
        }
        log(f"  {tag} n={n}: phase_rms {phase_rms.mean():.3f}±{phase_rms.std():.3f} (max {phase_rms.max():.3f}) "
            f"w_rms {w_rms.mean():.3f} | corr(phase,deg)={report['corr'][str(n)]['phase_vs_mean_deg']:+.2f}")
    # --- B90 tail on TEB n=20 (use stored test p if present, else eval) ---
    te = d.get("test", {}).get("winner", {})
    if "20" in te and "p" in te["20"]:
        pp = np.array(te["20"]["p"]); ins20 = sim.instances(20, len(pp), TEB)
    else:
        ins20 = sim.instances(20, 100, TEB)
        lk = open("/tmp/v3_n20.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
        try: pp = eval_n(20, ins20, p, b, 2)
        finally: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    inv = 1 / np.maximum(pp, 1e-300)
    thr = np.quantile(inv, 0.9)
    lay20 = V.Layout(20, a0.get("code", "gray"))
    traits20 = instance_traits(20, ins20, lay20, V)
    tail = [traits20[i] | {"inv_p": float(inv[i])} for i in range(len(ins20)) if inv[i] >= thr]
    body = [traits20[i] | {"inv_p": float(inv[i])} for i in range(len(ins20)) if inv[i] < thr]
    def avg(key, rows): return float(np.mean([x[key] for x in rows])) if rows else float("nan")
    report["tail"] = {
        "B90_thr": float(thr), "n_tail": len(tail),
        "tail_mean_nsol": avg("nsol", tail), "body_mean_nsol": avg("nsol", body),
        "tail_mean_max_deg": avg("max_deg", tail), "body_mean_max_deg": avg("max_deg", body),
        "tail_mean_deg": avg("mean_deg", tail), "body_mean_deg": avg("mean_deg", body),
        "tail_seeds": [t["seed"] for t in tail[:15]],
    }
    log(f"  {tag} n=20 tail: nsol {report['tail']['tail_mean_nsol']:.2f} vs body {report['tail']['body_mean_nsol']:.2f}; "
        f"max_deg {report['tail']['tail_mean_max_deg']:.1f} vs {report['tail']['body_mean_max_deg']:.1f}")
    # transfer summary: ratio of phase_rms n=20 / n=8
    t8 = report["transfer"].get("8", {}).get("phase_rms_mean", 1e-9)
    t20 = report["transfer"].get("20", {}).get("phase_rms_mean", 0.0)
    report["transfer_ratio_20_over_8"] = float(t20 / max(t8, 1e-12))
    outp = f"results/v3/analyze_{tag}.json"
    json.dump(report, open(outp, "w"), indent=2)
    log(f"  wrote {outp} (transfer_ratio_20/8={report['transfer_ratio_20_over_8']:.2f})")
    if r.ablate:
        log(f"  --ablate requested but not implemented in this scaffolding pass; hooks listed in ablate_plan")
    return report


os.makedirs("results/v3", exist_ok=True)
all_r = {}
for t in r.tags:
    log(f"analyze {t}")
    all_r[t] = probe(t)
summary = {k: v for k, v in all_r.items() if v}
json.dump(summary, open("results/v3/analyze_arch_summary.json", "w"), indent=2)
# stub so run_v3 finalize/done_ JSON path is happy (ALLDONE in log is primary)
tag = "A_arch_scan"
stub = {"args": {"tag": tag, "L": 4, "tags": r.tags}, "winner": 0,
        "test": {"winner": {"id": 0, "8": {"median_inv_p": 0.0, "B90_inv_p": 0.0}}}, "summary_path": "results/v3/analyze_arch_summary.json",
        "n_ok": len(summary)}
# if invoked with --tag via queue, queue tag is A_arch_scan; write that name
out_tag = os.environ.get("V3_ANALYZE_TAG", tag)
# detect tag from argv roughly: queue uses --tags not --tag; write A_arch_scan.json always when that is the log name
json.dump(stub, open(f"results/v3/{out_tag}.json", "w"), indent=2)
log("ALLDONE")
