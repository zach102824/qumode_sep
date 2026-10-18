"""Phase-1 multi-architecture validation re-selection (16:09 CST 2026-10-18, Zach approved).

Scores every trained init across one or more finished gnn_snap.py tags (may differ in --arch),
with Zach's fixed criterion BEFORE looking at the held-out n=20 TEST set:
  S = mean_{n in vn} [ (log median(1/p) + log B90(1/p)) / 2 ]
Default vn = [10, 14, 20] with per-n counts [600, 300, 100].
Validation base VBASE = 37e9 (disjoint from train 31e9 / test 32e9 / old val 33e9 / R_* 36e9).
Final TEST: only the winning init, TEB 32e9, 100 instances at n=20 (same fixed set as all other reports).
Each source is loaded with ITS OWN args (different arches have different param trees).
"""
import argparse, json, os, sys, time, fcntl
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))  # 21:55 fix: sim.py lives in oct_explore/
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--vn", type=int, nargs="*", default=[10, 14, 20])
ap.add_argument("--nval", type=int, nargs="*", default=[600, 300, 100],
                help="instance counts matching --vn (default 600/300/100 for n=10/14/20)")
ap.add_argument("--vbase", type=int, default=37_000_000_000)
ap.add_argument("--test", type=int, nargs="*", default=[20])
ap.add_argument("--chunk", type=int, default=30)
r = ap.parse_args()
assert len(r.nval) == len(r.vn), "--nval length must match --vn"
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
CRIT = ("min mean_n[(log median 1/p + log B90 1/p)/2] over fresh val n=%s counts=%s base=%d; "
        "tie-break mean -log p; TEST only winner on TEB n=20") % (r.vn, r.nval, r.vbase)
SRC = {t: json.load(open(f"results/v3/{t}.json")) for t in r.src}
for t, d in SRC.items():
    if "winner" not in d:
        log(f"source {t} not finished training (no winner) - exiting"); sys.exit(3)

# Cache gnn_snap library namespaces keyed by structure (arch + snap heads + L/H/...)
STRUCT = ("arch", "code", "L", "hidden", "rounds", "clause_w", "var_mix", "snap", "snap_form",
          "free_m", "plain", "feat")
_LIB = {}


def lib_for(a0):
    key = tuple(a0.get(k) for k in STRUCT)
    if key in _LIB: return _LIB[key]
    argv = []
    for k, v in a0.items():
        if k in ("smoke", "tag", "warm", "warm_ids", "init_from", "ninit", "ntrain", "maxiter",
                 "val", "val_count", "test", "seed", "chunk"): continue
        if isinstance(v, bool):
            if v: argv.append(f"--{k}")
        elif isinstance(v, list):
            if v: argv += [f"--{k}"] + [str(x) for x in v]
        elif v is None or v == "": continue
        else: argv += [f"--{k}", str(v)]
    # older results lack --arch: default mp
    if "arch" not in a0: argv += ["--arch", "mp"]
    argv += ["--tag", "_lib_reselect_arch"]  # 21:58 fix: gnn_snap requires --tag (dummy; lib part only defines save())
    path = os.path.join(HERE, "gnn_snap.py")
    code = open(path).read(); code = code[: code.index("# ---------- main ----------")]
    sys.argv = [path] + argv
    G = {"__name__": "gnn_snap_lib", "__file__": path}
    exec(compile(code, path, "exec"), G)
    _LIB[key] = G
    log(f"  loaded model arch={G['a'].arch} L={G['L']} H={G['H']} form={G['a'].snap_form} params~OK")
    return G


OUT = f"results/v3/{r.tag}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res["args"] = {"tag": r.tag, "L": SRC[r.src[0]]["args"]["L"], "src": r.src, "vn": r.vn, "nval": r.nval, "vbase": r.vbase, "criterion": CRIT}
def save(): json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

runs = res.setdefault("runs", [])
cands = [(t, rr) for t in r.src for rr in SRC[t]["runs"]]
# Build validation sets once (shared across arches)
import sim as _sim
VAL = {n: _sim.instances(n, c, r.vbase) for n, c in zip(r.vn, r.nval)}
log(f"reselect_arch {r.tag}: {len(cands)} inits from {r.src}; {CRIT}")
log(f"  val sizes: " + ", ".join(f"n={n}:{len(VAL[n])}" for n in r.vn))

for j, (t, rr) in enumerate(cands):
    if j < len(runs): continue
    a0 = SRC[t]["args"]; G = lib_for(a0)
    unr, eval_n, V, jnp = G["unr"], G["eval_n"], G["V"], G["jnp"]
    p, b = unr(jnp.asarray(rr["theta"], jnp.float32)); t0 = time.time(); per = {}
    for n in r.vn:
        ch = 50 if n <= 12 else (10 if n < 20 else 2)
        # serialize n>=18 like training jobs
        lk = open("/tmp/v3_n20.lock", "w") if n >= 18 else None
        if lk: fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            pp = eval_n(n, VAL[n], p, b, ch); s = V.summ(pp)
        finally:
            if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        per[str(n)] = {"median_inv_p": s["median_inv_p"], "B90_inv_p": s["B90_inv_p"],
                       "nll": float(np.mean(-np.log(pp + 1e-12)))}
    S = float(np.mean([(np.log(per[k]["median_inv_p"]) + np.log(per[k]["B90_inv_p"])) / 2 for k in per]))
    nll = float(np.mean([per[k]["nll"] for k in per]))
    runs.append({"i": j, "src": t, "src_i": rr["i"], "origin": rr.get("origin", ""), "arch": a0.get("arch", "mp"),
                 "loss": rr["loss"], "old_valmean": rr.get("valmean"),
                 "valmean": S, "nll": nll, "val": per}); save()
    log(f"  cand {j} {t}#{rr['i']} arch={a0.get('arch','mp')} ({rr.get('origin','')}): train {rr['loss']:.4f} | "
        + " ".join(f"n={k} {v['median_inv_p']:.1f}/{v['B90_inv_p']:.1f}" for k, v in per.items())
        + f" | S={S:.4f} nll={nll:.4f} ({time.time()-t0:.0f}s)")

if "winner" not in res:
    w = min(runs, key=lambda x: (round(x["valmean"], 6), x["nll"]))
    res["winner"] = int(w["i"]); res["trainbest"] = int(min(runs, key=lambda x: x["loss"])["i"]); save()
    log(f"RE-PICK (fixed before TEST): cand {w['i']} = {w['src']}#{w['src_i']} arch={w['arch']} S={w['valmean']:.4f}")

# Final TEST on TEB — winner only
te = res.setdefault("test", {}); k = res["winner"]; w = runs[k]; te.setdefault("winner", {"id": k})
G = lib_for(SRC[w["src"]]["args"])
unr, eval_n, V, jnp, TEB, L = G["unr"], G["eval_n"], G["V"], G["jnp"], G["TEB"], G["L"]
p, b = unr(jnp.asarray(SRC[w["src"]]["runs"][w["src_i"]]["theta"], jnp.float32))
for n in r.test:
    if str(n) in te["winner"]: continue
    N = 200 if n == 14 else 100; ins = _sim.instances(n, N, TEB); ch = 50 if n <= 12 else (10 if n < 20 else 2)
    lk = open("/tmp/v3_n20.lock", "w") if n >= 18 else None
    if lk: fcntl.flock(lk, fcntl.LOCK_EX)
    t0 = time.time(); s = V.summ(eval_n(n, ins, p, b, ch))
    if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    te["winner"][str(n)] = s; save()
    log(f"TEST re-picked cand {k} ({w['src']}#{w['src_i']} arch={w['arch']}) n={n} L={L} "
        f"med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} ({time.time()-t0:.0f}s)")
# also dump all cands' val scores for the report
res["leaderboard"] = sorted([{"i": x["i"], "src": x["src"], "src_i": x["src_i"], "arch": x["arch"],
                              "S": x["valmean"], "loss": x["loss"], "val": x["val"]} for x in runs],
                            key=lambda z: z["S"])
save()
log("ALLDONE")
