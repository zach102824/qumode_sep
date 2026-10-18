"""GNN-SNAP large-validation re-selection (15:29 CST 2026-10-18, Zach approved refocus on GNN-SNAP).

Takes every trained init (runs[*].theta) of one or more finished gnn_snap.py tags with identical model structure,
scores each on a FRESH, much larger n=10 and n=12 validation set, picks one init with a criterion fixed in advance
(before any n=20 number is looked at), and only then evaluates the re-picked init on the n=20 test set.
  * fresh validation instances: sim.instances(n, nval, VBASE), VBASE = 36e9 (train 31e9 / test 32e9 / old val 33e9 untouched;
    seeds = base + 1e7 n + k, so ranges cannot overlap).
  * criterion (pre-registered): minimise  S = mean_{n in 10,12} [ (log median(1/p) + log B90(1/p)) / 2 ]
    (B90-aware: tail weighted equally with median); tie-break mean -log p.
  * test: identical to gnn_snap.py (TEB 32e9, 100 instances at n=20; n=20 serialised by /tmp/v3_n20.lock).
No training happens here; selection only. Model code is gnn_snap.py itself (exec'd up to its main section).
"""
import argparse, json, os, sys, time, fcntl
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--nval", type=int, default=600); ap.add_argument("--vn", type=int, nargs="*", default=[10, 12])
ap.add_argument("--vbase", type=int, default=36_000_000_000); ap.add_argument("--test", type=int, nargs="*", default=[20])
ap.add_argument("--chunk", type=int, default=30)
r = ap.parse_args()
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
CRIT = "min mean_n[(log median 1/p + log B90 1/p)/2] over fresh val n=%s (%d inst each, base %d); tie-break mean -log p" % (r.vn, r.nval, r.vbase)
SRC = {t: json.load(open(f"results/v3/{t}.json")) for t in r.src}
for t, d in SRC.items():
    if "winner" not in d: log(f"source {t} not finished training (no winner) - exiting without result"); sys.exit(3)
STRUCT = ("code", "L", "hidden", "rounds", "clause_w", "var_mix", "snap", "snap_form", "free_m", "plain", "feat")
a0 = SRC[r.src[0]]["args"]
for t, d in SRC.items():
    bad = [k for k in STRUCT if d["args"].get(k) != a0.get(k)]
    assert not bad, f"{t}: structure differs from {r.src[0]} in {bad}"
# build gnn_snap.py argv from the first source's args (model structure only matters) and exec its library part
argv = []
for k, v in a0.items():
    if k in ("smoke",): continue
    if isinstance(v, bool):
        if v: argv.append(f"--{k}")
    elif isinstance(v, list):
        if v: argv += [f"--{k}"] + [str(x) for x in v]
    elif v is None or v == "": continue
    else: argv += [f"--{k}", str(v)]
path = os.path.join(HERE, "gnn_snap.py"); code = open(path).read(); code = code[: code.index("# ---------- main ----------")]
sys.argv = [path] + argv; G = {"__name__": "gnn_snap_lib", "__file__": path}
exec(compile(code, path, "exec"), G)
unr, eval_n, sim, V, jnp, TEB, L = G["unr"], G["eval_n"], G["sim"], G["V"], G["jnp"], G["TEB"], G["L"]

OUT = f"results/v3/{r.tag}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res["args"] = {"tag": r.tag, "L": L, "src": r.src, "nval": r.nval, "vn": r.vn, "vbase": r.vbase, "criterion": CRIT}
def save(): json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
runs = res.setdefault("runs", [])
cands = [(t, rr) for t in r.src for rr in SRC[t]["runs"]]
VAL = {n: sim.instances(n, r.nval, r.vbase) for n in r.vn}
log(f"reselect {r.tag}: {len(cands)} inits from {r.src}; {CRIT}")
for j, (t, rr) in enumerate(cands):
    if j < len(runs): continue
    p, b = unr(jnp.asarray(rr["theta"], jnp.float32)); t0 = time.time(); per = {}
    for n in r.vn:
        pp = eval_n(n, VAL[n], p, b, r.chunk); s = V.summ(pp)
        per[str(n)] = {"median_inv_p": s["median_inv_p"], "B90_inv_p": s["B90_inv_p"], "nll": float(np.mean(-np.log(pp + 1e-12)))}
    S = float(np.mean([(np.log(per[k]["median_inv_p"]) + np.log(per[k]["B90_inv_p"])) / 2 for k in per]))
    nll = float(np.mean([per[k]["nll"] for k in per]))
    known = {role: d.get("20") and {"median_inv_p": d["20"]["median_inv_p"], "B90_inv_p": d["20"]["B90_inv_p"]}
             for role, d in SRC[t].get("test", {}).items() if d.get("id") == rr["i"]}
    runs.append({"i": j, "src": t, "src_i": rr["i"], "origin": rr["origin"], "loss": rr["loss"], "old_valmean": rr.get("valmean"),
                 "valmean": S, "nll": nll, "val": per, "src_role": list(known)}); save()
    log(f"  cand {j} {t}#{rr['i']} ({rr['origin']}): train {rr['loss']:.4f} old-val {rr.get('valmean', float('nan')):.3f} | "
        + " ".join(f"n={k} {v['median_inv_p']:.1f}/{v['B90_inv_p']:.1f}" for k, v in per.items()) + f" | S={S:.4f} nll={nll:.4f} ({time.time()-t0:.0f}s)")
if "winner" not in res:
    w = min(runs, key=lambda x: (round(x["valmean"], 6), x["nll"]))
    res["winner"] = int(w["i"]); res["trainbest"] = int(min(runs, key=lambda x: x["loss"])["i"]); save()
    log(f"RE-PICK (fixed before n=20): cand {w['i']} = {w['src']}#{w['src_i']} S={w['valmean']:.4f} (train-best cand {res['trainbest']})")
te = res.setdefault("test", {}); k = res["winner"]; w = runs[k]; te.setdefault("winner", {"id": k})
p, b = unr(jnp.asarray(SRC[w["src"]]["runs"][w["src_i"]]["theta"], jnp.float32))
for n in r.test:
    if str(n) in te["winner"]: continue
    N = 200 if n == 14 else 100; ins = sim.instances(n, N, TEB); ch = 50 if n <= 12 else (10 if n < 20 else 2)
    lk = open("/tmp/v3_n20.lock", "w") if n >= 18 else None
    if lk: fcntl.flock(lk, fcntl.LOCK_EX)
    t0 = time.time(); s = V.summ(eval_n(n, ins, p, b, ch))
    if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    te["winner"][str(n)] = s; save()
    log(f"test re-picked cand {k} ({w['src']}#{w['src_i']}) n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} ({time.time()-t0:.0f}s)")
log("ALLDONE")
