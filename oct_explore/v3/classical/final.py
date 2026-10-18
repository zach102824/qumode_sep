from runcls import *
import time
R = 200; CFG = {"walksat_p0.5": (0, 0.5, 0, 0), "SA_tuned_n8train": (1, 1.0, 2.0, 5), "SA_tuned_n20train": (1, 1.0, 3.0, 50)}
out = {}
for n in (8, 10, 12, 14, 20):
    T = I[f"test_{n}"]; t0 = time.time(); d = {"N": len(T), "seeds": [i["seed"] for i in T], "nsol": [i["nsol"] for i in T]}
    d["random_guess"] = {"per_inst": [2 ** n / i["nsol"] for i in T]}
    for k, (mode, a, b, s) in CFG.items():
        f = [run(i, n, R, mode, a, b, s * n, seed=7) for i in T]
        d[k] = {"flips_per_inst": [x[0] for x in f], "evals_per_inst": [x[1] for x in f]}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk in list(v):
                if kk.endswith("per_inst"): v[kk.replace("per_inst", "stats")] = stats(v[kk])
    out[str(n)] = d
    print(n, f"{time.time()-t0:.0f}s", {k: (v.get("evals_stats") or v.get("per_stats")) for k, v in d.items() if isinstance(v, dict)}, flush=True)
json.dump(out, open(os.path.join(HERE, "final_raw.json"), "w"))
