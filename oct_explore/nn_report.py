"""Append markdown table for an nn_costgx run vs QAOA_raw / G_qfe."""
import json, sys, time
q = json.load(open("results/QAOA_raw.json"))["test"]
gq = json.load(open("results/G_qfe.json"))["test"] if __import__("os").path.exists("results/G_qfe.json") else {}
for tag in sys.argv[1:]:
    d = json.load(open(f"results/{tag}.json"))
    a = d.get("args_last", d["args"])
    flags = " ".join(f"{k}={v}" for k, v in a.items()
                     if k in ("mode", "rich", "per_cav", "per_bit", "beta0", "beta_min", "beta_w",
                              "hidden", "depth", "delta_scale", "ntrain", "stages", "final_ecd", "fix_beta", "phys_beta", "ebins", "bmax"))
    print(f"\n## {tag} ({time.strftime('%a %H:%M CST')})  [{flags}]\n")
    # collect by L -> n -> metrics from test keys like "8+10|L4|n20"
    from collections import defaultdict
    by = defaultdict(dict)  # L -> n -> s
    for k, s in d.get("test", {}).items():
        # key: stage|L{L}|n{n}
        try:
            parts = k.split("|"); L = int(parts[1][1:]); n = int(parts[2][1:])
        except Exception:
            continue
        # keep best (lowest median) across stages for each (L,n), prefer fullest stage
        prev = by[L].get(n)
        if prev is None or len(parts[0]) >= len(prev[0]) or s["median_inv_p"] < prev[1]["median_inv_p"]:
            by[L][n] = (parts[0], s)
    ns = sorted({n for L in by for n in by[L]})
    Ls = sorted(by)
    if not Ls:
        print("(no test results yet)\n"); continue
    print("| L | " + " | ".join(f"n={n}" for n in ns) + " |")
    print("|---|" + "|".join(["---"] * len(ns)) + "|")
    for L in Ls:
        print(f"| {L} | " + " | ".join(
            f"{by[L][n][1]['median_inv_p']:.1f}/{by[L][n][1]['B90_inv_p']:.1f}" if n in by[L] else "-"
            for n in ns) + " |")
        print(f"| QAOA {L} | " + " | ".join(
            f"{q[str(n)][str(L)]['median_inv_p']:.1f}/{q[str(n)][str(L)]['B90_inv_p']:.1f}"
            if str(n) in q and str(L) in q[str(n)] else "-" for n in ns) + " |")
        if gq:
            print(f"| G_qfe {L} | " + " | ".join(
                f"{gq[str(n)][str(L)]['median_inv_p']:.1f}/{gq[str(n)][str(L)]['B90_inv_p']:.1f}"
                if str(n) in gq and str(L) in gq[str(n)] else "-" for n in ns) + " |")
