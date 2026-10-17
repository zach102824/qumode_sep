"""Append a markdown table for a costgx run vs QAOA_raw at same L and SATLike."""
import json, sys, time
SAT = {8: "16/70", 10: "26/111", 12: "36/157", 14: "50/208", 20: "~124/~530"}
q = json.load(open("results/QAOA_raw.json"))["test"]
for tag in sys.argv[1:]:
    d = json.load(open(f"results/{tag}.json")); te = d["test"]; ns = sorted(te, key=int)
    a = d.get("args_last", d["args"])
    flags = " ".join(f"{k}={v}" for k, v in a.items() if k in ("final_ecd", "ecd2", "init", "obj", "alpha", "w", "nt2", "beta_l2", "layout", "code", "ntrain") )
    print(f"\n## {tag} finished ({time.strftime('%a %H:%M CST')})  [{flags}]\n")
    print("| L | " + " | ".join(f"n={n}" for n in ns) + " |"); print("|---|" + "|".join(["---"] * len(ns)) + "|")
    for L in sorted({L for n in te for L in te[n]}, key=int):
        print(f"| {L} | " + " | ".join(f"{te[n][L]['median_inv_p']:.1f}/{te[n][L]['B90_inv_p']:.1f}" if L in te[n] else "-" for n in ns) + " |")
        if L in q.get(ns[-1], {}):
            print(f"| QAOA {L} | " + " | ".join(f"{q[n][L]['median_inv_p']:.1f}/{q[n][L]['B90_inv_p']:.1f}" if L in q.get(n, {}) else "-" for n in ns) + " |")
    print("| SATLike | " + " | ".join(SAT.get(int(n), "-") for n in ns) + " |")
