"""SATLike lookups (strict count) to reach 90% and 99% ground-state success.

Source: logs/strong/eval.jsonl (tuned SATLike-style, 200 runs per n = 10 instances x 20 seeds).
A lookup = one bitstring energy evaluated; success = first time the ground state is scored.
Run `python noiseless/SATLike.py` to recompute from the raw data.
"""
import json, collections, numpy as np, os

# Saved results (n: (B90, B99)) from logs/strong/eval.jsonl, 200 runs per n
SATLIKE_B90_B99 = {8: (153, 298), 10: (162, 287), 12: (356, 1183), 14: (299, 962),
                   16: (532, 1050), 18: (1004, 2538), 20: (795, 2718)}
# More reliable 99% from 10,000 runs per n (logs/strong/satlike_many.jsonl, 10 instances x 1000 seeds)
SATLIKE_B99_10K = {8: 357, 10: 280, 12: 1047, 14: 633, 16: 1078}

def recompute(path=os.path.join(os.path.dirname(__file__), "..", "logs/strong/eval.jsonl")):
    d = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r.get("phase", "eval") != "eval" or r["arm"] != "satlike": continue
        for h in r["hits"]:
            v = h[2] if isinstance(h, list) else h
            d[r["n"]].append(v if v >= 0 else 10**9)
    return {n: (int(np.percentile(a, 90, method="higher")), int(np.percentile(a, 99, method="higher")))
            for n, a in sorted(d.items())}

if __name__ == "__main__":
    print("n  B90  B99  (B99 10k runs)")
    for n, (b90, b99) in recompute().items():
        print(n, b90, b99, SATLIKE_B99_10K.get(n, "-"))
