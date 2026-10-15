"""SATLike lookups (strict count) to reach 90% and 99% ground-state success.

Source: logs/strong/eval.jsonl (tuned SATLike-style, 200 runs per n = 10 instances x 20 seeds).
A lookup = one bitstring energy evaluated; success = first time the ground state is scored.
Run `python noiseless/SATLike.py` to recompute from the raw data.
"""
import json, collections, numpy as np, os

# MAIN: fresh instances, one run per instance (logs/strong/satlike_manyinst.jsonl)
# n: (instances, median, B90, B99); 2000 instances per n, 6000 at n=20
SATLIKE_FRESH = {8: (2000, 31, 107, 249), 10: (2000, 50, 180, 429), 12: (2000, 91, 320, 951),
                 14: (2000, 113, 370, 1095), 16: (2000, 142, 540, 1730), 18: (2000, 177, 620, 1740),
                 20: (6000, 227, 886, 2905)}
# CONV (logs/conv/satlike_conv.jsonl, noiseless/satlike_convergence.py): 2000 new instances x 5 runs per n,
# value and bootstrap std over instances; unsolved=inf. n=12 std(B99)=inf: 95/10000 unsolved (cfg mode=1), near 1%.
SATLIKE_CONV = {8: {'median': (32, 0.7), 'B90': (114, 2.1), 'B99': (252, 8.6), 'unsolved': 0},
                10: {'median': (51, 1.0), 'B90': (183, 3.6), 'B99': (433, 19.4), 'unsolved': 0},
                12: {'median': (95, 2.1), 'B90': (323, 6.0), 'B99': (1170, float('inf')), 'unsolved': 95},
                14: {'median': (104, 1.8), 'B90': (373, 7.3), 'B99': (929, 46.4), 'unsolved': 1},
                16: {'median': (140, 2.5), 'B90': (514, 10.5), 'B99': (1439, 85.0), 'unsolved': 5},
                18: {'median': (183, 3.5), 'B90': (673, 15.1), 'B99': (2031, 162.6), 'unsolved': 0},
                20: {'median': (221, 3.9), 'B90': (871, 18.8), 'B99': (2877, 161.8), 'unsolved': 0}}
# OLD (10 instances only, bumpy): Saved results (n: (B90, B99)) from logs/strong/eval.jsonl, 200 runs per n
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

def recompute_fresh(path=os.path.join(os.path.dirname(__file__), "..", "logs/strong/satlike_manyinst.jsonl")):
    d = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l); d[r["n"]] += [h if h >= 0 else 10**9 for h in r["hits"]]
    return {n: (len(a), int(np.median(a)), int(np.percentile(a, 90, method="higher")), int(np.percentile(a, 99, method="higher")))
            for n, a in ((n, np.array(v)) for n, v in sorted(d.items()))}

if __name__ == "__main__":
    print("FRESH: n inst median B90 B99")
    for n, v in recompute_fresh().items(): print(n, *v)
    print("n  B90  B99  (B99 10k runs)")
    for n, (b90, b99) in recompute().items():
        print(n, b90, b99, SATLIKE_B99_10K.get(n, "-"))
