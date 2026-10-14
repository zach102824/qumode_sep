"""SATLike: N fresh unique-GS instances per n (same generator as wmaxsat clause rule), one random run each."""
import json, random, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
BEST = json.load(open('logs/strong/best_cfg.json')); N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
from noiseless.make_fresh_instances import fast_energies  # same E, ~16x faster
def job(a):
    n, k0 = a; out = []; k = k0
    while len(out) < 250:
        r = random.Random(900_000_000 + 1_000_000 * n + k); k += 1; m = int(round(9.93 * n)); cl = []
        for _ in range(m):
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; w = r.randint(1, 5)
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": w})
        d = {"n": n, "clauses": cl}; E = fast_energies(n, cl)
        if (E == E.min()).sum() > 1: continue
        d["gsv"] = int(np.argmin(E))
        h, _ = sc.runone("satlike", d, 77_000_000 + 1_000_000 * n + k, min(40 * 2 ** n, 4_000_000), BEST[f"satlike|{n}"]["cfg"])
        out.append(int(h))
    return n, out
if __name__ == "__main__":
    jobs = [(n, 100_000 * b) for n in [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else '20,18,16,14,12,10,8'.split(','))] for b in range(N // 250)]
    with Pool(8) as p, open('logs/strong/satlike_manyinst.jsonl', 'a') as f:
        for n, h in p.imap_unordered(job, jobs):
            f.write(json.dumps(dict(n=n, hits=h)) + "\n"); f.flush()
    print("ALLDONE")
