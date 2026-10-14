"""SATLike on the 100 fresh instances per n, exactly as satlike_manyinst.job (same seed, cap, cfg).
Verifies: some satlike_manyinst.jsonl block's first 100 hits equal these (=> same instances, same runs)."""
import json, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
BEST = json.load(open('logs/strong/best_cfg.json'))
def job(a):
    n, i = a; d = json.load(open(f"Hamiltonians/wmaxsat_fresh/fresh_n{n:02d}_{i}.json")); d["gsv"] = int(d["gs"], 2)
    h, _ = sc.runone("satlike", d, 77_000_000 + 1_000_000 * n + d["k"] + 1, min(40 * 2 ** n, 4_000_000), BEST[f"satlike|{n}"]["cfg"])
    return n, i, int(h)
if __name__ == "__main__":
    NS = (8, 10, 12, 14, 16, 18, 20); H = {n: [0] * 100 for n in NS}
    with Pool(8) as p:
        for n, i, h in p.imap_unordered(job, [(n, i) for n in NS for i in range(100)]): H[n][i] = h
    blocks = [json.loads(l) for l in open('logs/strong/satlike_manyinst.jsonl')]
    out = {}
    for n in NS:
        m = sum(b["hits"][:100] == H[n] for b in blocks if b["n"] == n)
        out[n] = dict(hits=H[n], matches_manyinst_block=m); print(n, "matching blocks:", m, flush=True)
    json.dump(out, open('logs/fresh/satlike_fresh100.json', 'w'))
