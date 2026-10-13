"""SA-alone lookups-to-first-GS-hit, same SA as classical_layout.sa_sample (QM_SA_COUNT=raw).

sa_sample's budget only truncates the RNG stream (restart chains of 8n steps from new random
starts, 64 counted probes once per run), so the lookup index of the first GS evaluation is
budget-independent: success(B) = P(hit <= B). One long run per (inst, seed) gives every budget.
"""
import json, os, sys, numpy as np
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noiseless.run_relayout_kbit import ham_paths, energies_for

NS = [8, 10, 12, 14, 16, 18, 20]; SEEDS = 20; SEED0 = 20261013
OUT = Path("logs/3cav/saalone"); STOP = OUT / "SAALONE.STOP"

def first_hit(E, n, gsv, rng, cap, T0=2.0, T1=0.05, npr=64):
    cnt = 0
    scale = float(np.std(E[rng.integers(0, len(E), npr)]))
    for v in rng.integers(0, len(E), npr):
        cnt += 1
        if int(v) == gsv: return cnt
    scale = max(scale, 1e-9)
    nsteps = 8 * n
    Ts = [scale * T0 * (T1 / T0) ** (t / max(1, nsteps - 1)) for t in range(nsteps)]
    while cnt < cap:
        v = int(rng.integers(len(E))); cnt += 1; e = E[v]
        if v == gsv: return cnt
        for t in range(nsteps):
            if cnt >= cap: break
            w = v ^ (1 << int(rng.integers(n))); cnt += 1; ew = E[w]
            if w == gsv: return cnt
            if ew <= e or rng.random() < np.exp(-(ew - e) / Ts[t]):
                v, e = w, ew
    return None

def job(a):
    n, i, s = a
    if STOP.exists(): return None
    p = ham_paths(n, "wmaxsat")[i]
    E, gs = energies_for(str(p), n, "wmaxsat")
    E = E.tolist()  # python floats: faster scalar access, same values
    E = np.asarray(E)
    rng = np.random.default_rng(SEED0 + 1000 * n + 100 * i + s)
    return dict(n=n, inst=i, seed=s, hit=first_hit(E, n, int(gs, 2), rng, cap=40 * 2 ** n))

if __name__ == "__main__":
    f = OUT / "hits.jsonl"
    done = set()
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l); done.add((r["n"], r["inst"], r["seed"]))
    jobs = [(n, i, s) for n in NS for i in range(10) for s in range(SEEDS) if (n, i, s) not in done]
    with Pool(8) as pool, f.open("a") as fh:
        for r in pool.imap_unordered(job, jobs):
            if r is None: continue
            fh.write(json.dumps(r) + "\n"); fh.flush()
    (OUT / "SAALONE.done").write_text("ok\n")
