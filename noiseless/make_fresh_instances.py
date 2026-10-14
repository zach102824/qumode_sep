"""Write the first 100 unique-GS instances of block b=0 of satlike_manyinst.py (k from 0) as wmaxsat JSON."""
import json, random, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
from noiseless.run_relayout_kbit import wmaxsat_energies
K = 100
def fast_energies(n, cl):
    """Same E as strong_classical.energies (x_v = bit n-v of the index), by adding w on each clause's violated subcube."""
    T = np.zeros([2] * n, np.int64)
    for c in cl:
        ix = [slice(None)] * n
        for v, s in zip(c["vars"], c["signs"]): ix[v - 1] = 0 if s > 0 else 1
        T[tuple(ix)] += c["w"]
    return T.reshape(-1)
def gen(n):
    k = 0; c = 0
    while c < K:
        seed = 900_000_000 + 1_000_000 * n + k; r = random.Random(seed); k += 1; m = int(round(9.93 * n)); cl = []
        for _ in range(m):  # identical to satlike_manyinst.job
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; w = r.randint(1, 5)
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": w})
        E = fast_energies(n, cl)
        if k <= 3: assert np.array_equal(E, sc.energies({"n": n, "clauses": cl}))
        if (E == E.min()).sum() > 1: continue
        g = int(np.argmin(E)); u = np.unique(E)
        d = {"n": n, "m": m, "ratio": 9.93, "seed": seed, "k": k - 1, "gs": format(g, f"0{n}b"), "E_gs": int(E[g]),
             "n_viol_gs": int(sum(all(((g >> (n - v)) & 1) == (0 if s > 0 else 1) for v, s in zip(c_["vars"], c_["signs"])) for c_ in cl)),
             "E_2nd": int(u[1]), "n_first_excited": int((E == u[1]).sum()), "clauses": cl}
        p = f"Hamiltonians/wmaxsat_fresh/fresh_n{n:02d}_{c}.json"; open(p, "w").write(json.dumps(d))
        E2, meta = wmaxsat_energies(p, n)  # runner convention check
        assert np.array_equal(E2, E) and int(np.argmin(E2)) == int(meta["ground_bitstring"], 2)
        c += 1
    return n, k
if __name__ == "__main__":
    with Pool(2) as p:
        for n, k in p.imap_unordered(gen, tuple(int(x) for x in sys.argv[1:]) or (8, 10, 12, 14, 16, 18, 20)): print(n, "used k<", k, flush=True)
