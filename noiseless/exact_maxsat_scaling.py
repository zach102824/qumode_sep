"""Exact MaxSAT (PySAT RC2 / FM / LSU) cost scaling on Hamiltonians/wmaxsat instances.
Clauses map directly to WCNF soft clauses (literal +v if sign>0 else -v; integer weights 1..5; no scaling).
Bit order: gs string char i-1 = x_i (as in run_relayout_kbit.wmaxsat_energies)."""
import json, random, sys, time, statistics, os
from pathlib import Path
from pysat.formula import WCNF
from pysat.solvers import Solver
from pysat.examples.rc2 import RC2
from pysat.examples.fm import FM
from pysat.examples.lsu import LSU
REPO = Path(__file__).resolve().parents[1]
STOP = REPO / "logs/3cav/exact_maxsat/STOP"
CNT = {"calls": 0, "stats": {}}
for meth in ("solve", "solve_limited"):
    orig = getattr(Solver, meth)
    def wrap(self, *a, _o=orig, **k):
        CNT["calls"] += 1
        r = _o(self, *a, **k)
        try: CNT["stats"][id(self)] = dict(self.accum_stats() or {})
        except Exception: pass
        return r
    setattr(Solver, meth, wrap)

def wcnf_of(d):
    w = WCNF()
    for c in d["clauses"]:
        w.append([int(v) if int(s) > 0 else -int(v) for v, s in zip(c["vars"], c["signs"])], weight=int(c["w"]))
    return w

def gen(n, k):  # same clause generator as Hamiltonians/wmaxsat_gen.py, no uniqueness filter (no enumeration)
    r = random.Random(100000 * n + 50000 + k); m = int(round(9.93 * n)); cl = []
    for _ in range(m):
        v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; w = r.randint(1, 5)
        cl.append({"vars": [i + 1 for i in v], "signs": s, "w": w})
    return {"n": n, "m": m, "clauses": cl}

def cost(d, bits):
    return sum(c["w"] for c in d["clauses"] if all(bits[v-1] == (0 if s > 0 else 1) for v, s in zip(c["vars"], c["signs"])))

def run(alg, w):
    CNT["calls"] = 0; CNT["stats"] = {}; cores = None
    t = time.perf_counter()
    if alg == "rc2":
        s = RC2(w, solver="g4"); ncore = [0]; og = s.get_core
        def gc(): r = og(); ncore[0] += bool(s.core); return r
        s.get_core = gc; m = s.compute(); c = s.cost; cores = ncore[0]; s.delete()
    elif alg == "fm":
        s = FM(w, solver="g4"); s.compute(); m = s.model; c = s.cost; s.delete()
    else:
        s = LSU(w, solver="g4"); s.solve(); m = s.model; c = s.cost; s.delete()
    dt = time.perf_counter() - t
    agg = {}
    for st in CNT["stats"].values():
        for k_, v in st.items(): agg[k_] = agg.get(k_, 0) + v
    return m, c, dt, CNT["calls"], cores, agg

def main():
    out = REPO / "logs/3cav/exact_maxsat/results.jsonl"; f = open(out, "a")
    jobs = [(n, p, None) for n in (8, 10, 12, 14, 16, 18, 20) for p in sorted((REPO/"Hamiltonians/wmaxsat").glob(f"default_n{n:02d}_*.json"), key=lambda q: int(q.stem.rsplit('_',1)[1]))]
    jobs += [(n, None, k) for n in (24, 28, 32, 40, 48, 56, 64) for k in range(10)]
    for n, p, k in jobs:
        if STOP.exists(): print("STOP"); return
        d = json.loads(p.read_text()) if p else gen(n, k); w = wcnf_of(d)
        for alg in ("rc2", "fm"):
            if alg != "rc2" and n > 32: continue
            times = []; rec = None
            for rep in range(5):
                m, c, dt, calls, cores, agg = run(alg, w); times.append(dt)
                rec = rec or dict(calls=calls, cores=cores, stats=agg, model=m, cost=c)
            bits = [1 if x > 0 else 0 for x in sorted(rec["model"], key=abs)[:n]]
            s = "".join(map(str, bits)); ok = None
            assert cost(d, bits) == rec["cost"]
            if p: ok = (s == d["gs"] and rec["cost"] == d["E_gs"])
            r = dict(n=n, inst=(int(p.stem.rsplit('_',1)[1]) if p else k), new=p is None, alg=alg, t_med=statistics.median(times),
                     times=times, calls=rec["calls"], cores=rec["cores"], cost=rec["cost"], sol=s, match_gs=ok, **{k_: rec["stats"].get(k_) for k_ in ("conflicts","decisions","propagations","restarts")})
            f.write(json.dumps(r) + "\n"); f.flush(); print(n, r["inst"], alg, r["t_med"], r["calls"], r["conflicts"], ok, flush=True)
            if n > 20 and r["t_med"] > 120: print("slow, stop", alg, n)
main()
