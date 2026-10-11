#!/usr/bin/env python3
"""Offline policy evaluation for explore_raw.py logs: hit rate of the min-energy pooled candidate after E runs for
top-K per run and fix-up radius rho (1 or 2); also mean distinct lookups. Usage: explore_raw_analyze.py TAG [TAG...]"""
import sys, json, itertools
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_REPO))
from noiseless.run_relayout_kbit import ham_paths, energies_for
RAW = _REPO / "noiseless" / "results" / "relayout_runs" / "raw"


def masks(n, rho):
    ms = [0] + [1 << b for b in range(n)]
    if rho >= 2:
        ms += [(1 << a) | (1 << b) for a, b in itertools.combinations(range(n), 2)]
    return np.array(ms, dtype=np.int64)


def analyze(tag, Ks=(1, 2, 4, 8), rhos=(1, 2), Es=None, verbose=True):
    recs = [json.loads(l) for l in (RAW / f"{tag}.jsonl").read_text().splitlines() if l.strip()]
    n = int(tag[1:3]); cache = {}
    M = {r: masks(n, r) for r in rhos}
    out = {}
    nr = len(recs[0]["runs"])
    for K in Ks:
        for rho in rhos:
            hit = np.zeros((len(recs), nr)); lk = np.zeros((len(recs), nr))
            for i, rec in enumerate(recs):
                E = cache.get(rec["inst"])
                if E is None:
                    E = cache[rec["inst"]] = energies_for(str(ham_paths(n, "scaling")[rec["inst"]]), n, "scaling")[0]
                best = (np.inf, -1); seen = set()
                for m, run in enumerate(rec["runs"]):
                    for v in run["raw"][:K]:
                        ball = v ^ M[rho]
                        e = E[ball]; j = int(np.argmin(e))
                        cand = (float(e[j]), int(ball[j]))
                        if cand < best: best = cand
                        seen.update(ball.tolist())
                    hit[i, m] = best[1] == rec["gs"]; lk[i, m] = len(seen)
            out[(K, rho)] = (hit.mean(0), lk.mean(0))
    if verbose:
        print(f"== {tag} ({len(recs)} trials, 2^n={1<<n})")
        rk = np.array([[r["rank"] for r in rec["runs"]] for rec in recs])
        print("  GS prob rank per run: P(rank0)=%.2f P(<2)=%.2f P(<4)=%.2f P(<8)=%.2f P(<16)=%.2f ; mean p(GS) run0 %.3f, mean over runs %.3f ; leak %.3f" % (
            (rk == 0).mean(), (rk < 2).mean(), (rk < 4).mean(), (rk < 8).mean(), (rk < 16).mean(),
            np.mean([rec["runs"][0]["pgs"] for rec in recs]), np.mean([[r["pgs"] for r in rec["runs"]] for rec in recs]),
            np.mean([[r["leak"] for r in rec["runs"]] for rec in recs])))
        for (K, rho), (h, l) in out.items():
            pick = [e for e in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24) if e <= nr]
            print(f"  K={K} rho={rho}: " + " ".join(f"E{e}={h[e-1]:.2f}({l[e-1]/(1<<n):.3f})" for e in pick))
    return out




def energy_ranked(tag, Krs=(16, 32, 64, 128), Ts=(1, 4), Es=(2, 3, 4, 5, 6, 8, 10, 12), rho=1):
    """Energy-ranked pool: raw top-Kr per run (1 lookup each) -> sort by classical energy -> fix up the T lowest."""
    recs = [json.loads(l) for l in (RAW / f"{tag}.jsonl").read_text().splitlines() if l.strip()]
    n = int(tag[1:3]); cache = {}
    Mk = masks(n, rho)
    nr = len(recs[0]["runs"]); Es = [e for e in Es if e <= nr]
    print(f"== energy-ranked pool {tag} rho={rho}: hit(lookups/2^n)")
    for Kr in Krs:
        if Kr > len(recs[0]["runs"][0]["raw"]):
            continue
        for T in Ts:
            row = []
            for Ee in Es:
                h = lk = 0
                for rec in recs:
                    E = cache.get(rec["inst"])
                    if E is None:
                        E = cache[rec["inst"]] = energies_for(str(ham_paths(n, "scaling")[rec["inst"]]), n, "scaling")[0]
                    raw = set()
                    for r in rec["runs"][:Ee]:
                        raw.update(r["raw"][:Kr])
                    srt = sorted(raw, key=lambda v: (E[v], v))[:T]
                    seen = set(raw); best = (np.inf, -1)
                    for v in srt:
                        ball = v ^ Mk; seen.update(ball.tolist())
                        e = E[ball]; j = int(np.argmin(e))
                        best = min(best, (float(e[j]), int(ball[j])))
                    h += best[1] == rec["gs"]; lk += len(seen)
                row.append(f"E{Ee}={h/len(recs):.2f}({lk/len(recs)/(1<<n):.3f})")
            print(f"  Kr={Kr} T={T}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        analyze(t)
        energy_ranked(t)
