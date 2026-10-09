#!/usr/bin/env python3
"""Step-1 diagnosis: why does relayout cost more at n=10 than n=8 (dense four_sat_scaling)?
Reads per-trial jsonl in relayout_runs/kbit, no new simulation."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from noiseless.run_relayout_kbit import RUN_ROOT, ham_paths, energies_for, load_records

def hd(a, b):
    return sum(x != y for x, y in zip(a, b))

def is_local_min(E, v, n):
    return all(E[v ^ (1 << j)] > E[v] for j in range(n))

def analyse(tag, trials=None):
    recs = load_records(RUN_ROOT / f"{tag}.jsonl", trials)
    n = recs[0]["n"]
    paths = ham_paths(n, "scaling")
    Es = {i: energies_for(str(p), n, "scaling") for i, p in enumerate(paths)}
    R = max(len(x["rounds"]) for x in recs) - 1
    print(f"\n=== {tag}  trials={len(recs)}  success={np.mean([x['success'] for x in recs]):.3f} "
          f"meanp={np.mean([x['p_gs'] for x in recs]):.3f} leak={np.mean([x['leakage'] for x in recs]):.4f} "
          f"evals={np.mean([x['nfev'] for x in recs]):.0f}")
    print("rnd | guessIN_hit | round_p | p|guessOK | p|guessBad | leak|OK | leak|Bad | maxb|OK | maxb|Bad | guessOUT_hit | bsf_change | wrongHD")
    for j in range(R + 1):
        rows = []
        for x in recs:
            rs = x["rounds"]
            q = rs[j]
            E, gs = Es[x["inst"]]
            gin = None
            if j > 0:
                gin = rs[j - 1]["next_guess_is_ground"]
            rows.append((gin, q["p_gs"], q["leakage"], q["max_abs_beta"], q["next_guess_is_ground"],
                         j > 0 and q["bsf_round"] != rs[j - 1]["bsf_round"],
                         None if (q["next_guess"] is None or q["next_guess_is_ground"]) else hd(q["next_guess"], gs)))
        ok = [r for r in rows if r[0]]
        bad = [r for r in rows if r[0] is False]
        f = lambda L, i: f"{np.mean([r[i] for r in L]):.3f}" if L else "  -  "
        hds = Counter(r[6] for r in rows if r[6] is not None)
        print(f"{j:3d} | {np.mean([r[0] for r in rows if r[0] is not None]) if j else float('nan'):.3f} | "
              f"{np.mean([r[1] for r in rows]):.3f} | {f(ok,1)} | {f(bad,1)} | {f(ok,2)} | {f(bad,2)} | {f(ok,3)} | {f(bad,3)} | "
              f"{np.mean([r[4] for r in rows]):.3f} | {np.mean([r[5] for r in rows]):.3f} | {dict(sorted(hds.items()))}")
    # round 0 detail
    r0 = [x["rounds"][0] for x in recs]
    print(f"round0: raw hit {np.mean([q['success'] for q in r0]):.3f}  polished hit {np.mean([q['polished_is_ground'] for q in r0]):.3f} "
          f"p {np.mean([q['p_gs'] for q in r0]):.3f} leak {np.mean([q['leakage'] for q in r0]):.3f} "
          f"maxb median {np.median([q['max_abs_beta'] for q in r0]):.2f}  MLleaked {np.mean([q['most_likely_bitstring']=='LEAKED' for q in r0]):.3f}")
    # wrong guesses: are they local minima (radius-1 fixup can never escape) ?
    lm = []; Ewrong = []; stuck = []
    for x in recs:
        E, gs = Es[x["inst"]]
        guesses = [q["next_guess"] for q in x["rounds"]]
        wrong = [g for g in guesses if g is not None and g != gs]
        for g in wrong:
            v = int(g, 2); lm.append(is_local_min(E, v, n)); Ewrong.append(E[v])
        # number of rounds whose input guess equalled the previous input guess (repeat) and was wrong
        ins = [(q["xa"], q["xb"]) for q in x["rounds"][1:]]
        stuck.append(sum(1 for a, b in zip(ins, ins[1:]) if a == b))
    print(f"wrong guesses: n={len(lm)} local-min frac {np.mean(lm) if lm else float('nan'):.3f}  E dist {dict(sorted(Counter(np.round(Ewrong,3)).items()))}")
    fails = [x for x in recs if not x["success"]]
    def rounds_ok(x):
        return [q for q in x["rounds"][1:] if True]
    # final state given that the final guess was ground
    allok = [x for x in recs if x["rounds"][-2]["next_guess_is_ground"]] if R >= 1 else []
    print(f"final guess correct: {np.mean([x['rounds'][-1]['next_guess_is_ground'] for x in recs]):.3f};  "
          f"selected round p (given any round guessIn correct) "
          f"{np.mean([x['p_gs'] for x in recs if any(q['next_guess_is_ground'] for q in x['rounds'][:-1])]):.3f}")
    # per-H
    ph = {}
    for x in recs: ph.setdefault(x["inst"], []).append(x)
    print("per-H: inst | succ | meanp | r0 polished hit | local minima (E>0) count | #E=1 states")
    for i, v in sorted(ph.items()):
        E, gs = Es[i]
        nlm = sum(is_local_min(E, u, n) for u in range(1 << n)) - 1
        print(f"  {i:2d} | {np.mean([y['success'] for y in v]):.2f} | {np.mean([y['p_gs'] for y in v]):.3f} | "
              f"{np.mean([y['rounds'][0]['polished_is_ground'] for y in v]):.2f} | {nlm} | {int(np.sum(np.isclose(E,1)))}")
    # among relabel rounds with correct input guess: p distribution, and with correct guess the 'fun' at vacuum etc.
    okr = [q for x in recs for j, q in enumerate(x["rounds"]) if j > 0 and x["rounds"][j-1]["next_guess_is_ground"]]
    if okr:
        ps = np.array([q["p_gs"] for q in okr])
        print(f"relabel rounds w/ correct guess: N={len(ps)} mean p {ps.mean():.3f} median {np.median(ps):.3f} "
              f"frac p>0.95 {np.mean(ps>0.95):.3f} frac p<0.5 {np.mean(ps<0.5):.3f} mean leak {np.mean([q['leakage'] for q in okr]):.4f} "
              f"mean maxb {np.mean([q['max_abs_beta'] for q in okr]):.2f}")
    badr = [q for x in recs for j, q in enumerate(x["rounds"]) if j > 0 and not x["rounds"][j-1]["next_guess_is_ground"]]
    if badr:
        ps = np.array([q["p_gs"] for q in badr])
        print(f"relabel rounds w/ wrong guess: N={len(ps)} mean p {ps.mean():.3f}  frac success {np.mean([q['success'] for q in badr]):.3f} "
              f"frac polished hit {np.mean([q['polished_is_ground'] for q in badr]):.3f} frac stay-at-vacuum(ML==guess) "
              f"{np.mean([q['most_likely_bitstring']==None for q in badr]):.3f}")
    return recs

if __name__ == "__main__":
    for t in sys.argv[1:]:
        analyse(t)
