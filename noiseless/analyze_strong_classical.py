import json, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "logs/strong"
NS = [8, 10, 12, 14, 16, 18, 20]; HB = dict(zip(NS, [152, 190, 228, 266, 400, 558, 740]))
SA0 = dict(zip(NS, [462, 1005, 2127, 3735, 5234, 9909, 18236])); HP = dict(zip(NS, [0.987, 0.986, 0.990, 0.961, 0.972, 0.963, 0.957]))
def b90(v): v = sorted(v); return v[int(np.ceil(0.9 * len(v))) - 1]
def fits(y):
    x = np.array(NS, float); y = np.array(y, float); ly = np.log(y); r = {}
    for name, X in (("exp", x), ("pow", np.log(x))):
        k, c = np.polyfit(X, ly, 1); pred = k * X + c
        r2l = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
        r2 = 1 - ((y - np.exp(pred)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r[name] = (np.exp(c), np.exp(k) if name == "exp" else k, r2l, r2)
    return r
rows = [json.loads(l) for l in (OUT / "eval.jsonl").read_text().splitlines()]
best = json.load(open(OUT / "best_cfg.json"))
L = []; summ = {}
names = {"sa": "Tuned SA", "walksat": "Weighted WalkSAT", "satlike": "SATLike-style"}
for arm in ("sa", "walksat", "satlike"):
    L.append(f"\n### {names[arm]}\n")
    L.append("| n | tuned config | succ @ hybrid budget (strict) | B90 strict | B90/hybrid | median strict | B90 flips | B90 full-neighbourhood (1+n*(flips+1)) | succ @ hybrid budget (full-nbhd) | B90 full/hybrid | unsolved@cap |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    bs, bf = [], []
    for n in NS:
        r = next(x for x in rows if x["arm"] == arm and x["n"] == n); cap = r["cap"]; big = 10 ** 18
        h = [x[2] if x[2] > 0 else big for x in r["hits"]]; f = [x[3] for x in r["hits"]]
        full = [1 + n * (x[3] + 1) if x[2] > 0 else big for x in r["hits"]]
        if arm == "sa": full = h
        s = np.mean([v <= HB[n] for v in h]); sf = np.mean([v <= HB[n] for v in full])
        B = b90(h); BF = b90(full); Bfl = b90([x[3] if x[2] > 0 else big for x in r["hits"]])
        bs.append(B); bf.append(BF); summ[(arm, n)] = (s, B, sf, BF)
        L.append(f"| {n} | {json.dumps(best[f'{arm}|{n}']['cfg'])} | {s:.3f} | {B} | {B/HB[n]:.2f} | {int(np.median(h))} | {Bfl} | {BF} | {sf:.3f} | {BF/HB[n]:.2f} | {sum(v==big for v in h)} |")
    for lab, y in (("strict", bs), ("full-nbhd", bf)):
        ft = fits(y); e, p = ft["exp"], ft["pow"]
        L.append(f"\nB90 {lab}: exp a={e[0]:.3f} b={e[1]:.4f} R2(log)={e[2]:.4f} R2(lin)={e[3]:.4f}; power a={p[0]:.4f} k={p[1]:.3f} R2(log)={p[2]:.4f} R2(lin)={p[3]:.4f}")
ft = fits([HB[n] for n in NS]); L.append(f"\nHybrid lookups: exp b={ft['exp'][1]:.4f} R2(log)={ft['exp'][2]:.4f}; power k={ft['pow'][1]:.3f} R2(log)={ft['pow'][2]:.4f}")
L.append("\n### Head-to-head (B90 strict / full-nbhd vs hybrid lookups; hybrid succ 10/10, mean p(GS) listed)\n")
L.append("| n | hybrid lookups | hybrid p(GS) | untuned SA B90 | tuned SA B90 | WalkSAT B90 strict/full | SATLike B90 strict/full |")
L.append("|---|---|---|---|---|---|---|")
for n in NS:
    L.append(f"| {n} | {HB[n]} | {HP[n]} | {SA0[n]} | {summ[('sa',n)][1]} | {summ[('walksat',n)][1]} / {summ[('walksat',n)][3]} | {summ[('satlike',n)][1]} / {summ[('satlike',n)][3]} |")
print("\n".join(L))
