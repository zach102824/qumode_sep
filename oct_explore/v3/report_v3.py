"""Markdown table (median/B90 of 1/p) for a finished v3 tag, with QAOA_raw at equal L and the G_fe baseline."""
import json, sys, time
t = sys.argv[1]; r = json.load(open(f"results/v3/{t}.json")); L = r["args"]["L"]; te = r.get("test", {})
q = json.load(open("results/QAOA_raw.json"))["test"]; g = json.load(open("results/G_fe.json"))["test"]
ns = sorted({int(n) for role in te.values() for n in role if n != "id"})
fmt = lambda s: f"{s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}" if s else "-"
a = r["args"]; desc = " ".join(f"{k}={v}" for k, v in a.items() if k not in ("tag",) and v not in (None, "", [], False))
print(f"\n### {t} finished {time.strftime('%Y-%m-%d %H:%M')} CST  (L={L})\n\n`{desc}`\n")
print("| row | " + " | ".join(f"n={n}" for n in ns) + " |"); print("|---|" + "|".join(["---"] * len(ns)) + "|")
for role, d in te.items():
    extra = ""
    if "keep" in r: k = {x["id"]: x for x in r["keep"]}.get(d["id"]); extra = f" (train {k['loss']:.3f}, val {k.get('valmean', 0):.3f})" if k else ""
    if "runs" in r: k = r["runs"][d["id"]]; extra = f" (train {k['loss']:.3f}, val {k.get('valmean', 0):.3f})"
    print(f"| {t} {role} {d['id']}{extra} | " + " | ".join(fmt(d.get(str(n))) for n in ns) + " |")
print(f"| QAOA_raw L={L} | " + " | ".join(fmt(q.get(str(n), {}).get(str(L))) for n in ns) + " |")
if any(str(L) in g.get(str(n), {}) for n in ns): print(f"| G_fe L={L} | " + " | ".join(fmt(g.get(str(n), {}).get(str(L))) for n in ns) + " |")
w = te.get("winner", {}).get("20"); qq = q["20"].get(str(L))
if w and qq: print(f"\nn=20 winner vs QAOA L={L}: median {w['median_inv_p']:.1f} vs {qq['median_inv_p']:.1f}, B90 {w['B90_inv_p']:.1f} vs {qq['B90_inv_p']:.1f} -> " + ("**BEATS QAOA (both)**" if w['median_inv_p'] < qq['median_inv_p'] and w['B90_inv_p'] < qq['B90_inv_p'] else "no beat"))
