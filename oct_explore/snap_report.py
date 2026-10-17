"""Append SNAP_costgx result table to LOG.md."""
import json, sys, os, time
tag = sys.argv[1]
path = f"results/{tag}.json"
if not os.path.exists(path):
    print(f"missing {path}", file=sys.stderr); sys.exit(1)
d = json.load(open(path))
qa = json.load(open("results/QAOA_raw.json"))
gq = json.load(open("results/G_qfe.json")) if os.path.exists("results/G_qfe.json") else None
args = d.get("args_last") or d.get("args") or {}
Ls = [int(k) for k in d.get("train", {})]
ns = sorted(int(k) for k in d.get("test", {}))
lines = []
lines.append(f"\n## {tag} ({time.strftime('%a %H:%M CST')})  [kind={args.get('kind')} snap={args.get('snap')} layout={args.get('layout')} nt2={args.get('nt2')}]\n")
hdr = "| L | " + " | ".join(f"n={n}" for n in ns if n != 8) + " |"
# always show n=10,12,14,20 if present
show = [n for n in (10, 12, 14, 20) if str(n) in d["test"]]
hdr = "| L | " + " | ".join(f"n={n}" for n in show) + " |"
sep = "|" + "|".join(["---"] * (len(show) + 1)) + "|"
lines.append(hdr); lines.append(sep)
for L in sorted(Ls):
    cells = []
    for n in show:
        t = d["test"].get(str(n), {}).get(str(L))
        if not t: cells.append("—"); continue
        cells.append(f"{t['median_inv_p']:.1f}/{t['B90_inv_p']:.1f}")
    lines.append(f"| {tag} {L} | " + " | ".join(cells) + " |")
    # QAOA / G_qfe refs
    qcells, gcells = [], []
    for n in show:
        qt = qa.get("test", {}).get(str(n), {}).get(str(L)) or qa.get("test", {}).get(n, {}).get(L)
        if isinstance(qa.get("test", {}).get(str(n)), dict):
            qt = qa["test"][str(n)].get(str(L)) or qa["test"][str(n)].get(L)
        # QAOA_raw uses int keys sometimes
        qt = None
        block = qa["test"].get(str(n)) or qa["test"].get(n)
        if block:
            qt = block.get(str(L)) or block.get(L)
        qcells.append(f"{qt['median_inv_p']:.1f}/{qt['B90_inv_p']:.1f}" if qt else "—")
        if gq:
            gb = gq["test"].get(str(n), {}).get(str(L))
            gcells.append(f"{gb['median_inv_p']:.1f}/{gb['B90_inv_p']:.1f}" if gb else "—")
    lines.append(f"| QAOA {L} | " + " | ".join(qcells) + " |")
    if gq:
        lines.append(f"| G_qfe {L} | " + " | ".join(gcells) + " |")
# snap mag
for L in sorted(Ls):
    sm = d["train"][str(L)].get("snap_mag")
    if sm is not None:
        lines.append(f"\nL={L} snap_mag={sm:.4f} (mean |s0-s1| for csnap, else mean |s|).")
lines.append("")
open("LOG.md", "a").write("\n".join(lines))
print("\n".join(lines))
