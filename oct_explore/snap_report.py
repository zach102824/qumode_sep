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
Ls = sorted(int(k) for k in d.get("train", {}))


def get_test(blob, n, L):
    block = blob.get("test", {})
    nb = block.get(str(n))
    if nb is None:
        nb = block.get(n)
    if not isinstance(nb, dict):
        return None
    t = nb.get(str(L))
    if t is None:
        t = nb.get(L)
    return t


show = [n for n in (8, 10, 12, 14, 20) if get_test(d, n, Ls[0]) is not None or str(n) in d.get("test", {})]
# keep order, only sizes that have any L result
show = [n for n in (8, 10, 12, 14, 20) if any(get_test(d, n, L) for L in Ls)]
if not show:
    show = sorted(int(k) for k in d.get("test", {}))

lines = []
lines.append(f"\n## {tag} ({time.strftime('%a %H:%M CST')})  [kind={args.get('kind')} snap={args.get('snap')} layout={args.get('layout')} nt2={args.get('nt2')}]\n")
hdr = "| L | " + " | ".join(f"n={n}" for n in show) + " |"
sep = "|" + "|".join(["---"] * (len(show) + 1)) + "|"
lines.append(hdr); lines.append(sep)
for L in Ls:
    cells = []
    for n in show:
        t = get_test(d, n, L)
        cells.append(f"{t['median_inv_p']:.1f}/{t['B90_inv_p']:.1f}" if t else "—")
    lines.append(f"| {tag} {L} | " + " | ".join(cells) + " |")
    qcells, gcells = [], []
    for n in show:
        qt = get_test(qa, n, L)
        qcells.append(f"{qt['median_inv_p']:.1f}/{qt['B90_inv_p']:.1f}" if qt else "—")
        if gq:
            gt = get_test(gq, n, L)
            gcells.append(f"{gt['median_inv_p']:.1f}/{gt['B90_inv_p']:.1f}" if gt else "—")
    lines.append(f"| QAOA {L} | " + " | ".join(qcells) + " |")
    if gq:
        lines.append(f"| G_qfe {L} | " + " | ".join(gcells) + " |")
for L in Ls:
    sm = d["train"][str(L)].get("snap_mag")
    if sm is not None:
        lines.append(f"\nL={L} snap_mag={sm:.4f} (mean |s0−s1| for csnap, else mean |s|).")
lines.append("")
text = "\n".join(lines)
open("LOG.md", "a").write(text)
print(text)
