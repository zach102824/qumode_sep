#!/usr/bin/env python3
"""Rebuild RESULTS.md / PARETO.md from scale1000_*_full.json + BEAT markers."""
from __future__ import annotations

import json
import time
from pathlib import Path

from Nov.scripts.nov1st_final_train import SEALED_REF

EASY = Path(__file__).resolve().parents[1] / "runs" / "snap_easy"


def load_full(path: Path):
    d = json.loads(path.read_text())
    # normalize scale dict
    scale = d.get("scale") or {}
    # sometimes nested under rows only
    out = {"tag": d.get("tag") or path.stem.replace("scale1000_", ""), "path": str(path)}
    # params heuristic
    tag = out["tag"]
    if "few_mM2" in tag or "fewm2" in tag or "kerr" in tag:
        out["params"] = 2
    elif "gauss_plus1" in tag or "poly3" in tag or "shell" in tag:
        out["params"] = 4
    elif "gauss" in tag or "poly2" in tag or "few_mM3" in tag:
        out["params"] = 3
    else:
        out["params"] = d.get("params")
    cells = {}
    if scale:
        for n in (9, 12, 15, 18):
            m = scale.get(str(n)) or scale.get(n)
            if not m:
                continue
            med = m.get("median_inv_p") or m.get("Med")
            b90 = m.get("B90_inv_p") or m.get("B90")
            if med is None:
                continue
            ref = SEALED_REF[n]
            cells[n] = (med, b90, med / ref["Med"], b90 / ref["B90"])
    else:
        for r in d.get("rows") or []:
            n = int(r["n"])
            cells[n] = (r["Med"], r["B90"], r["Med_x"], r["B90_x"])
    out["cells"] = cells
    if len(cells) == 4:
        out["beat"] = all(cells[n][2] <= 1.0 and cells[n][3] <= 1.0 for n in (9, 12, 15, 18))
    else:
        out["beat"] = False
    return out


def main():
    fulls = sorted(EASY.glob("scale1000_*_full.json"))
    fulls += sorted((EASY / "gauss_polish").glob("scale1000_*_full.json"))
    fulls += sorted((EASY / "kerr_polish").glob("scale1000_*_full.json"))
    rows = []
    seen = set()
    for p in fulls:
        r = load_full(p)
        if r["tag"] in seen:
            continue
        if len(r["cells"]) < 4:
            continue
        seen.add(r["tag"])
        rows.append(r)
    rows.sort(key=lambda r: (r.get("params") or 99, max(r["cells"][n][2] for n in r["cells"])))

    beat = EASY / "BEAT_SEALED.json"
    beat2 = EASY / "BEAT_SEALED_2PARAM.json"
    notes = []
    if beat.exists():
        bj = json.loads(beat.read_text())
        notes.append(f"## ★ BEAT_SEALED — `{bj.get('tag')}` (see BEAT_SEALED.json)\n")
    if beat2.exists():
        bj = json.loads(beat2.read_text())
        notes.append(f"## ★ BEAT_SEALED ≤2-param — `{bj.get('tag')}`\n")

    lines = [
        "# SNAP easiness — RESULTS",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}_",
        "",
        "Motif locked: Nov1st-final. Train n∈{6,9} max_train=200. Eval scale1000 B=1000.",
        "",
        f"Sealed: n9 {SEALED_REF[9]['Med']}/{SEALED_REF[9]['B90']}, "
        f"n12 {SEALED_REF[12]['Med']}/{SEALED_REF[12]['B90']}, "
        f"n15 {SEALED_REF[15]['Med']}/{SEALED_REF[15]['B90']}, "
        f"n18 {SEALED_REF[18]['Med']}/{SEALED_REF[18]['B90']}.",
        "",
        *notes,
        "## Full B=1000 finalists",
        "",
        "| tag | params | n9 | n12 | n15 | n18 | beat? |",
        "|-----|--------|----|-----|-----|-----|-------|",
    ]
    for r in rows:
        def fmt(n):
            med, b90, mx, bx = r["cells"][n]
            return f"{med:.3f}/{b90:.3f} ({mx:.3f}×/{bx:.3f}×)"
        lines.append(
            f"| `{r['tag']}` | {r.get('params')} | {fmt(9)} | {fmt(12)} | {fmt(15)} | {fmt(18)} | "
            f"{'YES' if r['beat'] else 'NO'} |"
        )
    lines += [
        "",
        "## Most-restricted survivors",
        "",
        "- **gauss CVaR polish** (3 params/L): dual-beats sealed (BEAT_SEALED.json)",
        "- **few_m M=2 / kerr** (2 params/L): see table; polish in `kerr_polish/`",
        "",
        "Stop file: `Nov/runs/STOP_SNAP_EASY`.",
        "",
    ]
    (EASY / "RESULTS.md").write_text("\n".join(lines) + "\n")

    # PARETO: pick best by params then max Med×
    pareto = [
        "# SNAP-easy pareto",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}_",
        "",
    ]
    if beat.exists():
        bj = json.loads(beat.read_text())
        pareto += [
            f"## Clear pareto + sealed dual-beat: `{bj.get('tag')}`",
            "",
            "See BEAT_SEALED.json / RESULTS.md.",
            "",
        ]
    if beat2.exists():
        bj = json.loads(beat2.read_text())
        pareto += [f"## ≤2-param dual-beat: `{bj.get('tag')}`", ""]
    else:
        pareto += [
            "## ≤2-param (kerr / few_m M=2)",
            "",
            "No dual-beat yet. Kerr B90 gap + few_m polish running in `kerr_polish/`.",
            "",
        ]
    # best 2-param row
    twos = [r for r in rows if r.get("params") == 2]
    if twos:
        # bottleneck = max(Med×, B90×) over sealed n (B90 is the usual killer)
        best = min(
            twos,
            key=lambda r: max(max(r["cells"][n][2], r["cells"][n][3]) for n in (9, 12, 15, 18)),
        )
        pareto += [f"Best 2-param full table so far: `{best['tag']}`", ""]
        pareto += ["| n | Med / B90 | ×Med / ×B90 |", "|---|-----------|-------------|"]
        for n in (9, 12, 15, 18):
            med, b90, mx, bx = best["cells"][n]
            pareto.append(f"| {n} | {med:.3f} / {b90:.3f} | {mx:.3f}× / {bx:.3f}× |")
        pareto.append("")
    (EASY / "PARETO.md").write_text("\n".join(pareto) + "\n")
    print("Wrote", EASY / "RESULTS.md", "and", EASY / "PARETO.md", "n_full=", len(rows))


if __name__ == "__main__":
    main()
