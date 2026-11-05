#!/usr/bin/env python3
"""Hold-only BFGS table: Fourier (2L) vs SNAP (5L). No freeze."""
import json, sys
from pathlib import Path

def load(d, method):
    p = Path(d) / "train_results.json"
    rows = json.loads(p.read_text())
    out = []
    for r in rows:
        r = dict(r)
        r["method"] = method
        r["n_params"] = int(r.get("n_params", (2 if method == "fourier" else 5) * int(r["L"])))
        out.append(r)
    return out

def champ_hold(rows):
    # best_hold_score is mean of hold medians across sizes in that mode
    champ = {}
    for r in rows:
        key = (r["method"], int(r["L"]), r["mode"])
        h = float(r["best_hold_score"])
        prev = champ.get(key)
        if prev is None or h > prev[0]:
            champ[key] = (h, r)
    return champ

def main():
    out = Path(sys.argv[1])
    fdir, sdir = sys.argv[2], sys.argv[3]
    rows = load(fdir, "fourier") + load(sdir, "snap")
    champ = champ_hold(rows)
    # also per-n hold med from champ row
    lines = [
        "# Γ=0.5 same-L BFGS hold — Fourier (2L) vs SNAP (5L)",
        "",
        "Goal: more SNAP params edge at same L (no freeze).",
        "",
        "| L | mode | n_params F/S | Fourier best_hold | SNAP best_hold | Δ | n6 hold F/S | n9 hold F/S |",
        "|---|------|--------------|-------------------|----------------|---|-------------|-------------|",
    ]
    Ls = sorted({k[1] for k in champ})
    modes = sorted({k[2] for k in champ})
    for L in Ls:
        for mode in modes:
            fh = champ.get(("fourier", L, mode))
            sh = champ.get(("snap", L, mode))
            if not fh and not sh:
                continue
            npF = 2 * L
            npS = 5 * L
            fv = f"{fh[0]:.4f}" if fh else "—"
            sv = f"{sh[0]:.4f}" if sh else "—"
            d = f"{sh[0]-fh[0]:+.4f}" if fh and sh else "—"
            def nmed(h, n):
                if not h: return "—"
                hold = h[1].get("hold") or {}
                cell = hold.get(str(n)) or hold.get(n)
                return f"{cell['median_F']:.4f}" if cell else "—"
            n6 = f"{nmed(fh,6)}/{nmed(sh,6)}"
            n9 = f"{nmed(fh,9)}/{nmed(sh,9)}"
            lines.append(f"| {L} | {mode} | {npF}/{npS} | {fv} | {sv} | {d} | {n6} | {n9} |")
    text = "\n".join(lines) + "\n"
    (out / "RESULTS.md").write_text(text)
    payload = {f"{a}|L{b}|{c}": {"hold": h, "seed": r["seed"], "n_params": r["n_params"],
                                  "hold_detail": r.get("hold")} for (a,b,c),(h,r) in champ.items()}
    (out / "hold_champs.json").write_text(json.dumps(payload, indent=2))
    print(text)

if __name__ == "__main__":
    main()
