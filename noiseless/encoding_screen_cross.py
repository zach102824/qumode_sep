#!/usr/bin/env python3
"""Cross-Hamiltonian transfer analysis of the full encoding screen.

For every finished Hamiltonian (summary with all 20160 classes x inits) builds the per-class mean
p(GS) matrix M[h, class] and reports: identity rank / mean p(GS) vs best class, the rank of each
H's best class on every other finished H, Spearman rank correlation between Hs, and the class with
the best average mean p(GS) over finished Hs vs identity.  Per-class matrix cached to
<dir>/.cross_cache.npz keyed by summary (size, mtime_ns).

  PYTHONPATH=/workspace/qumode_sep /workspace/venv-qumode/bin/python noiseless/encoding_screen_cross.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
ROOT = _REPO / "noiseless" / "results" / "encoding_screen"
N_CLASSES = 20160


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """1-based rank, 1 = highest."""
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v), dtype=int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def load_matrix(d: Path, n_ham: int, inits: int):
    cache_p = d / ".cross_cache.npz"
    cache = {}
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=False)
        keys = list(z["keys"])
        for i, k in enumerate(keys):
            cache[str(k)] = z["M"][i]
    hs, rows, keys, slots = [], [], [], {}
    for h in range(n_ham):
        sp = d / f"four_sat_{h:03d}_summary.json"
        if not sp.exists():
            continue
        st = sp.stat()
        key = f"{h}:{st.st_size}:{st.st_mtime_ns}"
        if key in cache:
            row = cache[key]
        else:
            s = json.loads(sp.read_text())
            if s.get("n_trials", 0) < N_CLASSES * inits:
                continue
            row = np.full(N_CLASSES, np.nan)
            for r in s["ranking"]:
                row[int(r["class_idx"])] = float(r["mean_p_gs"])
                if int(r["class_idx"]) not in slots:
                    slots[int(r["class_idx"])] = r.get("slots")
        hs.append(h)
        rows.append(row)
        keys.append(key)
    M = np.array(rows) if rows else np.zeros((0, N_CLASSES))
    if rows:
        np.savez(cache_p, keys=np.array(keys), M=M)
    return hs, M


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="full")
    p.add_argument("--n-ham", type=int, default=20)
    p.add_argument("--inits", type=int, default=3)
    args = p.parse_args(argv)
    d = ROOT / args.tag
    hs, M = load_matrix(d, args.n_ham, args.inits)
    if not hs:
        print("no finished Hamiltonians")
        return 0
    R = np.array([_rank_desc(m) for m in M])
    best = M.argmax(axis=1)
    print(f"finished Hamiltonians: {hs}\n")
    print(f"{'H':>3} {'id rank':>8} {'id p':>7} {'best cls':>9} {'best p':>7} {'#<0.01':>7}  rank of this H's best class on other Hs")
    for i, h in enumerate(hs):
        other = ", ".join(f"H{hs[j]}:{R[j, best[i]]}" for j in range(len(hs)) if j != i)
        near = int((M[i] >= M[i].max() - 0.01).sum())
        print(f"{h:>3} {R[i, 0]:>8} {M[i, 0]:>7.4f} {best[i]:>9} {M[i, best[i]]:>7.4f} {near:>7}  {other}")
    if len(hs) >= 2:
        print("\nSpearman rank correlation of per-class mean p(GS) between Hs:")
        C = np.corrcoef(R)
        print("     " + " ".join(f"{'H' + str(h):>6}" for h in hs))
        for i, h in enumerate(hs):
            print(f"{'H' + str(h):>4} " + " ".join(f"{C[i, j]:>6.3f}" for j in range(len(hs))))
    avg = M.mean(axis=0)
    ra = _rank_desc(avg)
    bc = int(avg.argmax())
    print(f"\nclass with best average mean p(GS) over {len(hs)} Hs: {bc}  avg {avg[bc]:.4f}  per-H "
          + ", ".join(f"H{h}:{M[i, bc]:.3f} (rank {R[i, bc]})" for i, h in enumerate(hs)))
    print(f"identity (class 0): avg {avg[0]:.4f}, rank {ra[0]}/{N_CLASSES} on the average; median avg {np.median(avg):.4f}")
    print(f"mean of per-H best: {M.max(axis=1).mean():.4f}  (per-instance oracle choice)")
    worst_rank = R.max(axis=0)
    rb = int(worst_rank.argmin())
    print(f"most robust class (best worst-case rank): {rb}  worst rank {worst_rank[rb]}, avg {avg[rb]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
