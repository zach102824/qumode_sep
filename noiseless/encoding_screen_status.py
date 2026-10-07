#!/usr/bin/env python3
"""Status of the full encoding screen (noiseless/run_encoding_screen_full.sh).

Prints trials done per Hamiltonian / total, overall %, recent throughput and ETA (CST),
whether the driver is alive, and for each completed Hamiltonian: identity rank + mean p(GS),
best class + mean p(GS), spread of per-class mean p(GS) (min / median / max).

  PYTHONPATH=/workspace/qumode_sep /workspace/venv-qumode/bin/python noiseless/encoding_screen_status.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
ROOT = _REPO / "noiseless" / "results" / "encoding_screen"
N_CLASSES = 20160
CST = timezone(timedelta(hours=8))


def _count_lines(p: Path) -> int:
    n = 0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def _tail_times(p: Path, nbytes: int = 4 << 20) -> list[float]:
    size = p.stat().st_size
    with p.open("rb") as fh:
        fh.seek(max(0, size - nbytes))
        data = fh.read().split(b"\n")
    out = []
    for line in data[1:] if size > nbytes else data:
        try:
            t = json.loads(line).get("t_done")
        except (json.JSONDecodeError, ValueError, AttributeError):
            continue
        if t is not None:
            out.append(float(t))
    return out


def _headline(sp: Path, cache: dict) -> dict | None:
    """Small headline of a (13 MB) summary JSON, cached by (size, mtime_ns) in .status_cache.json."""
    try:
        st = sp.stat()
    except OSError:
        return None
    key = f"{st.st_size}:{st.st_mtime_ns}"
    hit = cache.get(sp.name)
    if hit and hit.get("key") == key:
        return hit["head"]
    try:
        s = json.loads(sp.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    head = {k: s.get(k) for k in ("n_trials", "n_assignments", "identity", "spread_mean_p_gs")}
    head["best"] = s["ranking"][0] if s.get("ranking") else None
    cache[sp.name] = {"key": key, "head": head}
    return head


def _alive(pidfile: Path) -> tuple[bool, int | None]:
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except OSError:
        return False, pid


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="full")
    p.add_argument("--n-ham", type=int, default=20)
    p.add_argument("--inits", type=int, default=3)
    p.add_argument("--window", type=float, default=600.0, help="throughput window (s)")
    args = p.parse_args(argv)
    d = ROOT / args.tag
    per_h = N_CLASSES * args.inits
    total = per_h * args.n_ham
    alive, pid = _alive(d / "driver.pid")
    now = time.time()
    print(f"encoding screen '{args.tag}'  {datetime.now(CST):%Y-%m-%d %H:%M:%S} CST  dir {d}")
    print(f"driver: {'ALIVE' if alive else 'NOT running'} (pid {pid})")
    done_total, times = 0, []
    rows = []
    cache_path = d / ".status_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    for h in range(args.n_ham):
        f = d / f"four_sat_{h:03d}.jsonl"
        sp = d / f"four_sat_{h:03d}_summary.json"
        summ = _headline(sp, cache) if sp.exists() else None
        if summ and (summ.get("n_trials") or 0) >= per_h:
            n = int(summ["n_trials"])  # finished: no need to scan the 30 MB JSONL
        else:
            summ = None
            n = _count_lines(f) if f.exists() else 0
        done_total += min(n, per_h)
        if f.exists() and now - f.stat().st_mtime < args.window + 60:
            times += _tail_times(f)
        rows.append((h, n, summ))
    try:
        cache_path.write_text(json.dumps(cache))
    except OSError:
        pass
    print(f"\n{'H':>4} {'trials':>13} {'%':>6}  identity rank / mean p   best class (mean p)   spread min / median / max")
    for h, n, s in rows:
        line = f"{h:>4} {n:>6}/{per_h:<6} {100 * n / per_h:>5.1f}%"
        if s:
            idn, best, spr = s["identity"], s["best"], s["spread_mean_p_gs"]
            line += (f"  {idn['rank']:>5}/{s['n_assignments']} {idn['mean_p_gs']:.4f}"
                     f"   {best['class_idx']:>5} ({best['mean_p_gs']:.4f})"
                     f"      {spr['min']:.4f} / {spr['median']:.4f} / {spr['max']:.4f}")
        print(line)
    recent = [t for t in times if now - t <= args.window]
    rate = len(recent) / (now - min(recent)) if len(recent) >= 2 else 0.0
    left = total - done_total
    print(f"\ntotal {done_total}/{total} ({100 * done_total / total:.2f}%)  remaining {left}")
    if rate > 0:
        eta = datetime.now(CST) + timedelta(seconds=left / rate)
        print(f"throughput {rate:.2f} trials/s (last {len(recent)} trials, {args.window:.0f}s window)  "
              f"ETA {eta:%Y-%m-%d %H:%M} CST ({left / rate / 3600:.1f} h)")
    else:
        print("throughput: no trials finished in the window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
