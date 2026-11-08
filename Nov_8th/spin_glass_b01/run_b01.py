"""Historical B=0.1 confirmation of shared-gamma KM with the |J|-max packing.

No per-layer gamma. Ground states are not loaded.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_cri"))
import layout_gate  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "d147eb571151c819cff450bb0788db20c438570f98f45821414252d3c4b0b5a1"


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    if "KM_gl" in PREREG["family_order"] or "KM_gl" in PREREG["families"]:
        raise SystemExit("per-layer family is in the protocol")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = tuple(PREREG["family_order"])
    m4.CAP = int(PREREG["maxfun"])
    if "KM_gl" in m4.FAMS:
        raise SystemExit("per-layer family reached the runner")


def grad_and_layout():
    layout_gate.main()
    inst = m4.m3.instance(9, 44)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}, not the historical 0.1")
    part, _ = m4.best_packing(inst.J)
    ip = m4.with_packing(inst, part)
    if float(np.max(np.abs(ip.w1 - ip.wX))) > 1e-12:
        raise SystemExit("packed w1 is not sum X")
    if float(np.max(np.abs(inst.w1 - inst.wX))) > 1e-12:
        raise SystemExit("interleaved w1 is not sum X")
    if m4.nparam("KM_pack", 6) != 21 or m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
        raise SystemExit("parameter count is not the shared-gamma count")
    L = 4
    shells = np.arange(3 * L, dtype=float).reshape(L, 3) * 0.01
    x = m4.r1.pack("KM", 0.2, {"K": -0.1, "M": 0.05}, shells)
    _, snaps = m4.r1.forward(ip, "KM", x, L, store=True)
    gammas = [float(s[-1]) for s in snaps]
    if max(gammas) - min(gammas) > 1e-12:
        raise SystemExit("packed circuit does not share gamma")
    g0 = snaps[0][4]
    if any(float(np.max(np.abs(s[4] - g0))) > 1e-12 for s in snaps):
        raise SystemExit("D changes across layers")
    rng = np.random.default_rng(2)
    out = {}
    for fam, use, depth in (("KM_pack", ip, 2), ("KM", inst, 2), ("qaoa", inst, 3)):
        z = m4.clip_of(use, fam, rng.normal(scale=0.15, size=m4.nparam(fam, depth)), depth)
        _, g, _ = m4.energy_of(use, fam, z, depth)
        err = 0.0
        for i in range(z.size):
            e = np.zeros_like(z)
            e[i] = 1e-6
            num = (m4.energy_of(use, fam, z + e, depth)[0] - m4.energy_of(use, fam, z - e, depth)[0]) / 2e-6
            err = max(err, abs(num - g[i]))
        out[fam] = float(err)
        if err > 1e-6:
            raise SystemExit(f"gradient check failed {fam} {err}")
    return out


def run_job(job):
    N, k, r, _deadline = job
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}")
    return m4.run_job(job)


def main():
    digest = check_hash()
    bind()
    checks = grad_and_layout()
    for d in ("logs", "jobs"):
        (HERE / d).mkdir(exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(PREREG["optimize_wall_seconds"])
    jobs = [(N, k, r, deadline) for k in PREREG["k_confirm"] for N in (12, 9, 6) for r in range(PREREG["repeats"])]
    mp.set_start_method("fork", force=True)
    with mp.Pool(int(PREREG["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
