"""Historical B=0.1 comparison of shared-gamma KM on maximin triples.

No per-layer gamma. Ground states are not loaded. The ferromagnetic-sum,
|J|-max, and Fiedler rules are not fit.
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
sys.path.insert(0, str(HERE))
import packing_maximin  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_cri"))
import layout_gate  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_b01_fe"))
import packing_ferro  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_b01_fid"))
import packing_fiedler  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "5aeff7cd082ba7bc63570ee49867f1a6beb2b79e308f349451972761e3f22aed"


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    if any(name in PREREG["family_order"] for name in ("KM_gl", "KM_pack", "KM_fid", "KM_fe")):
        raise SystemExit("a forbidden family is in the protocol")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = tuple(PREREG["family_order"])
    m4.CAP = int(PREREG["maxfun"])
    m4.BASE.update({"nodata_mm": "nodata", "K_mm": "K", "KM_mm": "KM"})
    m4.packed = lambda fam: str(fam).endswith("_mm")
    m4.best_packing = packing_maximin.maximin_packing
    if any(name in m4.FAMS for name in ("KM_gl", "KM_pack", "KM_fid", "KM_fe")):
        raise SystemExit("forbidden family reached the runner")


def grad_and_layout():
    layout_gate.main()
    inst = m4.m3.instance(6, 4)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}, not the historical 0.1")
    zero, zscore = packing_maximin.maximin_packing(np.zeros((6, 6)))
    again, _ = packing_maximin.maximin_packing(np.zeros((6, 6)))
    if zero != [[0, 1, 2], [3, 4, 5]] or zero != again or abs(zscore) > 1e-12:
        raise SystemExit("the all-zero tie is not the lexicographic partition")
    dev6 = m4.m3.instance(6, 3)
    got, got_score = packing_maximin.maximin_packing(dev6.J)
    brute, brute_score = _brute(dev6.J)
    if got != brute or abs(got_score - brute_score) > 1e-9:
        raise SystemExit("maximin search disagrees with the N=6 enumeration")
    if abs(got_score - min(_triple(dev6.J, t) for t in got)) > 1e-9:
        raise SystemExit("stored score is not the worst triple")
    for N, k in ((9, 0), (12, 0), (12, 4)):
        dev = m4.m3.instance(N, k)
        mm, mm_score = packing_maximin.maximin_packing(dev.J)
        if abs(mm_score - min(_triple(dev.J, t) for t in mm)) > 1e-9:
            raise SystemExit("score is not the worst triple")
        fe, _ = packing_ferro.ferro_packing(dev.J)
        if mm == fe:
            raise SystemExit(f"maximin packing equals the ferromagnetic sum at N={N} k={k}")
        if mm == _abs_packing(dev.J):
            raise SystemExit(f"maximin packing equals |J|-max at N={N} k={k}")
        fied, _ = packing_fiedler.fiedler_packing(dev.J)
        if mm == fied:
            raise SystemExit(f"maximin packing equals Fiedler at N={N} k={k}")
        inter = [[a, a + N // 3, a + 2 * N // 3] for a in range(N // 3)]
        if mm == inter:
            raise SystemExit(f"maximin packing equals interleaved at N={N} k={k}")
        fe_worst = min(_triple(dev.J, t) for t in fe)
        if mm_score + 1e-9 < fe_worst:
            raise SystemExit("maximin did not raise the worst triple")
    return _circuit(inst)


def _triple(J, t):
    return -float(J[t[0], t[1]] + J[t[0], t[2]] + J[t[1], t[2]])


def _brute(J):
    N = J.shape[0]
    sites = list(range(N))
    best, best_score = None, -1.0e300

    def rec(rem, acc):
        nonlocal best, best_score
        if not rem:
            part = tuple(sorted(tuple(sorted(t)) for t in acc))
            score = min(_triple(J, t) for t in part)
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or part < best)):
                best, best_score = part, score
            return
        s0, rest = rem[0], rem[1:]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                rec([rest[k] for k in range(len(rest)) if k != i and k != j], acc + [(s0, rest[i], rest[j])])

    rec(sites, [])
    return [list(t) for t in best], float(best_score)


def _abs_packing(J):
    """The recorded |J|-max search, copied for the development distinctness check only."""
    sites = list(range(J.shape[0]))
    best, best_score = None, -1.0

    def rec(rem, acc):
        nonlocal best, best_score
        if not rem:
            part = tuple(sorted(tuple(sorted(t)) for t in acc))
            score = 0.0
            for t in part:
                score += abs(J[t[0], t[1]]) + abs(J[t[0], t[2]]) + abs(J[t[1], t[2]])
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or part < best)):
                best, best_score = part, score
            return
        s0, rest = rem[0], rem[1:]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                rec([rest[k] for k in range(len(rest)) if k != i and k != j], acc + [(s0, rest[i], rest[j])])

    rec(sites, [])
    return [list(t) for t in best]


def _circuit(inst):
    part, _ = packing_maximin.maximin_packing(inst.J)
    ip = m4.with_packing(inst, part)
    import importlib.util
    spec = importlib.util.spec_from_file_location("mm_replay", HERE / "replay_independent.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    Kh, Mh = replay.features(inst.J, part)
    if float(np.max(np.abs(Kh - ip.K_hat))) > 1e-9 or float(np.max(np.abs(Mh - ip.M_hat))) > 1e-9:
        raise SystemExit("independent features disagree with the model")
    if replay.maximin_packing(inst.J) != part:
        raise SystemExit("independent packing disagrees on the layout instance")
    if float(np.max(np.abs(ip.w1 - ip.wX))) > 1e-12:
        raise SystemExit("maximin w1 is not sum X")
    if float(np.max(np.abs(inst.w1 - inst.wX))) > 1e-12:
        raise SystemExit("interleaved w1 is not sum X")
    if m4.nparam("KM_mm", 6) != 21 or m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
        raise SystemExit("parameter count is not the shared-gamma count")
    L = 4
    shells = np.arange(3 * L, dtype=float).reshape(L, 3) * 0.01
    x = m4.r1.pack("KM", 0.2, {"K": -0.1, "M": 0.05}, shells)
    _, snaps = m4.r1.forward(ip, "KM", x, L, store=True)
    gammas = [float(s[-1]) for s in snaps]
    if max(gammas) - min(gammas) > 1e-12:
        raise SystemExit("maximin circuit does not share gamma")
    g0 = snaps[0][4]
    if any(float(np.max(np.abs(s[4] - g0))) > 1e-12 for s in snaps):
        raise SystemExit("D changes across layers")
    rng = np.random.default_rng(2)
    out = {}
    for fam, use, depth in (("KM_mm", ip, 2), ("KM", inst, 2), ("qaoa", inst, 3)):
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
    if k not in PREREG["k_confirm"] or k in (100, 101):
        raise SystemExit(f"k={k} is outside the frozen window")
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
        "packing": "maximin",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
