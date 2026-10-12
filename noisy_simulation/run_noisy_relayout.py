#!/usr/bin/env python3
"""Explore + XOR-relabel SPSA run (input_circuit/circuit.json) on the noisy device.

Same protocol, seeds, checkpoint format and summary as noiseless/run_relayout_kbit.py; the only change
is the simulator: NoisyXorKbitSim evolves the (d, e, A, B) density matrix through ideal ECD blocks +
exact T1/T2 Kraus channels, and the JP gate as the physical q-sequence (see NOISE.md).

Defaults reproduce the input circuit at the Eickbusch device point:
    python noisy_simulation/run_noisy_relayout.py --trials 5
Noise sweep: --noise-scale multiplies every dissipative rate (0 = noiseless reference).
Smoke test: --max-h 1 --trials 1 --s 2 --r 4 --nf 12
"""

from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from noiseless.run_relayout_kbit import config_tag, run_config, summarize  # noqa: E402
from noisy_simulation.circuit import CHANNEL_CACHE_ENV, get_cavity_channel  # noqa: E402
from noisy_simulation.device import PRESETS  # noqa: E402
from noisy_simulation.simulator import NoisyXorKbitSim  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "results"
SPEC = HERE / "input_circuit" / "circuit.json"


def spec_defaults(path: Path = SPEC) -> dict:
    d = json.loads(Path(path).read_text())
    hw, al = d["hardware"], d["algorithm"]
    ex, rl = al["explore"], al["relabel"]
    return dict(n=int(d["noiseless_reference"]["n"]), ham_set=d["hamiltonians"]["set"], code=hw["cavity_code"],
                nf=int(hw["fock_levels_per_cavity"]), L=int(d["ansatz"]["L_max"]), s=int(ex["spsa_steps_per_layer"]),
                explore=int(ex["E"]), explore_mask=ex["explore_mask"], topk=1, xeta=16.0, reta=2.0,
                lr=float(rl["lr"]), R=int(rl["R"]), r=int(rl["spsa_steps"]))


def main(argv=None) -> int:
    sd = spec_defaults()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, typ in (("n", int), ("L", int), ("s", int), ("r", int), ("R", int), ("lr", float), ("nf", int),
                      ("explore", int), ("topk", int), ("xeta", float), ("reta", float)):
        p.add_argument(f"--{name}", type=typ, default=sd[name])
    p.add_argument("--explore-mask", default=sd["explore_mask"])
    p.add_argument("--code", default=sd["code"])
    p.add_argument("--ham-set", default=sd["ham_set"])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--max-h", type=int, default=None)
    p.add_argument("--workers", type=int, default=4, help="8 GB RAM: ~4 (each worker holds a few 2304^2 rho copies)")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--outdir", default=str(RUN_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--preset", choices=sorted(PRESETS), default="eickbusch")
    p.add_argument("--noise-scale", "--kappa-tau-scale", dest="noise_scale", type=float, default=1.0,
                   help="multiplies kappa, gamma, gamma_phi (1 = device, 0 = noiseless check)")
    p.add_argument("--n-slice", type=int, default=None, help="Strang sub-steps per JP wait (default: device, 8)")
    p.add_argument("--jp-method", choices=("fast", "direct"), default="fast")
    p.add_argument("--jumps", type=int, default=3, help="max total photon jumps kept in the fast JP channel")
    a = p.parse_args(argv)

    base = PRESETS[a.preset]
    dev = base.with_(noise_scale=a.noise_scale * base.noise_scale, n_slice=a.n_slice or base.n_slice)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or (config_tag(a.n, a.L, a.s, a.r, a.R, a.lr, a.nf, a.ham_set, "fast", a.explore, a.topk,
                               a.explore_mask, a.xeta, a.reta, a.L, None, code=a.code)
                    + f"_{a.preset}x{a.noise_scale:g}_{a.jp_method}" + (f"_ns{dev.n_slice}" if a.n_slice else ""))
    print(json.dumps({"tag": tag, "device": dev.summary()}, indent=1), flush=True)

    if a.jp_method == "fast":
        _os.environ[CHANNEL_CACHE_ENV] = str(outdir / "channel_cache")
        t0 = time.perf_counter()
        get_cavity_channel(dev, a.nf, a.jumps, dev.tau_ecd)
        print(f"JP cavity channel ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    extra = {"sim_cls": NoisyXorKbitSim,
             "sim_kwargs": {"device": dev, "jp_method": a.jp_method, "jumps": a.jumps}}
    tag, recs = run_config(n=a.n, L=a.L, s=a.s, r=a.r, R=a.R, lr=a.lr, nf=a.nf, trials=a.trials,
                           trial_offset=a.trial_offset, ham_set=a.ham_set, method="fast", workers=a.workers,
                           seed0=a.seed, outdir=outdir, nf_check=0, max_h=a.max_h, tag=tag,
                           explore=a.explore, topk=a.topk, explore_mask=a.explore_mask, xeta=a.xeta, reta=a.reta,
                           xL=a.L, code=a.code, job_extra=extra)
    sm = summarize(recs)
    sm.pop("per_h", None)
    print(json.dumps({"tag": tag, "device": dev.summary(), **sm}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
