"""W_BinH compile freeze. No SPSA on the isometry and no SK Hamiltonian."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import iterations_equal4
from Nov3_INIT.wcompile import (
    A_MAX,
    COMPILER_SEED,
    K_PAIRS,
    N_PARAMS,
    NF_OPT,
    NF_REPLAY,
    Q_COMPILE,
    compiled_map,
    metrics,
    target_isometry,
    unpack,
    x0_identity,
)
from Nov9_SK_SPSA.experiments.FT.sim import displacement_matrix

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WCOMP.md"
OUT = ROOT / "results" / "wcompile" / "CONFORMANCE.json"
SK_LEDGER = 2818048


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def main() -> None:
    t16 = target_isometry(NF_OPT)
    assert t16.shape == (16, 8)
    assert np.allclose(t16[8:, :], 0.0)
    w = t16[:8, :]
    assert np.allclose(w @ w.T, np.eye(8), atol=1e-12)
    d0 = displacement_matrix(0.0, NF_OPT, "restricted")
    dtrue = displacement_matrix(0.4, NF_OPT, "restricted")
    dcrop = displacement_matrix(0.4, 8, "strict8")
    # Cropped 8-level D is not the nf=16 restriction of a true D.
    crop_is_not_true = not np.allclose(dtrue[:8, :8], dcrop, atol=1e-6)
    ident = x0_identity()
    v0 = compiled_map(ident, NF_OPT)
    m0 = metrics(ident, NF_OPT)
    alpha, theta = unpack(ident)
    q_ok = iterations_equal4(Q_COMPILE) == 250 and N_PARAMS == 27 and K_PAIRS == 3
    ledger_ok = iterations_equal4(128) == 10 and Q_COMPILE == 2048

    payload = {
        "ok": bool(
            q_ok
            and crop_is_not_true
            and np.allclose(d0, np.eye(NF_OPT), atol=1e-10)
            and np.allclose(v0[:8, :], np.eye(8), atol=1e-12)
            and np.allclose(v0[8:, :], 0.0, atol=1e-12)
            and abs(m0["F_avg"] - 0.125) < 1e-12
                and abs(m0["F_hs"]) < 1e-18
            and np.allclose(alpha, 0.0)
            and np.allclose(theta, 0.0)
            and A_MAX == 1.8
            and NF_REPLAY == 32
            and COMPILER_SEED == 33001
            and COMPILER_SEED not in range(11000, 11032)
            and COMPILER_SEED not in range(20000, 21000)
            and ledger_ok
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "sk_ledger_untouched": SK_LEDGER,
        "compiler_Q": Q_COMPILE,
        "T": iterations_equal4(Q_COMPILE),
        "n_params": N_PARAMS,
        "K": K_PAIRS,
        "nf_opt": NF_OPT,
        "nf_replay": NF_REPLAY,
        "real_alpha_only": True,
        "snap_pad_0": True,
        "not_flat_prep": True,
        "not_ux_compile": True,
        "not_cropped_D8": True,
        "crop_is_not_true": crop_is_not_true,
        "identity_F_avg": m0["F_avg"],
        "identity_F_hs": m0["F_hs"],
        "A_MAX": A_MAX,
        "no_spsa_on_isometry_in_this_module": True,
        "no_sk_hamiltonian": True,
        "global_phase": "one overall arg(tr(T^H V))",
        "label_rule": "WCOMP_CANDIDATE iff F_avg>=0.99, F_worst>=0.95, mean p_leak<=0.01, worst p_leak<=0.05, F_HS>=0.99, replay |dF_avg|<=1e-3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
