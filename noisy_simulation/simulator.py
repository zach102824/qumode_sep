"""Noisy drop-in for ``noiseless.relayout_kbit.XorKbitSim``: same encoding, XOR relabel, leakage and
Gibbs cost; only ``probs_from_x`` changes (Born probabilities of the noisy density matrix)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noiseless.relayout_kbit import XorKbitSim

from .circuit import apply_noisy_circuit
from .device import EICKBUSCH, Device
from .kraus import probabilities


@dataclass
class NoisyXorKbitSim(XorKbitSim):
    device: Device = EICKBUSCH
    jp_method: str = "fast"
    jumps: int = 3

    def __post_init__(self):
        super().__post_init__()
        self._last_key = None
        self._last_probs = None

    def probs_from_x(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        key = (int(self.n_layers), x.tobytes())
        if key != self._last_key:
            rho = apply_noisy_circuit(x, int(self.n_layers), self.layout.nf, self.device,
                                      method=self.jp_method, jumps=self.jumps)
            p = np.clip(probabilities(rho), 0.0, None)
            self._last_probs = p / p.sum()
            self._last_key = key
        return self._last_probs.copy()
