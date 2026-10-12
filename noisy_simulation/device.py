"""Hardware parameters of the 3-transmon (d, q, e) + 2-cavity (A, B) device.

All inputs are laboratory quantities in SI units (seconds, rad/s). The circuit code only ever uses
the dimensionless products rate * duration (kappa*tau, gamma*tau, chi*t), so no unit system is
baked into the simulator. See NOISE.md for where each number comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Device:
    t1_cav: float = 436e-6          # cavity photon lifetime T1,cav (Eickbusch 2022)
    t1_q: float = 50e-6             # transmon T1, same for d, q, e
    t2_q: float = 30e-6             # transmon T2 (Ramsey), same for d, q, e
    chi_qa: float = TWO_PI * 1e6    # matched shared shift chi_qA = chi_qB (rad/s)
    tau_ecd: float = 0.65e-6        # one parallel (R_d, R_e, ECD_dA || ECD_eB) block (Dutta, conservative)
    chi_spectator: float = 0.0      # chi_dA = chi_eB during the JP waits (rad/s); 0 = echoed out
    kerr: float = 0.0               # self-Kerr K_A = K_B during the JP waits (rad/s)
    cross_kerr: float = 0.0         # K_AB during the JP waits (rad/s)
    noise_scale: float = 1.0        # multiplies every dissipative rate (0 = noiseless, 1 = device)
    n_slice: int = 8                # Strang sub-steps per JP wait (8: < 1e-4 trace distance, NOISE.md)
    jmax: int | None = 2            # photon-loss Kraus truncation (None = exact, all jumps)

    def __post_init__(self):
        if self.t2_q > 2.0 * self.t1_q:
            raise ValueError("T2 must be <= 2 T1")
        if self.n_slice < 1:
            raise ValueError("n_slice must be >= 1")

    # dissipative rates (1/s), scaled by noise_scale
    @property
    def kappa(self) -> float:
        return self.noise_scale / self.t1_cav

    @property
    def gamma(self) -> float:
        return self.noise_scale / self.t1_q

    @property
    def gamma_phi(self) -> float:
        """Pure dephasing rate in the (gamma_phi / 2) D[sigma_z] convention: 1/T2 = gamma/2 + gamma_phi."""
        return self.noise_scale * (1.0 / self.t2_q - 0.5 / self.t1_q)

    # timings (s)
    @property
    def tau_wait(self) -> float:
        """One JP wait, chi_qA * tau = pi."""
        return np.pi / self.chi_qa

    @property
    def tau_jp(self) -> float:
        """Total JP idle time (two waits); the four qubit pi/2 pulses are treated as instantaneous."""
        return 2.0 * self.tau_wait

    def with_(self, **kw) -> "Device":
        return replace(self, **kw)

    def summary(self) -> dict:
        return {
            "kappa_tau_ecd": self.kappa * self.tau_ecd,
            "kappa_tau_jp": self.kappa * self.tau_jp,
            "gamma_tau_ecd": self.gamma * self.tau_ecd,
            "gamma_tau_jp": self.gamma * self.tau_jp,
            "gamma_phi_tau_jp": self.gamma_phi * self.tau_jp,
            "gamma_over_chi": self.gamma / self.chi_qa,
            "gamma_phi_over_chi": self.gamma_phi / self.chi_qa,
            "kappa_over_chi": self.kappa / self.chi_qa,
            "tau_jp_us": self.tau_jp * 1e6,
        }


EICKBUSCH = Device()
NOISELESS = Device(noise_scale=0.0)
PRESETS = {"eickbusch": EICKBUSCH, "noiseless": NOISELESS}
