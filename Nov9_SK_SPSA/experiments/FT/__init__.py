"""N9-FT dev simulator: GCS-inspired features and transfer.

D2-D5 were approved by Zach at 18:41 Asia/Shanghai on 2026-11-09 as simulation
definitions: SNAP phase 0 above Fock level 7, Uz (a ZZ term that touches a
leaked cavity is off and every other term still acts), per-term leakage energy
(|J_ij| on a touched pair and +b per leaked qubit), and ideal logical readout.
That approval is not a hardware-feasibility claim. Compiled W_L is unmodelled.
Validation and final seeds stay unread.
"""

CLAIM_LEVEL = "approved-simulation-definition"
D2_D5_CONFIRMED = True
APPROVAL = {
    "who": "Zach",
    "when_local": "2026-11-09 18:41 Asia/Shanghai",
    "scope": "simulation definitions D2-D5, not hardware feasibility",
    "snap_f_ge_8": "phase 0",
    "UcZ": "terms touching a leaked cavity are off; all other terms act",
    "energy": "per-term |J_ij| on a leaked pair and +b per leaked qubit",
    "readout": "ideal logical readout",
}
