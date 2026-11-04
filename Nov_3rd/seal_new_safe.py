"""Gate for the NEW_SAFE λ=0.75, μ=0 shallow seal through n=18.

H = diag(unsat) − 0.75 Σ_i X_i. There is no cavity XX term.

Walsh Hamming-shell SNAP is compared with Fourier QAOA. HVA is not a
column: β^XX = 0 is the Fourier state, and H has nothing for an XX mixer
to match.

Med ε is the median over seeds of each seed's median-over-instances excess
⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier. A seed BEATs when its own
n=12 L=2 ratio is ≤ 0.98.

PASS when all of the following hold:

1. n=12 L=2 R_Fourier ≤ 0.85 and at least 4/5 seeds BEAT.
2. Median cavity Λ ≤ 0.80 at n=12 and at n=18. Λ is the median over
   cavities and instances of ⟨(Σ_{i∈c} X_i / 3)^2⟩ on the true ground
   state's X histogram.
3. Median product gap (E_prod − E_0)/|E_0| ≥ 0.03 at n=12 and at n=18.
4. L=2 R_Fourier does not rise by more than 0.05 on either +3-qubit step
   n=12→15 or n=15→18.

L=4 is reported and does not enter the verdict. A prior n=12 L=4 ratio
near 0.90 is a soft lead.

needs_edit when no kill fires and at least one PASS bar is missed. Λ above
0.80 at n=18, with n=12 still under that line, is this case.

KILL when n=18 median Λ ≥ 0.90, or when L=2 R_Fourier > 0.98 at n=12,
n=15, or n=18. A kill wins over a soft miss.
"""
from __future__ import annotations

import math

import numpy as np

LAM = 0.75
MU = 0.0
LAMBDA_X = -0.75
R_PASS = 0.85
R_BEAT = 0.98
R_JUMP = 0.05
LAMBDA_PASS = 0.80
LAMBDA_KILL = 0.90
GAP_PASS = 0.03
N_SEEDS = 5
BEAT_NEED = 4
_EPS = 1e-12


def shell_rms(phi) -> float:
    """RMS distance of φ_k from the best line a + b k, pooled over layers.

    A Fourier X mixer has φ_k = −β k, so this is 0. A bent Hamming shell is > 0.
    """
    arr = np.asarray(phi, dtype=np.float64).reshape(-1, 4)
    k = np.arange(4.0)
    design = np.stack([np.ones(4), k], axis=1)
    resid = []
    for row in arr:
        coef, *_ = np.linalg.lstsq(design, row, rcond=None)
        resid.append(row - design @ coef)
    r = np.concatenate(resid)
    return float(np.sqrt(np.mean(r * r)))


def _finite(name: str, val: float) -> float:
    out = float(val)
    if not math.isfinite(out):
        raise ValueError(f"non-finite {name}")
    return out


def classify_seal(
    r12: float,
    r15: float,
    r18: float,
    beats: int,
    lam12: float,
    lam18: float,
    gap12: float,
    gap18: float,
    *,
    n_seeds: int = N_SEEDS,
) -> dict:
    """Classify the NEW_SAFE λ=0.75 μ=0 seal. Ratios are L=2 R_Fourier."""
    r12 = _finite("n12 R", r12)
    r15 = _finite("n15 R", r15)
    r18 = _finite("n18 R", r18)
    lam12 = _finite("n12 Λ", lam12)
    lam18 = _finite("n18 Λ", lam18)
    gap12 = _finite("n12 gap", gap12)
    gap18 = _finite("n18 gap", gap18)
    beats = int(beats)
    if beats < 0 or beats > int(n_seeds):
        raise ValueError(f"beats {beats} outside 0..{n_seeds}")

    jump_12_15 = r15 - r12
    jump_15_18 = r18 - r15
    kill = []
    if lam18 >= LAMBDA_KILL - _EPS:
        kill.append("n18 median Λ ≥ 0.90")
    if r12 > R_BEAT + _EPS:
        kill.append("n12 L=2 R_Fourier > 0.98")
    if r15 > R_BEAT + _EPS:
        kill.append("n15 L=2 R_Fourier > 0.98")
    if r18 > R_BEAT + _EPS:
        kill.append("n18 L=2 R_Fourier > 0.98")

    soft = []
    if r12 > R_PASS + _EPS:
        soft.append("n12 L=2 R_Fourier > 0.85")
    if beats < BEAT_NEED:
        soft.append(f"L=2 only {beats}/{n_seeds} seeds BEAT")
    if lam12 > LAMBDA_PASS + _EPS:
        soft.append("n12 median Λ > 0.80")
    if lam18 > LAMBDA_PASS + _EPS:
        soft.append("n18 median Λ > 0.80")
    if gap12 + _EPS < GAP_PASS:
        soft.append("n12 median product gap < 0.03")
    if gap18 + _EPS < GAP_PASS:
        soft.append("n18 median product gap < 0.03")
    if jump_12_15 > R_JUMP + _EPS:
        soft.append("L=2 R jump n12→n15 > 0.05")
    if jump_15_18 > R_JUMP + _EPS:
        soft.append("L=2 R jump n15→n18 > 0.05")

    if kill:
        verdict = "KILL"
        reasons = kill
    elif soft:
        verdict = "needs_edit"
        reasons = soft
    else:
        verdict = "PASS"
        reasons = []
    return {
        "verdict": verdict,
        "reasons": reasons,
        "kill": kill,
        "soft": soft,
        "jump_12_15": float(jump_12_15),
        "jump_15_18": float(jump_15_18),
        "beats": beats,
        "r12": r12,
        "r15": r15,
        "r18": r18,
        "lam12": lam12,
        "lam18": lam18,
        "gap12": gap12,
        "gap18": gap18,
    }


def decision_self_test() -> None:
    ok = classify_seal(0.75, 0.76, 0.78, 5, 0.56, 0.57, 0.11, 0.11)
    if ok["verdict"] != "PASS" or ok["reasons"]:
        raise AssertionError(ok)
    # Inclusive edges.
    edge = classify_seal(0.85, 0.90, 0.95, 4, 0.80, 0.80, 0.03, 0.03)
    if edge["verdict"] != "PASS":
        raise AssertionError(edge)
    if abs(edge["jump_12_15"] - 0.05) > 1e-12 or abs(edge["jump_15_18"] - 0.05) > 1e-12:
        raise AssertionError(edge)
    # A drop along the ladder is not a jump.
    drop = classify_seal(0.80, 0.70, 0.60, 5, 0.55, 0.55, 0.10, 0.10)
    if drop["verdict"] != "PASS":
        raise AssertionError(drop)
    # Λ drifts above 0.80 at n=18 only.
    drift = classify_seal(0.75, 0.76, 0.77, 5, 0.56, 0.81, 0.11, 0.11)
    if drift["verdict"] != "needs_edit" or "n18 median Λ > 0.80" not in drift["soft"]:
        raise AssertionError(drift)
    # n=12 Λ miss, n=18 still fine.
    lam12 = classify_seal(0.75, 0.76, 0.77, 5, 0.81, 0.56, 0.11, 0.11)
    if lam12["verdict"] != "needs_edit" or "n12 median Λ > 0.80" not in lam12["soft"]:
        raise AssertionError(lam12)
    # Ratio just over the pass bar.
    soft_r = classify_seal(0.851, 0.86, 0.87, 5, 0.56, 0.57, 0.11, 0.11)
    if soft_r["verdict"] != "needs_edit" or "n12 L=2 R_Fourier > 0.85" not in soft_r["soft"]:
        raise AssertionError(soft_r)
    few = classify_seal(0.75, 0.76, 0.77, 3, 0.56, 0.57, 0.11, 0.11)
    if few["verdict"] != "needs_edit" or not any("3/5" in s for s in few["soft"]):
        raise AssertionError(few)
    gap = classify_seal(0.75, 0.76, 0.77, 5, 0.56, 0.57, 0.029, 0.11)
    if gap["verdict"] != "needs_edit" or "n12 median product gap < 0.03" not in gap["soft"]:
        raise AssertionError(gap)
    gap18 = classify_seal(0.75, 0.76, 0.77, 5, 0.56, 0.57, 0.11, 0.029)
    if gap18["verdict"] != "needs_edit" or "n18 median product gap < 0.03" not in gap18["soft"]:
        raise AssertionError(gap18)
    jump = classify_seal(0.75, 0.81, 0.80, 5, 0.56, 0.57, 0.11, 0.11)
    if jump["verdict"] != "needs_edit" or "L=2 R jump n12→n15 > 0.05" not in jump["soft"]:
        raise AssertionError(jump)
    jump2 = classify_seal(0.75, 0.76, 0.82, 5, 0.56, 0.57, 0.11, 0.11)
    if jump2["verdict"] != "needs_edit" or "L=2 R jump n15→n18 > 0.05" not in jump2["soft"]:
        raise AssertionError(jump2)
    # Kills.
    kill_l = classify_seal(0.75, 0.76, 0.77, 5, 0.56, 0.90, 0.11, 0.11)
    if kill_l["verdict"] != "KILL" or "n18 median Λ ≥ 0.90" not in kill_l["kill"]:
        raise AssertionError(kill_l)
    kill_r = classify_seal(0.99, 0.76, 0.77, 0, 0.56, 0.57, 0.11, 0.11)
    if kill_r["verdict"] != "KILL" or "n12 L=2 R_Fourier > 0.98" not in kill_r["kill"]:
        raise AssertionError(kill_r)
    kill_r18 = classify_seal(0.75, 0.76, 0.981, 5, 0.56, 0.57, 0.11, 0.11)
    if kill_r18["verdict"] != "KILL" or "n18 L=2 R_Fourier > 0.98" not in kill_r18["kill"]:
        raise AssertionError(kill_r18)
    kill_r15 = classify_seal(0.75, 0.99, 0.80, 5, 0.56, 0.57, 0.11, 0.11)
    if kill_r15["verdict"] != "KILL":
        raise AssertionError(kill_r15)
    # Exactly 0.98 is a BEAT, not a kill. The median ratio at 0.98 still passes the kill line.
    tie = classify_seal(0.98, 0.98, 0.98, 4, 0.80, 0.80, 0.03, 0.03)
    if tie["verdict"] != "needs_edit":
        raise AssertionError(tie)
    if any("0.98" in s for s in tie["kill"]):
        raise AssertionError(tie)
    # Kill wins when a soft miss is also present.
    both = classify_seal(1.05, 1.10, 1.20, 0, 0.95, 0.94, 0.01, 0.01)
    if both["verdict"] != "KILL" or len(both["kill"]) < 2 or not both["soft"]:
        raise AssertionError(both)
    # Shell of a pure line is zero; a bent shell is not.
    line = np.array([[0.0, -0.2, -0.4, -0.6], [0.1, -0.1, -0.3, -0.5]])
    if shell_rms(line) > 1e-12:
        raise AssertionError(shell_rms(line))
    bent = np.array([[0.0, -0.2, 0.4, -0.6]])
    if shell_rms(bent) <= 1e-8:
        raise AssertionError(shell_rms(bent))


if __name__ == "__main__":
    decision_self_test()
    print("seal_new_safe decision self-test ok")
