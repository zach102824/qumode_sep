"""Decision rule for the transverse-field family, μ = 0.

    H = diag(unsat) − λ Σ_i X_i,   λ ∈ {0.50, 0.75, 1.00}

There is no cavity XX. Fourier QAOA is the baseline: its mixer is ⊗ exp(−i β X/2).
Hamming-shell Walsh SNAP is the method. HVA is not a third column. With μ = 0
the XX mixer has nothing in H to match, and β^XX = 0 is exactly Fourier.

Λ, the product gap, and the Z-peak are medians on the frozen n=12 pool.
Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state.
The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state.
A classical ground state means the median Z-basis peak is at least 0.5.

Per λ, first match wins:

- PASS when median Λ ≤ 0.80, product gap ≥ 0.03, the ground state is not
  classical, and n=12 R_Fourier ≤ 0.85 at L=2 or at L=4.
- KILL when median Λ > 0.85, or the ground state is classical, or
  R_Fourier > 0.98 at both L=2 and L=4 on the n=6 and n=9 holdouts.
- needs_edit when the Λ / gap / non-classical bar holds and the better n=12
  depth has R_Fourier in (0.85, 0.98].
- needs_edit also covers a cell that misses PASS and does not meet a KILL
  clause (Λ in (0.80, 0.85], a short gap, or n=12 R above 0.98 while the
  train-size ratios are not all above 0.98).

The family is PASS if any λ is PASS. It is KILL only when every λ is KILL.
Otherwise it is needs_edit. A PASS cell is not overridden by the train-size
ratio kill: the n=12 bar is the one the PASS line names.
"""
from __future__ import annotations

import math

import numpy as np

LAMBDAS = (0.50, 0.75, 1.00)
MU = 0.0
DEPTHS = (2, 4)
TRAIN_NS = (6, 9)
FREEZE_N = 12

LAMBDA_PASS = 0.80
LAMBDA_KILL = 0.85
GAP_PASS = 0.03
R_PASS = 0.85
R_SOFT = 0.98
CLASSICAL_PEAK = 0.50
_EPS = 1e-12


def lambda_coeff(lam: float) -> float:
    """Coefficient of Σ_i X_i in H = diag(unsat) − λ Σ X."""
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError(f"λ must be a nonnegative finite field, got {lam}")
    return -lam


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


def _train_ratios(r_train: dict) -> list[float]:
    out = []
    for n in TRAIN_NS:
        for depth in DEPTHS:
            if (n, depth) in r_train:
                out.append(float(r_train[(n, depth)]))
            elif f"{n},{depth}" in r_train:
                out.append(float(r_train[f"{n},{depth}"]))
            else:
                raise KeyError(f"missing train ratio n={n} L={depth}")
    if len(out) != 4:
        raise RuntimeError(f"expected 4 train ratios, got {out}")
    return out


def classify_cell(
    *,
    lam_med: float,
    gap_med: float,
    peak_med: float,
    r12_by_depth: dict,
    r_train: dict,
) -> dict:
    """Classify one λ. Ratios are Medε_SNAP / Medε_Fourier."""
    lam = float(lam_med)
    gap = float(gap_med)
    peak = float(peak_med)
    for name, val in (("Λ", lam), ("gap", gap), ("peak", peak)):
        if not math.isfinite(val):
            raise ValueError(f"non-finite {name}")
    r2 = float(r12_by_depth[2] if 2 in r12_by_depth else r12_by_depth["2"])
    r4 = float(r12_by_depth[4] if 4 in r12_by_depth else r12_by_depth["4"])
    if not math.isfinite(r2) or not math.isfinite(r4):
        raise ValueError(f"non-finite n12 ratio L2={r2} L4={r4}")
    train_vals = _train_ratios(r_train)
    if not all(math.isfinite(v) for v in train_vals):
        raise ValueError(f"non-finite train ratios {train_vals}")

    classical = peak + _EPS >= CLASSICAL_PEAK
    lambda_good = lam <= LAMBDA_PASS + _EPS
    gap_good = gap + _EPS >= GAP_PASS
    quantum_bar = bool(lambda_good and gap_good and not classical)
    r_pass = bool(r2 <= R_PASS + _EPS or r4 <= R_PASS + _EPS)
    r_best = float(min(r2, r4))
    r_soft = bool((not r_pass) and r_best <= R_SOFT + _EPS)
    train_kill = bool(all(v > R_SOFT + _EPS for v in train_vals))

    kill: list[str] = []
    if lam > LAMBDA_KILL + _EPS:
        kill.append(f"median Λ {lam:.4f} > 0.85")
    if classical:
        kill.append(f"classical GS (median Z-peak {peak:.4f} ≥ 0.5)")
    if train_kill:
        kill.append("R_Fourier > 0.98 at L=2 and L=4 on n6 and n9")

    if quantum_bar and r_pass:
        call = "PASS"
        detail = "n12 R≤0.85 with Λ≤0.80, gap≥0.03, and a non-classical GS"
    elif kill:
        call = "KILL"
        detail = "; ".join(kill)
    elif quantum_bar and r_soft:
        call = "needs_edit"
        detail = f"soft n12 R={r_best:.4f} in (0.85, 0.98] with Λ and the gap inside the bar"
    else:
        call = "needs_edit"
        misses = []
        if not lambda_good:
            misses.append(f"Λ {lam:.4f} is inside (0.80, 0.85]")
        if not gap_good:
            misses.append(f"product gap {gap:.4f} < 0.03")
        if not r_pass and not r_soft:
            misses.append(f"better n12 R={r_best:.4f} > 0.98")
        detail = "no PASS and no KILL clause"
        if misses:
            detail = detail + ": " + "; ".join(misses)

    return {
        "call": call,
        "detail": detail,
        "kill": kill,
        "quantum_bar": quantum_bar,
        "r_pass": r_pass,
        "r_soft": r_soft,
        "r_best": r_best,
        "train_kill": train_kill,
        "classical": classical,
        "lambda_med": lam,
        "gap_med": gap,
        "peak_med": peak,
        "r12_L2": r2,
        "r12_L4": r4,
    }


def classify_family(cells: list[dict]) -> dict:
    """Family verdict. PASS if any λ passes. KILL only when every λ is KILL."""
    if len(cells) != len(LAMBDAS):
        raise ValueError(f"expected {len(LAMBDAS)} λ cells, got {len(cells)}")
    calls = [str(c["call"]) for c in cells]
    allowed = {"PASS", "needs_edit", "KILL"}
    if any(c not in allowed for c in calls):
        raise ValueError(f"bad cell call in {calls}")
    if any(c == "PASS" for c in calls):
        verdict = "PASS"
    elif all(c == "KILL" for c in calls):
        verdict = "KILL"
    else:
        verdict = "needs_edit"
    return {"verdict": verdict, "calls": calls}


def decision_self_test() -> None:
    """Raise AssertionError if the gate disagrees with the written rule."""
    good = dict(lam_med=0.60, gap_med=0.05, peak_med=0.10)
    train_win = {(6, 2): 0.40, (6, 4): 0.50, (9, 2): 0.45, (9, 4): 0.55}
    train_dead = {(6, 2): 0.99, (6, 4): 0.99, (9, 2): 0.991, (9, 4): 1.01}

    def cell(**kw):
        base = dict(good)
        base["r_train"] = dict(train_win)
        base["r12_by_depth"] = {2: 0.50, 4: 0.70}
        base.update(kw)
        return classify_cell(**base)

    if cell()["call"] != "PASS":
        raise AssertionError("clear PASS")
    if cell(r12_by_depth={2: 0.90, 4: 0.70})["call"] != "PASS":
        raise AssertionError("L=4 alone is enough")
    if cell(r12_by_depth={2: 0.85, 4: 1.10})["call"] != "PASS":
        raise AssertionError("R=0.85 is inside the PASS bar")
    soft = cell(r12_by_depth={2: 0.90, 4: 0.98})
    if soft["call"] != "needs_edit" or not soft["r_soft"]:
        raise AssertionError(soft)
    if cell(r12_by_depth={2: 0.99, 4: 1.05})["call"] != "needs_edit":
        raise AssertionError("n12 above 0.98 without a train kill is needs_edit")

    locked = cell(lam_med=0.90, r12_by_depth={2: 0.30, 4: 0.40})
    if locked["call"] != "KILL" or not any("Λ" in s for s in locked["kill"]):
        raise AssertionError(locked)
    classical = cell(lam_med=0.40, peak_med=0.50, gap_med=0.0, r12_by_depth={2: 0.2, 4: 0.2})
    if classical["call"] != "KILL" or not classical["classical"]:
        raise AssertionError(classical)
    # Λ=0.85 exactly is not above the kill line and not inside the pass line.
    between = cell(lam_med=0.85)
    if between["call"] != "needs_edit" or between["kill"]:
        raise AssertionError(between)
    dominated = cell(r_train=train_dead, r12_by_depth={2: 0.90, 4: 0.90})
    if dominated["call"] != "KILL" or not dominated["train_kill"]:
        raise AssertionError(dominated)
    # n12 PASS outranks a train-size ratio kill.
    saved = cell(r_train=train_dead, r12_by_depth={2: 0.50, 4: 0.99})
    if saved["call"] != "PASS" or not saved["train_kill"]:
        raise AssertionError(saved)
    short = cell(gap_med=0.029)
    if short["call"] != "needs_edit" or short["quantum_bar"]:
        raise AssertionError(short)

    passing = cell()
    killing = cell(lam_med=0.94)
    editing = cell(r12_by_depth={2: 0.90, 4: 0.92})
    if classify_family([passing, killing, killing])["verdict"] != "PASS":
        raise AssertionError("one PASS λ saves the family")
    if classify_family([killing, killing, killing])["verdict"] != "KILL":
        raise AssertionError("family KILL needs every λ")
    if classify_family([editing, killing, killing])["verdict"] != "needs_edit":
        raise AssertionError("a soft λ blocks KILL")
    edge = cell(lam_med=0.80, gap_med=0.03, peak_med=0.499)
    if edge["call"] != "PASS":
        raise AssertionError(edge)

    phi = np.array([[0.0, -0.2, -0.4, -0.6], [0.0, -0.1, -0.2, -0.3]])
    if shell_rms(phi) > 1e-12:
        raise AssertionError(f"linear shell rms {shell_rms(phi)}")
    bent = phi.copy()
    bent[0, 2] += 0.3
    if shell_rms(bent) <= 0.0:
        raise AssertionError("bent shell")
    if lambda_coeff(0.75) != -0.75 or lambda_coeff(0.0) != 0.0:
        raise AssertionError("lambda coeff")
    try:
        lambda_coeff(-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative λ must fail")
