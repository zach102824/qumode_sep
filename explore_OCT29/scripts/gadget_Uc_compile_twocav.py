#!/usr/bin/env python3
"""Two-cavity projector compile (simulation): when U(γ)=I+(e^{-iγ}-1) C_a⊗C_b
factors as locals vs needs entangling; fidelity table.

Writes grape_library/COMPILE_TWOCAV.md + compile_twocav.json.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_grape_library import (  # noqa: E402
    U_from_C,
    process_fidelity,
    projector_Z,
    snap_from_phases,
    exact_snap_phases_for_CZ,
)

OUT = _PKG / "runs" / "gadget_Uc" / "grape_library"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[2cav]", *a, flush=True)


def support_fock(C: np.ndarray) -> list[int]:
    return [int(f) for f in range(C.shape[0]) if abs(C[f, f]) > 0.5]


def local_snap_U(C: np.ndarray, gamma: float) -> np.ndarray:
    th = exact_snap_phases_for_CZ(C, gamma)
    return snap_from_phases(th, nf=8)


def kron_local_snaps(Ca, Cb, gamma_a, gamma_b) -> np.ndarray:
    return np.kron(local_snap_U(Ca, gamma_a), local_snap_U(Cb, gamma_b))


def joint_diag_phase_U(Ca, Cb, gamma: float) -> np.ndarray:
    """Exact U on 64-dim: phase γ iff both cavities in projector support (Z-frame)."""
    C = np.kron(Ca, Cb)
    return U_from_C(C, gamma)


def separable_phase_best_fit(Ca, Cb, gamma: float) -> tuple[np.ndarray, float]:
    """Best U = SNAP_a ⊗ SNAP_b with θ_a on supp(Ca), θ_b on supp(Cb) (2 scalars).

    Target phase pattern: γ on supp_a × supp_b, 0 elsewhere.
    Separable phases: φ(fa,fb)=α·1_a(fa)+β·1_b(fb). Cannot match AND; optimize α,β.
    """
    sa, sb = set(support_fock(Ca)), set(support_fock(Cb))
    # grid of candidate α,β in [0, γ] and outside
    best_U, best_fid = None, -1.0
    U_tgt = joint_diag_phase_U(Ca, Cb, gamma)
    for alpha, beta in itertools.product(
        np.linspace(0, gamma, 11), np.linspace(0, gamma, 11)
    ):
        U = kron_local_snaps(Ca, Cb, float(alpha), float(beta))
        f = process_fidelity(U_tgt, U)
        if f > best_fid:
            best_fid, best_U = f, U
    # also try α=γ, β=0 and vice versa and α=β=γ/2
    for alpha, beta in [(gamma, 0), (0, gamma), (gamma / 2, gamma / 2), (gamma, gamma)]:
        U = kron_local_snaps(Ca, Cb, float(alpha), float(beta))
        f = process_fidelity(U_tgt, U)
        if f > best_fid:
            best_fid, best_U = f, U
    return best_U, float(best_fid)


def beamsplitter(nf: int = 8, theta: float = np.pi / 4) -> np.ndarray:
    """Two-mode beamsplitter on Fock⊗Fock truncated: exp(-i θ (a†b + ab†))."""
    a = np.diag(np.sqrt(np.arange(1, nf, dtype=np.float64)), 1).astype(np.complex128)
    ad = a.T.conj()
    # operators on 64 dim
    A = np.kron(a, np.eye(nf))
    Ad = np.kron(ad, np.eye(nf))
    B = np.kron(np.eye(nf), a)
    Bd = np.kron(np.eye(nf), ad)
    H = theta * (Ad @ B + A @ Bd)
    # expm via eig
    w, V = np.linalg.eig(H)
    return (V * np.exp(-1j * w)) @ np.linalg.inv(V)


def fit_bs_sandwich(Ca, Cb, gamma: float, n_starts: int = 8, steps: int = 80, seed: int = 0) -> dict:
    """Fit BS(θ) · (SNAP_a ⊗ SNAP_b) · BS(θ)† ≈ U_tgt (joint projector phase).

    Params: θ, θa[8], θb[8] — enough to show entangling help vs local-only.
    """
    rng = np.random.default_rng(seed)
    U_tgt = torch.as_tensor(joint_diag_phase_U(Ca, Cb, gamma), dtype=torch.complex128)
    nf = 8
    d = nf * nf

    def snap8(th):
        return torch.diag(torch.exp(-1j * th.to(torch.complex128)))

    # precompute BS generator pieces for autodiff via matrix exp on small H is heavy;
    # use finite set of θ and optimize SNAP phases only, plus optimize θ by grid.
    sa = support_fock(Ca)
    sb = support_fock(Cb)
    best = {"fid": 0.0, "theta_bs": None}

    theta_grid = np.linspace(0, np.pi / 2, 9)
    for th_bs in theta_grid:
        BS = torch.as_tensor(beamsplitter(nf, float(th_bs)), dtype=torch.complex128)
        BSd = BS.mH
        # init SNAP phases: γ on supports
        tha = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64))
        thb = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64))
        with torch.no_grad():
            for f in sa:
                tha[f] = gamma / 2
            for f in sb:
                thb[f] = gamma / 2
        # random jitter starts
        for start in range(n_starts):
            if start > 0:
                with torch.no_grad():
                    tha.add_(torch.as_tensor(rng.normal(0, 0.2, size=8)))
                    thb.add_(torch.as_tensor(rng.normal(0, 0.2, size=8)))
            opt = torch.optim.Adam([tha, thb], lr=0.1)
            for _ in range(steps):
                opt.zero_grad()
                Uloc = torch.kron(snap8(tha), snap8(thb))
                U = BS @ Uloc @ BSd
                # process fidelity
                fid = (torch.abs(torch.trace(U_tgt.mH @ U)) ** 2 / (d * d)).real
                loss = 1.0 - fid
                loss.backward()
                opt.step()
            with torch.no_grad():
                Uloc = torch.kron(snap8(tha), snap8(thb))
                U = BS @ Uloc @ BSd
                fid = float((torch.abs(torch.trace(U_tgt.mH @ U)) ** 2 / (d * d)).real)
            if fid > best["fid"]:
                best = {
                    "fid": fid,
                    "theta_bs": float(th_bs),
                    "tha": tha.detach().cpu().tolist(),
                    "thb": thb.detach().cpu().tolist(),
                }
    return best


def make_reps():
    """Representative local projectors for (1,3) and (2,2) splits."""
    # (1,3): cavity A bit0 +, cavity B all three bits +++
    Ca_13 = projector_Z([(0, 1)])
    Cb_13 = projector_Z([(0, 1), (1, 1), (2, 1)])
    # (2,2): A bits 0,1 ++; B bits 0,1 ++
    Ca_22 = projector_Z([(0, 1), (1, 1)])
    Cb_22 = projector_Z([(0, 1), (1, 1)])
    return {
        "(1,3)": {"Ca": Ca_13, "Cb": Cb_13, "ka": 1, "kb": 3},
        "(2,2)": {"Ca": Ca_22, "Cb": Cb_22, "ka": 2, "kb": 2},
    }


def analytic_notes(Ca, Cb, gamma: float) -> dict:
    C = np.kron(Ca, Cb)
    U = U_from_C(C, gamma)
    # Try factor as Ua ⊗ Ub for projector-form locals with same γ
    Ua = U_from_C(Ca, gamma)
    Ub = U_from_C(Cb, gamma)
    U_prod = np.kron(Ua, Ub)
    fro = float(np.linalg.norm(U - U_prod, ord="fro"))
    fid = process_fidelity(U, U_prod)
    # Rank / support sizes
    return {
        "rank_C": int(np.linalg.matrix_rank(C.real.round(10))),
        "rank_Ca": int(np.linalg.matrix_rank(Ca.real.round(10))),
        "rank_Cb": int(np.linalg.matrix_rank(Cb.real.round(10))),
        "supp_a": support_fock(Ca),
        "supp_b": support_fock(Cb),
        "fro_U_vs_Ua_kron_Ub": fro,
        "fid_U_vs_Ua_kron_Ub": fid,
        "factors_as_local_projector_unitaries": fro < 1e-8,
        "reason": (
            "Ua⊗Ub expands to terms Ca⊗I, I⊗Cb, Ca⊗Cb — not equal to I+(e^{-iγ}-1)Ca⊗Cb "
            "unless one factor is trivial. Joint phase is AND of supports → nonseparable."
        ),
    }


def run_table(gammas=(0.5, 1.0, math.pi)):
    reps = make_reps()
    rows = []
    W = build_W_BinH(nf=8, dtype=torch.complex128).numpy()[:8, :8]
    for label, rep in reps.items():
        Ca, Cb = rep["Ca"], rep["Cb"]
        notes = analytic_notes(Ca, Cb, gamma=1.0)
        log(label, "factors_local?", notes["factors_as_local_projector_unitaries"],
            "fid U vs Ua⊗Ub", f"{notes['fid_U_vs_Ua_kron_Ub']:.6f}")
        for g in gammas:
            U_tgt = joint_diag_phase_U(Ca, Cb, g)
            # exact joint SNAP on 64 (diagonal) — identity check
            C = np.kron(Ca, Cb)
            th = exact_snap_phases_for_CZ(C, g)  # works on 64 if we pad — C is 64x64
            # exact_snap_phases_for_CZ assumes diag of C
            diag = np.real(np.diag(C))
            theta64 = np.where(diag > 0.5, g, 0.0)
            U_joint_snap = np.diag(np.exp(-1j * theta64))
            fid_exact = process_fidelity(U_tgt, U_joint_snap)

            _, fid_sep = separable_phase_best_fit(Ca, Cb, g)
            bs = fit_bs_sandwich(Ca, Cb, g, n_starts=4, steps=60, seed=hash(label + str(g)) % 10000)

            # BinH: conjugate each local then kron — still wrong for same reason
            Wd = W.conj().T
            Ua_x = Wd @ U_from_C(Ca, g) @ W
            Ub_x = Wd @ U_from_C(Cb, g) @ W
            fid_x_kron = process_fidelity(
                (np.kron(Wd, Wd) @ U_tgt @ np.kron(W, W)),  # U_tgt in X frame
                np.kron(Ua_x, Ub_x),
            )
            # Actually compare X-frame target properly:
            U_tgt_X = np.kron(Wd, Wd) @ U_tgt @ np.kron(W, W)
            fid_x_kron = process_fidelity(U_tgt_X, np.kron(Ua_x, Ub_x))

            row = {
                "split": label,
                "gamma": g,
                "fid_joint_SNAP_exact": fid_exact,
                "fid_best_local_SNAP_kron": fid_sep,
                "fid_BS_sandwich": bs["fid"],
                "BS_theta": bs["theta_bs"],
                "fid_BinH_local_kron_Xframe": fid_x_kron,
                "needs_entangling": fid_sep < 0.999,
            }
            rows.append(row)
            log(
                f"{label} γ={g:.3f}: jointSNAP={fid_exact:.6f} "
                f"localKron={fid_sep:.4f} BS={bs['fid']:.4f} (θ={bs['theta_bs']:.3f}) "
                f"Xkron={fid_x_kron:.4f}"
            )
        notes["split"] = label
        notes["ka"] = rep["ka"]
        notes["kb"] = rep["kb"]
        yield notes, rows
    return notes, rows  # noqa — generator used differently


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reps = make_reps()
    gammas = (0.5, 1.0, math.pi)
    all_notes = {}
    all_rows = []
    for label, rep in reps.items():
        notes = analytic_notes(rep["Ca"], rep["Cb"], 1.0)
        notes["ka"] = rep["ka"]
        notes["kb"] = rep["kb"]
        all_notes[label] = notes
        log(label, "local factor?", notes["factors_as_local_projector_unitaries"],
            f"fid={notes['fid_U_vs_Ua_kron_Ub']:.6f}")

    for label, rep in reps.items():
        Ca, Cb = rep["Ca"], rep["Cb"]
        W = build_W_BinH(nf=8, dtype=torch.complex128).numpy()[:8, :8]
        Wd = W.conj().T
        for g in gammas:
            U_tgt = joint_diag_phase_U(Ca, Cb, g)
            C = np.kron(Ca, Cb)
            theta64 = np.where(np.real(np.diag(C)) > 0.5, g, 0.0)
            U_joint_snap = np.diag(np.exp(-1j * theta64))
            fid_exact = process_fidelity(U_tgt, U_joint_snap)
            _, fid_sep = separable_phase_best_fit(Ca, Cb, g)
            bs = fit_bs_sandwich(Ca, Cb, g, n_starts=4, steps=50, seed=abs(hash((label, g))) % 10000)
            U_tgt_X = np.kron(Wd, Wd) @ U_tgt @ np.kron(W, W)
            Ua_x = Wd @ U_from_C(Ca, g) @ W
            Ub_x = Wd @ U_from_C(Cb, g) @ W
            fid_x_kron = process_fidelity(U_tgt_X, np.kron(Ua_x, Ub_x))
            row = {
                "split": label,
                "gamma": float(g),
                "fid_joint_SNAP_exact": fid_exact,
                "fid_best_local_SNAP_kron": fid_sep,
                "fid_BS_sandwich": bs["fid"],
                "BS_theta": bs["theta_bs"],
                "fid_BinH_local_kron_Xframe": fid_x_kron,
                "needs_entangling": fid_sep < 0.999,
            }
            all_rows.append(row)
            log(
                f"{label} γ={g:.4f}: joint={fid_exact:.6f} localKron={fid_sep:.4f} "
                f"BS={bs['fid']:.4f} Xkron={fid_x_kron:.4f}"
            )

    payload = {
        "notes": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("Ca", "Cb")}
            for k, v in all_notes.items()
        },
        "fidelity_rows": all_rows,
        "conclusion": (
            "Two-cavity clause U(γ)=I+(e^{-iγ}-1)Ca⊗Cb does NOT factor as Ua⊗Ub. "
            "Z-frame: exact as joint number-dependent SNAP (phase on supp_a×supp_b). "
            "Local SNAP⊗SNAP is separable and cannot realize the AND phase; "
            "BS sandwich improves fidelity but joint SNAP / cross-Kerr / EigenSNAP is the native compile."
        ),
    }
    # json-safe notes (lists)
    for k, v in payload["notes"].items():
        pass
    (OUT / "compile_twocav.json").write_text(json.dumps(payload, indent=2))

    # Markdown
    lines = [
        "# Two-cavity projector compile (simulation)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "## Setup",
        "",
        "Clause spanning two cavities: \(C=C_a\\otimes C_b\) with local projectors from the frozen library.",
        r"Target \(U(\gamma)=I+(e^{-i\gamma}-1)C\) on \(\mathbb{C}^8\otimes\mathbb{C}^8\) (Fock trunc 0..7).",
        "",
        "Representatives:",
        "",
        "- **(1,3):** \(C_a=\\Pi_{b0}\), \(C_b=\\Pi_{b0,b1,b2}\) (rank 4 ⊗ rank 1)",
        "- **(2,2):** \(C_a=\\Pi_{b0,b1}\), \(C_b=\\Pi_{b0,b1}\) (rank 2 ⊗ rank 2)",
        "",
        "## Does \(U\) factor as locals?",
        "",
        "**No.** If \(U_a=I+(e^{-i\gamma}-1)C_a\) and \(U_b=I+(e^{-i\gamma}-1)C_b\), then",
        "",
        r"\[ U_a\otimes U_b = I + (e^{-i\gamma}-1)(C_a\otimes I + I\otimes C_b) + (e^{-i\gamma}-1)^2 C_a\otimes C_b \]",
        "",
        r"which is **not** \(I+(e^{-i\gamma}-1)C_a\otimes C_b\). The joint unitary phases only the",
        "**AND** of supports \(\\mathrm{supp}(C_a)\\times\\mathrm{supp}(C_b)\); product of local",
        "projector-unitaries also phases when only one cavity is in support.",
        "",
    ]
    for label, notes in all_notes.items():
        lines += [
            f"### {label}",
            "",
            f"- ranks (Ca, Cb, C) = ({notes['rank_Ca']}, {notes['rank_Cb']}, {notes['rank_C']})",
            f"- supp_a = {notes['supp_a']}, supp_b = {notes['supp_b']}",
            f"- Fid(\(U\), \(U_a\\otimes U_b\)) = **{notes['fid_U_vs_Ua_kron_Ub']:.6f}** "
            f"(fro={notes['fro_U_vs_Ua_kron_Ub']:.3e})",
            f"- factors as local projector-unitaries? **{notes['factors_as_local_projector_unitaries']}**",
            "",
        ]
    lines += [
        "## Compile paths",
        "",
        "| path | what | exact? |",
        "|------|------|--------|",
        "| **Joint Fock SNAP** (Z-frame) | phase γ on \|fa,fb⟩ iff fa∈supp_a and fb∈supp_b | **YES** |",
        "| BinH sandwich | \(W^\\dagger\\otimes W^\\dagger\) · jointSNAP · \(W\\otimes W\) | **YES** (X-frame) |",
        "| Local SNAP⊗SNAP | separable phases θa(fa)+θb(fb) | **NO** (misses AND) |",
        "| BS · (SNAP⊗SNAP) · BS† | entangling sandwich | approximate (table) |",
        "| Cross-Kerr / EigenSNAP | native #a#b-dependent phase | hardware-native AND |",
        "",
        "## Fidelity table",
        "",
        "| split | γ | Fid joint SNAP | Fid best local SNAP⊗SNAP | Fid BS sandwich | Fid BinH local⊗ in X | needs entangling |",
        "|-------|---|----------------|--------------------------|-----------------|----------------------|------------------|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['split']} | {r['gamma']:.4f} | {r['fid_joint_SNAP_exact']:.6f} | "
            f"{r['fid_best_local_SNAP_kron']:.4f} | {r['fid_BS_sandwich']:.4f} | "
            f"{r['fid_BinH_local_kron_Xframe']:.4f} | {r['needs_entangling']} |"
        )
    lines += [
        "",
        "## Takeaway for GRAPE / hardware",
        "",
        "1. Single-cavity factors stay **exact local SNAP** (library FREEZE).",
        "2. Two-cavity clauses **require a joint / entangling compile** — not SNAP_a⊗SNAP_b.",
        "3. Preferred sim compile: **joint number-dependent diagonal** (or BinH-conjugated).",
        "4. Preferred hardware story: **cross-Kerr / EigenSNAP / BS+SNAP sandwich** GRAPE with",
        "   target \(U(\\gamma)=I+(e^{-i\\gamma}-1)C_a\\otimes C_b\) — still **not** densified diag(E).",
        "5. Instance only picks which \((C_a,C_b)\) pair from the finite library.",
        "",
        "Artifacts: `compile_twocav.json`. Champ classical angles unchanged.",
        "",
    ]
    (OUT / "COMPILE_TWOCAV.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "COMPILE_TWOCAV.md")


if __name__ == "__main__":
    main()
