#!/usr/bin/env python3
"""GRAPE library + SNAP fidelity for clause-projector gadgets (BinH).

Enumerates finite local unsat projectors on a 3-bit cavity for 4-SAT
(spans 2/3/4 cavities → local k∈{1,2,3}), shows U(γ)=I+(e^{-iγ}-1)C is
fixed-op + γ phase. Fidelity tables: exact SNAP (Z-frame), W†SNAP W (X-frame),
and SNAP+D least-squares fit to U_X on d=8.

Writes runs/gadget_Uc/grape_library/{LIBRARY.md,fidelity.json,RESULTS.md}.
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

OUT = _PKG / "runs" / "gadget_Uc" / "grape_library"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[grape_lib]", *a, flush=True)


def projector_Z(bits_signs: list[tuple[int, int]]) -> np.ndarray:
    """Rank-|S|? Actually rank 2^{3-|S|}: ∏_{j,s} (I - s Z_j)/2 on 8-dim.

    bits_signs: list of (bit j in 0..2, sign s=±1). Unsat when bit==0 if s=+1
    (literal x), bit==1 if s=-1 (literal ¬x) — matching sat_gen: ss>0 → ix=0.
    """
    P = np.eye(8, dtype=np.complex128)
    Zs = []
    for j in range(3):
        d = np.ones(8, dtype=np.complex128)
        for f in range(8):
            d[f] = 1.0 if ((f >> j) & 1) == 0 else -1.0
        Zs.append(np.diag(d))
    for j, s in bits_signs:
        # (I - s Z)/2 : eigenvalue 1 when s*z_eig = -1? 
        # z_eig = +1 for bit0, -1 for bit1.
        # sat_gen: unsat contrib when bit matches "false literal"
        # s>0 false at bit=0 → want projector |0⟩; (I+Z)/2 = |0⟩⟨0| when Z|+⟩=+
        # Standard: (I+Z)/2 = |0⟩⟨0|, (I-Z)/2 = |1⟩⟨1|
        # s>0 → unsat at 0 → (I+Z)/2
        # s<0 → unsat at 1 → (I-Z)/2
        if s > 0:
            Pj = 0.5 * (np.eye(8) + Zs[j])
        else:
            Pj = 0.5 * (np.eye(8) - Zs[j])
        P = Pj @ P
    return P


def U_from_C(C: np.ndarray, gamma: float) -> np.ndarray:
    """I + (e^{-iγ}-1) C  for projector C (C²=C, C†=C)."""
    return np.eye(C.shape[0], dtype=np.complex128) + (np.exp(-1j * gamma) - 1.0) * C


def process_fidelity(A: np.ndarray, B: np.ndarray) -> float:
    """Entanglement fidelity for unitary channels on pure states: |Tr(A†B)|²/d²."""
    d = A.shape[0]
    return float(abs(np.trace(A.conj().T @ B)) ** 2 / (d * d))


def fro_fidelity(A: np.ndarray, B: np.ndarray) -> float:
    """|⟨A,B⟩_F|² / (||A||_F² ||B||_F²) — for unitaries = |Tr(A†B)|²/d² same as process if unitary."""
    num = abs(np.vdot(A.ravel(), B.ravel())) ** 2
    den = (np.linalg.norm(A) ** 2) * (np.linalg.norm(B) ** 2)
    return float(num / max(den, 1e-30))


def snap_from_phases(theta: np.ndarray, nf: int = 8) -> np.ndarray:
    ph = np.zeros(nf, dtype=np.float64)
    ph[: len(theta)] = theta
    return np.diag(np.exp(-1j * ph)).astype(np.complex128)


def exact_snap_phases_for_CZ(C: np.ndarray, gamma: float) -> np.ndarray:
    """C diagonal in Z → SNAP phases: γ on support of C, else 0."""
    # C should be diagonal 0/1
    diag = np.real(np.diag(C))
    theta = np.zeros(8)
    for f in range(8):
        if diag[f] > 0.5:
            theta[f] = gamma
        else:
            theta[f] = 0.0
    return theta


def pauli_x_bit(j: int) -> np.ndarray:
    X = np.zeros((8, 8), dtype=np.complex128)
    for f in range(8):
        X[f ^ (1 << j), f] = 1.0
    return X


def projector_X(bits_signs: list[tuple[int, int]]) -> np.ndarray:
    """BinH conjugate of projector_Z: replace Z→X in the same formula."""
    P = np.eye(8, dtype=np.complex128)
    Xs = [pauli_x_bit(j) for j in range(3)]
    for j, s in bits_signs:
        if s > 0:
            Pj = 0.5 * (np.eye(8) + Xs[j])  # H(I+Z)H/2
        else:
            Pj = 0.5 * (np.eye(8) - Xs[j])
        P = Pj @ P
    # numerical hermitize / project
    P = 0.5 * (P + P.conj().T)
    return P


def _expm_skew(H: np.ndarray) -> np.ndarray:
    """expm for anti-Hermitian / general via eig (d=8 tiny)."""
    w, V = np.linalg.eig(H)
    return (V * np.exp(w)) @ np.linalg.inv(V)


def displacement_matrix(alpha: complex, nf: int = 8) -> np.ndarray:
    """Truncated displacement D(α) on Fock 0..nf-1."""
    a = np.diag(np.sqrt(np.arange(1, nf, dtype=np.float64)), 1)
    ad = a.T.astype(np.complex128)
    a = a.astype(np.complex128)
    H = alpha * ad - np.conj(alpha) * a
    return _expm_skew(H).astype(np.complex128)


def fit_snap_D_to_target(U_tgt: np.ndarray, n_starts: int = 6, seed: int = 0, steps: int = 120) -> dict:
    """Fit SNAP(θ)·D(α) or D·SNAP·D to U_tgt on d=8 via torch Adam."""
    rng = np.random.default_rng(seed)
    Ut = torch.as_tensor(U_tgt, dtype=torch.complex128)
    nf = 8
    # precompute a, ad
    a = torch.diag(torch.sqrt(torch.arange(1, nf, dtype=torch.float64)), 1).to(torch.complex128)
    ad = a.mH.contiguous()

    def D_of(re, im):
        alpha = torch.complex(re, im)
        H = alpha * ad - torch.conj(alpha) * a
        # expm via eigen
        w, V = torch.linalg.eig(H)
        return V * torch.exp(w) @ torch.linalg.inv(V)

    def snap(theta):
        return torch.diag(torch.exp(-1j * theta.to(torch.complex128)))

    def fid(U):
        d = nf
        return (torch.abs(torch.trace(Ut.mH @ U)) ** 2 / (d * d)).real

    best = {"fid_SD": 0.0, "fid_DSD": 0.0}

    for _ in range(n_starts):
        theta = torch.nn.Parameter(torch.as_tensor(rng.normal(0, 0.5, size=8), dtype=torch.float64))
        re = torch.nn.Parameter(torch.tensor(float(rng.normal(0, 0.3)), dtype=torch.float64))
        im = torch.nn.Parameter(torch.tensor(float(rng.normal(0, 0.3)), dtype=torch.float64))
        opt = torch.optim.Adam([theta, re, im], lr=0.08)
        for _s in range(steps):
            opt.zero_grad()
            U = snap(theta) @ D_of(re, im)
            loss = 1.0 - fid(U)
            loss.backward()
            opt.step()
        f = float(fid(snap(theta) @ D_of(re, im)).detach())
        if f > best["fid_SD"]:
            best["fid_SD"] = f

        theta = torch.nn.Parameter(torch.as_tensor(rng.normal(0, 0.5, size=8), dtype=torch.float64))
        re1 = torch.nn.Parameter(torch.tensor(float(rng.normal(0, 0.3)), dtype=torch.float64))
        im1 = torch.nn.Parameter(torch.tensor(float(rng.normal(0, 0.3)), dtype=torch.float64))
        re2 = torch.nn.Parameter(torch.tensor(float(rng.normal(0, 0.3)), dtype=torch.float64))
        im2 = torch.nn.Parameter(torch.tensor(float(rng.normal(0, 0.3)), dtype=torch.float64))
        opt = torch.optim.Adam([theta, re1, im1, re2, im2], lr=0.08)
        for _s in range(steps):
            opt.zero_grad()
            U = D_of(re1, im1) @ snap(theta) @ D_of(re2, im2)
            loss = 1.0 - fid(U)
            loss.backward()
            opt.step()
        f = float(fid(D_of(re1, im1) @ snap(theta) @ D_of(re2, im2)).detach())
        if f > best["fid_DSD"]:
            best["fid_DSD"] = f
    return best


def enumerate_library() -> dict:
    """All nonempty local (bit,sign) patterns on 3 bits — finite GRAPE family."""
    bits = [0, 1, 2]
    lib = {"k1": [], "k2": [], "k3": [], "note": {}}
    # signs ±1 for each involved bit
    for k in (1, 2, 3):
        for S in itertools.combinations(bits, k):
            for signs in itertools.product([-1, 1], repeat=k):
                bs = list(zip(S, signs))
                CZ = projector_Z(bs)
                CX = projector_X(bs)
                # rank
                ev = np.linalg.eigvalsh(CZ.real) if np.allclose(CZ.imag, 0) else np.linalg.eigvalsh(0.5 * (CZ + CZ.conj().T))
                rank = int(np.sum(ev > 0.5))
                # support Fock states
                supp = [int(f) for f in range(8) if abs(CZ[f, f]) > 0.5]
                entry = {
                    "bits_signs": [(int(j), int(s)) for j, s in bs],
                    "k": k,
                    "rank": rank,
                    "fock_support": supp,
                    "label": "b" + "".join(str(j) for j in S) + "_s" + "".join("+" if s > 0 else "-" for s in signs),
                }
                lib[f"k{k}"].append(entry)
    lib["note"] = {
        "4SAT_cavity_spans": "With 3 bits/cavity, 4-var clauses never fit in 1 cavity. Spans ∈{2,3,4}.",
        "local_k_from_data": "Empirical: (1,3), (2,2), (1,1,2), (1,1,1,1) — local k∈{1,2,3}.",
        "U_gamma": "U(γ)=I+(e^{-iγ}-1)C for each projector C (or C^X). Only γ changes.",
        "count": {f"k{k}": len(lib[f"k{k}"]) for k in (1, 2, 3)},
        "total_local_patterns": sum(len(lib[f"k{k}"]) for k in (1, 2, 3)),
    }
    # sanity C^2=C
    for k in (1, 2, 3):
        for e in lib[f"k{k}"][:2]:
            CZ = projector_Z(e["bits_signs"])
            err = np.linalg.norm(CZ @ CZ - CZ)
            e["proj_err"] = float(err)
    return lib


def run_fidelity_table(gammas=(0.2, 0.5, 1.0, 1.5), max_patterns: int = 6) -> dict:
    W = build_W_BinH(nf=8, dtype=torch.complex128).numpy()[:8, :8]
    Wd = W.conj().T
    lib = enumerate_library()
    # pick representative patterns: one per k, two signs
    reps = []
    for k in (1, 2, 3):
        reps.append(lib[f"k{k}"][0])  # all +
        reps.append(lib[f"k{k}"][1] if len(lib[f"k{k}"]) > 1 else lib[f"k{k}"][0])
    reps = reps[:max_patterns]

    rows = []
    for e in reps:
        CZ = projector_Z(e["bits_signs"])
        CX = projector_X(e["bits_signs"])
        # verify BinH: W† CZ W ≈ CX?  Actually H Z H = X so W† f(Z) W = f(X) if W=H
        # U_Z conjugated: W† U_Z W should equal U_X
        for g in gammas:
            UZ = U_from_C(CZ, g)
            UX = U_from_C(CX, g)
            U_via = Wd @ UZ @ W
            conj_err = float(np.linalg.norm(U_via - UX, ord="fro"))
            # exact SNAP in Z-frame
            th = exact_snap_phases_for_CZ(CZ, g)
            U_snap = snap_from_phases(th)
            fid_snap_Z = process_fidelity(UZ, U_snap)
            # W† SNAP W vs UX
            U_Wsnap = Wd @ U_snap @ W
            fid_Wsnap_X = process_fidelity(UX, U_Wsnap)
            # SNAP+D fit to UX (harder — UX not diagonal in Z)
            fit = fit_snap_D_to_target(UX, n_starts=4, seed=hash(e["label"] + str(g)) % 10_000)
            row = {
                "label": e["label"],
                "k": e["k"],
                "rank": e["rank"],
                "gamma": g,
                "BinH_conj_fro": conj_err,
                "fid_SNAP_vs_UZ": fid_snap_Z,
                "fid_Wdag_SNAP_W_vs_UX": fid_Wsnap_X,
                "fid_SNAP_D_fit_UX": fit["fid_SD"],
                "fid_D_SNAP_D_fit_UX": fit["fid_DSD"],
            }
            rows.append(row)
            log(
                f"{e['label']} γ={g}: SNAP/Z={fid_snap_Z:.6f} W†SNAP W/X={fid_Wsnap_X:.6f} "
                f"SD={fit['fid_SD']:.4f} DSD={fit['fid_DSD']:.4f} conj_fro={conj_err:.2e}"
            )
    return {"gammas": list(gammas), "rows": rows, "reps": [e["label"] for e in reps]}


def write_library_md(lib: dict, fid: dict, path: Path):
    lines = [
        "# GRAPE library — clause-projector gadgets (3-bit cavity, BinH)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "## Finite operator family",
        "",
        "4-SAT + K=3 packing ⇒ clause cavity-spans ∈{2,3,4} (never 1). Local support per cavity k∈{1,2,3}.",
        "",
        "Each local factor is a **fixed** projector",
        "",
        r"\[ C_S = \prod_{(j,s)\in S} \frac{I - \mathrm{sgn}'(s)\,Z_j}{2} \]",
        "",
        "with BinH image \(C^X_S\) (Z→X). Unitary",
        "",
        r"\[ U(\gamma)= I + (e^{-i\gamma}-1)\,C \]",
        "",
        "— **same C**, only γ changes. GRAPE / pulse library = these finitely many C (and two-cavity products \(C_a\\otimes C_b\)).",
        "",
        f"- Patterns k=1: **{lib['note']['count']['k1']}** (choose 1 bit × 2 signs)",
        f"- Patterns k=2: **{lib['note']['count']['k2']}**",
        f"- Patterns k=3: **{lib['note']['count']['k3']}**",
        f"- **Total local: {lib['note']['total_local_patterns']}**",
        "",
        "Two-cavity clause: \(C=C^{(a)}\\otimes C^{(b)}\) with \((k_a,k_b)\\in\\{(1,3),(3,1),(2,2)\\}\). Still fixed op × γ.",
        "",
        "### Catalog (labels)",
        "",
    ]
    for k in (1, 2, 3):
        labels = ", ".join(e["label"] for e in lib[f"k{k}"])
        lines.append(f"- **k={k}** (rank={lib[f'k{k}'][0]['rank']}): {labels}")
    lines += [
        "",
        "## Exact pulse identities (proof sketch)",
        "",
        "1. **Z-frame (lab Fock):** \(C_Z\) is diagonal 0/1 on Fock 0..7 ⇒ "
        r"\(U_Z(\gamma)=\mathrm{SNAP}(\theta)\) with \(\theta_f=\gamma\) on \(\mathrm{supp}(C_Z)\), else 0. "
        "**Exact** — no GRAPE needed for single-cavity factors.",
        "2. **X-frame (BinH):** \(U_X=W^\dagger U_Z W\) with \(W=H^{\otimes 3}\) ⇒ "
        r"\(U_X=W^\dagger\mathrm{SNAP}(\theta)W\). Exact if W is available (Arm C sandwich).",
        "3. **SNAP+D without W** fitting \(U_X\) in Fock basis: numerical (table below) — optional compile path.",
        "",
        "## Fidelity table (d=8)",
        "",
        "| pattern | k | γ | Fid SNAP↔UZ | Fid W†SNAP W↔UX | Fid SNAP·D↔UX | Fid D·SNAP·D↔UX | BinH conj fro |",
        "|---------|---|---|-------------|-----------------|---------------|-----------------|---------------|",
    ]
    for r in fid["rows"]:
        lines.append(
            f"| {r['label']} | {r['k']} | {r['gamma']} | {r['fid_SNAP_vs_UZ']:.6f} | "
            f"{r['fid_Wdag_SNAP_W_vs_UX']:.6f} | {r['fid_SNAP_D_fit_UX']:.4f} | "
            f"{r['fid_D_SNAP_D_fit_UX']:.4f} | {r['BinH_conj_fro']:.2e} |"
        )
    lines += [
        "",
        "### Takeaway",
        "",
        "- SNAP↔UZ and W†SNAP W↔UX should be **=1** (exact) within numerics.",
        "- SNAP+D fit to UX is a fallback if W is not native; expect high but not exact fidelity.",
        "- **GRAPE never needs instance `diag(E)`** — only library ops \(U(\\gamma;C)\) for C in this finite set, scheduled by the clause list.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("enumerate library")
    lib = enumerate_library()
    (OUT / "library.json").write_text(json.dumps(lib, indent=2))
    log("total patterns", lib["note"]["total_local_patterns"])
    log("fidelity table")
    fid = run_fidelity_table()
    (OUT / "fidelity.json").write_text(json.dumps(fid, indent=2))
    write_library_md(lib, fid, OUT / "LIBRARY.md")
    # short RESULTS
    exact_ok = all(
        r["fid_SNAP_vs_UZ"] > 1 - 1e-10 and r["fid_Wdag_SNAP_W_vs_UX"] > 1 - 1e-10
        for r in fid["rows"]
    )
    lines = [
        "# GRAPE library RESULTS",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"- Local projector patterns: **{lib['note']['total_local_patterns']}** (k1+k2+k3)",
        f"- Exact SNAP / W†SNAP W identities: **{'PASS' if exact_ok else 'FAIL'}**",
        f"- Full write-up: `LIBRARY.md`",
        "",
    ]
    # mean SD fit
    sd = np.mean([r["fid_SNAP_D_fit_UX"] for r in fid["rows"]])
    dsd = np.mean([r["fid_D_SNAP_D_fit_UX"] for r in fid["rows"]])
    lines += [f"- Mean Fid SNAP·D fit to UX: **{sd:.4f}**", f"- Mean Fid D·SNAP·D fit to UX: **{dsd:.4f}**", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("exact_ok", exact_ok, "mean SD", f"{sd:.4f}", "DSD", f"{dsd:.4f}")
    log("wrote", OUT)


if __name__ == "__main__":
    main()
