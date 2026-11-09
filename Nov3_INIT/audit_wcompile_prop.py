"""PR109 propagator audit and frozen-pulse replay. No SPSA. No SK fits."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.run_v1 import _dump
from Nov3_INIT.wcompile import compiled_map as map_pr109, metrics as metrics_pr109, x0_identity
from Nov3_INIT.wcompile_prop import (
    EVALS,
    LADDER,
    column_overlap,
    isometry_metrics,
    partition_columns,
    pr109_partition_explanation,
    pulse_from_json,
    tests,
    workspace_map,
)

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "wcompile"
PULSE = RES / "pulse.json"
OUT = RES / "AUDIT_PROP.json"
MD = RES / "ADDENDUM_PROP.md"
SK_LEDGER = 2818048
HIST_COMPILER = 2048


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def main() -> None:
    EVALS.clear()
    pulse = json.loads(PULSE.read_text())
    x, alpha, theta = pulse_from_json(pulse)
    x0 = x0_identity()

    # Historical map at nf=16 must match workspace_map(16): that IS the bug.
    v_old = map_pr109(x, 16)
    v_ws16 = workspace_map(alpha, theta, 16)
    match16 = float(np.linalg.norm(v_old - v_ws16))
    m_old = metrics_pr109(x, 16)
    part_old = partition_columns(v_old)
    euc_mean = float(np.mean(np.sqrt(np.clip(np.array(part_old["p_total"]), 0.0, None))))

    ladder = {}
    maps = {}
    for nf in LADDER:
        v = workspace_map(alpha, theta, nf)
        maps[nf] = v
        ladder[str(nf)] = isometry_metrics(v, nf)

    ident = {}
    for nf in LADDER:
        a0 = np.zeros_like(alpha)
        th0 = np.zeros_like(theta)
        ident[str(nf)] = isometry_metrics(workspace_map(a0, th0, nf), nf)

    conv = {
        "32_vs_64": column_overlap(maps[32], maps[64]),
        "64_vs_96": column_overlap(maps[64], maps[96]),
        "32_vs_96": column_overlap(maps[32], maps[96]),
    }
    t96 = tests(alpha, theta, nf=96)
    t32 = tests(alpha, theta, nf=32)

    L_ident = ident["96"]["L"]
    L_sel = ladder["96"]["L"]
    L_old_sel = m_old["L"]
    L_old_ident = metrics_pr109(x0, 16)["L"]

    payload = {
        "version": "N3INIT_wcompile_prop_audit",
        "no_new_spsa": True,
        "no_new_pulse_depth": True,
        "no_sk_fits": True,
        "sk_ledger_unchanged": SK_LEDGER,
        "historical_compiler_evals_recorded": HIST_COMPILER,
        "new_conformance_evals": dict(EVALS),
        "bug": {
            "what": "each D was P_nf exp_{nf+32}(α(a†-a)) P_nf; SNAP then the next P D P. Pad amplitudes were discarded between gates.",
            "not": "unrestricted oscillator evolution, nor a single final truncation of a large-workspace unitary",
            "pr109_vs_workspace16_fro": match16,
            "pr109_equals_workspace16": match16 < 1e-12,
        },
        "pr109_nf16_partition": part_old,
        "pr109_mean_euclid_norm": euc_mean,
        "pr109_reported_mean_col_norm_0.402_is": "mean ||psi||^2 (retained probability), not Euclidean ||psi||",
        "pr109_arithmetic": pr109_partition_explanation(),
        "F_definitions": {
            "F_avg": "mean_k |<T e_k | V e_k>|^2 after one global phase; unnormalized; not a postselected state fidelity",
            "F_worst": "min of those 8",
            "F_hs": "|tr(T^H V)|^2 / 64, unnormalized Hilbert-Schmidt",
            "old_p_leak": "1-p_code, which includes lost boundary mass",
            "corrected": "p_code + p_pad_retained + p_lost = 1",
        },
        "identity_was_start0": True,
        "identity_evaluated": True,
        "identity_e_init_nf16": L_old_ident,
        "selected_L_nf16_broken_prop": L_old_sel,
        "identity_L_ws96": L_ident,
        "selected_L_ws96": L_sel,
        "selected_beats_identity_on_broken_L": bool(L_old_sel < L_old_ident),
        "selected_beats_identity_on_workspace96_L": bool(L_sel < L_ident),
        "why_broken_L_improved": "nonunitary P D P shrinks ||V|| so ||V-T||_F can fall even when F_avg does not beat identity (0.117 vs 0.125)",
        "ladder_selected": ladder,
        "ladder_identity": ident,
        "isometry_convergence": conv,
        "tests_nf96": t96,
        "tests_nf32": t32,
        "not_a_physical_validation_claim": True,
        "wcomp_fail_budget_historical": True,
        "no_gr_patch": True,
    }
    unit_ok = bool(
        t96["D0_ok"]
        and t96["snap_ok"]
        and t96["roundtrip_code_ok"]
        and t96["inverse_ok"]
        and payload["bug"]["pr109_equals_workspace16"]
        and ident["96"]["partition"]["mean_p_lost"] < 1e-12
        and ident["96"]["partition"]["mean_p_total"] > 1.0 - 1e-12
    )
    favg = [ladder[str(n)]["F_avg"] for n in LADDER]
    code_iso = bool(
        abs(favg[0] - favg[2]) < 1e-6
        and conv["64_vs_96"]["code_block_fro"] < 1e-10
    )
    full_space_ok = bool(
        conv["64_vs_96"]["mean_col_overlap_sq"] >= 0.999
        and ladder["96"]["partition"]["mean_p_lost"] < 1e-6
        and ladder["64"]["partition"]["mean_p_lost"] < 1e-3
    )
    payload["code_isometry_converged"] = code_iso
    payload["F_avg_across_ladder"] = favg
    conv_ok = full_space_ok
    payload["propagation_tests_pass"] = unit_ok
    payload["workspace_convergence_pass"] = conv_ok
    payload["full_space_pad_tail_converged_64_vs_96"] = full_space_ok
    payload["physical_validation_claim"] = False
    payload["ok"] = bool(unit_ok)  # audit ran; convergence may fail and that is a finding
    _dump(OUT, json.loads(json.dumps(payload, default=_plain)))

    p96 = ladder["96"]["partition"]
    p32 = ladder["32"]["partition"]
    p64 = ladder["64"]["partition"]
    lines = [
        "# Addendum: PR109 propagator was repeated truncation, not oscillator evolution",
        "",
        "Historical **WCOMP_FAIL_BUDGET** numbers are preserved. They are not evidence that native SNAP+D compilation is impossible. No refit. SK ledger 2818048. Historical compiler evals 2048 remain recorded.",
        "",
        "## What PR109 did",
        "",
        "Each displacement was `P_nf D_{nf+32}(α) P_nf` with nf=16 (replay 32). That block is non-unitary. Applying it after every D **throws away pad amplitudes before the next gate**. That is implicit repeated projection, not a sequence on one oscillator. SNAP pad phase 0 was already unitary.",
        "",
        f"Audit check: `workspace_map` at nf=16 matches PR109 `compiled_map` (Frobenius {match16:.2e}).",
        "",
        "## 0.402 vs 0.672",
        "",
        f"PR109 `col_norm` is **mean ‖ψ‖²** (retained probability), not Euclidean ‖ψ‖. Mean Euclidean norm was {euc_mean:.4f}.",
        "",
        f"- mean p_total = {part_old['mean_p_total']:.4f} (the 0.402 figure)",
        f"- mean p_code = {part_old['mean_p_code']:.4f}",
        f"- mean p_pad in 8..15 = {part_old['mean_p_pad_retained']:.4f}",
        f"- mean p_lost = {part_old['mean_p_lost']:.4f} (discarded by the three projections)",
        f"- old p_leak = 1-p_code = {part_old['old_p_leak_was_one_minus_p_code']:.4f} = pad + lost",
        "",
        "F_avg / F_worst / F_HS used unnormalized overlaps against the padded isometry T. They are not postselected state fidelities. When ‖ψ‖²≈0.40 they cannot be read as physical gate error.",
        "",
        "## Identity vs selected on the actual objective",
        "",
        "Start 0 was the identity compile and **was evaluated** (not excluded). NF=16 broken propagator: identity L = 2.000, selected polyak64 L = 0.822. Selected **beats** identity on L because shrinking ‖V‖ lowers ‖V−T‖_F even though F_avg 0.117 **loses** to identity 0.125.",
        "",
        f"Common workspace NF=96 (no refit, unit norm): identity L = {L_ident:.4f}, selected L = {L_sel:.4f}. Selected still beats identity on L ({payload['selected_beats_identity_on_workspace96_L']}) because tr(W)=0 so identity has HS overlap 0, while the pulse has a globally alignable positive overlap. F_avg 0.112 still **loses** to identity 0.125. The extra drop from L=1.434 (unitary workspace) to L=0.822 (PR109) is the nonunitary projection shrinking ‖V‖.",
        "",
        "## Frozen-pulse replay on a common workspace (32, 64, 96)",
        "",
        "Same `pulse.json`. D defined as restrict(exp_{W+32}) onto W, then the whole (D,SNAP)^3 stays in C^W. No renormalization.",
        "",
        "| W | mean p_total | mean p_code | mean p_pad | mean p_lost | mean p_boundary | F_avg | F_HS | L |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nf, tag in ((32, "32"), (64, "64"), (96, "96")):
        m = ladder[tag]
        p = m["partition"]
        lines.append(
            f"| {nf} | {p['mean_p_total']:.6f} | {p['mean_p_code']:.4f} | {p['mean_p_pad_retained']:.4f} | {p['mean_p_lost']:.3e} | {p['mean_p_boundary']:.3e} | {m['F_avg']:.4f} | {m['F_hs']:.4f} | {m['L']:.4f} |"
        )
    c = conv
    lines += [
        "",
        f"Full-space column overlap (pad-embed, one global phase): 32 vs 64 mean |⟨col⟩|² {c['32_vs_64']['mean_col_overlap_sq']:.4f}; 64 vs 96 {c['64_vs_96']['mean_col_overlap_sq']:.6f}; 32 vs 96 {c['32_vs_96']['mean_col_overlap_sq']:.4f}.",
        f"Code-block Frobenius 32 vs 96: {c['32_vs_96']['code_block_fro']:.3e}; 64 vs 96: {c['64_vs_96']['code_block_fro']:.3e}.",
        f"F_avg is identical on the ladder ({favg[0]:.6f}): T lives on 0..7, so F_avg/F_HS are code-isometry diagnostics. Code-isometry converged={code_iso}.",
        "",
        f"W=96 has unit norm (p_lost {p96['mean_p_lost']:.3e}) but 64 vs 96 full-column overlap is 0.980 because the pad tail (levels ≥64) is still moving. Unit norm ≠ full-space convergence. full_space_pad_tail_converged_64_vs_96={full_space_ok}.",
        "",
        "## Conformance tests (W=96 unless noted)",
        "",
        f"- D(0)=I: err {t96['D0_minus_I_fro']:.2e}. SNAP unimodular max err {t96['snap_unimodular_max_err']:.2e}.",
        f"- D(−α)D(α) on |0⟩,|3⟩,|7⟩ norms {t96['roundtrip_norm_fock_0_3_7']}; on |+⟩ {t96['roundtrip_norm_plus']:.6f}. Full-space D†D−I fro/n {t96['Dminus_Dplus_I_fro_over_n']:.3e} (residual on high-n basis states near the cutoff).",
        f"- Sequence then inverse, back to code basis: fro/col {t96['sequence_inverse_fro_per_col']:.3e}.",
        f"- Same tests at W=32 inverse_ok={t32['inverse_ok']} roundtrip_code_ok={t32['roundtrip_code_ok']} (32 is too small for three |α|≈1.8 displacements).",
        f"- Cropped expm(α(a†−a)) on W vs +32-restricted true D, fro/n {t96['cropped_expm_vs_restricted_fro_over_n']:.3e}.",
        "",
        f"propagation_tests_pass={unit_ok}. New conformance evals {int(EVALS['total'])} (not SK, not added to 2048).",
        "",
        "## Claim",
        "",
        "No physical validation of compiled W_BinH. WCOMP_FAIL_BUDGET stays the historical SPSA result under a broken inter-gate projection. Corrected frozen-pulse numbers above are diagnostics of that same pulse, not a new compile. No GR.",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(json.dumps({"ok": payload["ok"], "propagation_tests_pass": unit_ok, "workspace_convergence_pass": conv_ok, "evals": dict(EVALS)}, indent=2))
    if match16 >= 1e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
